from datetime import UTC, datetime, timedelta

import pytest

from ai_video_generator.domain import (
    ArtifactState,
    ConditioningArtifact,
    ProjectSpec,
    TaskKind,
    TaskSpec,
    TaskState,
    WorkflowApproval,
    WorkflowOutput,
    WorkflowTemplate,
)
from ai_video_generator.persistence import (
    IdempotencyConflictError,
    InvalidTaskTransitionError,
    LeaseError,
    SQLiteTaskStore,
    StoreConflictError,
)
from ai_video_generator.workers.workflow import canonical_json_sha256


def make_task(
    task_id: str,
    *,
    state: TaskState = TaskState.BLOCKED,
    depends_on: tuple[str, ...] = (),
    fingerprint: str | None = None,
    max_attempts: int = 3,
) -> TaskSpec:
    ordinal = sum(ord(character) for character in task_id) % 16
    return TaskSpec(
        task_id=task_id,
        project_id="project-1",
        kind=TaskKind.H3_GENERATION,
        state=state,
        idempotency_key=f"{ordinal:x}" * 64,
        input_fingerprint=fingerprint or f"{(ordinal + 1) % 16:x}" * 64,
        depends_on=depends_on,
        max_attempts=max_attempts,
    )


@pytest.fixture
def store(tmp_path) -> SQLiteTaskStore:
    return SQLiteTaskStore(tmp_path / "project.db")


def test_database_uses_wal_and_duplicate_submit_returns_original_task(
    store: SQLiteTaskStore,
) -> None:
    original = make_task("segment-1")
    duplicate = original.model_copy(update={"task_id": "another-request-id"})

    assert store.journal_mode() == "wal"
    assert store.add_task(original) == original
    assert store.add_task(duplicate) == original
    assert len(store.list_tasks()) == 1


def test_idempotency_key_rejects_different_inputs(store: SQLiteTaskStore) -> None:
    task = make_task("segment-1")
    store.add_task(task)

    with pytest.raises(IdempotencyConflictError):
        store.add_task(
            task.model_copy(update={"task_id": "segment-2", "input_fingerprint": "f" * 64})
        )


def test_invalid_transition_is_rejected(store: SQLiteTaskStore) -> None:
    store.add_task(make_task("segment-1", state=TaskState.SUCCEEDED))

    with pytest.raises(InvalidTaskTransitionError, match="succeeded to running"):
        store.transition_task("segment-1", TaskState.RUNNING)


def test_claim_renew_and_recover_expired_lease(store: SQLiteTaskStore) -> None:
    start = datetime(2026, 8, 14, 12, tzinfo=UTC)
    store.add_task(make_task("segment-1", state=TaskState.READY))
    store.transition_task("segment-1", TaskState.QUEUED)

    claimed = store.claim_next("local-gpu-0", lease_duration=timedelta(seconds=30), now=start)
    assert claimed is not None
    assert claimed.state == TaskState.RUNNING
    assert claimed.attempt == 1
    assert claimed.lease_expires_at == start + timedelta(seconds=30)

    renewed = store.renew_lease(
        claimed.task_id,
        "local-gpu-0",
        lease_duration=timedelta(minutes=1),
        now=start + timedelta(seconds=10),
    )
    assert renewed.lease_expires_at == start + timedelta(seconds=70)
    assert store.recover_expired_leases(now=start + timedelta(seconds=69)) == ()
    assert store.recover_expired_leases(now=start + timedelta(seconds=71)) == ("segment-1",)

    recovered = store.get_task("segment-1")
    assert recovered.state == TaskState.READY
    assert recovered.lease_expires_at is None
    assert recovered.error_code == "lease_expired"

    with pytest.raises(LeaseError):
        store.renew_lease(
            recovered.task_id,
            "local-gpu-0",
            lease_duration=timedelta(seconds=30),
            now=start + timedelta(seconds=72),
        )


def test_submitted_comfy_prompt_is_reconciled_instead_of_resubmitted(
    store: SQLiteTaskStore,
) -> None:
    start = datetime(2026, 8, 14, 12, tzinfo=UTC)
    store.add_task(make_task("segment-1", state=TaskState.READY))
    store.transition_task("segment-1", TaskState.QUEUED)
    claimed = store.claim_next("local-gpu-0", lease_duration=timedelta(seconds=10), now=start)
    assert claimed is not None
    prompt = store.record_comfyui_prompt(
        claimed.task_id,
        "prompt-123",
        client_id="desktop-1",
        submitted_at=start + timedelta(seconds=1),
    )
    assert prompt.prompt_id == "prompt-123"

    store.recover_expired_leases(now=start + timedelta(seconds=11))
    recovered = store.get_task(claimed.task_id)
    assert recovered.state == TaskState.RUNNING
    assert recovered.lease_expires_at is None
    assert store.claim_next("local-gpu-0", lease_duration=timedelta(seconds=10)) is None
    assert store.list_unreconciled_comfyui_prompts() == (prompt,)

    reconciled = store.reconcile_comfyui_prompt(claimed.task_id, "succeeded")
    assert reconciled.status == "succeeded"
    assert store.list_unreconciled_comfyui_prompts() == ()


def test_success_promotes_dependant_and_stale_propagates(store: SQLiteTaskStore) -> None:
    first = make_task("a", state=TaskState.READY)
    second = make_task("b", depends_on=("a",))
    third = make_task("c", depends_on=("b",))
    store.add_task(first)
    store.add_task(second)
    store.add_task(third)

    store.transition_task("a", TaskState.QUEUED)
    assert store.claim_next("gpu-0", lease_duration=timedelta(minutes=1)).task_id == "a"
    store.transition_task("a", TaskState.SUCCEEDED)
    assert store.get_task("b").state == TaskState.READY
    store.transition_task("b", TaskState.QUEUED)
    assert store.claim_next("gpu-0", lease_duration=timedelta(minutes=1)).task_id == "b"
    store.transition_task("b", TaskState.SUCCEEDED)
    store.transition_task("c", TaskState.QUEUED)
    assert store.claim_next("gpu-0", lease_duration=timedelta(minutes=1)).task_id == "c"
    store.transition_task("c", TaskState.SUCCEEDED)

    stale = store.mark_stale("a")
    assert set(stale) == {"a", "b", "c"}
    assert {store.get_task(task_id).state for task_id in stale} == {TaskState.STALE}


def test_ready_task_requires_explicit_queue_before_worker_claim(store: SQLiteTaskStore) -> None:
    store.add_task(make_task("guided-ready", state=TaskState.READY))

    assert store.claim_next("gpu-0", lease_duration=timedelta(minutes=1)) is None

    store.transition_task("guided-ready", TaskState.QUEUED)
    claimed = store.claim_next("gpu-0", lease_duration=timedelta(minutes=1))
    assert claimed is not None
    assert claimed.task_id == "guided-ready"


def test_workflow_revisions_and_artifacts_are_immutable(store: SQLiteTaskStore) -> None:
    workflow = WorkflowTemplate(
        template_id="image",
        revision=1,
        name="Image",
        workflow_sha256=canonical_json_sha256({"1": {"class_type": "SaveImage", "inputs": {}}}),
        node_schema_sha256="b" * 64,
        raw_workflow={"1": {"class_type": "SaveImage", "inputs": {}}},
        outputs=(WorkflowOutput(output_id="image", node_id="1", title="Image"),),
        required_node_types=("SaveImage",),
        approval=WorkflowApproval.APPROVED,
    )
    artifact = ConditioningArtifact(
        artifact_id="conditioning-1",
        fingerprint="c" * 64,
        tensor_path="artifacts/conditioning-1.safetensors",
        manifest_path="artifacts/conditioning-1.json",
        segment_ids=("segment-1",),
        state=ArtifactState.READY,
        blob_sha256="d" * 64,
        byte_size=42,
    )

    assert store.put_workflow_revision(workflow) == workflow
    assert store.get_workflow_revision("image", 1) == workflow
    assert store.put_artifact(artifact) == artifact
    assert store.get_artifact("conditioning-1") == artifact


def test_project_revisions_are_immutable_and_latest_is_selected(
    store: SQLiteTaskStore,
) -> None:
    first = ProjectSpec(
        project_id="project-1",
        revision=1,
        name="First cut",
        target_duration_seconds=60,
    )
    second = first.model_copy(update={"revision": 2, "name": "Second cut"})

    assert store.put_project_revision(first) == first
    assert store.put_project_revision(first) == first
    assert store.put_project_revision(second) == second
    assert store.get_latest_project_revision("project-1") == second
    assert store.list_project_revisions("project-1") == (first, second)
    assert store.list_projects() == (second,)

    with pytest.raises(StoreConflictError, match="immutable"):
        store.put_project_revision(first.model_copy(update={"name": "Overwrite"}))


def test_task_review_is_atomic_and_idempotent(store: SQLiteTaskStore) -> None:
    accepted = make_task("accept", state=TaskState.NEEDS_REVIEW)
    rejected = make_task("reject", state=TaskState.NEEDS_REVIEW)
    unreviewable = make_task("unreviewable", state=TaskState.READY)
    store.add_task(accepted)
    store.add_task(rejected)
    store.add_task(unreviewable)

    reviewed = store.review_task("accept", accepted=True, feedback=" Looks good ")
    assert reviewed.state == TaskState.SUCCEEDED
    assert store.review_task("accept", accepted=True, feedback="Looks good") == reviewed

    declined = store.review_task("reject", accepted=False, feedback=" Fix continuity ")
    assert declined.state == TaskState.FAILED
    assert declined.error_code == "review_rejected"
    assert declined.error_message == "Fix continuity"

    with pytest.raises(StoreConflictError, match="different review"):
        store.review_task("accept", accepted=False)
    with pytest.raises(InvalidTaskTransitionError, match="not awaiting review"):
        store.review_task("unreviewable", accepted=True)
