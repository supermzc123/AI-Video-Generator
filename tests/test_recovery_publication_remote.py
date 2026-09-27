"""Behavioral regression coverage for publication, budgets and remote ownership."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from ai_video_generator.domain import ExecutionTarget, TaskKind, TaskSpec, TaskState
from ai_video_generator.domain.orchestration import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ProjectRunState,
    ProjectWorkspaceRevision,
    TaskCheckpoint,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.persistence.execution_runtime import execution_guard
from ai_video_generator.persistence.task_store import (
    InvalidTaskTransitionError,
    LeaseError,
    StoreConflictError,
)
from ai_video_generator.services.remote import TaskResultReport, WorkerHeartbeat, WorkerResultStatus
from ai_video_generator.workers.remote_worker import (
    RemoteWorkerClient,
    RemoteWorkerError,
    RetryPolicy,
)


def task(name, *, remote=False, kind=TaskKind.H3_GENERATION, max_attempts=3):
    return TaskSpec(
        task_id=name,
        project_id="project",
        kind=kind,
        state=TaskState.READY,
        idempotency_key=hashlib.sha256(name.encode()).hexdigest(),
        input_fingerprint=hashlib.sha256((name + "input").encode()).hexdigest(),
        execution_target=ExecutionTarget.REMOTE if remote else ExecutionTarget.LOCAL,
        worker_id="worker" if remote else None,
        max_attempts=max_attempts,
    )


def claim_local(store, name="t", **kwargs):
    store.add_task(task(name, **kwargs))
    store.transition_task(name, TaskState.QUEUED)
    assert store.acquire_dispatcher("dispatcher")
    claimed = store.claim_local_task(name, "dispatcher")
    assert claimed is not None
    return claimed


def claim_remote(store, name="remote", **kwargs):
    store.add_task(task(name, remote=True, **kwargs))
    store.transition_task(name, TaskState.QUEUED)
    claimed = store.claim_next(
        "worker",
        execution_target=ExecutionTarget.REMOTE,
        lease_duration=timedelta(seconds=120),
    )
    assert claimed is not None
    return claimed


@pytest.mark.parametrize("write", ["checkpoint", "workspace", "prompt", "child", "manifest"])
def test_late_guard_fences_all_transaction_writes(tmp_path, write):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    claimed = claim_local(store)
    store.request_task_cancellation("t")
    token = execution_guard.set(("t", claimed.attempt_id))
    try:
        with pytest.raises(ValueError, match="no longer allowed"):
            if write == "checkpoint":
                store.put_task_checkpoint(
                    TaskCheckpoint(
                        checkpoint_id="late",
                        task_id="t",
                        sequence=0,
                        phase="published",
                        created_at=datetime.now(UTC),
                    )
                )
            elif write == "workspace":
                store.put_project_workspace_revision(
                    ProjectWorkspaceRevision(
                        project_id="project",
                        revision=1,
                        payload={},
                        payload_sha256="a" * 64,
                        created_at=datetime.now(UTC),
                    )
                )
            elif write == "prompt":
                store.record_comfyui_prompt("t", "late-prompt")
            elif write == "child":
                store.add_task(task("late-child"))
            else:
                # Generic transaction fence applies before any manifest SQL is issued.
                with store._transaction(immediate=True) as conn:
                    conn.execute("DELETE FROM workload_manifests")
    finally:
        execution_guard.reset(token)
    assert store.get_task("t").state == TaskState.CANCELLING


def test_guard_allows_own_terminal_transition_but_not_followup_write(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    claimed = claim_local(store)
    token = execution_guard.set(("t", claimed.attempt_id))
    try:
        assert store.transition_task("t", TaskState.SUCCEEDED).state == TaskState.SUCCEEDED
        with pytest.raises(ValueError, match="no longer allowed"):
            store.record_comfyui_prompt("t", "late")
    finally:
        execution_guard.reset(token)


def test_retry_preserves_deadline_and_attempt_budget(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_local(store)
    store.transition_task("t", TaskState.FAILED)
    retried = store.prepare_task_retry("t")
    assert retried.attempt == first.attempt
    assert retried.deadline_at == first.deadline_at
    store.transition_task("t", TaskState.QUEUED)
    second = store.claim_local_task("t", "dispatcher")
    assert second.attempt == 2 and second.deadline_at == first.deadline_at
    store.transition_task("t", TaskState.FAILED)
    with pytest.raises(InvalidTaskTransitionError, match="deadline"):
        store.prepare_task_retry("t", now=first.deadline_at + timedelta(seconds=1))


def test_cancelled_and_exhausted_tasks_cannot_be_revived(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    claimed = claim_local(store, max_attempts=1)
    store.transition_task("t", TaskState.FAILED)
    with pytest.raises(InvalidTaskTransitionError, match="budget"):
        store.prepare_task_retry("t")
    store.transition_task("t", TaskState.CANCELLED)
    for mutation in (
        lambda: store.prepare_task_retry("t"),
        lambda: store.safe_explicit_retry("t"),
        lambda: store.schedule_task_retry("t", "late error"),
        lambda: store.confirm_ambiguous_retry(
            "t",
            expected_attempt_id=claimed.attempt_id,
            expected_submission_token=None,
        ),
    ):
        with pytest.raises((ValueError, InvalidTaskTransitionError)):
            mutation()
    assert store.get_task("t").state == TaskState.CANCELLED


def test_ambiguous_retry_requires_cas_and_audits_risk(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    claimed = claim_local(store)
    submission, _ = store.ensure_submission_intent("t")
    store.record_comfyui_prompt("t", "old-prompt")
    store.transition_task("t", TaskState.NEEDS_ATTENTION)
    with pytest.raises(ValueError, match="stop"):
        store.safe_explicit_retry("t")
    with pytest.raises(ValueError, match="superseded"):
        store.confirm_ambiguous_retry(
            "t",
            expected_attempt_id="wrong",
            expected_submission_token=submission,
        )
    ready = store.confirm_ambiguous_retry(
        "t",
        expected_attempt_id=claimed.attempt_id,
        expected_submission_token=submission,
    )
    assert ready.state == TaskState.READY and ready.comfyui_prompt_id is None
    assert ready.attempt == claimed.attempt and ready.deadline_at == claimed.deadline_at
    audit = next(
        event
        for event in store.list_execution_events("t")
        if event["kind"] == "duplicate_execution_risk_accepted"
    )
    assert audit["payload"]["prompt_id"] == "old-prompt"
    assert audit["payload"]["external_stop_confirmed"] is False


def test_two_recovered_gpu_observations_do_not_deadlock(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_local(store, "first")
    store.record_comfyui_prompt("first", "prompt-first")
    store.transition_task("first", TaskState.FAILED)
    second = claim_local(store, "second")
    store.record_comfyui_prompt("second", "prompt-second")
    store.transition_task("first", TaskState.RECOVERING)
    store.transition_task("second", TaskState.RECOVERING)
    for original in (first, second):
        observed = store.claim_local_task(original.task_id, "dispatcher")
        assert observed is not None
        assert observed.attempt == original.attempt
        with pytest.raises(ValueError, match="observation"):
            store.ensure_submission_intent(original.task_id)


def test_orchestration_yield_excludes_waiting_from_step_deadline(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db", llm_operation_timeout_seconds=41)
    first = claim_local(store, kind=TaskKind.LLM_PLANNING)
    assert 40 <= (first.deadline_at - datetime.now(UTC)).total_seconds() <= 41
    store.defer_orchestration("t")
    assert store.get_task("t").deadline_at is None
    store.transition_task("t", TaskState.QUEUED)
    second = store.claim_local_task("t", "dispatcher")
    assert second.attempt == first.attempt
    assert second.deadline_at >= first.deadline_at


def test_orchestration_interruption_reenters_checkpointed_service(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_local(store, kind=TaskKind.LLM_PLANNING)
    store.release_dispatcher("dispatcher")
    assert store.recover_expired_leases() == ("t",)
    assert store.get_task("t").state == TaskState.RECOVERING
    assert store.acquire_dispatcher("new-dispatcher")
    resumed = store.claim_local_task("t", "new-dispatcher")
    assert resumed.attempt == first.attempt and resumed.deadline_at == first.deadline_at


def test_remote_claim_is_atomic_across_store_instances(tmp_path):
    path = tmp_path / "tasks.db"
    store = SQLiteTaskStore(path)
    for name in ("a", "b"):
        store.add_task(task(name, remote=True))
        store.transition_task(name, TaskState.QUEUED)
    other = SQLiteTaskStore(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda current: current.claim_next(
                    "worker",
                    execution_target=ExecutionTarget.REMOTE,
                    lease_duration=timedelta(seconds=120),
                ),
                (store, other),
            )
        )
    assert len([item for item in results if item is not None]) == 1


def test_remote_claim_rechecks_stale_dependency_and_pause(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    parent = task("parent", kind=TaskKind.EXPORT).model_copy(update={"state": TaskState.SUCCEEDED})
    store.add_task(parent)
    child = task("child", remote=True).model_copy(update={"depends_on": ("parent",)})
    store.add_task(child)
    store.transition_task("child", TaskState.QUEUED)
    store.put_project_run_state(
        ProjectRunState(
            project_id="project",
            paused=True,
            updated_at=datetime.now(UTC),
        )
    )
    assert store.claim_next("worker", lease_duration=timedelta(seconds=120)) is None
    store.put_project_run_state(
        ProjectRunState(
            project_id="project",
            paused=False,
            updated_at=datetime.now(UTC),
        )
    )
    with store._transaction(immediate=True) as conn:
        conn.execute("UPDATE tasks SET state='stale' WHERE task_id='parent'")
    assert store.claim_next("worker", lease_duration=timedelta(seconds=120)) is None


def test_remote_claim_respects_member_pause_payload(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    store.add_task(task("remote", remote=True))
    store.transition_task("remote", TaskState.QUEUED)
    now = datetime.now(UTC)
    batch = BatchRun(
        batch_id="batch",
        name="batch",
        state=BatchState.RUNNING,
        items=(BatchRunItem(project_id="project", task_ids=("remote",)),),
        created_at=now,
        updated_at=now,
    )
    store.put_batch_run(batch)
    payload = batch.model_dump(mode="json")
    payload["items"][0]["paused"] = True
    with store._transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE batch_runs SET payload_json=? WHERE batch_id='batch'", (json.dumps(payload),)
        )
    assert store.claim_next("worker", lease_duration=timedelta(seconds=120)) is None


def test_remote_expiry_fences_renew_and_late_result_and_retains_slot(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_remote(store)
    assert first.attempt_id
    with pytest.raises(LeaseError, match="identity"):
        store.renew_lease("remote", "worker", lease_duration=timedelta(seconds=120))
    assert (
        store.renew_lease(
            "remote", "worker", attempt_id=first.attempt_id, lease_duration=timedelta(seconds=120)
        ).attempt
        == 1
    )
    expired_at = datetime.now(UTC) + timedelta(seconds=130)
    assert store.recover_expired_leases(now=expired_at) == ("remote",)
    assert store.get_task("remote").state == TaskState.NEEDS_ATTENTION
    report = TaskResultReport(
        report_id="a" * 64,
        task_id="remote",
        worker_id="worker",
        attempt=1,
        attempt_id=first.attempt_id,
        status=WorkerResultStatus.SUCCEEDED,
    )
    with pytest.raises(LeaseError):
        store.submit_remote_result(report, now=expired_at)
    store.add_task(task("other", remote=True))
    store.transition_task("other", TaskState.QUEUED)
    assert store.claim_next("worker", lease_duration=timedelta(seconds=120)) is None


def test_remote_result_identity_and_expiry_are_checked_before_mutation(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_remote(store)
    report = TaskResultReport(
        report_id="a" * 64,
        task_id="remote",
        worker_id="worker",
        attempt=1,
        attempt_id="wrong",
        status=WorkerResultStatus.SUCCEEDED,
    )
    with pytest.raises(LeaseError, match="identity"):
        store.submit_remote_result(report)
    correct = report.model_copy(update={"attempt_id": first.attempt_id})
    with pytest.raises(LeaseError, match="expired"):
        store.submit_remote_result(correct, now=first.lease_expires_at + timedelta(seconds=1))
    receipt = store.submit_remote_result(correct)
    assert store.submit_remote_result(correct) == receipt
    assert store.get_task("remote").state == TaskState.SUCCEEDED


def test_remote_unknown_outcome_is_not_safe_retryable(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_remote(store)
    store.submit_remote_result(
        TaskResultReport(
            report_id="a" * 64,
            task_id="remote",
            worker_id="worker",
            attempt=1,
            attempt_id=first.attempt_id,
            status=WorkerResultStatus.NEEDS_ATTENTION,
            error_code="worker_executor_error",
            error_message="network disconnected after submit",
        )
    )
    assert store.get_task("remote").state == TaskState.NEEDS_ATTENTION
    with pytest.raises(ValueError, match="stop"):
        store.safe_explicit_retry("remote")


def test_confirmed_failure_persists_backoff_and_new_attempt_fences_old_callback(tmp_path):
    path = tmp_path / "tasks.db"
    store = SQLiteTaskStore(path)
    first = claim_local(store)
    submission, _ = store.ensure_submission_intent("t")
    store.record_comfyui_prompt("t", "failed-prompt")
    waiting = store.finish_attempt_failure(
        "t",
        expected_attempt_id=first.attempt_id,
        expected_submission_token=submission,
        error_code="worker_unavailable",
        error_message="temporary resource shortage",
        retryable=True,
        stopped_evidence="history says execution failed",
    )
    assert waiting.state == TaskState.RETRY_WAIT
    assert waiting.next_retry_at is not None and waiting.comfyui_prompt_id is None
    assert waiting.attempt == 1 and waiting.deadline_at == first.deadline_at
    restarted = SQLiteTaskStore(path)
    assert restarted.get_task("t").next_retry_at == waiting.next_retry_at
    restarted.promote_due_retries(now=waiting.next_retry_at + timedelta(seconds=1))
    second = restarted.claim_local_task("t", "dispatcher")
    assert second.attempt == 2 and second.deadline_at == first.deadline_at
    with pytest.raises(ValueError, match="superseded"):
        restarted.finish_attempt_failure(
            "t",
            expected_attempt_id=first.attempt_id,
            expected_submission_token=submission,
            error_code="late",
            error_message="old completion",
            retryable=True,
        )
    assert restarted.get_task("t").attempt_id == second.attempt_id


def test_unconfirmed_failure_and_cancellation_never_schedule_replay(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_local(store)
    submission, _ = store.ensure_submission_intent("t")
    unknown = store.finish_attempt_failure(
        "t",
        expected_attempt_id=first.attempt_id,
        expected_submission_token=submission,
        error_code="read_timeout",
        error_message="response lost",
        retryable=True,
    )
    assert unknown.state == TaskState.NEEDS_ATTENTION and unknown.next_retry_at is None
    store.request_task_cancellation("t")
    assert store.get_task("t").state == TaskState.CANCELLING
    with pytest.raises(ValueError, match="superseded"):
        store.finish_attempt_failure(
            "t",
            expected_attempt_id=first.attempt_id,
            expected_submission_token=submission,
            error_code="late",
            error_message="late failure",
            retryable=True,
        )
    assert store.get_task("t").state == TaskState.CANCELLING


def test_remote_confirmed_retryable_failure_uses_persistent_budget(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    first = claim_remote(store)
    receipt = store.submit_remote_result(
        TaskResultReport(
            report_id="a" * 64,
            task_id="remote",
            worker_id="worker",
            attempt=1,
            attempt_id=first.attempt_id,
            status=WorkerResultStatus.FAILED,
            error_code="worker_busy",
            error_message="execution not started",
            retryable=True,
        )
    )
    assert receipt.state == "retry_wait"
    waiting = store.get_task("remote")
    assert waiting.next_retry_at is not None
    assert waiting.deadline_at == first.deadline_at and waiting.attempt == 1


@pytest.mark.asyncio
async def test_claim_transport_timeout_does_not_replay_claim():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("response lost", request=request)

    async with RemoteWorkerClient(
        control_plane_url="https://worker.test",
        token="test",
        retry_policy=RetryPolicy(attempts=3, initial_delay_seconds=0),
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RemoteWorkerError):
            await client.claim(
                WorkerHeartbeat(
                    worker_id="worker",
                    sent_at=datetime.now(UTC),
                    available_gpu_slots=1,
                )
            )
    assert calls == 1


def test_task_registration_checks_workspace_before_insert_and_idempotent_return(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    draft = task("image", kind=TaskKind.IMAGE_GENERATION)
    first = ProjectWorkspaceRevision(
        project_id="project",
        revision=1,
        payload={},
        payload_sha256="a" * 64,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(StoreConflictError, match="workspace changed"):
        store.add_task(draft, expected_workspace_sha256=first.payload_sha256)
    store.put_project_workspace_revision(first)
    stored = store.add_task(draft, expected_workspace_sha256=first.payload_sha256)
    assert stored.task_id == draft.task_id
    store.put_project_workspace_revision(
        first.model_copy(
            update={
                "revision": 2,
                "payload_sha256": "b" * 64,
            }
        )
    )
    with pytest.raises(StoreConflictError, match="workspace changed"):
        store.add_task(draft, expected_workspace_sha256=first.payload_sha256)
    assert len(store.list_tasks()) == 1
