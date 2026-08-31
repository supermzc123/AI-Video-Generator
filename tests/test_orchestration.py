import hashlib
import io
import json
import sqlite3
import zipfile
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from ai_video_generator.api import _apply_structured_patches, create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    DecisionLedger,
    DecisionSource,
    ExecutionMode,
    ExecutionTarget,
    HarnessBundle,
    HarnessRevision,
    HarnessSource,
    MemoryEventKind,
    ProjectMemoryEvent,
    ProjectRunState,
    ProjectSpec,
    ReviewDeadline,
    ReviewMode,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.llm import (
    JsonPatchOp,
    JsonPatchOperation,
    StructuredOperationResponse,
)
from ai_video_generator.persistence import SQLiteTaskStore, StoreConflictError
from ai_video_generator.services.harness_sources import (
    HarnessSourceInstallError,
    install_h3_harness_source,
    validate_h3_harness_source,
)


def task(
    task_id: str,
    *,
    state: TaskState = TaskState.READY,
    affinity: str | None = None,
    kind: TaskKind = TaskKind.H3_GENERATION,
) -> TaskSpec:
    value = sum(map(ord, task_id)) % 16
    return TaskSpec(
        task_id=task_id,
        project_id="project-1",
        kind=kind,
        state=state,
        idempotency_key=f"{value:x}" * 64,
        input_fingerprint=f"{(value + 1) % 16:x}" * 64,
        affinity_key=affinity,
    )


def test_mode_switch_waits_for_atomic_task_boundary(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "state.db")
    now = datetime(2026, 8, 15, tzinfo=UTC)
    store.put_project_run_state(ProjectRunState(project_id="project-1", updated_at=now))
    store.add_task(task("running", state=TaskState.RUNNING))

    pending = store.request_project_mode("project-1", ExecutionMode.BATCH, now=now)
    assert pending.execution_mode == ExecutionMode.GUIDED
    assert pending.pending_mode == ExecutionMode.BATCH

    store.transition_task("running", TaskState.SUCCEEDED, now=now)
    applied = store.apply_pending_project_mode("project-1", now=now)
    assert applied.execution_mode == ExecutionMode.BATCH
    assert applied.pending_mode is None


def test_review_timeout_switches_whole_project_to_ai_only(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "review.db")
    opened = datetime(2026, 8, 15, tzinfo=UTC)
    store.put_project_run_state(ProjectRunState(project_id="project-1", updated_at=opened))
    store.add_task(
        task("review", state=TaskState.NEEDS_REVIEW, kind=TaskKind.AI_REVIEW)
    )
    store.put_review_deadline(
        ReviewDeadline(
            project_id="project-1",
            task_id="review",
            opened_at=opened,
            deadline_at=opened + timedelta(minutes=10),
        )
    )

    assert store.apply_expired_review_deadlines(now=opened + timedelta(minutes=9)) == ()
    changed = store.apply_expired_review_deadlines(now=opened + timedelta(minutes=11))
    assert changed[0].review_policy.configured_mode == ReviewMode.HUMAN_AI
    assert changed[0].review_policy.effective_mode == ReviewMode.AI_ONLY


@pytest.mark.parametrize("accepted", [True, False])
def test_human_review_resolves_deadline_before_timeout(tmp_path, accepted: bool) -> None:
    store = SQLiteTaskStore(tmp_path / f"resolved-{accepted}.db")
    opened = datetime(2026, 8, 15, tzinfo=UTC)
    store.put_project_run_state(ProjectRunState(project_id="project-1", updated_at=opened))
    store.add_task(
        task("review", state=TaskState.NEEDS_REVIEW, kind=TaskKind.AI_REVIEW)
    )
    store.put_review_deadline(
        ReviewDeadline(
            project_id="project-1",
            task_id="review",
            opened_at=opened,
            deadline_at=opened + timedelta(seconds=30),
        )
    )

    store.review_task(
        "review",
        accepted=accepted,
        feedback="接受" if accepted else "连续性需要返工",
        now=opened + timedelta(seconds=10),
    )

    assert store.apply_expired_review_deadlines(now=opened + timedelta(seconds=31)) == ()
    assert (
        store.get_project_run_state("project-1").review_policy.effective_mode
        == ReviewMode.HUMAN_AI
    )
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM review_deadlines").fetchone()[0] == 0


def test_expired_deadlines_ignore_terminal_and_non_review_tasks(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "irrelevant-deadlines.db")
    opened = datetime(2026, 8, 15, tzinfo=UTC)
    store.put_project_run_state(ProjectRunState(project_id="project-1", updated_at=opened))
    store.add_task(task("finished", state=TaskState.SUCCEEDED, kind=TaskKind.AI_REVIEW))
    store.add_task(task("not-ai", state=TaskState.NEEDS_REVIEW))
    for task_id in ("finished", "not-ai"):
        store.put_review_deadline(
            ReviewDeadline(
                project_id="project-1",
                task_id=task_id,
                opened_at=opened,
                deadline_at=opened + timedelta(seconds=30),
            )
        )

    assert store.apply_expired_review_deadlines(now=opened + timedelta(seconds=31)) == ()
    assert (
        store.get_project_run_state("project-1").review_policy.effective_mode
        == ReviewMode.HUMAN_AI
    )
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM review_deadlines").fetchone()[0] == 0


def test_memory_search_and_locked_decision_ledger(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "memory.db")
    now = datetime(2026, 8, 15, tzinfo=UTC)
    event = ProjectMemoryEvent(
        event_id="event-1",
        project_id="project-1",
        kind=MemoryEventKind.CONSTRAINT,
        source=DecisionSource.USER,
        role="user",
        content="The protagonist always wears a red coat.",
        created_at=now,
    )
    store.add_memory_event(event)
    assert store.list_memory_events("project-1", query="protagonist") == (event,)

    decision = DecisionLedger(
        decision_id="decision-1",
        project_id="project-1",
        key="character.costume",
        value={"coat": "red"},
        rationale="User approved it.",
        source=DecisionSource.USER,
        locked=True,
        created_at=now,
    )
    store.add_decision(decision)
    with pytest.raises(StoreConflictError, match="locked"):
        store.add_decision(
            decision.model_copy(update={"decision_id": "decision-2", "value": {"coat": "blue"}})
        )


def test_affinity_bonus_does_not_prevent_aging(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "scheduler.db")
    start = datetime(2026, 8, 15, tzinfo=UTC)
    store.add_task(task("old-other", affinity="image"))
    store.add_task(task("new-match", affinity="h3"))
    store.transition_task("old-other", TaskState.QUEUED)
    store.transition_task("new-match", TaskState.QUEUED)
    with sqlite3.connect(tmp_path / "scheduler.db") as connection:
        connection.execute(
            "UPDATE tasks SET created_at = ? WHERE task_id = ?",
            ((start - timedelta(hours=2)).timestamp(), "old-other"),
        )
        connection.execute(
            "UPDATE tasks SET created_at = ? WHERE task_id = ?",
            (start.timestamp(), "new-match"),
        )
    claimed = store.claim_next(
        "gpu-0",
        lease_duration=timedelta(minutes=1),
        resident_affinity_key="h3",
        now=start,
    )
    assert claimed is not None
    assert claimed.task_id == "old-other"


@pytest.mark.asyncio
async def test_batch_pause_freezes_dispatch_and_pauses_queued_tasks(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    now = datetime(2026, 8, 15, tzinfo=UTC)
    project = ProjectSpec(
        project_id="project-1", name="Film", target_duration_seconds=30
    )
    queued = task("queued", state=TaskState.QUEUED)
    batch = BatchRun(
        batch_id="batch-1",
        name="Night run",
        state=BatchState.RUNNING,
        items=(BatchRunItem(project_id="project-1", task_ids=(queued.task_id,)),),
        created_at=now,
        updated_at=now,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created_project = await client.post(
            "/api/v1/projects", json=project.model_dump(mode="json")
        )
        created_task = await client.post(
            "/api/v1/tasks", json=queued.model_dump(mode="json")
        )
        created_batch = await client.post(
            "/api/v1/batches", json=batch.model_dump(mode="json")
        )
        assert created_project.status_code == 201
        assert created_task.status_code == 201
        assert created_batch.status_code == 201

        paused = await client.post("/api/v1/batches/batch-1/pause")
        stored_task = await client.get("/api/v1/tasks/queued")
        run_state = await client.get("/api/v1/projects/project-1/run-state")

    assert paused.status_code == 200
    assert paused.json()["state"] == "paused"
    assert stored_task.json()["state"] == "paused"
    assert run_state.json()["paused"] is True


@pytest.mark.asyncio
async def test_batch_cancel_cancels_selected_queue_and_returns_project_to_guided(
    tmp_path,
) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    now = datetime(2026, 8, 15, tzinfo=UTC)
    project = ProjectSpec(
        project_id="project-1", name="Film", target_duration_seconds=30
    )
    queued = task("queued", state=TaskState.QUEUED)
    batch = BatchRun(
        batch_id="batch-1",
        name="Night run",
        state=BatchState.RUNNING,
        items=(BatchRunItem(project_id="project-1", task_ids=(queued.task_id,)),),
        created_at=now,
        updated_at=now,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post("/api/v1/projects", json=project.model_dump(mode="json"))
        await client.post("/api/v1/tasks", json=queued.model_dump(mode="json"))
        await client.post("/api/v1/batches", json=batch.model_dump(mode="json"))

        cancelled = await client.post("/api/v1/batches/batch-1/cancel")
        stored_task = await client.get("/api/v1/tasks/queued")
        run_state = await client.get("/api/v1/projects/project-1/run-state")

    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert stored_task.json()["state"] == "cancelled"
    assert run_state.json()["execution_mode"] == "guided"
    assert run_state.json()["paused"] is False


@pytest.mark.asyncio
async def test_running_local_task_cancels_comfyui_before_database_transition(tmp_path) -> None:
    requests: list[tuple[str, str]] = []

    def comfy_handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(
                200,
                json={"queue_running": [[1, "prompt-1", {}, {}, []]], "queue_pending": []},
            )
        if request.method == "POST" and request.url.path == "/interrupt":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    app = create_app(
        Settings(_env_file=None, data_root=tmp_path, comfyui_base_url="http://comfy"),
        comfyui_transport=httpx.MockTransport(comfy_handler),
    )
    running = task("running", state=TaskState.RUNNING).model_copy(
        update={
            "execution_target": ExecutionTarget.LOCAL,
            "comfyui_prompt_id": "prompt-1",
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/v1/tasks", json=running.model_dump(mode="json")
        )
        assert created.status_code == 201
        cancelled = await client.post("/api/v1/tasks/running/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert requests == [("GET", "/queue"), ("POST", "/interrupt")]


def test_harness_revisions_are_versioned_and_immutable(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "harness.db")
    bundle = store.put_harness_bundle(
        HarnessBundle(harness_id="image:test", name="Test", purpose="image_prompting")
    )
    refreshed = store.put_harness_bundle(
        bundle.model_copy(update={"name": "Updated", "workflow_template_id": "image-v2"})
    )
    assert refreshed.name == "Updated"
    assert store.list_harness_bundles()[0].workflow_template_id == "image-v2"
    markdown = "Write an image prompt for {{prompt}}."
    payload = {
        "markdown": markdown,
        "input_schema": {},
        "output_schema": {},
        "workflow_template_id": None,
        "workflow_revision": None,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    revision = HarnessRevision(
        harness_id="image:test",
        revision=1,
        markdown=markdown,
        content_sha256=digest,
        created_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    assert store.put_harness_revision(revision) == revision
    with pytest.raises(StoreConflictError, match="immutable"):
        store.put_harness_revision(revision.model_copy(update={"markdown": "Changed"}))


def test_harness_source_accepts_pinned_git_sha() -> None:
    source = HarnessSource(
        repository_url="https://github.com/example/repository",
        commit="a" * 40,
    )
    assert source.commit == "a" * 40


@pytest.mark.asyncio
async def test_workspace_and_guided_task_control_api(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    project = {
        "project_id": "project-1",
        "name": "First full run",
        "target_duration_seconds": 60,
    }
    queued_task = task("guided-task")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.post("/api/v1/projects", json=project)).status_code == 201
        saved = await client.post(
            "/api/v1/projects/project-1/workspace",
            json={"revision": 1, "payload": {"outline": [{"title": "Opening"}]}},
        )
        created = await client.post("/api/v1/tasks", json=queued_task.model_dump(mode="json"))
        queued = await client.post("/api/v1/tasks/guided-task/run")
        paused = await client.post("/api/v1/tasks/guided-task/pause")
        resumed = await client.post("/api/v1/tasks/guided-task/resume")
        workspace = await client.get("/api/v1/projects/project-1/workspace")

    assert saved.status_code == 201
    assert created.status_code == 201
    assert queued.json()["state"] == "queued"
    assert paused.json()["state"] == "paused"
    assert resumed.json()["state"] == "ready"
    assert workspace.json()["payload"]["outline"][0]["title"] == "Opening"


@pytest.mark.asyncio
async def test_run_endpoint_resets_failed_comfyui_execution_before_retry(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    failed = task("failed-comfy", state=TaskState.FAILED).model_copy(
        update={
            "comfyui_prompt_id": "old-failed-prompt",
            "error_code": "comfy_failed",
            "error_message": "old failure",
            "attempt": 3,
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/api/v1/tasks", json=failed.model_dump(mode="json"))
        retried = await client.post("/api/v1/tasks/failed-comfy/run")

    assert created.status_code == 201
    assert retried.status_code == 200
    assert retried.json()["state"] == "queued"
    assert retried.json()["comfyui_prompt_id"] is None
    assert retried.json()["error_code"] is None
    assert retried.json()["error_message"] is None
    assert retried.json()["attempt"] == 2


@pytest.mark.asyncio
async def test_run_endpoint_reuses_completed_comfyui_job_when_output_collection_failed(
    tmp_path,
) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    failed = task("failed-collection", state=TaskState.FAILED).model_copy(
        update={
            "comfyui_prompt_id": "completed-prompt",
            "error_code": "local_output_collection_failed",
            "error_message": "ReadTimeout",
            "attempt": 3,
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/api/v1/tasks", json=failed.model_dump(mode="json"))
        retried = await client.post("/api/v1/tasks/failed-collection/run")

    assert created.status_code == 201
    assert retried.status_code == 200
    assert retried.json()["state"] == "queued"
    assert retried.json()["comfyui_prompt_id"] == "completed-prompt"
    assert retried.json()["error_code"] is None
    assert retried.json()["error_message"] is None
    assert retried.json()["attempt"] == 2


def test_project_agent_patches_are_applied_without_touching_locked_siblings() -> None:
    response = StructuredOperationResponse(
        operation_id="op-1",
        patches=(
            JsonPatchOperation(
                op=JsonPatchOp.REPLACE,
                path="/shots/0/prompt",
                value="Revised prompt",
            ),
        ),
        rationale="Improve motion clarity.",
    )
    source = {"shots": [{"prompt": "Old", "seed": 42}]}
    patched = _apply_structured_patches(source, response)
    assert patched == {"shots": [{"prompt": "Revised prompt", "seed": 42}]}
    assert source["shots"][0]["prompt"] == "Old"


@pytest.mark.asyncio
async def test_pinned_harness_installer_extracts_documents_only(tmp_path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("repo-commit/skills/director/SKILL.md", "# Director")
        bundle.writestr("repo-commit/skills/director/tool.py", "raise RuntimeError()")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=archive.getvalue())

    installed = await install_h3_harness_source(
        "community", tmp_path, transport=httpx.MockTransport(handler)
    )
    assert installed.files == ("skills/director/SKILL.md",)
    assert (
        tmp_path / "harness-sources" / "community" / installed.commit / "install-manifest.json"
    ).is_file()
    assert not (
        tmp_path / "harness-sources" / "community" / installed.commit / "skills/director/tool.py"
    ).exists()


@pytest.mark.asyncio
async def test_official_harness_installer_uses_tree_documents(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "git/trees" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {"path": "skills/h3-prompt-writing/SKILL.md", "type": "blob"},
                        {
                            "path": "skills/h3-prompt-writing/references/base-en.txt",
                            "type": "blob",
                        },
                        {
                            "path": "skills/h3-prompt-writing/references/ref-en.txt",
                            "type": "blob",
                        },
                        {"path": "large-model.bin", "type": "blob"},
                    ]
                },
            )
        return httpx.Response(200, content=b"# Official H3 prompt writing")

    installed = await install_h3_harness_source(
        "official", tmp_path, transport=httpx.MockTransport(handler)
    )
    assert installed.files == (
        "skills/h3-prompt-writing/SKILL.md",
        "skills/h3-prompt-writing/references/base-en.txt",
        "skills/h3-prompt-writing/references/ref-en.txt",
    )
    assert installed.redistribution_allowed is False
    validate_h3_harness_source(installed.install_root, source_id="official")

    installed_skill = (
        tmp_path
        / "harness-sources"
        / "official"
        / installed.commit
        / "skills"
        / "h3-prompt-writing"
        / "SKILL.md"
    )
    installed_skill.write_text("tampered", encoding="utf-8")
    with pytest.raises(HarnessSourceInstallError, match="hash mismatch"):
        validate_h3_harness_source(installed.install_root, source_id="official")


@pytest.mark.asyncio
async def test_official_harness_installer_rejects_missing_references(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/SKILL.md"):
            return httpx.Response(200, content=b"# Skill")
        return httpx.Response(404)

    with pytest.raises(HarnessSourceInstallError, match="failed to download"):
        await install_h3_harness_source(
            "official", tmp_path, transport=httpx.MockTransport(handler)
        )
