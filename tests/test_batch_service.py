import hashlib
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ExecutionTarget,
    ProjectSpec,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore, StoreConflictError
from ai_video_generator.services.batch_runs import (
    batch_project_tasks,
    reconcile_batch_runs,
    resolve_batch_run,
)


def _task(
    task_id: str,
    kind: TaskKind,
    *,
    state: TaskState,
    depends_on: tuple[str, ...] = (),
    project_id: str = "project-1",
) -> TaskSpec:
    fingerprint = hashlib.sha256(task_id.encode()).hexdigest()
    return TaskSpec(
        task_id=task_id,
        project_id=project_id,
        kind=kind,
        state=state,
        idempotency_key=fingerprint,
        input_fingerprint=fingerprint,
        depends_on=depends_on,
    )


def _batch(boundary: str, *, task_ids: tuple[str, ...] = ()) -> BatchRun:
    now = datetime(2026, 8, 16, tzinfo=UTC)
    return BatchRun(
        batch_id="batch-1",
        name="Night run",
        items=(
            BatchRunItem(
                project_id="project-1",
                task_ids=task_ids,
                start_boundary=boundary,
                priority=17,
            ),
        ),
        created_at=now,
        updated_at=now,
    )


def test_review_boundary_resolves_required_ancestors_and_all_successors(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch.db")
    store.add_task(_task("image", TaskKind.IMAGE_GENERATION, state=TaskState.SUCCEEDED))
    store.add_task(
        _task(
            "h3",
            TaskKind.H3_GENERATION,
            state=TaskState.READY,
            depends_on=("image",),
        )
    )
    store.add_task(
        _task("review", TaskKind.AI_REVIEW, state=TaskState.BLOCKED, depends_on=("h3",))
    )
    store.add_task(
        _task("export", TaskKind.EXPORT, state=TaskState.BLOCKED, depends_on=("review",))
    )

    resolved = resolve_batch_run(store, _batch("review"))

    assert resolved.items[0].task_ids == ("h3", "review", "export")
    assert store.get_task("h3").priority == 17


def test_client_task_ids_cannot_create_an_orphaned_batch_successor(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch.db")
    store.add_task(_task("h3", TaskKind.H3_GENERATION, state=TaskState.READY))
    store.add_task(
        _task("review", TaskKind.AI_REVIEW, state=TaskState.BLOCKED, depends_on=("h3",))
    )
    store.add_task(
        _task("export", TaskKind.EXPORT, state=TaskState.BLOCKED, depends_on=("review",))
    )

    resolved = resolve_batch_run(store, _batch("review", task_ids=("export",)))

    assert resolved.items[0].task_ids == ("h3", "review", "export")


def test_project_cannot_join_two_active_batches(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch.db")
    store.add_task(_task("selected", TaskKind.H3_GENERATION, state=TaskState.READY))
    first = _batch("next_ready", task_ids=("selected",))
    store.put_batch_run(resolve_batch_run(store, first))
    second = first.model_copy(
        update={
            "batch_id": "batch-2",
            "name": "Second",
            "created_at": first.created_at + timedelta(seconds=1),
            "updated_at": first.updated_at + timedelta(seconds=1),
        }
    )

    with pytest.raises(StoreConflictError, match="already belongs to an active batch"):
        resolve_batch_run(store, second)


def test_batch_project_tasks_are_limited_to_frozen_member_ids(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch.db")
    store.add_task(_task("selected", TaskKind.H3_GENERATION, state=TaskState.READY))
    store.add_task(_task("outside", TaskKind.EXPORT, state=TaskState.READY))
    batch = _batch("next_ready", task_ids=("selected",))

    tasks = batch_project_tasks(store, batch, "project-1")

    assert tuple(task.task_id for task in tasks) == ("selected",)


def test_failed_project_is_settled_without_blocking_other_batch_items(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch.db")
    store.add_task(_task("failed", TaskKind.H3_GENERATION, state=TaskState.FAILED))
    store.add_task(
        _task("blocked", TaskKind.AI_REVIEW, state=TaskState.BLOCKED, depends_on=("failed",))
    )
    store.add_task(
        _task(
            "other-ready",
            TaskKind.H3_GENERATION,
            state=TaskState.READY,
            project_id="project-2",
        )
    )
    now = datetime(2026, 8, 16, tzinfo=UTC)
    running = _batch("next_ready").model_copy(
        update={
            "state": BatchState.RUNNING,
            "items": (
                BatchRunItem(
                    project_id="project-1",
                    task_ids=("failed", "blocked"),
                    start_boundary="next_ready",
                ),
                BatchRunItem(
                    project_id="project-2",
                    task_ids=("other-ready",),
                    start_boundary="next_ready",
                ),
            ),
            "updated_at": now,
        }
    )
    store.put_batch_run(running)

    reconciled = reconcile_batch_runs(store, now=now)

    assert store.get_task("blocked").state == TaskState.CANCELLED
    assert store.get_task("other-ready").state == TaskState.READY
    assert reconciled[0].state == BatchState.RUNNING

    store.transition_task("other-ready", TaskState.QUEUED)
    store.transition_task("other-ready", TaskState.RUNNING)
    store.transition_task("other-ready", TaskState.SUCCEEDED)
    reconciled = reconcile_batch_runs(store, now=now)

    assert reconciled[0].state == BatchState.COMPLETED


@pytest.mark.asyncio
async def test_create_batch_api_returns_server_resolved_task_closure(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    h3 = _task("h3", TaskKind.H3_GENERATION, state=TaskState.READY)
    review = _task(
        "review", TaskKind.AI_REVIEW, state=TaskState.BLOCKED, depends_on=("h3",)
    )
    export = _task(
        "export", TaskKind.EXPORT, state=TaskState.BLOCKED, depends_on=("review",)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for task in (h3, review, export):
            response = await client.post("/api/v1/tasks", json=task.model_dump(mode="json"))
            assert response.status_code == 201
        response = await client.post(
            "/api/v1/batches", json=_batch("review", task_ids=("export",)).model_dump(mode="json")
        )

    assert response.status_code == 201
    assert response.json()["items"][0]["task_ids"] == ["h3", "review", "export"]


@pytest.mark.asyncio
async def test_outline_approved_project_enters_batch_before_dag_compilation(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    now = datetime.now(UTC)
    batch = BatchRun(
        batch_id="batch-outline",
        name="Outline-first batch",
        items=(BatchRunItem(project_id="project-outline"),),
        created_at=now,
        updated_at=now,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/v1/projects",
            json=ProjectSpec(
                project_id="project-outline",
                name="Outline project",
                target_duration_seconds=30,
            ).model_dump(mode="json"),
        )
        assert created.status_code == 201
        workspace = await client.post(
            "/api/v1/projects/project-outline/workspace",
            json={
                "revision": 1,
                "payload": {
                    "revision": 1,
                    "activeStage": "outline",
                    "stageApprovals": {"outline": now.isoformat()},
                    "outline": [{"id": "beat-1", "title": "开端"}],
                    "shots": [],
                    "assetPlans": [],
                    "referenceAssetMode": "planned",
                    "prompts": {"imagePrompts": [], "h3Prompts": []},
                },
            },
        )
        assert workspace.status_code == 201
        response = await client.post("/api/v1/batches", json=batch.model_dump(mode="json"))

    assert response.status_code == 201
    task_ids = response.json()["items"][0]["task_ids"]
    assert len(task_ids) == 1
    assert task_ids[0].startswith("batch-plan:project-outline:")


@pytest.mark.asyncio
async def test_batch_admission_uses_latest_committed_workspace_progress(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    now = datetime.now(UTC)
    project = ProjectSpec(
        project_id="committed-assets",
        revision=1,
        name="Committed assets project",
        target_duration_seconds=30,
    )
    batch = BatchRun(
        batch_id="batch-committed-assets",
        name="Continue from assets",
        items=(BatchRunItem(project_id=project.project_id),),
        created_at=now,
        updated_at=now,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/v1/projects", json=project.model_dump(mode="json")
        )
        assert created.status_code == 201
        committed_project = project.model_copy(update={"revision": 2})
        committed = await client.post(
            f"/api/v1/projects/{project.project_id}/commit",
            json={
                "project": committed_project.model_dump(mode="json"),
                "payload": {
                    "revision": 2,
                    "activeStage": "assets",
                    "stageApprovals": {
                        "outline": now.isoformat(),
                        "storyboard": now.isoformat(),
                    },
                    "outline": [{"id": "beat-1", "title": "Opening"}],
                    "shots": [{"id": "shot-1", "title": "Arrival"}],
                },
            },
        )
        assert committed.status_code == 201

        state = await client.get(f"/api/v1/projects/{project.project_id}/run-state")
        admitted = await client.post(
            "/api/v1/batches", json=batch.model_dump(mode="json")
        )

    assert state.status_code == 200
    assert state.json()["outline_approved"] is True
    assert state.json()["current_stage"] == "assets"
    assert admitted.status_code == 201


@pytest.mark.asyncio
async def test_outline_only_project_defers_boundary_filter_until_after_compilation(
    tmp_path,
) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    now = datetime.now(UTC)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/projects",
            json=ProjectSpec(
                project_id="outline-review",
                name="Outline review",
                target_duration_seconds=30,
            ).model_dump(mode="json"),
        )
        await client.post(
            "/api/v1/projects/outline-review/workspace",
            json={
                "revision": 1,
                "payload": {
                    "stageApprovals": {"outline": now.isoformat()},
                    "outline": [{"id": "beat-1", "title": "Opening"}],
                },
            },
        )
        batch = BatchRun(
            batch_id="deferred-review",
            name="Deferred review",
            items=(
                BatchRunItem(project_id="outline-review", start_boundary="review"),
            ),
            created_at=now,
            updated_at=now,
        )
        response = await client.post(
            "/api/v1/batches", json=batch.model_dump(mode="json")
        )

    assert response.status_code == 201
    assert response.json()["items"][0]["start_boundary"] == "review"
    assert response.json()["items"][0]["task_ids"][0].startswith(
        "batch-plan:outline-review:"
    )


@pytest.mark.asyncio
async def test_cancel_batch_project_interrupts_its_job_and_keeps_other_project_running(
    tmp_path,
) -> None:
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
    running = _task(
        "project-1-running", TaskKind.H3_GENERATION, state=TaskState.RUNNING
    ).model_copy(
        update={
            "execution_target": ExecutionTarget.LOCAL,
            "comfyui_prompt_id": "prompt-1",
        }
    )
    other = _task(
        "project-2-queued",
        TaskKind.H3_GENERATION,
        state=TaskState.QUEUED,
        project_id="project-2",
    )
    now = datetime(2026, 8, 16, tzinfo=UTC)
    batch = BatchRun(
        batch_id="batch-two-projects",
        name="Two projects",
        state=BatchState.RUNNING,
        items=(
            BatchRunItem(project_id="project-1", start_boundary="next_ready"),
            BatchRunItem(project_id="project-2", start_boundary="next_ready"),
        ),
        created_at=now,
        updated_at=now,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for project_id in ("project-1", "project-2"):
            response = await client.post(
                "/api/v1/projects",
                json=ProjectSpec(
                    project_id=project_id,
                    name=project_id,
                    target_duration_seconds=4,
                ).model_dump(mode="json"),
            )
            assert response.status_code == 201
        for task in (running, other):
            response = await client.post("/api/v1/tasks", json=task.model_dump(mode="json"))
            assert response.status_code == 201
        response = await client.post("/api/v1/batches", json=batch.model_dump(mode="json"))
        assert response.status_code == 201

        cancelled = await client.post(
            "/api/v1/batches/batch-two-projects/projects/project-1/cancel"
        )
        first = await client.get("/api/v1/tasks/project-1-running")
        second = await client.get("/api/v1/tasks/project-2-queued")

    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "running"
    assert first.json()["state"] == "cancelled"
    assert second.json()["state"] == "queued"
    assert ("POST", "/interrupt") in requests
