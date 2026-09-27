import hashlib
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from ai_video_generator.api import (
    _batch_settings_for_task,
    _reactivate_completed_batch_for_task,
    create_app,
)
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


def test_exhausted_local_orchestration_restarts_as_new_batch_member(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "restart.db")
    original = _task("plan", TaskKind.LLM_PLANNING, state=TaskState.FAILED)
    store.add_task(original)
    with store._transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE tasks SET attempt=3,error_code='local_control_task_failed',"
            "error_message='missing workflow' WHERE task_id='plan'"
        )
    batch = _batch("next_ready", task_ids=("plan",)).model_copy(
        update={"state": BatchState.COMPLETED}
    )
    store.put_batch_run(batch)
    assert "restart" in store.get_task("plan").available_actions

    replacement = store.restart_failed_orchestration("plan")
    assert replacement.task_id != "plan"
    assert (replacement.state, replacement.attempt, replacement.max_attempts) == (
        TaskState.QUEUED, 0, 3
    )
    assert (store.get_task("plan").state, store.get_task("plan").attempt) == (
        TaskState.FAILED, 3
    )
    updated = store.list_batch_runs()[0]
    assert updated.state == BatchState.RUNNING
    assert updated.items[0].task_ids == (replacement.task_id,)
    assert "restart" not in store.get_task("plan").available_actions


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
    store.add_task(_task("review", TaskKind.AI_REVIEW, state=TaskState.BLOCKED, depends_on=("h3",)))
    store.add_task(
        _task("export", TaskKind.EXPORT, state=TaskState.BLOCKED, depends_on=("review",))
    )

    resolved = resolve_batch_run(store, _batch("review"))

    assert resolved.items[0].task_ids == ("h3", "review", "export")
    assert store.get_task("h3").priority == 17


def test_client_task_ids_cannot_create_an_orphaned_batch_successor(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch.db")
    store.add_task(_task("h3", TaskKind.H3_GENERATION, state=TaskState.READY))
    store.add_task(_task("review", TaskKind.AI_REVIEW, state=TaskState.BLOCKED, depends_on=("h3",)))
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


def test_next_ready_resumes_failed_task_without_creating_new_planning(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch.db")
    store.add_task(_task("completed", TaskKind.H3_GENERATION, state=TaskState.SUCCEEDED))
    store.add_task(
        _task(
            "failed",
            TaskKind.H3_GENERATION,
            state=TaskState.FAILED,
            depends_on=("completed",),
        )
    )
    store.add_task(
        _task("delivery", TaskKind.EXPORT, state=TaskState.BLOCKED, depends_on=("failed",))
    )

    resolved = resolve_batch_run(store, _batch("next_ready"))

    assert resolved.items[0].task_ids == ("failed", "delivery")
    assert not any(task.kind == TaskKind.LLM_PLANNING for task in store.list_tasks())


def test_failed_dependency_stays_blocked_without_stalling_other_batch_items(tmp_path) -> None:
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

    assert store.get_task("blocked").state == TaskState.BLOCKED
    assert "failed" in store.get_task("blocked").blocked_reason
    assert store.get_task("other-ready").state == TaskState.READY
    assert reconciled[0].state == BatchState.RUNNING

    store.transition_task("other-ready", TaskState.QUEUED)
    store.transition_task("other-ready", TaskState.RUNNING)
    store.transition_task("other-ready", TaskState.SUCCEEDED)
    reconciled = reconcile_batch_runs(store, now=now)

    assert reconciled[0].state == BatchState.RUNNING


@pytest.mark.asyncio
async def test_create_batch_api_returns_server_resolved_task_closure(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    h3 = _task("h3", TaskKind.H3_GENERATION, state=TaskState.READY)
    review = _task("review", TaskKind.AI_REVIEW, state=TaskState.BLOCKED, depends_on=("h3",))
    export = _task("export", TaskKind.EXPORT, state=TaskState.BLOCKED, depends_on=("review",))
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
async def test_batch_api_accepts_default_and_project_settings_with_existing_queue(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    queued = _task("queued", TaskKind.H3_GENERATION, state=TaskState.QUEUED)
    batch = _batch("next_ready").model_copy(
        update={
            "settings": {
                "imageWorkflowId": "image-default",
                "seedvrUpscaleFactor": 1.5,
                "rifeEnabled": True,
            },
            "items": (
                _batch("next_ready")
                .items[0]
                .model_copy(update={"settings": {"rifeWorkflowId": "rife-project"}}),
            ),
        }
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.post("/api/v1/tasks", json=queued.model_dump(mode="json"))
        ).status_code == 201
        response = await client.post("/api/v1/batches", json=batch.model_dump(mode="json"))

    assert response.status_code == 201, response.text
    assert response.json()["settings"]["rifeEnabled"] is True
    assert response.json()["settings"]["seedvrUpscaleFactor"] == 1.5
    assert response.json()["items"][0]["settings"]["rifeWorkflowId"] == "rife-project"


def test_completed_batch_retains_settings_for_failed_automation_retry(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "batch-retry.db")
    automation = _task("automation", TaskKind.LLM_PLANNING, state=TaskState.FAILED)
    store.add_task(automation)
    now = datetime.now(UTC)
    store.put_batch_run(
        BatchRun(
            batch_id="completed-batch",
            name="Completed batch",
            state=BatchState.COMPLETED,
            settings={
                "seedvrEnabled": True,
                "seedvrWorkflowId": "user:restoration",
            },
            items=(
                BatchRunItem(
                    project_id=automation.project_id,
                    task_ids=(automation.task_id,),
                    settings={
                        "rifeEnabled": True,
                        "rifeWorkflowId": "user:interpolation",
                    },
                ),
            ),
            created_at=now,
            updated_at=now,
        )
    )

    assert _batch_settings_for_task(store, automation) == {
        "seedvrEnabled": True,
        "seedvrWorkflowId": "user:restoration",
        "rifeEnabled": True,
        "rifeWorkflowId": "user:interpolation",
    }
    reactivated = _reactivate_completed_batch_for_task(store, automation)
    assert reactivated is not None
    assert reactivated.state == BatchState.RUNNING


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
        created = await client.post("/api/v1/projects", json=project.model_dump(mode="json"))
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
        admitted = await client.post("/api/v1/batches", json=batch.model_dump(mode="json"))

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
            items=(BatchRunItem(project_id="outline-review", start_boundary="review"),),
            created_at=now,
            updated_at=now,
        )
        response = await client.post("/api/v1/batches", json=batch.model_dump(mode="json"))

    assert response.status_code == 201
    assert response.json()["items"][0]["start_boundary"] == "review"
    assert response.json()["items"][0]["task_ids"][0].startswith("batch-plan:outline-review:")


@pytest.mark.asyncio
async def test_cancel_batch_project_interrupts_its_job_and_keeps_other_project_running(
    tmp_path,
) -> None:
    requests: list[tuple[str, str]] = []
    interrupted = False

    def comfy_handler(request: httpx.Request) -> httpx.Response:
        nonlocal interrupted
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(
                200,
                json={
                    "queue_running": [] if interrupted else [[1, "prompt-1", {}, {}, []]],
                    "queue_pending": [],
                },
            )
        if request.method == "POST" and request.url.path == "/interrupt":
            interrupted = True
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

        dynamically_created = _task(
            "project-1-dynamic",
            TaskKind.IMAGE_GENERATION,
            state=TaskState.QUEUED,
        )
        response = await client.post(
            "/api/v1/tasks", json=dynamically_created.model_dump(mode="json")
        )
        assert response.status_code == 201

        cancelled = await client.post(
            "/api/v1/batches/batch-two-projects/projects/project-1/cancel"
        )
        first = await client.get("/api/v1/tasks/project-1-running")
        dynamic = await client.get("/api/v1/tasks/project-1-dynamic")
        second = await client.get("/api/v1/tasks/project-2-queued")

    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "running"
    assert first.json()["state"] == "cancelled"
    # This task was created outside the orchestrator context and never joined
    # this batch. A batch-scoped command must not cancel unrelated project work.
    assert dynamic.json()["state"] == "queued"
    assert second.json()["state"] == "queued"
    assert ("POST", "/interrupt") in requests
