from datetime import UTC, datetime

import httpx
import pytest
from fastapi import HTTPException
from test_batch_orchestration_flow import Scenario, _workspace

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ProjectRunState,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.services.batch_orchestration import OrchestrationNeedsAttention, _Progress
from ai_video_generator.services.batch_runs import reconcile_batch_runs, resolve_batch_task_ids


def setup(tmp_path, *, batch_state=BatchState.COMPLETED, attempts=1):
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    store.add_task(
        TaskSpec(
            task_id="parent",
            project_id="p",
            kind=TaskKind.LLM_PLANNING,
            state=TaskState.FAILED,
            attempt=attempts,
            idempotency_key="a" * 64,
            input_fingerprint="b" * 64,
        )
    )
    now = datetime.now(UTC)
    store.put_project_run_state(
        ProjectRunState(
            project_id="p",
            outline_approved=True,
            updated_at=now,
        )
    )
    store.put_batch_run(
        BatchRun(
            batch_id="batch",
            name="retry",
            state=batch_state,
            items=(BatchRunItem(project_id="p", task_ids=("parent",)),),
            created_at=now,
            updated_at=now,
        )
    )
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    return store, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize("route", ["command", "run", "redo"])
async def test_retry_entry_points_reactivate_completed_batch_without_resetting_budget(
    tmp_path, route
):
    store, client = setup(tmp_path)
    async with client:
        if route == "command":
            result = await client.post(
                "/api/v1/task-commands",
                json={
                    "scope": "project",
                    "scope_id": "p",
                    "action": "retry",
                    "task_ids": ["parent"],
                    "idempotency_key": "retry-parent",
                },
            )
            assert result.json()["results"][0]["ok"], result.text
        else:
            result = await client.post(f"/api/v1/tasks/parent/{route}")
        assert result.status_code == 200, result.text
    assert store.get_task("parent").state == TaskState.QUEUED
    assert store.get_task("parent").attempt == 1
    assert store.list_batch_runs()[0].state == BatchState.RUNNING


async def test_batch_start_requeues_failure_and_does_not_finish_immediately(tmp_path):
    store, client = setup(tmp_path, batch_state=BatchState.DRAFT)
    async with client:
        result = await client.post("/api/v1/batches/batch/start")
        assert result.status_code == 200, result.text
    reconcile_batch_runs(store)
    assert store.get_task("parent").state == TaskState.QUEUED
    assert store.list_batch_runs()[0].state == BatchState.RUNNING


async def test_exhausted_retry_does_not_reactivate_batch_and_remains_cancellable(tmp_path):
    store, client = setup(tmp_path, attempts=3)
    assert "retry" not in store.get_task("parent").available_actions
    assert "cancel" in store.get_task("parent").available_actions
    async with client:
        rejected = await client.post("/api/v1/batches/batch/start")
        assert rejected.status_code == 409
        cancelled = await client.post(
            "/api/v1/task-commands",
            json={
                "scope": "project",
                "scope_id": "p",
                "action": "cancel",
                "task_ids": ["parent"],
                "idempotency_key": "cancel-parent",
            },
        )
        assert cancelled.json()["results"][0]["state"] == "cancelled"
    assert store.list_batch_runs()[0].state == BatchState.COMPLETED


def test_generation_boundary_keeps_failed_generation():
    task = TaskSpec(
        task_id="image",
        project_id="p",
        kind=TaskKind.IMAGE_GENERATION,
        state=TaskState.FAILED,
        idempotency_key="a" * 64,
        input_fingerprint="b" * 64,
    )
    assert resolve_batch_task_ids((task,), "generation") == ("image",)
    assert (
        resolve_batch_task_ids(
            (task.model_copy(update={"state": TaskState.CANCELLED}),), "generation"
        )
        == ()
    )


def test_rejected_operation_can_retry_but_unanswered_intent_cannot(tmp_path):
    store, _ = setup(tmp_path)
    task = store.get_task("parent")
    progress = _Progress(store, task)
    first = progress.start("storyboard", {"idea": "scene"})
    with pytest.raises(HTTPException), progress.rejection_boundary("storyboard"):
        raise HTTPException(502, "provider rejected request")
    second = _Progress(store, task.model_copy(update={"attempt": 2})).start(
        "storyboard",
        {"idea": "scene"},
    )
    assert second != first
    with pytest.raises(OrchestrationNeedsAttention):
        _Progress(store, task.model_copy(update={"attempt": 3})).start(
            "storyboard", {"idea": "scene"}
        )


async def test_actual_orchestrator_continues_after_explicit_provider_rejection(tmp_path):
    scenario = Scenario(tmp_path / "flow.db", _workspace())

    async def rejected(project_id, request):
        raise HTTPException(502, "provider rejected request")

    with pytest.raises(HTTPException):
        await scenario.advance(operation=rejected)
    scenario.store.transition_task("parent", TaskState.FAILED)
    scenario.store.prepare_task_retry("parent")
    assert await scenario.advance() is False
    assert scenario.operations == ["initialize_storyboard"]
    assert scenario.payload()["shots"]
    assert scenario.store.get_task("parent").attempt == 2
