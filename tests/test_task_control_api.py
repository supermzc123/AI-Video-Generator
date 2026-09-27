import hashlib
from datetime import UTC, datetime

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


def _task(task_id, project_id="p", state=TaskState.QUEUED, **extra):
    fingerprint = hashlib.sha256(task_id.encode()).hexdigest()
    return TaskSpec(
        task_id=task_id,
        project_id=project_id,
        kind=TaskKind.H3_GENERATION,
        state=state,
        idempotency_key=fingerprint,
        input_fingerprint=fingerprint,
        **extra,
    )


def _client(tmp_path, handler=None):
    app = create_app(
        Settings(_env_file=None, data_root=tmp_path, comfyui_base_url="http://mock-comfy"),
        comfyui_transport=httpx.MockTransport(handler) if handler else None,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _put_task(client, task):
    response = await client.post("/api/v1/tasks", json=task.model_dump(mode="json"))
    assert response.status_code == 201, response.text


@pytest.mark.asyncio
async def test_restart_command_replaces_exhausted_planning_and_replays_receipt(tmp_path):
    from ai_video_generator.persistence import SQLiteTaskStore

    async with _client(tmp_path) as client:
        planning = _task("batch-plan:p:old", state=TaskState.FAILED).model_copy(update={
            "kind": TaskKind.LLM_PLANNING, "attempt": 3,
            "error_code": "local_control_task_failed", "error_message": "missing workflow",
        })
        await _put_task(client, planning)
        store = SQLiteTaskStore(tmp_path / "control-plane.db")
        now = datetime.now(UTC)
        store.put_batch_run(BatchRun(
            batch_id="restart-batch", name="Restart", state=BatchState.COMPLETED,
            items=(BatchRunItem(project_id="p", task_ids=(planning.task_id,)),),
            created_at=now, updated_at=now,
        ))
        command = {"scope": "project", "scope_id": "p", "action": "restart",
                   "task_ids": [planning.task_id], "idempotency_key": "restart-command"}
        response = await client.post("/api/v1/task-commands", json=command)
        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["ok"], response.text
        new_id = response.json()["results"][0]["task_id"]
        assert new_id != planning.task_id
        replay = await client.post("/api/v1/task-commands", json=command)
        assert replay.json() == response.json()
        assert len(store.list_tasks()) == 2


@pytest.mark.asyncio
async def test_cancel_offline_then_reconcile_confirms_stop_without_resubmitting(tmp_path):
    online = False
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if not online:
            raise httpx.ConnectError("offline", request=request)
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        return httpx.Response(404)

    async with _client(tmp_path, handler) as client:
        await _put_task(client, _task("cancel", state=TaskState.RUNNING, comfyui_prompt_id="ext"))
        response = await client.post("/api/v1/tasks/cancel/cancel")
        assert response.json()["state"] == "cancelling"
        online = True
        response = await client.post("/api/v1/tasks/cancel/reconcile")
        assert response.json()["state"] == "cancelled"
        repeated = await client.post("/api/v1/tasks/cancel/reconcile")
        assert repeated.json()["state"] == "cancelled"
        assert not any(method == "POST" and path == "/prompt" for method, path in requests)


@pytest.mark.asyncio
async def test_remote_cancel_never_interrupts_the_local_gpu(tmp_path):
    def handler(request):
        raise AssertionError("remote cancellation touched local ComfyUI")

    async with _client(tmp_path, handler) as client:
        await _put_task(
            client,
            _task(
                "remote",
                state=TaskState.RUNNING,
                execution_target=ExecutionTarget.REMOTE,
                worker_id="worker",
                comfyui_prompt_id="external",
            ),
        )
        response = await client.post("/api/v1/tasks/remote/cancel")
        assert response.status_code == 200
        assert response.json()["state"] == "cancelling"


@pytest.mark.asyncio
async def test_bulk_commands_reject_cross_project_scope_before_any_side_effect(tmp_path):
    async with _client(tmp_path) as client:
        await _put_task(client, _task("one"))
        await _put_task(client, _task("two", project_id="other"))
        response = await client.post(
            "/api/v1/task-commands",
            json={
                "scope": "project",
                "scope_id": "p",
                "action": "cancel",
                "task_ids": ["one", "two"],
                "idempotency_key": "scope-check",
            },
        )
        assert response.status_code == 422
        assert (await client.get("/api/v1/tasks/one")).json()["state"] == "queued"
        assert (await client.get("/api/v1/tasks/two")).json()["state"] == "queued"


@pytest.mark.asyncio
async def test_command_receipt_survives_app_restart_and_rejects_key_reuse(tmp_path):
    command = {
        "scope": "project",
        "scope_id": "p",
        "action": "pause",
        "task_ids": ["one"],
        "idempotency_key": "persistent-command",
    }
    async with _client(tmp_path) as client:
        await _put_task(client, _task("one"))
        first = await client.post("/api/v1/task-commands", json=command)
        assert first.json()["results"] == [{"task_id": "one", "ok": True, "state": "paused"}]
    async with _client(tmp_path) as client:
        second = await client.post("/api/v1/task-commands", json=command)
        assert second.json() == first.json()
        conflict = await client.post("/api/v1/task-commands", json={**command, "action": "cancel"})
        assert conflict.status_code == 409
        assert (await client.get("/api/v1/tasks/one")).json()["state"] == "paused"


@pytest.mark.asyncio
async def test_ambiguous_execution_requires_separate_explicit_confirmation(tmp_path):
    async with _client(tmp_path) as client:
        await _put_task(client, _task("unknown", state=TaskState.NEEDS_ATTENTION))
        assert (await client.post("/api/v1/tasks/unknown/run")).status_code == 409
        no = await client.post(
            "/api/v1/tasks/unknown/confirm-retry", json={"idempotency_key": "yes"}
        )
        assert no.status_code == 422
        yes = await client.post(
            "/api/v1/tasks/unknown/confirm-retry",
            json={
                "idempotency_key": "yes",
                "confirm_duplicate_execution": True,
            },
        )
        assert yes.status_code == 200
        assert yes.json()["results"][0]["ok"]
        assert (await client.get("/api/v1/tasks/unknown")).json()["state"] == "queued"


@pytest.mark.asyncio
async def test_reconcile_completed_external_prompt_only_reattaches(tmp_path):
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/history/ext":
            return httpx.Response(200, json={"ext": {"status": {"completed": True}, "outputs": {}}})
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        raise AssertionError("unexpected external operation")

    async with _client(tmp_path, handler) as client:
        await _put_task(
            client, _task("finished", state=TaskState.NEEDS_ATTENTION, comfyui_prompt_id="ext")
        )
        response = await client.post("/api/v1/tasks/finished/reconcile")
        assert response.json()["state"] == "recovering"
        assert response.json()["comfyui_prompt_id"] == "ext"
        assert all(method == "GET" for method, _ in requests)


@pytest.mark.asyncio
async def test_batch_member_cancel_rejects_nonmember_and_reports_mixed_outcomes(tmp_path):
    now = datetime.now(UTC)
    async with _client(tmp_path) as client:
        for project_id in ("p", "other"):
            await client.post(
                "/api/v1/projects",
                json=ProjectSpec(
                    project_id=project_id,
                    name=project_id,
                    target_duration_seconds=4,
                ).model_dump(mode="json"),
            )
        await _put_task(client, _task("own", state=TaskState.FAILED))
        await _put_task(client, _task("elsewhere", project_id="other"))
        batch = BatchRun(
            batch_id="batch",
            name="Batch",
            state=BatchState.RUNNING,
            items=(BatchRunItem(project_id="p", task_ids=("own",)),),
            created_at=now,
            updated_at=now,
        )
        assert (
            await client.post("/api/v1/batches", json=batch.model_dump(mode="json"))
        ).status_code == 201
        denied = await client.post("/api/v1/batches/batch/projects/other/cancel")
        assert denied.status_code == 422
        assert (await client.get("/api/v1/tasks/elsewhere")).json()["state"] == "queued"
        listed = (await client.get("/api/v1/batches")).json()[0]
        assert listed["all_tasks_ended"]
        assert not listed["all_tasks_succeeded"]
        assert listed["task_counts"]["failed"] == 1


@pytest.mark.asyncio
async def test_health_and_events_expose_backend_state(tmp_path):
    async with _client(tmp_path) as client:
        await _put_task(client, _task("visible"))
        await client.post("/api/v1/tasks/visible/pause")
        health = await client.get("/api/v1/scheduler/health")
        assert health.json()["task_counts"]["paused"] == 1
        events = await client.get("/api/v1/tasks/visible/events")
        assert any(event["kind"] == "state_changed" for event in events.json())
        assert (await client.get("/api/v1/tasks/missing/events")).status_code == 404


@pytest.mark.asyncio
async def test_compiled_execution_does_not_require_an_llm(tmp_path):
    async with _client(tmp_path) as client:
        await client.post(
            "/api/v1/projects",
            json=ProjectSpec(
                project_id="p",
                name="Prepared",
                target_duration_seconds=4,
            ).model_dump(mode="json"),
        )
        await client.post(
            "/api/v1/projects/p/workspace",
            json={
                "revision": 1,
                "payload": {"stageApprovals": {"outline": datetime.now(UTC).isoformat()}},
            },
        )
        await _put_task(client, _task("compiled", state=TaskState.READY))
        result = await client.get("/api/v1/projects/p/preflight")
        assert result.status_code == 200
        assert not result.json()["requires_llm"]
        assert result.json()["blockers"] == []
