"""Existing HTTP routes only; future repair routes are documented, not invented."""

import socket
from datetime import UTC, datetime
from hashlib import sha256
from importlib import import_module

import httpx
import pytest

from ai_video_generator import config
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ProjectRunState,
    ProjectSpec,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore, execution_runtime, task_store

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def make_task(task_id, project_id="project", state=TaskState.READY):
    return TaskSpec(
        task_id=task_id,
        project_id=project_id,
        state=state,
        kind=TaskKind.H3_GENERATION,
        idempotency_key=sha256(task_id.encode()).hexdigest(),
        input_fingerprint=sha256(f"input:{task_id}".encode()).hexdigest(),
    )


@pytest.fixture
async def api(tmp_path, monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(execution_runtime, "datetime", FrozenDateTime)
    monkeypatch.setattr(task_store, "datetime", FrozenDateTime)

    def reject_network(*args, **kwargs):
        raise AssertionError("Independent verification must not open a network socket")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    def disconnected(request):
        raise httpx.ConnectError("injected offline worker", request=request)

    settings = Settings(
        _env_file=None,
        data_root=tmp_path,
        comfyui_root=None,
        comfyui_base_url="http://mock-worker.invalid",
    )
    original_get_settings = config.get_settings
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(config, "load_secret", lambda: None)
    api_module = import_module("ai_video_generator.api")
    api_module.get_settings = original_get_settings

    app = api_module.create_app(
        settings,
        comfyui_transport=httpx.MockTransport(disconnected),
    )
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client, store


def batch(store):
    for project_id in ("project", "foreign"):
        store.put_project_revision(
            ProjectSpec(
                project_id=project_id,
                name=project_id,
                target_duration_seconds=10,
            )
        )
        store.put_project_run_state(ProjectRunState(project_id=project_id, updated_at=NOW))
    for task_id, project_id in (
        ("member", "project"),
        ("outside", "project"),
        ("foreign-task", "foreign"),
    ):
        store.add_task(make_task(task_id, project_id))
    return store.put_batch_run(
        BatchRun(
            batch_id="batch",
            name="batch",
            state=BatchState.RUNNING,
            items=(BatchRunItem(project_id="project", task_ids=("member",)),),
            created_at=NOW,
            updated_at=NOW,
        )
    )


async def test_task_submission_and_cancel_are_idempotent(api):
    client, store = api
    original = make_task("work")
    first = await client.post("/api/v1/tasks", json=original.model_dump(mode="json"))
    duplicate = await client.post(
        "/api/v1/tasks",
        json=original.model_copy(
            update={"task_id": "duplicate"},
        ).model_dump(mode="json"),
    )
    assert first.status_code == duplicate.status_code == 201
    assert first.json()["task_id"] == duplicate.json()["task_id"] == "work"
    first_cancel = await client.post("/api/v1/tasks/work/cancel")
    second_cancel = await client.post("/api/v1/tasks/work/cancel")
    assert first_cancel.status_code == second_cancel.status_code == 200
    assert first_cancel.json() == second_cancel.json()
    assert len(store.list_tasks()) == 1
    for action in ("run", "resume", "redo"):
        response = await client.post(f"/api/v1/tasks/work/{action}")
        assert response.status_code == 409
    assert store.get_task("work").state == TaskState.CANCELLED


@pytest.mark.parametrize("action", ["run", "pause", "resume", "redo", "cancel"])
async def test_missing_task_is_404_without_mutations(api, action):
    client, store = api
    response = await client.post(f"/api/v1/tasks/missing/{action}")
    assert response.status_code == 404
    assert store.list_tasks() == ()


async def test_unknown_submission_cancel_reserves_ownership(api):
    client, store = api
    store.add_task(make_task("work", state=TaskState.QUEUED))
    assert store.acquire_dispatcher("owner", now=NOW)
    assert store.claim_local_task("work", "owner", now=NOW) is not None
    submission, fresh = store.ensure_submission_intent("work")
    assert fresh
    for _ in range(2):
        response = await client.post("/api/v1/tasks/work/cancel")
        assert response.status_code == 200
        assert response.json()["state"] == "cancelling"
    assert store.inspect_submission("work")["submission_token"] == submission
    store.add_task(make_task("next", state=TaskState.QUEUED))
    assert store.claim_local_task("next", "owner", now=NOW) is None


async def test_offline_worker_cancel_remains_cancelling(api):
    client, store = api
    store.add_task(make_task("work", state=TaskState.QUEUED))
    assert store.acquire_dispatcher("owner", now=NOW)
    assert store.claim_local_task("work", "owner", now=NOW) is not None
    store.record_comfyui_prompt("work", "external-id")
    response = await client.post("/api/v1/tasks/work/cancel")
    assert response.status_code == 200
    assert response.json()["state"] == "cancelling"
    store.add_task(make_task("next", state=TaskState.QUEUED))
    assert store.claim_local_task("next", "owner", now=NOW) is None


async def test_batch_cancel_rejects_nonmember_project_without_side_effects(api):
    client, store = api
    batch(store)
    before = store.get_task("foreign-task")
    response = await client.post("/api/v1/batches/batch/projects/foreign/cancel")
    after = store.get_task("foreign-task")
    assert after == before, f"nonmember changed: HTTP {response.status_code}, {after.state}"
    assert response.status_code in {404, 409, 422}


@pytest.mark.parametrize(
    "path", ["/api/v1/batches/batch/projects/project/cancel", "/api/v1/batches/batch/cancel"]
)
async def test_batch_cancel_does_not_cancel_unselected_project_tasks(api, path):
    client, store = api
    batch(store)
    response = await client.post(path)
    assert response.status_code == 200, response.text
    assert store.get_task("member").state == TaskState.CANCELLED
    assert store.get_task("outside").state == TaskState.READY
    assert store.get_task("foreign-task").state == TaskState.READY


async def test_api_exposes_persisted_attempt_metadata(api):
    client, store = api
    store.add_task(make_task("work", state=TaskState.QUEUED))
    assert store.acquire_dispatcher("owner", now=NOW)
    claimed = store.claim_local_task("work", "owner", now=NOW)
    assert claimed is not None
    response = await client.get("/api/v1/tasks/work")
    assert response.status_code == 200
    body = response.json()
    for field in (
        "attempt_id",
        "created_at",
        "updated_at",
        "last_activity_at",
        "next_retry_at",
        "deadline_at",
        "current_phase",
        "blocked_reason",
        "available_actions",
    ):
        assert field in body
    assert body["attempt_id"] == claimed.attempt_id
    assert body["available_actions"] == ["cancel"]
