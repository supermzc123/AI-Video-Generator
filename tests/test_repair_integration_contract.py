"""Independent public control-plane contracts using isolated SQLite and mock I/O."""

import socket
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import httpx
import pytest

from ai_video_generator.api import create_app
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


def task(task_id, *, state=TaskState.QUEUED, project_id="p", **kwargs):
    digest = sha256(task_id.encode()).hexdigest()
    return TaskSpec(
        task_id=task_id,
        project_id=project_id,
        kind=TaskKind.H3_GENERATION,
        state=state,
        idempotency_key=digest,
        input_fingerprint=digest,
        **kwargs,
    )


@pytest.fixture
async def harness(tmp_path, monkeypatch):
    clock = {"now": NOW}

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"] if tz is not None else clock["now"].replace(tzinfo=None)

    monkeypatch.setattr(execution_runtime, "datetime", FrozenDateTime)
    monkeypatch.setattr(task_store, "datetime", FrozenDateTime)

    def reject_network(*args, **kwargs):
        raise AssertionError("Independent tests must never open a network socket")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    for project_id in ("p", "q"):
        store.put_project_revision(
            ProjectSpec(project_id=project_id, name=project_id, target_duration_seconds=5)
        )
        store.put_project_run_state(ProjectRunState(project_id=project_id, updated_at=NOW))
    assert store.acquire_dispatcher("test-owner", now=NOW)

    def unexpected(request):
        raise AssertionError(f"Unexpected external request: {request.method} {request.url.path}")

    @asynccontextmanager
    async def open_client(handler=unexpected):
        app = create_app(
            Settings(
                _env_file=None,
                data_root=tmp_path,
                comfyui_root=None,
                comfyui_base_url="http://mock-worker.invalid",
            ),
            comfyui_transport=httpx.MockTransport(handler),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client

    return store, open_client, clock


def put_batch(store, task_ids):
    return store.put_batch_run(
        BatchRun(
            batch_id="b",
            name="Batch",
            state=BatchState.RUNNING,
            items=(BatchRunItem(project_id="p", task_ids=tuple(task_ids)),),
            created_at=NOW,
            updated_at=NOW,
        )
    )


async def test_cancel_lost_submission_response_reconciles_token_without_resubmission(harness):
    store, open_client, _ = harness
    store.add_task(task("work"))
    assert store.claim_local_task("work", "test-owner", now=NOW) is not None
    submission, _ = store.ensure_submission_intent("work")
    store.transition_task("work", TaskState.NEEDS_ATTENTION, now=NOW)
    running = True
    calls = []

    def worker(request):
        nonlocal running
        calls.append((request.method, request.url.path))
        if request.url.path == "/queue":
            row = [1, "accepted", {}, {"avg_submission_token": submission}]
            return httpx.Response(
                200, json={"queue_running": [row] if running else [], "queue_pending": []}
            )
        if request.url.path.startswith("/history"):
            return httpx.Response(200, json={})
        if request.method == "POST" and request.url.path == "/interrupt":
            running = False
            return httpx.Response(200, json={})
        raise AssertionError(f"Unexpected external operation: {request.method} {request.url.path}")

    async with open_client(worker) as client:
        cancelled = await client.post("/api/v1/tasks/work/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["state"] in {"cancelling", "cancelled"}
        reconciled = await client.post("/api/v1/tasks/work/reconcile")
        assert reconciled.status_code == 200, reconciled.text
        assert reconciled.json()["state"] == "cancelled"
        assert store.inspect_submission("work")["submission_token"] == submission
        assert not running
        assert not any(method == "POST" and path == "/prompt" for method, path in calls)


async def test_late_reconciliation_cannot_replace_a_new_attempt(harness):
    store, open_client, _ = harness
    store.add_task(task("work"))
    old = store.claim_local_task("work", "test-owner", now=NOW)
    assert old is not None
    old_token, _ = store.ensure_submission_intent("work")
    store.record_comfyui_prompt("work", "old-prompt")
    store.transition_task("work", TaskState.NEEDS_ATTENTION, now=NOW)
    replacement = None

    def worker(request):
        nonlocal replacement
        if request.url.path == "/history/old-prompt":
            store.confirm_task_stopped(
                "work",
                expected_attempt_id=old.attempt_id,
                expected_submission_token=old_token,
                evidence="mock worker confirms old generation stopped",
                now=NOW,
            )
            store.safe_explicit_retry("work", now=NOW)
            store.transition_task("work", TaskState.QUEUED, now=NOW)
            replacement = store.claim_local_task("work", "test-owner", now=NOW)
            store.ensure_submission_intent("work")
            store.record_comfyui_prompt("work", "new-prompt")
            return httpx.Response(
                200, json={"old-prompt": {"status": {"completed": True}, "outputs": {}}}
            )
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        raise AssertionError(f"Unexpected external operation: {request.method} {request.url.path}")

    async with open_client(worker) as client:
        response = await client.post("/api/v1/tasks/work/reconcile")
        assert response.status_code in {200, 409}, response.text
    after = store.get_task("work")
    assert replacement is not None
    assert after.state == TaskState.RUNNING
    assert after.attempt_id == replacement.attempt_id
    assert after.comfyui_prompt_id == "new-prompt"


async def test_member_pause_does_not_pause_independent_work_in_same_project(harness):
    store, open_client, _ = harness
    store.add_task(task("member"))
    store.add_task(task("outside"))
    put_batch(store, ("member",))
    async with open_client() as client:
        paused = await client.post("/api/v1/batches/b/projects/p/pause")
        assert paused.status_code == 200, paused.text
        assert paused.json()["items"][0]["paused"]
        assert store.claim_local_task("member", "test-owner", now=NOW) is None
        outside = store.claim_local_task("outside", "test-owner", now=NOW)
        assert outside is not None, "member scope leaked into a project-wide pause"
        store.transition_task("outside", TaskState.SUCCEEDED, now=NOW)
        resumed = await client.post("/api/v1/batches/b/projects/p/resume")
        assert resumed.status_code == 200, resumed.text
        assert not resumed.json()["items"][0]["paused"]
        assert store.claim_local_task("member", "test-owner", now=NOW) is not None


async def test_batch_command_validates_all_members_before_mutating_any(harness):
    store, open_client, _ = harness
    for task_id in ("member", "outside"):
        store.add_task(task(task_id))
    put_batch(store, ("member",))
    async with open_client() as client:
        response = await client.post(
            "/api/v1/task-commands",
            json={
                "scope": "batch",
                "scope_id": "b",
                "action": "cancel",
                "task_ids": ["member", "outside"],
                "idempotency_key": "invalid-batch-command",
            },
        )
        assert response.status_code == 422
    assert {t.state for t in store.list_tasks()} == {TaskState.QUEUED}


async def test_partial_command_receipt_is_durable_and_never_replays(harness):
    store, open_client, _ = harness
    store.add_task(task("queued"))
    store.add_task(task("done", state=TaskState.SUCCEEDED))
    command = {
        "scope": "project",
        "scope_id": "p",
        "action": "pause",
        "task_ids": ["queued", "done"],
        "idempotency_key": "mixed-outcome",
    }
    async with open_client() as client:
        first = await client.post("/api/v1/task-commands", json=command)
        assert first.status_code == 200, first.text
        receipt = first.json()
        assert receipt["status"] == "completed"
        assert [item["ok"] for item in receipt["results"]] == [True, False]
        assert store.get_task("queued").state == TaskState.PAUSED
        assert store.get_task("done").state == TaskState.SUCCEEDED
        assert (await client.post("/api/v1/tasks/queued/resume")).status_code == 200
    async with open_client() as client:
        duplicate = await client.post("/api/v1/task-commands", json=command)
        assert duplicate.json() == receipt
        assert store.get_task("queued").state != TaskState.PAUSED
        conflict = await client.post(
            "/api/v1/task-commands", json={**command, "task_ids": ["queued"]}
        )
        assert conflict.status_code == 409


async def test_confirm_retry_cannot_reset_exhausted_budget(harness):
    store, open_client, _ = harness
    store.add_task(task("work", max_attempts=1))
    claimed = store.claim_local_task("work", "test-owner", now=NOW)
    assert claimed is not None
    token, _ = store.ensure_submission_intent("work")
    store.transition_task("work", TaskState.NEEDS_ATTENTION, now=NOW)
    async with open_client() as client:
        response = await client.post(
            "/api/v1/tasks/work/confirm-retry",
            json={"idempotency_key": "budget", "confirm_duplicate_execution": True},
        )
        assert response.status_code == 200, response.text
        assert not response.json()["results"][0]["ok"]
    after = store.get_task("work")
    assert after.state == TaskState.NEEDS_ATTENTION
    assert after.attempt == claimed.attempt
    assert after.deadline_at == claimed.deadline_at
    assert store.inspect_submission("work")["submission_token"] == token


async def test_public_retry_preserves_absolute_deadline_and_attempt_count(harness):
    store, open_client, clock = harness
    store.add_task(task("work"))
    claimed = store.claim_local_task("work", "test-owner", now=NOW)
    assert claimed is not None
    store.transition_task("work", TaskState.FAILED, now=NOW)
    clock["now"] += timedelta(seconds=10)
    async with open_client() as client:
        response = await client.post(
            "/api/v1/task-commands",
            json={
                "scope": "project",
                "scope_id": "p",
                "action": "retry",
                "task_ids": ["work"],
                "idempotency_key": "retry",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["ok"]
    retried = store.claim_local_task("work", "test-owner", now=clock["now"])
    assert retried is not None
    assert retried.attempt == claimed.attempt + 1
    assert retried.deadline_at == claimed.deadline_at


async def test_mixed_batch_outcomes_are_not_reported_as_success(harness):
    store, open_client, _ = harness
    for state in (TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED, TaskState.STALE):
        store.add_task(task(state.value, state=state))
    put_batch(store, ("succeeded", "failed", "cancelled", "stale"))
    async with open_client() as client:
        response = await client.get("/api/v1/batches")
        assert response.status_code == 200
        batch = response.json()[0]
    assert batch["all_tasks_ended"]
    assert not batch["all_tasks_succeeded"]
    assert batch["task_counts"]["pending"] == 0
    assert batch["task_counts"]["total"] == 4
    for state in ("succeeded", "failed", "cancelled", "stale"):
        assert batch["task_counts"][state] == 1


async def test_health_and_events_are_read_only_ordered_and_bounded(harness):
    store, open_client, _ = harness
    store.add_task(task("work"))
    store.claim_local_task("work", "test-owner", now=NOW)
    store.transition_task("work", TaskState.CANCELLING, now=NOW)
    before = store.get_task("work")
    async with open_client() as client:
        health = await client.get("/api/v1/scheduler/health")
        assert health.status_code == 200
        assert health.json()["task_counts"]["cancelling"] == 1
        full = await client.get("/api/v1/tasks/work/events")
        limited = await client.get("/api/v1/tasks/work/events", params={"limit": 1})
        assert limited.json() == full.json()[:1]
        assert full.json()[0]["event_id"] > full.json()[-1]["event_id"]
        assert (await client.get("/api/v1/tasks/work/events?limit=1001")).status_code == 422
        assert (await client.get("/api/v1/tasks/missing/events")).status_code == 404
    assert store.get_task("work") == before
