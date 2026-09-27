"""Run the production lifespan/dispatcher against isolated, CPU-only failures."""

import asyncio
from datetime import UTC, datetime
from hashlib import sha256

import httpx
import pytest

from ai_video_generator import api as api_module
from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ComfyUIOutput,
    ProjectRunState,
    ProjectSpec,
    TaskKind,
    TaskSpec,
    TaskState,
    TaskWorkloadManifest,
)
from ai_video_generator.persistence import SQLiteTaskStore


def task(task_id, *, project_id="p", state=TaskState.QUEUED, **kwargs):
    digest = sha256(task_id.encode()).hexdigest()
    return TaskSpec(
        task_id=task_id,
        project_id=project_id,
        state=state,
        kind=TaskKind.H3_GENERATION,
        idempotency_key=digest,
        input_fingerprint=digest,
        **kwargs,
    )


def setup(tmp_path):
    def unexpected(request):
        raise AssertionError(f"Unexpected external operation: {request.method} {request.url.path}")

    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path,
            comfyui_root=None,
            comfyui_base_url="http://mock-worker.invalid",
        ),
        comfyui_transport=httpx.MockTransport(unexpected),
    )
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    now = datetime.now(UTC)
    for project_id in ("p", "q"):
        store.put_project_revision(
            ProjectSpec(project_id=project_id, name=project_id, target_duration_seconds=5)
        )
        store.put_project_run_state(ProjectRunState(project_id=project_id, updated_at=now))
    return app, store, now


async def wait_for_terminal(store, task_id, timeout=5):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        current = store.get_task(task_id)
        if current.state in {TaskState.FAILED, TaskState.SUCCEEDED, TaskState.NEEDS_ATTENTION}:
            return current
        await asyncio.sleep(0.05)
    return store.get_task(task_id)


@pytest.mark.parametrize("unclaimable", ["paused_member", "failed_dependency"])
async def test_unclaimable_gpu_candidate_cannot_starve_independent_project(tmp_path, unclaimable):
    app, store, now = setup(tmp_path)
    if unclaimable == "failed_dependency":
        store.add_task(task("failed-parent", state=TaskState.FAILED))
        store.add_task(task("first", priority=100, depends_on=("failed-parent",)))
    else:
        store.add_task(task("first", priority=100))
        store.put_batch_run(
            BatchRun(
                batch_id="b",
                name="Paused member",
                state=BatchState.RUNNING,
                items=(BatchRunItem(project_id="p", task_ids=("first",), paused=True),),
                created_at=now,
                updated_at=now,
            )
        )
    # No manifest: the real executor must reach deterministic validation and
    # fail before touching ComfyUI. Remaining queued means it was never claimed.
    store.add_task(task("independent", project_id="q", priority=0))
    async with app.router.lifespan_context(app):
        result = await wait_for_terminal(store, "independent")
    assert result.state == TaskState.FAILED, (
        f"{unclaimable} blocked independent execution: state={result.state}; "
        f"events={store.list_execution_events('independent')}"
    )
    assert result.attempt == 1
    assert result.error_code == "local_comfy_execution_failed"
    assert "manifest" in result.error_message
    assert store.get_task("first").attempt == 0


async def test_actual_executor_failure_releases_slot_for_next_project(tmp_path):
    app, store, _ = setup(tmp_path)
    store.add_task(task("first", priority=100))
    store.add_task(task("next", project_id="q"))
    async with app.router.lifespan_context(app):
        result = await wait_for_terminal(store, "next")
    assert store.get_task("first").state == TaskState.FAILED
    assert result.state == TaskState.FAILED
    assert store.get_task("first").attempt == result.attempt == 1
    assert all(store.list_execution_events(task_id) for task_id in ("first", "next"))


async def test_restart_after_output_checkpoint_before_success_reuses_checkpoint(
    tmp_path, monkeypatch
):
    _, store, _ = setup(tmp_path)
    manifest = TaskWorkloadManifest(
        task_kind=TaskKind.H3_GENERATION,
        workflow_sha256="a" * 64,
        node_schema_sha256="b" * 64,
        prompt={"1": {"class_type": "TestVideoOutput", "inputs": {}}},
        outputs=(ComfyUIOutput(node_id="1", media_type="video/mp4"),),
    )
    record = store.put_workload_manifest(manifest)
    store.add_task(task("video", workload_manifest_sha256=record.sha256))
    calls = []
    checkpoint_written = asyncio.Event()
    interrupt_publication = True
    transition = SQLiteTaskStore.transition_task

    def crash_between_checkpoint_and_success(self, task_id, new_state, **kwargs):
        nonlocal interrupt_publication
        if task_id == "video" and new_state == TaskState.SUCCEEDED and interrupt_publication:
            interrupt_publication = False
            checkpoint_written.set()
            raise asyncio.CancelledError("injected process interruption after output checkpoint")
        return transition(self, task_id, new_state, **kwargs)

    monkeypatch.setattr(SQLiteTaskStore, "transition_task", crash_between_checkpoint_and_success)

    async def valid_dimensions(*args, **kwargs):
        return (640, 360)

    monkeypatch.setattr(api_module, "probe_video_dimensions", valid_dimensions)

    def worker(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "external", "node_errors": {}})
        if request.url.path == "/history/external":
            return httpx.Response(
                200,
                json={
                    "external": {
                        "status": {"completed": True},
                        "outputs": {"1": {"videos": [{"filename": "fake.mp4"}]}},
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=b"mock-video-bytes")
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    def application():
        return create_app(
            Settings(
                _env_file=None,
                data_root=tmp_path,
                comfyui_root=None,
                comfyui_base_url="http://mock-worker.invalid",
            ),
            comfyui_transport=httpx.MockTransport(worker),
        )

    first = application()
    async with first.router.lifespan_context(first):
        await asyncio.wait_for(checkpoint_written.wait(), timeout=5)
    previous = store.list_task_checkpoints("video")
    assert len(previous) == 1 and previous[0].phase == "video_saved"
    assert store.get_task("video").state == TaskState.RUNNING
    second = application()
    async with second.router.lifespan_context(second):
        result = await wait_for_terminal(store, "video")
    assert result.state == TaskState.SUCCEEDED, result.error_message
    assert store.list_task_checkpoints("video") == previous
    assert calls.count(("POST", "/prompt")) == 1
