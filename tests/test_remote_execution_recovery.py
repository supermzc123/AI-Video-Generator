import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from ai_video_generator.domain import (
    ComfyUIOutput,
    ExecutionTarget,
    TaskKind,
    TaskSpec,
    TaskState,
    TaskWorkloadManifest,
    WorkerCapabilities,
)
from ai_video_generator.services.remote import TaskResultReport, WorkerResultStatus
from ai_video_generator.services.remote_execution import RemoteExecutionJournal
from ai_video_generator.worker_cli import _workload_executor
from ai_video_generator.workers.remote_worker import (
    RemoteLeaseLostError,
    RemoteWorkerRuntime,
    WorkerExecutionOutcome,
)


def workload():
    manifest = TaskWorkloadManifest(
        task_kind=TaskKind.H3_GENERATION,
        workflow_sha256="a" * 64,
        node_schema_sha256="b" * 64,
        prompt={"1": {"class_type": "SaveVideo", "inputs": {}}},
        outputs=(ComfyUIOutput(node_id="1", media_type="video/mp4"),),
    )
    task = TaskSpec(
        task_id="t",
        project_id="p",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.RUNNING,
        idempotency_key="c" * 64,
        input_fingerprint="d" * 64,
        execution_target=ExecutionTarget.REMOTE,
        worker_id="w",
        attempt=1,
        attempt_id="attempt",
        deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        workload_manifest_sha256=manifest.sha256,
    )
    return task, manifest


class Adapter:
    def __init__(self):
        self.submissions = 0
        self.history = {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {"1": {"videos": [{"filename": "result.mp4"}]}},
        }
        self.found = "prompt"
        self.active = set()
        self.metadata = None
        self.lose_response = False
        self.cancel_stops = True
        self.queue_malformed = False
        self.started = asyncio.Event()

    async def submit_prompt(self, prompt, *, client_id, extra_data):
        self.submissions += 1
        self.metadata = extra_data
        self.active.add("prompt")
        self.started.set()
        if self.lose_response:
            raise httpx.ReadTimeout("response lost")
        return SimpleNamespace(prompt_id="prompt", node_errors={})

    async def find_submission(self, token):
        return self.found

    async def get_history(self, prompt_id):
        return self.history

    async def get_output_image(self, filename, **kwargs):
        return b"generated-video"

    async def cancel_prompt(self, prompt_id):
        if self.cancel_stops:
            self.active.discard(prompt_id)

    def _client(self):
        def handler(request):
            return httpx.Response(
                200,
                json={}
                if self.queue_malformed
                else {
                    "queue_running": [[0, item, {}, {}] for item in self.active],
                    "queue_pending": [],
                },
            )

        return httpx.AsyncClient(base_url="http://mock", transport=httpx.MockTransport(handler))


def executor(tmp_path, adapter):
    return _workload_executor(
        client=SimpleNamespace(),
        adapter=adapter,
        comfyui_root=tmp_path / "comfy",
        work_root=tmp_path / "work",
        poll_seconds=0.001,
    )


@pytest.mark.asyncio
async def test_lost_submission_response_reattaches_without_repeat(tmp_path):
    task, manifest = workload()
    adapter = Adapter()
    adapter.lose_response = True
    execute = executor(tmp_path, adapter)
    result = await execute(task, manifest)
    assert result.status == WorkerResultStatus.SUCCEEDED
    assert adapter.submissions == 1
    assert adapter.metadata["avg_attempt_id"] == task.attempt_id
    assert adapter.metadata["avg_submission_token"] == result.submission_token
    assert result.artifacts[0].path.read_bytes() == b"generated-video"
    assert hashlib.sha256(b"generated-video").hexdigest() in result.artifacts[0].path.name
    restarted = executor(tmp_path, adapter)
    assert (await restarted(task, manifest)).status == WorkerResultStatus.SUCCEEDED
    assert adapter.submissions == 1


@pytest.mark.asyncio
async def test_unknown_intent_survives_restart_and_never_resubmits(tmp_path):
    task, manifest = workload()
    adapter = Adapter()
    adapter.lose_response, adapter.found = True, None
    first = await executor(tmp_path, adapter)(task, manifest)
    second = await executor(tmp_path, adapter)(task, manifest)
    assert first.status == second.status == WorkerResultStatus.NEEDS_ATTENTION
    assert first.submission_token == second.submission_token
    assert adapter.submissions == 1


@pytest.mark.asyncio
async def test_expired_original_deadline_does_not_get_four_more_hours(tmp_path):
    task, manifest = workload()
    task = task.model_copy(update={"deadline_at": datetime.now(UTC) - timedelta(seconds=1)})
    adapter = Adapter()
    result = await executor(tmp_path, adapter)(task, manifest)
    assert result.status == WorkerResultStatus.FAILED
    assert result.error_code == "deadline_exceeded" and adapter.submissions == 0


@pytest.mark.asyncio
async def test_completed_history_collects_after_original_deadline(tmp_path):
    task, manifest = workload()
    task = task.model_copy(update={"deadline_at": datetime.now(UTC) - timedelta(seconds=1)})
    adapter = Adapter()
    execute = executor(tmp_path, adapter)
    execute.journal.prepare(task, manifest)
    execute.journal.claim_submission(task)
    execute.journal.update(task, phase="submitted", prompt_id="prompt")
    assert (await execute(task, manifest)).status == WorkerResultStatus.SUCCEEDED
    assert adapter.submissions == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed,stops,expected",
    [
        (False, True, WorkerResultStatus.CANCELLED),
        (False, False, WorkerResultStatus.NEEDS_ATTENTION),
        (True, True, WorkerResultStatus.NEEDS_ATTENTION),
    ],
)
async def test_cancellation_requires_validated_absent_queue(tmp_path, malformed, stops, expected):
    task, manifest = workload()
    adapter = Adapter()
    adapter.history = None
    adapter.cancel_stops, adapter.queue_malformed = stops, malformed
    execute = executor(tmp_path, adapter)
    running = asyncio.create_task(execute(task, manifest))
    await adapter.started.wait()
    running.cancel()
    result = await running
    assert result.status == expected
    assert bool(result.stop_evidence) == (expected == WorkerResultStatus.CANCELLED)
    assert execute.journal.get(task)["phase"] == (
        "stopped" if expected == WorkerResultStatus.CANCELLED else "unknown"
    )


@pytest.mark.asyncio
async def test_corrupt_completed_output_is_not_reused_or_regenerated(tmp_path):
    task, manifest = workload()
    adapter = Adapter()
    execute = executor(tmp_path, adapter)
    first = await execute(task, manifest)
    first.artifacts[0].path.write_bytes(b"corrupt")
    second = await execute(task, manifest)
    assert second.status == WorkerResultStatus.NEEDS_ATTENTION
    assert adapter.submissions == 1


def test_journal_attempt_identity_and_deadline_are_immutable(tmp_path):
    task, manifest = workload()
    journal = RemoteExecutionJournal(tmp_path)
    first = journal.prepare(task, manifest)
    assert journal.prepare(task, manifest)["submission_token"] == first["submission_token"]
    with pytest.raises(ValueError, match="deadline"):
        journal.prepare(
            task.model_copy(update={"deadline_at": task.deadline_at + timedelta(hours=1)}), manifest
        )
    assert journal.claim_submission(task)
    assert not RemoteExecutionJournal(tmp_path).claim_submission(task)


class RuntimeClient:
    def __init__(self, task, manifest, *, cancelled=False, lost=False):
        self.task, self.manifest = task, manifest
        self.cancelled, self.lost = cancelled, lost
        self.reports = []

    async def register(self, capabilities):
        pass

    async def claim(self, heartbeat):
        return self.task

    async def renew(self, worker_id, task_id, attempt_id):
        if self.lost:
            raise RemoteLeaseLostError("lease lost")
        return (
            self.task.model_copy(update={"state": TaskState.CANCELLING})
            if self.cancelled
            else self.task
        )

    async def get_workload_manifest(self, digest):
        return self.manifest

    async def report_result(self, report):
        self.reports.append(report)


def runtime(client, execute):
    return RemoteWorkerRuntime(
        client=client,
        capabilities=WorkerCapabilities(
            worker_id="w",
            platform="test",
            node_schema_sha256="b" * 64,
        ),
        executor=execute,
        workload_executor=execute,
        lease_renew_interval_seconds=0.003,
    )


@pytest.mark.asyncio
async def test_renew_cancelling_cleans_up_executor_then_reports_confirmed_stop(tmp_path):
    task, manifest = workload()
    adapter = Adapter()
    adapter.history = None
    client = RuntimeClient(task, manifest, cancelled=True)
    await runtime(client, executor(tmp_path, adapter)).run_once()
    assert client.reports[0].status == WorkerResultStatus.CANCELLED
    assert client.reports[0].stop_evidence
    assert not adapter.active


@pytest.mark.asyncio
async def test_lease_loss_cleans_up_without_late_result_publication(tmp_path):
    task, manifest = workload()
    adapter = Adapter()
    adapter.history = None
    client = RuntimeClient(task, manifest, lost=True)
    with pytest.raises(RemoteLeaseLostError):
        await runtime(client, executor(tmp_path, adapter)).run_once()
    assert not client.reports and not adapter.active


@pytest.mark.asyncio
async def test_stop_event_does_not_leave_orphan_executor(tmp_path):
    task, manifest = workload()
    adapter = Adapter()
    adapter.history = None
    client = RuntimeClient(task, manifest)
    worker = runtime(client, executor(tmp_path, adapter))
    stop = asyncio.Event()
    running = asyncio.create_task(worker.run(stop_event=stop))
    await adapter.started.wait()
    stop.set()
    await asyncio.wait_for(running, timeout=2)
    assert not adapter.active and not client.reports


def test_cancelled_report_requires_stop_evidence():
    with pytest.raises(ValueError, match="stop evidence"):
        TaskResultReport(
            report_id="a" * 64,
            task_id="t",
            worker_id="w",
            attempt=1,
            status=WorkerResultStatus.CANCELLED,
        )
    with pytest.raises(ValueError, match="stop evidence"):
        WorkerExecutionOutcome(status=WorkerResultStatus.CANCELLED)
