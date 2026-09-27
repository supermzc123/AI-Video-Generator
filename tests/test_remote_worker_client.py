import hashlib
from pathlib import Path

import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    ExecutionTarget,
    TaskKind,
    TaskSpec,
    TaskState,
    WorkerCapabilities,
)
from ai_video_generator.services.remote import WorkerResultStatus
from ai_video_generator.workers.remote_worker import (
    ProducedArtifact,
    RemoteWorkerClient,
    RemoteWorkerRuntime,
    RetryPolicy,
    WorkerExecutionOutcome,
)


def capabilities() -> WorkerCapabilities:
    return WorkerCapabilities(
        worker_id="ubuntu-1",
        platform="linux",
        node_schema_sha256="a" * 64,
    )


def remote_task() -> TaskSpec:
    return TaskSpec(
        task_id="remote-segment-1",
        project_id="project-1",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.READY,
        idempotency_key="b" * 64,
        input_fingerprint="c" * 64,
        execution_target=ExecutionTarget.REMOTE,
        worker_id="ubuntu-1",
    )


@pytest.mark.asyncio
async def test_remote_client_claim_is_retry_safe_and_result_is_idempotent(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path / "data",
            worker_auth_token="secret-token",
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as setup:
        response = await setup.post("/api/v1/tasks", json=remote_task().model_dump(mode="json"))
        assert response.status_code == 201
        queued = await setup.post(f"/api/v1/tasks/{remote_task().task_id}/run")
        assert queued.status_code == 200

    artifact = tmp_path / "result.bin"
    artifact.write_bytes(b"remote-result")
    async with RemoteWorkerClient(
        control_plane_url="http://test",
        token="secret-token",
        allow_insecure_http=True,
        transport=transport,
        retry_policy=RetryPolicy(attempts=1),
    ) as client:
        await client.register(capabilities())

        async def execute_once(leased: TaskSpec) -> WorkerExecutionOutcome:
            assert leased.attempt == 1
            assert leased.attempt_id
            # A second client/poll cannot replay an active execution.
            assert await client.claim(_heartbeat()) is None
            return await execute(leased)

        async def execute(_: TaskSpec) -> WorkerExecutionOutcome:
            return WorkerExecutionOutcome(
                status=WorkerResultStatus.SUCCEEDED,
                artifacts=(ProducedArtifact(artifact),),
            )

        runtime = RemoteWorkerRuntime(
            client=client,
            capabilities=capabilities(),
            executor=execute_once,
            lease_renew_interval_seconds=60,
        )
        completed = await runtime.run_once()
        assert completed is not None
        assert completed.attempt == 1

        destination = tmp_path / "downloaded.bin"
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        await client.download_artifact(digest, destination)
        assert destination.read_bytes() == b"remote-result"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as inspect:
        response = await inspect.get("/api/v1/tasks/remote-segment-1")
        assert response.json()["state"] == "succeeded"


@pytest.mark.asyncio
async def test_remote_client_retries_transient_registration_failure() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "capabilities": capabilities().model_dump(mode="json"),
                "connection_state": "online",
                "registered_at": "2026-08-15T00:00:00Z",
            },
        )

    async with RemoteWorkerClient(
        control_plane_url="https://control.example",
        token="token",
        transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(attempts=2, initial_delay_seconds=0, jitter_ratio=0),
    ) as client:
        registration = await client.register(capabilities())

    assert registration.capabilities.worker_id == "ubuntu-1"
    assert calls == 2


@pytest.mark.asyncio
async def test_remote_worker_does_not_claim_local_tasks(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path / "data",
            worker_auth_token="secret-token",
        )
    )
    transport = httpx.ASGITransport(app=app)
    local_task = remote_task().model_copy(
        update={
            "task_id": "local-segment",
            "execution_target": ExecutionTarget.LOCAL,
            "worker_id": None,
        }
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as setup:
        response = await setup.post("/api/v1/tasks", json=local_task.model_dump(mode="json"))
        assert response.status_code == 201

    async with RemoteWorkerClient(
        control_plane_url="http://test",
        token="secret-token",
        allow_insecure_http=True,
        transport=transport,
        retry_policy=RetryPolicy(attempts=1),
    ) as client:
        await client.register(capabilities())
        assert await client.claim(_heartbeat()) is None


def test_remote_client_rejects_plain_http_by_default() -> None:
    with pytest.raises(ValueError, match="require HTTPS"):
        RemoteWorkerClient(control_plane_url="http://control.example", token="token")


def _heartbeat():
    from datetime import UTC, datetime

    from ai_video_generator.services.remote import WorkerHeartbeat

    return WorkerHeartbeat(
        worker_id="ubuntu-1",
        sent_at=datetime.now(UTC),
        available_gpu_slots=1,
    )
