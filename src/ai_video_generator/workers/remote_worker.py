from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ai_video_generator.domain import (
    TaskSpec,
    TaskWorkloadManifest,
    WorkerCapabilities,
    WorkloadManifestRecord,
    worker_can_execute_manifest,
)
from ai_video_generator.services.remote import (
    ArtifactTransfer,
    TaskResultReceipt,
    TaskResultReport,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerResultStatus,
)


class RemoteWorkerError(RuntimeError):
    pass


class RemoteLeaseLostError(RemoteWorkerError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 5
    initial_delay_seconds: float = 0.5
    maximum_delay_seconds: float = 15.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("retry attempts must be positive")
        if self.initial_delay_seconds < 0 or self.maximum_delay_seconds < 0:
            raise ValueError("retry delays must not be negative")
        if self.multiplier < 1:
            raise ValueError("retry multiplier must be at least one")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("retry jitter ratio must be between zero and one")


@dataclass(frozen=True, slots=True)
class ProducedArtifact:
    path: Path
    media_type: str = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class WorkerExecutionOutcome:
    status: WorkerResultStatus
    artifacts: tuple[ProducedArtifact, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status == WorkerResultStatus.FAILED and not self.error_code:
            raise ValueError("failed execution outcomes require an error code")
        if self.status != WorkerResultStatus.FAILED and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("only failed execution outcomes may include error details")


class RemoteWorkerClient:
    """Outbound-only client for the control plane's remote Worker API."""

    def __init__(
        self,
        *,
        control_plane_url: str,
        token: str,
        timeout_seconds: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        allow_insecure_http: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base_url = control_plane_url.rstrip("/")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("control plane URL must be an absolute HTTPS URL")
        if parsed.scheme != "https" and not allow_insecure_http:
            raise ValueError(
                "remote Workers require HTTPS unless insecure HTTP is explicitly enabled"
            )
        if not token:
            raise ValueError("Worker token must not be empty")
        self.base_url = base_url
        self.token = token
        self.retry_policy = retry_policy or RetryPolicy()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> RemoteWorkerClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def register(self, capabilities: WorkerCapabilities) -> WorkerRegistration:
        response = await self._request(
            "POST",
            "/api/v1/workers/register",
            json={"capabilities": capabilities.model_dump(mode="json")},
        )
        return WorkerRegistration.model_validate(response.json())

    async def claim(self, heartbeat: WorkerHeartbeat) -> TaskSpec | None:
        response = await self._request(
            "POST",
            f"/api/v1/workers/{heartbeat.worker_id}/leases/claim",
            json={"heartbeat": heartbeat.model_dump(mode="json")},
        )
        payload = response.json()
        return None if payload is None else TaskSpec.model_validate(payload)

    async def renew(self, worker_id: str, task_id: str) -> TaskSpec:
        response = await self._request(
            "POST", f"/api/v1/workers/{worker_id}/leases/{task_id}/renew"
        )
        if response.status_code == 409:
            raise RemoteLeaseLostError(response.text)
        return TaskSpec.model_validate(response.json())

    async def report_result(self, report: TaskResultReport) -> TaskResultReceipt:
        response = await self._request(
            "POST",
            f"/api/v1/workers/{report.worker_id}/leases/{report.task_id}/result",
            json={"result": report.model_dump(mode="json")},
        )
        if response.status_code == 409:
            raise RemoteLeaseLostError(response.text)
        return TaskResultReceipt.model_validate(response.json())

    async def get_workload_manifest(self, sha256_value: str) -> TaskWorkloadManifest:
        if len(sha256_value) != 64 or any(
            character not in "0123456789abcdef" for character in sha256_value
        ):
            raise ValueError("invalid workload manifest SHA-256")
        response = await self._request("GET", f"/api/v1/workload-manifests/{sha256_value}")
        record = WorkloadManifestRecord.model_validate(response.json())
        if record.sha256 != sha256_value:
            raise RemoteWorkerError("control plane returned the wrong workload manifest")
        return record.manifest

    async def upload_artifact(self, path: str | Path, media_type: str) -> ArtifactTransfer:
        source = Path(path)
        sha256_value = await asyncio.to_thread(_file_sha256, source)

        async def chunks() -> AsyncIterator[bytes]:
            with source.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    yield chunk

        response = await self._request_with_content_factory(
            "PUT",
            f"/api/v1/artifacts/blobs/{sha256_value}",
            headers={"X-Artifact-Media-Type": media_type},
            content_factory=chunks,
        )
        transfer = ArtifactTransfer.model_validate(response.json())
        if transfer.sha256 != sha256_value or transfer.byte_size != source.stat().st_size:
            raise RemoteWorkerError("control plane returned inconsistent artifact metadata")
        return transfer

    async def download_artifact(self, sha256_value: str, destination: str | Path) -> Path:
        if len(sha256_value) != 64 or any(c not in "0123456789abcdef" for c in sha256_value):
            raise ValueError("invalid artifact SHA-256")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".part", dir=target.parent
            )
            os.close(file_descriptor)
            temporary = Path(temporary_name)
            digest = await self._stream_download(
                f"/api/v1/artifacts/blobs/{sha256_value}", temporary
            )
            if digest != sha256_value:
                raise RemoteWorkerError("downloaded artifact SHA-256 mismatch")
            os.replace(temporary, target)
            temporary = None
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def lease_events(
        self,
        heartbeat_factory: Callable[[], WorkerHeartbeat],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[TaskSpec | None]:
        """Poll lease offers over WSS and reconnect with bounded backoff."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:  # pragma: no cover - depends on Worker extra
            raise RemoteWorkerError("install the 'worker' extra for WSS support") from exc

        parsed = urlsplit(self.base_url)
        websocket_url = urlunsplit(
            (
                "wss" if parsed.scheme == "https" else "ws",
                parsed.netloc,
                f"{parsed.path}/api/v1/workers/{heartbeat_factory().worker_id}/events",
                "",
                "",
            )
        )
        delay = self.retry_policy.initial_delay_seconds
        while stop_event is None or not stop_event.is_set():
            try:
                async with connect(
                    websocket_url,
                    additional_headers={"Authorization": f"Bearer {self.token}"},
                ) as websocket:
                    delay = self.retry_policy.initial_delay_seconds
                    while stop_event is None or not stop_event.is_set():
                        heartbeat = heartbeat_factory()
                        await websocket.send(
                            json.dumps({"heartbeat": heartbeat.model_dump(mode="json")})
                        )
                        payload = json.loads(await websocket.recv())
                        if payload.get("type") == "error":
                            raise RemoteWorkerError(f"Worker event error: {payload.get('code')}")
                        task_payload = payload.get("task")
                        yield (
                            None if task_payload is None else TaskSpec.model_validate(task_payload)
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if stop_event is not None and stop_event.is_set():
                    return
                if delay > self.retry_policy.maximum_delay_seconds:
                    raise RemoteWorkerError("WSS reconnect budget exhausted") from exc
                await asyncio.sleep(_jitter(delay, self.retry_policy.jitter_ratio))
                delay *= self.retry_policy.multiplier

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        policy = self.retry_policy
        delay = policy.initial_delay_seconds
        for attempt in range(policy.attempts):
            try:
                response = await self._client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt + 1 == policy.attempts:
                    raise RemoteWorkerError(
                        f"control plane request failed: {method} {url}"
                    ) from exc
            else:
                if response.status_code not in {408, 429} and response.status_code < 500:
                    if response.status_code >= 400 and response.status_code != 409:
                        raise RemoteWorkerError(
                            f"control plane rejected {method} {url}: "
                            f"HTTP {response.status_code} {response.text}"
                        )
                    return response
                if attempt + 1 == policy.attempts:
                    raise RemoteWorkerError(
                        f"control plane unavailable: {method} {url} returned {response.status_code}"
                    )
            await asyncio.sleep(
                _jitter(min(delay, policy.maximum_delay_seconds), policy.jitter_ratio)
            )
            delay *= policy.multiplier
        raise AssertionError("unreachable")

    async def _request_with_content_factory(
        self,
        method: str,
        url: str,
        *,
        content_factory: Callable[[], AsyncIterator[bytes]],
        **kwargs: Any,
    ) -> httpx.Response:
        policy = self.retry_policy
        delay = policy.initial_delay_seconds
        for attempt in range(policy.attempts):
            try:
                response = await self._client.request(
                    method, url, content=content_factory(), **kwargs
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt + 1 == policy.attempts:
                    raise RemoteWorkerError(f"artifact upload failed: {url}") from exc
            else:
                if response.status_code not in {408, 429} and response.status_code < 500:
                    if response.status_code >= 400:
                        raise RemoteWorkerError(
                            f"control plane rejected artifact upload: HTTP {response.status_code}"
                        )
                    return response
                if attempt + 1 == policy.attempts:
                    raise RemoteWorkerError(
                        f"artifact upload unavailable: HTTP {response.status_code}"
                    )
            await asyncio.sleep(
                _jitter(min(delay, policy.maximum_delay_seconds), policy.jitter_ratio)
            )
            delay *= policy.multiplier
        raise AssertionError("unreachable")

    async def _stream_download(self, url: str, temporary: Path) -> str:
        policy = self.retry_policy
        delay = policy.initial_delay_seconds
        for attempt in range(policy.attempts):
            digest = hashlib.sha256()
            try:
                async with self._client.stream("GET", url) as response:
                    if response.status_code not in {408, 429} and response.status_code < 500:
                        if response.status_code >= 400:
                            raise RemoteWorkerError(
                                "control plane rejected artifact download: "
                                f"HTTP {response.status_code}"
                            )
                        with temporary.open("wb") as handle:
                            async for chunk in response.aiter_bytes(1024 * 1024):
                                digest.update(chunk)
                                handle.write(chunk)
                        return digest.hexdigest()
                    if attempt + 1 == policy.attempts:
                        raise RemoteWorkerError(
                            f"artifact download unavailable: HTTP {response.status_code}"
                        )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt + 1 == policy.attempts:
                    raise RemoteWorkerError(f"artifact download failed: {url}") from exc
            await asyncio.sleep(
                _jitter(min(delay, policy.maximum_delay_seconds), policy.jitter_ratio)
            )
            delay *= policy.multiplier
        raise AssertionError("unreachable")


class RemoteWorkerRuntime:
    def __init__(
        self,
        *,
        client: RemoteWorkerClient,
        capabilities: WorkerCapabilities,
        executor: Callable[[TaskSpec], Awaitable[WorkerExecutionOutcome]],
        workload_executor: Callable[
            [TaskSpec, TaskWorkloadManifest], Awaitable[WorkerExecutionOutcome]
        ]
        | None = None,
        lease_renew_interval_seconds: float = 20.0,
    ) -> None:
        if lease_renew_interval_seconds <= 0:
            raise ValueError("lease renewal interval must be positive")
        self.client = client
        self.capabilities = capabilities
        self.executor = executor
        self.workload_executor = workload_executor
        self.lease_renew_interval_seconds = lease_renew_interval_seconds

    async def run_once(self) -> TaskSpec | None:
        await self.client.register(self.capabilities)
        heartbeat = WorkerHeartbeat(
            worker_id=self.capabilities.worker_id,
            sent_at=datetime.now(UTC),
            available_gpu_slots=1,
        )
        task = await self.client.claim(heartbeat)
        if task is None:
            return None

        stop_renewal = asyncio.Event()
        renewal = asyncio.create_task(self._renew_lease(task, stop_renewal))
        execution = asyncio.create_task(self._execute(task))
        try:
            done, _ = await asyncio.wait({execution, renewal}, return_when=asyncio.FIRST_COMPLETED)
            if renewal in done:
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                error = renewal.exception()
                if error is None:
                    raise RemoteWorkerError("lease renewal stopped before task completion")
                if isinstance(error, RemoteWorkerError):
                    raise error
                raise RemoteWorkerError(
                    f"lease renewal failed: {type(error).__name__}: {error}"
                ) from error
            try:
                outcome = await execution
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                outcome = WorkerExecutionOutcome(
                    status=WorkerResultStatus.FAILED,
                    error_code="worker_executor_error",
                    error_message=str(exc)[:2000],
                )
            artifact_hashes = []
            for artifact in outcome.artifacts:
                if renewal.done():
                    await renewal
                transfer = await self.client.upload_artifact(artifact.path, artifact.media_type)
                artifact_hashes.append(transfer.sha256)
            if renewal.done():
                await renewal
            report = _build_result_report(task, outcome, tuple(artifact_hashes))
            await self.client.report_result(report)
            return task
        finally:
            stop_renewal.set()
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)

    async def run(
        self,
        *,
        stop_event: asyncio.Event,
        idle_poll_seconds: float = 2.0,
        error_backoff_seconds: float = 5.0,
    ) -> None:
        while not stop_event.is_set():
            try:
                task = await self.run_once()
            except RemoteWorkerError:
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=error_backoff_seconds)
                continue
            if task is None:
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=idle_poll_seconds)

    async def _renew_lease(self, task: TaskSpec, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.lease_renew_interval_seconds)
            except TimeoutError:
                try:
                    await self.client.renew(self.capabilities.worker_id, task.task_id)
                except RemoteLeaseLostError:
                    raise
                except (RemoteWorkerError, httpx.TimeoutException, httpx.NetworkError) as exc:
                    raise RemoteWorkerError(
                        f"lease renewal request failed for {task.task_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

    async def _execute(self, task: TaskSpec) -> WorkerExecutionOutcome:
        if task.workload_manifest_sha256 is None:
            return await self.executor(task)
        manifest = await self.client.get_workload_manifest(task.workload_manifest_sha256)
        if manifest.task_kind != task.kind:
            return WorkerExecutionOutcome(
                status=WorkerResultStatus.FAILED,
                error_code="workload_task_kind_mismatch",
                error_message="workload manifest task kind does not match its lease",
            )
        if not worker_can_execute_manifest(self.capabilities, manifest):
            return WorkerExecutionOutcome(
                status=WorkerResultStatus.FAILED,
                error_code="worker_capability_mismatch",
                error_message="Worker does not satisfy the workload manifest requirements",
            )
        if self.workload_executor is None:
            return WorkerExecutionOutcome(
                status=WorkerResultStatus.FAILED,
                error_code="workload_executor_unavailable",
                error_message="Worker has no declarative workload executor configured",
            )
        return await self.workload_executor(task, manifest)


def _build_result_report(
    task: TaskSpec,
    outcome: WorkerExecutionOutcome,
    artifact_hashes: tuple[str, ...],
) -> TaskResultReport:
    payload = {
        "task_id": task.task_id,
        "worker_id": task.worker_id,
        "attempt": task.attempt,
        "status": outcome.status.value,
        "artifact_sha256_values": artifact_hashes,
        "error_code": outcome.error_code,
        "error_message": outcome.error_message,
    }
    report_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return TaskResultReport(report_id=report_id, **payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jitter(delay: float, ratio: float) -> float:
    return max(0.0, delay * (1 + random.uniform(-ratio, ratio)))
