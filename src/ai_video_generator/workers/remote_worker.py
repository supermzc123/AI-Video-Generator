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
from urllib.parse import urlsplit

import httpx

from ai_video_generator.domain import (
    TaskSpec,
    TaskState,
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


class RemoteCancellationRequested(RemoteWorkerError):
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
    retryable: bool = False
    stop_evidence: str | None = None
    submission_token: str | None = None
    external_prompt_id: str | None = None

    def __post_init__(self) -> None:
        if self.status == WorkerResultStatus.CANCELLED and not (self.stop_evidence or "").strip():
            raise ValueError("cancelled outcomes require confirmed stop evidence")
        if self.retryable and self.status != WorkerResultStatus.FAILED:
            raise ValueError("only confirmed failed execution may authorize automatic retry")
        error_status = self.status in {
            WorkerResultStatus.FAILED,
            WorkerResultStatus.NEEDS_ATTENTION,
        }
        if error_status and not self.error_code:
            raise ValueError("failed execution outcomes require an error code")
        if not error_status and (self.error_code is not None or self.error_message is not None):
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
            retry_transport=False,
            json={"heartbeat": heartbeat.model_dump(mode="json")},
        )
        payload = response.json()
        return None if payload is None else TaskSpec.model_validate(payload)

    async def renew(self, worker_id: str, task_id: str, attempt_id: str | None = None) -> TaskSpec:
        if not attempt_id:
            raise RemoteLeaseLostError("lease renewal requires the current attempt identity")
        response = await self._request(
            "POST",
            f"/api/v1/workers/{worker_id}/leases/{task_id}/renew",
            json={"attempt_id": attempt_id},
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

    async def _request(
        self, method: str, url: str, *, retry_transport: bool = True, **kwargs: Any
    ) -> httpx.Response:
        policy = self.retry_policy
        attempts = policy.attempts if retry_transport else 1
        delay = policy.initial_delay_seconds
        for attempt in range(attempts):
            try:
                response = await self._client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt + 1 == attempts:
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
                if attempt + 1 == attempts:
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
        self._unrecoverable_attempts: set[tuple[str, str | None]] = set()

    async def run_once(self) -> TaskSpec | None:
        await self.client.register(self.capabilities)
        journal = getattr(self.workload_executor, "journal", None)
        task = None
        if journal is not None:
            for previous in journal.pending_tasks():
                key = (previous.task_id, previous.attempt_id)
                if key in self._unrecoverable_attempts:
                    continue
                try:
                    task = await self.client.renew(
                        self.capabilities.worker_id, previous.task_id, previous.attempt_id
                    )
                except RemoteLeaseLostError:
                    self._unrecoverable_attempts.add(key)
                    journal.update(
                        previous, phase="fenced", detail="control-plane lease no longer owned"
                    )
                    continue
                if task.attempt_id != previous.attempt_id:
                    raise RemoteLeaseLostError("renewal returned a different attempt")
                break
        if task is None:
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
        cancellation_requested = task.state == TaskState.CANCELLING
        try:
            # Let the executor enter its cleanup boundary before cancellation.
            if cancellation_requested:
                await asyncio.sleep(0)
                execution.cancel()
            else:
                done, _ = await asyncio.wait(
                    {execution, renewal}, return_when=asyncio.FIRST_COMPLETED
                )
                if renewal in done:
                    error = renewal.exception()
                    execution.cancel()
                    if isinstance(error, RemoteCancellationRequested):
                        cancellation_requested = True
                    else:
                        await asyncio.gather(execution, return_exceptions=True)
                        if isinstance(error, RemoteWorkerError):
                            raise error
                        raise RemoteWorkerError(f"lease ownership was lost: {error}")
            try:
                outcome = await execution
            except asyncio.CancelledError:
                if not cancellation_requested:
                    raise
                outcome = WorkerExecutionOutcome(
                    status=WorkerResultStatus.NEEDS_ATTENTION,
                    error_code="cancellation_unconfirmed",
                    error_message="executor ended without confirmed external stop",
                )
            except Exception as exc:
                outcome = WorkerExecutionOutcome(
                    status=WorkerResultStatus.NEEDS_ATTENTION,
                    error_code="worker_executor_error",
                    error_message=str(exc)[:2000],
                )
            if cancellation_requested and outcome.status not in {
                WorkerResultStatus.CANCELLED,
                WorkerResultStatus.NEEDS_ATTENTION,
            }:
                outcome = WorkerExecutionOutcome(
                    status=WorkerResultStatus.NEEDS_ATTENTION,
                    error_code="cancellation_completion_race",
                    error_message="execution completed while cancellation required reconciliation",
                    submission_token=outcome.submission_token,
                    external_prompt_id=outcome.external_prompt_id,
                )
            artifact_hashes = []
            for artifact in outcome.artifacts:
                if renewal.done() and not cancellation_requested:
                    await renewal
                transfer = await self.client.upload_artifact(artifact.path, artifact.media_type)
                artifact_hashes.append(transfer.sha256)
            if renewal.done() and not cancellation_requested:
                await renewal
            report = _build_result_report(task, outcome, tuple(artifact_hashes))
            await self.client.report_result(report)
            if journal is not None:
                # Rejected manifests never entered the submission journal.
                with suppress(ValueError):
                    journal.update(task, phase="reported", detail=outcome.status.value)
            return task
        finally:
            stop_renewal.set()
            if not execution.done():
                execution.cancel()
            renewal.cancel()
            await asyncio.gather(execution, renewal, return_exceptions=True)

    async def run(
        self,
        *,
        stop_event: asyncio.Event,
        idle_poll_seconds: float = 2.0,
        error_backoff_seconds: float = 5.0,
    ) -> None:
        while not stop_event.is_set():
            operation = asyncio.create_task(self.run_once())
            stopping = asyncio.create_task(stop_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {operation, stopping}, return_when=asyncio.FIRST_COMPLETED
                )
                if stopping in done:
                    operation.cancel()
                    await asyncio.gather(operation, return_exceptions=True)
                    return
                try:
                    task = await operation
                except (RemoteWorkerError, httpx.HTTPError):
                    with suppress(TimeoutError):
                        await asyncio.wait_for(stop_event.wait(), timeout=error_backoff_seconds)
                    continue
                if task is None:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(stop_event.wait(), timeout=idle_poll_seconds)
            finally:
                stopping.cancel()
                if not operation.done():
                    operation.cancel()
                await asyncio.gather(operation, stopping, return_exceptions=True)

    async def _renew_lease(self, task: TaskSpec, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.lease_renew_interval_seconds)
            except TimeoutError:
                try:
                    renewed = await self.client.renew(
                        self.capabilities.worker_id, task.task_id, task.attempt_id
                    )
                    if renewed.attempt_id != task.attempt_id:
                        raise RemoteLeaseLostError("renewal returned a different attempt")
                    if renewed.state == TaskState.CANCELLING:
                        raise RemoteCancellationRequested("control plane requested cancellation")
                    if renewed.state != TaskState.RUNNING:
                        raise RemoteLeaseLostError("control plane no longer authorizes execution")
                except (RemoteLeaseLostError, RemoteCancellationRequested):
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
        "attempt_id": task.attempt_id,
        "status": outcome.status.value,
        "artifact_sha256_values": artifact_hashes,
        "error_code": outcome.error_code,
        "error_message": outcome.error_message,
        "retryable": outcome.retryable,
        "stop_evidence": outcome.stop_evidence,
        "submission_token": outcome.submission_token,
        "external_prompt_id": outcome.external_prompt_id,
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
