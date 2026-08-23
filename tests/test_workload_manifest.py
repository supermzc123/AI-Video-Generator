import hashlib
import sqlite3
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    ComfyUIOutput,
    ExecutionTarget,
    TaskKind,
    TaskSpec,
    TaskState,
    TaskWorkloadManifest,
    WorkerCapabilities,
    WorkloadBlob,
    canonical_workload_manifest_bytes,
)
from ai_video_generator.persistence import SQLiteTaskStore, StoreConflictError
from ai_video_generator.services.remote import WorkerResultStatus
from ai_video_generator.workers.remote_worker import (
    ProducedArtifact,
    RemoteWorkerClient,
    RemoteWorkerRuntime,
    RetryPolicy,
    WorkerExecutionOutcome,
)


def workload_manifest() -> TaskWorkloadManifest:
    return TaskWorkloadManifest(
        task_kind=TaskKind.H3_GENERATION,
        workflow_sha256="1" * 64,
        node_schema_sha256="2" * 64,
        prompt={
            "1": {"class_type": "LoadImage", "inputs": {"image": "inputs/a.png"}},
            "2": {"class_type": "SaveVideo", "inputs": {"images": ["1", 0]}},
        },
        input_blobs=(
            WorkloadBlob(
                sha256="3" * 64,
                media_type="image/png",
                mount_path="inputs/a.png",
                role="reference_image",
            ),
        ),
        outputs=(ComfyUIOutput(node_id="2", media_type="video/mp4"),),
        required_node_types=("LoadImage", "SaveVideo"),
        required_model_sha256_values=("4" * 64,),
        workflow_template_id="h3:turbo",
    )


def manifest_task(manifest: TaskWorkloadManifest) -> TaskSpec:
    return TaskSpec(
        task_id="remote-h3-1",
        project_id="project-1",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.READY,
        idempotency_key="5" * 64,
        input_fingerprint="6" * 64,
        workload_manifest_sha256=manifest.sha256,
        execution_target=ExecutionTarget.REMOTE,
        worker_id="ubuntu-1",
    )


def capabilities() -> WorkerCapabilities:
    return WorkerCapabilities(
        worker_id="ubuntu-1",
        platform="linux",
        node_schema_sha256="2" * 64,
        node_types=("LoadImage", "SaveVideo"),
        model_sha256_values=("4" * 64,),
        workflow_template_ids=("h3:turbo",),
    )


def test_manifest_hash_is_canonical_and_rejects_unsafe_mount_path() -> None:
    manifest = workload_manifest()
    assert (
        manifest.sha256 == hashlib.sha256(canonical_workload_manifest_bytes(manifest)).hexdigest()
    )

    with pytest.raises(ValidationError, match="safe relative POSIX path"):
        WorkloadBlob(
            sha256="3" * 64,
            media_type="image/png",
            mount_path="../outside.png",
            role="reference_image",
        )


def test_store_persists_manifest_reference_and_migrates_legacy_tasks(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control-plane.db"
    store = SQLiteTaskStore(database)
    manifest = workload_manifest()
    record = store.put_workload_manifest(manifest)
    assert store.get_workload_manifest(record.sha256) == record
    assert store.add_task(manifest_task(manifest)).workload_manifest_sha256 == record.sha256

    unknown = manifest_task(manifest).model_copy(
        update={
            "task_id": "unknown",
            "idempotency_key": "7" * 64,
            "workload_manifest_sha256": "8" * 64,
        }
    )
    with pytest.raises(StoreConflictError, match="unknown workload manifest"):
        store.add_task(unknown)

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
    assert "workload_manifest_sha256" in columns


@pytest.mark.asyncio
async def test_manifest_api_and_remote_runtime_execute_resolved_contract(
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
    manifest = workload_manifest()
    task = manifest_task(manifest)
    headers = {"Authorization": "Bearer secret-token"}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as setup:
        registered_manifest = await setup.post(
            "/api/v1/workload-manifests", json=manifest.model_dump(mode="json")
        )
        created_task = await setup.post("/api/v1/tasks", json=task.model_dump(mode="json"))
        queued_task = await setup.post(f"/api/v1/tasks/{task.task_id}/run")
    assert registered_manifest.status_code == 201
    assert registered_manifest.json()["sha256"] == manifest.sha256
    assert created_task.status_code == 201
    assert queued_task.status_code == 200

    executed: list[TaskWorkloadManifest] = []

    async def legacy_executor(_: TaskSpec) -> WorkerExecutionOutcome:
        raise AssertionError("manifest task must not use the legacy executor")

    async def workload_executor(
        _: TaskSpec, resolved: TaskWorkloadManifest
    ) -> WorkerExecutionOutcome:
        executed.append(resolved)
        artifact = tmp_path / "result.mp4"
        artifact.write_bytes(b"video-result")
        return WorkerExecutionOutcome(
            status=WorkerResultStatus.SUCCEEDED,
            artifacts=(ProducedArtifact(artifact, "video/mp4"),),
        )

    async with RemoteWorkerClient(
        control_plane_url="http://test",
        token="secret-token",
        allow_insecure_http=True,
        transport=transport,
        retry_policy=RetryPolicy(attempts=1),
    ) as client:
        runtime = RemoteWorkerRuntime(
            client=client,
            capabilities=capabilities(),
            executor=legacy_executor,
            workload_executor=workload_executor,
            lease_renew_interval_seconds=60,
        )
        completed = await runtime.run_once()

    assert completed is not None
    assert executed == [manifest]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as inspect:
        fetched = await inspect.get(
            f"/api/v1/workload-manifests/{manifest.sha256}", headers=headers
        )
        finished = await inspect.get(f"/api/v1/tasks/{task.task_id}")
    assert fetched.status_code == 200
    assert finished.json()["state"] == "succeeded"
