from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, model_validator

from .chain import SHA256_PATTERN, FrozenModel
from .tasks import TaskKind, WorkerCapabilities

WORKLOAD_MANIFEST_FORMAT = "avg-comfyui-workload+json-v1"


class WorkloadBlob(FrozenModel):
    """A content-addressed file the Worker materializes before execution."""

    sha256: str = Field(pattern=SHA256_PATTERN)
    media_type: str = Field(min_length=1)
    mount_path: str = Field(min_length=1)
    role: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_mount_path(self) -> WorkloadBlob:
        path = PurePosixPath(self.mount_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.mount_path:
            raise ValueError("workload blob mount_path must be a safe relative POSIX path")
        return self


class ComfyUIOutput(FrozenModel):
    node_id: str = Field(min_length=1)
    media_type: str = Field(min_length=1)


class TaskWorkloadManifest(FrozenModel):
    """Declarative, content-addressed workload accepted by a remote Worker.

    The contract deliberately supports ComfyUI API prompts only. It cannot carry a
    shell command, Python object, pickle payload, or arbitrary executable entrypoint.
    """

    format: Literal["avg-comfyui-workload+json-v1"] = WORKLOAD_MANIFEST_FORMAT
    task_kind: TaskKind
    workflow_sha256: str = Field(pattern=SHA256_PATTERN)
    node_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt: dict[str, dict[str, Any]] = Field(min_length=1)
    input_blobs: tuple[WorkloadBlob, ...] = ()
    outputs: tuple[ComfyUIOutput, ...] = Field(min_length=1)
    required_node_types: tuple[str, ...] = ()
    required_model_sha256_values: tuple[str, ...] = ()
    workflow_template_id: str | None = None
    context: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> TaskWorkloadManifest:
        for node_id, node in self.prompt.items():
            if not node_id or not isinstance(node, dict):
                raise ValueError("ComfyUI prompt node IDs and definitions must be objects")
            if not isinstance(node.get("class_type"), str) or not node["class_type"]:
                raise ValueError(f"ComfyUI prompt node {node_id} requires class_type")
            if not isinstance(node.get("inputs"), dict):
                raise ValueError(f"ComfyUI prompt node {node_id} requires object inputs")
        if any(output.node_id not in self.prompt for output in self.outputs):
            raise ValueError("workload outputs must reference prompt nodes")
        if len(self.input_blobs) != len({blob.mount_path for blob in self.input_blobs}):
            raise ValueError("workload blob mount paths must be unique")
        if len(self.required_node_types) != len(set(self.required_node_types)):
            raise ValueError("required node types must be unique")
        if len(self.required_model_sha256_values) != len(set(self.required_model_sha256_values)):
            raise ValueError("required model SHA-256 values must be unique")
        canonical_workload_manifest_bytes(self)
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_workload_manifest_bytes(self)).hexdigest()


class WorkloadManifestRecord(FrozenModel):
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=1)
    manifest: TaskWorkloadManifest

    @model_validator(mode="after")
    def validate_digest(self) -> WorkloadManifestRecord:
        payload = canonical_workload_manifest_bytes(self.manifest)
        if self.sha256 != hashlib.sha256(payload).hexdigest():
            raise ValueError("workload manifest SHA-256 mismatch")
        if self.byte_size != len(payload):
            raise ValueError("workload manifest byte size mismatch")
        return self


def canonical_workload_manifest_bytes(manifest: TaskWorkloadManifest) -> bytes:
    try:
        serialized = json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("workload manifest must contain JSON-safe values") from exc
    return serialized.encode("utf-8")


def worker_can_execute_manifest(
    capabilities: WorkerCapabilities,
    manifest: TaskWorkloadManifest,
) -> bool:
    return (
        capabilities.node_schema_sha256 == manifest.node_schema_sha256
        and set(manifest.required_node_types).issubset(capabilities.node_types)
        and set(manifest.required_model_sha256_values).issubset(capabilities.model_sha256_values)
        and (
            manifest.workflow_template_id is None
            or manifest.workflow_template_id in capabilities.workflow_template_ids
        )
    )
