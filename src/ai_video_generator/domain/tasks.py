from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from .chain import SHA256_PATTERN, FrozenModel


class TaskKind(StrEnum):
    LLM_PLANNING = "llm_planning"
    IMAGE_GENERATION = "image_generation"
    CONDITIONING_ENCODING = "conditioning_encoding"
    H3_GENERATION = "h3_generation"
    AI_REVIEW = "ai_review"
    MODEL_SWITCH = "model_switch"
    SEEDVR2 = "seedvr2"
    RIFE = "rife"
    WHISPER = "whisper"
    MASTER_ASSEMBLY = "master_assembly"
    EXPORT = "export"
    ASSET_TRANSFER = "asset_transfer"


class TaskState(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"
    QUEUED = "queued"
    PAUSED = "paused"
    RUNNING = "running"
    RECOVERING = "recovering"
    RETRY_WAIT = "retry_wait"
    CANCELLING = "cancelling"
    NEEDS_ATTENTION = "needs_attention"
    NEEDS_REVIEW = "needs_review"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


class ExecutionTarget(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class TaskSpec(FrozenModel):
    schema_version: str = "1.0"
    task_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    kind: TaskKind
    state: TaskState = TaskState.BLOCKED
    idempotency_key: str = Field(pattern=SHA256_PATTERN)
    input_fingerprint: str = Field(pattern=SHA256_PATTERN)
    workload_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    depends_on: tuple[str, ...] = ()
    execution_target: ExecutionTarget = ExecutionTarget.LOCAL
    worker_id: str | None = None
    affinity_key: str | None = Field(default=None, max_length=500)
    priority: int = Field(default=0, ge=-100, le=100)
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=20)
    lease_expires_at: datetime | None = None
    comfyui_prompt_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    attempt_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_activity_at: datetime | None = None
    next_retry_at: datetime | None = None
    deadline_at: datetime | None = None
    current_phase: str | None = None
    blocked_reason: str | None = None
    available_actions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution(self) -> "TaskSpec":
        if self.execution_target == ExecutionTarget.REMOTE and not self.worker_id:
            raise ValueError("remote tasks require worker_id")
        if self.lease_expires_at is not None and self.state not in {
            TaskState.RUNNING,
            TaskState.RECOVERING,
            TaskState.CANCELLING,
        }:
            raise ValueError("only active execution attempts may have a lease")
        if self.attempt > self.max_attempts:
            raise ValueError("task attempts exceed max_attempts")
        return self


class WorkerCapabilities(FrozenModel):
    schema_version: str = "1.0"
    worker_id: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    gpu_names: tuple[str, ...] = ()
    total_vram_bytes: tuple[int, ...] = ()
    comfyui_version: str | None = None
    comfyui_commit: str | None = None
    node_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    node_types: tuple[str, ...] = ()
    model_sha256_values: tuple[str, ...] = ()
    workflow_template_ids: tuple[str, ...] = ()
    generation_concurrency_per_gpu: int = Field(default=1, ge=1, le=1)
