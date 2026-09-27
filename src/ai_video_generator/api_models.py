from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ai_video_generator.domain import (
    ChainSpec,
    ConditioningStack,
    ExecutionMode,
    H3WorkflowProfile,
    PerformanceSignature,
    ProjectSpec,
    ReviewMode,
    ReworkAction,
    WorkerCapabilities,
    WorkflowInvocation,
    WorkflowTemplate,
)
from ai_video_generator.services.export import ExportSpec
from ai_video_generator.services.remote import TaskResultReport, WorkerHeartbeat


class DryRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chain: ChainSpec
    conditioning_stack: ConditioningStack


class WorkflowInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_workflow: dict[str, Any] = Field(min_length=1)
    object_info: dict[str, Any] | None = None


class WorkflowCompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: WorkflowTemplate
    invocation: WorkflowInvocation


class H3WorkflowInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: H3WorkflowProfile
    object_info: dict[str, Any] | None = None


class H3WorkflowCompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: H3WorkflowProfile
    object_info: dict[str, Any] | None = None


class ExportPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ExportSpec


class WorkerRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capabilities: WorkerCapabilities


class WorkerLeaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heartbeat: WorkerHeartbeat


class WorkerLeaseRenewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(min_length=1)


class WorkerResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TaskResultReport


class TaskReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback: str | None = Field(default=None, max_length=4000)


class WorkspaceSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=1)
    payload: dict[str, Any]


class ProjectCommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectSpec
    payload: dict[str, Any]


class ProjectModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ExecutionMode


class ProjectPauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paused: bool


class RuntimeSettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comfyui_root: str | None = None
    comfyui_base_url: str
    request_timeout_seconds: float = Field(gt=0, le=60)
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    clear_llm_api_key: bool = False
    llm_timeout_seconds: float = Field(gt=0, le=300)
    llm_first_token_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_stream_idle_timeout_seconds: float = Field(default=600.0, gt=0, le=3600)
    llm_operation_timeout_seconds: float = Field(default=300, gt=0, le=3600)
    llm_concurrency: int = Field(default=2, ge=1, le=16)
    llm_video_capable: bool = False
    network_proxy: str | None = None
    h3_diffusion_model: str = Field(
        default="minimax_h3_fl2va_pruned_int8_convrot.safetensors", min_length=1
    )
    h3_text_encoder: str = Field(
        default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors", min_length=1
    )
    h3_video_vae: str = Field(default="minimax_h3_video_vae_fp16.safetensors", min_length=1)
    h3_audio_vae: str = Field(default="minimax_h3_audio_vae_fp32.safetensors", min_length=1)
    h3_turbo_lora: str = Field(
        default="minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        min_length=1,
    )
    h3_turbo_enabled: bool = True
    h3_sage_attention_enabled: bool = False
    h3_low_vram: bool = True
    h3_steps: int = Field(default=6, ge=4, le=50)
    h3_conditioning_workflow_template_id: str | None = None
    h3_conditioning_workflow_revision: int | None = Field(default=None, ge=1)
    h3_diffusion_workflow_template_id: str | None = None
    h3_diffusion_workflow_revision: int | None = Field(default=None, ge=1)


class ComfyNodeInstallStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str
    label: str
    succeeded: bool
    message: str


class ComfyNodeInstallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    succeeded: bool
    restart_required: bool
    comfyui_root: str
    steps: tuple[ComfyNodeInstallStep, ...]


class ReviewModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ReviewMode
    human_timeout_seconds: int = Field(default=600, ge=30, le=86_400)


class PerformanceSampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signature: PerformanceSignature
    duration_seconds: float = Field(gt=0)


class PerformanceEstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signature: PerformanceSignature
    projected_tasks: int | None = Field(default=None, ge=1)


class ReviewDeadlineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opened_at: datetime
    deadline_at: datetime


class ProjectAgentOperationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    instruction: str = Field(min_length=1, max_length=50_000)
    display_instruction: str | None = Field(default=None, min_length=1, max_length=1000)
    conversation_parent_event_id: str | None = Field(default=None, min_length=1)
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    locked_paths: tuple[str, ...] = ()
    commit: bool = False


class SegmentReworkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_h3_task_id: str = Field(min_length=1)
    review_task_id: str | None = None
    action: ReworkAction
    feedback: str = Field(min_length=1, max_length=4000)
    replacement_seed: int | None = Field(default=None, ge=0)


class ReworkMarkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: str = Field(min_length=1)
    action: ReworkAction
    feedback: str | None = Field(default=None, max_length=4000)


class ConfirmTaskRetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_duplicate_execution: bool = False
    idempotency_key: str = Field(min_length=1, max_length=200)


class TaskCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["project", "batch"]
    scope_id: str = Field(min_length=1, max_length=200)
    action: Literal["pause", "resume", "cancel", "retry", "reconcile", "confirm_retry", "restart"]
    task_ids: tuple[str, ...] = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=1, max_length=200)
    confirm_duplicate_execution: bool = False
    replacement_seed: int | None = Field(default=None, ge=0)
    source: str = Field(default="human", pattern="^(human|ai)$")


class ReworkMarkerUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ReworkAction | None = None
    feedback: str | None = Field(default=None, min_length=1, max_length=4000)
    replacement_seed: int | None = Field(default=None, ge=0)


class ImageTaskRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_workspace_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
