from datetime import datetime
from typing import Any

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
    h3_sage_attention_enabled: bool = True
    h3_low_vram: bool = True
    h3_steps: int = Field(default=6, ge=4, le=50)


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
