from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .chain import SHA256_PATTERN, FrozenModel
from .workflow import WorkflowApproval


class H3AccelerationMode(StrEnum):
    STANDARD = "standard"
    TURBO = "turbo"


class H3AttentionMode(StrEnum):
    NATIVE = "native"
    KITCHEN = "kitchen"
    # Accepted for stored v1 profiles. It now resolves to the native Kitchen backend.
    SAGE = "sage"


class H3InputTarget(FrozenModel):
    node_id: str = Field(min_length=1)
    input_name: str = Field(min_length=1)


class H3TurboProfile(FrozenModel):
    source_repository: str = "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
    source_commit: str = Field(
        default="4274783a23afcfdbea3b4876cb79effd6c510785",
        pattern=r"^[0-9a-f]{40}$",
    )
    lora_loader_node_id: str = Field(min_length=1)
    sampler_node_id: str = Field(min_length=1)
    scheduler_node_id: str | None = None
    lora_name_target: H3InputTarget
    strength_target: H3InputTarget
    low_vram_target: H3InputTarget | None = None
    steps_target: H3InputTarget
    scheduler_name_target: H3InputTarget | None = None
    lora_name: str = "minimax_h3_turbo_v4_step600_ema.safetensors"
    strength: float = Field(default=1.0, ge=0.5, le=1.2)
    steps: int = Field(default=6, ge=4, le=8)
    low_vram: bool = False

    @model_validator(mode="after")
    def validate_targets(self) -> "H3TurboProfile":
        loader_targets = (
            (self.lora_name_target, "lora_name"),
            (self.strength_target, "strength"),
            (self.low_vram_target, "low_vram"),
        )
        for target, expected_input in loader_targets:
            if target is None:
                continue
            if target.node_id != self.lora_loader_node_id:
                raise ValueError(f"{expected_input} target must belong to the LoRA loader node")
            if target.input_name != expected_input:
                raise ValueError(f"official Turbo LoRA input must be {expected_input}")
        if self.scheduler_node_id is not None:
            if self.steps_target.node_id != self.scheduler_node_id:
                raise ValueError("steps target must belong to the BasicScheduler node")
            if self.steps_target.input_name != "steps":
                raise ValueError("official BasicScheduler steps input must be steps")
            if (
                self.scheduler_name_target is not None
                and self.scheduler_name_target.node_id != self.scheduler_node_id
            ):
                raise ValueError("scheduler name target must belong to the scheduler node")
            if (
                self.scheduler_name_target is not None
                and self.scheduler_name_target.input_name != "scheduler"
            ):
                raise ValueError("official BasicScheduler scheduler input must be scheduler")
        elif self.steps_target.node_id != self.sampler_node_id:
            raise ValueError("steps target must belong to the sampler node")
        return self


class H3WorkflowProfile(FrozenModel):
    schema_version: str = "1.0"
    profile_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    name: str = Field(min_length=1, max_length=200)
    approval: WorkflowApproval = WorkflowApproval.DRAFT
    workflow_sha256: str = Field(pattern=SHA256_PATTERN)
    node_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    raw_workflow: dict[str, dict[str, Any]]
    acceleration: H3AccelerationMode = H3AccelerationMode.STANDARD
    attention: H3AttentionMode = H3AttentionMode.NATIVE
    turbo: H3TurboProfile | None = None
    sage_attention_node_id: str | None = None

    @model_validator(mode="after")
    def validate_modes(self) -> "H3WorkflowProfile":
        if self.acceleration == H3AccelerationMode.TURBO and self.turbo is None:
            raise ValueError("turbo acceleration requires a turbo profile")
        if self.acceleration != H3AccelerationMode.TURBO and self.turbo is not None:
            raise ValueError("turbo settings require turbo acceleration")
        uses_backend = self.attention in {H3AttentionMode.KITCHEN, H3AttentionMode.SAGE}
        if uses_backend and not self.sage_attention_node_id:
            raise ValueError("Kitchen Attention requires an attention backend node id")
        if not uses_backend and self.sage_attention_node_id:
            raise ValueError("attention backend node id requires Kitchen Attention mode")
        return self
