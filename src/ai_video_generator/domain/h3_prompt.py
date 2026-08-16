from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .chain import FrozenModel, GenerationMode


class H3AssetKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class H3AssetPromptRole(StrEnum):
    REFERENCE = "reference"
    IDENTITY = "identity"
    CHARACTER = "character"
    OBJECT = "object"
    SCENE = "scene"
    STYLE = "style"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"


class H3ShotStrategy(StrEnum):
    AUTO = "auto"
    SINGLE = "single"
    MULTI = "multi"


class H3AssetInput(FrozenModel):
    asset_id: str = Field(min_length=1)
    label: str = Field(pattern=r"^<(Picture|Video|Audio) [1-9][0-9]*>$")
    kind: H3AssetKind
    role: H3AssetPromptRole
    preservation: str = Field(min_length=1)


class H3PromptRequest(FrozenModel):
    operation_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    creative_brief: str = Field(min_length=1)
    duration_seconds: float = Field(ge=4, le=15)
    fps: int = Field(default=24)
    assets: tuple[H3AssetInput, ...] = ()
    shot_strategy: H3ShotStrategy = H3ShotStrategy.AUTO
    project_memory: str = ""
    constraints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> H3PromptRequest:
        if self.fps != 24:
            raise ValueError("MiniMax H3 prompts require 24 FPS")
        labels = [asset.label for asset in self.assets]
        if len(labels) != len(set(labels)):
            raise ValueError("asset labels must be unique")
        ids = [asset.asset_id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("asset IDs must be unique")
        return self


class H3DirectorDecision(FrozenModel):
    operation_id: str = Field(min_length=1)
    mode: GenerationMode
    use_multishot: bool
    rationale: str = Field(min_length=1)
    assumptions: tuple[str, ...] = ()


class H3ShotBeat(FrozenModel):
    shot_number: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    composition: str = Field(min_length=2)
    subjects: str = Field(min_length=2)
    environment: str = Field(min_length=2)
    action: str = Field(min_length=2)
    camera: str = Field(min_length=2)
    sound: str = Field(min_length=2)
    end_state: str = Field(min_length=2)

    @model_validator(mode="after")
    def validate_interval(self) -> H3ShotBeat:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("shot end must be after shot start")
        return self


class H3MultishotPlan(FrozenModel):
    operation_id: str = Field(min_length=1)
    shots: tuple[H3ShotBeat, ...] = Field(min_length=2)
    continuity_strategy: str = Field(min_length=2)


class H3PromptCandidate(FrozenModel):
    operation_id: str = Field(min_length=1)
    mode: GenerationMode
    timeline: tuple[H3ShotBeat, ...] = Field(min_length=1)
    integrated_multimodal_description: str | None = None
    overall_soundscape: str = Field(min_length=2)
    non_diegetic_music: str = Field(min_length=1)
    subject_definitions: str | None = None
    summary: str | None = None
    retention_analysis: str | None = None
    detailed_description: str | None = None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> H3PromptCandidate:
        ref_fields = (
            self.subject_definitions,
            self.summary,
            self.retention_analysis,
            self.detailed_description,
        )
        if self.mode == GenerationMode.REF2VA:
            if self.integrated_multimodal_description is not None or any(
                value is None or not value.strip() for value in ref_fields
            ):
                raise ValueError("Ref2VA requires exactly the six reference prompt sections")
        elif self.mode in {
            GenerationMode.T2VA,
            GenerationMode.I2VA,
            GenerationMode.FL2VA,
            GenerationMode.L2VA,
        }:
            if not self.integrated_multimodal_description or any(
                value is not None for value in ref_fields
            ):
                raise ValueError("base H3 modes require exactly the three base prompt sections")
        else:
            raise ValueError("unsupported H3 prompt mode")
        return self


class H3ReviewSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class H3ReviewFinding(FrozenModel):
    severity: H3ReviewSeverity
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class H3ReviewerDecision(FrozenModel):
    operation_id: str = Field(min_length=1)
    approved: bool
    findings: tuple[H3ReviewFinding, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_approval(self) -> H3ReviewerDecision:
        has_errors = any(item.severity == H3ReviewSeverity.ERROR for item in self.findings)
        if self.approved == has_errors:
            raise ValueError("review approval must agree with error findings")
        return self


class H3PromptResult(FrozenModel):
    request: H3PromptRequest
    director: H3DirectorDecision
    multishot_plan: H3MultishotPlan | None = None
    candidate: H3PromptCandidate
    review_history: tuple[H3ReviewerDecision, ...]
    repair_passes: int = Field(ge=0, le=2)
    execution_prompt: str = Field(min_length=1, max_length=7000)
