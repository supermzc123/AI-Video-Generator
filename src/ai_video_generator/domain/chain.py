from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetRole(StrEnum):
    IDENTITY = "identity"
    LOCATION = "location"
    PROP = "prop"
    STYLE = "style"
    KEYFRAME = "keyframe"


class GenerationMode(StrEnum):
    T2VA = "t2va"
    I2VA = "i2va"
    FL2VA = "fl2va"
    L2VA = "l2va"
    REF2VA = "ref2va"
    HYBRID = "hybrid"
    V2VA = "v2va"
    RV2VA = "rv2va"


class ContextMode(StrEnum):
    NONE = "none"
    MOTION_CONTEXT = "motion_context"
    RESET = "reset"


class AssetBinding(FrozenModel):
    asset_revision_id: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    role: AssetRole
    priority: int = Field(default=0, ge=0, le=1000)


class IncomingContext(FrozenModel):
    mode: ContextMode = ContextMode.NONE
    predecessor_segment_id: str | None = None
    visual: bool = True
    audio: bool = True
    context_frames: int = Field(default=0, ge=0)
    audio_context_video_frames: int = Field(default=0, ge=0)
    trim_head_frames: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_mode(self) -> "IncomingContext":
        if self.mode == ContextMode.MOTION_CONTEXT:
            if not self.predecessor_segment_id:
                raise ValueError("motion_context requires predecessor_segment_id")
            if not (self.visual or self.audio):
                raise ValueError("motion_context must inherit visual or audio context")
            if self.context_frames == 0 and self.audio_context_video_frames == 0:
                raise ValueError("motion_context requires a positive context window")
        elif self.predecessor_segment_id is not None:
            raise ValueError("only motion_context may reference a predecessor")
        return self


class GenerationSegment(FrozenModel):
    segment_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    prompt_revision_id: str = Field(min_length=1)
    normalized_prompt: str = Field(min_length=1)
    generation_mode: GenerationMode
    fps: int = Field(default=24, ge=1, le=120)
    width: int = Field(ge=32)
    height: int = Field(ge=32)
    sample_frames: int = Field(ge=1)
    visible_frames: int = Field(ge=1)
    common_assets: tuple[AssetBinding, ...] = ()
    local_assets: tuple[AssetBinding, ...] = ()
    incoming_context: IncomingContext = IncomingContext()

    @model_validator(mode="after")
    def validate_frame_budget(self) -> "GenerationSegment":
        if self.width % 32 or self.height % 32:
            raise ValueError("H3 width and height must be multiples of 32")
        if self.sample_frames > self.fps * 15:
            raise ValueError("H3 sample duration must not exceed 15 seconds")
        if (self.sample_frames - 5) % 17:
            raise ValueError("H3 sample_frames must follow the 17k+5 grid")
        if self.visible_frames > self.sample_frames:
            raise ValueError("visible_frames must not exceed sample_frames")
        if self.incoming_context.trim_head_frames >= self.sample_frames:
            raise ValueError("trim_head_frames must leave at least one visible frame")
        if self.visible_frames + self.incoming_context.trim_head_frames > self.sample_frames:
            raise ValueError("visible frames and trimmed head exceed the sample frame budget")
        if self.incoming_context.mode != ContextMode.MOTION_CONTEXT:
            if self.incoming_context.trim_head_frames:
                raise ValueError("only motion_context segments may trim a context head")
        elif self.incoming_context.trim_head_frames != self.incoming_context.context_frames:
            raise ValueError("motion_context trim_head_frames must equal context_frames")
        unused_frames = (
            self.sample_frames - self.incoming_context.trim_head_frames - self.visible_frames
        )
        if unused_frames >= 17:
            raise ValueError("unused sampled tail must be smaller than one H3 grid interval")
        return self


class ChainSpec(FrozenModel):
    schema_version: str = "1.0"
    project_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    shot_revision_id: str = Field(min_length=1)
    segments: tuple[GenerationSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chain(self) -> "ChainSpec":
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")
        if [segment.ordinal for segment in self.segments] != list(range(1, len(self.segments) + 1)):
            raise ValueError("segment ordinals must be consecutive and ordered")

        first = self.segments[0]
        if first.incoming_context.mode == ContextMode.MOTION_CONTEXT:
            raise ValueError("the first segment cannot inherit motion context")

        for previous, current in zip(
            self.segments[:-1],
            self.segments[1:],
            strict=True,
        ):
            incoming = current.incoming_context
            if (
                incoming.mode == ContextMode.MOTION_CONTEXT
                and incoming.predecessor_segment_id != previous.segment_id
            ):
                raise ValueError("motion context must reference the immediately preceding segment")
        return self


class ConditioningStack(FrozenModel):
    schema_version: str = "1.0"
    text_encoder_sha256: str = Field(pattern=SHA256_PATTERN)
    h3_model_sha256: str = Field(pattern=SHA256_PATTERN)
    video_vae_sha256: str = Field(pattern=SHA256_PATTERN)
    audio_vae_sha256: str = Field(pattern=SHA256_PATTERN)
    lora_sha256_values: tuple[str, ...] = ()
    comfyui_commit: str = Field(min_length=7)
    worker_engine_commit: str = Field(min_length=7)
    node_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lora_hashes(self) -> "ConditioningStack":
        import re

        if any(re.fullmatch(SHA256_PATTERN, value) is None for value in self.lora_sha256_values):
            raise ValueError("all LoRA hashes must be lowercase SHA-256 values")
        return self
