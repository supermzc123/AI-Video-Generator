from enum import StrEnum

from pydantic import Field, model_validator

from .chain import SHA256_PATTERN, AssetRole, FrozenModel


class AudioPolicy(StrEnum):
    H3_NATIVE = "h3_native"
    H3_WITH_EXTERNAL = "h3_with_external"
    MUTED = "muted"


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class AssetScope(StrEnum):
    COMMON = "common"
    SHOT = "shot"
    SEGMENT = "segment"


class ReviewState(StrEnum):
    DRAFT = "draft"
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"


class ProjectSpec(FrozenModel):
    schema_version: str = "1.0"
    project_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    name: str = Field(min_length=1, max_length=200)
    width: int = Field(default=1024, ge=32)
    height: int = Field(default=608, ge=32)
    fps: int = Field(default=24, ge=1, le=120)
    target_duration_seconds: float = Field(gt=0)
    audio_policy: AudioPolicy = AudioPolicy.H3_NATIVE
    external_audio_asset_id: str | None = None

    @model_validator(mode="after")
    def validate_dimensions_and_audio(self) -> "ProjectSpec":
        if self.width % 32 or self.height % 32:
            raise ValueError("project width and height must be multiples of 32")
        if self.audio_policy == AudioPolicy.H3_WITH_EXTERNAL and not self.external_audio_asset_id:
            raise ValueError("external audio policy requires external_audio_asset_id")
        if self.audio_policy != AudioPolicy.H3_WITH_EXTERNAL and self.external_audio_asset_id:
            raise ValueError("external_audio_asset_id requires h3_with_external audio policy")
        return self


class ShotSpec(FrozenModel):
    schema_version: str = "1.0"
    shot_revision_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    target_duration_seconds: float = Field(gt=0)
    segment_ids: tuple[str, ...] = ()
    review_state: ReviewState = ReviewState.DRAFT


class AssetRecord(FrozenModel):
    schema_version: str = "1.0"
    asset_revision_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    media_type: MediaType
    scope: AssetScope
    role: AssetRole | None = None
    sha256: str = Field(pattern=SHA256_PATTERN)
    storage_uri: str = Field(min_length=1)
    original_name: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    duration_seconds: float | None = Field(default=None, gt=0)
    source_task_id: str | None = None

    @model_validator(mode="after")
    def validate_media_metadata(self) -> "AssetRecord":
        if self.media_type == MediaType.IMAGE and (self.width is None or self.height is None):
            raise ValueError("image assets require width and height")
        return self
