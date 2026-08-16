from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from .chain import SHA256_PATTERN, FrozenModel


class ArtifactState(StrEnum):
    PLANNED = "planned"
    READY = "ready"
    STALE = "stale"
    INVALID = "invalid"


class ConditioningArtifact(FrozenModel):
    schema_version: str = "1.0"
    artifact_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=SHA256_PATTERN)
    payload_format: Literal["safetensors+json-v1"] = "safetensors+json-v1"
    tensor_path: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    segment_ids: tuple[str, ...] = Field(min_length=1)
    state: ArtifactState = ArtifactState.PLANNED
    blob_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    byte_size: int | None = Field(default=None, ge=0)


class ArtifactKind(StrEnum):
    VIDEO_SEGMENT = "video_segment"
    VIDEO_MASTER = "video_master"
    VIDEO_EXPORT = "video_export"
    SUBTITLE = "subtitle"


class ArtifactDescriptor(FrozenModel):
    schema_version: str = "1.0"
    artifact_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    kind: ArtifactKind
    media_type: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    segment_id: str | None = None
    media_url: str = Field(min_length=1)
    created_at: datetime
