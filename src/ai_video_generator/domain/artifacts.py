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
