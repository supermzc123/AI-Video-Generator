from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .chain import SHA256_PATTERN, FrozenModel


class PostProcessKind(StrEnum):
    RESTORATION = "restoration"
    INTERPOLATION = "interpolation"
    TRANSCRIPTION = "transcription"


class PostProcessProfileRef(FrozenModel):
    profile_id: str = Field(min_length=1)
    revision: int = Field(ge=1)


class PostProcessModelSelection(FrozenModel):
    profile: PostProcessProfileRef
    model_id: str = Field(min_length=1)
    model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    auxiliary_model_id: str | None = None
    auxiliary_model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class PostProcessProfile(FrozenModel):
    schema_version: str = "1.0"
    profile_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    name: str = Field(min_length=1)
    kind: PostProcessKind
    engine: str = Field(min_length=1)
    required_node_types: tuple[str, ...] = ()
    model_node_type: str | None = None
    model_input_name: str | None = None
    auxiliary_model_node_type: str | None = None
    auxiliary_model_input_name: str | None = None
    supported_target_fps: tuple[Literal[48, 60, 120], ...] = ()
    workflow_file: str | None = None

    @model_validator(mode="after")
    def validate_model_binding(self) -> PostProcessProfile:
        if bool(self.model_node_type) != bool(self.model_input_name):
            raise ValueError("model node type and input name must be configured together")
        if bool(self.auxiliary_model_node_type) != bool(self.auxiliary_model_input_name):
            raise ValueError("auxiliary model binding must be configured together")
        if self.kind != PostProcessKind.INTERPOLATION and self.supported_target_fps:
            raise ValueError("target frame rates only apply to interpolation profiles")
        return self


class PostProcessProfileCapability(FrozenModel):
    profile: PostProcessProfile
    available: bool
    models: tuple[str, ...] = ()
    auxiliary_models: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


class PostProcessingCapabilities(FrozenModel):
    schema_version: str = "1.0"
    server_online: bool
    profiles: tuple[PostProcessProfileCapability, ...]
