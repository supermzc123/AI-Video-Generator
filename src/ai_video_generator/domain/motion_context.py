from enum import StrEnum

from pydantic import Field, model_validator

from .chain import FrozenModel


class MotionContextHandoff(StrEnum):
    AV_LATENT = "av_latent"


class MotionContextProfile(FrozenModel):
    """Pinned execution contract for one Motion Context implementation."""

    provider_id: str = Field(min_length=1)
    source_repository: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    node_types: tuple[str, ...] = Field(min_length=1)
    native_fps: int = Field(gt=0)
    context_lengths: tuple[int, ...] = Field(min_length=1)
    handoff: MotionContextHandoff

    @model_validator(mode="after")
    def validate_contract(self) -> "MotionContextProfile":
        if len(self.node_types) != len(set(self.node_types)):
            raise ValueError("Motion Context node types must be unique")
        if any(not node_type for node_type in self.node_types):
            raise ValueError("Motion Context node types must not be blank")
        if tuple(sorted(set(self.context_lengths))) != self.context_lengths:
            raise ValueError("Motion Context lengths must be sorted and unique")
        if any(length <= 0 for length in self.context_lengths):
            raise ValueError("Motion Context lengths must be positive")
        return self
