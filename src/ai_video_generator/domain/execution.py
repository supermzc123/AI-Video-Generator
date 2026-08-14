from enum import StrEnum
from typing import Literal

from pydantic import Field

from .artifacts import ConditioningArtifact
from .chain import FrozenModel


class ModelResidency(StrEnum):
    UNLOADED = "unloaded"
    CONDITIONING = "conditioning"
    DIFFUSION = "diffusion"


class TaskType(StrEnum):
    ENCODE_CONDITIONING = "encode_conditioning"
    UNLOAD_CONDITIONING = "unload_conditioning"
    LOAD_DIFFUSION = "load_diffusion"
    GENERATE_SEGMENT = "generate_segment"


class PlannedTask(FrozenModel):
    task_id: str = Field(min_length=1)
    task_type: TaskType
    segment_ids: tuple[str, ...] = ()
    conditioning_fingerprint: str | None = None
    depends_on: tuple[str, ...] = ()
    required_residency: ModelResidency
    resulting_residency: ModelResidency


class DryRunPlan(FrozenModel):
    schema_version: str = "1.0"
    project_id: str
    run_id: str
    shot_revision_id: str
    engine: str = "motion-director"
    engine_commit: str
    ready_for_submission: Literal[False] = False
    conditioning_artifacts: tuple[ConditioningArtifact, ...]
    tasks: tuple[PlannedTask, ...]
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
