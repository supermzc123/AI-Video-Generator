from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from .chain import FrozenModel
from .review import ReworkAction


class GenerationBatchKind(StrEnum):
    INITIAL = "initial"
    REWORK = "rework"


class GenerationBatchState(StrEnum):
    PREPARING = "preparing"
    SEALED = "sealed"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class GenerationBatch(FrozenModel):
    batch_id: str
    project_id: str
    kind: GenerationBatchKind
    state: GenerationBatchState = GenerationBatchState.PREPARING
    generation_number: int = Field(ge=1)
    segment_ids: tuple[str, ...]
    task_ids: tuple[str, ...] = ()
    encoding_task_ids: tuple[str, ...] = ()
    model_switch_task_id: str | None = None
    dispatch_requested: bool = False
    sealed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class SegmentVersionState(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISCARDED = "discarded"


class SegmentGenerationVersion(FrozenModel):
    version_id: str
    project_id: str
    shot_id: str
    segment_id: str
    runtime_segment_id: str = ""
    segment_index: int = Field(ge=0)
    generation_number: int = Field(ge=1)
    batch_id: str
    task_id: str
    parent_version_id: str | None = None
    predecessor_version_id: str | None = None
    prompt_revision_id: str | None = None
    seed: int = Field(ge=0)
    state: SegmentVersionState = SegmentVersionState.PLANNED
    artifact_id: str | None = None
    discard_reason: str | None = None
    created_at: datetime
    activated_at: datetime | None = None


class ReworkMarkerState(StrEnum):
    DRAFT = "draft"
    PREPARING = "preparing"
    SEALED = "sealed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class ReworkMarker(FrozenModel):
    marker_id: str
    project_id: str
    shot_id: str
    segment_id: str
    segment_index: int = Field(ge=0)
    source_version_id: str
    action: ReworkAction
    feedback: str = Field(min_length=1, max_length=4000)
    replacement_seed: int | None = Field(default=None, ge=0)
    state: ReworkMarkerState = ReworkMarkerState.DRAFT
    batch_id: str | None = None
    source: str = Field(pattern="^(human|ai)$")
    created_at: datetime
    updated_at: datetime

    def model_post_init(self, __context: object) -> None:
        if self.action == ReworkAction.CHANGE_SEED and self.replacement_seed is None:
            raise ValueError("change_seed marker requires replacement_seed")
