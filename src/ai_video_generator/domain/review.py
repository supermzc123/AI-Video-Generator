from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from .chain import FrozenModel


class ReviewIssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ReviewIssueCategory(StrEnum):
    MEDIA = "media"
    BLACK_FRAME = "black_frame"
    FREEZE = "freeze"
    AUDIO = "audio"
    ANATOMY = "anatomy"
    IDENTITY = "identity"
    MOTION = "motion"
    CONTINUITY = "continuity"
    SEMANTIC = "semantic"
    COMPOSITION = "composition"
    ARTIFACT = "artifact"
    OTHER = "other"


class ReviewIssue(FrozenModel):
    category: ReviewIssueCategory
    severity: ReviewIssueSeverity
    message: str = Field(min_length=1, max_length=4000)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    evidence: str | None = Field(default=None, max_length=4000)
    suggested_action: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_range(self) -> ReviewIssue:
        if self.end_seconds is not None and self.start_seconds is None:
            raise ValueError("issue end time requires a start time")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("issue end time cannot precede its start time")
        return self


class ReviewDisposition(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_HUMAN = "needs_human"


class ReviewInputMode(StrEnum):
    VIDEO = "video"
    FRAMES = "frames"


class ReviewDecision(FrozenModel):
    schema_version: str = "1.0"
    decision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    disposition: ReviewDisposition
    confidence: float = Field(ge=0, le=1)
    issues: tuple[ReviewIssue, ...] = ()
    input_mode: ReviewInputMode
    fallback_reason: str | None = Field(default=None, max_length=2000)
    deterministic_checks_passed: bool = True
    created_at: datetime

    @model_validator(mode="after")
    def validate_disposition(self) -> ReviewDecision:
        has_error = any(issue.severity == ReviewIssueSeverity.ERROR for issue in self.issues)
        if self.disposition == ReviewDisposition.ACCEPTED and (
            has_error or not self.deterministic_checks_passed or self.confidence < 0.85
        ):
            raise ValueError("accepted review requires clean checks and confidence >= 0.85")
        return self


class ReworkAction(StrEnum):
    RETRY = "retry"
    CHANGE_SEED = "change_seed"
    REVISE_PROMPT = "revise_prompt"


class ReworkState(StrEnum):
    REQUESTED = "requested"
    QUEUED = "queued"
    NEEDS_PROMPT_REVISION = "needs_prompt_revision"


class ReworkRequest(FrozenModel):
    schema_version: str = "1.0"
    request_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    source_h3_task_id: str = Field(min_length=1)
    review_task_id: str | None = None
    action: ReworkAction
    feedback: str = Field(min_length=1, max_length=4000)
    replacement_seed: int | None = Field(default=None, ge=0)
    state: ReworkState = ReworkState.REQUESTED
    replacement_task_id: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def validate_action(self) -> ReworkRequest:
        if self.action == ReworkAction.CHANGE_SEED and self.replacement_seed is None:
            raise ValueError("change_seed rework requires replacement_seed")
        if self.state == ReworkState.QUEUED and not self.replacement_task_id:
            raise ValueError("queued rework requires replacement_task_id")
        return self
