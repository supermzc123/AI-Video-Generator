from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .chain import SHA256_PATTERN, FrozenModel
from .tasks import TaskKind

H3_COMMUNITY_SKILLS_COMMIT = "2e2096fe39ef9b5ddd5998cdcf422c6534065115"
H3_OFFICIAL_SKILL_COMMIT = "6da473b48daf91e5aebfb56451f8a0b116348df5"


class ExecutionMode(StrEnum):
    GUIDED = "guided"
    BATCH = "batch"


class ReviewMode(StrEnum):
    HUMAN_AI = "human_ai"
    AI_ONLY = "ai_only"
    MANUAL = "manual"
    NONE = "none"


class ApprovalState(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class BatchState(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MemoryEventKind(StrEnum):
    MESSAGE = "message"
    TOOL = "tool"
    FACT = "fact"
    CONSTRAINT = "constraint"
    APPROVAL = "approval"
    SUMMARY = "summary"


class DecisionSource(StrEnum):
    USER = "user"
    PROJECT_AGENT = "project_agent"
    SUBAGENT = "subagent"
    SYSTEM = "system"


class HarnessSource(FrozenModel):
    repository_url: str = Field(min_length=1)
    commit: str = Field(pattern="^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    license: str | None = None
    redistribution_allowed: bool = False
    installed_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class HarnessDocumentSnapshot(FrozenModel):
    source_id: str = Field(min_length=1, max_length=100)
    source_commit: str = Field(pattern="^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=SHA256_PATTERN)
    content: str = Field(min_length=1, max_length=100_000)
    stages: tuple[str, ...] = Field(min_length=1)


class H3HarnessManifest(FrozenModel):
    schema_version: str = "2.0"
    documents: tuple[HarnessDocumentSnapshot, ...] = Field(min_length=1)
    stage_documents: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_documents(self) -> H3HarnessManifest:
        paths = [item.path for item in self.documents]
        if len(paths) != len(set(paths)):
            raise ValueError("harness manifest document paths must be unique")
        known = set(paths)
        for stage, stage_paths in self.stage_documents.items():
            missing = set(stage_paths) - known
            if not stage.strip() or missing:
                raise ValueError(
                    f"invalid harness stage binding {stage!r}: unknown documents {sorted(missing)}"
                )
        return self


class HarnessBundle(FrozenModel):
    schema_version: str = "1.0"
    harness_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=100)
    workflow_template_id: str | None = None
    sources: tuple[HarnessSource, ...] = ()


class HarnessRevision(FrozenModel):
    schema_version: str = "1.0"
    harness_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=1, ge=1)
    markdown: str = Field(min_length=1, max_length=200_000)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    approval: ApprovalState = ApprovalState.DRAFT
    workflow_template_id: str | None = None
    workflow_revision: int | None = Field(default=None, ge=1)
    runtime_manifest: H3HarnessManifest | None = None
    created_at: datetime

    @model_validator(mode="after")
    def validate_workflow_binding(self) -> HarnessRevision:
        if (self.workflow_template_id is None) != (self.workflow_revision is None):
            raise ValueError("workflow template ID and revision must be provided together")
        return self


class ProjectWorkspaceRevision(FrozenModel):
    schema_version: str = "1.0"
    project_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    payload: dict[str, Any]
    payload_sha256: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime


class ReviewPolicy(FrozenModel):
    configured_mode: ReviewMode = ReviewMode.HUMAN_AI
    effective_mode: ReviewMode = ReviewMode.HUMAN_AI
    human_timeout_seconds: int = Field(default=600, ge=30, le=86_400)
    ai_takeover_at: datetime | None = None
    takeover_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_takeover(self) -> ReviewPolicy:
        if (
            self.effective_mode == ReviewMode.AI_ONLY
            and self.configured_mode == ReviewMode.HUMAN_AI
        ):
            if self.ai_takeover_at is None:
                raise ValueError("AI takeover requires a timestamp")
        elif self.ai_takeover_at is not None:
            raise ValueError("AI takeover timestamp is only valid after human review timeout")
        return self


class ReviewDeadline(FrozenModel):
    project_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    opened_at: datetime
    deadline_at: datetime
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def validate_deadline(self) -> ReviewDeadline:
        if self.deadline_at <= self.opened_at:
            raise ValueError("review deadline must be after its open time")
        return self


class TimeBudget(FrozenModel):
    total_seconds: float = Field(gt=0)
    spent_active_seconds: float = Field(default=0, ge=0)
    estimated_remaining_seconds: float | None = Field(default=None, ge=0)
    max_ai_retries: int = Field(default=2, ge=0, le=20)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.spent_active_seconds)

    @property
    def retry_budget_exhausted(self) -> bool:
        estimate = self.estimated_remaining_seconds
        return estimate is not None and estimate > self.remaining_seconds


class PerformanceSignature(FrozenModel):
    worker_id: str = Field(min_length=1)
    task_kind: TaskKind
    model_fingerprint: str = Field(min_length=1)
    workflow_fingerprint: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    frames: int = Field(ge=1)
    steps: int = Field(ge=1)
    acceleration: str = Field(min_length=1)


class PerformanceEstimate(FrozenModel):
    signature: PerformanceSignature
    sample_count: int = Field(ge=1)
    mean_seconds: float = Field(gt=0)
    p90_seconds: float = Field(gt=0)
    projected_project_seconds: float | None = Field(default=None, gt=0)
    exceeds_budget: bool = False


class ProjectRunState(FrozenModel):
    schema_version: str = "1.0"
    project_id: str = Field(min_length=1)
    execution_mode: ExecutionMode = ExecutionMode.GUIDED
    pending_mode: ExecutionMode | None = None
    paused: bool = False
    outline_approved: bool = False
    generation_revision: int = Field(default=0, ge=0)
    current_stage: str = Field(default="idea", min_length=1, max_length=100)
    review_policy: ReviewPolicy = Field(default_factory=ReviewPolicy)
    time_budget: TimeBudget | None = None
    updated_at: datetime


class ProjectMemoryEvent(FrozenModel):
    schema_version: str = "1.0"
    event_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    kind: MemoryEventKind
    source: DecisionSource
    role: str = Field(min_length=1, max_length=50)
    content: str = Field(min_length=1, max_length=1_000_000)
    input_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    created_at: datetime


class DecisionLedger(FrozenModel):
    schema_version: str = "1.0"
    decision_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    key: str = Field(min_length=1, max_length=200)
    value: dict[str, Any]
    rationale: str = Field(min_length=1, max_length=20_000)
    source: DecisionSource
    assumptions: tuple[str, ...] = ()
    locked: bool = False
    created_at: datetime


class PendingProjectRun(FrozenModel):
    project_id: str = Field(min_length=1)
    start_boundary: str = Field(min_length=1, max_length=100)
    priority: int = Field(default=0, ge=-100, le=100)
    requested_at: datetime


class BatchRunItem(FrozenModel):
    project_id: str = Field(min_length=1)
    task_ids: tuple[str, ...] = ()
    start_boundary: str = Field(default="next_ready", min_length=1, max_length=100)
    priority: int = Field(default=0, ge=-100, le=100)


class BatchRun(FrozenModel):
    schema_version: str = "1.0"
    batch_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    state: BatchState = BatchState.DRAFT
    items: tuple[BatchRunItem, ...] = Field(min_length=1)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_members(self) -> BatchRun:
        project_ids = [item.project_id for item in self.items]
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("a project may appear only once in a batch")
        if self.updated_at < self.created_at:
            raise ValueError("batch updated_at cannot precede created_at")
        return self


class TaskCheckpoint(FrozenModel):
    checkpoint_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    phase: str = Field(min_length=1, max_length=100)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ModelResidencyEvent(FrozenModel):
    event_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    action: str = Field(pattern="^(load|unload|reuse)$")
    task_id: str | None = None
    cache_hit: bool = False
    duration_seconds: float | None = Field(default=None, ge=0)
    created_at: datetime
