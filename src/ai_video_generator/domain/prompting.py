from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from .chain import SHA256_PATTERN, FrozenModel, GenerationMode
from .project import AssetScope


class ProjectAssetState(StrEnum):
    AVAILABLE = "available"
    MISSING_BLOB = "missing_blob"
    RETIRED = "retired"


class ProjectAssetMediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ProjectAssetPurpose(StrEnum):
    REFERENCE = "reference"
    CHARACTER = "character"
    SCENE = "scene"
    PROP = "prop"
    STYLE = "style"
    KEYFRAME = "keyframe"


class ProjectAssetSource(StrEnum):
    UPLOAD = "upload"
    GENERATED = "generated"
    LEGACY = "legacy"


class AssetGenerationCandidateState(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DISCARDED = "discarded"


class AssetGenerationCandidate(FrozenModel):
    """A generated image awaiting explicit replacement approval."""

    schema_version: str = "1.0"
    candidate_id: str = Field(min_length=1, max_length=200)
    project_id: str = Field(min_length=1, max_length=200)
    asset_plan_id: str = Field(min_length=1, max_length=200)
    source_task_id: str = Field(min_length=1)
    current_asset_id: str | None = Field(default=None, min_length=1)
    name: str = Field(min_length=1, max_length=200)
    kind: ProjectAssetPurpose = ProjectAssetPurpose.REFERENCE
    scope: AssetScope = AssetScope.COMMON
    shot_id: str | None = Field(default=None, min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    preview_sha256: str = Field(pattern=SHA256_PATTERN)
    mime_type: str = Field(min_length=1, max_length=200)
    byte_size: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    state: AssetGenerationCandidateState = AssetGenerationCandidateState.PENDING
    accepted_asset_id: str | None = Field(default=None, min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def validate_state(self) -> AssetGenerationCandidate:
        if self.scope == AssetScope.SHOT and self.shot_id is None:
            raise ValueError("shot-scoped candidates require shot_id")
        if self.scope != AssetScope.SHOT and self.shot_id is not None:
            raise ValueError("only shot-scoped candidates may carry shot_id")
        if self.state == AssetGenerationCandidateState.ACCEPTED:
            if self.accepted_asset_id is None:
                raise ValueError("accepted candidate requires accepted_asset_id")
        elif self.accepted_asset_id is not None:
            raise ValueError("only accepted candidates may reference an accepted asset")
        return self


class ProjectAsset(FrozenModel):
    """An immutable revision of a user-facing project image."""

    schema_version: str = "1.0"
    asset_id: str = Field(min_length=1, max_length=200)
    project_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=1, ge=1)
    name: str = Field(min_length=1, max_length=200)
    original_name: str = Field(min_length=1, max_length=500)
    state: ProjectAssetState = ProjectAssetState.AVAILABLE
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    preview_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    mime_type: str | None = Field(default=None, min_length=1, max_length=200)
    media_kind: ProjectAssetMediaKind = ProjectAssetMediaKind.IMAGE
    byte_size: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    duration_seconds: float | None = Field(default=None, gt=0)
    frame_rate: float | None = Field(default=None, gt=0)
    has_audio: bool | None = None
    kind: ProjectAssetPurpose = ProjectAssetPurpose.REFERENCE
    scope: AssetScope = AssetScope.COMMON
    shot_id: str | None = Field(default=None, min_length=1)
    shot_ids: tuple[str, ...] = ()
    segment_id: str | None = Field(default=None, min_length=1)
    source: ProjectAssetSource = ProjectAssetSource.UPLOAD
    source_task_id: str | None = Field(default=None, min_length=1)
    created_at: datetime

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized.startswith("@"):
            raise ValueError("asset name must be non-empty and must not start with @")
        return normalized

    @model_validator(mode="after")
    def validate_blob_and_scope(self) -> ProjectAsset:
        metadata = (self.sha256, self.mime_type, self.byte_size)
        if self.state == ProjectAssetState.AVAILABLE and any(value is None for value in metadata):
            raise ValueError(
                "available project assets require blob and image metadata or media metadata"
            )
        if self.state == ProjectAssetState.AVAILABLE:
            if self.media_kind == ProjectAssetMediaKind.IMAGE and (
                self.width is None or self.height is None
            ):
                raise ValueError("available images require dimensions")
            if self.media_kind == ProjectAssetMediaKind.VIDEO and (
                self.width is None
                or self.height is None
                or self.duration_seconds is None
                or self.frame_rate is None
            ):
                raise ValueError("available videos require dimensions, duration, and frame rate")
            if (
                self.media_kind == ProjectAssetMediaKind.AUDIO
                and self.duration_seconds is None
            ):
                raise ValueError("available audio references require duration")
        if self.state == ProjectAssetState.MISSING_BLOB and self.sha256 is not None:
            raise ValueError("missing_blob assets cannot reference a blob")
        if any(not shot_id.strip() for shot_id in self.shot_ids):
            raise ValueError("asset shot_ids must contain non-empty strings")
        if len(set(self.shot_ids)) != len(self.shot_ids):
            raise ValueError("asset shot_ids must be unique")
        if self.scope == AssetScope.SHOT:
            if self.shot_id is None or self.segment_id is not None:
                raise ValueError("shot-scoped assets require only shot_id")
        elif self.scope == AssetScope.SEGMENT:
            if self.segment_id is None:
                raise ValueError("segment-scoped assets require segment_id")
        elif self.shot_id is not None or self.segment_id is not None:
            raise ValueError("common assets cannot reference a shot or segment")
        return self


class TextNode(FrozenModel):
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=1_000_000)


class AssetMentionNode(FrozenModel):
    kind: Literal["asset_mention"] = "asset_mention"
    asset_id: str = Field(min_length=1, max_length=200)
    display_name: str = Field(min_length=1, max_length=200)


RichTextNode = Annotated[TextNode | AssetMentionNode, Field(discriminator="kind")]
_RICH_TEXT_NODE_ADAPTER = TypeAdapter(RichTextNode)


class RichTextDocument(FrozenModel):
    schema_version: str = "1.0"
    nodes: tuple[RichTextNode, ...] = ()

    @property
    def referenced_asset_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                node.asset_id for node in self.nodes if isinstance(node, AssetMentionNode)
            )
        )

    @classmethod
    def from_plain_text(cls, text: str) -> RichTextDocument:
        return cls(nodes=() if not text else (TextNode(text=text),))

    @field_validator("nodes", mode="before")
    @classmethod
    def parse_nodes(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(_RICH_TEXT_NODE_ADAPTER.validate_python(item) for item in value)
        return value


class AssetPlanState(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    SATISFIED = "satisfied"
    STALE = "stale"


class AssetResolutionSource(StrEnum):
    DEFAULT = "default"
    AI = "ai"
    MANUAL = "manual"


class AssetPlan(FrozenModel):
    schema_version: str = "1.0"
    plan_id: str = Field(min_length=1, max_length=200)
    project_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=1, ge=1)
    name: str = Field(min_length=1, max_length=200)
    description: RichTextDocument = Field(default_factory=RichTextDocument)
    purpose: ProjectAssetPurpose
    scope: AssetScope = AssetScope.COMMON
    shot_id: str | None = Field(default=None, min_length=1)
    segment_id: str | None = Field(default=None, min_length=1)
    fulfilled_by_asset_id: str | None = Field(default=None, min_length=1)
    state: AssetPlanState = AssetPlanState.DRAFT
    width: int = Field(default=1280, ge=64, le=4096, multiple_of=8)
    height: int = Field(default=1280, ge=64, le=4096, multiple_of=8)
    resolution_source: AssetResolutionSource = AssetResolutionSource.DEFAULT
    created_at: datetime

    @model_validator(mode="after")
    def validate_scope(self) -> AssetPlan:
        if self.scope == AssetScope.SHOT and not self.shot_id:
            raise ValueError("shot-scoped asset plans require shot_id")
        if self.scope == AssetScope.SEGMENT and not self.segment_id:
            raise ValueError("segment-scoped asset plans require segment_id")
        if self.scope == AssetScope.COMMON and (self.shot_id or self.segment_id):
            raise ValueError("common asset plans cannot reference a shot or segment")
        if self.state == AssetPlanState.SATISFIED and not self.fulfilled_by_asset_id:
            raise ValueError("satisfied asset plans require a project asset")
        return self


class PromptRevisionState(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"


class PromptIssueSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class PromptReviewIssue(FrozenModel):
    code: str = Field(min_length=1, max_length=100)
    severity: PromptIssueSeverity
    message: str = Field(min_length=1, max_length=10_000)
    path: str | None = Field(default=None, max_length=500)


class PromptReviewResult(FrozenModel):
    schema_version: str = "1.0"
    reviewer_harness_id: str = Field(min_length=1, max_length=200)
    reviewer_harness_revision: int = Field(ge=1)
    issues: tuple[PromptReviewIssue, ...] = ()
    structure_complete: bool
    references_complete: bool
    timeline_complete: bool
    contradictions_absent: bool
    audio_consistent: bool
    within_length_limit: bool
    reviewed_at: datetime

    @property
    def execution_ready(self) -> bool:
        checks = (
            self.structure_complete,
            self.references_complete,
            self.timeline_complete,
            self.contradictions_absent,
            self.audio_consistent,
            self.within_length_limit,
        )
        return all(checks) and not any(
            issue.severity == PromptIssueSeverity.ERROR for issue in self.issues
        )


class ImagePromptRevision(FrozenModel):
    schema_version: str = "1.0"
    prompt_revision_id: str = Field(min_length=1, max_length=200)
    project_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=1, ge=1)
    asset_plan_id: str = Field(min_length=1, max_length=200)
    workflow_template_id: str = Field(min_length=1, max_length=200)
    workflow_revision: int = Field(ge=1)
    harness_id: str = Field(min_length=1, max_length=200)
    harness_revision: int = Field(ge=1)
    prompt: RichTextDocument
    negative_prompt: RichTextDocument = Field(default_factory=RichTextDocument)
    reference_asset_ids: tuple[str, ...] = ()
    state: PromptRevisionState = PromptRevisionState.DRAFT
    locked: bool = False
    created_at: datetime


class H3PromptRevision(FrozenModel):
    schema_version: str = "2.0"
    prompt_revision_id: str = Field(min_length=1, max_length=200)
    project_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=1, ge=1)
    segment_id: str = Field(min_length=1, max_length=200)
    generation_mode: GenerationMode
    harness_id: str = Field(min_length=1, max_length=200)
    harness_revision: int = Field(ge=1)
    reference_asset_ids: tuple[str, ...] = ()
    execution_prompt: str = Field(min_length=1, max_length=7000)
    harness_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    route: str | None = Field(default=None, max_length=100)
    asset_role_ledger: tuple[dict[str, object], ...] = ()
    assumptions: tuple[str, ...] = ()
    stage_trace: tuple[dict[str, object], ...] = ()
    terminal_state: str | None = Field(default=None, min_length=2, max_length=4000)
    review: PromptReviewResult
    state: PromptRevisionState = PromptRevisionState.DRAFT
    locked: bool = False
    created_at: datetime

    @model_validator(mode="after")
    def validate_approval(self) -> H3PromptRevision:
        if self.state == PromptRevisionState.APPROVED and not self.review.execution_ready:
            raise ValueError("approved H3 prompts must pass deterministic and harness review")
        if self.generation_mode == GenerationMode.REF2VA and not self.reference_asset_ids:
            raise ValueError("Ref2VA prompts require at least one reference asset")
        return self

class PromptSet(FrozenModel):
    schema_version: str = "1.0"
    prompt_set_id: str = Field(min_length=1, max_length=200)
    project_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=1, ge=1)
    asset_plan_ids: tuple[str, ...] = ()
    image_prompt_revision_ids: tuple[str, ...] = ()
    h3_prompt_revision_ids: tuple[str, ...] = ()
    created_at: datetime


class StageGenerationState(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class StageGenerationCheckpoint(FrozenModel):
    schema_version: str = "1.0"
    checkpoint_id: str = Field(min_length=1, max_length=200)
    project_id: str = Field(min_length=1, max_length=200)
    stage: str = Field(min_length=1, max_length=100)
    idempotency_key: str = Field(pattern=SHA256_PATTERN)
    state: StageGenerationState = StageGenerationState.STARTED
    before_workspace_revision: int = Field(ge=1)
    after_workspace_revision: int | None = Field(default=None, ge=1)
    error_code: str | None = Field(default=None, max_length=200)
    error_message: str | None = Field(default=None, max_length=20_000)
    created_at: datetime
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> StageGenerationCheckpoint:
        if self.state == StageGenerationState.STARTED:
            if self.after_workspace_revision is not None or self.completed_at is not None:
                raise ValueError("started checkpoints cannot contain completion data")
        elif self.completed_at is None:
            raise ValueError("completed checkpoints require completed_at")
        if self.state == StageGenerationState.SUCCEEDED:
            if self.after_workspace_revision is None or self.error_code or self.error_message:
                raise ValueError("successful checkpoints require an output revision and no error")
        elif self.state == StageGenerationState.PARTIAL:
            if self.after_workspace_revision is None or not self.error_code:
                raise ValueError("partial checkpoints require an output revision and error code")
        elif self.state == StageGenerationState.FAILED:
            if not self.error_code:
                raise ValueError("failed checkpoints require error_code")
            if self.after_workspace_revision is not None:
                raise ValueError("failed checkpoints cannot commit an output revision")
        return self
