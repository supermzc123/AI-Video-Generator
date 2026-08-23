from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ai_video_generator.domain import (
    AssetMentionNode,
    AssetPlan,
    AssetPlanState,
    AssetScope,
    GenerationMode,
    H3PromptRevision,
    ImagePromptRevision,
    ProjectAsset,
    ProjectAssetPurpose,
    ProjectAssetState,
    PromptIssueSeverity,
    PromptReviewIssue,
    PromptReviewResult,
    PromptRevisionState,
    PromptSet,
    RichTextDocument,
    StageGenerationCheckpoint,
    StageGenerationState,
    TextNode,
)
from ai_video_generator.persistence import SQLiteTaskStore, StoreConflictError

NOW = datetime(2026, 8, 15, tzinfo=UTC)


def project_asset(
    asset_id: str,
    name: str,
    *,
    revision: int = 1,
    state: ProjectAssetState = ProjectAssetState.AVAILABLE,
) -> ProjectAsset:
    available = state == ProjectAssetState.AVAILABLE
    return ProjectAsset(
        asset_id=asset_id,
        project_id="project-1",
        revision=revision,
        name=name,
        original_name=f"{name}.png",
        state=state,
        sha256="a" * 64 if available else None,
        mime_type="image/png" if available else None,
        byte_size=100 if available else None,
        width=64 if available else None,
        height=96 if available else None,
        kind=ProjectAssetPurpose.CHARACTER,
        created_at=NOW + timedelta(seconds=revision),
    )


def passing_review(*, with_error: bool = False) -> PromptReviewResult:
    issues = (
        PromptReviewIssue(
            code="timeline_gap",
            severity=PromptIssueSeverity.ERROR,
            message="时间线未完整覆盖。",
        ),
    ) if with_error else ()
    return PromptReviewResult(
        reviewer_harness_id="h3:reviewer",
        reviewer_harness_revision=2,
        issues=issues,
        structure_complete=True,
        references_complete=True,
        timeline_complete=True,
        contradictions_absent=True,
        audio_consistent=True,
        within_length_limit=True,
        reviewed_at=NOW,
    )


def test_rich_text_mentions_keep_stable_asset_ids() -> None:
    document = RichTextDocument(
        nodes=(
            TextNode(text="主角 "),
            AssetMentionNode(asset_id="asset-1", display_name="林夏"),
            TextNode(text=" 走进车站，随后再次看向 "),
            AssetMentionNode(asset_id="asset-1", display_name="林夏"),
        )
    )
    renamed = document.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"display_name": "女主角"})
                if isinstance(node, AssetMentionNode)
                else node
                for node in document.nodes
            )
        }
    )

    assert document.referenced_asset_ids == ("asset-1",)
    assert renamed.referenced_asset_ids == ("asset-1",)


def test_project_asset_revisions_are_immutable_and_names_are_unique(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "assets.db")
    first = project_asset("asset-1", "Hero")
    assert store.put_project_asset(first) == first
    assert store.put_project_asset(first) == first

    with pytest.raises(StoreConflictError, match="name is already in use"):
        store.put_project_asset(project_asset("asset-2", "hero"))
    with pytest.raises(StoreConflictError, match="immutable"):
        store.put_project_asset(first.model_copy(update={"name": "Changed"}))

    renamed = project_asset("asset-1", "Lead", revision=2)
    store.put_project_asset(renamed)
    second = project_asset("asset-2", "Hero")
    store.put_project_asset(second)
    assert store.get_project_asset("asset-1") == renamed
    assert store.list_project_assets("project-1") == (second, renamed)
    assert store.list_project_assets("project-1", include_history=True) == (
        first,
        renamed,
        second,
    )


def test_legacy_missing_blob_asset_is_explicit_and_not_executable() -> None:
    missing = project_asset(
        "legacy-1", "旧参考图", state=ProjectAssetState.MISSING_BLOB
    )
    assert missing.sha256 is None
    with pytest.raises(ValidationError, match="require blob and image metadata"):
        ProjectAsset.model_validate(
            {**project_asset("broken", "Broken").model_dump(), "sha256": None}
        )


def test_prompt_revisions_and_sets_round_trip_without_overwrite(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "prompts.db")
    plan = AssetPlan(
        plan_id="plan-1",
        project_id="project-1",
        revision=1,
        name="主角参考图",
        description=RichTextDocument.from_plain_text("保持主角服装一致"),
        purpose=ProjectAssetPurpose.CHARACTER,
        state=AssetPlanState.READY,
        created_at=NOW,
    )
    image_prompt = ImagePromptRevision(
        prompt_revision_id="image-prompt-1",
        project_id="project-1",
        asset_plan_id=plan.plan_id,
        workflow_template_id="z-image",
        workflow_revision=1,
        harness_id="image:z-image",
        harness_revision=1,
        prompt=RichTextDocument.from_plain_text("电影质感人物定妆照"),
        created_at=NOW,
    )
    h3_prompt = H3PromptRevision(
        prompt_revision_id="h3-prompt-1",
        project_id="project-1",
        segment_id="segment-1",
        generation_mode=GenerationMode.REF2VA,
        harness_id="h3:default",
        harness_revision=2,
        reference_asset_ids=("asset-1",),
        execution_prompt_zh=(
            "subject_definitions: [角色1]保持参考图身份与服装。\n"
            "summary: 角色走进车站。\nretention_analysis: 保留身份，允许姿态变化。\n"
            "detailed_description: 0-4秒，中景跟拍角色走入车站，环境声连续。\n"
            "overall_soundscape: 脚步与车站广播。\nnon_diegetic_music: 无。"
        ),
        review=passing_review(),
        state=PromptRevisionState.APPROVED,
        created_at=NOW,
    )
    prompt_set = PromptSet(
        prompt_set_id="prompt-set-1",
        project_id="project-1",
        asset_plan_ids=(plan.plan_id,),
        image_prompt_revision_ids=(image_prompt.prompt_revision_id,),
        h3_prompt_revision_ids=(h3_prompt.prompt_revision_id,),
        created_at=NOW,
    )

    assert store.put_asset_plan(plan) == plan
    assert store.put_image_prompt_revision(image_prompt) == image_prompt
    assert store.put_h3_prompt_revision(h3_prompt) == h3_prompt
    assert store.put_prompt_set(prompt_set) == prompt_set
    assert store.list_asset_plans("project-1") == (plan,)
    assert store.get_asset_plan(plan.plan_id) == plan
    assert store.list_image_prompt_revisions("project-1") == (image_prompt,)
    assert store.get_image_prompt_revision(image_prompt.prompt_revision_id) == image_prompt
    assert store.list_h3_prompt_revisions("project-1", segment_id="segment-1") == (
        h3_prompt,
    )
    assert store.get_h3_prompt_revision(h3_prompt.prompt_revision_id) == h3_prompt
    assert store.list_prompt_sets("project-1") == (prompt_set,)
    assert store.get_prompt_set(prompt_set.prompt_set_id) == prompt_set

    with pytest.raises(StoreConflictError, match="immutable"):
        store.put_h3_prompt_revision(
            h3_prompt.model_copy(update={"execution_prompt_zh": "被覆盖"})
        )


def test_approved_h3_prompt_requires_passing_review() -> None:
    with pytest.raises(ValidationError, match="must pass"):
        H3PromptRevision(
            prompt_revision_id="h3-prompt-1",
            project_id="project-1",
            segment_id="segment-1",
            generation_mode=GenerationMode.T2VA,
            harness_id="h3:default",
            harness_revision=2,
            execution_prompt_zh="integrated_multimodal_description: 完整描述",
            review=passing_review(with_error=True),
            state=PromptRevisionState.APPROVED,
            created_at=NOW,
        )


def test_stage_checkpoint_is_idempotent_and_completes_once(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "checkpoint.db")
    started = StageGenerationCheckpoint(
        checkpoint_id="checkpoint-1",
        project_id="project-1",
        stage="prompts",
        idempotency_key="c" * 64,
        before_workspace_revision=3,
        created_at=NOW,
    )
    assert store.put_stage_generation_checkpoint(started) == started
    assert store.put_stage_generation_checkpoint(started) == started

    succeeded = started.model_copy(
        update={
            "state": StageGenerationState.SUCCEEDED,
            "after_workspace_revision": 4,
            "completed_at": NOW + timedelta(seconds=5),
        }
    )
    assert store.put_stage_generation_checkpoint(succeeded) == succeeded
    assert store.get_stage_generation_checkpoint("project-1", "prompts", "c" * 64) == succeeded
    assert store.list_stage_generation_checkpoints("project-1", stage="prompts") == (
        succeeded,
    )

    with pytest.raises(StoreConflictError, match="immutable"):
        store.put_stage_generation_checkpoint(
            succeeded.model_copy(update={"after_workspace_revision": 5})
        )


def test_project_asset_scope_requires_matching_owner() -> None:
    with pytest.raises(ValidationError, match="shot-scoped"):
        ProjectAsset.model_validate(
            {
                **project_asset("asset-1", "Hero").model_dump(),
                "scope": AssetScope.SHOT,
            }
        )
