import hashlib
from datetime import UTC, datetime

import httpx
import pytest

from ai_video_generator.api import _asset_plan_resolution, create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BindingSemantic,
    BindingValueType,
    ComfyUIOutput,
    ProjectRunState,
    ReviewMode,
    ReviewPolicy,
    SegmentGenerationVersion,
    SegmentVersionState,
    TaskCheckpoint,
    TaskKind,
    TaskSpec,
    TaskState,
    TaskWorkloadManifest,
    WorkflowApproval,
    WorkflowBinding,
    WorkflowBindingDraft,
    WorkflowOutput,
    WorkflowOutputType,
    WorkflowTemplate,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.services.project_tasks import (
    ProjectTaskPlan,
    compile_delivery_task_plan,
    compile_project_task_plan,
    persist_project_task_plan,
)
from ai_video_generator.workers import inspect_api_workflow, workflow_template_from_inspection
from ai_video_generator.workers.workflow import canonical_json_sha256


def ready_h3_prompt(segment_id: str, continuation_of: str | None = None) -> dict[str, object]:
    return {
        "segmentId": segment_id,
        "continuationOf": continuation_of,
        "prompt": (
            "integrated_multimodal_description: 完整片段\n"
            "overall_soundscape: 环境声\nnon_diegetic_music: N/A"
        ),
        "review": {"ready": True},
    }


def test_asset_image_resolution_is_independent_from_video_resolution() -> None:
    assert _asset_plan_resolution({"kind": "character", "width": 1024, "height": 1600}) == (
        1024,
        1600,
    )
    assert _asset_plan_resolution({"kind": "scene"}) == (1600, 1024)


def test_delivery_plan_uses_active_video_inputs_without_compiling_h3() -> None:
    plan = compile_delivery_task_plan(
        project_id="project-1",
        payload={
            "fps": 24,
            "postProcessing": {
                "seedvr": {"enabled": False},
                "rife": {"enabled": False},
                "whisper": {"enabled": False},
            },
        },
        segment_outputs={"segment-1": "active-video-snapshot"},
    )

    assert plan.blockers == ()
    assert [task.kind for task in plan.tasks] == [TaskKind.MASTER_ASSEMBLY, TaskKind.EXPORT]
    assert plan.tasks[0].depends_on == ("active-video-snapshot",)


@pytest.mark.asyncio
async def test_delivery_source_snapshot_registers_active_video_checkpoint(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    payload = {
        "prompts": {
            "imagePrompts": [],
            "h3Prompts": [ready_h3_prompt("segment-1")],
        },
        "postProcessing": {
            "seedvr": {"enabled": False},
            "rife": {"enabled": False},
            "whisper": {"enabled": False},
        },
    }
    video_path = tmp_path / "videos" / "segment-1.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/projects",
            json={
                "project_id": "delivery-snapshot",
                "name": "Delivery snapshot",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        await client.post(
            "/api/v1/projects/delivery-snapshot/workspace",
            json={"revision": 1, "payload": payload},
        )

        store = SQLiteTaskStore(tmp_path / "control-plane.db")
        manifest = TaskWorkloadManifest(
            task_kind=TaskKind.H3_GENERATION,
            workflow_sha256="a" * 64,
            node_schema_sha256="b" * 64,
            prompt={"1": {"class_type": "SaveVideo", "inputs": {}}},
            outputs=(ComfyUIOutput(node_id="1", media_type="video/mp4"),),
            context={"project_id": "delivery-snapshot", "segment_id": "segment-1"},
        )
        manifest_record = store.put_workload_manifest(manifest)
        source = TaskSpec(
            task_id="source-h3",
            project_id="delivery-snapshot",
            kind=TaskKind.H3_GENERATION,
            state=TaskState.SUCCEEDED,
            idempotency_key="c" * 64,
            input_fingerprint="d" * 64,
            workload_manifest_sha256=manifest_record.sha256,
        )
        store.add_task(source)
        store.put_task_checkpoint(
            TaskCheckpoint(
                checkpoint_id="source-h3:video",
                task_id=source.task_id,
                sequence=1,
                phase="video_saved",
                payload={"path": str(video_path), "sha256": "e" * 64, "segment_id": "segment-1"},
                created_at=datetime.now(UTC),
            )
        )
        store.put_segment_generation_version(
            SegmentGenerationVersion(
                version_id="segment-version:1",
                project_id="delivery-snapshot",
                shot_id="shot-1",
                segment_id="segment-1",
                segment_index=0,
                generation_number=1,
                batch_id="batch-1",
                task_id=source.task_id,
                seed=1,
                state=SegmentVersionState.ACTIVE,
                artifact_id="source-h3:video",
                created_at=datetime.now(UTC),
            )
        )

        compiled = await client.post(
            "/api/v1/projects/delivery-snapshot/delivery/tasks/compile"
        )

    assert compiled.status_code == 201
    snapshot = next(
        task
        for task in store.list_tasks(project_id="delivery-snapshot")
        if task.task_id.startswith("delivery-source:delivery-snapshot:")
    )
    checkpoint = next(
        item
        for item in store.list_task_checkpoints(snapshot.task_id)
        if item.phase == "video_saved"
    )
    assert checkpoint.payload["path"] == str(video_path)


def test_project_task_plan_batches_conditioning_before_h3_and_export() -> None:
    state = ProjectRunState(
        project_id="project-1",
        review_policy=ReviewPolicy(
            configured_mode=ReviewMode.AI_ONLY,
            effective_mode=ReviewMode.AI_ONLY,
        ),
        updated_at=datetime.now(UTC),
    )
    plan = compile_project_task_plan(
        project_id="project-1",
        workspace_revision=4,
        payload={
            "prompts": {
                "imagePrompts": [],
                "h3Prompts": [ready_h3_prompt("s1"), ready_h3_prompt("s2", "s1")],
            },
            "postProcessing": {
                "seedvr": {"enabled": False},
                "rife": {"enabled": False},
                "whisper": {"enabled": False},
            },
        },
        run_state=state,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )
    assert plan.blockers == ()
    kinds = [task.kind for task in plan.tasks]
    assert kinds.count(TaskKind.CONDITIONING_ENCODING) == 2
    assert kinds.count(TaskKind.H3_GENERATION) == 2
    assert kinds.count(TaskKind.AI_REVIEW) == 2
    assert kinds[-1] == TaskKind.EXPORT
    switch = next(task for task in plan.tasks if task.kind == TaskKind.MODEL_SWITCH)
    assert len(switch.depends_on) == 2
    h3 = [task for task in plan.tasks if task.kind == TaskKind.H3_GENERATION]
    assert h3[0].task_id in h3[1].depends_on


@pytest.mark.parametrize("review", [None, {"ready": False}])
def test_project_task_plan_uses_final_h3_text_without_prompt_review_gate(
    review: dict[str, bool] | None,
) -> None:
    state = ProjectRunState(project_id="manual-prompt", updated_at=datetime.now(UTC))
    prompt: dict[str, object] = {
        "segmentId": "segment-1",
        "continuationOf": None,
        "prompt": "用户确认后的完整中文 MiniMax H3 视频提示词",
    }
    if review is not None:
        prompt["review"] = review

    plan = compile_project_task_plan(
        project_id="manual-prompt",
        workspace_revision=1,
        payload={"prompts": {"imagePrompts": [], "h3Prompts": [prompt]}},
        run_state=state,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )

    assert plan.blockers == ()
    h3_task = next(task for task in plan.tasks if task.kind == TaskKind.H3_GENERATION)
    assert h3_task.state == TaskState.BLOCKED


def test_prompt_review_provenance_does_not_change_h3_task_identity() -> None:
    state = ProjectRunState(project_id="prompt-identity", updated_at=datetime.now(UTC))

    def compile_with_review(ready: bool) -> ProjectTaskPlan:
        prompt = ready_h3_prompt("segment-1")
        prompt["review"] = {"ready": ready, "issues": [{"code": "audit-only"}]}
        return compile_project_task_plan(
            project_id=state.project_id,
            workspace_revision=1,
            payload={"prompts": {"imagePrompts": [], "h3Prompts": [prompt]}},
            run_state=state,
            approved_image_workflows=set(),
            approved_image_harnesses={},
        )

    reviewed = compile_with_review(True)
    manually_confirmed = compile_with_review(False)

    assert reviewed.fingerprint == manually_confirmed.fingerprint
    assert [task.task_id for task in reviewed.tasks] == [
        task.task_id for task in manually_confirmed.tasks
    ]


def test_human_review_timeout_takeover_keeps_compiled_dag_identity() -> None:
    payload = {
        "prompts": {
            "imagePrompts": [],
            "h3Prompts": [ready_h3_prompt("s1")],
        }
    }
    human = ProjectRunState(
        project_id="project-review-takeover",
        review_policy=ReviewPolicy(
            configured_mode=ReviewMode.HUMAN_AI,
            effective_mode=ReviewMode.HUMAN_AI,
        ),
        updated_at=datetime.now(UTC),
    )
    ai_takeover = human.model_copy(
        update={
            "review_policy": human.review_policy.model_copy(
                update={"effective_mode": ReviewMode.AI_ONLY}
            )
        }
    )

    before = compile_project_task_plan(
        project_id=human.project_id,
        workspace_revision=1,
        payload=payload,
        run_state=human,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )
    after = compile_project_task_plan(
        project_id=human.project_id,
        workspace_revision=1,
        payload=payload,
        run_state=ai_takeover,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )

    assert before.fingerprint == after.fingerprint
    assert [task.task_id for task in before.tasks] == [task.task_id for task in after.tasks]


def test_project_task_plan_rejects_unbound_image_prompt() -> None:
    state = ProjectRunState(project_id="p", updated_at=datetime.now(UTC))
    plan = compile_project_task_plan(
        project_id="p",
        workspace_revision=1,
        payload={
            "prompts": {
                "imagePrompts": [{"prompt": "portrait", "workflowTemplateId": None}],
                "h3Prompts": [ready_h3_prompt("s1")],
            }
        },
        run_state=state,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )
    assert "未绑定工作流" in "; ".join(plan.blockers)


def test_project_task_plan_accepts_approved_workflow_without_custom_harness() -> None:
    state = ProjectRunState(project_id="p", updated_at=datetime.now(UTC))
    plan = compile_project_task_plan(
        project_id="p",
        workspace_revision=1,
        payload={
            "assetPlans": [{"id": "hero", "fulfilledByAssetId": None}],
            "prompts": {
                "imagePrompts": [
                    {
                        "assetPlanId": "hero",
                        "prompt": "电影感主角全身角色设定图",
                        "workflowTemplateId": "user:image-workflow",
                    }
                ],
                "h3Prompts": [ready_h3_prompt("s1")],
            },
        },
        run_state=state,
        approved_image_workflows={"user:image-workflow"},
        approved_image_harnesses={},
    )

    assert plan.blockers == ()
    assert any(task.kind == TaskKind.IMAGE_GENERATION for task in plan.tasks)


def test_project_task_plan_skips_already_fulfilled_image_requirement() -> None:
    state = ProjectRunState(project_id="p", updated_at=datetime.now(UTC))
    plan = compile_project_task_plan(
        project_id="p",
        workspace_revision=2,
        payload={
            "assetPlans": [{"id": "hero", "fulfilledByAssetId": "asset-1"}],
            "prompts": {
                "imagePrompts": [
                    {
                        "assetPlanId": "hero",
                        "prompt": "old prompt",
                        "workflowTemplateId": "removed-template",
                    }
                ],
                "h3Prompts": [ready_h3_prompt("s1")],
            },
        },
        run_state=state,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )
    assert not plan.blockers
    assert all(task.kind != TaskKind.IMAGE_GENERATION for task in plan.tasks)


def test_project_task_plan_skips_all_images_in_no_reference_mode() -> None:
    state = ProjectRunState(project_id="p", updated_at=datetime.now(UTC))
    plan = compile_project_task_plan(
        project_id="p",
        workspace_revision=3,
        payload={
            "referenceAssetMode": "none",
            "assetPlans": [{"id": "unused", "fulfilledByAssetId": None}],
            "prompts": {
                "imagePrompts": [
                    {
                        "assetPlanId": "unused",
                        "prompt": "must not run",
                        "workflowTemplateId": "missing-workflow",
                    }
                ],
                "h3Prompts": [ready_h3_prompt("s1")],
            },
        },
        run_state=state,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )

    assert plan.blockers == ()
    assert all(task.kind != TaskKind.IMAGE_GENERATION for task in plan.tasks)
    conditioning = next(task for task in plan.tasks if task.kind == TaskKind.CONDITIONING_ENCODING)
    assert conditioning.depends_on == ()


def test_none_review_mode_still_creates_deterministic_media_check() -> None:
    state = ProjectRunState(
        project_id="project-none",
        review_policy=ReviewPolicy(
            configured_mode=ReviewMode.NONE,
            effective_mode=ReviewMode.NONE,
        ),
        updated_at=datetime.now(UTC),
    )
    plan = compile_project_task_plan(
        project_id="project-none",
        workspace_revision=1,
        payload={
            "referenceAssetMode": "none",
            "prompts": {
                "imagePrompts": [],
                "h3Prompts": [ready_h3_prompt("segment-1")],
            },
        },
        run_state=state,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )

    reviews = [task for task in plan.tasks if task.kind == TaskKind.AI_REVIEW]
    exports = [task for task in plan.tasks if task.kind == TaskKind.EXPORT]
    assert not reviews
    master = next(task for task in plan.tasks if task.kind == TaskKind.MASTER_ASSEMBLY)
    assert master.depends_on
    assert exports[0].depends_on == (master.task_id,)


def _task_ids_by_kind(plan: object) -> dict[TaskKind, tuple[str, ...]]:
    return {
        kind: tuple(task.task_id for task in plan.tasks if task.kind == kind) for kind in TaskKind
    }


def _postprocessing_plan(post_processing: dict[str, object], *, prompt_text: str = "完整片段"):
    state = ProjectRunState(
        project_id="post-project",
        review_policy=ReviewPolicy(
            configured_mode=ReviewMode.AI_ONLY,
            effective_mode=ReviewMode.AI_ONLY,
        ),
        updated_at=datetime.now(UTC),
    )
    prompt = ready_h3_prompt("segment-1")
    prompt["prompt"] = (
        f"integrated_multimodal_description: {prompt_text}\n"
        "overall_soundscape: 环境声\nnon_diegetic_music: N/A"
    )
    return compile_project_task_plan(
        project_id=state.project_id,
        workspace_revision=1,
        payload={
            "width": 320,
            "height": 480,
            "referenceAssetMode": "none",
            "prompts": {"imagePrompts": [], "h3Prompts": [prompt]},
            "postProcessing": post_processing,
        },
        run_state=state,
        approved_image_workflows=set(),
        approved_image_harnesses={},
    )


def test_enabling_postprocessing_reuses_completed_generation_and_review(tmp_path) -> None:
    disabled = {
        "outputWidth": 640,
        "outputHeight": 960,
        "crf": 18,
        "seedvr": {"enabled": False},
        "rife": {"enabled": False, "targetFps": 48},
        "whisper": {"enabled": False},
    }
    before = _postprocessing_plan(disabled)
    enabled = {
        **disabled,
        "rife": {
            "enabled": True,
            "workflowTemplateId": "user:interpolation",
            "workflowRevision": 1,
            "modelId": "rife49.pth",
            "targetFps": 48,
        },
    }
    after = _postprocessing_plan(enabled)
    before_ids = _task_ids_by_kind(before)
    after_ids = _task_ids_by_kind(after)

    upstream = (
        TaskKind.CONDITIONING_ENCODING,
        TaskKind.MODEL_SWITCH,
        TaskKind.H3_GENERATION,
        TaskKind.AI_REVIEW,
    )
    for kind in upstream:
        assert before_ids[kind] == after_ids[kind]
    assert before_ids[TaskKind.EXPORT] != after_ids[TaskKind.EXPORT]
    assert after_ids[TaskKind.RIFE]

    store = SQLiteTaskStore(tmp_path / "tasks.db")
    persist_project_task_plan(store, before)
    for task in before.tasks:
        if task.kind == TaskKind.EXPORT:
            continue
        current = store.get_task(task.task_id)
        if current.state == TaskState.BLOCKED:
            current = store.get_task(task.task_id)
        store.transition_task(current.task_id, TaskState.QUEUED)
        store.transition_task(current.task_id, TaskState.RUNNING)
        store.transition_task(current.task_id, TaskState.SUCCEEDED)
    persist_project_task_plan(store, after)
    for kind in upstream:
        for task_id in after_ids[kind]:
            assert store.get_task(task_id).state == TaskState.SUCCEEDED


def test_rife_settings_only_replace_rife_and_export() -> None:
    base = {
        "outputWidth": 640,
        "outputHeight": 960,
        "crf": 18,
        "seedvr": {"enabled": False},
        "rife": {
            "enabled": True,
            "workflowTemplateId": "user:interpolation",
            "workflowRevision": 1,
            "modelId": "rife49.pth",
            "targetFps": 48,
        },
        "whisper": {
            "enabled": True,
            "workflowTemplateId": "user:transcription",
            "workflowRevision": 1,
            "profileId": "user:transcription",
            "profileRevision": 1,
            "modelId": "large-v3-turbo",
            "language": "zh",
        },
    }
    before = _postprocessing_plan(base)
    after = _postprocessing_plan({**base, "rife": {**base["rife"], "targetFps": 60}})
    before_ids = _task_ids_by_kind(before)
    after_ids = _task_ids_by_kind(after)

    assert before_ids[TaskKind.SEEDVR2] == after_ids[TaskKind.SEEDVR2]
    assert before_ids[TaskKind.WHISPER] != after_ids[TaskKind.WHISPER]
    assert before_ids[TaskKind.RIFE] != after_ids[TaskKind.RIFE]
    assert before_ids[TaskKind.EXPORT] != after_ids[TaskKind.EXPORT]


def test_export_settings_only_replace_export() -> None:
    base = {
        "outputWidth": 640,
        "outputHeight": 960,
        "crf": 18,
        "seedvr": {"enabled": False},
        "rife": {
            "enabled": True,
            "workflowTemplateId": "user:interpolation",
            "workflowRevision": 1,
            "modelId": "rife49.pth",
            "targetFps": 48,
        },
        "whisper": {"enabled": False},
    }
    before = _postprocessing_plan(base)
    after = _postprocessing_plan({**base, "crf": 22})
    before_ids = _task_ids_by_kind(before)
    after_ids = _task_ids_by_kind(after)

    for kind in TaskKind:
        if kind == TaskKind.EXPORT:
            assert before_ids[kind] != after_ids[kind]
        else:
            assert before_ids[kind] == after_ids[kind]


def test_h3_prompt_change_replaces_generation_and_all_downstream_tasks() -> None:
    post = {
        "seedvr": {"enabled": False},
        "rife": {
            "enabled": True,
            "workflowTemplateId": "user:interpolation",
            "workflowRevision": 1,
            "modelId": "rife49.pth",
            "targetFps": 48,
        },
        "whisper": {"enabled": False},
    }
    before = _postprocessing_plan(post, prompt_text="人物向左走")
    after = _postprocessing_plan(post, prompt_text="人物向右走")
    before_ids = _task_ids_by_kind(before)
    after_ids = _task_ids_by_kind(after)

    for kind in (
        TaskKind.CONDITIONING_ENCODING,
        TaskKind.MODEL_SWITCH,
        TaskKind.H3_GENERATION,
        TaskKind.AI_REVIEW,
        TaskKind.RIFE,
        TaskKind.MASTER_ASSEMBLY,
        TaskKind.EXPORT,
    ):
        assert before_ids[kind] != after_ids[kind]


def test_h3_execution_profile_restart_replaces_video_chain_but_reuses_images() -> None:
    state = ProjectRunState(project_id="h3-profile", updated_at=datetime.now(UTC))
    payload = {
        "assetPlans": [{"id": "hero", "fulfilledByAssetId": None}],
        "prompts": {
            "imagePrompts": [
                {
                    "assetPlanId": "hero",
                    "prompt": "角色设定图",
                    "workflowTemplateId": "image:user",
                }
            ],
            "h3Prompts": [ready_h3_prompt("segment-1")],
        },
    }

    def compile_with_profile(generation_revision: int, model: str) -> ProjectTaskPlan:
        return compile_project_task_plan(
            project_id=state.project_id,
            workspace_revision=1,
            payload=payload,
            run_state=state,
            approved_image_workflows={"image:user"},
            approved_image_harnesses={},
            h3_execution_profile={
                "generation_revision": generation_revision,
                "diffusion_model": model,
                "turbo_enabled": False,
                "steps": 12,
            },
        )

    before = _task_ids_by_kind(compile_with_profile(0, "h3-a.safetensors"))
    after = _task_ids_by_kind(compile_with_profile(1, "h3-b.safetensors"))

    assert before[TaskKind.IMAGE_GENERATION] == after[TaskKind.IMAGE_GENERATION]
    for kind in (
        TaskKind.CONDITIONING_ENCODING,
        TaskKind.MODEL_SWITCH,
        TaskKind.H3_GENERATION,
        TaskKind.AI_REVIEW,
        TaskKind.MASTER_ASSEMBLY,
        TaskKind.EXPORT,
    ):
        assert before[kind] != after[kind]


def test_post_recompile_can_adopt_matching_legacy_upstream_branch() -> None:
    disabled = {
        "seedvr": {"enabled": False},
        "rife": {"enabled": False},
        "whisper": {"enabled": False},
    }
    before = _postprocessing_plan(disabled)
    labels = {
        TaskKind.CONDITIONING_ENCODING: "conditioning:segment-1",
        TaskKind.MODEL_SWITCH: "conditioning-to-h3-diffusion",
        TaskKind.H3_GENERATION: "h3:segment-1",
        TaskKind.AI_REVIEW: "review:segment-1",
    }
    id_map: dict[str, str] = {}
    reusable: dict[str, TaskSpec] = {}
    for task in before.tasks:
        label = labels.get(task.kind)
        if label is None:
            continue
        label_identity = hashlib.sha256(f"{task.kind.value}:{label}".encode()).hexdigest()[:16]
        legacy_id = (
            f"run:{task.project_id}:{before.fingerprint[:16]}:{task.kind.value}:{label_identity}"
        )
        id_map[task.task_id] = legacy_id
        legacy = task.model_copy(
            update={
                "task_id": legacy_id,
                "depends_on": tuple(id_map[value] for value in task.depends_on),
                "state": TaskState.SUCCEEDED,
            }
        )
        reusable[f"{task.kind.value}:{label_identity}"] = legacy

    after = compile_project_task_plan(
        project_id="post-project",
        workspace_revision=2,
        payload={
            "width": 320,
            "height": 480,
            "referenceAssetMode": "none",
            "prompts": {
                "imagePrompts": [],
                "h3Prompts": [ready_h3_prompt("segment-1")],
            },
            "postProcessing": {
                **disabled,
                "rife": {
                    "enabled": True,
                    "workflowTemplateId": "user:interpolation",
                    "workflowRevision": 1,
                    "modelId": "rife49.pth",
                    "targetFps": 48,
                },
            },
        },
        run_state=ProjectRunState(project_id="post-project", updated_at=datetime.now(UTC)),
        approved_image_workflows=set(),
        approved_image_harnesses={},
        reusable_tasks=reusable,
    )
    for task in after.tasks:
        if task.kind in labels:
            original_id = next(value.task_id for value in before.tasks if value.kind == task.kind)
            assert task.task_id == id_map[original_id]
            assert task.state == TaskState.SUCCEEDED
    rife = next(task for task in after.tasks if task.kind == TaskKind.RIFE)
    review = next(task for task in after.tasks if task.kind == TaskKind.AI_REVIEW)
    assert rife.depends_on != (review.task_id,)


@pytest.mark.asyncio
async def test_compile_endpoint_preserves_successful_upstream_for_post_only_change(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_root=tmp_path,
        comfyui_base_url="http://comfy-must-not-be-contacted",
    )
    app = create_app(settings)
    disabled = {
        "outputWidth": 640,
        "outputHeight": 960,
        "crf": 18,
        "seedvr": {"enabled": False},
        "rife": {"enabled": False, "targetFps": 48},
        "whisper": {"enabled": False},
    }
    payload = {
        "width": 320,
        "height": 480,
        "referenceAssetMode": "none",
        "prompts": {
            "imagePrompts": [],
            "h3Prompts": [ready_h3_prompt("segment-1")],
        },
        "postProcessing": disabled,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/projects",
            json={
                "project_id": "post-project",
                "name": "Post reuse",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        await client.post(
            "/api/v1/projects/post-project/workspace",
            json={"revision": 1, "payload": payload},
        )
        store = SQLiteTaskStore(tmp_path / "control-plane.db")
        before = compile_project_task_plan(
            project_id="post-project",
            workspace_revision=1,
            payload=payload,
            run_state=store.get_project_run_state("post-project"),
            approved_image_workflows=set(),
            approved_image_harnesses={},
        )
        tasks = []
        for planned in before.tasks:
            task = planned
            if planned.kind in {
                TaskKind.CONDITIONING_ENCODING,
                TaskKind.H3_GENERATION,
            }:
                output_id = "output"
                manifest = TaskWorkloadManifest(
                    task_kind=planned.kind,
                    workflow_sha256="a" * 64,
                    node_schema_sha256="b" * 64,
                    prompt={
                        output_id: {
                            "class_type": "TestOutput",
                            "inputs": {"value": planned.kind.value},
                        }
                    },
                    outputs=(ComfyUIOutput(node_id=output_id, media_type="video/mp4"),),
                    context={"project_id": "post-project", "segment_id": "segment-1"},
                )
                record = store.put_workload_manifest(manifest)
                task = task.model_copy(update={"workload_manifest_sha256": record.sha256})
            tasks.append(task.model_copy(update={"state": TaskState.SUCCEEDED}))
        for task in tasks:
            store.add_task(task)
        transcription_raw = {
            "1": {"class_type": "LoadVideo", "inputs": {"video": "input.mp4"}},
            "2": {"class_type": "SaveSubtitle", "inputs": {"video": ["1", 0]}},
        }
        store.put_workflow_revision(
            WorkflowTemplate(
                template_id="user:transcription",
                revision=1,
                name="Transcription",
                kind="transcription",
                workflow_sha256=canonical_json_sha256(transcription_raw),
                node_schema_sha256="c" * 64,
                raw_workflow=transcription_raw,
                bindings=(
                    WorkflowBinding(
                        binding_id="source",
                        semantic=BindingSemantic.SOURCE_VIDEO,
                        node_id="1",
                        input_name="video",
                        value_type=BindingValueType.VIDEO_PATH,
                        title="Source",
                    ),
                ),
                outputs=(
                    WorkflowOutput(
                        output_id="subtitle:2",
                        node_id="2",
                        output_type=WorkflowOutputType.SUBTITLE,
                        title="Subtitle",
                    ),
                ),
                required_node_types=("LoadVideo", "SaveSubtitle"),
                approval=WorkflowApproval.APPROVED,
            )
        )

        await client.post(
            "/api/v1/projects/post-project/revisions",
            json={
                "project_id": "post-project",
                "revision": 2,
                "name": "Post reuse",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        await client.post(
            "/api/v1/projects/post-project/workspace",
            json={
                "revision": 2,
                "payload": {
                    **payload,
                    "postProcessing": {
                        **disabled,
                        "whisper": {
                            "enabled": True,
                            "workflowTemplateId": "user:transcription",
                            "workflowRevision": 1,
                            "profileId": "user:transcription",
                            "profileRevision": 1,
                            "modelId": "small",
                            "language": "zh",
                        },
                    },
                },
            },
        )
        compiled = await client.post("/api/v1/projects/post-project/tasks/compile")

    assert compiled.status_code == 201, compiled.text
    compiled_tasks = compiled.json()["tasks"]
    compiled_ids = {task["task_id"] for task in compiled_tasks}
    old_upstream = [task for task in tasks if task.kind != TaskKind.EXPORT]
    assert all(task.task_id in compiled_ids for task in old_upstream)
    assert all(store.get_task(task.task_id).state == TaskState.SUCCEEDED for task in old_upstream)
    old_export = next(task for task in tasks if task.kind == TaskKind.EXPORT)
    assert store.get_task(old_export.task_id).state == TaskState.STALE
    whisper = next(task for task in compiled_tasks if task["kind"] == TaskKind.WHISPER.value)
    assert whisper["state"] == TaskState.READY.value


@pytest.mark.asyncio
async def test_execution_status_marks_plan_stale_when_postprocessing_changes(
    tmp_path,
) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    disabled = {
        "seedvr": {"enabled": False},
        "rife": {"enabled": False},
        "whisper": {"enabled": False},
    }
    payload = {
        "width": 320,
        "height": 480,
        "referenceAssetMode": "none",
        "prompts": {
            "imagePrompts": [],
            "h3Prompts": [ready_h3_prompt("segment-1")],
        },
        "postProcessing": disabled,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/projects",
            json={
                "project_id": "post-freshness",
                "name": "Post freshness",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        await client.post(
            "/api/v1/projects/post-freshness/workspace",
            json={"revision": 1, "payload": payload},
        )
        store = SQLiteTaskStore(tmp_path / "control-plane.db")
        plan = compile_project_task_plan(
            project_id="post-freshness",
            workspace_revision=1,
            payload=payload,
            run_state=store.get_project_run_state("post-freshness"),
            approved_image_workflows=set(),
            approved_image_harnesses={},
        )
        for task in plan.tasks:
            store.add_task(task)
        current = await client.get(
            "/api/v1/projects/post-freshness/execution-status"
        )
        await client.post(
            "/api/v1/projects/post-freshness/revisions",
            json={
                "project_id": "post-freshness",
                "revision": 2,
                "name": "Post freshness",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        await client.post(
            "/api/v1/projects/post-freshness/workspace",
            json={
                "revision": 2,
                "payload": {
                    **payload,
                    "postProcessing": {
                        **disabled,
                        "seedvr": {
                            "enabled": True,
                            "workflowTemplateId": "user:restoration",
                            "workflowRevision": 1,
                        },
                    },
                },
            },
        )
        stale = await client.get(
            "/api/v1/projects/post-freshness/execution-status"
        )

    assert current.json()["compiled"] is True
    assert stale.json()["compiled"] is False


@pytest.mark.asyncio
async def test_restart_h3_compile_creates_new_video_chain_and_stales_old_tasks(
    tmp_path,
) -> None:
    def comfy_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    app = create_app(
        Settings(_env_file=None, data_root=tmp_path, comfyui_base_url="http://comfy"),
        comfyui_transport=httpx.MockTransport(comfy_handler),
    )
    h3_prompt = ready_h3_prompt("segment-1")
    h3_prompt.update(
        {
            "durationSeconds": 4,
            "inputMode": "t2va",
            "assetIds": [],
            "seed": 42,
        }
    )
    payload = {
        "width": 320,
        "height": 480,
        "referenceAssetMode": "none",
        "prompts": {
            "imagePrompts": [],
            "h3Prompts": [h3_prompt],
        },
        "postProcessing": {
            "seedvr": {"enabled": False},
            "rife": {"enabled": False},
            "whisper": {"enabled": False},
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/projects",
            json={
                "project_id": "restart-h3",
                "name": "Restart H3",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        await client.post(
            "/api/v1/projects/restart-h3/workspace",
            json={"revision": 1, "payload": payload},
        )
        first = await client.post("/api/v1/projects/restart-h3/tasks/compile")
        first_conditioning = next(
            task
            for task in first.json()["tasks"]
            if task["kind"] == "conditioning_encoding"
        )
        queued = await client.post(f"/api/v1/tasks/{first_conditioning['task_id']}/run")
        restarted = await client.post(
            "/api/v1/projects/restart-h3/tasks/compile?restart_h3=true"
        )
        run_state = await client.get("/api/v1/projects/restart-h3/run-state")

    assert first.status_code == 201, first.text
    assert queued.json()["state"] == TaskState.QUEUED.value
    assert restarted.status_code == 201, restarted.text
    assert run_state.json()["generation_revision"] == 1
    first_h3 = next(task for task in first.json()["tasks"] if task["kind"] == "h3_generation")
    restarted_h3 = next(
        task for task in restarted.json()["tasks"] if task["kind"] == "h3_generation"
    )
    assert first_h3["task_id"] != restarted_h3["task_id"]
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    assert store.get_task(first_conditioning["task_id"]).state == TaskState.CANCELLED
    assert store.get_task(first_h3["task_id"]).state == TaskState.STALE
    assert store.get_task(restarted_h3["task_id"]).state == TaskState.BLOCKED


@pytest.mark.asyncio
async def test_release_does_not_register_machine_specific_image_workflow(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        templates = await client.get("/api/v1/workflows/templates")
    assert templates.status_code == 200
    assert templates.json() == []


@pytest.mark.asyncio
async def test_project_compile_endpoint_persists_dag_and_image_manifest(tmp_path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    workspace = {
        "width": 320,
        "height": 480,
        "prompts": {
            "imagePrompts": [
                {
                    "assetPlanId": "hero",
                    "prompt": "黄色卡通龙角色设定图",
                    "negativePrompt": "文字水印",
                    "workflowTemplateId": "builtin:z-image-turbo",
                }
            ],
            "h3Prompts": [ready_h3_prompt("segment-1")],
        },
        "postProcessing": {
            "seedvr": {"enabled": False},
            "rife": {"enabled": False},
            "whisper": {"enabled": False},
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/api/v1/projects",
            json={
                "project_id": "project-compile",
                "name": "Compile",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        saved = await client.post(
            "/api/v1/projects/project-compile/workspace",
            json={"revision": 1, "payload": workspace},
        )
        compiled = await client.post("/api/v1/projects/project-compile/tasks/compile")
        status = await client.get("/api/v1/projects/project-compile/execution-status")
        next_project = await client.post(
            "/api/v1/projects/project-compile/revisions",
            json={
                "project_id": "project-compile",
                "revision": 2,
                "name": "Compile",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        next_workspace = await client.post(
            "/api/v1/projects/project-compile/workspace",
            json={
                "revision": 2,
                "payload": {
                    **workspace,
                    "activeStage": "review",
                    "stageApprovals": {"generation": "now"},
                },
            },
        )
        status_after_stage_change = await client.get(
            "/api/v1/projects/project-compile/execution-status"
        )
    assert created.status_code == 201
    assert saved.status_code == 201
    assert compiled.status_code == 409
    assert "未批准" in str(compiled.json())
    assert status.json()["compiled"] is False
    assert next_project.status_code == 201
    assert next_workspace.status_code == 201
    assert status_after_stage_change.json()["compiled"] is False


@pytest.mark.asyncio
async def test_single_image_run_compiles_without_any_h3_prompts(tmp_path) -> None:
    object_info = {
        "CLIPTextEncode": {
            "input": {"required": {"text": ["STRING"]}},
            "output": ["CONDITIONING"],
        },
        "SaveImage": {
            "input": {"required": {"images": ["CONDITIONING"]}},
            "output": [],
            "output_node": True,
        },
    }
    workflow = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "default"},
            "_meta": {"title": "正面提示词"},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0]},
            "_meta": {"title": "保存图片"},
        },
    }
    inspection = inspect_api_workflow(workflow, object_info).model_copy(
        update={
            "bindings": (
                WorkflowBindingDraft(
                    binding_id="prompt",
                    semantic=BindingSemantic.PROMPT,
                    node_id="1",
                    input_name="text",
                    value_type=BindingValueType.STRING,
                    title="正面提示词",
                    default_value="default",
                ),
            )
        }
    )
    template = workflow_template_from_inspection(
        inspection,
        template_id="image:test",
        name="Image test",
        approval=WorkflowApproval.APPROVED,
    )

    def comfy_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=object_info)
        return httpx.Response(404)

    app = create_app(
        Settings(_env_file=None, data_root=tmp_path, comfyui_base_url="http://comfy"),
        comfyui_transport=httpx.MockTransport(comfy_handler),
    )
    workspace = {
        "width": 320,
        "height": 480,
        "assetPlans": [
            {
                "id": "set",
                "name": "雨夜街道",
                "description": "湿润路面与霓虹灯",
                "kind": "scene",
                "scope": "public",
                "shotId": None,
                "fulfilledByAssetId": None,
                "state": "ready",
            }
        ],
        "prompts": {
            "imagePrompts": [
                {
                    "assetPlanId": "set",
                    "prompt": "雨夜街道，湿润路面反射霓虹灯，电影感广角构图",
                    "negativePrompt": "文字水印，低清晰度",
                    "workflowTemplateId": "image:test",
                    "revision": 1,
                }
            ],
            "h3Prompts": [],
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        registered = await client.post(
            "/api/v1/workflows/templates",
            json=template.model_dump(mode="json"),
        )
        await client.post(
            "/api/v1/projects",
            json={
                "project_id": "image-first",
                "name": "Image first",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 4,
            },
        )
        await client.post(
            "/api/v1/projects/image-first/workspace",
            json={"revision": 1, "payload": workspace},
        )
        response = await client.post("/api/v1/projects/image-first/image-prompts/set/run")

    assert registered.status_code == 201
    assert response.status_code == 202
    assert response.json()["kind"] == "image_generation"
    assert response.json()["workload_manifest_sha256"]
