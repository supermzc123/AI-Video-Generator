from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ai_video_generator.domain import (
    ProjectRunState,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore


@dataclass(frozen=True)
class ProjectTaskPlan:
    fingerprint: str
    tasks: tuple[TaskSpec, ...]
    blockers: tuple[str, ...] = ()


def compile_project_task_plan(
    *,
    project_id: str,
    workspace_revision: int,
    payload: dict[str, Any],
    run_state: ProjectRunState,
    approved_image_workflows: set[str],
    approved_image_harnesses: dict[str, int],
    reusable_tasks: dict[str, TaskSpec] | None = None,
) -> ProjectTaskPlan:
    """Compile the approved prompt set into a deterministic persisted DAG.

    The plan is intentionally independent from a live Worker. A project may be
    prepared while ComfyUI is offline, but dispatch remains a separate action.
    """
    prompts = payload.get("prompts") if isinstance(payload.get("prompts"), dict) else {}
    image_prompts = (
        prompts.get("imagePrompts") if isinstance(prompts.get("imagePrompts"), list) else []
    )
    reference_assets_disabled = payload.get("referenceAssetMode") == "none"
    if reference_assets_disabled:
        image_prompts = []
    h3_prompts = prompts.get("h3Prompts") if isinstance(prompts.get("h3Prompts"), list) else []
    asset_plans = payload.get("assetPlans") if isinstance(payload.get("assetPlans"), list) else []
    fulfilled_plan_ids = {
        str(plan.get("id") or "")
        for plan in asset_plans
        if isinstance(plan, dict) and str(plan.get("fulfilledByAssetId") or "").strip()
    }
    blockers: list[str] = []

    valid_images: list[dict[str, Any]] = []
    for index, prompt in enumerate(image_prompts):
        if isinstance(prompt, dict) and str(prompt.get("assetPlanId") or "") in fulfilled_plan_ids:
            continue
        if not isinstance(prompt, dict) or not str(prompt.get("prompt") or "").strip():
            blockers.append(f"图片提示词 {index + 1} 为空")
            continue
        workflow_id = str(prompt.get("workflowTemplateId") or "")
        if not workflow_id:
            blockers.append(f"图片提示词 {index + 1} 未绑定工作流")
        elif workflow_id not in approved_image_workflows:
            blockers.append(f"图片提示词 {index + 1} 绑定的工作流未批准或不存在")
        else:
            valid_images.append(prompt)

    valid_h3: list[dict[str, Any]] = []
    for index, prompt in enumerate(h3_prompts):
        review = prompt.get("review") if isinstance(prompt, dict) else None
        if not isinstance(prompt, dict) or not str(prompt.get("prompt") or "").strip():
            blockers.append(f"H3 提示词 {index + 1} 为空")
        elif not isinstance(review, dict) or review.get("ready") is not True:
            blockers.append(f"H3 提示词 {index + 1} 尚未通过 Reviewer")
        else:
            valid_h3.append(prompt)
    if not valid_h3:
        blockers.append("至少需要一个通过 Reviewer 的 H3 片段提示词")

    canonical = json.dumps(
        {
            "project_id": project_id,
            "reference_asset_mode": payload.get("referenceAssetMode", "planned"),
            "image_prompts": valid_images,
            "h3_prompts": valid_h3,
            # AI takeover after a human-review timeout changes dispatch behavior,
            # not the structure or identity of the already compiled DAG.
            "review_mode": run_state.review_policy.configured_mode.value,
            "post_processing": payload.get("postProcessing", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    if blockers:
        return ProjectTaskPlan(fingerprint=fingerprint, tasks=(), blockers=tuple(blockers))

    tasks: list[TaskSpec] = []
    reusable_tasks = reusable_tasks or {}

    def canonical_bytes(value: Any) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def add(
        kind: TaskKind,
        label: str,
        *,
        depends_on: tuple[str, ...] = (),
        inputs: Any = None,
        affinity: str | None = None,
        max_attempts: int = 3,
    ) -> str:
        legacy_identity = hashlib.sha256(f"{kind.value}:{label}".encode()).hexdigest()[:16]
        reusable = reusable_tasks.get(f"{kind.value}:{legacy_identity}")
        if (
            reusable is not None
            and reusable.project_id == project_id
            and reusable.kind == kind
            and reusable.depends_on == depends_on
            and reusable.state not in {TaskState.CANCELLED, TaskState.STALE}
        ):
            tasks.append(reusable)
            return reusable.task_id
        # A task is identified only by its own inputs and its effective upstream
        # tasks. Project-wide options must not invalidate unrelated completed work.
        task_input = canonical_bytes(
            {
                "project_id": project_id,
                "kind": kind.value,
                "label": label,
                "inputs": inputs,
                "depends_on": depends_on,
            }
        )
        input_fingerprint = hashlib.sha256(task_input).hexdigest()
        task_id = f"run:{project_id}:{kind.value}:{input_fingerprint[:16]}"
        tasks.append(
            TaskSpec(
                task_id=task_id,
                project_id=project_id,
                kind=kind,
                state=TaskState.READY if not depends_on else TaskState.BLOCKED,
                idempotency_key=hashlib.sha256(
                    f"{task_id}:{input_fingerprint}".encode()
                ).hexdigest(),
                input_fingerprint=input_fingerprint,
                depends_on=depends_on,
                affinity_key=affinity,
                max_attempts=max_attempts,
            )
        )
        return task_id

    image_tasks: dict[str, str] = {}
    for index, prompt in enumerate(valid_images):
        plan_id = str(prompt.get("assetPlanId") or index)
        image_tasks[plan_id] = add(
            TaskKind.IMAGE_GENERATION,
            f"image:{plan_id}:{prompt.get('workflowTemplateId')}",
            inputs=prompt,
            affinity=f"image:{prompt.get('workflowTemplateId')}",
        )

    encode_tasks: dict[str, str] = {}
    all_image_dependencies = tuple(image_tasks.values())
    for prompt in valid_h3:
        segment_id = str(prompt.get("segmentId"))
        encode_tasks[segment_id] = add(
            TaskKind.CONDITIONING_ENCODING,
            f"conditioning:{segment_id}",
            depends_on=all_image_dependencies,
            inputs={
                "prompt": prompt,
                "width": payload.get("width"),
                "height": payload.get("height"),
                "reference_asset_mode": payload.get("referenceAssetMode", "planned"),
            },
            affinity="h3:conditioning",
        )

    switch_dependencies = tuple(encode_tasks.values())
    switch_task = add(
        TaskKind.MODEL_SWITCH,
        "conditioning-to-h3-diffusion",
        depends_on=switch_dependencies,
        inputs={"target": "h3-diffusion"},
        affinity="h3:diffusion",
        max_attempts=2,
    )
    h3_tasks: dict[str, str] = {}
    for prompt in valid_h3:
        segment_id = str(prompt.get("segmentId"))
        dependencies = [encode_tasks[segment_id], switch_task]
        continuation_of = str(prompt.get("continuationOf") or "")
        if continuation_of and continuation_of in h3_tasks:
            dependencies.append(h3_tasks[continuation_of])
        h3_tasks[segment_id] = add(
            TaskKind.H3_GENERATION,
            f"h3:{segment_id}",
            depends_on=tuple(dict.fromkeys(dependencies)),
            inputs={
                "prompt": prompt,
                "width": payload.get("width"),
                "height": payload.get("height"),
            },
            affinity="h3:diffusion",
        )

    review_tasks: dict[str, str] = {}
    for segment_id, h3_task in h3_tasks.items():
        review_tasks[segment_id] = add(
            TaskKind.AI_REVIEW,
            f"review:{segment_id}",
            depends_on=(h3_task,),
            inputs={
                "segment_id": segment_id,
                "mode": run_state.review_policy.configured_mode.value,
            },
            affinity=(
                "media:review"
                if run_state.review_policy.effective_mode.value == "none"
                else "llm:review"
            ),
            max_attempts=2,
        )

    post = payload.get("postProcessing") if isinstance(payload.get("postProcessing"), dict) else {}
    segment_outputs = dict(review_tasks)
    if isinstance(post.get("seedvr"), dict) and post["seedvr"].get("enabled") is True:
        config = post["seedvr"]
        if not config.get("profileId") or not config.get("profileRevision"):
            blockers.append("视频修复已启用，但尚未选择可用 Profile")
        elif config.get("profileId") == "seedvr2:official-video":
            blockers.append("SeedVR2 Profile 缺少可发布的 ComfyUI API 工作流，暂不可执行")
        if not config.get("modelId") or not config.get("vaeId"):
            blockers.append("视频修复已启用，但扩散模型或 VAE 尚未选择")
        for segment_id, dependency in tuple(segment_outputs.items()):
            segment_outputs[segment_id] = add(
                TaskKind.SEEDVR2,
                f"seedvr2:{segment_id}",
                depends_on=(dependency,),
                inputs={
                    **config,
                    "segment_id": segment_id,
                    "output_width": post.get("outputWidth"),
                    "output_height": post.get("outputHeight"),
                },
                affinity=(
                    f"postprocess:{config.get('profileId') or 'seedvr2'}:"
                    f"{config.get('modelId') or 'unselected'}"
                ),
            )
    if isinstance(post.get("rife"), dict) and post["rife"].get("enabled") is True:
        config = post["rife"]
        target_fps = config.get("targetFps")
        if not config.get("profileId") or not config.get("profileRevision"):
            blockers.append("插帧已启用，但尚未选择可用 Profile")
        elif config.get("profileId") != "interpolation:rife":
            blockers.append("所选插帧 Profile 尚无可执行的 ComfyUI API 工作流")
        if not config.get("modelId"):
            blockers.append("插帧已启用，但尚未选择 checkpoint/model")
        if target_fps not in {48, 60, 120}:
            blockers.append("插帧目标帧率必须为 48、60 或 120 fps")
        for segment_id, dependency in tuple(segment_outputs.items()):
            segment_outputs[segment_id] = add(
                TaskKind.RIFE,
                f"interpolation:{segment_id}",
                depends_on=(dependency,),
                inputs={**config, "segment_id": segment_id, "source_fps": payload.get("fps")},
                affinity=(
                    f"postprocess:{config.get('profileId') or 'interpolation'}:"
                    f"{config.get('modelId') or 'unselected'}"
                ),
            )
    master_task = add(
        TaskKind.MASTER_ASSEMBLY,
        "ffmpeg-master",
        depends_on=tuple(segment_outputs.values()),
        inputs={
            "output_width": post.get("outputWidth"),
            "output_height": post.get("outputHeight"),
            "fps": (
                post.get("rife", {}).get("targetFps")
                if isinstance(post.get("rife"), dict) and post["rife"].get("enabled") is True
                else payload.get("fps")
            ),
        },
        affinity="ffmpeg",
        max_attempts=2,
    )
    export_dependencies = [master_task]
    if isinstance(post.get("whisper"), dict) and post["whisper"].get("enabled") is True:
        config = post["whisper"]
        if not config.get("profileId") or not config.get("profileRevision"):
            blockers.append("字幕转写已启用，但尚未选择可用 Profile")
        if not config.get("modelId"):
            blockers.append("字幕转写已启用，但尚未选择模型")
        export_dependencies.append(
            add(
                TaskKind.WHISPER,
                "whisper",
                depends_on=(master_task,),
                inputs=config,
                affinity=f"whisper:{config.get('modelId') or 'unselected'}",
            )
        )
    export_settings = {
        key: value for key, value in post.items() if key not in {"seedvr", "rife", "whisper"}
    }
    add(
        TaskKind.EXPORT,
        "ffmpeg-delivery",
        depends_on=tuple(export_dependencies),
        inputs=export_settings,
        affinity="ffmpeg",
        max_attempts=2,
    )
    return ProjectTaskPlan(fingerprint=fingerprint, tasks=tuple(tasks), blockers=tuple(blockers))


def persist_project_task_plan(
    store: SQLiteTaskStore, plan: ProjectTaskPlan
) -> tuple[TaskSpec, ...]:
    if plan.blockers:
        raise ValueError("; ".join(plan.blockers))
    persisted: list[TaskSpec] = []
    for task in plan.tasks:
        candidate = task
        if task.state == TaskState.BLOCKED and task.depends_on:
            dependencies = tuple(store.get_task(task_id) for task_id in task.depends_on)
            if all(dependency.state == TaskState.SUCCEEDED for dependency in dependencies):
                candidate = task.model_copy(update={"state": TaskState.READY})
        persisted.append(store.add_task(candidate))
    return tuple(persisted)
