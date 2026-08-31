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
from ai_video_generator.services.generation_batches import canonical_motion_context_chain
from ai_video_generator.services.h3_loras import diffusion_affinity


@dataclass(frozen=True)
class ProjectTaskPlan:
    fingerprint: str
    tasks: tuple[TaskSpec, ...]
    blockers: tuple[str, ...] = ()


def append_delivery_tasks(
    *, add: Any, payload: dict[str, Any], segment_outputs: dict[str, str], blockers: list[str]
) -> None:
    """Append post-processing tasks to an existing task graph.

    ``segment_outputs`` may point at freshly compiled H3 tasks or at the
    stable task IDs behind already active segment versions. Keeping this
    builder independent from generation makes delivery-only compilation safe.
    """
    post = payload.get("postProcessing") if isinstance(payload.get("postProcessing"), dict) else {}
    # Assemble the generated segments first. Post-processing operates on the
    # single ordered master, never on individual BFS-produced segments.
    master_task = add(
        TaskKind.MASTER_ASSEMBLY,
        "ffmpeg-master",
        depends_on=tuple(segment_outputs.values()),
        inputs={"fps": payload.get("fps")},
        affinity="ffmpeg",
        max_attempts=2,
    )
    post_video_source = master_task
    if isinstance(post.get("seedvr"), dict) and post["seedvr"].get("enabled") is True:
        config = post["seedvr"]
        workflow_id = config.get("workflowTemplateId")
        workflow_revision = config.get("workflowRevision")
        if not workflow_id or not workflow_revision:
            blockers.append("视频修复已启用，但尚未选择用户工作流")
        seedvr_task = add(
            TaskKind.SEEDVR2,
            "seedvr2:master",
            depends_on=(master_task,),
            inputs={**config, "segment_id": "master"},
            affinity=(f"postprocess:{workflow_id or 'restoration'}:"
                      f"{config.get('modelId') or 'unselected'}"),
        )
        post_video_source = seedvr_task
    if isinstance(post.get("rife"), dict) and post["rife"].get("enabled") is True:
        config = post["rife"]
        target_fps = config.get("targetFps")
        workflow_id = config.get("workflowTemplateId")
        workflow_revision = config.get("workflowRevision")
        if not workflow_id or not workflow_revision:
            blockers.append("插帧已启用，但尚未选择用户工作流")
        if target_fps not in {48, 60, 120}:
            blockers.append("插帧目标帧率必须为 48、60 或 120 fps")
        post_video_source = add(
            TaskKind.RIFE,
            "interpolation:master",
            depends_on=(post_video_source,),
            inputs={**config, "segment_id": "master", "source_fps": payload.get("fps")},
            affinity=(f"postprocess:{workflow_id or 'interpolation'}:"
                      f"{config.get('modelId') or 'unselected'}"),
        )
    export_dependencies = [post_video_source]
    if isinstance(post.get("whisper"), dict) and post["whisper"].get("enabled") is True:
        config = post["whisper"]
        workflow_id = config.get("workflowTemplateId")
        workflow_revision = config.get("workflowRevision")
        if not workflow_id or not workflow_revision:
            blockers.append("字幕转写已启用，但尚未选择用户语音识别工作流")
        export_dependencies.append(
            add(
                TaskKind.WHISPER,
                "transcription",
                depends_on=(master_task,),
                inputs=config,
                affinity=(
                    f"postprocess:{workflow_id or 'transcription'}:"
                    f"{config.get('modelId') or 'workflow-default'}"
                ),
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


def compile_project_task_plan(
    *,
    project_id: str,
    workspace_revision: int,
    payload: dict[str, Any],
    run_state: ProjectRunState,
    approved_image_workflows: set[str],
    approved_image_harnesses: dict[str, int],
    h3_execution_profile: dict[str, Any] | None = None,
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
        if not isinstance(prompt, dict) or not str(prompt.get("prompt") or "").strip():
            blockers.append(f"H3 提示词 {index + 1} 为空")
        else:
            # Review provenance is audit-only. The text confirmed in the editor,
            # plus its execution metadata, is the complete MiniMax task input.
            valid_h3.append({key: value for key, value in prompt.items() if key != "review"})
    if not valid_h3:
        blockers.append("至少需要一个非空 H3 片段提示词")
    valid_h3 = list(canonical_motion_context_chain(valid_h3))

    # Asset ids are stable across revisions. Include the active blob identity
    # in the plan input so accepting a replacement image cannot reuse a task
    # manifest that was compiled with the previous bytes.
    workspace_assets = payload.get("assets") if isinstance(payload.get("assets"), list) else []
    asset_inputs = {
        str(asset.get("id")): {
            "sha256": str(asset.get("sha256") or ""),
            "revision": asset.get("revision"),
        }
        for asset in workspace_assets
        if isinstance(asset, dict) and asset.get("id")
    }

    canonical = json.dumps(
        {
            "project_id": project_id,
            "reference_asset_mode": payload.get("referenceAssetMode", "planned"),
            "image_prompts": valid_images,
            "h3_prompts": valid_h3,
            "asset_inputs": asset_inputs,
            "h3_execution_profile": h3_execution_profile or {},
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
        priority: int = 0,
    ) -> str:
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
                priority=priority,
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
    ordered_h3 = sorted(
        enumerate(valid_h3),
        key=lambda item: (int(item[1].get("segmentIndex") or 0), item[0]),
    )
    for _, prompt in ordered_h3:
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
                "h3_execution_profile": h3_execution_profile or {},
            },
            affinity="h3:conditioning",
        )

    switch_dependencies = tuple(encode_tasks.values())
    h3_affinity = diffusion_affinity(h3_execution_profile)
    switch_task = add(
        TaskKind.MODEL_SWITCH,
        "conditioning-to-h3-diffusion",
        depends_on=switch_dependencies,
        inputs={
            "target": "h3-diffusion",
            "h3_execution_profile": h3_execution_profile or {},
        },
        affinity=h3_affinity,
        max_attempts=2,
    )
    h3_tasks: dict[str, str] = {}
    for stable_index, prompt in ordered_h3:
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
                "h3_execution_profile": h3_execution_profile or {},
            },
            affinity=h3_affinity,
            priority=100 - min(99, int(prompt.get("segmentIndex") or 0) * 10 + stable_index),
        )

    review_tasks: dict[str, str] = {}
    if run_state.review_policy.effective_mode.value not in {"manual", "none"}:
        for segment_id, h3_task in h3_tasks.items():
            review_tasks[segment_id] = add(
                TaskKind.AI_REVIEW,
                f"review:{segment_id}",
                depends_on=(h3_task,),
                inputs={
                    "segment_id": segment_id,
                    "mode": run_state.review_policy.configured_mode.value,
                },
                affinity="llm:review",
                max_attempts=2,
            )

    # Review is a side-channel consumer. Delivery can also be compiled later
    # against active segment versions through the same builder.
    append_delivery_tasks(add=add, payload=payload, segment_outputs=h3_tasks, blockers=blockers)
    return ProjectTaskPlan(fingerprint=fingerprint, tasks=tuple(tasks), blockers=tuple(blockers))


def compile_delivery_task_plan(
    *,
    project_id: str,
    payload: dict[str, Any],
    segment_outputs: dict[str, str],
) -> ProjectTaskPlan:
    """Compile only delivery tasks from an already active video chain."""
    canonical = json.dumps(
        {
            "project_id": project_id,
            "post_processing": payload.get("postProcessing", {}),
            "segment_outputs": segment_outputs,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    fingerprint = hashlib.sha256(canonical).hexdigest()
    tasks: list[TaskSpec] = []

    def add(
        kind: TaskKind,
        label: str,
        *,
        depends_on: tuple[str, ...] = (),
        inputs: Any = None,
        affinity: str | None = None,
        max_attempts: int = 3,
        priority: int = 0,
    ) -> str:
        task_input = json.dumps(
            {
                "project_id": project_id,
                "kind": kind.value,
                "label": label,
                "inputs": inputs,
                "depends_on": depends_on,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        input_fingerprint = hashlib.sha256(task_input).hexdigest()
        task_id = f"delivery:{project_id}:{kind.value}:{input_fingerprint[:16]}"
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
                priority=priority,
                max_attempts=max_attempts,
            )
        )
        return task_id

    blockers: list[str] = []
    append_delivery_tasks(
        add=add, payload=payload, segment_outputs=segment_outputs, blockers=blockers
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
