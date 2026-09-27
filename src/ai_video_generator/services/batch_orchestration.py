from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException

from ai_video_generator.api_models import ProjectAgentOperationRequest
from ai_video_generator.domain import BatchState, ExecutionMode, TaskKind, TaskSpec, TaskState
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.persistence.execution_runtime import execution_guard
from ai_video_generator.prompting_api import _workspace_segments
from ai_video_generator.services.batch_runs import resolve_batch_task_ids


class OrchestrationNeedsAttention(ValueError):
    """An existing output or uncertain operation needs explicit reconciliation."""


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stable_operation_id(task: TaskSpec, scope: str, inputs: object) -> str:
    return (
        "batch-op:"
        + _fingerprint([task.task_id, task.input_fingerprint, scope, _fingerprint(inputs)])[:40]
    )


def _semantic_inputs(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Exclude UI timestamps/revisions from resumable operation identity."""
    return {key: payload.get(key) for key in keys}


def _image_execution_inputs(
    store: SQLiteTaskStore,
    payload: dict[str, Any],
    plan: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    references = set(entry.get("referenceAssetIds", []))
    return {
        "plan": plan,
        "prompt": entry,
        "references": [
            asset
            for asset in payload.get("assets", [])
            if isinstance(asset, dict) and asset.get("id") in references
        ],
        "workflow_revisions": [
            template.model_dump(mode="json")
            for template in store.list_workflow_revisions()
            if template.template_id == entry.get("workflowTemplateId")
        ],
    }


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(_text(item) for item in value)


def _revision(entry: dict[str, Any]) -> bool:
    return (
        _text(entry.get("id"))
        and type(entry.get("revision")) is int
        and entry["revision"] >= 1
        and type(entry.get("locked")) is bool
        and (
            entry.get("harnessRevision") is None
            or (type(entry["harnessRevision"]) is int and entry["harnessRevision"] >= 1)
        )
    )


def reusable_image_prompt(entry: dict[str, Any], plan_id: str) -> bool:
    """Validate the workspace image entry, not the distinct persisted revision model."""
    return (
        _revision(entry)
        and entry.get("assetPlanId") == plan_id
        and _text(entry.get("prompt"))
        and isinstance(entry.get("negativePrompt"), str)
        and _text(entry.get("workflowTemplateId"))
        and _strings(entry.get("referenceAssetIds"))
    )


def reusable_h3_prompt(entry: dict[str, Any], segment: dict[str, Any]) -> bool:
    """Match execution metadata to the authoritative storyboard segment topology."""
    duration = entry.get("durationSeconds")
    return (
        _revision(entry)
        and _text(entry.get("prompt"))
        and len(entry["prompt"]) <= 7000
        and entry.get("inputMode") in {"t2va", "i2va", "fl2va", "l2va", "ref2va"}
        and _strings(entry.get("assetIds"))
        and (entry["inputMode"] != "ref2va" or bool(entry["assetIds"]))
        and type(entry.get("seed")) is int
        and type(entry.get("segmentIndex")) is int
        and all(
            entry.get(key) == segment.get(key)
            for key in ("segmentId", "shotId", "segmentIndex", "continuationOf")
        )
        and type(duration) in {int, float}
        and math.isfinite(duration)
        and abs(duration - segment["durationSeconds"]) < 0.01
        and (entry.get("endState") is None or _text(entry["endState"]))
    )


def _entries(payload: dict[str, Any], name: str, identity: str) -> dict[str, dict[str, Any]]:
    prompts = payload.get("prompts", {})
    if not isinstance(prompts, dict):
        raise OrchestrationNeedsAttention("prompts must be a workspace prompt object")
    entries = prompts.get(name, [])
    if not isinstance(entries, list):
        raise OrchestrationNeedsAttention(f"{name} must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not _text(entry.get(identity)):
            raise OrchestrationNeedsAttention(f"{name} contains an invalid {identity}")
        key = entry[identity]
        if key in indexed:
            raise OrchestrationNeedsAttention(f"{name} contains duplicate {identity}: {key}")
        indexed[key] = entry
    return indexed


class _Progress:
    def __init__(self, store: SQLiteTaskStore, task: TaskSpec):
        self.store = store
        self.task = task
        checkpoints = store.list_task_checkpoints(task.task_id)
        self.records = {
            (record.phase, record.payload.get("scope", "")): record.payload
            for record in checkpoints
        }
        self.started = {
            (
                record.payload.get("scope", ""),
                record.payload.get("input_fingerprint"),
            ): record.payload
            for record in checkpoints
            if record.phase == "batch_operation_started"
        }

    def record(self, phase: str, scope: str = "", **payload: Any) -> dict[str, Any]:
        value = {"scope": scope, **payload}
        if self.records.get((phase, scope)) != value:
            self.store.append_task_checkpoint(self.task.task_id, phase, value)
            self.records[phase, scope] = value
        return value

    def start(self, scope: str, inputs: object) -> str:
        fingerprint = _fingerprint(inputs)
        existing = self.started.get((scope, fingerprint))
        failure = self.records.get(("batch_operation_rejected", scope))
        rejected = (
            existing is not None
            and failure
            and (failure.get("operation_id") == existing["operation_id"])
        )
        if existing is not None and not rejected:
            raise OrchestrationNeedsAttention(
                f"Operation {existing['operation_id']} has no validated output; "
                "reconcile it before retrying, rather than repeating generation"
            )
        operation_id = stable_operation_id(self.task, scope, inputs)
        if rejected:
            operation_id += f":retry:{self.task.attempt}"
        self.started[scope, fingerprint] = self.record(
            "batch_operation_started",
            scope,
            input_fingerprint=fingerprint,
            operation_id=operation_id,
        )
        return operation_id

    @contextmanager
    def rejection_boundary(self, scope: str):
        # Only a completed local handler's explicit HTTP rejection is evidence
        # of failure. Transport loss/cancellation still leaves an ambiguous intent.
        try:
            yield
        except HTTPException as exc:
            started = self.records[("batch_operation_started", scope)]
            self.record(
                "batch_operation_rejected",
                scope,
                operation_id=started["operation_id"],
                status_code=exc.status_code,
            )
            raise

    def yield_task(self, scope: str, dependencies: tuple[str, ...] = ()) -> bool:
        self.record("batch_yield", scope, dependencies=list(dependencies))
        self.store.defer_orchestration(self.task.task_id, dependencies=dependencies)
        return False


async def run_batch_project_orchestration(
    task: TaskSpec,
    *,
    store: SQLiteTaskStore,
    internal: httpx.AsyncClient,
    operate_project_with_agent: Callable[[str, ProjectAgentOperationRequest], Awaitable[Any]],
    batch_settings: dict[str, Any] | None = None,
) -> bool:
    """Advance at most one generation operation, or register the compiled DAG.

    The caller owns the HTTP client and the claimed execution_guard context.
    False always means the store durably released this attempt for resumption.
    Ambiguous prior calls and invalid confirmed entries require explicit review.
    """
    guard = execution_guard.get()
    if not task.attempt_id or guard != (task.task_id, task.attempt_id):
        raise ValueError("batch orchestration requires its claimed execution_guard")
    progress = _Progress(store, task)
    project_id = task.project_id
    base = f"/api/v1/projects/{quote(project_id, safe='')}"
    state = store.get_project_run_state(project_id)
    if not state.outline_approved:
        raise OrchestrationNeedsAttention("batch orchestration requires an approved outline")
    if state.execution_mode != ExecutionMode.BATCH:
        store.put_project_run_state(
            state.model_copy(
                update={
                    "execution_mode": ExecutionMode.BATCH,
                    "updated_at": datetime.now(UTC),
                }
            )
        )
    progress.record(
        "batch_orchestration_started", review_policy=state.review_policy.model_dump(mode="json")
    )

    async def post(path: str, payload: dict[str, Any]) -> httpx.Response:
        response = await internal.post(base + path, json=payload)
        if response.is_error:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Batch operation {path} failed: {response.text[:2000]}",
            )
        return response

    payload = store.get_latest_project_workspace(project_id).payload
    for scope, operation, paths, instruction in (
        (
            "storyboard",
            "initialize_storyboard",
            ("/shots",),
            "根据已批准大纲生成完整电影分镜。motionSegments 逐段填写 durationSeconds 和 summary；"
            "分段合计等于镜头时长，每段至少4秒，首段最多15秒，续段最多12秒。"
            "接缝放在运动与机位较稳定处，summary 写清交给下一段继承的结束状态。",
        ),
        (
            "asset_plan",
            "initialize_assets",
            ("/assetPlans", "/referenceAssetMode"),
            "根据已批准创意、大纲、分镜和现有项目图片规划仍缺少的素材。不得重复已有素材；"
            "每项图片总像素约1280×1280并独立决定构图。无需素材时设置 referenceAssetMode=none。",
        ),
    ):
        ready = (
            bool(payload.get("shots"))
            if scope == "storyboard"
            else (bool(payload.get("assetPlans")) or payload.get("referenceAssetMode") == "none")
        )
        inputs = _semantic_inputs(payload, "idea", "outline", "highestInstruction", "name")
        if scope == "asset_plan":
            inputs.update(_semantic_inputs(payload, "shots", "assets"))
        input_fingerprint = _fingerprint(inputs)
        if ready:
            progress.record("batch_output_ready", scope, input_fingerprint=input_fingerprint)
            continue
        ready_record = progress.records.get(("batch_output_ready", scope))
        if ready_record and ready_record.get("input_fingerprint") == input_fingerprint:
            raise OrchestrationNeedsAttention(f"Previously completed {scope} output is missing")
        operation_id = progress.start(scope, inputs)
        with progress.rejection_boundary(scope):
            await operate_project_with_agent(
                project_id,
                ProjectAgentOperationRequest(
                    operation_id=operation_id,
                    operation=operation,
                    instruction=instruction,
                    display_instruction=instruction,
                    allowed_paths=paths,
                    commit=True,
                ),
            )
        return progress.yield_task(scope)

    segments = _workspace_segments(payload)
    if not segments:
        raise OrchestrationNeedsAttention("Storyboard has no valid execution segments")
    expected_ids = {segment["segmentId"] for segment in segments}
    if len(expected_ids) != len(segments):
        raise OrchestrationNeedsAttention("Storyboard contains duplicate segment IDs")

    plans = payload.get("assetPlans", [])
    if not isinstance(plans, list) or any(
        not isinstance(plan, dict) or not _text(plan.get("id")) for plan in plans
    ):
        raise OrchestrationNeedsAttention("Invalid workspace asset plans")
    if len({plan["id"] for plan in plans}) != len(plans):
        raise OrchestrationNeedsAttention("Duplicate asset plan IDs")
    image_prompts = _entries(payload, "imagePrompts", "assetPlanId")
    for plan in plans if payload.get("referenceAssetMode") != "none" else []:
        plan_id = plan["id"]
        if plan.get("fulfilledByAssetId"):
            continue
        scope = f"image:{plan_id}"
        entry = image_prompts.get(plan_id)
        if entry is not None and not reusable_image_prompt(entry, plan_id):
            raise OrchestrationNeedsAttention(f"Review invalid existing image prompt: {plan_id}")
        if entry is None:
            progress.start(
                scope,
                {
                    "plan": plan,
                    **_semantic_inputs(payload, "idea", "shots", "assets"),
                    "workflow_template_id": (batch_settings or {}).get("imageWorkflowId"),
                },
            )
            with progress.rejection_boundary(scope):
                await post(
                    f"/prompts/images/{quote(plan_id, safe='')}/generate",
                    {
                        "instruction": None,
                        "workflow_template_id": (batch_settings or {}).get("imageWorkflowId"),
                    },
                )
            payload = store.get_latest_project_workspace(project_id).payload
            image_prompts = _entries(payload, "imagePrompts", "assetPlanId")
            entry = image_prompts.get(plan_id)
            if entry is None or not reusable_image_prompt(entry, plan_id):
                raise OrchestrationNeedsAttention(
                    f"Image prompt operation did not commit a valid result: {plan_id}"
                )
            # Prompt generation may also choose the plan's resolution.
            updated_plans = payload.get("assetPlans")
            plan = (
                next(
                    (
                        item
                        for item in updated_plans
                        if isinstance(item, dict) and item.get("id") == plan_id
                    ),
                    None,
                )
                if isinstance(updated_plans, list)
                else None
            )
            if plan is None:
                raise OrchestrationNeedsAttention(
                    f"Image prompt operation removed its asset plan: {plan_id}"
                )
        progress.record("batch_output_ready", scope, output_fingerprint=_fingerprint(entry))
        execution_fingerprint = _fingerprint(_image_execution_inputs(store, payload, plan, entry))
        saved = progress.records.get(("batch_image_child", scope))
        child = (
            store.get_task(saved["task_id"])
            if saved and saved.get("input_fingerprint") == execution_fingerprint
            else None
        )
        if child is None:
            # The endpoint freezes and identifies exact execution inputs. A task-ID
            # prefix only identifies the plan, and can point at an obsolete image.
            response = await post(
                f"/image-prompts/{quote(plan_id, safe='')}/run",
                {
                    "expected_workspace_sha256": _fingerprint(payload),
                },
            )
            child = store.get_task(response.json()["task_id"])
        if child.project_id != project_id or child.kind != TaskKind.IMAGE_GENERATION:
            raise OrchestrationNeedsAttention("Image operation returned an unrelated task")
        progress.record(
            "batch_image_child",
            scope,
            task_id=child.task_id,
            input_fingerprint=execution_fingerprint,
        )
        _register(store, task, (child.task_id,), resolve_boundary=False)
        if child.state != TaskState.SUCCEEDED:
            return progress.yield_task(scope, (child.task_id,))
        raise OrchestrationNeedsAttention(
            f"Image {child.task_id} succeeded but plan {plan_id} is not fulfilled; "
            "reconcile or accept its existing output"
        )

    h3_prompts = _entries(payload, "h3Prompts", "segmentId")
    if set(h3_prompts) - expected_ids:
        raise OrchestrationNeedsAttention("H3 entries reference segments outside the storyboard")
    for segment in segments:
        segment_id = segment["segmentId"]
        entry = h3_prompts.get(segment_id)
        if entry is not None and not reusable_h3_prompt(entry, segment):
            raise OrchestrationNeedsAttention(f"Review invalid existing H3 prompt: {segment_id}")
        if entry is None:
            scope = f"h3:{segment_id}"
            progress.start(
                scope,
                {
                    "segment": segment,
                    **_semantic_inputs(
                        payload, "idea", "assets", "assetPlans", "highestInstruction"
                    ),
                },
            )
            with progress.rejection_boundary(scope):
                await post(f"/prompts/h3/{quote(segment_id, safe='')}/regenerate", {})
            return progress.yield_task(scope)

    compile_fingerprint = _fingerprint(
        {
            "workspace": _semantic_inputs(
                payload,
                "prompts",
                "shots",
                "assets",
                "assetPlans",
                "referenceAssetMode",
                "postProcessing",
                "width",
                "height",
                "fps",
                "h3Loras",
            ),
            "batch_settings": batch_settings,
            "review_policy": state.review_policy.configured_mode.value,
            "generation_revision": state.generation_revision,
        }
    )
    compiled = progress.records.get(("batch_dag_compiled", ""))
    if compiled is not None:
        if compiled["input_fingerprint"] != compile_fingerprint:
            raise OrchestrationNeedsAttention("Workspace changed after DAG compilation")
        task_ids = tuple(compiled["task_ids"])
    else:
        progress.record("batch_dag_compiling")
        response = await post(
            "/tasks/compile",
            {
                "batch_settings": batch_settings,
            }
            if batch_settings
            else {},
        )
        result = response.json()
        entries = result.get("tasks") if isinstance(result, dict) else None
        if not isinstance(entries, list) or any(
            not isinstance(item, dict) or not _text(item.get("task_id")) for item in entries
        ):
            raise OrchestrationNeedsAttention("Compiler returned an invalid task list")
        task_ids = tuple(item["task_id"] for item in entries)
        if not task_ids or task.task_id in task_ids or len(set(task_ids)) != len(task_ids):
            raise OrchestrationNeedsAttention("Compiler did not return a child DAG")
        progress.record(
            "batch_dag_compiled", input_fingerprint=compile_fingerprint, task_ids=list(task_ids)
        )
    _register(store, task, task_ids, resolve_boundary=True)
    progress.record("batch_dag_registered", task_ids=list(task_ids))
    return True


def _register(
    store: SQLiteTaskStore, task: TaskSpec, task_ids: tuple[str, ...], *, resolve_boundary: bool
) -> None:
    tasks = tuple(store.get_task(task_id) for task_id in task_ids)
    if any(child.project_id != task.project_id for child in tasks):
        raise OrchestrationNeedsAttention("Compiled DAG contains another project's tasks")
    registered = False
    for batch in store.list_batch_runs():
        if batch.state not in {BatchState.RUNNING, BatchState.PAUSED}:
            continue
        for member in batch.items:
            if member.project_id != task.project_id or task.task_id not in member.task_ids:
                continue
            store.expand_running_batch_project_tasks(
                batch_id=batch.batch_id,
                project_id=task.project_id,
                orchestration_task_id=task.task_id,
                task_ids=(
                    resolve_batch_task_ids(tasks, member.start_boundary)
                    if resolve_boundary
                    else task_ids
                ),
            )
            registered = True
    if not registered:
        raise OrchestrationNeedsAttention("Orchestration no longer belongs to an active batch")
