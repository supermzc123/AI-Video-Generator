from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_video_generator.config import Settings, load_runtime_settings
from ai_video_generator.domain import (
    ApprovalState,
    DecisionSource,
    H3PromptRevision,
    MemoryEventKind,
    ProjectMemoryEvent,
    ProjectWorkspaceRevision,
    PromptIssueSeverity,
    PromptReviewIssue,
    PromptReviewResult,
    PromptRevisionState,
    StageGenerationCheckpoint,
    StageGenerationState,
)
from ai_video_generator.domain.h3_prompt import (
    H3AssetInput,
    H3AssetKind,
    H3AssetPromptRole,
    H3PromptRequest,
    H3ShotStrategy,
)
from ai_video_generator.llm import (
    ChatMessage,
    H3HarnessLibrary,
    H3PromptHarnessError,
    ImageURL,
    ImageURLContentPart,
    LLMClientError,
    TextContentPart,
    complete_json,
    complete_text,
    exception_messages,
    remote_config,
)
from ai_video_generator.llm.h3_prompt import deterministic_director_decision
from ai_video_generator.llm.harness_files import load_harness
from ai_video_generator.persistence import SQLiteTaskStore, StoreConflictError
from ai_video_generator.persistence.project_assets import (
    ProjectAssetNotFoundError,
    ProjectAssetStore,
)
from ai_video_generator.services.llm_streaming import stream_llm_operation
from ai_video_generator.services.workspace_topology import normalize_workspace_topology


class ImagePromptGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str | None = Field(default=None, max_length=4_000)
    workflow_template_id: str | None = Field(default=None, min_length=1, max_length=200)


class H3PromptTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_revision_id: str
    source_sha256: str
    language: str = "zh-CN"
    translation: str = Field(min_length=1, max_length=20_000)
    executable: bool = False


class _TranslationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    translation: str = Field(min_length=1, max_length=20_000)


_prompt_commit_locks_guard = threading.Lock()
_prompt_commit_locks: dict[str, threading.Lock] = {}


def _system_with_highest_instruction(system: str, payload: dict[str, Any]) -> str:
    """Prepend the project's instruction to any task-specific Harness system prompt."""
    highest = str(payload.get("highestInstruction") or "").strip()
    if not highest:
        return system
    return (
        "PROJECT HIGHEST INSTRUCTION\n"
        f"{highest}\n\n"
        "The following Harness documents define the task-specific rules and output contract.\n\n"
        f"{system}"
    )


def _commit_prompt_result(project_id: str, commit: Callable[[], Any]) -> Any:
    """Serialize only the short revision merge, while LLM calls remain concurrent."""
    with _prompt_commit_locks_guard:
        lock = _prompt_commit_locks.setdefault(project_id, threading.Lock())
    with lock:
        return commit()


def _merge_h3_prompt_entries(
    existing: list[Any], incoming: list[Any]
) -> list[Any]:
    """Merge segment drafts without allowing an empty result to erase text."""
    incoming_by_id = {
        str(item.get("segmentId")): item
        for item in incoming
        if isinstance(item, dict) and str(item.get("segmentId") or "")
    }
    merged: list[Any] = []
    existing_ids: set[str] = set()
    for current in existing:
        if not isinstance(current, dict):
            merged.append(current)
            continue
        segment_id = str(current.get("segmentId") or "")
        existing_ids.add(segment_id)
        candidate = incoming_by_id.get(segment_id)
        if candidate is None:
            merged.append(current)
            continue
        current_text = str(current.get("prompt") or "").strip()
        candidate_text = str(candidate.get("prompt") or "").strip()
        merged.append(current if current_text and not candidate_text else candidate)
    merged.extend(item for key, item in incoming_by_id.items() if key not in existing_ids)
    return merged


def _h3_generation_summary(entries: list[Any]) -> dict[str, Any]:
    failed = [
        item
        for item in entries
        if isinstance(item, dict)
        and isinstance(item.get("review"), dict)
        and item["review"].get("ready") is not True
    ]
    return {
        "total": len(entries),
        "succeeded": len(entries) - len(failed),
        "failed": len(failed),
        "failedSegmentIds": [str(item.get("segmentId") or "") for item in failed],
    }


def create_prompting_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["project-prompts"])
    task_store = SQLiteTaskStore(Path(settings.data_root) / "control-plane.db")
    asset_store = ProjectAssetStore(
        Path(settings.data_root) / "control-plane.db",
        Path(settings.data_root) / "project-assets",
    )

    @router.post("/api/v1/projects/{project_id}/stages/prompts/generate")
    async def generate_prompt_stage(project_id: str) -> ProjectWorkspaceRevision:
        return await _generate_and_commit(
            settings,
            task_store,
            asset_store,
            project_id,
            only_segment_id=None,
        )

    @router.post("/api/v1/projects/{project_id}/prompts/h3/{segment_id}/regenerate")
    async def regenerate_h3_prompt(project_id: str, segment_id: str) -> ProjectWorkspaceRevision:
        return await _generate_and_commit(
            settings,
            task_store,
            asset_store,
            project_id,
            only_segment_id=segment_id,
        )

    @router.post("/api/v1/projects/{project_id}/prompts/h3/{segment_id}/regenerate/stream")
    async def stream_h3_prompt(
        project_id: str,
        segment_id: str,
        x_llm_operation_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        return stream_llm_operation(
            lambda: _generate_and_commit(
                settings,
                task_store,
                asset_store,
                project_id,
                only_segment_id=segment_id,
            ),
            project_id=project_id,
            kind="h3_prompt",
            scope=segment_id,
            operation_id=x_llm_operation_id,
        )

    @router.post(
        "/api/v1/projects/{project_id}/prompts/h3/{prompt_revision_id}/translate",
        response_model=H3PromptTranslation,
    )
    async def translate_h3_prompt(project_id: str, prompt_revision_id: str) -> H3PromptTranslation:
        try:
            revision = task_store.get_h3_prompt_revision(prompt_revision_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="H3 prompt revision not found") from exc
        if revision.project_id != project_id:
            raise HTTPException(status_code=404, detail="H3 prompt revision not found")
        source_sha256 = hashlib.sha256(revision.execution_prompt.encode("utf-8")).hexdigest()
        cached = task_store.get_h3_prompt_translation(prompt_revision_id, source_sha256)
        if cached is not None:
            return H3PromptTranslation(
                prompt_revision_id=prompt_revision_id,
                source_sha256=source_sha256,
                translation=cached,
            )
        runtime = load_runtime_settings(settings)
        workspace = task_store.get_latest_project_workspace(project_id)
        messages = (
            ChatMessage(
                role="system",
                content=_system_with_highest_instruction(
                    load_harness("h3-translation.md"), workspace.payload
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "response_schema": _TranslationEnvelope.model_json_schema(),
                        "execution_prompt": revision.execution_prompt,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        try:
            raw = await complete_json(
                remote_config(runtime),
                messages,
            )
            translated = _TranslationEnvelope.model_validate_json(raw).translation
        except (LLMClientError, ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail=f"H3 prompt translation failed: {exc}"
            ) from exc
        cached = task_store.put_h3_prompt_translation(prompt_revision_id, source_sha256, translated)
        return H3PromptTranslation(
            prompt_revision_id=prompt_revision_id,
            source_sha256=source_sha256,
            translation=cached,
        )

    @router.post("/api/v1/projects/{project_id}/prompts/images/{asset_plan_id}/generate")
    async def generate_image_prompt(
        project_id: str,
        asset_plan_id: str,
        request: ImagePromptGenerationRequest,
    ) -> ProjectWorkspaceRevision:
        return await _generate_image_prompt_and_commit(
            settings,
            task_store,
            asset_store,
            project_id,
            asset_plan_id,
            instruction=request.instruction,
            workflow_template_id=request.workflow_template_id,
        )

    @router.post("/api/v1/projects/{project_id}/prompts/images/{asset_plan_id}/generate/stream")
    async def stream_image_prompt(
        project_id: str,
        asset_plan_id: str,
        request: ImagePromptGenerationRequest,
        x_llm_operation_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        return stream_llm_operation(
            lambda: _generate_image_prompt_and_commit(
                settings,
                task_store,
                asset_store,
                project_id,
                asset_plan_id,
                instruction=request.instruction,
                workflow_template_id=request.workflow_template_id,
            ),
            project_id=project_id,
            kind="image_prompt",
            scope=asset_plan_id,
            operation_id=x_llm_operation_id,
        )

    return router


async def _generate_image_prompt_and_commit(
    settings: Settings,
    store: SQLiteTaskStore,
    asset_store: ProjectAssetStore,
    project_id: str,
    asset_plan_id: str,
    instruction: str | None = None,
    workflow_template_id: str | None = None,
) -> ProjectWorkspaceRevision:
    runtime = load_runtime_settings(settings)
    if not runtime.llm_base_url or not runtime.llm_model:
        raise HTTPException(status_code=503, detail="LLM provider is not configured")
    try:
        project = store.get_latest_project_revision(project_id)
        workspace = store.get_latest_project_workspace(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project workspace not found") from exc

    plans = workspace.payload.get("assetPlans")
    plan = (
        next(
            (item for item in plans if isinstance(item, dict) and item.get("id") == asset_plan_id),
            None,
        )
        if isinstance(plans, list)
        else None
    )
    if plan is None:
        raise HTTPException(status_code=404, detail="asset plan not found")
    workflow_id, harness_revision_number = _image_workflow_binding(store, workflow_template_id)
    if not workflow_id:
        raise HTTPException(
            status_code=409,
            detail="没有可用的已批准图片工作流",
        )
    bundles = [
        bundle
        for bundle in store.list_harness_bundles()
        if bundle.purpose == "image_prompting" and bundle.workflow_template_id == workflow_id
    ]
    revisions = [
        revision
        for bundle in bundles
        for revision in store.list_harness_revisions(bundle.harness_id)
        if revision.approval == ApprovalState.APPROVED
        and (harness_revision_number is None or revision.revision == harness_revision_number)
    ]
    harness_revision = revisions[-1] if revisions else None

    assets = [
        asset for asset in asset_store.list_assets(project_id) if asset.state.value == "available"
    ]
    bound_asset_ids: set[str] = set()
    for plan_item in (
        workspace.payload.get("assetPlans", [])
        if isinstance(workspace.payload.get("assetPlans"), list)
        else []
    ):
        if (
            isinstance(plan_item, dict)
            and str(plan_item.get("id")) == asset_plan_id
            and plan_item.get("fulfilledByAssetId")
        ):
            bound_asset_ids.add(str(plan_item["fulfilledByAssetId"]))
    asset_context = [
        {
            "asset_id": asset.asset_id,
            "name": asset.name,
            "kind": asset.kind.value,
            "scope": asset.scope.value,
            "shot_id": asset.shot_id,
            "shot_ids": list(asset.shot_ids),
        }
        for asset in assets
    ]
    existing_prompts = workspace.payload.get("prompts", {}).get("imagePrompts", [])
    existing_prompt = (
        next(
            (
                item
                for item in existing_prompts
                if isinstance(item, dict) and item.get("assetPlanId") == asset_plan_id
            ),
            None,
        )
        if isinstance(existing_prompts, list)
        else None
    )
    context = {
        "project": {
            "name": workspace.payload.get("name"),
            "idea": workspace.payload.get("idea"),
            "shots": workspace.payload.get("shots"),
        },
        "asset_plan": plan,
        "existing_image_prompt": existing_prompt,
        "available_reference_assets": asset_context,
        "workflow_template_id": workflow_id,
        "user_revision_instruction": instruction,
        "image_resolution_policy": {
            "independent_from_video_resolution": True,
            "target_total_pixels": 1280 * 1280,
            "dimension_multiple": 8,
            "minimum_dimension": 64,
            "maximum_dimension": 4096,
            "manual_resolution_is_locked": plan.get("resolutionSource") == "manual",
            "instruction": (
                "按素材构图选择横图、竖图或方图；总像素控制在1280×1280左右。"
                "若 resolutionSource 为 manual，必须原样返回 asset_plan 中的 width 和 height。"
            ),
        },
    }
    system = "\n\n".join(
        (
            harness_revision.markdown
            if harness_revision
            else load_harness("image-default.md"),
            load_harness("image-writer.md"),
        )
    )
    system = _system_with_highest_instruction(system, workspace.payload)
    text_payload = (
        "项目主管交给你的完整图片生成任务如下。请直接完成，不要反问。只返回提示词正文：\n\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
    )
    content: str | tuple[TextContentPart | ImageURLContentPart, ...] = text_payload
    if assets:
        content = (
            TextContentPart(text=text_payload),
            *(
                ImageURLContentPart(
                    image_url=ImageURL(url=_preview_data_url(asset_store, asset), detail="low")
                )
                for asset in assets
            ),
        )
    messages = (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=content),
    )
    try:
        prompt_text = await complete_text(
            remote_config(runtime),
            messages,
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    prompt_text = prompt_text.strip()
    if len(prompt_text) < 20:
        raise HTTPException(status_code=422, detail="LLM 返回的图片提示词过短")

    payload = json.loads(json.dumps(workspace.payload))
    prompts = payload.setdefault("prompts", {})
    image_prompts = prompts.get("imagePrompts")
    image_prompts = image_prompts if isinstance(image_prompts, list) else []
    old = next(
        (
            item
            for item in image_prompts
            if isinstance(item, dict) and item.get("assetPlanId") == asset_plan_id
        ),
        None,
    )
    entry = {
        "id": (old or {}).get("id")
        or f"image-{hashlib.sha256(asset_plan_id.encode()).hexdigest()[:24]}",
        "assetPlanId": asset_plan_id,
        "prompt": prompt_text,
        "negativePrompt": str((old or {}).get("negativePrompt") or "N/A"),
        "workflowTemplateId": workflow_id,
        "harnessRevision": harness_revision_number,
        "referenceAssetIds": list(bound_asset_ids or (old or {}).get("referenceAssetIds") or []),
        "locked": bool((old or {}).get("locked", False)),
        "revision": int((old or {}).get("revision") or 0) + 1,
    }
    manual_resolution = plan.get("resolutionSource") == "manual"
    selected_width = int(plan.get("width") or 1024)
    selected_height = int(plan.get("height") or 1024)
    prompts["imagePrompts"] = [
        entry if isinstance(item, dict) and item.get("assetPlanId") == asset_plan_id else item
        for item in image_prompts
    ]
    if old is None:
        prompts["imagePrompts"].append(entry)
    payload["assetPlans"] = [
        {
            **item,
            "state": "ready",
            "width": selected_width,
            "height": selected_height,
            "resolutionSource": "manual" if manual_resolution else "ai",
        }
        if isinstance(item, dict) and item.get("id") == asset_plan_id
        else item
        for item in plans
    ]
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    next_revision = project.revision + 1
    payload["revision"] = next_revision

    def commit_latest_image_prompt() -> ProjectWorkspaceRevision:
        latest_project = store.get_latest_project_revision(project_id)
        latest_workspace = store.get_latest_project_workspace(project_id)
        latest_payload = json.loads(json.dumps(latest_workspace.payload))
        latest_prompts = latest_payload.setdefault("prompts", {})
        latest_images = (
            latest_prompts.get("imagePrompts")
            if isinstance(latest_prompts.get("imagePrompts"), list)
            else []
        )
        new_images = payload["prompts"]["imagePrompts"]
        by_plan = {
            str(item.get("assetPlanId")): item for item in new_images if isinstance(item, dict)
        }
        merged_images = [
            by_plan.get(str(item.get("assetPlanId")), item) if isinstance(item, dict) else item
            for item in latest_images
        ]
        existing_plan_ids = {
            str(item.get("assetPlanId")) for item in merged_images if isinstance(item, dict)
        }
        merged_images.extend(item for key, item in by_plan.items() if key not in existing_plan_ids)
        latest_prompts["imagePrompts"] = merged_images
        latest_plans = (
            latest_payload.get("assetPlans")
            if isinstance(latest_payload.get("assetPlans"), list)
            else []
        )
        changed_plan = next(
            (
                item
                for item in payload["assetPlans"]
                if isinstance(item, dict) and str(item.get("id")) == asset_plan_id
            ),
            None,
        )
        latest_payload["assetPlans"] = [
            changed_plan
            if isinstance(item, dict) and str(item.get("id")) == asset_plan_id and changed_plan
            else item
            for item in latest_plans
        ]
        latest_payload["updatedAt"] = datetime.now(UTC).isoformat()
        latest_payload["revision"] = latest_project.revision + 1
        latest_canonical = json.dumps(
            latest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        try:
            return store.put_project_and_workspace_revision(
                latest_project.model_copy(update={"revision": latest_project.revision + 1}),
                ProjectWorkspaceRevision(
                    project_id=project_id,
                    revision=latest_project.revision + 1,
                    payload=latest_payload,
                    payload_sha256=hashlib.sha256(latest_canonical).hexdigest(),
                    created_at=datetime.now(UTC),
                ),
            )
        except StoreConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _commit_prompt_result(project_id, commit_latest_image_prompt)


async def _complete_image_prompt(
    client: object,
    messages: tuple[ChatMessage, ...],
) -> str:
    return (await client.complete_text(messages)).strip()  # type: ignore[attr-defined]


def _bound_asset_ids_for_shot(payload: dict[str, Any], shot_id: str) -> set[str]:
    return {
        str(plan["fulfilledByAssetId"])
        for plan in (payload.get("assetPlans") or [])
        if isinstance(plan, dict)
        and plan.get("fulfilledByAssetId")
        and (
            shot_id in {str(value) for value in plan.get("shotIds", [])}
            or (not plan.get("shotIds") and str(plan.get("shotId") or "") == shot_id)
        )
    }


def _asset_is_bound_to_shot(asset: Any, shot_id: str, planned_asset_ids: set[str]) -> bool:
    return (
        asset.asset_id in planned_asset_ids
        or shot_id in asset.shot_ids
        or (not asset.shot_ids and asset.shot_id == shot_id)
    )


async def _generate_and_commit(
    settings: Settings,
    store: SQLiteTaskStore,
    asset_store: ProjectAssetStore,
    project_id: str,
    *,
    only_segment_id: str | None,
) -> ProjectWorkspaceRevision:
    runtime = load_runtime_settings(settings)
    if not runtime.llm_base_url or not runtime.llm_model:
        raise HTTPException(status_code=503, detail="LLM provider is not configured")
    try:
        project = store.get_latest_project_revision(project_id)
        workspace = store.get_latest_project_workspace(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project workspace not found") from exc

    harness_revision = _active_h3_revision(store)
    library = _load_h3_library(harness_revision)
    segments = _workspace_segments(workspace.payload)
    if only_segment_id and all(item["segmentId"] != only_segment_id for item in segments):
        raise HTTPException(status_code=404, detail="H3 segment not found")

    operation_scope = only_segment_id or "all"
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "project_id": project_id,
                "workspace_revision": workspace.revision,
                "segment": operation_scope,
                "harness_revision": harness_revision.revision,
                "harness_manifest_sha256": harness_revision.content_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    try:
        existing = store.get_stage_generation_checkpoint(project_id, "prompts", fingerprint)
        if existing.state == StageGenerationState.SUCCEEDED:
            return store.get_latest_project_workspace(project_id)
        if existing.state == StageGenerationState.FAILED:
            # A completed failed attempt is immutable. A manual retry receives a
            # distinct idempotency key while a crashed STARTED attempt is resumed.
            fingerprint = hashlib.sha256(f"{fingerprint}:retry:{uuid.uuid4()}".encode()).hexdigest()
            existing = None
    except KeyError:
        existing = None
    checkpoint = existing or StageGenerationCheckpoint(
        checkpoint_id=str(uuid.uuid4()),
        project_id=project_id,
        stage="prompts",
        idempotency_key=fingerprint,
        before_workspace_revision=workspace.revision,
        created_at=datetime.now(UTC),
    )
    if existing is None:
        store.put_stage_generation_checkpoint(checkpoint)

    payload = json.loads(json.dumps(workspace.payload))
    prompts = payload.setdefault("prompts", {})
    prior_h3 = prompts.get("h3Prompts") if isinstance(prompts.get("h3Prompts"), list) else []
    locked_by_segment = {
        str(item.get("segmentId")): item
        for item in prior_h3
        if isinstance(item, dict) and item.get("locked") is True
    }
    assets = asset_store.list_assets(project_id)
    generated: dict[str, dict[str, Any]] = {}
    continuity_end_states = {
        str(item.get("segmentId")): str(item["endState"])
        for item in prior_h3
        if isinstance(item, dict)
        and str(item.get("segmentId") or "")
        and str(item.get("endState") or "").strip()
    }
    for revision in store.list_h3_prompt_revisions(project_id):
        if revision.terminal_state:
            continuity_end_states.setdefault(revision.segment_id, revision.terminal_state)

    try:
        remote = remote_config(runtime)
        for segment in segments:
            segment_id = str(segment["segmentId"])
            if only_segment_id and segment_id != only_segment_id:
                continue
            prior_end_state = _continuation_end_state(
                segment,
                continuity_end_states,
                require_for_single_regeneration=only_segment_id is not None,
            )
            if segment_id in locked_by_segment:
                generated[segment_id] = locked_by_segment[segment_id]
                continue
            # Explicit shot bindings are the only source of truth for reference use.
            bound_asset_ids = _bound_asset_ids_for_shot(payload, str(segment["shotId"]))
            relevant_candidates = [
                asset
                for asset in assets
                if asset.state.value == "available"
                and _asset_is_bound_to_shot(asset, str(segment["shotId"]), bound_asset_ids)
            ]
            relevant: list[Any] = []
            media_urls_list: list[str | None] = []
            for asset in relevant_candidates:
                relevant.append(asset)
                if asset.media_kind.value == "image":
                    try:
                        media_urls_list.append(_preview_data_url(asset_store, asset))
                    except (OSError, ProjectAssetNotFoundError):
                        # Keep the asset in the request even when its preview
                        # is unavailable; the execution workflow still mounts
                        # the original blob and must retain the reference ID.
                        media_urls_list.append(None)
                else:
                    media_urls_list.append(None)
            h3_assets_list: list[H3AssetInput] = []
            ordinals = {"image": 0, "video": 0, "audio": 0}
            soundtrack_ordinals = {
                asset.asset_id: index
                for index, asset in enumerate(
                    (
                        item
                        for item in relevant
                        if item.media_kind.value == "video" and item.has_audio
                    ),
                    start=1,
                )
            }
            ordinals["audio"] = len(soundtrack_ordinals)
            label_types = {"image": "Picture", "video": "Video", "audio": "Audio"}
            for asset in relevant:
                media_kind = asset.media_kind.value
                ordinals[media_kind] += 1
                h3_assets_list.append(
                    H3AssetInput(
                        asset_id=asset.asset_id,
                        label=f"<{label_types[media_kind]} {ordinals[media_kind]}>",
                        kind=H3AssetKind(media_kind),
                        companion_audio_label=(
                            f"<Audio {soundtrack_ordinals[asset.asset_id]}>"
                            if asset.asset_id in soundtrack_ordinals
                            else None
                        ),
                        role=_asset_role(asset.kind.value, media_kind=media_kind),
                        preservation=_asset_preservation(
                            asset.name, asset.kind.value, media_kind=media_kind
                        ),
                        preserve_attributes=(asset.kind.value, asset.name),
                        forbidden_propagation_targets=(
                            "face",
                            "skin",
                            "limbs",
                            "other subjects",
                            "background",
                        )
                        if media_kind == "image"
                        else (),
                    )
                )
            request = H3PromptRequest(
                operation_id=str(uuid.uuid4()),
                segment_id=segment_id,
                creative_brief=_creative_brief(payload, segment, prior_end_state),
                duration_seconds=float(segment["durationSeconds"]),
                assets=tuple(h3_assets_list),
                shot_strategy=H3ShotStrategy.AUTO,
                project_memory=_relevant_project_memory(store, project_id, payload, segment),
                highest_instruction=str(payload.get("highestInstruction") or ""),
                constraints=_segment_constraints(segment),
                prior_continuity_state=prior_end_state,
            )
            try:
                director = deterministic_director_decision(request)
                prompt_text = (
                    await complete_text(
                        remote,
                        _plain_h3_messages(
                            request,
                            library,
                            director.mode.value,
                            tuple(media_urls_list),
                        ),
                    )
                ).strip()
                persisted = _persist_plain_h3_prompt(
                    store,
                    project_id,
                    segment_id,
                    harness_revision.revision,
                    harness_revision.content_sha256,
                    request,
                    director.mode,
                    prompt_text,
                )
                generated[segment_id] = _workspace_h3_prompt(segment, persisted, 0)
                continuity_end_states[segment_id] = prompt_text
            except (LLMClientError, ValueError) as exc:
                generated[segment_id] = _failed_workspace_prompt(
                    segment, harness_revision.revision, exception_messages(exc)
                )
    except LLMClientError as exc:
        _fail_checkpoint(store, checkpoint, "llm_unavailable", str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    prompts["h3Prompts"] = [
        generated[str(segment["segmentId"])]
        for segment in segments
        if str(segment["segmentId"]) in generated
    ]
    default_workflow_id, default_harness_revision = _default_image_binding(store)
    prompts["imagePrompts"] = _image_prompt_drafts(
        payload,
        default_workflow_id=default_workflow_id,
        default_harness_revision=default_harness_revision,
    )
    prompts["generatedAt"] = datetime.now(UTC).isoformat()
    failed_segments = [
        item
        for item in prompts["h3Prompts"]
        if isinstance(item, dict)
        and isinstance(item.get("review"), dict)
        and item["review"].get("ready") is not True
    ]
    prompts["generationSummary"] = _h3_generation_summary(prompts["h3Prompts"])
    next_revision = project.revision + 1
    payload["revision"] = next_revision
    payload["updatedAt"] = datetime.now(UTC).isoformat()

    def commit_latest_h3_prompt() -> ProjectWorkspaceRevision:
        # Concurrent prompt requests may have started from the same revision.
        # Merge only the generated prompt entries into the newest workspace so
        # one completed stream cannot erase another completed stream.
        latest_project = store.get_latest_project_revision(project_id)
        latest_workspace = store.get_latest_project_workspace(project_id)
        latest_payload = json.loads(json.dumps(latest_workspace.payload))
        latest_prompts = latest_payload.setdefault("prompts", {})
        latest_h3 = (
            latest_prompts.get("h3Prompts")
            if isinstance(latest_prompts.get("h3Prompts"), list)
            else []
        )
        merged_h3 = _merge_h3_prompt_entries(latest_h3, prompts["h3Prompts"])
        latest_prompts["h3Prompts"] = merged_h3
        latest_images = (
            latest_prompts.get("imagePrompts")
            if isinstance(latest_prompts.get("imagePrompts"), list)
            else []
        )
        generated_images = {
            str(item.get("assetPlanId")): item
            for item in prompts["imagePrompts"]
            if isinstance(item, dict)
        }
        merged_images = [
            generated_images.get(str(item.get("assetPlanId")), item)
            if isinstance(item, dict)
            else item
            for item in latest_images
        ]
        existing_plan_ids = {
            str(item.get("assetPlanId")) for item in merged_images if isinstance(item, dict)
        }
        merged_images.extend(
            item for key, item in generated_images.items() if key not in existing_plan_ids
        )
        latest_prompts["imagePrompts"] = merged_images
        latest_prompts["generatedAt"] = prompts["generatedAt"]
        latest_prompts["generationSummary"] = _h3_generation_summary(merged_h3)
        latest_payload = normalize_workspace_topology(latest_payload)
        latest_payload["revision"] = latest_project.revision + 1
        latest_payload["updatedAt"] = datetime.now(UTC).isoformat()
        latest_canonical = json.dumps(
            latest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        try:
            return store.put_project_and_workspace_revision(
                latest_project.model_copy(update={"revision": latest_project.revision + 1}),
                ProjectWorkspaceRevision(
                    project_id=project_id,
                    revision=latest_project.revision + 1,
                    payload=latest_payload,
                    payload_sha256=hashlib.sha256(latest_canonical).hexdigest(),
                    created_at=datetime.now(UTC),
                ),
            )
        except StoreConflictError as exc:
            _fail_checkpoint(store, checkpoint, "revision_conflict", str(exc))
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    committed = _commit_prompt_result(project_id, commit_latest_h3_prompt)
    checkpoint_state = (
        StageGenerationState.PARTIAL if failed_segments else StageGenerationState.SUCCEEDED
    )
    store.put_stage_generation_checkpoint(
        checkpoint.model_copy(
            update={
                "state": checkpoint_state,
                "after_workspace_revision": next_revision,
                "error_code": "segment_generation_failed" if failed_segments else None,
                "error_message": (
                    f"{len(failed_segments)} H3 segment(s) failed" if failed_segments else None
                ),
                "completed_at": datetime.now(UTC),
            }
        )
    )
    store.add_memory_event(
        ProjectMemoryEvent(
            event_id=str(uuid.uuid4()),
            project_id=project_id,
            kind=MemoryEventKind.TOOL,
            source=DecisionSource.PROJECT_AGENT,
            role="h3_prompt_harness",
            content=(
                f"提示词阶段完成：{len(generated) - len(failed_segments)} 成功，"
                f"{len(failed_segments)} 失败，"
                f"Harness R{harness_revision.revision}。"
            ),
            created_at=datetime.now(UTC),
        )
    )
    return committed


def _incomplete_asset_plan_names(payload: dict[str, Any]) -> tuple[str, ...]:
    asset_plans = payload.get("assetPlans")
    if not isinstance(asset_plans, list):
        return ()
    return tuple(
        str(item.get("name") or item.get("id") or "未命名素材")
        for item in asset_plans
        if isinstance(item, dict) and not item.get("fulfilledByAssetId")
    )


def _relevant_project_memory(
    store: SQLiteTaskStore,
    project_id: str,
    payload: dict[str, Any],
    segment: dict[str, Any],
) -> str:
    """Retrieve a small, segment-specific memory window through SQLite FTS5."""
    search_source = " ".join(
        str(value)
        for value in (
            segment.get("title"),
            segment.get("description"),
            segment.get("shotId"),
            payload.get("idea"),
        )
        if value
    )
    terms = [
        item
        for item in re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,8}", search_source)
        if item.casefold() not in {"segment", "shot"}
    ][:8]
    relevant = ()
    if terms:
        query = " OR ".join(f'"{item}"' for item in dict.fromkeys(terms))
        relevant = store.list_memory_events(project_id, query=query, limit=12)
    recent = store.list_memory_events(project_id, limit=4)
    by_id = {
        event.event_id: event for event in (*relevant, *recent) if event.role != "h3_llm_telemetry"
    }
    ordered = sorted(by_id.values(), key=lambda event: event.created_at)
    return "\n".join(event.content for event in ordered)[-8_000:]


def _load_h3_library(revision) -> H3HarnessLibrary:
    if revision.schema_version != "2.0" or revision.runtime_manifest is None:
        raise HTTPException(
            status_code=409,
            detail="活动 H3 Harness 不是可执行 v2 快照；请重新组装并批准",
        )
    try:
        return H3HarnessLibrary.from_manifest(revision.runtime_manifest)
    except H3PromptHarnessError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _active_h3_revision(store: SQLiteTaskStore):
    revisions = [
        item
        for item in store.list_harness_revisions("h3:default")
        if item.approval == ApprovalState.APPROVED
        and item.schema_version == "2.0"
        and item.runtime_manifest is not None
        and "references/base-en.txt" in item.markdown
        and "references/ref-en.txt" in item.markdown
    ]
    if not revisions:
        raise HTTPException(
            status_code=409,
            detail="完整的 H3 Harness 尚未批准；请重新安装官方来源并组装新修订",
        )
    return revisions[-1]


def _workspace_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    shots = payload.get("shots") if isinstance(payload.get("shots"), list) else []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("id") or "")
        duration = float(shot.get("durationSeconds") or 0)
        if not shot_id or duration <= 0:
            continue
        execution_duration = max(4.0, duration)
        configured = shot.get("motionSegments")
        configured_ids: list[str] = []
        configured_predecessors: list[str | None] = []
        if isinstance(configured, list) and configured:
            durations: list[float] = []
            summaries: list[str] = []
            for index, item in enumerate(configured):
                if not isinstance(item, dict):
                    raise HTTPException(
                        status_code=422,
                        detail=f"分镜 {shot_id} 的 Motion Context 第 {index + 1} 段无效",
                    )
                segment_duration = float(
                    item.get("durationSeconds") or item.get("duration_seconds") or 0
                )
                limit = 15.0 if index == 0 else 12.0
                summary = str(item.get("summary") or item.get("segmentSummary") or "").strip()
                if not 4.0 <= segment_duration <= limit or not summary:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"分镜 {shot_id} 的第 {index + 1} 段必须填写内容，"
                            f"时长须在 4 到 {limit:g} 秒之间"
                        ),
                    )
                durations.append(segment_duration)
                summaries.append(summary)
                configured_ids.append(
                    str(item.get("segmentId") or item.get("segment_id") or item.get("id") or "")
                )
                predecessor = item.get("continuationOf", item.get("continuation_of"))
                configured_predecessors.append(str(predecessor) if predecessor else None)
            if abs(sum(durations) - execution_duration) >= 0.01:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"分镜 {shot_id} 的 Motion Context 分段合计 {sum(durations):g} 秒，"
                        f"必须等于镜头执行时长 {execution_duration:g} 秒"
                    ),
                )
        else:
            # The inherited head counts against H3's sampling limit, so
            # continuation clips expose at most roughly 12 seconds of new footage.
            count = _h3_segment_count(duration)
            piece = round(execution_duration / count, 3)
            durations = [piece] * count
            durations[-1] = round(execution_duration - sum(durations[:-1]), 3)
            summaries = [str(shot.get("summary") or "").strip()] * count
        count = len(durations)
        previous: str | None = None
        for index in range(count):
            configured_id = configured_ids[index] if index < len(configured_ids) else ""
            segment_id = configured_id or f"{shot_id}.C{index + 1:02d}"
            continuation = (
                configured_predecessors[index] or previous
                if index < len(configured_predecessors)
                else previous
            )
            segments.append(
                {
                    "shotId": shot_id,
                    "segmentId": segment_id,
                    "segmentIndex": index,
                    "segmentCount": count,
                    "shotDurationSeconds": duration,
                    "durationSeconds": durations[index],
                    "segmentSummary": summaries[index],
                    "continuationOf": continuation,
                    "seed": int(shot.get("seed") or 0),
                    "shot": shot,
                }
            )
            previous = segment_id
    _validate_workspace_segment_chain(segments)
    return segments


def _validate_workspace_segment_chain(segments: list[dict[str, Any]]) -> None:
    """Validate the normalized segment chain before any LLM or task work."""
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        segment_id = str(segment.get("segmentId") or "")
        shot_id = str(segment.get("shotId") or "")
        if not segment_id or not shot_id:
            raise HTTPException(
                status_code=422,
                detail="Motion Context 分段缺少 segmentId 或 shotId",
            )
        by_shot.setdefault(shot_id, []).append(segment)
    for shot_id, chain in by_shot.items():
        ids = [str(item["segmentId"]) for item in chain]
        if len(ids) != len(set(ids)):
            raise HTTPException(status_code=422, detail=f"分镜 {shot_id} 存在重复 segmentId")
        for index, segment in enumerate(chain):
            expected = None if index == 0 else ids[index - 1]
            actual = segment.get("continuationOf")
            if actual != expected:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"分镜 {shot_id} 的 segment 链断裂："
                        f"{ids[index]} continuationOf 应为 {expected or 'null'}"
                    ),
                )


def _h3_segment_count(duration_seconds: float) -> int:
    if duration_seconds <= 15:
        return 1
    continuation_visible_limit = 12.0
    return max(2, math.ceil(duration_seconds / continuation_visible_limit))


def _creative_brief(
    payload: dict[str, Any],
    segment: dict[str, Any],
    prior_end_state: str | None,
) -> str:
    idea = payload.get("idea") if isinstance(payload.get("idea"), dict) else {}
    shot = segment["shot"]
    long_shot_context = ""
    if segment["segmentCount"] > 1:
        long_shot_context = (
            f"电影分镜总时长：{segment['shotDurationSeconds']} 秒，已规划为 "
            f"{segment['segmentCount']} 个 Motion Context 视频片段，分别编写 "
            "H3 提示词，再通过 Motion Context 按顺序拼接。续段会在模型的15秒预算内"
            "保留至少2秒继承上一段末尾潜空间，并在输出时裁掉，因此续段的新内容不能写满15秒。\n"
        )
    if segment["continuationOf"]:
        continuation_context = "本段必须从上一段尾部状态自然继续，不重复上一段完整事件。"
    elif segment["segmentCount"] > 1:
        continuation_context = "本段是连续链首段，结尾需为下一段留下明确可延续的动作与视听状态。"
    else:
        continuation_context = "本段是独立执行片段，完整描述本段事件及其结束状态。"
    return (
        f"项目：{payload.get('name', '')}\n"
        f"核心创意：{idea.get('concept', '')}\n"
        f"类型与视觉风格：{idea.get('genre', '')}；{idea.get('visualStyle', '')}\n"
        f"电影分镜：{shot.get('title', '')}\n"
        f"画面与事件：{shot.get('summary', '')}\n"
        f"当前分段内容：{segment.get('segmentSummary', '')}\n"
        f"摄影机：{shot.get('camera', '')}\n"
        + long_shot_context
        + f"当前只编写第 {segment['segmentIndex'] + 1}/{segment['segmentCount']} 段，"
        f"本段时长 {segment['durationSeconds']} 秒。\n"
        + (f"上一段结束状态：{prior_end_state}\n" if prior_end_state else "")
        + continuation_context
    )


def _continuation_end_state(
    segment: dict[str, Any],
    end_states: dict[str, str],
    *,
    require_for_single_regeneration: bool,
) -> str | None:
    prior_segment_id = str(segment.get("continuationOf") or "")
    state = end_states.get(prior_segment_id) if prior_segment_id else None
    return state


def _segment_constraints(segment: dict[str, Any]) -> tuple[str, ...]:
    constraints = [
        (
            "执行描述严格使用官方建议的英文；只有对白、歌词和画面中实际可见的文字"
            "保留原语言。保留官方字段名、引用标签、时间戳和控制标记。"
        ),
        (
            "Ref2VA 逐项限定参考素材的职责；保留人物身份不等于复制所有表面纹理，"
            "禁止把服装图案、材质或局部特征传播到脸部、皮肤、肢体、其他主体或背景。"
        ),
        "MiniMax H3 生成原生立体声音频。",
        (
            f"只为当前第 {segment['segmentIndex'] + 1}/{segment['segmentCount']} 段编写一份"
            f"独立、可执行的 {segment['durationSeconds']} 秒 H3 提示词。"
        ),
    ]
    if segment["segmentCount"] > 1:
        constraints.append(
            f"当前电影分镜已规划为 {segment['segmentCount']} 份提示词和视频片段；"
            "最终由 Motion Context 顺序拼接，不得把整条长镜头的事件压入当前提示词。"
            "分段不是固定15+15：续段必须为继承的末尾潜空间预留至少2秒，30秒可以规划为"
            "10+10+10；优先把接缝放在密集信息或关键动作结束之后、人物运动和机位相对稳定处。"
        )
        if segment["continuationOf"]:
            constraints.append(
                "当前是续段：开头必须继承上一段尾部的人物状态、运动方向、机位、光线与"
                "声音，并继续事件而非重演。"
            )
        else:
            constraints.append(
                "当前是连续链首段：结尾必须留下清晰的人物、动作、机位、光线和声音状态，"
                "供下一段 Motion Context 继承。"
            )
    return tuple(constraints)


def _asset_role(kind: str, *, media_kind: str = "image") -> H3AssetPromptRole:
    if media_kind == "video":
        return H3AssetPromptRole.ACTION
    if media_kind == "audio":
        return H3AssetPromptRole.SOUND
    return {
        "character": H3AssetPromptRole.CHARACTER,
        "scene": H3AssetPromptRole.SCENE,
        "prop": H3AssetPromptRole.OBJECT,
        "style": H3AssetPromptRole.STYLE,
        "keyframe": H3AssetPromptRole.REFERENCE,
        "reference": H3AssetPromptRole.REFERENCE,
    }.get(kind, H3AssetPromptRole.REFERENCE)


def _asset_preservation(name: str, kind: str, *, media_kind: str = "image") -> str:
    if media_kind == "video":
        return f"视频“{name}”提供动作、镜头、时间结构及其配对音轨参考。"
    if media_kind == "audio":
        return f"音频“{name}”提供声音、音色、节奏或需要保留的音频内容参考。"
    return f"素材“{name}”作为{kind}参考，保持其可辨识外观、颜色、材质和跨镜头连续性。"


def _preview_data_url(store: ProjectAssetStore, asset) -> str:
    content = store.preview_for(asset.project_id, asset.asset_id).read_bytes()
    mime_type = str(getattr(asset, "mime_type", None) or "image/jpeg")
    return f"data:{mime_type};base64," + base64.b64encode(content).decode("ascii")


def _plain_h3_messages(
    request: H3PromptRequest,
    library: H3HarnessLibrary,
    mode: str,
    media_urls: tuple[str | None, ...],
) -> tuple[ChatMessage, ...]:
    if mode == "ref2va":
        official_guide = library.official_reference
        writer_guide = library.community_reference_writer
    elif mode == "t2va":
        official_guide = library.official_base
        writer_guide = library.community_text_writer
    else:
        official_guide = library.official_base
        writer_guide = library.community_keyframe_writer

    system = "\n\n".join(
        (
            library.official_skill,
            official_guide,
            writer_guide,
            load_harness("h3-direct-writer.md"),
        )
    )
    system = _system_with_highest_instruction(
        system, {"highestInstruction": request.highest_instruction}
    )
    assets = (
        "\n".join(
            f"- {item.label}: {item.kind.value}; role={item.role.value}; "
            f"preserve={item.preservation}"
            for item in request.assets
        )
        or "- None"
    )
    continuity = request.prior_continuity_state or "No prior Motion Context segment."
    user_text = (
        f"Project memory:\n{request.project_memory or 'No additional project memory.'}\n\n"
        f"Current segment ({request.duration_seconds:g}s, {mode}):\n"
        f"{request.creative_brief}\n\n"
        f"Motion Context from the previous segment:\n{continuity}\n\n"
        f"Active reference assets:\n{assets}\n\n"
        f"Constraints:\n{chr(10).join(request.constraints) or 'None'}\n\n"
        "Write the prompt now. Only output the prompt itself and nothing else."
    )
    image_parts = tuple(
        ImageURLContentPart(image_url=ImageURL(url=url, detail="high"))
        for asset, url in zip(request.assets, media_urls, strict=True)
        if asset.kind == H3AssetKind.IMAGE and url
    )
    content: str | tuple[TextContentPart | ImageURLContentPart, ...] = user_text
    if image_parts:
        content = (TextContentPart(text=user_text), *image_parts)
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=content),
    )


def _persist_plain_h3_prompt(
    store: SQLiteTaskStore,
    project_id: str,
    segment_id: str,
    harness_revision: int,
    manifest_sha256: str,
    request: H3PromptRequest,
    mode,
    prompt_text: str,
) -> H3PromptRevision:
    if not prompt_text:
        raise LLMClientError("OpenAI-compatible response message content was empty")
    prompt_id = "h3-" + hashlib.sha256(f"{project_id}:{segment_id}".encode()).hexdigest()[:32]
    history = store.list_h3_prompt_revisions(
        project_id, segment_id=segment_id, include_history=True
    )
    revision = (
        max(
            (item.revision for item in history if item.prompt_revision_id == prompt_id),
            default=0,
        )
        + 1
    )
    review = PromptReviewResult(
        reviewer_harness_id="h3:direct-writer",
        reviewer_harness_revision=harness_revision,
        issues=(),
        structure_complete=True,
        references_complete=True,
        timeline_complete=True,
        contradictions_absent=True,
        audio_consistent=True,
        within_length_limit=True,
        reviewed_at=datetime.now(UTC),
    )
    return store.put_h3_prompt_revision(
        H3PromptRevision(
            prompt_revision_id=prompt_id,
            project_id=project_id,
            revision=revision,
            segment_id=segment_id,
            generation_mode=mode,
            harness_id="h3:direct-writer",
            harness_revision=harness_revision,
            reference_asset_ids=tuple(item.asset_id for item in request.assets),
            execution_prompt=prompt_text,
            harness_manifest_sha256=manifest_sha256,
            route=mode.value,
            asset_role_ledger=tuple(
                {
                    "asset_id": item.asset_id,
                    "label": item.label,
                    "kind": item.kind.value,
                    "role": item.role.value,
                }
                for item in request.assets
            ),
            assumptions=(),
            stage_trace=({"stage": "direct_writer", "status": "succeeded", "attempt": 0},),
            terminal_state=_prompt_terminal_context(prompt_text),
            review=review,
            state=PromptRevisionState.DRAFT,
            created_at=datetime.now(UTC),
        )
    )


def _prompt_terminal_context(prompt_text: str) -> str:
    """Keep the ending context for continuation without duplicating the full prompt."""
    paragraphs = [value.strip() for value in prompt_text.split("\n\n") if value.strip()]
    selected: list[str] = []
    length = 0
    for paragraph in reversed(paragraphs):
        added = len(paragraph) + (2 if selected else 0)
        if selected and length + added > 4000:
            break
        selected.append(paragraph)
        length += added
        if length >= 2000:
            break
    context = "\n\n".join(reversed(selected)) or prompt_text
    return context[-4000:]


def _persist_h3_result(
    store: SQLiteTaskStore,
    project_id: str,
    segment_id: str,
    harness_revision: int,
    result,
) -> H3PromptRevision:
    prompt_id = "h3-" + hashlib.sha256(f"{project_id}:{segment_id}".encode()).hexdigest()[:32]
    history = store.list_h3_prompt_revisions(
        project_id, segment_id=segment_id, include_history=True
    )
    revision = (
        max(
            (item.revision for item in history if item.prompt_revision_id == prompt_id),
            default=0,
        )
        + 1
    )
    # Prompt authoring uses the deterministic validator as its quality gate;
    # the optional LLM reviewer is intentionally not called on this path.
    findings = ()
    review = PromptReviewResult(
        reviewer_harness_id="h3:default",
        reviewer_harness_revision=harness_revision,
        issues=tuple(
            PromptReviewIssue(
                code=item.code,
                severity=PromptIssueSeverity(item.severity.value),
                message=item.message,
            )
            for item in findings
        ),
        structure_complete=True,
        references_complete=True,
        timeline_complete=True,
        contradictions_absent=True,
        audio_consistent=True,
        within_length_limit=len(result.execution_prompt) <= 7000,
        reviewed_at=datetime.now(UTC),
    )
    stored = H3PromptRevision(
        prompt_revision_id=prompt_id,
        project_id=project_id,
        revision=revision,
        segment_id=segment_id,
        generation_mode=result.director.mode,
        harness_id="h3:default",
        harness_revision=harness_revision,
        reference_asset_ids=tuple(item.asset_id for item in result.request.assets),
        execution_prompt=result.execution_prompt,
        harness_manifest_sha256=result.harness_manifest_sha256,
        route=result.director.mode.value,
        asset_role_ledger=tuple(
            {
                "asset_id": item.asset_id,
                "label": item.label,
                "kind": item.kind.value,
                "role": item.role.value,
                "preservation": item.preservation,
                "preserve_attributes": item.preserve_attributes,
                "mutable_attributes": item.mutable_attributes,
                "forbidden_propagation_targets": item.forbidden_propagation_targets,
                "active_shots": item.active_shots,
            }
            for item in result.request.assets
        ),
        assumptions=result.assumptions,
        stage_trace=tuple(item.model_dump(mode="json") for item in result.stage_trace),
        terminal_state=result.candidate.timeline[-1].end_state,
        review=review,
        state=PromptRevisionState.DRAFT,
        created_at=datetime.now(UTC),
    )
    return store.put_h3_prompt_revision(stored)


def _workspace_h3_prompt(
    segment: dict[str, Any], revision: H3PromptRevision, repair_passes: int
) -> dict[str, Any]:
    return {
        "id": revision.prompt_revision_id,
        "shotId": segment["shotId"],
        "segmentId": segment["segmentId"],
        "segmentIndex": segment["segmentIndex"],
        "durationSeconds": segment["durationSeconds"],
        "continuationOf": segment["continuationOf"],
        "inputMode": revision.generation_mode.value,
        "prompt": revision.execution_prompt,
        "endState": revision.terminal_state,
        "assetIds": list(revision.reference_asset_ids),
        "seed": segment["seed"],
        "harnessRevision": revision.harness_revision,
        "harnessManifestSha256": revision.harness_manifest_sha256,
        "route": revision.route,
        "assetRoles": list(revision.asset_role_ledger),
        "assumptions": list(revision.assumptions),
        "stageTrace": list(revision.stage_trace),
        "locked": revision.locked,
        "revision": revision.revision,
        "legacy": False,
        "review": {
            "ready": revision.review.execution_ready,
            "issues": [item.model_dump(mode="json") for item in revision.review.issues]
            + (
                [
                    {
                        "severity": "warning",
                        "code": "repaired",
                        "message": f"Reviewer 自动修复 {repair_passes} 次",
                    }
                ]
                if repair_passes
                else []
            ),
            "reviewedAt": revision.review.reviewed_at.isoformat(),
        },
    }


def _failed_workspace_prompt(
    segment: dict[str, Any], harness_revision: int, errors: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "id": "failed-" + hashlib.sha256(str(segment["segmentId"]).encode()).hexdigest()[:24],
        "shotId": segment["shotId"],
        "segmentId": segment["segmentId"],
        "segmentIndex": segment["segmentIndex"],
        "durationSeconds": segment["durationSeconds"],
        "continuationOf": segment["continuationOf"],
        "inputMode": "t2va",
        "prompt": "",
        "endState": None,
        "assetIds": [],
        "seed": segment["seed"],
        "harnessRevision": harness_revision,
        "locked": False,
        "revision": 1,
        "legacy": False,
        "review": {
            "ready": False,
            "issues": [
                {"severity": "error", "code": "harness_failed", "message": message}
                for message in errors
            ],
            "reviewedAt": datetime.now(UTC).isoformat(),
        },
    }


def _image_workflow_binding(
    store: SQLiteTaskStore,
    requested_workflow_id: str | None = None,
) -> tuple[str | None, int | None]:
    workflows = {
        item.template_id: item
        for item in store.list_workflow_revisions()
        if item.approval.value == "approved" and not item.template_id.startswith("builtin:")
    }
    if requested_workflow_id and requested_workflow_id not in workflows:
        raise HTTPException(status_code=409, detail="选择的图片工作流未批准或不存在")
    workflow_id = requested_workflow_id or (sorted(workflows)[0] if workflows else None)
    if workflow_id is None:
        return None, None

    harness_revisions: list[int] = []
    for bundle in store.list_harness_bundles():
        if bundle.purpose != "image_prompting" or bundle.workflow_template_id != workflow_id:
            continue
        approved = [
            revision
            for revision in store.list_harness_revisions(bundle.harness_id)
            if revision.approval == ApprovalState.APPROVED
        ]
        if approved:
            harness_revisions.append(approved[-1].revision)
    return workflow_id, max(harness_revisions) if harness_revisions else None


def _default_image_binding(store: SQLiteTaskStore) -> tuple[str | None, int | None]:
    return _image_workflow_binding(store)


def _image_prompt_drafts(
    payload: dict[str, Any],
    *,
    default_workflow_id: str | None = None,
    default_harness_revision: int | None = None,
) -> list[dict[str, Any]]:
    plans = payload.get("assetPlans") if isinstance(payload.get("assetPlans"), list) else []
    existing = payload.get("prompts") if isinstance(payload.get("prompts"), dict) else {}
    old = existing.get("imagePrompts") if isinstance(existing.get("imagePrompts"), list) else []
    old_by_plan = {str(item.get("assetPlanId")): item for item in old if isinstance(item, dict)}
    result: list[dict[str, Any]] = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        plan_id = str(plan.get("id") or plan.get("plan_id") or "")
        if not plan_id:
            continue
        if plan.get("fulfilledByAssetId"):
            if plan_id in old_by_plan:
                result.append(old_by_plan[plan_id])
            continue
        if plan_id in old_by_plan and old_by_plan[plan_id].get("locked"):
            result.append(old_by_plan[plan_id])
            continue
        description = str(plan.get("description") or "").strip()
        result.append(
            {
                "id": f"image-{hashlib.sha256(plan_id.encode()).hexdigest()[:24]}",
                "assetPlanId": plan_id,
                "prompt": description,
                "negativePrompt": "低清晰度，错误结构，多余肢体，文字水印，画面污染",
                "workflowTemplateId": default_workflow_id,
                "harnessRevision": default_harness_revision,
                "referenceAssetIds": [],
                "locked": False,
                "revision": int(old_by_plan.get(plan_id, {}).get("revision") or 0) + 1,
            }
        )
    return result


def _fail_checkpoint(
    store: SQLiteTaskStore,
    checkpoint: StageGenerationCheckpoint,
    code: str,
    message: str,
) -> None:
    store.put_stage_generation_checkpoint(
        checkpoint.model_copy(
            update={
                "state": StageGenerationState.FAILED,
                "error_code": code,
                "error_message": message,
                "completed_at": datetime.now(UTC),
            }
        )
    )
