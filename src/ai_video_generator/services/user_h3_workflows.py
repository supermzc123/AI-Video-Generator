from __future__ import annotations

import hashlib
import json
from typing import Any

from ai_video_generator.domain import (
    BindingSemantic,
    ComfyUIOutput,
    TaskKind,
    TaskWorkloadManifest,
    WorkflowOutputType,
    WorkflowTemplate,
    WorkloadBlob,
)
from ai_video_generator.workers.workflow import compile_workflow

H3_TEMPLATE_REQUIREMENTS = {
    "h3_conditioning": frozenset(
        {
            BindingSemantic.PROMPT,
            BindingSemantic.WIDTH,
            BindingSemantic.HEIGHT,
            BindingSemantic.FRAME_COUNT,
            BindingSemantic.CONDITIONING_FINGERPRINT,
        }
    ),
    "h3_diffusion": frozenset(
        {
            BindingSemantic.SEED,
            BindingSemantic.CONDITIONING_FINGERPRINT,
            BindingSemantic.OUTPUT_PREFIX,
            BindingSemantic.MOTION_CONTEXT_INPUT,
            BindingSemantic.MOTION_CONTEXT_OUTPUT_PREFIX,
        }
    ),
}


def validate_h3_template_contract(template: WorkflowTemplate) -> tuple[str, ...]:
    required = H3_TEMPLATE_REQUIREMENTS.get(template.kind)
    if required is None:
        return ()
    semantics = {binding.semantic for binding in template.bindings}
    missing = sorted(item.value for item in required - semantics)
    issues = [f"缺少标准绑定：{value}" for value in missing]
    expected_output = (
        WorkflowOutputType.CONDITIONING
        if template.kind == "h3_conditioning"
        else WorkflowOutputType.VIDEO
    )
    if not any(output.output_type == expected_output for output in template.outputs):
        issues.append(f"必须声明一个 {expected_output.value} 输出")
    return tuple(issues)


def compile_user_h3_segment_manifests(
    *,
    conditioning_template: WorkflowTemplate,
    diffusion_template: WorkflowTemplate,
    project_id: str,
    prompt: dict[str, Any],
    width: int,
    height: int,
    node_schema_sha256: str,
    asset_blobs: tuple[tuple[str, str, str, str], ...] = (),
) -> tuple[TaskWorkloadManifest, TaskWorkloadManifest]:
    """Compile an approved two-part H3 kit through the generic binding engine."""
    for template in (conditioning_template, diffusion_template):
        issues = validate_h3_template_contract(template)
        if issues:
            raise ValueError("; ".join(issues))
    segment_id = str(prompt["segmentId"])
    duration = float(prompt["durationSeconds"])
    visible_frames = max(5, round(duration * 24))
    context_frames = 56 if prompt.get("continuationOf") else 0
    frame_count = visible_frames + context_frames
    frame_count += (5 - frame_count % 17) % 17
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "prompt": prompt.get("prompt", ""),
                "width": width,
                "height": height,
                "frames": frame_count,
                "conditioning_template": [
                    conditioning_template.template_id,
                    conditioning_template.revision,
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    def values(
        template: WorkflowTemplate, source: dict[BindingSemantic, object]
    ) -> dict[str, object]:
        return {
            binding.binding_id: source[binding.semantic]
            for binding in template.bindings
            if binding.semantic in source
        }

    shared: dict[BindingSemantic, object] = {
        BindingSemantic.PROMPT: str(prompt.get("prompt") or ""),
        BindingSemantic.WIDTH: width,
        BindingSemantic.HEIGHT: height,
        BindingSemantic.FRAME_COUNT: frame_count,
        BindingSemantic.CONDITIONING_FINGERPRINT: fingerprint,
        BindingSemantic.SEED: int(prompt.get("seed") or 0),
        BindingSemantic.OUTPUT_PREFIX: f"ai-video-generator/{project_id}/{segment_id}",
        BindingSemantic.MOTION_CONTEXT_INPUT: (
            f"ai-video-generator/{project_id}/motion/"
            f"{prompt.get('continuationOf')}_00001.safetensors"
            if prompt.get("continuationOf")
            else ""
        ),
        BindingSemantic.MOTION_CONTEXT_OUTPUT_PREFIX: (
            f"ai-video-generator/{project_id}/motion/{segment_id}"
        ),
    }
    media_paths: dict[tuple[BindingSemantic, int], str] = {}
    input_blobs: list[WorkloadBlob] = []
    counters = {"image": 0, "video": 0, "audio": 0}
    semantic_for = {
        "image": BindingSemantic.REFERENCE_IMAGE,
        "video": BindingSemantic.REFERENCE_VIDEO,
        "audio": BindingSemantic.REFERENCE_AUDIO,
    }
    for asset_id, sha256_value, suffix, media_type in asset_blobs:
        media_kind = media_type.split("/", 1)[0]
        if media_kind not in counters:
            continue
        counters[media_kind] += 1
        mount_path = f"ai-video-generator/{project_id}/{sha256_value}{suffix}"
        media_paths[(semantic_for[media_kind], counters[media_kind])] = mount_path
        input_blobs.append(
            WorkloadBlob(
                sha256=sha256_value,
                media_type=media_type,
                mount_path=mount_path,
                role=f"reference:{asset_id}",
            )
        )

    def template_values(template: WorkflowTemplate) -> dict[str, object]:
        result = values(template, shared)
        for binding in template.bindings:
            key = (binding.semantic, binding.reference_index or 0)
            if key in media_paths:
                result[binding.binding_id] = media_paths[key]
        return result
    encode = compile_workflow(conditioning_template, template_values(conditioning_template))
    diffusion = compile_workflow(diffusion_template, template_values(diffusion_template))
    context = {
        "project_id": project_id,
        "segment_id": segment_id,
        "prompt_text": str(prompt.get("prompt") or ""),
        "conditioning_fingerprint": fingerprint,
        "visible_frames": str(visible_frames),
        "motion_context_frames": str(context_frames),
        "sample_frames": str(frame_count),
        "continuation_of": str(prompt.get("continuationOf") or ""),
    }
    encode_output = next(
        output for output in conditioning_template.outputs
        if output.output_type == WorkflowOutputType.CONDITIONING
    )
    video_output = next(
        output for output in diffusion_template.outputs
        if output.output_type == WorkflowOutputType.VIDEO
    )
    return (
        TaskWorkloadManifest(
            task_kind=TaskKind.CONDITIONING_ENCODING,
            workflow_sha256=encode.workflow_sha256,
            node_schema_sha256=node_schema_sha256,
            prompt=encode.workflow,
            input_blobs=tuple(input_blobs),
            outputs=(ComfyUIOutput(node_id=encode_output.node_id, media_type="application/json"),),
            required_node_types=conditioning_template.required_node_types,
            workflow_template_id=conditioning_template.template_id,
            context=context,
        ),
        TaskWorkloadManifest(
            task_kind=TaskKind.H3_GENERATION,
            workflow_sha256=diffusion.workflow_sha256,
            node_schema_sha256=node_schema_sha256,
            prompt=diffusion.workflow,
            input_blobs=tuple(input_blobs),
            outputs=(ComfyUIOutput(node_id=video_output.node_id, media_type="video/mp4"),),
            required_node_types=diffusion_template.required_node_types,
            workflow_template_id=diffusion_template.template_id,
            context=context,
        ),
    )
