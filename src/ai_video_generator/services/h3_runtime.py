from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from ai_video_generator.config import Settings
from ai_video_generator.domain import ComfyUIOutput, TaskKind, TaskWorkloadManifest, WorkloadBlob
from ai_video_generator.workers.h3_conditioning import compile_h3_static_encode_workflow
from ai_video_generator.workers.workflow import canonical_json_sha256, parse_api_workflow

RESOURCE_ROOT = Path(__file__).resolve().parent.parent / "resources" / "h3"
H3_NATIVE_FPS = 24
H3_MAX_SAMPLE_SECONDS = 15
MOTION_CONTEXT_FRAMES = 56
AssetBlob = tuple[str, str, str] | tuple[str, str, str, str]


def compile_h3_segment_manifests(
    *,
    settings: Settings,
    project_id: str,
    prompt: dict[str, Any],
    width: int,
    height: int,
    asset_blobs: tuple[AssetBlob, ...],
    node_schema_sha256: str,
) -> tuple[TaskWorkloadManifest, TaskWorkloadManifest]:
    """Compile the tested H3 cache/diffusion graphs with typed deployment values."""
    segment_id = str(prompt["segmentId"])
    duration = float(prompt["durationSeconds"])
    continuation = bool(prompt.get("continuationOf"))
    visible_frames = max(5, round(duration * H3_NATIVE_FPS))
    context_frames = MOTION_CONTEXT_FRAMES if continuation else 0
    frame_count = visible_frames + context_frames
    frame_count += (5 - frame_count % 17) % 17
    if frame_count > H3_MAX_SAMPLE_SECONDS * H3_NATIVE_FPS + 2:
        raise ValueError(
            "H3 segment exceeds the 15-second sampling budget after Motion Context reserve"
        )
    fingerprint_payload = {
        "prompt": prompt.get("prompt", ""),
        "assets": [_asset_blob_parts(item)[1] for item in asset_blobs],
        "width": width,
        "height": height,
        "frames": frame_count,
        "encoder": settings.h3_text_encoder,
        "video_vae": settings.h3_video_vae,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    source_name = "multi-reference-source.api.json" if asset_blobs else "conditioning.api.json"
    source = _load(source_name)
    input_blobs: list[WorkloadBlob] = []
    if asset_blobs:
        source = _configure_reference_source(
            source,
            project_id=project_id,
            prompt_text=str(prompt.get("prompt") or ""),
            width=width,
            height=height,
            frames=frame_count,
            settings=settings,
            asset_blobs=asset_blobs,
            workload_blobs=input_blobs,
        )
        conditioning_node_id = "131"
        encode = compile_h3_static_encode_workflow(
            source,
            conditioning_node_id=conditioning_node_id,
            fingerprint=fingerprint,
        )
    else:
        _configure_conditioning_source(
            source,
            prompt_text=str(prompt.get("prompt") or ""),
            width=width,
            height=height,
            frames=frame_count,
            settings=settings,
        )
        source["avg-save-h3-static"]["inputs"]["fingerprint"] = fingerprint
        encode = source
    encode_output_id = next(
        node_id
        for node_id, node in encode.items()
        if node["class_type"] == "AVGSaveH3StaticConditioning"
    )

    diffusion = _load("continuation.api.json" if continuation else "initial.api.json")
    _configure_diffusion(
        diffusion,
        settings=settings,
        fingerprint=fingerprint,
        seed=int(prompt.get("seed") or 0),
        project_id=project_id,
        segment_id=segment_id,
        continuation_of=str(prompt.get("continuationOf") or ""),
    )
    common_context = {
        "project_id": project_id,
        "segment_id": segment_id,
        "conditioning_fingerprint": fingerprint,
        "visible_frames": str(visible_frames),
        "motion_context_frames": str(context_frames),
        "sample_frames": str(frame_count),
    }
    encode_manifest = TaskWorkloadManifest(
        task_kind=TaskKind.CONDITIONING_ENCODING,
        workflow_sha256=canonical_json_sha256(encode),
        node_schema_sha256=node_schema_sha256,
        prompt=encode,
        input_blobs=tuple(input_blobs),
        outputs=(ComfyUIOutput(node_id=encode_output_id, media_type="application/json"),),
        required_node_types=tuple(sorted({str(node["class_type"]) for node in encode.values()})),
        workflow_template_id="h3:controlled-v1:conditioning",
        context=common_context,
    )
    output_node = "92"
    diffusion_manifest = TaskWorkloadManifest(
        task_kind=TaskKind.H3_GENERATION,
        workflow_sha256=canonical_json_sha256(diffusion),
        node_schema_sha256=node_schema_sha256,
        prompt=diffusion,
        outputs=(ComfyUIOutput(node_id=output_node, media_type="video/mp4"),),
        required_node_types=tuple(sorted({str(node["class_type"]) for node in diffusion.values()})),
        workflow_template_id=(
            "h3:controlled-v1:continuation" if continuation else "h3:controlled-v1:initial"
        ),
        context={**common_context, "continuation_of": str(prompt.get("continuationOf") or "")},
    )
    return encode_manifest, diffusion_manifest


def _load(name: str) -> dict[str, dict[str, Any]]:
    return parse_api_workflow(json.loads((RESOURCE_ROOT / name).read_text(encoding="utf-8")))


def _configure_conditioning_source(
    workflow: dict[str, dict[str, Any]],
    *,
    prompt_text: str,
    width: int,
    height: int,
    frames: int,
    settings: Settings,
) -> None:
    workflow["128"]["inputs"]["clip_name"] = settings.h3_text_encoder
    workflow["119"]["inputs"]["vae_name"] = settings.h3_video_vae
    workflow["131"]["inputs"].update(
        {"prompt": prompt_text, "width": width, "height": height, "length": frames}
    )


def _configure_reference_source(
    workflow: dict[str, dict[str, Any]],
    *,
    project_id: str,
    prompt_text: str,
    width: int,
    height: int,
    frames: int,
    settings: Settings,
    asset_blobs: tuple[AssetBlob, ...],
    workload_blobs: list[WorkloadBlob],
) -> dict[str, dict[str, Any]]:
    configured = copy.deepcopy(workflow)
    configured["128"]["inputs"]["clip_name"] = settings.h3_text_encoder
    configured["119"]["inputs"]["vae_name"] = settings.h3_video_vae
    configured["120"]["inputs"]["vae_name"] = settings.h3_audio_vae
    inputs = configured["131"]["inputs"]
    inputs.update({"prompt": prompt_text, "width": width, "height": height, "length": frames})
    reference_prefixes = (
        "ref_images.",
        "ref_videos.",
        "ref_video_audios.",
        "ref_audios.",
    )
    for input_name in tuple(inputs):
        if input_name.startswith(reference_prefixes):
            inputs.pop(input_name)
    for node_id in tuple(configured):
        if node_id.startswith("ref-"):
            configured.pop(node_id)
    counts = {"image": 0, "video": 0, "audio": 0}
    limits = {"image": 9, "video": 3, "audio": 3}
    for asset_blob in asset_blobs:
        asset_id, sha256_value, suffix, media_type = _asset_blob_parts(asset_blob)
        media_kind = media_type.split("/", 1)[0]
        if media_kind not in counts:
            raise ValueError(f"unsupported H3 reference media type: {media_type}")
        index = counts[media_kind]
        if index >= limits[media_kind]:
            raise ValueError(f"H3 supports at most {limits[media_kind]} {media_kind} references")
        counts[media_kind] += 1
        node_id = f"avg-ref-{index}" if media_kind == "image" else f"avg-ref-{media_kind}-{index}"
        mount_path = f"ai-video-generator/{project_id}/{sha256_value}{suffix}"
        if media_kind == "image":
            configured[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": mount_path},
                "_meta": {"title": f"AVG_PICTURE_REFERENCE_{index + 1}"},
            }
            inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        elif media_kind == "video":
            components_id = f"{node_id}-components"
            configured[node_id] = {
                "class_type": "LoadVideo",
                "inputs": {"file": mount_path},
                "_meta": {"title": f"AVG_VIDEO_REFERENCE_{index + 1}"},
            }
            configured[components_id] = {
                "class_type": "GetVideoComponents",
                "inputs": {"video": [node_id, 0]},
                "_meta": {"title": f"AVG_VIDEO_COMPONENTS_{index + 1}"},
            }
            inputs[f"ref_videos.ref_video_{index}"] = [components_id, 0]
            inputs[f"ref_video_audios.ref_video_audio_{index}"] = [components_id, 1]
        else:
            configured[node_id] = {
                "class_type": "LoadAudio",
                "inputs": {"audio": mount_path},
                "_meta": {"title": f"AVG_AUDIO_REFERENCE_{index + 1}"},
            }
            inputs[f"ref_audios.ref_audio_{index}"] = [node_id, 0]
        workload_blobs.append(
            WorkloadBlob(
                sha256=sha256_value,
                media_type=media_type,
                mount_path=mount_path,
                role=f"reference:{asset_id}",
            )
        )
    return configured


def _asset_blob_parts(asset_blob: AssetBlob) -> tuple[str, str, str, str]:
    if len(asset_blob) == 3:
        asset_id, sha256_value, suffix = asset_blob
        return asset_id, sha256_value, suffix, "image/*"
    return asset_blob


def _configure_diffusion(
    workflow: dict[str, dict[str, Any]],
    *,
    settings: Settings,
    fingerprint: str,
    seed: int,
    project_id: str,
    segment_id: str,
    continuation_of: str,
) -> None:
    workflow["127"]["inputs"]["unet_name"] = settings.h3_diffusion_model
    workflow["119"]["inputs"]["vae_name"] = settings.h3_video_vae
    workflow["120"]["inputs"]["vae_name"] = settings.h3_audio_vae
    workflow["124"]["inputs"]["steps"] = settings.h3_steps
    workflow["129"]["inputs"]["noise_seed"] = seed
    workflow["avg-load-h3-static"]["inputs"]["fingerprint"] = fingerprint
    workflow["92"]["inputs"]["filename_prefix"] = (
        f"ai-video-generator/{project_id}/{segment_id}"
    )
    if settings.h3_turbo_enabled:
        workflow["134"]["inputs"].update(
            {
                "lora_name": settings.h3_turbo_lora,
                "low_vram": settings.h3_low_vram,
            }
        )
        base_model = ["134", 0]
    else:
        # The official non-Turbo H3 graph uses the base model directly and a
        # stock Euler sampler. Removing both Turbo nodes is necessary: merely
        # setting LoRA strength to zero still leaves a Turbo-only dependency.
        workflow.pop("134", None)
        workflow["135"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
            "_meta": {"title": "H3 standard Euler sampler"},
        }
        base_model = ["127", 0]
    if settings.h3_sage_attention_enabled:
        workflow["136"]["inputs"]["model"] = base_model
    else:
        workflow["126"]["inputs"]["model"] = base_model
        workflow["124"]["inputs"]["model"] = base_model
        workflow.pop("136", None)
    if "mc-apply" in workflow:
        workflow["mc-apply"]["inputs"]["context_length"] = str(MOTION_CONTEXT_FRAMES)
    if "mc-save-1" in workflow:
        workflow["mc-save-1"]["inputs"]["filename_prefix"] = (
            f"ai-video-generator/{project_id}/motion/{segment_id}"
        )
        workflow["mc-save-1"]["inputs"]["clip_index"] = 1
    if "mc-save-2" in workflow:
        workflow["mc-save-2"]["inputs"]["filename_prefix"] = (
            f"ai-video-generator/{project_id}/motion/{segment_id}"
        )
        workflow["mc-save-2"]["inputs"]["clip_index"] = 1
    if "mc-load-1" in workflow:
        if not continuation_of:
            raise ValueError("continuation workflow requires predecessor segment identity")
        workflow["mc-load-1"]["inputs"].update(
            {
                "latent_path": (
                    f"ai-video-generator/{project_id}/motion/"
                    f"{continuation_of}_00001.safetensors"
                ),
                "clip_index": 0,
            }
        )
