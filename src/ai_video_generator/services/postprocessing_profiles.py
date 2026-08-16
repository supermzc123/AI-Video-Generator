from __future__ import annotations

from typing import Any

from ai_video_generator.domain.postprocessing import (
    PostProcessKind,
    PostProcessProfile,
    PostProcessProfileCapability,
)

BUILTIN_POSTPROCESS_PROFILES = (
    PostProcessProfile(
        profile_id="seedvr2:official-video",
        revision=1,
        name="SeedVR2 官方视频修复",
        kind=PostProcessKind.RESTORATION,
        engine="seedvr2",
        required_node_types=(
            "UNETLoader",
            "VAELoader",
            "SeedVR2Conditioning",
            "SeedVR2Preprocess",
            "SeedVR2PostProcessing",
            "SaveVideo",
        ),
        model_node_type="UNETLoader",
        model_input_name="unet_name",
        auxiliary_model_node_type="VAELoader",
        auxiliary_model_input_name="vae_name",
    ),
    PostProcessProfile(
        profile_id="interpolation:rife",
        revision=1,
        name="RIFE 插帧",
        kind=PostProcessKind.INTERPOLATION,
        engine="rife",
        required_node_types=("VHS_LoadVideo", "RIFE VFI", "VHS_VideoCombine"),
        model_node_type="RIFE VFI",
        model_input_name="ckpt_name",
        supported_target_fps=(48, 60, 120),
        workflow_file="rife49_2x_api.json",
    ),
    PostProcessProfile(
        profile_id="interpolation:gimm-vfi",
        revision=1,
        name="GIMM-VFI 插帧",
        kind=PostProcessKind.INTERPOLATION,
        engine="gimm-vfi",
        required_node_types=(
            "VHS_LoadVideo",
            "DownloadAndLoadGIMMVFIModel",
            "GIMMVFI_interpolate",
            "VHS_VideoCombine",
        ),
        model_node_type="DownloadAndLoadGIMMVFIModel",
        model_input_name="model",
        supported_target_fps=(48, 60, 120),
    ),
    PostProcessProfile(
        profile_id="transcription:faster-whisper",
        revision=1,
        name="Faster Whisper 转写",
        kind=PostProcessKind.TRANSCRIPTION,
        engine="faster-whisper",
    ),
)


WHISPER_MODELS = (
    "large-v3-turbo",
    "large-v3",
    "medium",
    "small",
)


def _choices(
    nodes: dict[str, Any], node_type: str | None, input_name: str | None
) -> tuple[str, ...]:
    if not node_type or not input_name:
        return ()
    definition = nodes.get(node_type, {}).get("input", {}).get("required", {}).get(input_name)
    if isinstance(definition, list) and definition and isinstance(definition[0], list):
        return tuple(sorted({str(value) for value in definition[0] if str(value)}))
    return ()


def inspect_postprocess_profiles(
    nodes: dict[str, Any], *, server_online: bool
) -> tuple[PostProcessProfileCapability, ...]:
    result: list[PostProcessProfileCapability] = []
    for profile in BUILTIN_POSTPROCESS_PROFILES:
        if profile.kind == PostProcessKind.TRANSCRIPTION:
            result.append(
                PostProcessProfileCapability(
                    profile=profile,
                    available=True,
                    models=WHISPER_MODELS,
                )
            )
            continue
        missing = tuple(sorted(set(profile.required_node_types) - set(nodes)))
        models = _choices(nodes, profile.model_node_type, profile.model_input_name)
        auxiliary = _choices(
            nodes,
            profile.auxiliary_model_node_type,
            profile.auxiliary_model_input_name,
        )
        blockers = list(f"缺少节点：{value}" for value in missing)
        if profile.kind != PostProcessKind.TRANSCRIPTION and not profile.workflow_file:
            blockers.append("此 Profile 尚未提供可执行的 ComfyUI API 工作流")
        if not models:
            blockers.append("未发现可选择的模型")
        if profile.auxiliary_model_node_type and not auxiliary:
            blockers.append("未发现可选择的辅助模型")
        if not server_online:
            blockers.insert(0, "ComfyUI服务离线")
        result.append(
            PostProcessProfileCapability(
                profile=profile,
                available=not blockers,
                models=models,
                auxiliary_models=auxiliary,
                blockers=tuple(blockers),
            )
        )
    return tuple(result)
