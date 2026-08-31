from ai_video_generator.config import Settings
from ai_video_generator.services.h3_runtime import compile_h3_segment_manifests


def test_controlled_h3_runtime_compiles_static_and_diffusion_manifests() -> None:
    settings = Settings(_env_file=None)
    encode, diffusion = compile_h3_segment_manifests(
        settings=settings,
        project_id="project",
        prompt={
            "segmentId": "segment-1",
            "durationSeconds": 4,
            "prompt": "完整中文 H3 提示词",
            "seed": 42,
            "assetIds": [],
            "continuationOf": None,
        },
        width=352,
        height=640,
        asset_blobs=(),
        node_schema_sha256="1" * 64,
    )

    assert encode.outputs[0].node_id in encode.prompt
    assert encode.context["conditioning_fingerprint"] == diffusion.context[
        "conditioning_fingerprint"
    ]
    assert encode.context["prompt_text"] == "完整中文 H3 提示词"
    assert diffusion.context["prompt_text"] == "完整中文 H3 提示词"
    assert diffusion.prompt["127"]["inputs"]["unet_name"] == settings.h3_diffusion_model
    assert diffusion.prompt["124"]["inputs"]["steps"] == settings.h3_steps


def test_single_reference_rebuilds_template_reference_inputs() -> None:
    settings = Settings(_env_file=None)
    encode, _diffusion = compile_h3_segment_manifests(
        settings=settings,
        project_id="project",
        prompt={
            "segmentId": "segment-1",
            "durationSeconds": 4,
            "prompt": "完整中文 Ref2VA 提示词",
            "seed": 42,
            "assetIds": ["asset-1"],
            "continuationOf": None,
        },
        width=352,
        height=640,
        asset_blobs=(("asset-1", "a" * 64, ".png"),),
        node_schema_sha256="1" * 64,
    )

    inputs = encode.prompt["131"]["inputs"]
    reference_inputs = {
        name: value
        for name, value in inputs.items()
        if name.startswith("ref_images.ref_image_")
    }
    assert reference_inputs == {"ref_images.ref_image_0": ["avg-ref-0", 0]}
    assert "avg-ref-0" in encode.prompt
    assert "ref-2" not in encode.prompt


def test_video_and_audio_references_use_independent_inputs_and_limits() -> None:
    settings = Settings(_env_file=None)
    encode, _diffusion = compile_h3_segment_manifests(
        settings=settings,
        project_id="project",
        prompt={
            "segmentId": "segment-1",
            "durationSeconds": 4,
            "prompt": "English Ref2VA prompt using <Video 1> and <Audio 2>.",
            "seed": 42,
            "assetIds": ["video-1", "audio-1"],
            "continuationOf": None,
        },
        width=352,
        height=640,
        asset_blobs=(
            ("video-1", "b" * 64, ".mp4", "video/mp4"),
            ("audio-1", "c" * 64, ".wav", "audio/wav"),
        ),
        node_schema_sha256="1" * 64,
    )

    inputs = encode.prompt["131"]["inputs"]
    assert inputs["ref_videos.ref_video_0"] == ["avg-ref-video-0-components", 0]
    assert inputs["ref_video_audios.ref_video_audio_0"] == [
        "avg-ref-video-0-components",
        1,
    ]
    assert inputs["ref_audios.ref_audio_0"] == ["avg-ref-audio-0", 0]
    assert encode.prompt["avg-ref-video-0"]["class_type"] == "LoadVideo"
    assert encode.prompt["avg-ref-audio-0"]["class_type"] == "LoadAudio"
    assert {blob.media_type for blob in encode.input_blobs} >= {"video/mp4", "audio/wav"}


def test_standard_profile_removes_turbo_nodes_and_uses_stock_sampler() -> None:
    settings = Settings(_env_file=None, h3_turbo_enabled=False, h3_steps=20)
    _encode, diffusion = compile_h3_segment_manifests(
        settings=settings,
        project_id="project",
        prompt={
            "segmentId": "segment-1",
            "durationSeconds": 4,
            "prompt": "完整中文 H3 提示词",
            "seed": 42,
            "assetIds": [],
            "continuationOf": None,
        },
        width=352,
        height=640,
        asset_blobs=(),
        node_schema_sha256="1" * 64,
    )

    assert "134" not in diffusion.prompt
    assert "136" not in diffusion.prompt
    assert diffusion.prompt["135"]["class_type"] == "KSamplerSelect"
    assert diffusion.prompt["135"]["inputs"]["sampler_name"] == "euler"
    assert diffusion.prompt["124"]["inputs"]["steps"] == 20


def test_project_loras_are_applied_in_order_after_turbo() -> None:
    settings = Settings(
        _env_file=None, h3_turbo_enabled=True, h3_sage_attention_enabled=True
    )
    _encode, diffusion = compile_h3_segment_manifests(
        settings=settings,
        project_id="project",
        prompt={
            "segmentId": "segment-1",
            "durationSeconds": 4,
            "prompt": "A cinematic shot",
            "seed": 42,
            "assetIds": [],
            "continuationOf": None,
        },
        width=352,
        height=640,
        asset_blobs=(),
        node_schema_sha256="1" * 64,
        project_loras=(
            {"name": "character.safetensors", "strength": 0.8},
            {"name": "style.safetensors", "strength": 1.2},
        ),
    )

    first = diffusion.prompt["avg-project-lora-000"]
    second = diffusion.prompt["avg-project-lora-001"]
    assert first["inputs"]["model"] == ["134", 0]
    assert first["inputs"]["lora_name"] == "character.safetensors"
    assert second["inputs"]["model"] == ["avg-project-lora-000", 0]
    assert diffusion.prompt["136"]["inputs"]["model"] == ["avg-project-lora-001", 0]
    assert diffusion.prompt["136"]["class_type"] == "ModelAttentionBackend"
    assert diffusion.prompt["136"]["inputs"]["attention"] == "comfy kitchen attention"
    assert "character.safetensors" in diffusion.context["project_loras"]


def test_continuation_reserves_motion_context_inside_sample_budget() -> None:
    settings = Settings(_env_file=None)
    encode, diffusion = compile_h3_segment_manifests(
        settings=settings,
        project_id="project",
        prompt={
            "segmentId": "segment-2",
            "durationSeconds": 12,
            "prompt": "从上一段末尾自然继续",
            "seed": 42,
            "assetIds": [],
            "continuationOf": "segment-1",
        },
        width=352,
        height=640,
        asset_blobs=(),
        node_schema_sha256="1" * 64,
    )

    assert encode.context["motion_context_frames"] == "56"
    assert int(encode.context["sample_frames"]) <= 362
    assert diffusion.prompt["mc-apply"]["inputs"]["context_length"] == "56"
