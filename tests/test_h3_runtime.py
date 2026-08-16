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
    assert diffusion.prompt["135"]["class_type"] == "KSamplerSelect"
    assert diffusion.prompt["135"]["inputs"]["sampler_name"] == "euler"
    assert diffusion.prompt["124"]["inputs"]["steps"] == 20


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
