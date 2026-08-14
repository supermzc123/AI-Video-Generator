from ai_video_generator.domain import (
    AssetBinding,
    AssetRole,
    ConditioningStack,
    GenerationMode,
    GenerationSegment,
)
from ai_video_generator.services import conditioning_fingerprint, normalize_prompt

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def make_segment(prompt: str, assets: tuple[AssetBinding, ...]) -> GenerationSegment:
    return GenerationSegment(
        segment_id="S001.C01",
        ordinal=1,
        prompt_revision_id="prompt-1@1",
        normalized_prompt=prompt,
        generation_mode=GenerationMode.REF2VA,
        width=1024,
        height=608,
        sample_frames=277,
        visible_frames=277,
        common_assets=assets,
    )


def make_stack() -> ConditioningStack:
    return ConditioningStack(
        text_encoder_sha256=HASH_A,
        h3_model_sha256=HASH_B,
        video_vae_sha256=HASH_C,
        audio_vae_sha256=HASH_D,
        comfyui_commit="344b43989e8c56b5bb4a66cf028c834192ab59dd",
        worker_engine_commit="1234567890abcdef",
        node_versions={"h3-audio-t8": "1.3.2"},
    )


def test_prompt_normalization_is_stable() -> None:
    assert normalize_prompt("  one\n two   three ") == "one two three"


def test_equivalent_prompt_whitespace_has_same_fingerprint() -> None:
    asset = AssetBinding(
        asset_revision_id="hero@1",
        sha256=HASH_A,
        role=AssetRole.IDENTITY,
        priority=100,
    )
    left = conditioning_fingerprint(make_segment("one  two", (asset,)), make_stack())
    right = conditioning_fingerprint(make_segment("one\ntwo", (asset,)), make_stack())

    assert left == right


def test_asset_binding_order_changes_fingerprint() -> None:
    hero = AssetBinding(
        asset_revision_id="hero@1",
        sha256=HASH_A,
        role=AssetRole.IDENTITY,
    )
    style = AssetBinding(
        asset_revision_id="style@1",
        sha256=HASH_B,
        role=AssetRole.STYLE,
    )

    left = conditioning_fingerprint(make_segment("prompt", (hero, style)), make_stack())
    right = conditioning_fingerprint(make_segment("prompt", (style, hero)), make_stack())

    assert left != right
