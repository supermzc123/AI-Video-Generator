from __future__ import annotations

import importlib

import pytest
import torch


def _helper():
    module = importlib.import_module(
        "_minimax_h3_motion_director_testpkg.nodes.conditioning"
    )
    return module.append_minimax_keyframe_anchors


class _VAE:
    def __init__(self):
        self.inputs = []

    def encode(self, frames):
        self.inputs.append(frames.clone())
        return torch.ones((1, 4, 1, 2, 2), dtype=torch.float32) * len(self.inputs)


def test_reference_conditioning_keeps_refs_and_appends_native_anchor_positions():
    try:
        helper = _helper()
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"full ComfyUI runtime is unavailable: {exc}")

    refs = [{"kind": "video", "latent": torch.zeros(1, 4, 1, 2, 2)}]
    positive = [[torch.zeros(1), {"minimax_refs": refs}]]
    vae = _VAE()
    result = helper(
        positive,
        vae=vae,
        first_frame=torch.zeros(1, 8, 8, 3),
        last_frame=torch.ones(1, 8, 8, 3),
        frame_count=5,
        width=8,
        height=8,
    )

    metadata = result[0][1]
    assert metadata["minimax_refs"] is refs
    assert metadata["minimax_frame_count"] == 5
    assert [item["motion_context_index"] for item in metadata["minimax_keyframes"]] == [0, 4]
    assert all(item["resolved_frame_index"] == 0 for item in metadata["minimax_keyframes"])
    assert [
        item["latent"].reshape(-1)[0].item()
        for item in metadata["minimax_keyframes"]
    ] == [1.0, 2.0]
    assert len(vae.inputs) == 2


def test_anchor_helper_rejects_wrong_frame_count_and_existing_keyframes():
    try:
        helper = _helper()
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"full ComfyUI runtime is unavailable: {exc}")

    kwargs = dict(
        vae=_VAE(),
        first_frame=torch.zeros(1, 8, 8, 3),
        last_frame=torch.ones(1, 8, 8, 3),
        width=8,
        height=8,
    )
    with pytest.raises(ValueError, match="exactly 5"):
        helper([[torch.zeros(1), {}]], frame_count=4, **kwargs)
    with pytest.raises(ValueError, match="already contains"):
        helper(
            [[torch.zeros(1), {"minimax_keyframes": [{}]}]],
            frame_count=5,
            **kwargs,
        )


def test_refs_and_bridge_anchors_land_on_h3_target_frames_zero_and_four():
    try:
        helper = _helper()
        patches = importlib.import_module(
            "_minimax_h3_motion_director_testpkg.patches"
        )
        layout_patch = importlib.import_module(
            "_minimax_h3_motion_director_testpkg.patches.h3_layout"
        )
        from comfy.ldm.minimax import model as minimax_model
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"full ComfyUI runtime is unavailable: {exc}")

    assert patches.apply_motion_context_patches()
    refs = [
        {
            "kind": "video",
            "latent_t": 2,
            "latent_h": 2,
            "latent_w": 2,
            "ref_audio_t": 0,
        }
    ]
    positive = helper(
        [[torch.zeros(1), {"minimax_refs": refs}]],
        vae=_VAE(),
        first_frame=torch.zeros(1, 8, 8, 3),
        last_frame=torch.ones(1, 8, 8, 3),
        frame_count=5,
        width=8,
        height=8,
    )
    metadata = positive[0][1]
    layout = minimax_model.PackedLayout(
        7,
        2,
        2,
        2,
        1,
        keyframes=metadata["minimax_keyframes"],
        refs=metadata["minimax_refs"],
        frame_count=metadata["minimax_frame_count"],
    )
    origin = layout_patch._target_origin(layout)
    times = [
        float(layout.position_ids[start, 0])
        for start, _end, kind in layout.segments
        if kind == "cond"
    ]

    assert times == pytest.approx(
        [origin, origin + minimax_model.FRAME_RESCALE * 4], abs=1e-4
    )
