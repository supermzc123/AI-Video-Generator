from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

from conftest import PACKAGE


prep = importlib.import_module(f"{PACKAGE}.lib.image_prep")
patches_name = f"{PACKAGE}.patches"
inserted_patches_stub = False
if patches_name not in sys.modules:
    patches_stub = types.ModuleType(patches_name)
    patches_stub.MC_AUDIO_KEY = "_mmx_motion_context_audio"
    patches_stub.MC_KEY = "_mmx_motion_context_frame"
    patches_stub.motion_context_patch_status = lambda: (True, "unit test")
    sys.modules[patches_name] = patches_stub
    inserted_patches_stub = True
motion = importlib.import_module(f"{PACKAGE}.director.motion_context")
if inserted_patches_stub:
    sys.modules.pop(patches_name, None)


def test_global_stride_32_snaps_656x864_to_640x864():
    assert prep.resolve_h3_canvas(656, 864, stride=32) == (640, 864)
    assert prep.resolve_output_dimensions(
        656,
        864,
        mode="long_edge",
        long_edge=864,
        stride=32,
    )[:2] == (640, 864)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"task_key": "i2v", "segment_count": 1, "motion_context_enabled": False}, 32),
        ({"task_key": "i2v", "segment_count": 3, "motion_context_enabled": True}, 32),
        ({"task_key": "r2v", "segment_count": 3, "motion_context_enabled": False}, 32),
        ({"task_key": "r2v", "segment_count": 3, "motion_context_enabled": True}, 32),
        ({"task_key": "r2v", "segment_count": 1, "has_reference_video": True}, 32),
        ({"task_key": "v2v", "segment_count": 1, "motion_context_enabled": False}, 32),
        ({"task_key": "rv2v", "segment_count": 1, "motion_context_enabled": False}, 32),
        ({"task_key": "v2v", "segment_count": 2, "source_bridge_frames": 5}, 32),
        ({"task_key": "t2v", "segment_count": 1, "motion_context_enabled": False}, 32),
        ({"task_key": "t2v", "segment_count": 2, "motion_context_enabled": True}, 32),
    ],
)
def test_all_h3_tasks_use_global_spatial_stride_32(kwargs, expected):
    assert prep.resolve_h3_spatial_stride(**kwargs) == expected


def test_visual_conditioning_preflight_fails_loud_with_diagnostics():
    invalid = torch.zeros((5, 864, 656, 3))
    with pytest.raises(ValueError) as exc_info:
        prep.preflight_h3_visual_conditioning(
            invalid,
            task_key="rv2v",
            path="motion_context",
        )
    message = str(exc_info.value)
    assert "task=rv2v" in message
    assert "path=motion_context" in message
    assert "size=656x864" in message
    assert "required_stride=32" in message


@pytest.mark.parametrize("width", [640, 672])
def test_visual_conditioning_preflight_accepts_32_safe_canvases(width):
    frames = torch.zeros((5, 864, width, 3))
    assert prep.preflight_h3_visual_conditioning(
        frames,
        task_key="v2v",
        path="reference_video",
    ) is frames


def test_motion_context_preflight_runs_before_vae_encode():
    class NeverEncode:
        def encode(self, _frames):
            raise AssertionError("VAE encode must not run before spatial preflight")

    with pytest.raises(ValueError, match="required_stride=32"):
        motion._encode_video_context(
            NeverEncode(),
            torch.zeros((1, 864, 656, 3)),
            width=656,
            height=864,
            span=1,
            task_key="i2v",
        )


def test_motion_context_resizes_then_reanchors_then_preflights_then_encodes(monkeypatch):
    events: list[str] = []

    def fake_resize(images, width, height):
        events.append("resize_anchor" if float(images.mean()) > 0.5 else "resize_context")
        return torch.full((int(images.shape[0]), height, width, 3), float(images.mean()))

    def fake_reanchor(frames, anchor):
        assert frames.shape[1:3] == (864, 640)
        assert anchor.shape[1:3] == (864, 640)
        events.append("color_reanchor")
        return frames

    def fake_preflight(frames, **_kwargs):
        assert frames.shape[1:3] == (864, 640)
        events.append("preflight")
        return frames

    class VAE:
        def encode(self, frames):
            assert frames.shape[1:3] == (864, 640)
            events.append("encode")
            return torch.zeros((1, 16, 1, 54, 40))

    monkeypatch.setattr(motion, "_resize_frames", fake_resize)
    monkeypatch.setattr(motion, "apply_color_reanchor", fake_reanchor)
    monkeypatch.setattr(motion, "preflight_h3_visual_conditioning", fake_preflight)

    motion._encode_video_context(
        VAE(),
        torch.full((1, 32, 32, 3), 0.25),
        width=640,
        height=864,
        span=1,
        task_key="i2v",
        color_reanchor_enabled=True,
        color_anchor=torch.full((1, 16, 16, 3), 0.75),
    )

    assert events == [
        "resize_context",
        "resize_anchor",
        "color_reanchor",
        "preflight",
        "encode",
    ]
