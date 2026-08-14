from __future__ import annotations

from types import SimpleNamespace
import sys
import types

import torch

from _minimax_h3_motion_director_testpkg.director import fl2v_timeline
from _minimax_h3_motion_director_testpkg.director import segment_runtime

_PATCHES = "_minimax_h3_motion_director_testpkg.patches"
if _PATCHES not in sys.modules:
    patches = types.ModuleType(_PATCHES)
    patches.MC_AUDIO_KEY = "_motion_context_audio_end"
    patches.MC_KEY = "_motion_context_frame_index"
    patches.motion_context_patch_status = lambda: (True, "test")
    sys.modules[_PATCHES] = patches

from _minimax_h3_motion_director_testpkg.director import motion_context


def test_end_only_explicit_ref_never_invents_first_frame_from_placeholder():
    end = torch.ones((1, 32, 32, 3))
    placeholder = torch.full((1, 16, 16, 3), 0.5)
    first_frame, last_frame = fl2v_timeline.resolve_fl2v_endpoint_frames(
        explicit_first=None,
        explicit_last=end,
        clip_frames=placeholder,
    )
    assert first_frame is None
    assert torch.equal(last_frame, end)


def test_gen_fl2v_without_source_clip_does_not_slice_placeholder_source_video():
    plan = SimpleNamespace(
        raw={"timelineMode": "fl2v"},
        source_video=torch.full((1, 16, 16, 3), 0.5),
    )
    segment = SimpleNamespace(
        task_key="fl2v",
        source_clip=None,
        start_frame=0,
        end_frame=124,
    )
    assert segment_runtime.resolve_segment_raw_clip(plan, segment).shape == (0, 16, 16, 3)
    assert segment_runtime.resolve_segment_raw_clip_with_lookahead(
        plan, segment, end_extra=17
    ).shape == (0, 16, 16, 3)


def test_legacy_end_only_marker_expands_to_image1_without_image0():
    endpoint = {
        "id": "end-only",
        "imageFile": "end.png",
        "start": 0,
        "length": 124,
        "frameCount": 124,
        "isStartFrame": True,
        "isEndFrame": True,
        "endOnly": True,
    }
    shots = fl2v_timeline._expand_shots([endpoint])
    assert len(shots) == 1
    assert shots[0]["start"] is None
    assert shots[0]["end"] is endpoint


def test_motion_context_keeps_explicit_fl2v_last_anchor_at_visible_end():
    marker = motion_context.MC_KEY
    existing_last = {
        "kind": "video",
        "resolved_frame_index": 123,
        "latent": torch.ones((1, 1, 1, 1, 1)),
    }
    merged, removed, preserved = motion_context._merge_one_metadata(
        {
            "minimax_frame_count": 124,
            "minimax_keyframes": [existing_last],
        },
        motion_keyframes=[
            {
                "kind": "video",
                "resolved_frame_index": 0,
                marker: 0,
                "latent": torch.zeros((1, 1, 1, 1, 1)),
            }
        ],
        motion_audio_ref=None,
        generation_frame_count=141,
        visible_last_index=128,
    )
    assert removed == 0
    assert preserved == 1
    assert len(merged["minimax_keyframes"]) == 2
    assert merged["minimax_keyframes"][1][marker] == 128
