from __future__ import annotations

from types import SimpleNamespace

import torch

from _minimax_h3_motion_director_testpkg.director.segment_runtime import (
    load_source_bridge_clip,
    resolve_source_bridge_window,
)


def _segment(index: int, start: int, end: int, task: str = "v2v"):
    return SimpleNamespace(
        index=index,
        timeline_index=index,
        start_frame=start,
        end_frame=end,
        frame_count=end - start,
        task_key=task,
        source_clip=None,
    )


def _plan(segments, count: int = 20, raw: dict | None = None):
    source = torch.arange(count, dtype=torch.float32).reshape(count, 1, 1, 1)
    return SimpleNamespace(
        segments=list(segments),
        source_video=source.repeat(1, 2, 2, 3),
        raw=raw or {"timelineMode": "gen_blank"},
    )


def test_exact_five_frame_source_window_is_loaded_without_padding():
    segments = [_segment(0, 0, 10), _segment(1, 10, 20)]
    plan = _plan(segments)

    window, reason = resolve_source_bridge_window(plan, segments[0], segments[1])
    clip = load_source_bridge_clip(plan, window)

    assert reason is None
    assert (window.source_start, window.source_end) == (8, 13)
    assert clip.shape[0] == 5
    assert clip[:, 0, 0, 0].tolist() == [8, 9, 10, 11, 12]


def test_bridge_skips_when_bof_or_eof_cannot_supply_five_real_frames():
    left = _segment(0, 0, 1)
    right = _segment(1, 1, 4)
    window, reason = resolve_source_bridge_window(_plan([left, right], 4), left, right)

    assert window is None
    assert "five continuous source frames" in reason.lower()


def test_bridge_does_not_cross_a_physical_source_file_boundary():
    frame_map = [
        *({"clip": 0, "frame": i} for i in range(10)),
        *({"clip": 1, "frame": i} for i in range(10)),
    ]
    plan = _plan(
        [_segment(0, 0, 10), _segment(1, 10, 20)],
        20,
        {"video": {"frameMap": frame_map}, "videoClips": [{}, {}]},
    )
    plan.source_video = torch.zeros((0, 2, 2, 3))

    window, reason = resolve_source_bridge_window(plan, *plan.segments)

    assert window is None
    assert "physical source" in reason.lower()


def test_bridge_does_not_cross_an_edited_source_jump():
    frame_map = [
        *({"clip": 0, "frame": i} for i in range(10)),
        *({"clip": 0, "frame": 20 + i} for i in range(10)),
    ]
    plan = _plan(
        [_segment(0, 0, 10), _segment(1, 10, 20)],
        20,
        {"video": {"frameMap": frame_map}, "videoClips": [{}]},
    )
    plan.source_video = torch.zeros((0, 2, 2, 3))

    window, reason = resolve_source_bridge_window(plan, *plan.segments)

    assert window is None
    assert "discontinuity" in reason.lower()


def test_non_video_edit_tasks_never_resolve_a_bridge():
    segments = [_segment(0, 0, 10, "r2v"), _segment(1, 10, 20, "r2v")]
    window, reason = resolve_source_bridge_window(_plan(segments), *segments)

    assert window is None
    assert "v2v/rv2v" in reason.lower()
