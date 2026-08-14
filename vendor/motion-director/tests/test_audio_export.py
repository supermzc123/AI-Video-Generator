from __future__ import annotations

from types import SimpleNamespace

import pytest

from _minimax_h3_motion_director_testpkg.director.audio_export import (
    AUDIO_MODE_GENERATE,
    AUDIO_MODE_MUTE,
    AUDIO_MODE_SOURCE,
    resolve_audio_mode,
)


def _plan(task_key: str, audio_mode: str):
    return SimpleNamespace(
        global_task_key=task_key,
        raw={"output": {"audioMode": audio_mode}},
    )


@pytest.mark.parametrize("task_key", ["t2v", "i2v", "r2v", "fl2v"])
@pytest.mark.parametrize("audio_mode", ["generate", "source", "mute"])
def test_generated_av_tasks_always_resolve_generated_audio(task_key, audio_mode):
    assert resolve_audio_mode(_plan(task_key, audio_mode)) == AUDIO_MODE_GENERATE


@pytest.mark.parametrize("task_key", ["v2v", "rv2v"])
@pytest.mark.parametrize(
    ("audio_mode", "expected"),
    [
        ("generate", AUDIO_MODE_GENERATE),
        ("source", AUDIO_MODE_SOURCE),
        ("mute", AUDIO_MODE_MUTE),
    ],
)
def test_video_edit_tasks_preserve_selected_audio_mode(task_key, audio_mode, expected):
    assert resolve_audio_mode(_plan(task_key, audio_mode)) == expected
