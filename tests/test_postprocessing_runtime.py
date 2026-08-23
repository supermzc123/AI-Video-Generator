from __future__ import annotations

import pytest

from ai_video_generator.services.postprocessing_profiles import inspect_postprocess_profiles
from ai_video_generator.services.postprocessing_runtime import interpolation_plan


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (24, 48, (2, 48)),
        (24, 60, (5, 120)),
        (24, 120, (5, 120)),
        (30, 60, (2, 60)),
        (30, 120, (4, 120)),
    ],
)
def test_interpolation_plan_uses_integer_generation(
    source: int, target: int, expected: tuple[int, int]
) -> None:
    assert interpolation_plan(source, target) == expected


def test_interpolation_plan_rejects_intermediate_above_profile_limit() -> None:
    with pytest.raises(ValueError, match="exceeding Profile limit"):
        interpolation_plan(25, 48)


def test_profiles_without_api_workflow_are_not_reported_as_executable() -> None:
    node_types = {
        "UNETLoader",
        "VAELoader",
        "SeedVR2Conditioning",
        "SeedVR2Preprocess",
        "SeedVR2PostProcessing",
        "SaveVideo",
        "VHS_LoadVideo",
        "DownloadAndLoadGIMMVFIModel",
        "GIMMVFI_interpolate",
        "VHS_VideoCombine",
    }
    nodes = {
        value: {"input": {"required": {}}}
        for value in node_types
    }
    profiles = {
        item.profile.profile_id: item
        for item in inspect_postprocess_profiles(nodes, server_online=True)
    }
    assert not profiles["seedvr2:official-video"].available
    assert not profiles["interpolation:gimm-vfi"].available
    assert any(
        "API" in blocker and "工作流" in blocker
        for blocker in profiles["seedvr2:official-video"].blockers
    )


def test_whisper_profile_reflects_packaged_executor_availability() -> None:
    unavailable = {
        item.profile.profile_id: item
        for item in inspect_postprocess_profiles(
            {}, server_online=False, whisper_available=False
        )
    }
    available = {
        item.profile.profile_id: item
        for item in inspect_postprocess_profiles(
            {}, server_online=False, whisper_available=True
        )
    }

    assert not unavailable["transcription:faster-whisper"].available
    assert unavailable["transcription:faster-whisper"].blockers
    assert available["transcription:faster-whisper"].available
