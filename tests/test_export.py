from pathlib import Path

import pytest

from ai_video_generator.services.export import (
    ExportInput,
    ExportSpec,
    compile_export_plan,
    order_segment_ids_for_export,
)


def test_export_plan_normalizes_video_and_audio() -> None:
    plan = compile_export_plan(
        ExportSpec(
            inputs=(ExportInput(path=Path("one.mp4")), ExportInput(path=Path("two.mp4"))),
            output_path=Path("final.mp4"),
            width=1024,
            height=608,
            fps=24,
            external_audio_path=Path("music.wav"),
            external_audio_volume=0.25,
            loop_external_audio=True,
        )
    )

    assert plan.concat_manifest.count("file '") == 2
    assert "-stream_loop" in plan.command
    assert "scale=1024:608" in plan.command[plan.command.index("-filter_complex") + 1]
    assert plan.command[-1].endswith("final.mp4")


def test_export_plan_does_not_require_an_external_track() -> None:
    plan = compile_export_plan(
        ExportSpec(
            inputs=(ExportInput(path=Path("one.mp4")),),
            output_path=Path("final.mp4"),
            width=1024,
            height=608,
        )
    )

    assert "-vf" in plan.command
    assert "0:a?" in plan.command


def test_tail_trim_uses_probed_absolute_outpoint() -> None:
    plan = compile_export_plan(
        ExportSpec(
            inputs=(
                ExportInput(
                    path=Path("one.mp4"),
                    duration_seconds=10,
                    trim_head_seconds=1,
                    trim_tail_seconds=2,
                ),
            ),
            output_path=Path("final.mp4"),
            width=1024,
            height=608,
        )
    )

    assert "inpoint 1.000000" in plan.concat_manifest
    assert "outpoint 8.000000" in plan.concat_manifest


def test_tail_trim_requires_media_duration() -> None:
    with pytest.raises(ValueError, match="probed input duration"):
        ExportInput(path=Path("one.mp4"), trim_tail_seconds=1)


def test_export_segment_order_follows_workspace_not_opaque_shot_ids() -> None:
    payload = {
        "prompts": {
            "h3Prompts": [
                {"segmentId": "z-shot.C01"},
                {"segmentId": "z-shot.C02"},
                {"segmentId": "a-shot.C01"},
            ]
        }
    }

    ordered = order_segment_ids_for_export(
        ("a-shot.C01", "z-shot.C02", "z-shot.C01"), payload
    )

    assert ordered == ("z-shot.C01", "z-shot.C02", "a-shot.C01")


def test_export_segment_order_rejects_stale_or_incomplete_dependencies() -> None:
    payload = {
        "prompts": {
            "h3Prompts": [
                {"segmentId": "shot-1.C01"},
                {"segmentId": "shot-2.C01"},
            ]
        }
    }

    with pytest.raises(ValueError, match="does not match"):
        order_segment_ids_for_export(("shot-1.C01",), payload)
