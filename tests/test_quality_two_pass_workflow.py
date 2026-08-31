import json
from pathlib import Path

from ai_video_generator.workers import parse_api_workflow

ROOT = Path(__file__).resolve().parents[1] / "workflows" / "h3" / "quality-two-pass"


def test_quality_two_pass_workflow_has_a_minimal_sequential_refinement_path() -> None:
    workflow = parse_api_workflow((ROOT / "initial.api.json").read_text(encoding="utf-8"))
    node_types = {node["class_type"] for node in workflow.values()}

    assert workflow["learned-video-upscale"]["inputs"]["latent"] == ["separate-av", 0]
    assert workflow["separate-av"]["inputs"]["av_latent"] == ["first-pass", 0]
    assert workflow["refine-pass"]["inputs"]["latent_image"] == ["merge-av", 0]
    assert workflow["refine-schedule"]["inputs"]["denoise"] == 0.35
    assert workflow["decode-video"]["inputs"]["samples"] == ["refine-pass", 0]
    assert "MiniMaxH3Cache" not in node_types
    assert "FrameInterpolate" not in node_types
    assert "ComfySwitchNode" not in node_types


def test_quality_director_contract_is_api_focused() -> None:
    schema = json.loads((ROOT / "director-input.schema.json").read_text(encoding="utf-8"))
    reference = schema["properties"]["references"]["items"]

    assert schema["additionalProperties"] is False
    assert schema["properties"]["fps"]["const"] == 24
    assert reference["properties"]["stream"]["enum"] == [
        "visual",
        "audio",
        "visual_and_audio",
    ]
    assert {
        "local_prompt",
        "preserve",
        "allow_change",
        "forbid_propagation",
    }.issubset(reference["required"])
