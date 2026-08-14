from __future__ import annotations

from types import SimpleNamespace
import json

import torch

from _minimax_h3_motion_director_testpkg.director import (
    external_groups,
    gen_timeline,
    plan as plan_module,
)
from _minimax_h3_motion_director_testpkg.director.effective_refs import semantic_reference_token


def _fake_picture(item):
    return SimpleNamespace(
        index=int(item.get("index", 0)),
        asset_id=item["assetId"],
        tensor=torch.zeros((1, 32, 32, 3)),
    )


def _install_loaders(monkeypatch):
    monkeypatch.setattr(plan_module, "_load_refs", lambda items: [_fake_picture(x) for x in items])
    monkeypatch.setattr(plan_module, "_load_ref_videos", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(plan_module, "_load_ref_audios", lambda *_args, **_kwargs: [])


def _timeline():
    common = [
        {"index": i, "assetId": asset_id, "imageB64": "unused"}
        for i, asset_id in enumerate(["A", "B", "C"])
    ]
    return {
        "timelineMode": "prompt_batch",
        "editMode": "segment",
        "global": {
            "taskType": "r2v",
            "prompt": "GLOBAL MUST NOT LEAK",
        },
        "r2vCommon": {
            "refs": common,
        },
        "output": {"width": 64, "height": 64, "mode": "fixed"},
        "segments": [
            {
                "frameCount": 5,
                "prompt": f"one {semantic_reference_token('picture', 'D')}",
                "useCommonAssets": True,
                "excludedCommonAssetIds": [],
                "refs": [{"index": 0, "assetId": "D", "imageB64": "unused"}],
            },
            {
                "frameCount": 5,
                "prompt": "pure scene",
                "useCommonAssets": False,
                "excludedCommonAssetIds": [],
                "refs": [],
            },
            {
                "frameCount": 5,
                "prompt": f"three {semantic_reference_token('picture', 'C')}",
                "useCommonAssets": True,
                "excludedCommonAssetIds": [],
                "refs": [{"index": 0, "assetId": "E", "imageB64": "unused"}],
            },
        ],
    }


def test_r2v_plan_uses_selected_common_plus_local_without_previous_bundle(monkeypatch):
    _install_loaders(monkeypatch)
    director_plan = gen_timeline.build_gen_director_plan(
        _timeline(),
        global_task_type="r2v",
        global_prompt="",
        total_frames=15,
        frame_rate=24,
        width=64,
        height=64,
        ref_max_size=64,
        motion_context_enabled=True,
    )

    assert [[x.asset_id for x in seg.refs] for seg in director_plan.segments] == [
        ["A", "B", "C", "D"],
        [],
        ["A", "B", "C", "E"],
    ]
    assert director_plan.segments[0].prompt == "one <Picture 4>"
    assert director_plan.segments[1].prompt == "pure scene"
    assert director_plan.segments[2].prompt == "three <Picture 3>"
    assert not hasattr(director_plan.segments[1], "material_inherited")


def test_disabled_common_asset_reference_fails_during_plan_preflight(monkeypatch):
    _install_loaders(monkeypatch)
    timeline = _timeline()
    timeline["segments"][1]["useCommonAssets"] = True
    timeline["segments"][1]["excludedCommonAssetIds"] = ["B"]
    timeline["segments"][1]["prompt"] = semantic_reference_token("picture", "B")

    try:
        gen_timeline.build_gen_director_plan(
            timeline,
            global_task_type="r2v",
            global_prompt="",
            total_frames=15,
            frame_rate=24,
            width=64,
            height=64,
            ref_max_size=64,
            motion_context_enabled=False,
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - documents the required RED behavior
        raise AssertionError("disabled Common prompt reference was accepted")

    assert "Segment 2" in message
    assert "disabled" in message
    assert "B" in message


def test_external_r2v_groups_use_same_common_selection_and_never_inherit_local(monkeypatch):
    _install_loaders(monkeypatch)
    local_d = torch.ones((1, 32, 32, 3))
    groups = [
        external_groups.pack_r2v_group(prompt="one", ref_images={0: local_d}),
        external_groups.pack_r2v_group(prompt="pure scene"),
    ]
    timeline = {
        "frameRate": 24,
        "global": {
            "prompt": "GLOBAL MUST NOT LEAK",
        },
        "r2vCommon": {
            "refs": [{"index": 0, "assetId": "A", "imageB64": "unused"}],
        },
        "output": {"width": 64, "height": 64, "mode": "fixed"},
        "segments": [
            {"useCommonAssets": True, "excludedCommonAssetIds": []},
            {"useCommonAssets": False, "excludedCommonAssetIds": []},
        ],
    }

    task_key, validated, family = external_groups.validate_external_group_inputs(
        task_type="r2v",
        i2v_groups=None,
        r2v_groups=groups,
        motion_context_enabled=True,
    )
    assert (task_key, family) == ("r2v", "r2v")
    result = external_groups.build_plan_from_external_groups(
        validated,
        family=family,
        timeline_data=json.dumps(timeline),
        task_type="r2v",
        global_prompt="",
        total_frames=10,
        frame_rate=24,
        width=64,
        height=64,
        ref_max_size=64,
        motion_context_enabled=True,
    )

    assert [x.asset_id for x in result.segments[0].refs] == ["A", "external-picture-0-0"]
    assert result.segments[1].refs == []
    assert result.segments[0].prompt.endswith("one")
    assert "GLOBAL MUST NOT LEAK" not in result.segments[0].prompt
    assert result.segments[1].prompt == "pure scene"
