import pytest
from fastapi import HTTPException

from ai_video_generator.prompting_api import (
    _bound_asset_ids_for_shot,
    _complete_image_prompt,
    _continuation_end_state,
    _creative_brief,
    _image_prompt_drafts,
    _incomplete_asset_plan_names,
    _prompt_terminal_context,
    _segment_constraints,
    _workspace_segments,
)


def test_asset_plan_binding_never_propagates_an_unused_material() -> None:
    payload = {
        "assetPlans": [
            {"fulfilledByAssetId": "unused", "scope": "public", "shotIds": []},
            {"fulfilledByAssetId": "hero", "scope": "public", "shotIds": ["shot-1"]},
            {"fulfilledByAssetId": "legacy", "scope": "shot", "shotId": "shot-2"},
        ]
    }

    assert _bound_asset_ids_for_shot(payload, "shot-1") == {"hero"}
    assert _bound_asset_ids_for_shot(payload, "shot-2") == {"legacy"}
    assert _bound_asset_ids_for_shot(payload, "shot-3") == set()


def test_prompt_terminal_context_does_not_duplicate_a_long_execution_prompt() -> None:
    prompt = "opening description\n\n" + ("detailed ending state " * 300)

    context = _prompt_terminal_context(prompt)

    assert len(context) <= 4000
    assert context.endswith("ending state")
    assert context != prompt


class _ImagePromptClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def complete_text(self, _messages):
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_image_prompt_generation_returns_plain_text_without_repair() -> None:
    client = _ImagePromptClient(
        [
            "电影感人物全身定妆照，清晰面部特征，柔和侧光，简洁背景，细节完整",
        ]
    )

    result = await _complete_image_prompt(client, ())  # type: ignore[arg-type]

    assert "电影感" in result


@pytest.mark.asyncio
async def test_image_prompt_generation_does_not_retry() -> None:
    client = _ImagePromptClient(["short"])
    assert await _complete_image_prompt(client, ()) == "short"  # type: ignore[arg-type]
    assert client.responses == []


def test_h3_prompting_reports_every_unfulfilled_image_plan() -> None:
    names = _incomplete_asset_plan_names(
        {
            "assetPlans": [
                {"id": "hero", "name": "主角", "fulfilledByAssetId": "asset-1"},
                {"id": "set", "name": "雨夜街道", "fulfilledByAssetId": None},
                {"id": "prop", "name": "红伞"},
            ]
        }
    )

    assert names == ("雨夜街道", "红伞")


def test_fulfilled_asset_keeps_its_image_prompt_for_regeneration() -> None:
    existing = {
        "assetPlanId": "hero",
        "prompt": "完整的主角定妆提示词",
        "workflowTemplateId": "workflow-1",
        "revision": 3,
    }
    payload = {
        "assetPlans": [
            {"id": "hero", "name": "主角", "fulfilledByAssetId": "asset-1"}
        ],
        "prompts": {"imagePrompts": [existing]},
    }

    assert _image_prompt_drafts(payload) == [existing]


def test_long_shot_is_split_into_separate_motion_context_prompt_segments() -> None:
    payload = {
        "name": "Long take",
        "idea": {
            "concept": "A continuous railway-platform pursuit",
            "genre": "thriller",
            "visualStyle": "cinematic",
        },
        "shots": [
            {
                "id": "shot-1",
                "title": "Pursuit",
                "summary": "The protagonist follows the suspect.",
                "camera": "handheld tracking",
                "seed": 42,
                "durationSeconds": 31,
            }
        ],
    }

    segments = _workspace_segments(payload)

    assert len(segments) == 3
    assert all(4 <= item["durationSeconds"] <= 15 for item in segments)
    assert segments[0]["continuationOf"] is None
    assert segments[1]["continuationOf"] == "shot-1.C01"
    assert segments[2]["continuationOf"] == "shot-1.C02"
    assert sum(item["durationSeconds"] for item in segments) == 31

    continuation_brief = _creative_brief(payload, segments[1], "人物向画面右侧奔跑")
    assert "3 个 Motion Context 视频片段，分别编写 H3 提示词" in continuation_brief
    assert "当前只编写第 2/3 段" in continuation_brief
    assert "上一段结束状态：人物向画面右侧奔跑" in continuation_brief
    assert "Motion Context" in continuation_brief

    first_constraints = "\n".join(_segment_constraints(segments[0]))
    continuation_constraints = "\n".join(_segment_constraints(segments[1]))
    assert "已规划为 3 份提示词和视频片段" in first_constraints
    assert "连续链首段" in first_constraints
    assert "当前是续段" in continuation_constraints
    assert "继续事件而非重演" in continuation_constraints


def test_regular_shot_is_not_described_as_a_long_motion_context_chain() -> None:
    payload = {
        "name": "Single segment",
        "shots": [{"id": "shot-1", "durationSeconds": 12, "seed": 0}],
    }
    segment = _workspace_segments(payload)[0]

    brief = _creative_brief(payload, segment, None)
    constraints = "\n".join(_segment_constraints(segment))

    assert "超过 H3 单次 15 秒上限" not in brief
    assert "Motion Context" not in brief
    assert "原电影分镜超过15秒" not in constraints
    assert "独立、可执行的 12.0 秒 H3 提示词" in constraints


def test_single_continuation_regeneration_uses_previous_terminal_state_when_available() -> None:
    segment = {
        "segmentId": "shot-1.C02",
        "continuationOf": "shot-1.C01",
    }

    assert _continuation_end_state(
        segment,
        {"shot-1.C01": "人物朝画面右侧奔跑，摄影机平行跟随"},
        require_for_single_regeneration=True,
    ) == "人物朝画面右侧奔跑，摄影机平行跟随"

    assert _continuation_end_state(
        segment, {}, require_for_single_regeneration=True
    ) is None


def test_manual_motion_context_plan_controls_segment_boundaries() -> None:
    payload = {
        "shots": [
            {
                "id": "manual-chain",
                "durationSeconds": 30,
                "seed": 7,
                "summary": "A continuous dance.",
                "motionSegments": [
                    {"durationSeconds": 9, "summary": "Entrance and first pose."},
                    {"durationSeconds": 10, "summary": "Continuous turn, ending in profile."},
                    {"durationSeconds": 11, "summary": "Continue from profile and bow."},
                ],
            }
        ]
    }

    segments = _workspace_segments(payload)

    assert [item["durationSeconds"] for item in segments] == [9, 10, 11]
    assert [item["segmentSummary"] for item in segments] == [
        "Entrance and first pose.",
        "Continuous turn, ending in profile.",
        "Continue from profile and bow.",
    ]
    assert segments[2]["continuationOf"] == "manual-chain.C02"
    assert "Continue from profile and bow." in _creative_brief(
        payload, segments[2], None
    )


def test_invalid_manual_motion_context_plan_is_rejected_at_generation_boundary() -> None:
    payload = {
        "shots": [
            {
                "id": "bad-chain",
                "durationSeconds": 20,
                "motionSegments": [
                    {"durationSeconds": 10, "summary": "First."},
                    {"durationSeconds": 15, "summary": "Continuation is too long."},
                ],
            }
        ]
    }

    with pytest.raises(HTTPException) as exc_info:
        _workspace_segments(payload)

    assert exc_info.value.status_code == 422
    assert "4 到 12 秒" in str(exc_info.value.detail)


def test_sub_four_second_shot_uses_h3_minimum_duration() -> None:
    segments = _workspace_segments(
        {"shots": [{"id": "short", "durationSeconds": 2, "seed": 0}]}
    )

    assert segments[0]["durationSeconds"] == 4
    assert segments[0]["shotDurationSeconds"] == 2
