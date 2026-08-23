from __future__ import annotations

import json

import pytest

from ai_video_generator.domain.chain import GenerationMode
from ai_video_generator.domain.h3_prompt import (
    H3AssetInput,
    H3AssetKind,
    H3AssetPromptRole,
    H3PromptCandidate,
    H3PromptRequest,
    H3ShotBeat,
)
from ai_video_generator.llm.h3_prompt import (
    H3CallTelemetry,
    H3HarnessLibrary,
    H3PromptHarness,
    H3PromptHarnessError,
    deterministic_director_decision,
    render_h3_prompt,
    route_h3_mode,
    validate_h3_candidate,
)


class QueueClient:
    def __init__(self, values: list[dict[str, object] | str]) -> None:
        self.values = values
        self.messages = []

    async def complete_json(self, messages) -> str:
        self.messages.append(messages)
        value = self.values.pop(0)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def library() -> H3HarnessLibrary:
    return H3HarnessLibrary(
        official_skill="官方入口",
        official_base="基础字段规范",
        official_reference="六段引用规范",
        community_director="导演路由",
        community_planner="多镜头规划",
        community_text_writer="文本模式编写",
        community_keyframe_writer="关键帧模式编写",
        community_reference_writer="引用模式编写",
        community_reviewer="审核器",
        community_review_checklist="完整性检查表",
    )


def test_v2_manifest_is_frozen_and_rejects_tampered_documents() -> None:
    manifest = library().to_manifest(official_commit="a" * 40, community_commit="b" * 40)
    restored = H3HarnessLibrary.from_manifest(manifest)
    first = manifest.documents[0]
    tampered = manifest.model_copy(
        update={
            "documents": (
                first.model_copy(update={"content": first.content + " changed"}),
                *manifest.documents[1:],
            )
        }
    )

    assert restored == library()
    with pytest.raises(H3PromptHarnessError, match="hash mismatch"):
        H3HarnessLibrary.from_manifest(tampered)


def request(*, assets=()) -> H3PromptRequest:
    return H3PromptRequest(
        operation_id="op-1",
        segment_id="segment-1",
        creative_brief="雨夜里，一位侦探穿过街道并发现关键线索。",
        duration_seconds=4,
        assets=assets,
    )


def beat() -> dict[str, object]:
    return {
        "shot_number": 1,
        "start_seconds": 0,
        "end_seconds": 4,
        "composition": "中景构图",
        "subjects": "穿深色风衣的侦探",
        "environment": "霓虹灯映照的雨夜街道",
        "action": "侦探走向反光的金属线索并停下",
        "camera": "摄影机稳定横移后轻微推进",
        "sound": "雨声、脚步声与远处车辆声",
        "end_state": "侦探蹲下看清金属线索",
    }


def valid_base_candidate() -> dict[str, object]:
    description = (
        "[Shot 1] A cinematic medium shot places a detective in a dark trench coat "
        "on the left side of a rain-soaked street, where saturated neon signs form "
        "sharp reflections across the pavement. He walks forward at a measured pace "
        "while the camera tracks parallel to him with restrained, steady movement. "
        "Raindrops strike his collar and splash around each footstep. After two seconds, "
        "a metallic clue flashes beside the curb. The camera slowly pushes closer as he "
        "stops, crouches, and reaches toward it without breaking spatial continuity. "
        "He ends alert and motionless, studying the clue while rain, fabric movement, "
        "footsteps, and distant traffic remain synchronized with the visible action."
    )
    return {
        "operation_id": "op-1",
        "mode": "t2va",
        "timeline": [beat()],
        "integrated_multimodal_description": description,
        "overall_soundscape": (
            "Continuous stereo rain surrounds approaching footsteps, subtle fabric "
            "movement, curbside splashes, and sparse traffic in the distance."
        ),
        "non_diegetic_music": (
            "Low strings maintain a restrained suspense pulse beneath the physical sounds."
        ),
    }


@pytest.mark.asyncio
async def test_h3_harness_routes_deterministically_without_llm_reviewer() -> None:
    client = QueueClient(
        [
            valid_base_candidate(),
        ]
    )

    result = await H3PromptHarness(client, library()).generate(request())

    assert result.repair_passes == 0
    assert result.execution_prompt.startswith("integrated_multimodal_description:")
    assert "rain-soaked street" in result.execution_prompt
    assert result.director.mode == GenerationMode.T2VA
    assert result.director.use_multishot is False
    assert len(client.messages) == 1
    assert "执行描述使用英文" in client.messages[0][0].content
    assert "350-500 个英文单词" in client.messages[0][0].content
    assert "分段契约（segment contract）" in client.messages[0][0].content
    assert "continuationOf" in client.messages[0][0].content
    assert "基础字段规范" in client.messages[0][0].content


@pytest.mark.asyncio
async def test_h3_stage_repairs_malformed_json_without_shortening_content() -> None:
    client = QueueClient(
        [
            '{"operation_id":"op-1","mode":"t2va"',
            valid_base_candidate(),
        ]
    )

    result = await H3PromptHarness(client, library()).generate(request())

    assert len(result.execution_prompt) > 120
    repair_request = json.loads(client.messages[1][-1].content)
    assert repair_request["task"] == "repair_invalid_h3_stage_json"
    assert "不要缩写提示词" in repair_request["instruction"]
    assert "官方英文规范" in repair_request["instruction"]


@pytest.mark.asyncio
async def test_h3_harness_never_accepts_short_prompt_and_stops_after_one_repair() -> None:
    short = valid_base_candidate()
    short["integrated_multimodal_description"] = "[Shot 1] The detective crosses the street."
    client = QueueClient(
        [
            short,
        short,
        ]
    )

    with pytest.raises(H3PromptHarnessError, match="one repair pass") as exc_info:
        await H3PromptHarness(client, library()).generate(request())

    assert "too short" in " ".join(exc_info.value.errors)
    assert len(client.messages) == 2


def test_h3_candidate_rejects_chinese_descriptive_prose() -> None:
    candidate_data = valid_base_candidate()
    candidate_data["integrated_multimodal_description"] = (
        "[Shot 1] 侦探沿着雨夜街道前行，镜头平稳跟随，他发现路边线索后停下查看。"
    )
    candidate = H3PromptCandidate.model_validate(candidate_data)

    errors = validate_h3_candidate(
        request(), deterministic_director_decision(request()), None, candidate
    )

    assert any("must use English descriptive content" in error for error in errors)


def test_h3_candidate_allows_original_language_inside_dialogue_tags() -> None:
    candidate_data = valid_base_candidate()
    candidate_data["integrated_multimodal_description"] += (
        " The detective (S1) whispers, <d>[Chinese] 我找到线索了。</d> "
        "His lips close before he returns his attention to the object."
    )
    candidate = H3PromptCandidate.model_validate(candidate_data)

    errors = validate_h3_candidate(
        request(), deterministic_director_decision(request()), None, candidate
    )

    assert not any("must use English descriptive content" in error for error in errors)


@pytest.mark.asyncio
async def test_only_writer_receives_reference_images_and_reviewer_has_occlusion_rules() -> None:
    first_frame = H3AssetInput(
        asset_id="frame",
        label="<Picture 1>",
        kind=H3AssetKind.IMAGE,
        role=H3AssetPromptRole.FIRST_FRAME,
        preservation="保持首帧人物、构图和光线",
    )
    candidate = valid_base_candidate()
    candidate["mode"] = "i2va"
    candidate["integrated_multimodal_description"] += (
        " <Picture 1> is used only as the opening frame anchor."
    )
    client = QueueClient(
        [
            candidate,
        ]
    )

    await H3PromptHarness(client, library()).generate(
        request(assets=(first_frame,)),
        asset_image_urls=("data:image/jpeg;base64,AAAA",),
    )

    assert not isinstance(client.messages[0][1].content, str)
    assert "不得把服装纹理" in client.messages[0][0].content


@pytest.mark.asyncio
async def test_h3_harness_records_per_call_telemetry() -> None:
    telemetry: list[H3CallTelemetry] = []
    client = QueueClient(
        [
            valid_base_candidate(),
        ]
    )

    await H3PromptHarness(client, library(), telemetry_sink=telemetry.append).generate(request())

    assert [item.stage for item in telemetry] == ["write"]
    assert all(item.succeeded and item.total_seconds >= 0 for item in telemetry)
    assert all(item.input_characters > 0 and item.output_characters > 0 for item in telemetry)


def test_auto_multishot_requires_an_explicit_editing_marker() -> None:
    ordinary = request()
    explicit = ordinary.model_copy(
        update={"creative_brief": "先跟随人物奔跑，随后切到室内的反应镜头。"}
    )

    assert deterministic_director_decision(ordinary).use_multishot is False
    assert deterministic_director_decision(explicit).use_multishot is True


def test_asset_routing_defaults_reusable_images_to_ref2va() -> None:
    reusable = H3AssetInput(
        asset_id="hero",
        label="<Picture 1>",
        kind=H3AssetKind.IMAGE,
        role=H3AssetPromptRole.IDENTITY,
        preservation="保持人物面部、发型和服装完全一致",
    )
    first = reusable.model_copy(
        update={"role": H3AssetPromptRole.FIRST_FRAME, "preservation": "只作为首帧边界"}
    )

    assert route_h3_mode(request(assets=(reusable,))) == GenerationMode.REF2VA
    assert route_h3_mode(request(assets=(first,))) == GenerationMode.I2VA


def test_ref_candidate_requires_all_labels_and_full_timeline() -> None:
    asset = H3AssetInput(
        asset_id="hero",
        label="<Picture 1>",
        kind=H3AssetKind.IMAGE,
        role=H3AssetPromptRole.CHARACTER,
        preservation="保持人物身份",
    )
    prompt_request = request(assets=(asset,))
    candidate = H3PromptCandidate(
        operation_id="op-1",
        mode=GenerationMode.REF2VA,
        timeline=(H3ShotBeat.model_validate(beat()),),
        subject_definitions="侦探：保持面部与服装特征，不得改变人物身份。",
        summary="在雨夜街道中发现线索的四秒悬疑连续镜头。",
        retention_analysis="人物身份强保留，街道布局弱参考，声音全部重新生成。",
        detailed_description=(
            "[Shot 1] 中景拍摄侦探穿过雨夜街道，摄影机平稳横移并逐渐推进。"
            "霓虹倒影随脚步在积水中波动，他发现路边金属线索后停下并蹲身查看。"
            "镜头保持人物面部清楚、动作连续、光线方向一致，最后停在警觉的表情和线索同框画面。"
        ),
        overall_soundscape="连续雨声、清晰脚步声、衣料摩擦声和远处车辆声。",
        non_diegetic_music="低音弦乐悬疑脉冲。",
    )
    director = {
        "operation_id": "op-1",
        "mode": "ref2va",
        "use_multishot": False,
        "rationale": "人物图片承担身份参考职责。",
    }
    from ai_video_generator.domain.h3_prompt import H3DirectorDecision

    errors = validate_h3_candidate(
        prompt_request,
        H3DirectorDecision.model_validate(director),
        None,
        candidate,
    )

    assert "prompt does not assign a role to <Picture 1>" in errors
    assert render_h3_prompt(candidate).splitlines()[0].startswith("subject_definitions:")
