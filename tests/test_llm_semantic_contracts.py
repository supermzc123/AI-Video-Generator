import json
from types import SimpleNamespace

import pytest
from test_h3_prompt_harness import request
from test_llm_harness import patch_request

from ai_video_generator.domain.h3_prompt import H3AssetInput, H3AssetKind, H3AssetPromptRole
from ai_video_generator.llm import (
    HarnessValidationError,
    build_structured_operation_messages,
    parse_structured_operation,
)
from ai_video_generator.llm.context import content_fingerprint
from ai_video_generator.prompting_api import (
    _can_reuse_h3_prompt,
    _generate_validated_plain_h3,
    _h3_input_fingerprint,
    _image_context_shots,
    _merge_h3_prompt_entries,
    _validate_plain_h3_prompt,
)


def patch_response(path, value):
    return json.dumps(
        {
            "patches": [{"op": "replace", "path": path, "value": value}],
            "rationale": "Improve narrative",
        }
    )


def semantic_shot():
    return {
        "title": "Arrival",
        "summary": "The protagonist arrives.",
        "camera": "Wide",
        "durationSeconds": 6,
    }


def test_server_supplies_stable_id_seed_and_lock_instead_of_requiring_model_fields():
    operation = patch_request().model_copy(
        update={"source_document": {"shots": []}, "allowed_paths": ("/shots",), "locked_paths": ()}
    )
    first = parse_structured_operation(patch_response("/shots", [semantic_shot()]), operation)
    second = parse_structured_operation(patch_response("/shots", [semantic_shot()]), operation)
    assert first == second
    value = first.patches[0].value[0]
    assert value["id"] and isinstance(value["seed"], int) and value["locked"] is False
    messages = build_structured_operation_messages(operation)
    contract = json.loads(messages[1].content)["contract"]["workspace_value_contracts"]["shot_item"]
    assert not {"id", "seed", "locked"} & set(contract["required"])


@pytest.mark.parametrize("replacement", [[], [{**semantic_shot(), "id": "shot", "locked": False}]])
def test_replacing_whole_array_cannot_bypass_implicit_item_lock(replacement):
    operation = patch_request().model_copy(
        update={
            "source_document": {
                "shots": [{**semantic_shot(), "id": "shot", "seed": 1, "locked": True}]
            },
            "allowed_paths": ("/shots",),
            "locked_paths": (),
        }
    )
    with pytest.raises(HarnessValidationError) as info:
        parse_structured_operation(patch_response("/shots", replacement), operation)
    assert "locked item" in " ".join(info.value.errors)


def test_nested_duration_patch_receives_the_same_validation_as_full_replacement():
    operation = patch_request().model_copy(
        update={
            "source_document": {
                "shots": [{**semantic_shot(), "id": "shot", "seed": 1, "locked": False}]
            },
            "allowed_paths": ("/shots",),
            "locked_paths": (),
        }
    )
    with pytest.raises(HarnessValidationError) as info:
        parse_structured_operation(patch_response("/shots/0/durationSeconds", -1), operation)
    assert "positive number" in " ".join(info.value.errors)


def plan():
    return {
        "id": "plan",
        "name": "Hero",
        "description": "Main character",
        "kind": "character",
        "scope": "shot",
        "shotId": "unknown",
        "shotIds": [],
        "width": 1024,
        "height": 1024,
    }


def test_invented_shot_reference_is_rejected_without_model_judgment():
    operation = patch_request().model_copy(
        update={
            "source_document": {"shots": [], "assetPlans": []},
            "allowed_paths": ("/assetPlans",),
            "locked_paths": (),
        }
    )
    with pytest.raises(HarnessValidationError) as info:
        parse_structured_operation(patch_response("/assetPlans", [plan()]), operation)
    assert "unknown shot" in " ".join(info.value.errors)


def test_manually_chosen_image_dimensions_are_locked():
    original = {
        **plan(),
        "shotId": None,
        "scope": "public",
        "fulfilledByAssetId": None,
        "state": "draft",
        "resolutionSource": "manual",
    }
    operation = patch_request().model_copy(
        update={
            "source_document": {"assetPlans": [original]},
            "allowed_paths": ("/assetPlans",),
            "locked_paths": (),
        }
    )
    with pytest.raises(HarnessValidationError) as info:
        parse_structured_operation(patch_response("/assetPlans/0/width", 2048), operation)
    assert "manually locked resolution" in " ".join(info.value.errors)


def test_context_omits_unrelated_prompts_and_execution_history():
    operation = patch_request().model_copy(
        update={
            "source_document": {
                "shots": [],
                "outline": [],
                "tasks": ["large" * 999],
                "prompts": {"h3Prompts": ["unrelated"]},
            },
            "allowed_paths": ("/shots",),
            "locked_paths": (),
        }
    )
    source = json.loads(build_structured_operation_messages(operation)[1].content)["context"][
        "source_document"
    ]
    assert source == {"shots": [], "outline": []}


def test_image_context_contains_only_shots_using_this_plan():
    payload = {"shots": [{"id": "used"}, {"id": "unrelated"}]}
    assert _image_context_shots(payload, {"shotIds": ["used"]}) == [{"id": "used"}]
    assert _image_context_shots(payload, {"scope": "public", "shotIds": []}) == []


def test_h3_content_fingerprint_ignores_unrelated_revision_and_prompt_edits():
    payload = {"name": "film", "revision": 1, "prompts": {"h3Prompts": []}}
    fingerprint = _h3_input_fingerprint(payload, {"segmentId": "current"}, [], "manifest", None)
    payload.update(revision=99, prompts={"h3Prompts": ["another segment output"]})
    assert (
        _h3_input_fingerprint(payload, {"segmentId": "current"}, [], "manifest", None)
        == fingerprint
    )
    assert (
        _h3_input_fingerprint(
            payload, {"segmentId": "current"}, [], "manifest", "changed predecessor"
        )
        != fingerprint
    )
    assert _can_reuse_h3_prompt(
        {"prompt": "ready prompt", "review": {"ready": True}, "inputFingerprint": fingerprint},
        fingerprint,
    )
    assert not _can_reuse_h3_prompt(
        {"prompt": "ready prompt", "review": {"ready": False}, "inputFingerprint": fingerprint},
        fingerprint,
    )


def test_late_generated_prompt_cannot_overwrite_locked_user_content():
    existing = {"segmentId": "a", "prompt": "approved content", "locked": True}
    assert _merge_h3_prompt_entries(
        [existing], [{"segmentId": "a", "prompt": "late generated text"}]
    ) == [existing]


@pytest.mark.asyncio
async def test_plain_h3_unknown_reference_gets_one_targeted_repair(monkeypatch):
    from ai_video_generator import prompting_api

    calls = []
    outputs = iter(
        [
            "A cinematic scene follows <Picture 99> with slow movement.",
            "A cinematic scene unfolds with slow movement and gentle rain.",
        ]
    )

    async def complete(remote, messages):
        calls.append(messages)
        return next(outputs)

    monkeypatch.setattr(prompting_api, "complete_text", complete)
    output = await _generate_validated_plain_h3(
        SimpleNamespace(operation_timeout_seconds=1), request(), ()
    )
    assert "Picture 99" not in output
    assert len(calls) == 2
    assert "unknown reference" in calls[1][-1].content


@pytest.mark.asyncio
async def test_plain_h3_repair_failure_never_persists_invalid_prompt(monkeypatch):
    from ai_video_generator import prompting_api

    calls = []

    async def complete(remote, messages):
        calls.append(messages)
        return "short"

    monkeypatch.setattr(prompting_api, "complete_text", complete)
    with pytest.raises(ValueError, match="too short"):
        await _generate_validated_plain_h3(
            SimpleNamespace(operation_timeout_seconds=1), request(), ()
        )
    assert len(calls) == 2


def test_plain_prompt_requires_each_bound_asset_label():
    asset = H3AssetInput(
        asset_id="hero",
        label="<Picture 1>",
        kind=H3AssetKind.IMAGE,
        role=H3AssetPromptRole.IDENTITY,
        preservation="identity",
    )
    errors = _validate_plain_h3_prompt(
        request(assets=(asset,)),
        "A long cinematic scene description with no bound reference label.",
    )
    assert "missing reference" in " ".join(errors)


def test_content_fingerprint_does_not_depend_on_dict_order():
    assert content_fingerprint({"a": 1, "b": 2}) == content_fingerprint({"b": 2, "a": 1})
