import json
from collections.abc import Sequence

import pytest

from ai_video_generator.domain import ProjectSpec, ShotSpec
from ai_video_generator.llm import (
    ChatMessage,
    HarnessValidationError,
    LLMHarness,
    StructuredOperationRequest,
    WorkflowMappingRequest,
    build_structured_operation_messages,
    build_workflow_mapping_messages,
    parse_structured_operation,
    parse_workflow_mapping,
)


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete_json(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


def project() -> ProjectSpec:
    return ProjectSpec(
        project_id="project-1",
        name="Film",
        width=1024,
        height=608,
        target_duration_seconds=30,
    )


def shot() -> ShotSpec:
    return ShotSpec(
        shot_revision_id="shot-1@1",
        project_id="project-1",
        ordinal=1,
        title="Arrival",
        description="The protagonist enters the station.",
        target_duration_seconds=12,
    )


def mapping_request() -> WorkflowMappingRequest:
    return WorkflowMappingRequest(
        operation_id="map-1",
        workflow_id="workflow-1",
        project=project(),
        shot=shot(),
        raw_workflow={
            "2": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 1024, "height": 608},
            },
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "old prompt"},
            },
        },
    )


def valid_mapping() -> str:
    return json.dumps(
        {
            "operation_id": "map-1",
            "bindings": [
                {
                    "binding_id": "prompt",
                    "semantic": "prompt",
                    "node_id": "1",
                    "input_name": "text",
                    "value_type": "string",
                    "title": "Prompt",
                    "default_value": "old prompt",
                    "confidence": 0.98,
                    "rationale": "Text encoder input",
                }
            ],
            "warnings": [],
        }
    )


def test_workflow_mapping_prompt_is_deterministic_and_contains_contracts() -> None:
    request = mapping_request()

    first = build_workflow_mapping_messages(request)
    second = build_workflow_mapping_messages(request)

    assert first == second
    payload = json.loads(first[1].content)
    assert payload["context"]["project"]["project_id"] == "project-1"
    assert payload["context"]["shot"]["shot_revision_id"] == "shot-1@1"
    assert payload["context"]["raw_workflow"]["1"]["inputs"]["text"] == "old prompt"
    assert "response_schema" in payload["contract"]


def test_mapping_parser_rejects_invented_node_or_input() -> None:
    data = json.loads(valid_mapping())
    data["bindings"][0]["node_id"] = "999"

    with pytest.raises(HarnessValidationError, match="invalid workflow mapping"):
        parse_workflow_mapping(json.dumps(data), mapping_request())


def test_mapping_parser_rejects_connected_inputs_that_would_change_topology() -> None:
    data = json.loads(valid_mapping())
    data["bindings"][0]["node_id"] = "2"
    data["bindings"][0]["input_name"] = "width"
    request = mapping_request().model_copy(
        update={
            "raw_workflow": {
                **mapping_request().raw_workflow,
                "2": {
                    "class_type": "ResizeImage",
                    "inputs": {"width": ["1", 0]},
                },
            }
        }
    )

    with pytest.raises(HarnessValidationError, match="invalid workflow mapping"):
        parse_workflow_mapping(json.dumps(data), request)


@pytest.mark.asyncio
async def test_harness_repairs_invalid_mapping_then_returns_valid_draft() -> None:
    client = FakeClient(["not json", valid_mapping()])
    harness = LLMHarness(client)

    result = await harness.map_workflow(mapping_request())

    assert result.bindings[0].binding_id == "prompt"
    assert len(client.calls) == 2
    repair_payload = json.loads(client.calls[1][-1].content)
    assert repair_payload["task"] == "repair_invalid_response"
    assert repair_payload["validation_errors"]


@pytest.mark.asyncio
async def test_harness_allows_at_most_three_repair_attempts() -> None:
    client = FakeClient(["invalid"] * 4)
    harness = LLMHarness(client)

    with pytest.raises(HarnessValidationError, match="remained invalid"):
        await harness.map_workflow(mapping_request())

    assert len(client.calls) == 4


def patch_request() -> StructuredOperationRequest:
    return StructuredOperationRequest(
        operation_id="edit-1",
        operation="revise_shot",
        instruction="Make the action more urgent without changing the title.",
        project=project(),
        shot=shot(),
        source_document={
            "title": "Arrival",
            "content": {"action": "walks", "camera": "wide"},
        },
        allowed_paths=("/content",),
        locked_paths=("/content/camera",),
    )


def test_patch_parser_accepts_allowed_unlocked_descendant() -> None:
    result = parse_structured_operation(
        json.dumps(
            {
                "operation_id": "edit-1",
                "patches": [{"op": "replace", "path": "/content/action", "value": "runs"}],
                "rationale": "Increase urgency",
                "warnings": [],
            }
        ),
        patch_request(),
    )

    assert result.patches[0].path == "/content/action"


def test_patch_parser_allows_conversation_without_document_changes() -> None:
    result = parse_structured_operation(
        json.dumps(
            {
                "operation_id": "edit-1",
                "patches": [],
                "rationale": "The current camera choice supports the intended tension.",
                "warnings": [],
            }
        ),
        patch_request(),
    )

    assert result.patches == ()


def test_storyboard_contract_rejects_aliases_and_incomplete_shots() -> None:
    request = patch_request().model_copy(
        update={
            "operation": "initialize_storyboard",
            "source_document": {"shots": []},
            "allowed_paths": ("/shots",),
            "locked_paths": (),
        }
    )
    response = {
        "operation_id": "edit-1",
        "patches": [
            {
                "op": "replace",
                "path": "/shots",
                "value": [
                    {
                        "id": "shot-1",
                        "title": "Arrival",
                        "description": "A person arrives.",
                        "prompt": "wide shot",
                        "durationSeconds": 6,
                    }
                ],
            }
        ],
        "rationale": "Create the storyboard",
        "warnings": [],
    }

    with pytest.raises(HarnessValidationError, match="invalid structured operation"):
        parse_structured_operation(json.dumps(response), request)


def test_structured_prompt_contains_exact_workspace_contracts() -> None:
    messages = build_structured_operation_messages(patch_request())
    assert "immediately applies non-empty patches" in messages[0].content
    assert "Never ask the user to approve" in messages[0].content
    payload = json.loads(messages[1].content)

    shot_contract = payload["contract"]["workspace_value_contracts"]["shot_item"]
    assert "h3Prompt" not in shot_contract["required"]
    assert "harnessRevision" not in shot_contract["required"]
    assert "description" in shot_contract["forbiddenAliases"]
    assert "asset_plan_item" in payload["contract"]["workspace_value_contracts"]
    assert "image_prompt_item" in payload["contract"]["workspace_value_contracts"]
    asset_contract = payload["contract"]["workspace_value_contracts"]["asset_plan_item"]
    assert {"width", "height", "resolutionSource"}.issubset(asset_contract["required"])
    motion = payload["contract"]["motion_context_contract"]
    assert motion["segment_duration_sum"] == "must equal shot.durationSeconds"
    assert motion["first_segment_seconds"] == {"minimum": 4, "maximum": 15}
    assert "motionSegments" not in shot_contract["required"]
    assert "motionSegments" in shot_contract["properties"]
    assert motion["optional_when"] == "ordinary single-segment shots"
    assert "shotIds may be empty" in messages[0].content
    assert "absolutely never add" in messages[0].content
    assert "scope=public means reusable, not automatically used" in messages[0].content
    assert "an empty array is valid" in asset_contract["properties"]["shotIds"]


def test_storyboard_contract_allows_single_segment_shot_without_motion_context() -> None:
    request = patch_request().model_copy(
        update={
            "operation": "initialize_storyboard",
            "source_document": {"shots": []},
            "allowed_paths": ("/shots",),
            "locked_paths": (),
        }
    )
    shot = {
        "id": "shot-1",
        "title": "Arrival",
        "summary": "A person arrives.",
        "camera": "wide shot",
        "seed": 1,
        "durationSeconds": 8,
        "locked": False,
    }
    response = {
        "operation_id": "edit-1",
        "patches": [{"op": "replace", "path": "/shots", "value": [shot]}],
        "rationale": "Create a single continuous shot",
        "warnings": [],
    }
    parsed = parse_structured_operation(json.dumps(response), request)
    assert parsed.patches[0].value == [shot]


def test_storyboard_contract_allows_empty_motion_context_array() -> None:
    request = patch_request().model_copy(
        update={
            "operation": "initialize_storyboard",
            "source_document": {"shots": []},
            "allowed_paths": ("/shots",),
            "locked_paths": (),
        }
    )
    shot = {
        "id": "shot-1",
        "title": "Arrival",
        "summary": "A person arrives.",
        "camera": "wide shot",
        "seed": 1,
        "durationSeconds": 8,
        "motionSegments": [],
        "locked": False,
    }
    response = {
        "operation_id": "edit-1",
        "patches": [{"op": "replace", "path": "/shots", "value": [shot]}],
        "rationale": "Create a single continuous shot",
        "warnings": [],
    }
    parse_structured_operation(json.dumps(response), request)


def test_storyboard_contract_rejects_invalid_motion_context_segments() -> None:
    request = patch_request().model_copy(
        update={
            "operation": "initialize_storyboard",
            "source_document": {"shots": []},
            "allowed_paths": ("/shots",),
            "locked_paths": (),
        }
    )
    shot = {
        "id": "shot-1",
        "title": "Arrival",
        "summary": "A person arrives.",
        "camera": "wide shot",
        "seed": 1,
        "durationSeconds": 16,
        "motionSegments": [
            {"id": "m1", "durationSeconds": 16, "summary": "arrives"}
        ],
        "locked": False,
    }
    response = {
        "operation_id": "edit-1",
        "patches": [{"op": "replace", "path": "/shots", "value": [shot]}],
        "rationale": "Create the storyboard",
        "warnings": [],
    }
    with pytest.raises(HarnessValidationError, match="invalid structured operation"):
        parse_structured_operation(json.dumps(response), request)


def test_structured_prompt_can_include_untrusted_project_images() -> None:
    messages = build_structured_operation_messages(
        patch_request(), asset_image_urls=("data:image/jpeg;base64,AAAA",)
    )

    assert isinstance(messages[1].content, tuple)
    assert messages[1].content[0].type == "text"
    assert messages[1].content[1].type == "image_url"


def test_image_prompt_patch_requires_a_complete_typed_item() -> None:
    request = patch_request().model_copy(
        update={
            "operation": "write_asset_image_prompt",
            "source_document": {"prompts": {"imagePrompts": []}},
            "allowed_paths": ("/prompts/imagePrompts/-",),
            "locked_paths": (),
        }
    )
    incomplete = {
        "operation_id": "edit-1",
        "patches": [
            {
                "op": "add",
                "path": "/prompts/imagePrompts/-",
                "value": {"assetPlanId": "plan-1", "prompt": "角色设定图"},
            }
        ],
        "rationale": "Create an image prompt",
        "warnings": [],
    }

    with pytest.raises(HarnessValidationError, match="invalid structured operation"):
        parse_structured_operation(json.dumps(incomplete), request)


@pytest.mark.parametrize("path", ["/title", "/content/camera", "/content"])
def test_patch_parser_rejects_disallowed_or_locked_paths(path: str) -> None:
    response = {
        "operation_id": "edit-1",
        "patches": [{"op": "replace", "path": path, "value": "changed"}],
        "rationale": "Change",
        "warnings": [],
    }

    with pytest.raises(HarnessValidationError, match="invalid structured operation"):
        parse_structured_operation(json.dumps(response), patch_request())


def test_remove_patch_must_not_include_a_value() -> None:
    response = {
        "operation_id": "edit-1",
        "patches": [{"op": "remove", "path": "/content/action", "value": None}],
        "rationale": "Remove action",
    }

    with pytest.raises(HarnessValidationError, match="invalid structured operation"):
        parse_structured_operation(json.dumps(response), patch_request())
