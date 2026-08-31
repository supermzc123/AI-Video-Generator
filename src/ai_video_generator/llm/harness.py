from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .client import ChatMessage, ImageURL, ImageURLContentPart, TextContentPart
from .harness_files import load_harness
from .models import (
    StructuredOperationRequest,
    StructuredOperationResponse,
    WorkflowMappingDraft,
    WorkflowMappingRequest,
    enforce_patch_scope,
    validate_workflow_mapping,
)

MAX_REPAIR_ATTEMPTS = 3

WORKSPACE_VALUE_CONTRACTS: dict[str, object] = {
    "idea": {
        "type": "object",
        "required": ["concept", "genre", "visualStyle", "audience"],
        "properties": {
            "concept": "string",
            "genre": "string",
            "visualStyle": "string",
            "audience": "string",
        },
        "additionalProperties": False,
    },
    "outline_item": {
        "type": "object",
        "required": ["id", "title", "summary", "durationSeconds"],
        "properties": {
            "id": "string",
            "title": "string",
            "summary": "string",
            "durationSeconds": "positive number",
        },
        "additionalProperties": False,
    },
    "shot_item": {
        "type": "object",
        "required": [
            "id",
            "title",
            "summary",
            "camera",
            "seed",
            "durationSeconds",
            "locked",
        ],
        "properties": {
            "id": "string",
            "title": "string",
            "summary": "string",
            "camera": "string",
            "seed": "non-negative integer",
            "durationSeconds": "positive number",
            "motionSegments": {
                "type": "array",
                "minItems": 0,
                "items": {
                    "type": "object",
                    "required": ["id", "durationSeconds", "summary"],
                    "properties": {
                        "id": "string",
                        "durationSeconds": (
                            "4-15 seconds for first segment, "
                            "4-12 seconds thereafter"
                        ),
                        "summary": "string including this segment's action and inherited end state",
                    },
                    "additionalProperties": False,
                },
            },
            "locked": "boolean",
        },
        "additionalProperties": False,
        "forbiddenAliases": ["description", "prompt", "targetDurationSeconds"],
    },
    "asset_plan_item": {
        "type": "object",
        "required": [
            "id",
            "name",
            "description",
            "kind",
            "scope",
            "shotId",
            "shotIds",
            "fulfilledByAssetId",
            "state",
            "width",
            "height",
            "resolutionSource",
        ],
        "properties": {
            "id": "string",
            "name": "string",
            "description": "string",
            "kind": ["character", "scene", "prop", "style"],
            "scope": ["public", "shot"],
            "shotId": (
                "existing shot id when scope is shot; null only for genuinely "
                "project-wide reusable material"
            ),
            "shotIds": (
                "array of existing shot ids that genuinely use this material; an empty array is "
                "valid and means the material is currently unused. Newly planned scene or prop "
                "references must be reused across at least two shots; a scene or prop used only "
                "once stays in the shot description and is generated directly by the video model"
            ),
            "fulfilledByAssetId": "string or null",
            "state": ["draft", "ready", "satisfied", "stale"],
            "width": "integer from 64 to 4096, divisible by 8",
            "height": "integer from 64 to 4096, divisible by 8",
            "resolutionSource": ["ai", "manual", "default"],
        },
        "additionalProperties": False,
    },
    "image_prompt_item": {
        "type": "object",
        "required": [
            "id",
            "assetPlanId",
            "prompt",
            "negativePrompt",
            "workflowTemplateId",
            "harnessRevision",
            "referenceAssetIds",
            "locked",
            "revision",
        ],
        "properties": {
            "id": "string",
            "assetPlanId": "string",
            "prompt": "string",
            "negativePrompt": "string",
            "workflowTemplateId": "string or null",
            "harnessRevision": "positive integer or null",
            "referenceAssetIds": "array of existing project asset IDs",
            "locked": "boolean",
            "revision": "positive integer",
        },
        "additionalProperties": False,
    },
}


def _system_with_highest_instruction(instruction: str, highest: str) -> str:
    """Place the project-level instruction before task-specific Harness rules."""
    value = highest.strip()
    if not value:
        return instruction
    return (
        "PROJECT HIGHEST INSTRUCTION\n"
        f"{value}\n\n"
        "The following Harness documents define the task-specific rules, output "
        "format, and safety boundaries; do not weaken those contracts.\n\n"
        f"{instruction}"
    )


class JSONCompletionClient(Protocol):
    async def complete_json(self, messages: Sequence[ChatMessage]) -> str: ...


class HarnessValidationError(RuntimeError):
    def __init__(self, message: str, errors: tuple[str, ...]) -> None:
        super().__init__(message)
        self.errors = errors


T = TypeVar("T", bound=BaseModel)


def _canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_workflow_mapping_messages(request: WorkflowMappingRequest) -> tuple[ChatMessage, ...]:
    system = _system_with_highest_instruction(
        load_harness("workflow-mapping.md"), request.highest_instruction
    )
    payload = {
        "task": "workflow_binding_mapping",
        "contract": {
            "requested_semantics": [value.value for value in request.requested_semantics],
            "response_schema": WorkflowMappingDraft.model_json_schema(),
        },
        "context": request.model_dump(mode="json"),
    }
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_canonical_json(payload)),
    )


def build_structured_operation_messages(
    request: StructuredOperationRequest,
    *,
    asset_image_urls: tuple[str, ...] = (),
) -> tuple[ChatMessage, ...]:
    system = _system_with_highest_instruction(
        load_harness("project-lead.md"), request.highest_instruction
    )
    payload = {
        "task": request.operation,
        "contract": {
            "allowed_paths": request.allowed_paths,
            "locked_paths": request.locked_paths,
            "response_schema": StructuredOperationResponse.model_json_schema(),
            "workspace_value_contracts": WORKSPACE_VALUE_CONTRACTS,
            "motion_context_contract": {
                "required_when": [
                    "shot duration exceeds the single-segment limit",
                    (
                        "a continuous picture spans multiple H3 executions and must inherit exact "
                        "action, pose, camera, movement direction, composition, lighting, "
                        "environment, or audio state"
                    ),
                ],
                "optional_when": "ordinary single-segment shots",
                "topology_rule": (
                    "represent an uninterrupted continuous take as one shot with motionSegments; "
                    "use separate shots only for an intentional cut or continuity reset"
                ),
                "segment_duration_sum": "must equal shot.durationSeconds",
                "first_segment_seconds": {"minimum": 4, "maximum": 15},
                "continuation_segment_seconds": {"minimum": 4, "maximum": 12},
                "summary_must_include": ["action", "end_state_for_next_segment"],
                "continuation_field": (
                    "continuationOf is derived by the system from array order; "
                    "do not emit it on shots"
                ),
            },
            "asset_planning_policy": {
                "one_off_scene_or_prop": (
                    "do not create an asset plan; describe it in the shot and let the video model "
                    "generate it directly"
                ),
                "reusable_scene_or_prop": (
                    "create one asset plan only when the same identifiable material is used by at "
                    "least two shots"
                ),
                "exceptions": ["character identity", "project-wide visual style"],
                "uploaded_assets": "preserve as project facts even when used once",
            },
            "field_name_policy": (
                "Use the exact camelCase workspace field names. Do not emit aliases. "
                "When replacing an array, every item must be complete and match its item contract."
            ),
        },
        "context": request.model_dump(mode="json"),
    }
    user_content: str | tuple[TextContentPart | ImageURLContentPart, ...] = _canonical_json(
        payload
    )
    if asset_image_urls:
        user_content = (
            TextContentPart(text=user_content),
            *(
                ImageURLContentPart(image_url=ImageURL(url=url, detail="low"))
                for url in asset_image_urls
            ),
        )
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user_content),
    )


def _decode_json_object(content: str) -> object:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json\n"):
                stripped = stripped.lstrip()[5:]
    return json.loads(stripped)


def parse_workflow_mapping(
    content: str,
    request: WorkflowMappingRequest,
) -> WorkflowMappingDraft:
    try:
        draft = WorkflowMappingDraft.model_validate(_decode_json_object(content))
        validate_workflow_mapping(request, draft)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise HarnessValidationError("invalid workflow mapping", (str(exc),)) from exc
    return draft


def parse_structured_operation(
    content: str,
    request: StructuredOperationRequest,
) -> StructuredOperationResponse:
    try:
        response = StructuredOperationResponse.model_validate(_decode_json_object(content))
        enforce_patch_scope(request, response)
        _validate_workspace_patch_values(response)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise HarnessValidationError("invalid structured operation", (str(exc),)) from exc
    return response


def _require_complete_object(
    value: object,
    *,
    required: set[str],
    allowed: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")
    return value


def _validate_workspace_patch_values(response: StructuredOperationResponse) -> None:
    collections = {
        "/outline": {
            "id",
            "title",
            "summary",
            "durationSeconds",
        },
        "/shots": {
            "id",
            "title",
            "summary",
            "camera",
            "seed",
            "durationSeconds",
            "locked",
        },
        "/assetPlans": {
            "id",
            "name",
            "description",
            "kind",
            "scope",
            "shotId",
            "shotIds",
            "fulfilledByAssetId",
            "state",
            "width",
            "height",
            "resolutionSource",
        },
        "/prompts/imagePrompts": {
            "id",
            "assetPlanId",
            "prompt",
            "negativePrompt",
            "workflowTemplateId",
            "harnessRevision",
            "referenceAssetIds",
            "locked",
            "revision",
        },
    }
    optional_fields = {"/shots": {"motionSegments"}, "/assetPlans": {"shotIds"}}
    for patch in response.patches:
        if patch.op.value == "remove":
            continue
        for root, fields in collections.items():
            if patch.path == root:
                if not isinstance(patch.value, list):
                    raise ValueError(f"{root} replacement must be an array")
                for index, item in enumerate(patch.value):
                    _require_complete_object(
                        item,
                        required=fields,
                        allowed=fields | optional_fields.get(root, set()),
                        label=f"{root}[{index}]",
                    )
                    _validate_workspace_item_types(root, item, f"{root}[{index}]")
            elif patch.path.startswith(f"{root}/"):
                remainder = patch.path[len(root) + 1 :]
                if "/" not in remainder:
                    item = _require_complete_object(
                        patch.value,
                        required=fields,
                        allowed=fields | optional_fields.get(root, set()),
                        label=patch.path,
                    )
                    _validate_workspace_item_types(root, item, patch.path)


def _validate_workspace_item_types(
    root: str, item: dict[str, object], label: str
) -> None:
    def is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    string_fields = {
        "/outline": ("id", "title", "summary"),
        "/shots": ("id", "title", "summary", "camera"),
        "/assetPlans": ("id", "name", "description"),
        "/prompts/imagePrompts": (
            "id",
            "assetPlanId",
            "prompt",
            "negativePrompt",
        ),
    }[root]
    for field in string_fields:
        if not isinstance(item[field], str):
            raise ValueError(f"{label}.{field} must be a string")
    if root in {"/outline", "/shots"}:
        duration = item["durationSeconds"]
        if not is_number(duration) or duration <= 0:
            raise ValueError(f"{label}.durationSeconds must be a positive number")
    if root == "/shots":
        seed = item["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"{label}.seed must be a non-negative integer")
        if not isinstance(item["locked"], bool):
            raise ValueError(f"{label}.locked must be a boolean")
        segments = item.get("motionSegments", [])
        if segments is None:
            segments = []
        if not isinstance(segments, list):
            raise ValueError(f"{label}.motionSegments must be an array when provided")
        if not segments:
            return
        total = 0.0
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise ValueError(f"{label}.motionSegments[{index}] must be an object")
            required = {"id", "durationSeconds", "summary"}
            if set(segment) != required:
                raise ValueError(
                    f"{label}.motionSegments[{index}] must contain exactly "
                    "id, durationSeconds, summary"
                )
            if not isinstance(segment["id"], str) or not segment["id"].strip():
                raise ValueError(f"{label}.motionSegments[{index}].id must be a string")
            duration = segment["durationSeconds"]
            limit = 15.0 if index == 0 else 12.0
            if not is_number(duration) or not 4.0 <= float(duration) <= limit:
                raise ValueError(
                    f"{label}.motionSegments[{index}].durationSeconds must be "
                    f"between 4 and {limit:g} seconds"
                )
            if not isinstance(segment["summary"], str) or not segment["summary"].strip():
                raise ValueError(f"{label}.motionSegments[{index}].summary must be non-empty")
            total += float(duration)
        if abs(total - float(item["durationSeconds"])) > 0.01:
            raise ValueError(
                f"{label}.motionSegments durations must sum to durationSeconds"
            )
    if root == "/assetPlans":
        if item["kind"] not in {"character", "scene", "prop", "style"}:
            raise ValueError(f"{label}.kind is invalid")
        if item["scope"] not in {"public", "shot"}:
            raise ValueError(f"{label}.scope is invalid")
        if item["shotId"] is not None and not isinstance(item["shotId"], str):
            raise ValueError(f"{label}.shotId must be a string or null")
        shot_ids = item.get("shotIds", [])
        if not isinstance(shot_ids, list) or any(
            not isinstance(value, str) or not value.strip() for value in shot_ids
        ):
            raise ValueError(f"{label}.shotIds must be an array of non-empty strings")
        if item["fulfilledByAssetId"] is not None and not isinstance(
            item["fulfilledByAssetId"], str
        ):
            raise ValueError(f"{label}.fulfilledByAssetId must be a string or null")
        if item["state"] not in {"draft", "ready", "satisfied", "stale"}:
            raise ValueError(f"{label}.state is invalid")
        for field in ("width", "height"):
            value = item[field]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 64 <= value <= 4096
                or value % 8
            ):
                raise ValueError(
                    f"{label}.{field} must be an integer from 64 to 4096 divisible by 8"
                )
        if item["resolutionSource"] not in {"ai", "manual", "default"}:
            raise ValueError(f"{label}.resolutionSource is invalid")
    if root == "/prompts/imagePrompts":
        if item["workflowTemplateId"] is not None and not isinstance(
            item["workflowTemplateId"], str
        ):
            raise ValueError(f"{label}.workflowTemplateId must be a string or null")
        if item["harnessRevision"] is not None and (
            not isinstance(item["harnessRevision"], int)
            or item["harnessRevision"] < 1
        ):
            raise ValueError(f"{label}.harnessRevision must be a positive integer or null")
        references = item["referenceAssetIds"]
        if not isinstance(references, list) or any(
            not isinstance(asset_id, str) for asset_id in references
        ):
            raise ValueError(f"{label}.referenceAssetIds must be an array of strings")
        if not isinstance(item["locked"], bool):
            raise ValueError(f"{label}.locked must be a boolean")
        if not isinstance(item["revision"], int) or item["revision"] < 1:
            raise ValueError(f"{label}.revision must be a positive integer")


def _repair_messages(
    base_messages: tuple[ChatMessage, ...],
    invalid_content: str,
    error: HarnessValidationError,
) -> tuple[ChatMessage, ...]:
    repair = {
        "task": "repair_invalid_response",
        "validation_errors": error.errors,
        "instruction": "Return a corrected JSON object only. Preserve valid fields.",
    }
    return (
        *base_messages,
        ChatMessage(role="assistant", content=invalid_content),
        ChatMessage(role="user", content=_canonical_json(repair)),
    )


class LLMHarness:
    def __init__(
        self,
        client: JSONCompletionClient,
        *,
        max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    ) -> None:
        if not 0 <= max_repair_attempts <= MAX_REPAIR_ATTEMPTS:
            raise ValueError(f"max_repair_attempts must be between 0 and {MAX_REPAIR_ATTEMPTS}")
        self._client = client
        self._max_repair_attempts = max_repair_attempts

    async def map_workflow(self, request: WorkflowMappingRequest) -> WorkflowMappingDraft:
        return await self._run_with_repairs(
            build_workflow_mapping_messages(request),
            lambda content: parse_workflow_mapping(content, request),
        )

    async def propose_patch(
        self,
        request: StructuredOperationRequest,
        *,
        asset_image_urls: tuple[str, ...] = (),
    ) -> StructuredOperationResponse:
        return await self._run_with_repairs(
            build_structured_operation_messages(
                request, asset_image_urls=asset_image_urls
            ),
            lambda content: parse_structured_operation(content, request),
        )

    async def _run_with_repairs(
        self,
        base_messages: tuple[ChatMessage, ...],
        parser: Callable[[str], T],
    ) -> T:
        messages = base_messages
        failures: list[str] = []
        for attempt in range(self._max_repair_attempts + 1):
            content = await self._client.complete_json(messages)
            try:
                return parser(content)
            except HarnessValidationError as exc:
                failures.extend(exc.errors)
                if attempt == self._max_repair_attempts:
                    raise HarnessValidationError(
                        "LLM response remained invalid after repair attempts",
                        tuple(failures),
                    ) from exc
                messages = _repair_messages(base_messages, content, exc)
        raise AssertionError("repair loop must return or raise")
