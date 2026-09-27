from __future__ import annotations

import copy
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_video_generator.domain.project import ProjectSpec, ShotSpec
from ai_video_generator.domain.workflow import (
    BindingSemantic,
    BindingValueType,
    WorkflowBindingDraft,
)


class HarnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowMappingRequest(HarnessModel):
    operation_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    project: ProjectSpec
    shot: ShotSpec | None = None
    raw_workflow: dict[str, dict[str, Any]] = Field(min_length=1)
    requested_semantics: tuple[BindingSemantic, ...] = tuple(BindingSemantic)
    instructions: str = "Identify safe user-editable workflow inputs."
    highest_instruction: str = ""

    @model_validator(mode="after")
    def validate_context(self) -> WorkflowMappingRequest:
        if self.shot is not None and self.shot.project_id != self.project.project_id:
            raise ValueError("shot and project IDs must match")
        if not self.requested_semantics:
            raise ValueError("at least one binding semantic must be requested")
        if len(self.requested_semantics) != len(set(self.requested_semantics)):
            raise ValueError("requested semantics must be unique")
        return self


class WorkflowMappingDraft(HarnessModel):
    operation_id: str = Field(min_length=1)
    bindings: tuple[WorkflowBindingDraft, ...]
    warnings: tuple[str, ...] = ()


class JsonPatchOp(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


def validate_json_pointer(path: str) -> str:
    if path != "" and not path.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    if re.search(r"~(?![01])", path):
        raise ValueError("JSON pointer contains an invalid escape")
    return path


class JsonPatchOperation(HarnessModel):
    op: JsonPatchOp
    path: str
    value: Any = None

    @model_validator(mode="after")
    def validate_operation(self) -> JsonPatchOperation:
        validate_json_pointer(self.path)
        value_was_supplied = "value" in self.model_fields_set
        if self.op == JsonPatchOp.REMOVE and value_was_supplied:
            raise ValueError("remove operations must not include value")
        if self.op != JsonPatchOp.REMOVE and not value_was_supplied:
            raise ValueError("add and replace operations require value")
        return self


class StructuredOperationRequest(HarnessModel):
    operation_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    project: ProjectSpec
    shot: ShotSpec | None = None
    source_document: dict[str, Any]
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    locked_paths: tuple[str, ...] = ()
    highest_instruction: str = ""
    reference_asset_ids: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> StructuredOperationRequest:
        if self.shot is not None and self.shot.project_id != self.project.project_id:
            raise ValueError("shot and project IDs must match")
        for path in (*self.allowed_paths, *self.locked_paths):
            validate_json_pointer(path)
        if len(self.allowed_paths) != len(set(self.allowed_paths)):
            raise ValueError("allowed paths must be unique")
        if len(self.locked_paths) != len(set(self.locked_paths)):
            raise ValueError("locked paths must be unique")
        return self


class StructuredOperationResponse(HarnessModel):
    operation_id: str = Field(min_length=1)
    patches: tuple[JsonPatchOperation, ...] = ()
    rationale: str = Field(min_length=1)
    warnings: tuple[str, ...] = ()


def pointer_contains(parent: str, child: str) -> bool:
    return parent == "" or child == parent or child.startswith(f"{parent}/")


def enforce_patch_scope(
    request: StructuredOperationRequest,
    response: StructuredOperationResponse,
) -> dict[str, Any]:
    if response.operation_id != request.operation_id:
        raise ValueError("response operation_id does not match request")

    for patch in response.patches:
        if not any(pointer_contains(path, patch.path) for path in request.allowed_paths):
            raise ValueError(f"patch path is outside the allowed paths: {patch.path}")
        if any(
            pointer_contains(locked, patch.path) or pointer_contains(patch.path, locked)
            for locked in request.locked_paths
        ):
            raise ValueError(f"patch path overlaps a locked path: {patch.path}")
    candidate = _apply_validation_patches(request.source_document, response)
    for root in ("shots", "assetPlans"):
        _preserve_locked_items(request.source_document.get(root), candidate.get(root))
    source_prompts = request.source_document.get("prompts", {})
    candidate_prompts = candidate.get("prompts", {})
    if isinstance(source_prompts, dict) and isinstance(candidate_prompts, dict):
        for key in ("imagePrompts", "h3Prompts"):
            _preserve_locked_items(source_prompts.get(key), candidate_prompts.get(key))
    _validate_references(request, candidate)
    return candidate


def _preserve_locked_items(original: object, candidate: object) -> None:
    if not isinstance(original, list):
        return
    values = candidate if isinstance(candidate, list) else []
    for item in original:
        if not isinstance(item, dict) or item.get("locked") is not True:
            continue
        identity = item.get("id") or item.get("segmentId") or item.get("assetPlanId")
        if not any(value == item for value in values):
            raise ValueError(f"patch changes or removes locked item: {identity}")


def _apply_validation_patches(
    source: dict[str, Any], response: StructuredOperationResponse
) -> dict[str, Any]:
    document = copy.deepcopy(source)
    for patch in response.patches:
        if not patch.path:
            raise ValueError("root document patches are not allowed")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in patch.path[1:].split("/")]
        parent: Any = document
        for part in parts[:-1]:
            if isinstance(parent, dict) and part in parent:
                parent = parent[part]
            elif isinstance(parent, list) and part.isdigit() and int(part) < len(parent):
                parent = parent[int(part)]
            else:
                raise ValueError(f"patch parent does not exist: {patch.path}")
        key = parts[-1]
        if isinstance(parent, dict):
            if patch.op != JsonPatchOp.ADD and key not in parent:
                raise ValueError(f"patch target does not exist: {patch.path}")
            if patch.op == JsonPatchOp.REMOVE:
                del parent[key]
            else:
                parent[key] = patch.value
        elif isinstance(parent, list):
            if patch.op == JsonPatchOp.ADD and key == "-":
                parent.append(patch.value)
                continue
            if not key.isdigit() or int(key) >= len(parent):
                raise ValueError(f"invalid array patch target: {patch.path}")
            index = int(key)
            if patch.op == JsonPatchOp.REMOVE:
                parent.pop(index)
            elif patch.op == JsonPatchOp.ADD:
                parent.insert(index, patch.value)
            else:
                parent[index] = patch.value
        else:
            raise ValueError(f"patch target is scalar: {patch.path}")
    return document


def _validate_references(request: StructuredOperationRequest, candidate: dict[str, Any]) -> None:
    shots = {str(item.get("id")) for item in candidate.get("shots", []) if isinstance(item, dict)}
    original_plans = {
        str(item.get("id")): item
        for item in request.source_document.get("assetPlans", [])
        if isinstance(item, dict)
    }
    for plan in candidate.get("assetPlans", []):
        if not isinstance(plan, dict):
            continue
        if plan == original_plans.get(str(plan.get("id"))):
            continue
        old = original_plans.get(str(plan.get("id")), {})
        if old.get("resolutionSource") == "manual" and any(
            plan.get(field) != old.get(field) for field in ("width", "height", "resolutionSource")
        ):
            raise ValueError("asset plan changes manually locked resolution")
        if old and plan.get("fulfilledByAssetId") != old.get("fulfilledByAssetId"):
            raise ValueError("asset binding is managed by the application")
        used = list(plan.get("shotIds") or [])
        if plan.get("shotId"):
            used.append(plan["shotId"])
        if any(shot not in shots for shot in used):
            raise ValueError("asset plan references an unknown shot")
        if (
            plan.get("fulfilledByAssetId")
            and request.reference_asset_ids is not None
            and plan["fulfilledByAssetId"] not in request.reference_asset_ids
        ):
            raise ValueError("asset plan references an unknown project asset")
    prompts = candidate.get("prompts", {})
    if request.reference_asset_ids is not None and isinstance(prompts, dict):
        for prompt in prompts.get("imagePrompts", []):
            if isinstance(prompt, dict) and any(
                value not in request.reference_asset_ids
                for value in prompt.get("referenceAssetIds", [])
            ):
                raise ValueError("image prompt references an unknown project asset")


def validate_workflow_mapping(
    request: WorkflowMappingRequest,
    draft: WorkflowMappingDraft,
) -> None:
    if draft.operation_id != request.operation_id:
        raise ValueError("response operation_id does not match request")

    binding_ids: set[str] = set()
    targets: set[tuple[str, str]] = set()
    reference_indexes: set[tuple[BindingSemantic, int]] = set()
    requested = set(request.requested_semantics)

    for binding in draft.bindings:
        if binding.binding_id in binding_ids:
            raise ValueError(f"duplicate binding_id: {binding.binding_id}")
        binding_ids.add(binding.binding_id)

        target = (binding.node_id, binding.input_name)
        if target in targets:
            raise ValueError(
                f"multiple bindings target node {binding.node_id} input {binding.input_name}"
            )
        targets.add(target)

        if binding.semantic not in requested:
            raise ValueError(f"unrequested binding semantic: {binding.semantic.value}")
        node = request.raw_workflow.get(binding.node_id)
        if node is None:
            raise ValueError(f"binding references unknown node: {binding.node_id}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or binding.input_name not in inputs:
            raise ValueError(
                f"binding references unknown input: {binding.node_id}.{binding.input_name}"
            )
        input_value = inputs[binding.input_name]
        if (
            isinstance(input_value, (list, tuple))
            and len(input_value) == 2
            and isinstance(input_value[0], str)
            and isinstance(input_value[1], int)
            and not isinstance(input_value[1], bool)
        ):
            raise ValueError(
                f"binding targets connected input: {binding.node_id}.{binding.input_name}"
            )

        if binding.reference_index is not None:
            reference_key = (binding.semantic, binding.reference_index)
            if reference_key in reference_indexes:
                raise ValueError(
                    f"duplicate {binding.semantic.value} reference_index: {binding.reference_index}"
                )
            reference_indexes.add(reference_key)

        _validate_default_value(binding)


def _validate_default_value(binding: WorkflowBindingDraft) -> None:
    value = binding.default_value
    if value is None:
        return
    if binding.value_type == BindingValueType.STRING and not isinstance(value, str):
        raise ValueError(f"binding {binding.binding_id} default must be a string")
    if binding.value_type in {
        BindingValueType.IMAGE_PATH,
        BindingValueType.VIDEO_PATH,
        BindingValueType.AUDIO_PATH,
    } and not isinstance(value, str):
        raise ValueError(f"binding {binding.binding_id} default must be an image path")
    if binding.value_type == BindingValueType.INTEGER and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        raise ValueError(f"binding {binding.binding_id} default must be an integer")
    if binding.value_type == BindingValueType.NUMBER and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ValueError(f"binding {binding.binding_id} default must be a number")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if binding.minimum is not None and value < binding.minimum:
            raise ValueError(f"binding {binding.binding_id} default is below minimum")
        if binding.maximum is not None and value > binding.maximum:
            raise ValueError(f"binding {binding.binding_id} default is above maximum")
