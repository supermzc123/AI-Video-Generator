from __future__ import annotations

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
) -> None:
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
                    f"duplicate {binding.semantic.value} reference_index: "
                    f"{binding.reference_index}"
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
