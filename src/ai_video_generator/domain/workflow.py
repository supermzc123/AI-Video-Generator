from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .chain import SHA256_PATTERN, FrozenModel


class WorkflowApproval(StrEnum):
    DRAFT = "draft"
    NEEDS_CONFIRMATION = "needs_confirmation"
    APPROVED = "approved"
    REJECTED = "rejected"


class BindingSemantic(StrEnum):
    PROMPT = "prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    WIDTH = "width"
    HEIGHT = "height"
    STEPS = "steps"
    CFG = "cfg"
    SEED = "seed"
    BATCH_SIZE = "batch_size"
    REFERENCE_IMAGE = "reference_image"
    SOURCE_VIDEO = "source_video"
    MODEL = "model"
    INTERPOLATION_FACTOR = "interpolation_factor"
    UPSCALE_FACTOR = "upscale_factor"
    LANGUAGE = "language"


class BindingValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    IMAGE_PATH = "image_path"
    VIDEO_PATH = "video_path"


class WorkflowOutputType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    SUBTITLE = "subtitle"


class WorkflowBinding(FrozenModel):
    binding_id: str = Field(min_length=1)
    semantic: BindingSemantic
    node_id: str = Field(min_length=1)
    input_name: str = Field(min_length=1)
    value_type: BindingValueType
    title: str = Field(min_length=1)
    default_value: Any = None
    string_template: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    reference_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_binding(self) -> WorkflowBinding:
        expected_types = {
            BindingSemantic.PROMPT: BindingValueType.STRING,
            BindingSemantic.NEGATIVE_PROMPT: BindingValueType.STRING,
            BindingSemantic.WIDTH: BindingValueType.INTEGER,
            BindingSemantic.HEIGHT: BindingValueType.INTEGER,
            BindingSemantic.STEPS: BindingValueType.INTEGER,
            BindingSemantic.CFG: BindingValueType.NUMBER,
            BindingSemantic.SEED: BindingValueType.INTEGER,
            BindingSemantic.BATCH_SIZE: BindingValueType.INTEGER,
            BindingSemantic.REFERENCE_IMAGE: BindingValueType.IMAGE_PATH,
            BindingSemantic.SOURCE_VIDEO: BindingValueType.VIDEO_PATH,
            BindingSemantic.MODEL: BindingValueType.STRING,
            BindingSemantic.INTERPOLATION_FACTOR: BindingValueType.NUMBER,
            BindingSemantic.UPSCALE_FACTOR: BindingValueType.NUMBER,
            BindingSemantic.LANGUAGE: BindingValueType.STRING,
        }
        if self.value_type != expected_types[self.semantic]:
            required_type = expected_types[self.semantic].value
            raise ValueError(f"{self.semantic.value} requires {required_type}")
        if self.string_template is not None and self.value_type != BindingValueType.STRING:
            raise ValueError("string_template is only valid for string bindings")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("binding minimum must not exceed maximum")
        if self.semantic == BindingSemantic.REFERENCE_IMAGE and self.reference_index is None:
            raise ValueError("reference image bindings require reference_index")
        if self.semantic != BindingSemantic.REFERENCE_IMAGE and self.reference_index is not None:
            raise ValueError("reference_index is only valid for reference image bindings")
        return self


class WorkflowBindingDraft(WorkflowBinding):
    confidence: float = Field(default=1.0, ge=0, le=1)
    rationale: str = Field(default="", max_length=500)


class WorkflowOutput(FrozenModel):
    output_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    output_type: WorkflowOutputType = WorkflowOutputType.IMAGE
    title: str = Field(min_length=1)


class WorkflowTemplate(FrozenModel):
    schema_version: str = "1.0"
    template_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(
        default="image", pattern="^(image|interpolation|restoration|transcription)$"
    )
    workflow_sha256: str = Field(pattern=SHA256_PATTERN)
    node_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    raw_workflow: dict[str, dict[str, Any]]
    bindings: tuple[WorkflowBinding, ...] = ()
    outputs: tuple[WorkflowOutput, ...] = Field(min_length=1)
    required_node_types: tuple[str, ...] = Field(min_length=1)
    unknown_node_types: tuple[str, ...] = ()
    approval: WorkflowApproval = WorkflowApproval.DRAFT
    built_in: bool = False

    @model_validator(mode="after")
    def validate_template(self) -> WorkflowTemplate:
        node_ids = set(self.raw_workflow)
        if any(binding.node_id not in node_ids for binding in self.bindings):
            raise ValueError("workflow binding references an unknown node")
        if any(output.node_id not in node_ids for output in self.outputs):
            raise ValueError("workflow output references an unknown node")
        binding_ids = [binding.binding_id for binding in self.bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("workflow binding IDs must be unique")
        if self.approval == WorkflowApproval.APPROVED and self.unknown_node_types:
            raise ValueError("unknown node types must be confirmed before approval")
        encoded = json.dumps(
            self.raw_workflow,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.workflow_sha256:
            raise ValueError("workflow SHA-256 does not match raw_workflow")
        actual_node_types: set[str] = set()
        for node_id, node in self.raw_workflow.items():
            if not isinstance(node.get("class_type"), str) or not isinstance(
                node.get("inputs"), dict
            ):
                raise ValueError(f"workflow node {node_id} is not API format")
            actual_node_types.add(node["class_type"])
        if tuple(sorted(actual_node_types)) != tuple(sorted(self.required_node_types)):
            raise ValueError("required_node_types do not match raw_workflow")
        for binding in self.bindings:
            if binding.input_name not in self.raw_workflow[binding.node_id]["inputs"]:
                raise ValueError("workflow binding references an unknown node input")
        return self


class WorkflowInvocation(FrozenModel):
    template_id: str = Field(min_length=1)
    template_revision: int = Field(ge=1)
    values: dict[str, Any] = Field(default_factory=dict)
