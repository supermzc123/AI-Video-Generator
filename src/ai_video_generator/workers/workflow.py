from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict

from ai_video_generator.domain import (
    BindingValueType,
    WorkflowApproval,
    WorkflowBinding,
    WorkflowBindingDraft,
    WorkflowInvocation,
    WorkflowOutput,
    WorkflowOutputType,
    WorkflowTemplate,
)


class WorkflowContractError(ValueError):
    pass


class WorkflowInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_sha256: str
    node_schema_sha256: str
    raw_workflow: dict[str, dict[str, Any]]
    bindings: tuple[WorkflowBindingDraft, ...]
    outputs: tuple[WorkflowOutput, ...]
    required_node_types: tuple[str, ...]
    unknown_node_types: tuple[str, ...]
    issues: tuple[str, ...] = ()

    @property
    def compatible(self) -> bool:
        return not self.unknown_node_types and not self.issues and bool(self.outputs)


class CompiledWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str
    template_revision: int
    workflow: dict[str, dict[str, Any]]
    workflow_sha256: str


class ComfyUIImageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_id: str
    node_id: str
    filename: str
    subfolder: str = ""
    storage_type: str = "output"


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_api_workflow(
    source: str | bytes | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Parse ComfyUI API-format JSON and reject UI-format workflows."""
    if isinstance(source, bytes):
        try:
            source = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkflowContractError("workflow JSON must be UTF-8") from exc
    if isinstance(source, str):
        try:
            value = json.loads(source)
        except json.JSONDecodeError as exc:
            raise WorkflowContractError(f"invalid workflow JSON: {exc.msg}") from exc
    elif isinstance(source, Mapping):
        value = dict(source)
    else:
        raise WorkflowContractError("workflow must be JSON text, bytes, or an object")

    if not isinstance(value, dict) or not value:
        raise WorkflowContractError("API workflow root must be a non-empty object")
    if isinstance(value.get("nodes"), list):
        raise WorkflowContractError(
            "ComfyUI UI workflow JSON is not executable; export API format instead"
        )

    workflow: dict[str, dict[str, Any]] = {}
    for raw_node_id, raw_node in value.items():
        node_id = str(raw_node_id)
        if not node_id or not isinstance(raw_node, dict):
            raise WorkflowContractError(f"workflow node {node_id!r} must be an object")
        class_type = raw_node.get("class_type")
        inputs = raw_node.get("inputs")
        if not isinstance(class_type, str) or not class_type.strip():
            raise WorkflowContractError(f"workflow node {node_id!r} is missing class_type")
        if not isinstance(inputs, dict):
            raise WorkflowContractError(f"workflow node {node_id!r} is missing inputs")
        metadata = raw_node.get("_meta", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise WorkflowContractError(f"workflow node {node_id!r} _meta must be an object")
        title = (metadata or {}).get("title")
        if title is not None and not isinstance(title, str):
            raise WorkflowContractError(f"workflow node {node_id!r} title must be a string")
        workflow[node_id] = copy.deepcopy(raw_node)
    for node_id, node in workflow.items():
        for input_name, input_value in node["inputs"].items():
            link = _as_link(input_value)
            if link is not None and link[0] not in workflow:
                raise WorkflowContractError(
                    f"workflow node {node_id!r} input {input_name!r} references "
                    f"unknown node {link[0]!r}"
                )
    return workflow


def inspect_api_workflow(
    source: str | bytes | Mapping[str, Any],
    object_info: Mapping[str, Any],
) -> WorkflowInspection:
    workflow = parse_api_workflow(source)
    schemas = _normalize_schemas(object_info)
    required = tuple(sorted({str(node["class_type"]) for node in workflow.values()}))
    unknown = tuple(node_type for node_type in required if node_type not in schemas)
    outputs: list[WorkflowOutput] = []
    issues: list[str] = []

    issues.extend(_workflow_schema_issues(workflow, schemas))

    for node_id, node in workflow.items():
        schema = schemas.get(str(node["class_type"]))
        if schema and schema.get("output_node") is True:
            title = str((node.get("_meta") or {}).get("title") or "").strip()
            outputs.append(
                WorkflowOutput(
                    output_id=f"image:{node_id}",
                    node_id=node_id,
                    output_type=WorkflowOutputType.IMAGE,
                    title=title or f"{node['class_type']} #{node_id}",
                )
            )
    return WorkflowInspection(
        workflow_sha256=canonical_json_sha256(workflow),
        node_schema_sha256=canonical_json_sha256(schemas),
        raw_workflow=workflow,
        bindings=(),
        outputs=tuple(outputs),
        required_node_types=required,
        unknown_node_types=unknown,
        issues=tuple(issues),
    )


def validate_workflow_template(
    template: WorkflowTemplate,
    object_info: Mapping[str, Any],
) -> tuple[str, ...]:
    """Revalidate an approved template against the Worker's current node schema."""
    inspection = inspect_api_workflow(template.raw_workflow, object_info)
    issues = list(inspection.issues)
    if inspection.required_node_types != tuple(sorted(template.required_node_types)):
        issues.append("workflow required node types changed")
    if inspection.unknown_node_types:
        issues.append("missing node types: " + ", ".join(inspection.unknown_node_types))
    schemas = _normalize_schemas(object_info)
    for output in template.outputs:
        node = template.raw_workflow[output.node_id]
        schema = schemas.get(str(node["class_type"]))
        if not schema or schema.get("output_node") is not True:
            issues.append(
                f"output {output.output_id}: node {output.node_id} is not a ComfyUI output node"
            )
    for binding in template.bindings:
        node = template.raw_workflow[binding.node_id]
        if _as_link(node["inputs"].get(binding.input_name)) is not None:
            issues.append(
                f"binding {binding.binding_id}: target is a connected input "
                "and would modify topology"
            )
            continue
        schema = schemas.get(str(node["class_type"]))
        issue, minimum, maximum = _validate_schema_input(
            schema,
            binding.input_name,
            binding.value_type,
        )
        if issue:
            issues.append(f"binding {binding.binding_id}: {issue}")
        if binding.minimum is not None and minimum is not None and binding.minimum < minimum:
            issues.append(f"binding {binding.binding_id}: minimum is outside Worker schema")
        if binding.maximum is not None and maximum is not None and binding.maximum > maximum:
            issues.append(f"binding {binding.binding_id}: maximum is outside Worker schema")
    return tuple(dict.fromkeys(issues))


def workflow_template_from_inspection(
    inspection: WorkflowInspection,
    *,
    template_id: str,
    name: str,
    revision: int = 1,
    approval: WorkflowApproval = WorkflowApproval.DRAFT,
    built_in: bool = False,
) -> WorkflowTemplate:
    return WorkflowTemplate(
        template_id=template_id,
        revision=revision,
        name=name,
        workflow_sha256=inspection.workflow_sha256,
        node_schema_sha256=inspection.node_schema_sha256,
        raw_workflow=inspection.raw_workflow,
        bindings=tuple(
            WorkflowBinding(**binding.model_dump(exclude={"confidence", "rationale"}))
            for binding in inspection.bindings
        ),
        outputs=inspection.outputs,
        required_node_types=inspection.required_node_types,
        unknown_node_types=inspection.unknown_node_types,
        approval=approval,
        built_in=built_in,
    )


def compile_workflow(
    template: WorkflowTemplate,
    invocation: WorkflowInvocation | Mapping[str, Any],
    *,
    require_approved: bool = True,
) -> CompiledWorkflow:
    if require_approved and template.approval != WorkflowApproval.APPROVED:
        raise WorkflowContractError("workflow template must be approved before compilation")
    if isinstance(invocation, WorkflowInvocation):
        if invocation.template_id != template.template_id:
            raise WorkflowContractError("workflow invocation template_id does not match")
        if invocation.template_revision != template.revision:
            raise WorkflowContractError("workflow invocation revision does not match")
        values = dict(invocation.values)
    elif isinstance(invocation, Mapping):
        values = dict(invocation)
    else:
        raise WorkflowContractError("workflow invocation must be an object")

    bindings = {binding.binding_id: binding for binding in template.bindings}
    unknown_values = sorted(set(values) - set(bindings))
    if unknown_values:
        raise WorkflowContractError("unknown workflow binding(s): " + ", ".join(unknown_values))

    compiled = copy.deepcopy(template.raw_workflow)
    for binding in template.bindings:
        supplied = binding.binding_id in values
        value = values.get(binding.binding_id, binding.default_value)
        if value is None and not supplied:
            continue
        value = _coerce_binding_value(binding, value)
        if binding.string_template is not None:
            value = binding.string_template.replace("{value}", str(value))
        node = compiled.get(binding.node_id)
        if node is None or binding.input_name not in node.get("inputs", {}):
            raise WorkflowContractError(f"binding {binding.binding_id!r} target no longer exists")
        node["inputs"][binding.input_name] = value

    return CompiledWorkflow(
        template_id=template.template_id,
        template_revision=template.revision,
        workflow=compiled,
        workflow_sha256=canonical_json_sha256(compiled),
    )


def extract_workflow_outputs(
    history: Mapping[str, Any],
    template: WorkflowTemplate,
) -> tuple[ComfyUIImageOutput, ...]:
    record: Mapping[str, Any] = history
    if "outputs" not in record and len(record) == 1:
        candidate = next(iter(record.values()))
        if isinstance(candidate, Mapping):
            record = candidate
    raw_outputs = record.get("outputs", {})
    if not isinstance(raw_outputs, Mapping):
        raise WorkflowContractError("ComfyUI history outputs must be an object")

    extracted: list[ComfyUIImageOutput] = []
    for declared in template.outputs:
        node_output = raw_outputs.get(declared.node_id, {})
        if not isinstance(node_output, Mapping):
            continue
        field = "images" if declared.output_type == WorkflowOutputType.IMAGE else "gifs"
        media = node_output.get(field, node_output.get("images", []))
        if not isinstance(media, list):
            raise WorkflowContractError(
                f"ComfyUI history node {declared.node_id} output must be a list"
            )
        for image in media:
            if not isinstance(image, Mapping) or not str(image.get("filename") or ""):
                raise WorkflowContractError(
                    f"ComfyUI history node {declared.node_id} contains an invalid image"
                )
            extracted.append(
                ComfyUIImageOutput(
                    output_id=declared.output_id,
                    node_id=declared.node_id,
                    filename=str(image["filename"]),
                    subfolder=str(image.get("subfolder") or ""),
                    storage_type=str(image.get("type") or "output"),
                )
            )
    return tuple(extracted)


def _normalize_schemas(object_info: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for node_type, schema in object_info.items():
        if isinstance(node_type, str) and isinstance(schema, dict):
            schemas[node_type] = schema
    return dict(sorted(schemas.items()))


def _schema_input(schema: Mapping[str, Any], input_name: str) -> Any:
    input_block = schema.get("input", {})
    if not isinstance(input_block, Mapping):
        return None
    for section in ("required", "optional"):
        fields = input_block.get(section, {})
        if isinstance(fields, Mapping) and input_name in fields:
            return fields[input_name]
    if "." not in input_name:
        return None

    dynamic_name, child_name = input_name.split(".", 1)
    for section in ("required", "optional"):
        fields = input_block.get(section, {})
        if not isinstance(fields, Mapping):
            continue
        definition = fields.get(dynamic_name)
        resolved = _dynamic_schema_input(definition, child_name)
        if resolved is not None:
            return resolved
    return None


def _dynamic_schema_input(definition: Any, child_name: str) -> Any:
    if not isinstance(definition, (list, tuple)) or len(definition) < 2:
        return None
    kind, options = definition[0], definition[1]
    if not isinstance(options, Mapping):
        return None
    if kind == "COMFY_DYNAMICCOMBO_V3":
        candidates = []
        for option in options.get("options", []):
            if not isinstance(option, Mapping):
                continue
            inputs = option.get("inputs", {})
            if not isinstance(inputs, Mapping):
                continue
            for section in ("required", "optional"):
                fields = inputs.get(section, {})
                if isinstance(fields, Mapping) and child_name in fields:
                    candidates.append(fields[child_name])
        if not candidates:
            return None
        first = candidates[0]
        return first if all(value == first for value in candidates) else None
    if kind != "COMFY_AUTOGROW_V3":
        return None
    template = options.get("template")
    if not isinstance(template, Mapping):
        return None
    prefix = template.get("prefix")
    names = template.get("names")
    if isinstance(prefix, str):
        suffix = child_name.removeprefix(prefix)
        if not suffix.isdigit() or child_name != f"{prefix}{suffix}":
            return None
        index = int(suffix)
        maximum = template.get("max")
        if isinstance(maximum, int) and index >= maximum:
            return None
    elif isinstance(names, list):
        if child_name not in names:
            return None
    else:
        return None
    template_inputs = template.get("input")
    if not isinstance(template_inputs, Mapping):
        return None
    for section in ("required", "optional"):
        fields = template_inputs.get(section, {})
        if isinstance(fields, Mapping) and fields:
            return next(iter(fields.values()))
    return None


def _as_link(value: Any) -> tuple[str, int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    ):
        return str(value[0]), value[1]
    return None


def _workflow_schema_issues(
    workflow: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    issues: list[str] = []
    for node_id, node in workflow.items():
        node_type = str(node["class_type"])
        schema = schemas.get(node_type)
        if schema is None:
            continue
        inputs = node.get("inputs", {})
        input_block = schema.get("input", {})
        required_inputs = (
            input_block.get("required", {}) if isinstance(input_block, Mapping) else {}
        )
        if isinstance(required_inputs, Mapping):
            for required_name, definition in required_inputs.items():
                if not _required_input_present(required_name, definition, inputs):
                    issues.append(
                        f"node {node_id} ({node_type}): required input {required_name!r} is missing"
                    )
        for input_name, input_value in inputs.items():
            definition = _schema_input(schema, input_name)
            if definition is None:
                issues.append(
                    f"node {node_id} ({node_type}): input {input_name!r} "
                    "is not exposed by object_info"
                )
                continue
            link = _as_link(input_value)
            if link is None:
                continue
            source_node = workflow.get(link[0])
            if source_node is None:
                continue
            source_schema = schemas.get(str(source_node["class_type"]))
            outputs = source_schema.get("output", []) if source_schema else []
            if not isinstance(outputs, (list, tuple)) or link[1] >= len(outputs):
                issues.append(
                    f"node {node_id}.{input_name}: source output "
                    f"{link[0]}[{link[1]}] does not exist"
                )
                continue
            expected = (
                definition[0]
                if isinstance(definition, (list, tuple)) and definition
                else definition
            )
            actual = outputs[link[1]]
            if (
                isinstance(expected, str)
                and isinstance(actual, str)
                and not _comfy_types_compatible(expected, actual)
            ):
                issues.append(
                    f"node {node_id}.{input_name}: expected {expected}, source provides {actual}"
                )
    return issues


def _comfy_types_compatible(expected: str, actual: str) -> bool:
    wildcard = "COMFY_MATCHTYPE_V3"
    if expected == wildcard or actual == wildcard:
        return True
    expected_types = {value.strip() for value in expected.split(",")}
    actual_types = {value.strip() for value in actual.split(",")}
    return bool(expected_types & actual_types)


def _required_input_present(
    input_name: str,
    definition: Any,
    inputs: Mapping[str, Any],
) -> bool:
    if input_name in inputs:
        return True
    if not (
        isinstance(definition, (list, tuple))
        and len(definition) > 1
        and definition[0] == "COMFY_AUTOGROW_V3"
        and isinstance(definition[1], Mapping)
    ):
        return False
    template = definition[1].get("template")
    if not isinstance(template, Mapping):
        return False
    minimum = template.get("min", 0)
    if not isinstance(minimum, int) or minimum <= 0:
        return True
    prefix = template.get("prefix")
    names = template.get("names")
    if isinstance(prefix, str):
        live = sum(
            _dynamic_schema_input(definition, key.split(".", 1)[1]) is not None
            for key in inputs
            if key.startswith(f"{input_name}.")
        )
    elif isinstance(names, list):
        live = sum(f"{input_name}.{name}" in inputs for name in names)
    else:
        return False
    return live >= minimum


def _validate_schema_input(
    schema: Mapping[str, Any] | None,
    input_name: str,
    value_type: BindingValueType,
) -> tuple[str | None, float | None, float | None]:
    if schema is None:
        return None, None, None
    definition = _schema_input(schema, input_name)
    if definition is None:
        return "input is not exposed by object_info", None, None
    type_spec = (
        definition[0] if isinstance(definition, (list, tuple)) and definition else definition
    )
    options = (
        definition[1]
        if isinstance(definition, (list, tuple))
        and len(definition) > 1
        and isinstance(definition[1], Mapping)
        else {}
    )
    actual = type_spec.upper() if isinstance(type_spec, str) else "COMBO"
    allowed = {
        BindingValueType.STRING: {"STRING", "COMBO"},
        BindingValueType.INTEGER: {"INT"},
        BindingValueType.NUMBER: {"FLOAT", "INT"},
        BindingValueType.IMAGE_PATH: {"STRING", "COMBO"},
        BindingValueType.VIDEO_PATH: {"STRING", "COMBO"},
    }[value_type]
    issue = (
        None if actual in allowed else f"expected {value_type.value}, object_info reports {actual}"
    )
    minimum = options.get("min")
    maximum = options.get("max")
    return (
        issue,
        float(minimum) if isinstance(minimum, (int, float)) else None,
        float(maximum) if isinstance(maximum, (int, float)) else None,
    )


def _coerce_binding_value(binding: WorkflowBinding, value: Any) -> Any:
    if binding.value_type in {
        BindingValueType.STRING,
        BindingValueType.IMAGE_PATH,
        BindingValueType.VIDEO_PATH,
    }:
        if not isinstance(value, str):
            raise WorkflowContractError(f"binding {binding.binding_id!r} requires a string")
        if binding.value_type in {BindingValueType.IMAGE_PATH, BindingValueType.VIDEO_PATH}:
            _validate_relative_media_path(value)
        coerced: Any = value
    elif binding.value_type == BindingValueType.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            raise WorkflowContractError(f"binding {binding.binding_id!r} requires an integer")
        coerced = value
    elif binding.value_type == BindingValueType.NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkflowContractError(f"binding {binding.binding_id!r} requires a number")
        coerced = float(value)
    else:  # pragma: no cover - enum is closed
        raise WorkflowContractError(f"unsupported binding type {binding.value_type}")

    if isinstance(coerced, (int, float)):
        if binding.minimum is not None and coerced < binding.minimum:
            raise WorkflowContractError(
                f"binding {binding.binding_id!r} is below minimum {binding.minimum}"
            )
        if binding.maximum is not None and coerced > binding.maximum:
            raise WorkflowContractError(
                f"binding {binding.binding_id!r} exceeds maximum {binding.maximum}"
            )
    return coerced


def _validate_relative_media_path(value: str) -> None:
    if not value.strip():
        raise WorkflowContractError("image path must not be empty")
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        raise WorkflowContractError("image path must stay relative to the ComfyUI input root")


def load_api_workflow_file(path: Path) -> dict[str, dict[str, Any]]:
    return parse_api_workflow(path.read_bytes())
