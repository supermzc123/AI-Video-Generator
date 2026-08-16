from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field

from ai_video_generator.domain import (
    H3AccelerationMode,
    H3AttentionMode,
    H3InputTarget,
    H3TurboProfile,
    H3WorkflowProfile,
    WorkflowApproval,
)

from .h3_policy import (
    OFFICIAL_H3_TURBO_MODULES,
    OFFICIAL_H3_TURBO_REPOSITORY,
    PINNED_H3_TURBO_COMMIT,
    SAGE_NODE_TYPE,
    SAGE_PYTHON_MODULE,
    TURBO_LORA_NODE_TYPE,
    TURBO_SAMPLER_NODE_TYPE,
    TURBO_SCHEDULER_NODE_TYPE,
)
from .workflow import WorkflowContractError, canonical_json_sha256, parse_api_workflow


class H3WorkflowInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: H3WorkflowProfile
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @computed_field
    @property
    def compatible(self) -> bool:
        return not self.issues


def inspect_h3_workflow_profile(
    profile: H3WorkflowProfile,
    object_info: dict[str, Any],
) -> H3WorkflowInspection:
    workflow = parse_api_workflow(profile.raw_workflow)
    issues: list[str] = []
    warnings: list[str] = []
    actual_workflow_sha = canonical_json_sha256(workflow)
    actual_schema_sha = canonical_json_sha256(object_info)
    if actual_workflow_sha != profile.workflow_sha256:
        issues.append("H3 workflow SHA-256 does not match its normalized API JSON")
    if actual_schema_sha != profile.node_schema_sha256:
        issues.append("H3 node schema SHA-256 does not match object_info")

    node_types = {node_id: str(node["class_type"]) for node_id, node in workflow.items()}
    for node_id, node_type in node_types.items():
        if node_type not in object_info:
            issues.append(f"H3 workflow node {node_id} uses unavailable type {node_type}")
        if "teacache" in node_type.casefold():
            issues.append(f"TeaCache node {node_id} is not supported for MiniMax H3")

    if profile.attention == H3AttentionMode.SAGE:
        node_id = profile.sage_attention_node_id or ""
        if node_types.get(node_id) != SAGE_NODE_TYPE:
            issues.append(
                "SageAttention binding must reference "
                f"{SAGE_NODE_TYPE}, got {node_types.get(node_id, 'missing node')}"
            )
        else:
            sage_schema = object_info.get(SAGE_NODE_TYPE, {})
            if sage_schema.get("python_module") != SAGE_PYTHON_MODULE:
                issues.append("SageAttention node must come from ComfyUI-KJNodes")
            warnings.append("The KJNodes MiniMax H3 SageAttention patch is experimental")

    if profile.acceleration == H3AccelerationMode.TURBO and profile.turbo:
        _validate_turbo(
            profile.turbo,
            workflow,
            node_types,
            object_info,
            issues,
            warnings,
        )

    return H3WorkflowInspection(
        profile=profile,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def compile_h3_workflow(profile: H3WorkflowProfile) -> dict[str, dict[str, Any]]:
    workflow = parse_api_workflow(profile.raw_workflow)
    issues, _warnings = _inspect_h3_structure(profile, workflow)
    if issues:
        raise WorkflowContractError("; ".join(issues))
    if profile.acceleration != H3AccelerationMode.TURBO or profile.turbo is None:
        return workflow

    turbo = profile.turbo
    compiled = copy.deepcopy(workflow)
    _set_target(compiled, turbo.lora_name_target, turbo.lora_name)
    _set_target(compiled, turbo.strength_target, turbo.strength)
    _set_target(compiled, turbo.steps_target, turbo.steps)
    if turbo.low_vram_target:
        _set_target(compiled, turbo.low_vram_target, turbo.low_vram)
    if turbo.scheduler_name_target:
        _set_target(compiled, turbo.scheduler_name_target, "simple")
    return compiled


def build_h3_profile(
    *,
    profile_id: str,
    name: str,
    raw_workflow: dict[str, Any],
    object_info: dict[str, Any],
    acceleration: H3AccelerationMode = H3AccelerationMode.STANDARD,
    attention: H3AttentionMode = H3AttentionMode.NATIVE,
    turbo: H3TurboProfile | None = None,
    sage_attention_node_id: str | None = None,
    approval: WorkflowApproval = WorkflowApproval.DRAFT,
    revision: int = 1,
) -> H3WorkflowProfile:
    workflow = parse_api_workflow(raw_workflow)
    return H3WorkflowProfile(
        profile_id=profile_id,
        revision=revision,
        name=name,
        approval=approval,
        workflow_sha256=canonical_json_sha256(workflow),
        node_schema_sha256=canonical_json_sha256(object_info),
        raw_workflow=workflow,
        acceleration=acceleration,
        attention=attention,
        turbo=turbo,
        sage_attention_node_id=sage_attention_node_id,
    )


def _validate_turbo(
    turbo: H3TurboProfile,
    workflow: dict[str, dict[str, Any]],
    node_types: dict[str, str],
    object_info: dict[str, Any],
    issues: list[str],
    warnings: list[str],
) -> None:
    structural_issues, structural_warnings = _inspect_turbo_structure(
        turbo,
        workflow,
        node_types,
    )
    issues.extend(structural_issues)
    warnings.extend(structural_warnings)

    loader_type = node_types.get(turbo.lora_loader_node_id)
    sampler_type = node_types.get(turbo.sampler_node_id)
    if loader_type == TURBO_LORA_NODE_TYPE:
        _validate_official_turbo_schema(
            object_info.get(TURBO_LORA_NODE_TYPE),
            TURBO_LORA_NODE_TYPE,
            {"model", "lora_name", "strength", "low_vram"},
            issues,
        )
    if sampler_type == TURBO_SAMPLER_NODE_TYPE:
        _validate_official_turbo_schema(
            object_info.get(TURBO_SAMPLER_NODE_TYPE),
            TURBO_SAMPLER_NODE_TYPE,
            set(),
            issues,
        )

    if loader_type == TURBO_LORA_NODE_TYPE and turbo.low_vram_target is None:
        warnings.append("Turbo LoRA low_vram is not exposed; workflow value will be preserved")


def _inspect_h3_structure(
    profile: H3WorkflowProfile,
    workflow: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    node_types = {node_id: str(node["class_type"]) for node_id, node in workflow.items()}
    for node_id, node_type in node_types.items():
        if "teacache" in node_type.casefold():
            issues.append(f"TeaCache node {node_id} is not supported for MiniMax H3")

    if profile.attention == H3AttentionMode.SAGE:
        node_id = profile.sage_attention_node_id or ""
        if node_types.get(node_id) != SAGE_NODE_TYPE:
            issues.append(f"SageAttention binding must reference {SAGE_NODE_TYPE}")

    if profile.acceleration == H3AccelerationMode.TURBO and profile.turbo:
        turbo_issues, turbo_warnings = _inspect_turbo_structure(
            profile.turbo,
            workflow,
            node_types,
        )
        issues.extend(turbo_issues)
        warnings.extend(turbo_warnings)
    return issues, warnings


def _inspect_turbo_structure(
    turbo: H3TurboProfile,
    workflow: dict[str, dict[str, Any]],
    node_types: dict[str, str],
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    if turbo.source_repository != OFFICIAL_H3_TURBO_REPOSITORY:
        issues.append("Turbo source repository does not match the pinned official plugin")
    if turbo.source_commit != PINNED_H3_TURBO_COMMIT:
        issues.append("Turbo source commit does not match the pinned official plugin")
    loader_type = node_types.get(turbo.lora_loader_node_id)
    if loader_type != TURBO_LORA_NODE_TYPE:
        issues.append(f"Turbo LoRA loader must be the official {TURBO_LORA_NODE_TYPE} node")
    sampler_type = node_types.get(turbo.sampler_node_id)
    if sampler_type != TURBO_SAMPLER_NODE_TYPE:
        issues.append(f"Turbo sampler must be the official {TURBO_SAMPLER_NODE_TYPE} node")

    for target in (
        turbo.lora_name_target,
        turbo.strength_target,
        turbo.steps_target,
        turbo.low_vram_target,
        turbo.scheduler_name_target,
    ):
        if target is None:
            continue
        node = workflow.get(target.node_id)
        if node is None or target.input_name not in node["inputs"]:
            issues.append(
                f"Turbo target {target.node_id}.{target.input_name} is not an editable input"
            )

    scheduler_id = turbo.scheduler_node_id
    if not scheduler_id:
        issues.append(f"{TURBO_SAMPLER_NODE_TYPE} requires a paired scheduler node")
    else:
        scheduler = workflow.get(scheduler_id)
        if scheduler is None:
            issues.append("Turbo scheduler node is missing")
        elif scheduler["class_type"] != TURBO_SCHEDULER_NODE_TYPE:
            issues.append(f"Turbo scheduler must be {TURBO_SCHEDULER_NODE_TYPE}")
        elif turbo.scheduler_name_target is None:
            issues.append("Turbo scheduler binding must expose scheduler_name_target")

    if scheduler_id and not _has_paired_advanced_sampler(
        workflow,
        turbo.sampler_node_id,
        scheduler_id,
    ):
        issues.append(
            "Official Turbo sampler and BasicScheduler must feed the same "
            "SamplerCustomAdvanced node"
        )
    if scheduler_id and not _node_depends_on(
        workflow,
        scheduler_id,
        turbo.lora_loader_node_id,
    ):
        issues.append("BasicScheduler model path must include the official Turbo LoRA loader")

    if turbo.steps == 4:
        warnings.append(
            "Four-step Turbo may smear large or fast motion; v4 is best at six to eight steps"
        )
    if turbo.strength != 1.0:
        warnings.append("Turbo LoRA is tuned for strength 1.0; overrides are clip-specific")
    return issues, warnings


def _validate_official_turbo_schema(
    schema: Any,
    node_type: str,
    required_inputs: set[str],
    issues: list[str],
) -> None:
    if not isinstance(schema, dict):
        issues.append(f"ComfyUI object_info is missing {node_type}")
        return
    python_module = schema.get("python_module")
    if python_module not in OFFICIAL_H3_TURBO_MODULES:
        issues.append(
            f"{node_type} must come from the pinned official plugin; got "
            f"{python_module or 'unknown provider'}"
        )
    required = schema.get("input", {}).get("required", {})
    if not isinstance(required, dict) or not required_inputs.issubset(required):
        issues.append(f"{node_type} object_info does not match the official input contract")


def _has_paired_advanced_sampler(
    workflow: dict[str, dict[str, Any]],
    sampler_node_id: str,
    scheduler_node_id: str,
) -> bool:
    return any(
        node["class_type"] == "SamplerCustomAdvanced"
        and _linked_node_id(node["inputs"].get("sampler")) == sampler_node_id
        and _linked_node_id(node["inputs"].get("sigmas")) == scheduler_node_id
        for node in workflow.values()
    )


def _node_depends_on(
    workflow: dict[str, dict[str, Any]],
    node_id: str,
    ancestor_node_id: str,
    visited: set[str] | None = None,
) -> bool:
    if node_id == ancestor_node_id:
        return True
    visited = set() if visited is None else visited
    if node_id in visited:
        return False
    visited.add(node_id)
    node = workflow.get(node_id)
    if node is None:
        return False
    return any(
        linked is not None and _node_depends_on(workflow, linked, ancestor_node_id, visited)
        for linked in (_linked_node_id(value) for value in node["inputs"].values())
    )


def _linked_node_id(value: Any) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        node_id, output_index = value
        if isinstance(node_id, (str, int)) and isinstance(output_index, int):
            return str(node_id)
    return None


def _set_target(
    workflow: dict[str, dict[str, Any]],
    target: H3InputTarget,
    value: Any,
) -> None:
    node = workflow.get(target.node_id)
    if node is None or target.input_name not in node["inputs"]:
        raise WorkflowContractError(
            f"H3 workflow target {target.node_id}.{target.input_name} does not exist"
        )
    node["inputs"][target.input_name] = value
