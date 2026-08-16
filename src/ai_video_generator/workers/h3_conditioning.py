from __future__ import annotations

import copy
from typing import Any

from ai_video_generator.domain.chain import SHA256_PATTERN

from .workflow import WorkflowContractError, parse_api_workflow

SAVE_NODE_TYPE = "AVGSaveConditioningArtifact"
LOAD_NODE_TYPE = "AVGLoadConditioningArtifact"
SAVE_STATIC_NODE_TYPE = "AVGSaveH3StaticConditioning"
LOAD_STATIC_NODE_TYPE = "AVGLoadH3StaticConditioning"


def compile_conditioning_encode_workflow(
    workflow: dict[str, Any],
    *,
    conditioning_node_id: str,
    conditioning_output_index: int,
    fingerprint: str,
) -> dict[str, dict[str, Any]]:
    compiled = parse_api_workflow(workflow)
    _validate_fingerprint(fingerprint)
    if conditioning_node_id not in compiled:
        raise WorkflowContractError("conditioning source node does not exist")
    if conditioning_output_index < 0:
        raise WorkflowContractError("conditioning output index must not be negative")
    node_id = _new_node_id(compiled, "avg-save-conditioning")
    compiled[node_id] = {
        "class_type": SAVE_NODE_TYPE,
        "inputs": {
            "conditioning": [conditioning_node_id, conditioning_output_index],
            "fingerprint": fingerprint,
        },
        "_meta": {"title": "AVG conditioning cache output"},
    }
    return compiled


def compile_conditioning_load_workflow(
    workflow: dict[str, Any],
    *,
    target_node_id: str,
    target_input_name: str,
    fingerprint: str,
) -> dict[str, dict[str, Any]]:
    compiled = parse_api_workflow(copy.deepcopy(workflow))
    _validate_fingerprint(fingerprint)
    target = compiled.get(target_node_id)
    if target is None:
        raise WorkflowContractError("conditioning target node does not exist")
    if target_input_name not in target["inputs"]:
        raise WorkflowContractError("conditioning target input does not exist")
    node_id = _new_node_id(compiled, "avg-load-conditioning")
    compiled[node_id] = {
        "class_type": LOAD_NODE_TYPE,
        "inputs": {"fingerprint": fingerprint},
        "_meta": {"title": "AVG cached conditioning input"},
    }
    target["inputs"][target_input_name] = [node_id, 0]
    return compiled


def compile_h3_static_encode_workflow(
    workflow: dict[str, Any],
    *,
    conditioning_node_id: str,
    conditioning_output_index: int = 0,
    latent_output_index: int = 1,
    fingerprint: str,
) -> dict[str, dict[str, Any]]:
    """Compile an encode-only graph that persists all static H3 sampler inputs."""
    compiled = parse_api_workflow(workflow)
    _validate_fingerprint(fingerprint)
    if conditioning_node_id not in compiled:
        raise WorkflowContractError("H3 conditioning source node does not exist")
    if conditioning_output_index < 0 or latent_output_index < 0:
        raise WorkflowContractError("H3 static output indexes must not be negative")
    if conditioning_output_index == latent_output_index:
        raise WorkflowContractError("conditioning and latent output indexes must be different")

    node_id = _new_node_id(compiled, "avg-save-h3-static")
    compiled[node_id] = {
        "class_type": SAVE_STATIC_NODE_TYPE,
        "inputs": {
            "conditioning": [conditioning_node_id, conditioning_output_index],
            "latent": [conditioning_node_id, latent_output_index],
            "fingerprint": fingerprint,
        },
        "_meta": {"title": "AVG H3 static cache output"},
    }
    return _dependency_subgraph(compiled, (node_id,))


def compile_h3_static_load_workflow(
    workflow: dict[str, Any],
    *,
    conditioning_target_node_id: str,
    conditioning_target_input_name: str,
    latent_target_node_id: str,
    latent_target_input_name: str,
    output_node_ids: tuple[str, ...],
    fingerprint: str,
) -> dict[str, dict[str, Any]]:
    """Compile a diffusion graph whose static H3 source is only the cache bundle."""
    compiled = parse_api_workflow(workflow)
    _validate_fingerprint(fingerprint)
    _validate_target_input(
        compiled, conditioning_target_node_id, conditioning_target_input_name, "conditioning"
    )
    _validate_target_input(compiled, latent_target_node_id, latent_target_input_name, "latent")
    if not output_node_ids:
        raise WorkflowContractError("at least one diffusion output node is required")
    missing_outputs = tuple(node_id for node_id in output_node_ids if node_id not in compiled)
    if missing_outputs:
        raise WorkflowContractError(
            "diffusion output node does not exist: " + ", ".join(missing_outputs)
        )

    node_id = _new_node_id(compiled, "avg-load-h3-static")
    compiled[node_id] = {
        "class_type": LOAD_STATIC_NODE_TYPE,
        "inputs": {"fingerprint": fingerprint},
        "_meta": {"title": "AVG cached H3 static input"},
    }
    compiled[conditioning_target_node_id]["inputs"][conditioning_target_input_name] = [node_id, 0]
    compiled[latent_target_node_id]["inputs"][latent_target_input_name] = [node_id, 1]
    return _dependency_subgraph(compiled, output_node_ids)


def _validate_target_input(
    workflow: dict[str, dict[str, Any]], node_id: str, input_name: str, label: str
) -> None:
    target = workflow.get(node_id)
    if target is None:
        raise WorkflowContractError(f"{label} target node does not exist")
    if input_name not in target["inputs"]:
        raise WorkflowContractError(f"{label} target input does not exist")


def _dependency_subgraph(
    workflow: dict[str, dict[str, Any]], roots: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    required: set[str] = set()
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        if node_id in required:
            continue
        required.add(node_id)
        for value in workflow[node_id]["inputs"].values():
            if _is_node_link(value):
                pending.append(value[0])
    return {node_id: node for node_id, node in workflow.items() if node_id in required}


def _is_node_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _new_node_id(workflow: dict[str, Any], prefix: str) -> str:
    candidate = prefix
    suffix = 1
    while candidate in workflow:
        suffix += 1
        candidate = f"{prefix}-{suffix}"
    return candidate


def _validate_fingerprint(fingerprint: str) -> None:
    import re

    if re.fullmatch(SHA256_PATTERN, fingerprint) is None:
        raise WorkflowContractError("conditioning fingerprint must be lowercase SHA-256")
