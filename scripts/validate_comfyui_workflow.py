from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def fetch_object_info(url: str) -> dict[str, Any]:
    endpoint = f"{url.rstrip('/')}/object_info"
    with urllib.request.urlopen(endpoint, timeout=30) as response:  # noqa: S310
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("ComfyUI /object_info did not return an object")
    return value


def schema_input(schema: dict[str, Any], name: str) -> Any | None:
    inputs = schema.get("input", {})
    for section in ("required", "optional"):
        fields = inputs.get(section, {})
        if name in fields:
            return fields[name]
    return None


def validate(
    workflow: dict[str, Any], manifest: dict[str, Any], object_info: dict[str, Any]
) -> list[str]:
    issues: list[str] = []
    expected_hash = manifest.get("workflow_sha256")
    actual_hash = canonical_sha256(workflow)
    if expected_hash != actual_hash:
        issues.append(f"workflow SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")

    required_types = set(manifest.get("requirements", {}).get("node_types", []))
    actual_types: set[str] = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            issues.append(f"node {node_id} is not ComfyUI API format")
            continue
        node_type = node.get("class_type")
        if not isinstance(node_type, str):
            issues.append(f"node {node_id} has no class_type")
            continue
        actual_types.add(node_type)
        schema = object_info.get(node_type)
        if not isinstance(schema, dict):
            issues.append(f"node {node_id}: Worker is missing {node_type}")
            continue
        required = schema.get("input", {}).get("required", {})
        for input_name in required:
            if input_name not in node["inputs"]:
                issues.append(f"node {node_id}: required input {input_name} is missing")
        for input_name, input_value in node["inputs"].items():
            definition = schema_input(schema, input_name)
            if definition is None:
                issues.append(f"node {node_id}: unknown input {input_name}")
                continue
            if not (
                isinstance(input_value, list)
                and len(input_value) == 2
                and isinstance(input_value[1], int)
            ):
                continue
            source_id, output_index = str(input_value[0]), input_value[1]
            source = workflow.get(source_id)
            if not isinstance(source, dict):
                issues.append(f"node {node_id}.{input_name}: source {source_id} is missing")
                continue
            source_schema = object_info.get(source.get("class_type"), {})
            outputs = source_schema.get("output", [])
            if output_index < 0 or output_index >= len(outputs):
                issues.append(
                    f"node {node_id}.{input_name}: output {source_id}[{output_index}] is missing"
                )
                continue
            expected = definition[0] if isinstance(definition, list) else definition
            if isinstance(expected, str) and outputs[output_index] != expected:
                issues.append(
                    f"node {node_id}.{input_name}: expected {expected}, "
                    f"got {outputs[output_index]}"
                )

    if actual_types != required_types:
        issues.append(
            f"node type set mismatch: expected {sorted(required_types)}, got {sorted(actual_types)}"
        )

    for binding in manifest.get("bindings", []):
        node_id = str(binding.get("node_id"))
        input_name = binding.get("input_name")
        if node_id not in workflow or input_name not in workflow[node_id].get("inputs", {}):
            issues.append(f"binding {binding.get('binding_id')} target is missing")

    output_id = str(manifest.get("output", {}).get("node_id"))
    output = workflow.get(output_id)
    output_schema = object_info.get(output.get("class_type"), {}) if output else {}
    if not output_schema.get("output_node"):
        issues.append(f"declared output node {output_id} is not an output node")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an AVG ComfyUI API workflow")
    parser.add_argument("workflow", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    issues = validate(
        load_json(args.workflow), load_json(args.manifest), fetch_object_info(args.comfyui_url)
    )
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print("OK: workflow matches the live ComfyUI node schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
