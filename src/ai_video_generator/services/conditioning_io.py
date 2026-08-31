from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .hashing import sha256_file

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONDITIONING_FORMAT = "safetensors+json-v1"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024


class ConditioningIOError(RuntimeError):
    pass


class ConditioningDependencyError(ConditioningIOError):
    pass


class ConditioningIntegrityError(ConditioningIOError):
    pass


class ConditioningWriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str
    tensor_path: str
    manifest_path: str
    blob_sha256: str
    byte_size: int = Field(ge=0)
    tensor_count: int = Field(ge=0)


def write_conditioning_artifact(
    payload: Any,
    *,
    fingerprint: str,
    tensor_path: Path,
    manifest_path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> ConditioningWriteResult:
    torch, save_file, _load_file = _load_dependencies()
    _validate_fingerprint(fingerprint)
    _validate_distinct_paths(tensor_path, manifest_path)
    tensors: dict[str, Any] = {}
    structure = _encode_structure(payload, tensors, torch)
    safe_metadata = _json_value(dict(metadata or {}), location="metadata")

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_tmp = _temporary_peer(tensor_path)
    manifest_tmp = _temporary_peer(manifest_path)
    try:
        save_file(tensors, str(tensor_tmp), metadata={"format": CONDITIONING_FORMAT})
        blob_sha256 = sha256_file(tensor_tmp)
        byte_size = tensor_tmp.stat().st_size
        manifest = {
            "format": CONDITIONING_FORMAT,
            "fingerprint": fingerprint,
            "tensor_file": tensor_path.name,
            "blob_sha256": blob_sha256,
            "byte_size": byte_size,
            "tensor_count": len(tensors),
            "structure": structure,
            "metadata": safe_metadata,
        }
        manifest_tmp.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(tensor_tmp, tensor_path)
        # The manifest is the commit marker and is published last.
        os.replace(manifest_tmp, manifest_path)
    finally:
        tensor_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)

    return ConditioningWriteResult(
        fingerprint=fingerprint,
        tensor_path=str(tensor_path),
        manifest_path=str(manifest_path),
        blob_sha256=blob_sha256,
        byte_size=byte_size,
        tensor_count=len(tensors),
    )


def read_conditioning_artifact(
    *,
    tensor_path: Path,
    manifest_path: Path,
    expected_fingerprint: str | None = None,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    _torch, _save_file, load_file = _load_dependencies()
    _validate_distinct_paths(tensor_path, manifest_path)
    if expected_fingerprint is not None:
        _validate_fingerprint(expected_fingerprint)
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise ConditioningIntegrityError("conditioning artifact is incomplete")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ConditioningIntegrityError("conditioning manifest exceeds the size limit")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ConditioningIntegrityError("conditioning manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != CONDITIONING_FORMAT:
        raise ConditioningIntegrityError("unsupported conditioning artifact format")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        raise ConditioningIntegrityError("conditioning manifest fingerprint is invalid")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ConditioningIntegrityError("conditioning fingerprint does not match")
    if manifest.get("tensor_file") != tensor_path.name:
        raise ConditioningIntegrityError("conditioning manifest references a different tensor file")

    actual_size = tensor_path.stat().st_size
    if manifest.get("byte_size") != actual_size:
        raise ConditioningIntegrityError("conditioning tensor byte size does not match")
    actual_sha256 = sha256_file(tensor_path)
    if manifest.get("blob_sha256") != actual_sha256:
        raise ConditioningIntegrityError("conditioning tensor SHA-256 does not match")

    try:
        tensors = load_file(str(tensor_path), device=device)
    except Exception as exc:
        raise ConditioningIntegrityError(
            "conditioning safetensors payload cannot be loaded"
        ) from exc
    if manifest.get("tensor_count") != len(tensors):
        raise ConditioningIntegrityError("conditioning tensor count does not match")
    referenced_tensors: set[str] = set()
    payload = _decode_structure(
        manifest.get("structure"),
        tensors,
        referenced_tensors=referenced_tensors,
        budget=[0],
    )
    if referenced_tensors != set(tensors):
        raise ConditioningIntegrityError(
            "conditioning structure does not reference exactly the stored tensors"
        )
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ConditioningIntegrityError("conditioning metadata must be an object")
    try:
        safe_metadata = _json_value(metadata, location="metadata")
    except ConditioningIOError as exc:
        raise ConditioningIntegrityError("conditioning metadata is not safe JSON") from exc
    return payload, safe_metadata


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise ConditioningDependencyError(
            "conditioning I/O requires both torch and safetensors; "
            "install them in the Worker environment"
        ) from exc
    return torch, save_file, load_file


def _encode_structure(value: Any, tensors: dict[str, Any], torch: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        name = f"tensor_{len(tensors):06d}"
        tensors[name] = value.detach().cpu().contiguous()
        return {"kind": "tensor", "name": name}
    if value is None or isinstance(value, (str, bool, int)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConditioningIOError("conditioning scalars must be finite")
        return {"kind": "scalar", "value": value}
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_encode_structure(item, tensors, torch) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_structure(item, tensors, torch) for item in value],
        }
    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConditioningIOError("conditioning dictionary keys must be strings")
            items[key] = _encode_structure(item, tensors, torch)
        return {"kind": "dict", "items": items}
    raise ConditioningIOError(
        f"conditioning contains unsupported object type {type(value).__name__}"
    )


def _decode_structure(
    node: Any,
    tensors: Mapping[str, Any],
    *,
    referenced_tensors: set[str],
    budget: list[int],
    depth: int = 0,
) -> Any:
    budget[0] += 1
    if depth > 128 or budget[0] > 100_000:
        raise ConditioningIntegrityError("conditioning structure exceeds safety limits")
    if not isinstance(node, dict):
        raise ConditioningIntegrityError("conditioning structure node must be an object")
    kind = node.get("kind")
    if kind == "tensor":
        name = node.get("name")
        if not isinstance(name, str) or name not in tensors:
            raise ConditioningIntegrityError("conditioning structure references a missing tensor")
        referenced_tensors.add(name)
        return tensors[name]
    if kind == "scalar":
        value = node.get("value")
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ConditioningIntegrityError("conditioning scalar has an invalid type")
        if isinstance(value, float) and not math.isfinite(value):
            raise ConditioningIntegrityError("conditioning scalar must be finite")
        return value
    if kind in {"list", "tuple"}:
        items = node.get("items")
        if not isinstance(items, list):
            raise ConditioningIntegrityError("conditioning sequence items must be a list")
        decoded = [
            _decode_structure(
                item,
                tensors,
                referenced_tensors=referenced_tensors,
                budget=budget,
                depth=depth + 1,
            )
            for item in items
        ]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        items = node.get("items")
        if not isinstance(items, dict) or any(not isinstance(key, str) for key in items):
            raise ConditioningIntegrityError("conditioning dictionary structure is invalid")
        return {
            key: _decode_structure(
                item,
                tensors,
                referenced_tensors=referenced_tensors,
                budget=budget,
                depth=depth + 1,
            )
            for key, item in items.items()
        }
    raise ConditioningIntegrityError(f"unknown conditioning structure kind {kind!r}")


def _json_value(value: Any, *, location: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ConditioningIOError(f"{location} must contain only JSON-safe values") from exc


def _validate_fingerprint(value: str) -> None:
    if not SHA256_RE.fullmatch(value):
        raise ConditioningIOError("conditioning fingerprint must be lowercase SHA-256")


def _validate_distinct_paths(tensor_path: Path, manifest_path: Path) -> None:
    if tensor_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise ConditioningIOError("tensor and manifest paths must be different")
    if tensor_path.suffix != ".safetensors":
        raise ConditioningIOError("conditioning tensor path must end in .safetensors")
    if manifest_path.suffix != ".json":
        raise ConditioningIOError("conditioning manifest path must end in .json")


def _temporary_peer(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
