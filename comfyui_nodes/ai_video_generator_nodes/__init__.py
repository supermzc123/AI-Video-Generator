from __future__ import annotations

import re
from pathlib import Path

import folder_paths
from comfy.nested_tensor import NestedTensor

try:
    from ai_video_generator.services import (
        read_conditioning_artifact,
        write_conditioning_artifact,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ai_video_generator":
        raise
    from .conditioning_io import read_conditioning_artifact, write_conditioning_artifact

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _artifact_paths(fingerprint: str) -> tuple[Path, Path]:
    if not SHA256_RE.fullmatch(fingerprint):
        raise ValueError("fingerprint must be a lowercase SHA-256 value")
    output_root = Path(folder_paths.get_output_directory()).resolve()
    root = output_root / "ai-video-generator" / "conditioning"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{fingerprint}.safetensors", root / f"{fingerprint}.json"


def _pack_h3_latent(latent):
    if not isinstance(latent, dict):
        raise ValueError("H3 latent must be a dictionary")
    samples = latent.get("samples")
    if not isinstance(samples, NestedTensor):
        raise ValueError("H3 latent samples must be a ComfyUI NestedTensor")
    packed = dict(latent)
    packed["samples"] = list(samples.tensors)
    return packed


def _unpack_h3_latent(latent):
    if not isinstance(latent, dict):
        raise ValueError("cached H3 latent must be a dictionary")
    samples = latent.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cached H3 latent samples must be a non-empty tensor list")
    unpacked = dict(latent)
    unpacked["samples"] = NestedTensor(samples)
    return unpacked


class AVGSaveConditioningArtifact:
    CATEGORY = "AI Video Generator/conditioning"
    FUNCTION = "save"
    INPUT_TYPES = classmethod(
        lambda cls: {
            "required": {
                "conditioning": ("CONDITIONING",),
                "fingerprint": ("STRING", {"default": ""}),
            }
        }
    )
    OUTPUT_NODE = True
    RETURN_NAMES = ("manifest_path",)
    RETURN_TYPES = ("STRING",)

    def save(self, conditioning, fingerprint: str):
        tensor_path, manifest_path = _artifact_paths(fingerprint)
        result = write_conditioning_artifact(
            conditioning,
            fingerprint=fingerprint,
            tensor_path=tensor_path,
            manifest_path=manifest_path,
            metadata={"producer": "AVGSaveConditioningArtifact", "schema": "h3-conditioning-v1"},
        )
        return (result.manifest_path,)


class AVGLoadConditioningArtifact:
    CATEGORY = "AI Video Generator/conditioning"
    FUNCTION = "load"
    INPUT_TYPES = classmethod(
        lambda cls: {
            "required": {"fingerprint": ("STRING", {"default": ""})}
        }
    )
    RETURN_NAMES = ("conditioning",)
    RETURN_TYPES = ("CONDITIONING",)

    def load(self, fingerprint: str):
        tensor_path, manifest_path = _artifact_paths(fingerprint)
        conditioning, _metadata = read_conditioning_artifact(
            tensor_path=tensor_path,
            manifest_path=manifest_path,
            expected_fingerprint=fingerprint,
        )
        return (conditioning,)


class AVGSaveH3StaticConditioning:
    CATEGORY = "AI Video Generator/conditioning"
    FUNCTION = "save"
    INPUT_TYPES = classmethod(
        lambda cls: {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "fingerprint": ("STRING", {"default": ""}),
            }
        }
    )
    OUTPUT_NODE = True
    RETURN_NAMES = ("manifest_path",)
    RETURN_TYPES = ("STRING",)

    def save(self, conditioning, latent, fingerprint: str):
        tensor_path, manifest_path = _artifact_paths(fingerprint)
        result = write_conditioning_artifact(
            {"conditioning": conditioning, "latent": _pack_h3_latent(latent)},
            fingerprint=fingerprint,
            tensor_path=tensor_path,
            manifest_path=manifest_path,
            metadata={"producer": "AVGSaveH3StaticConditioning", "schema": "h3-static-v1"},
        )
        return (result.manifest_path,)


class AVGLoadH3StaticConditioning:
    CATEGORY = "AI Video Generator/conditioning"
    FUNCTION = "load"
    INPUT_TYPES = classmethod(
        lambda cls: {"required": {"fingerprint": ("STRING", {"default": ""})}}
    )
    RETURN_NAMES = ("conditioning", "latent")
    RETURN_TYPES = ("CONDITIONING", "LATENT")

    def load(self, fingerprint: str):
        tensor_path, manifest_path = _artifact_paths(fingerprint)
        payload, metadata = read_conditioning_artifact(
            tensor_path=tensor_path,
            manifest_path=manifest_path,
            expected_fingerprint=fingerprint,
        )
        if metadata.get("schema") != "h3-static-v1":
            raise ValueError("artifact is not an H3 static conditioning bundle")
        if not isinstance(payload, dict) or set(payload) != {"conditioning", "latent"}:
            raise ValueError("H3 static conditioning bundle has an invalid structure")
        return payload["conditioning"], _unpack_h3_latent(payload["latent"])


NODE_CLASS_MAPPINGS = {
    "AVGSaveConditioningArtifact": AVGSaveConditioningArtifact,
    "AVGLoadConditioningArtifact": AVGLoadConditioningArtifact,
    "AVGSaveH3StaticConditioning": AVGSaveH3StaticConditioning,
    "AVGLoadH3StaticConditioning": AVGLoadH3StaticConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AVGSaveConditioningArtifact": "AVG Save H3 Conditioning",
    "AVGLoadConditioningArtifact": "AVG Load H3 Conditioning",
    "AVGSaveH3StaticConditioning": "AVG Save H3 Static Bundle",
    "AVGLoadH3StaticConditioning": "AVG Load H3 Static Bundle",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
