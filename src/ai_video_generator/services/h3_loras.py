from __future__ import annotations

import hashlib
import json
from typing import Any


def normalize_project_loras(payload: dict[str, Any]) -> tuple[dict[str, object], ...]:
    """Return the enabled, ordered project LoRA stack in execution form."""
    raw_loras = payload.get("h3Loras")
    if not isinstance(raw_loras, list):
        return ()
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for raw in raw_loras:
        if not isinstance(raw, dict) or raw.get("enabled") is False:
            continue
        name = str(raw.get("name") or "").strip()
        if not name or name in names:
            continue
        strength = float(raw.get("strength", 1.0))
        if not -4.0 <= strength <= 4.0:
            raise ValueError(f"LoRA strength must be between -4 and 4: {name}")
        names.add(name)
        result.append({"name": name, "strength": strength})
    return tuple(result)


def model_stack_sha256(profile: dict[str, object]) -> str:
    relevant = {
        key: profile.get(key)
        for key in (
            "diffusion_model",
            "turbo_enabled",
            "turbo_lora",
            "kitchen_attention_enabled",
            "low_vram",
            "project_loras",
        )
    }
    return hashlib.sha256(
        json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def diffusion_affinity(profile: dict[str, Any] | None) -> str:
    digest = str((profile or {}).get("model_stack_sha256") or "default")
    return f"h3:diffusion:{digest[:16]}"
