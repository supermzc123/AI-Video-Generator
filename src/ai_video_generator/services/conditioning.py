import hashlib
import json
import unicodedata
from typing import Any

from ai_video_generator.domain import ConditioningStack, GenerationSegment


def normalize_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFC", prompt)
    return " ".join(normalized.split())


def conditioning_fingerprint(
    segment: GenerationSegment,
    stack: ConditioningStack,
) -> str:
    payload: dict[str, Any] = {
        "schema": "conditioning-fingerprint-v1",
        "prompt": normalize_prompt(segment.normalized_prompt),
        "generation_mode": segment.generation_mode.value,
        "width": segment.width,
        "height": segment.height,
        "fps": segment.fps,
        "sample_frames": segment.sample_frames,
        "assets": [
            binding.model_dump(mode="json")
            for binding in (*segment.common_assets, *segment.local_assets)
        ],
        "stack": stack.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
