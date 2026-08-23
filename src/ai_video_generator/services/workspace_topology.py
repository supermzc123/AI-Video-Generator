from __future__ import annotations

import copy
import math
from typing import Any


def normalize_workspace_topology(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove derived prompt records that no longer belong to the shot topology."""
    normalized = copy.deepcopy(payload)
    valid_segment_ids = _valid_segment_ids(normalized)
    prompts = normalized.get("prompts")
    if not isinstance(prompts, dict):
        return normalized
    h3_prompts = prompts.get("h3Prompts")
    if isinstance(h3_prompts, list):
        prompts["h3Prompts"] = [
            item
            for item in h3_prompts
            if isinstance(item, dict)
            and str(item.get("segmentId") or item.get("segment_id") or "")
            in valid_segment_ids
        ]
    return normalized


def _valid_segment_ids(payload: dict[str, Any]) -> frozenset[str]:
    shots = payload.get("shots")
    if not isinstance(shots, list):
        return frozenset()
    values: set[str] = set()
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("id") or "")
        if not shot_id:
            continue
        configured = shot.get("motionSegments")
        if isinstance(configured, list) and configured:
            count = len(configured)
            for index, segment in enumerate(configured):
                segment = segment if isinstance(segment, dict) else {}
                values.add(
                    str(
                        segment.get("segmentId")
                        or segment.get("segment_id")
                        or segment.get("id")
                        or f"{shot_id}.C{index + 1:02d}"
                    )
                )
            continue
        duration = float(shot.get("durationSeconds") or 0)
        count = 1 if duration <= 15 else max(2, math.ceil(duration / 12))
        values.update(f"{shot_id}.C{index + 1:02d}" for index in range(count))
    return frozenset(values)
