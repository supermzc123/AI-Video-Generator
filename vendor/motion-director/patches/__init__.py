"""Guarded MiniMax H3 runtime patches used by Motion Director."""

from .markers import MC_AUDIO_KEY, MC_KEY
from .h3_layout import (
    apply_patch as apply_layout_patch,
    failure_reason as layout_failure_reason,
    is_applied as layout_patch_applied,
)
from .h3_payload import (
    apply_patch as apply_payload_patch,
    failure_reason as payload_failure_reason,
    is_applied as payload_patch_applied,
)


def apply_motion_context_patches() -> bool:
    """Apply both patches. Motion Context is usable only when both pass."""
    return bool(apply_layout_patch() and apply_payload_patch())


def motion_context_patch_status() -> tuple[bool, str]:
    problems = []
    if not layout_patch_applied():
        problems.append(layout_failure_reason() or "PackedLayout patch is inactive")
    if not payload_patch_applied():
        problems.append(payload_failure_reason() or "H3 payload patch is inactive")
    return not problems, "; ".join(problems)


__all__ = [
    "MC_AUDIO_KEY",
    "MC_KEY",
    "apply_motion_context_patches",
    "motion_context_patch_status",
]
