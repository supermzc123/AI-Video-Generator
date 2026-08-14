# Portions derived from ComfyUI-H3-Motion-Context
# Copyright (C) 2026 NikoDemon80 and contributors
# Modified for MiniMax H3 Motion Director, 2026-08-09
# Licensed under GNU GPL v3.0. See LICENSE and NOTICE.

"""Let keyframes and refs coexist.

`MiniMaxH3.extra_conds` in comfy/model_base.py fills the payload from two
independent `if` blocks. The keyframe block sets `cond_video_latents`, then
the refs block **overwrites** it:

    if keyframes is not None:
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
    if refs is not None:
        payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]
        payload["cond_audio_latents"] = [r["audio_latent"] for r in refs ...]

So attaching an audio-only ref alongside keyframes wipes the keyframe video
content: an audio-only block has no "latent" key, the list comes back empty,
and the cond rows the layout built have nothing to fill them.

The layout itself handles the combination fine. Keyframe cond rows are
emitted first, ref rows second, target rows last, which is exactly the order
the forward pass expects when it writes rows into the never-denoised slots.
Only this payload assignment is in the way.

This wrapper re-runs the same logic and concatenates instead, keeping
keyframe latents first to match the row order. Graphs using only one
mechanism are unaffected: with no refs the ref list is empty, with no
keyframes the keyframe list is.
"""

import inspect
import logging

import comfy.model_base as model_base

_LOG = logging.getLogger("h3_motion_context")

_orig_extra_conds = None
_applied = False
_failure_reason = None


def merge_payload_latents(payload, keyframes, refs, frame_count=None):
    """Keep row payload order identical to PackedLayout: keyframes, then refs."""
    if not isinstance(payload, dict):
        raise TypeError("MiniMax H3 payload must be a dictionary")
    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [ref["latent"] for ref in refs if "latent" in ref]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [
        ref["audio_latent"]
        for ref in refs
        if ref.get("audio_latent") is not None
    ]
    if frame_count is not None:
        payload["frame_count"] = frame_count
    return payload


def _patched_extra_conds(self, **kwargs):
    out = _orig_extra_conds(self, **kwargs)

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out  # only one mechanism in play, stock behaviour is correct

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "h3_motion_context: ComfyUI returned an unexpected MiniMax H3 "
            "payload while keyframes and refs coexist. Refusing to sample "
            "because conditioning row payloads may be misordered."
        )

    fc = kwargs.get("minimax_frame_count", None)
    merge_payload_latents(payload, keyframes, refs, frame_count=fc)
    return out


def apply_patch():
    global _orig_extra_conds, _applied, _failure_reason
    if _applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _failure_reason = "MiniMaxH3.extra_conds was not found"
        _LOG.warning("h3_motion_context: %s; keyframes and refs cannot be combined",
                     _failure_reason)
        return False
    try:
        sig = inspect.signature(cls.extra_conds)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in sig.parameters.values()):
            raise RuntimeError("MiniMaxH3.extra_conds no longer accepts **kwargs")
        source = inspect.getsource(cls.extra_conds)
        required = (
            "minimax_keyframes",
            "minimax_refs",
            "minimax_payload",
            "cond_video_latents",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise RuntimeError(
                "MiniMaxH3.extra_conds no longer contains expected fields: %s"
                % ", ".join(missing)
            )
    except Exception as exc:
        _failure_reason = str(exc)
        _LOG.warning("h3_motion_context: payload compatibility self-test failed "
                     "(%s), patch not applied", exc)
        return False
    _orig_extra_conds = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _applied = True
    _failure_reason = None
    _LOG.info("h3_motion_context: keyframe/ref coexistence enabled")
    return True


def is_applied():
    return _applied


def failure_reason():
    return _failure_reason
