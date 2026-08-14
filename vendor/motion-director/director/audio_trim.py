"""Frame-exact IMAGE/AUDIO trimming for exported Motion Director segments."""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.audio_trim")


def audio_has_samples(audio: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(audio, dict)
        and isinstance(audio.get("waveform"), torch.Tensor)
        and int(audio["waveform"].numel()) > 0
    )


def samples_for_frames(frame_count: int, fps: float, sample_rate: int) -> int:
    if fps <= 0 or sample_rate <= 0:
        raise ValueError("Motion Director: fps and audio sample rate must be positive.")
    return int(round(max(0, int(frame_count)) / float(fps) * int(sample_rate)))


def trim_segment_av(
    images: torch.Tensor,
    audio: dict[str, Any] | None,
    *,
    head_frames: int,
    target_frames: int,
    fps: float,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """Remove the context head and crop both streams to one exact duration.

    H3's audio latent is 40 Hz and commonly decodes a few milliseconds past
    the picture. Integer sample boundaries are computed from the absolute
    frame boundaries, so rounding never accumulates across a segment chain.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Motion Director: decoded video is not a valid IMAGE batch.")
    head = max(0, int(head_frames))
    target = max(1, int(target_frames))
    total = int(images.shape[0])
    required = head + target
    if total < required:
        raise ValueError(
            "Motion Director: H3 decoded %d frames, but %d are required "
            "(%d Motion Context head + %d requested output)."
            % (total, required, head, target)
        )
    out_images = images[head:required]
    if int(out_images.shape[0]) != target:
        raise RuntimeError(
            "Motion Director: exact-duration trim failed (%d != %d frames)."
            % (int(out_images.shape[0]), target)
        )

    if not audio_has_samples(audio):
        return out_images, audio

    waveform = audio["waveform"]
    if waveform.ndim != 3:
        raise ValueError(
            "Motion Director: decoded AUDIO waveform must be [batch, channels, samples]."
        )
    sr = int(audio.get("sample_rate") or 0)
    start = samples_for_frames(head, fps, sr)
    # Compute the exported duration directly, rather than subtracting two
    # independently rounded absolute boundaries (which can differ by one
    # sample and then accumulate across a long segment chain).
    wanted = samples_for_frames(target, fps, sr)
    stop = start + wanted
    have = int(waveform.shape[-1])
    if start >= have:
        raise ValueError(
            "Motion Director: audio is too short to remove the %.3fs context head."
            % (head / float(fps))
        )
    out_wave = waveform[..., start:min(stop, have)]
    missing = wanted - int(out_wave.shape[-1])
    if missing > 0:
        # At most one 40 Hz H3 audio step is a normal grid-rounding shortfall.
        tolerance = max(1, int(round(sr / 40.0)))
        if missing > tolerance:
            raise ValueError(
                "Motion Director: decoded audio is %.2fms too short for %d frames. "
                "Refusing to create a long silent tail that would corrupt the "
                "next segment's Motion Audio Context."
                % (missing / sr * 1000.0, target)
            )
        pad = torch.zeros(
            *out_wave.shape[:-1],
            missing,
            dtype=out_wave.dtype,
            device=out_wave.device,
        )
        out_wave = torch.cat((out_wave, pad), dim=-1)
        log.warning(
            "Motion Director: padded %d audio samples (%.2fms) after H3 grid rounding.",
            missing,
            missing / sr * 1000.0,
        )
    elif missing < 0:
        out_wave = out_wave[..., :wanted]

    if int(out_wave.shape[-1]) != wanted:
        raise RuntimeError("Motion Director: exact audio-duration trim failed.")
    return out_images, {"waveform": out_wave.contiguous(), "sample_rate": sr}


__all__ = ["audio_has_samples", "samples_for_frames", "trim_segment_av"]
