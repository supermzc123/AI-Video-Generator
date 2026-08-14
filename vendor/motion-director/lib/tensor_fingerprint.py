"""Stable, compact identities for in-memory media tensors."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


def tensor_fingerprint(
    tensor: torch.Tensor | None,
    *,
    samples: int = 8192,
) -> dict[str, Any] | None:
    """Return a content-sensitive identity without serializing the full tensor."""
    if not isinstance(tensor, torch.Tensor):
        return None
    value = tensor.detach().cpu().contiguous()
    flat = value.reshape(-1)
    count = int(flat.numel())
    if count <= samples:
        probe = flat
    else:
        indices = torch.linspace(
            0,
            count - 1,
            steps=samples,
            dtype=torch.float64,
        ).long()
        probe = flat.index_select(0, indices)
    # NumPy cannot serialize every Torch dtype (notably bfloat16). Keep the
    # original dtype in metadata and normalize floating samples for hashing.
    probe_for_hash = probe.float() if probe.is_floating_point() else probe
    digest = hashlib.sha256(probe_for_hash.numpy().tobytes()).hexdigest()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": count,
        "probe_sha256": digest,
        "mean": float(flat.double().mean().item()) if count else 0.0,
        "square_mean": float((flat.double() ** 2).mean().item()) if count else 0.0,
    }


__all__ = ["tensor_fingerprint"]
