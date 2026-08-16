import json
from pathlib import Path

import pytest

from ai_video_generator.services import (
    ConditioningIntegrityError,
    ConditioningIOError,
    read_conditioning_artifact,
    write_conditioning_artifact,
)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def test_conditioning_safe_round_trip(tmp_path: Path) -> None:
    tensor_path = tmp_path / "blob.safetensors"
    manifest_path = tmp_path / "manifest.json"
    fingerprint = "a" * 64
    payload = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        {"minimax_frame_count": 22, "keyframes": (torch.ones(1), None)},
    ]

    result = write_conditioning_artifact(
        payload,
        fingerprint=fingerprint,
        tensor_path=tensor_path,
        manifest_path=manifest_path,
        metadata={"segment_id": "S001.C01"},
    )
    restored, metadata = read_conditioning_artifact(
        tensor_path=tensor_path,
        manifest_path=manifest_path,
        expected_fingerprint=fingerprint,
    )

    assert result.tensor_count == 2
    assert torch.equal(restored[0], payload[0])
    assert isinstance(restored[1]["keyframes"], tuple)
    assert metadata == {"segment_id": "S001.C01"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "safetensors+json-v1"


def test_conditioning_tamper_is_detected(tmp_path: Path) -> None:
    tensor_path = tmp_path / "blob.safetensors"
    manifest_path = tmp_path / "manifest.json"
    write_conditioning_artifact(
        {"tensor": torch.zeros(1)},
        fingerprint="b" * 64,
        tensor_path=tensor_path,
        manifest_path=manifest_path,
    )
    with tensor_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ConditioningIntegrityError, match="byte size"):
        read_conditioning_artifact(tensor_path=tensor_path, manifest_path=manifest_path)


def test_conditioning_rejects_python_objects(tmp_path: Path) -> None:
    with pytest.raises(ConditioningIOError, match="unsupported object type"):
        write_conditioning_artifact(
            object(),
            fingerprint="c" * 64,
            tensor_path=tmp_path / "blob.safetensors",
            manifest_path=tmp_path / "manifest.json",
        )
