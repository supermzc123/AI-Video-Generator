from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ai_video_generator.domain.chain import FrozenModel


class ExportInput(FrozenModel):
    path: Path
    duration_seconds: float | None = Field(default=None, gt=0)
    trim_head_seconds: float = Field(default=0, ge=0)
    trim_tail_seconds: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_trim(self) -> ExportInput:
        if self.trim_tail_seconds and self.duration_seconds is None:
            raise ValueError("tail trimming requires the probed input duration")
        if (
            self.duration_seconds is not None
            and self.trim_head_seconds + self.trim_tail_seconds >= self.duration_seconds
        ):
            raise ValueError("input trimming must leave a positive duration")
        return self


class ExportSpec(FrozenModel):
    inputs: tuple[ExportInput, ...] = Field(min_length=1)
    output_path: Path
    width: int = Field(ge=32)
    height: int = Field(ge=32)
    fps: int = Field(default=24, ge=1, le=120)
    video_codec: Literal["libx264"] = "libx264"
    audio_codec: Literal["aac"] = "aac"
    external_audio_path: Path | None = None
    external_audio_volume: float = Field(default=1.0, ge=0, le=4)
    loop_external_audio: bool = False

    @model_validator(mode="after")
    def validate_paths(self) -> ExportSpec:
        if any(item.path == self.output_path for item in self.inputs):
            raise ValueError("output path must not overwrite an input segment")
        if self.external_audio_path == self.output_path:
            raise ValueError("output path must not overwrite the external audio input")
        return self


class ExportPlan(FrozenModel):
    command: tuple[str, ...]
    concat_manifest: str


class ExportResult(FrozenModel):
    output_path: str
    byte_size: int = Field(ge=0)
    sha256: str
    command: tuple[str, ...]


class ExportExecutionError(RuntimeError):
    pass


def order_segment_ids_for_export(
    segment_ids: Sequence[str], workspace_payload: dict[str, object]
) -> tuple[str, ...]:
    """Resolve media order from the approved prompt list, never from opaque IDs."""
    prompts = workspace_payload.get("prompts")
    h3_prompts = prompts.get("h3Prompts") if isinstance(prompts, dict) else None
    if not isinstance(h3_prompts, list):
        raise ValueError("workspace has no H3 prompt order for export")
    expected = [
        str(item.get("segmentId") or "")
        for item in h3_prompts
        if isinstance(item, dict) and str(item.get("segmentId") or "")
    ]
    if len(expected) != len(set(expected)):
        raise ValueError("workspace H3 prompt order contains duplicate segment IDs")
    actual = tuple(segment_ids)
    if len(actual) != len(set(actual)):
        raise ValueError("export dependencies contain duplicate segment IDs")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            "export segment set does not match the approved prompt order "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return tuple(item for item in expected if item in set(actual))


def compile_export_plan(spec: ExportSpec, ffmpeg_binary: str = "ffmpeg") -> ExportPlan:
    """Compile an FFmpeg command without touching media or starting a process."""
    concat_lines: list[str] = []
    for item in spec.inputs:
        escaped = str(item.path.resolve()).replace("'", "'\\''")
        concat_lines.append(f"file '{escaped}'")
        if item.trim_head_seconds:
            concat_lines.append(f"inpoint {item.trim_head_seconds:.6f}")
        if item.trim_tail_seconds:
            assert item.duration_seconds is not None
            concat_lines.append(f"outpoint {item.duration_seconds - item.trim_tail_seconds:.6f}")

    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "{concat_manifest}",
    ]
    filter_parts = [
        f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease",
        f"pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2",
        f"fps={spec.fps}",
        "format=yuv420p",
    ]
    if spec.external_audio_path is not None:
        if spec.loop_external_audio:
            command.extend(("-stream_loop", "-1"))
        command.extend(("-i", str(spec.external_audio_path.resolve())))
        command.extend(
            (
                "-filter_complex",
                f"[0:v]{','.join(filter_parts)}[v];"
                f"[0:a][1:a]amix=inputs=2:duration=first:weights='1 "
                f"{spec.external_audio_volume}'[a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
            )
        )
    else:
        command.extend(("-vf", ",".join(filter_parts), "-map", "0:v", "-map", "0:a?"))

    command.extend(
        (
            "-c:v",
            spec.video_codec,
            "-c:a",
            spec.audio_codec,
            "-movflags",
            "+faststart",
            "-shortest",
            "-y",
            str(spec.output_path.resolve()),
        )
    )
    return ExportPlan(command=tuple(command), concat_manifest="\n".join(concat_lines) + "\n")


async def run_export(
    spec: ExportSpec,
    *,
    work_directory: Path,
    ffmpeg_binary: str = "ffmpeg",
) -> ExportResult:
    """Execute one approved export using temporary files and an atomic final publish."""
    work_directory.mkdir(parents=True, exist_ok=True)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    manifest_path = work_directory / f"concat-{nonce}.txt"
    temporary_output = work_directory / f"export-{nonce}{spec.output_path.suffix or '.mp4'}"
    execution_spec = spec.model_copy(update={"output_path": temporary_output})
    plan = compile_export_plan(execution_spec, ffmpeg_binary)
    command = tuple(
        str(manifest_path) if value == "{concat_manifest}" else value for value in plan.command
    )
    manifest_path.write_text(plan.concat_manifest, encoding="utf-8")
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace")[-4000:]
            raise ExportExecutionError(f"FFmpeg export failed: {message}")
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise ExportExecutionError("FFmpeg completed without a non-empty output")
        sha256_value = _sha256_file(temporary_output)
        byte_size = temporary_output.stat().st_size
        temporary_output.replace(spec.output_path)
        return ExportResult(
            output_path=str(spec.output_path),
            byte_size=byte_size,
            sha256=sha256_value,
            command=command,
        )
    finally:
        manifest_path.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
