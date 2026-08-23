from __future__ import annotations

import hashlib
import json
import math
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

from ai_video_generator.domain import ComfyUIOutput, TaskKind, TaskWorkloadManifest


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def transcribe_to_srt(
    *,
    video_path: Path,
    output_path: Path,
    model_id: str,
    language: str,
    device: str,
    precision: str,
    model_root: Path,
) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Whisper执行器未安装；请重新安装完整控制平面") from exc
    resolved_device = "auto" if device == "auto" else device
    compute_type = "default" if precision == "auto" else precision
    model_root.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        model_id,
        device=resolved_device,
        compute_type=compute_type,
        download_root=str(model_root),
    )
    segments, _info = model.transcribe(
        str(video_path),
        language=None if language == "auto" else language,
        vad_filter=True,
    )
    lines: list[str] = []
    for index, segment in enumerate(segments, 1):
        text = str(segment.text).strip()
        if not text:
            continue
        lines.extend(
            (
                str(index),
                f"{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}",
                text,
                "",
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output_path)


async def finalize_delivery(
    *,
    master_path: Path,
    output_path: Path,
    subtitle_path: Path | None,
    burn_in: bool,
    ffmpeg_binary: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.mp4")
    if subtitle_path is None:
        shutil.copyfile(master_path, temporary)
    else:
        command = [ffmpeg_binary, "-hide_banner", "-nostdin", "-i", str(master_path)]
        if burn_in:
            escaped = (
                str(subtitle_path.resolve())
                .replace("\\", "/")
                .replace(":", "\\:")
                .replace("'", "\\'")
            )
            command.extend(("-vf", f"subtitles='{escaped}'", "-c:v", "libx264", "-c:a", "copy"))
        else:
            command.extend(
                ("-i", str(subtitle_path), "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text")
            )
        command.extend(("-movflags", "+faststart", "-y", str(temporary)))
        import asyncio

        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "FFmpeg字幕封装失败：" + stderr.decode("utf-8", errors="replace")[-2000:]
            )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("最终交付没有产生有效文件")
    temporary.replace(output_path)


def interpolation_plan(
    source_fps: int, target_fps: int, *, maximum_fps: int = 120
) -> tuple[int, int]:
    """Return (generation multiplier, intermediate fps).

    Fractional interpolation is never delegated to a model. The model generates
    the least common multiple and FFmpeg performs the deterministic final sample.
    """
    if source_fps <= 0 or target_fps not in {48, 60, 120}:
        raise ValueError("unsupported interpolation frame rate")
    intermediate = math.lcm(source_fps, target_fps)
    if intermediate > maximum_fps:
        raise ValueError(
            f"{source_fps}→{target_fps} requires {intermediate} fps, "
            f"exceeding Profile limit {maximum_fps}"
        )
    multiplier = intermediate // source_fps
    if multiplier < 2:
        raise ValueError("interpolation target must require at least 2x generation")
    return multiplier, intermediate


def compile_rife_manifest(
    *,
    project_id: str,
    segment_id: str,
    source_task_id: str,
    source_fps: int,
    target_fps: int,
    model_id: str,
    node_schema_sha256: str,
) -> TaskWorkloadManifest:
    resource = files("ai_video_generator.resources").joinpath("postprocessing", "rife.api.json")
    raw = resource.read_bytes()
    prompt: dict[str, dict[str, Any]] = json.loads(raw)
    multiplier, intermediate_fps = interpolation_plan(source_fps, target_fps)
    prompt["1"]["inputs"]["force_rate"] = float(source_fps)
    prompt["2"]["inputs"]["ckpt_name"] = model_id
    prompt["2"]["inputs"]["multiplier"] = multiplier
    prompt["3"]["inputs"]["frame_rate"] = float(intermediate_fps)
    prompt["3"]["inputs"]["filename_prefix"] = (
        f"AI-Video-Generator/postprocessing/{project_id}/{segment_id}/rife"
    )
    return TaskWorkloadManifest(
        task_kind=TaskKind.RIFE,
        workflow_sha256=hashlib.sha256(raw).hexdigest(),
        node_schema_sha256=node_schema_sha256,
        prompt=prompt,
        outputs=(ComfyUIOutput(node_id="3", media_type="video/mp4"),),
        required_node_types=("VHS_LoadVideo", "RIFE VFI", "VHS_VideoCombine"),
        workflow_template_id=None,
        context={
            "project_id": project_id,
            "segment_id": segment_id,
            "source_task_id": source_task_id,
            "input_mount_path": "AI-Video-Generator/postprocessing/input.mp4",
            "source_fps": str(source_fps),
            "target_fps": str(target_fps),
            "intermediate_fps": str(intermediate_fps),
            "profile_id": "interpolation:rife",
            "profile_revision": "1",
            "model_id": model_id,
        },
    )
