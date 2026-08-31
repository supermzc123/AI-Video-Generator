from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from pathlib import Path

from ai_video_generator.domain.orchestration import ReviewMode
from ai_video_generator.domain.review import (
    ReviewIssue,
    ReviewIssueCategory,
    ReviewIssueSeverity,
    ReworkAction,
)


def classify_automatic_rework(
    issues: tuple[ReviewIssue, ...],
) -> tuple[ReworkAction | None, str]:
    """Choose only repairs that do not require creative interpretation."""
    categories = {issue.category for issue in issues}
    if categories & {ReviewIssueCategory.MEDIA, ReviewIssueCategory.BLACK_FRAME}:
        return ReworkAction.RETRY, "媒体损坏或持续黑场，重新运行原 H3 任务"
    unsafe = categories & {
        ReviewIssueCategory.ANATOMY,
        ReviewIssueCategory.IDENTITY,
        ReviewIssueCategory.CONTINUITY,
        ReviewIssueCategory.SEMANTIC,
        ReviewIssueCategory.COMPOSITION,
    }
    if unsafe:
        return (
            ReworkAction.REVISE_PROMPT,
            "问题涉及语义、身份、构图或连续性；当前单片段重写接口不能携带审核反馈并"
            "原子重编译 conditioning，已登记提示词修订请求并等待人工处理",
        )
    if categories and categories.issubset(
        {
            ReviewIssueCategory.ARTIFACT,
            ReviewIssueCategory.MOTION,
            ReviewIssueCategory.FREEZE,
            ReviewIssueCategory.AUDIO,
        }
    ):
        return ReworkAction.CHANGE_SEED, "检测到瞬态伪影或运动异常，更换 seed 后重试"
    if categories == {ReviewIssueCategory.OTHER} and issues:
        return ReworkAction.RETRY, "Reviewer 报告了明确问题但分类不可识别，重新运行原 H3 任务"
    return None, "Reviewer 未提供足以确定自动返工方式的分类证据"


def automatic_rework_policy(
    issues: tuple[ReviewIssue, ...],
    *,
    effective_mode: ReviewMode,
    previous_reworks: int,
    max_reworks: int = 2,
) -> tuple[ReworkAction | None, str]:
    if effective_mode != ReviewMode.AI_ONLY:
        return None, "项目当前处于人工审核模式，不在人工门之前自动派发返工"
    if previous_reworks >= max_reworks:
        return None, f"该片段已达到最多{max_reworks}次自动返工限制"
    return classify_automatic_rework(issues)


async def inspect_video_media(
    video_path: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str | None = None,
) -> tuple[ReviewIssue, ...]:
    """Run cheap, deterministic media checks before semantic review."""
    probe = ffprobe_binary or _ffprobe_for(ffmpeg_binary)
    issues: list[ReviewIssue] = []
    try:
        payload = await _run_json(
            probe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration,width,height",
            "-of",
            "json",
            str(video_path),
        )
    except (OSError, ValueError) as exc:
        return (
            ReviewIssue(
                category=ReviewIssueCategory.MEDIA,
                severity=ReviewIssueSeverity.ERROR,
                message="视频容器无法读取",
                evidence=str(exc)[:1000],
                suggested_action="重新运行原 H3 任务",
            ),
        )
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    try:
        duration = float(payload.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0
    if not video_streams or duration <= 0:
        issues.append(
            ReviewIssue(
                category=ReviewIssueCategory.MEDIA,
                severity=ReviewIssueSeverity.ERROR,
                message="视频缺少可解码画面或有效时长",
                suggested_action="重新运行原 H3 任务",
            )
        )
        return tuple(issues)
    if not audio_streams:
        issues.append(
            ReviewIssue(
                category=ReviewIssueCategory.AUDIO,
                severity=ReviewIssueSeverity.WARNING,
                message="未检测到音轨",
                suggested_action="确认该片段是否应包含 H3 原生音频",
            )
        )

    stderr = await _run_stderr(
        ffmpeg_binary,
        "-hide_banner",
        "-nostats",
        "-i",
        str(video_path),
        "-vf",
        "blackdetect=d=0.5:pix_th=0.10,freezedetect=n=-60dB:d=1.5",
        "-an",
        "-f",
        "null",
        "-",
    )
    issues.extend(_parse_filter_findings(stderr, duration))
    return tuple(issues)


def _ffprobe_for(ffmpeg_binary: str) -> str:
    path = Path(ffmpeg_binary)
    name = "ffprobe.exe" if path.suffix.lower() == ".exe" else "ffprobe"
    return str(path.with_name(name)) if path.parent != Path(".") else name


async def _run_json(*args: str) -> dict[str, object]:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise ValueError(stderr.decode("utf-8", errors="replace")[-1000:])
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("ffprobe returned an invalid document")
    return result


async def _run_stderr(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _stdout, stderr = await process.communicate()
    if process.returncode != 0:
        return ""
    return stderr.decode("utf-8", errors="replace")


def _parse_filter_findings(stderr: str, duration: float) -> tuple[ReviewIssue, ...]:
    issues: list[ReviewIssue] = []
    for line in stderr.splitlines():
        if "black_start:" in line and "black_end:" in line:
            values = _filter_values(line)
            start = values.get("black_start")
            end = values.get("black_end")
            if start is not None and end is not None:
                severity = (
                    ReviewIssueSeverity.ERROR
                    if end - start >= min(1.0, duration * 0.25)
                    else ReviewIssueSeverity.WARNING
                )
                issues.append(
                    ReviewIssue(
                        category=ReviewIssueCategory.BLACK_FRAME,
                        severity=severity,
                        message="检测到持续黑场",
                        start_seconds=start,
                        end_seconds=end,
                        evidence=line[-500:],
                        suggested_action="检查生成产物；持续黑场应重新生成",
                    )
                )
        elif "freeze_start:" in line:
            values = _filter_values(line)
            start = values.get("freeze_start")
            if start is not None:
                issues.append(
                    ReviewIssue(
                        category=ReviewIssueCategory.FREEZE,
                        severity=ReviewIssueSeverity.WARNING,
                        message="检测到持续冻结画面",
                        start_seconds=start,
                        evidence=line[-500:],
                        suggested_action="结合镜头设计确认静止是否符合预期",
                    )
                )
    return tuple(issues)


def _filter_values(line: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for match in re.finditer(
        r"(?P<key>black_start|black_end|black_duration|freeze_start|freeze_end|freeze_duration)"
        r":\s*(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+))",
        line,
    ):
        with suppress(ValueError):
            result[match.group("key")] = float(match.group("value"))
    return result
