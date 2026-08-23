from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ai_video_generator.config import Settings, load_runtime_settings
from ai_video_generator.llm import (
    ChatMessage,
    ImageURL,
    ImageURLContentPart,
    LLMClientError,
    OpenAICompatibleClient,
    TextContentPart,
    VideoURL,
    VideoURLContentPart,
)
from ai_video_generator.services.media_review import inspect_video_media

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / ".tmp" / "unattended-20260816-020855" / "review-media"
REVIEW_INSTRUCTION = (
    "审核这个四秒合成测试视频。正常样本中测试图案从固定黑色遮挡条后经过，这是正常透视遮挡，"
    "不得判为身体残缺；纯黑样本应报告 black_frame 错误。只返回 JSON："
    '{"accepted":true|false,"confidence":0.0,"issues":['
    '{"category":"black_frame|artifact|other","severity":"warning|error",'
    '"message":"...","evidence":"..."}]}。'
)


def _data_url(path: Path, media_type: str) -> str:
    return f"data:{media_type};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_frames(video: Path) -> tuple[Path, ...]:
    frame_root = RUN_ROOT / f"{video.stem}-frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=1,scale=640:-2",
            "-frames:v",
            "4",
            "-y",
            str(frame_root / "frame-%02d.jpg"),
        ],
        check=True,
    )
    return tuple(sorted(frame_root.glob("frame-*.jpg")))


async def _complete(parts: tuple[object, ...]) -> tuple[dict[str, object], float]:
    runtime = load_runtime_settings(Settings(_env_file=None))
    if not runtime.llm_base_url or not runtime.llm_model or not runtime.llm_api_key:
        raise RuntimeError("Gemini runtime configuration is incomplete")
    started = time.perf_counter()
    async with OpenAICompatibleClient(
        base_url=runtime.llm_base_url,
        model=runtime.llm_model,
        api_key=runtime.llm_api_key.get_secret_value(),
        timeout_seconds=runtime.llm_timeout_seconds,
        proxy=runtime.network_proxy,
    ) as client:
        raw = await client.complete_json(
            (ChatMessage(role="user", content=parts),)  # type: ignore[arg-type]
        )
    elapsed = time.perf_counter() - started
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Reviewer response is not an object")
    return value, elapsed


async def _review_video(video: Path) -> dict[str, object]:
    try:
        result, elapsed = await _complete(
            (
                TextContentPart(text=REVIEW_INSTRUCTION),
                VideoURLContentPart(video_url=VideoURL(url=_data_url(video, "video/mp4"))),
            )
        )
        return {
            "requested_mode": "video",
            "actual_mode": "video",
            "fallback_reason": None,
            "elapsed_seconds": round(elapsed, 3),
            "result": result,
        }
    except LLMClientError as exc:
        frames = _extract_frames(video)
        parts: list[object] = [TextContentPart(text=REVIEW_INSTRUCTION)]
        parts.extend(
            ImageURLContentPart(
                image_url=ImageURL(url=_data_url(frame, "image/jpeg"), detail="low")
            )
            for frame in frames
        )
        result, elapsed = await _complete(tuple(parts))
        return {
            "requested_mode": "video",
            "actual_mode": "frames",
            "fallback_reason": str(exc)[:1000],
            "elapsed_seconds": round(elapsed, 3),
            "result": result,
        }


async def _review_frames(video: Path) -> dict[str, object]:
    frames = _extract_frames(video)
    parts: list[object] = [TextContentPart(text=REVIEW_INSTRUCTION)]
    parts.extend(
        ImageURLContentPart(
            image_url=ImageURL(url=_data_url(frame, "image/jpeg"), detail="low")
        )
        for frame in frames
    )
    result, elapsed = await _complete(tuple(parts))
    return {
        "requested_mode": "frames",
        "actual_mode": "frames",
        "fallback_reason": None,
        "elapsed_seconds": round(elapsed, 3),
        "result": result,
    }


async def main() -> None:
    samples = {
        name: RUN_ROOT / name
        for name in ("normal-occlusion.mp4", "black.mp4", "freeze.mp4")
    }
    deterministic: dict[str, object] = {}
    for name, path in samples.items():
        started = time.perf_counter()
        issues = await inspect_video_media(path)
        deterministic[name] = {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "issues": [issue.model_dump(mode="json") for issue in issues],
        }

    target = RUN_ROOT / "review-validation.json"
    if "--deterministic-only" in sys.argv and target.is_file():
        report = json.loads(target.read_text(encoding="utf-8"))
        report["updated_at"] = datetime.now(UTC).isoformat()
        report["deterministic"] = deterministic
        report["deterministic_retested_after_parser_fix"] = True
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(target)
        return

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "samples": {name: str(path) for name, path in samples.items()},
        "deterministic": deterministic,
        "gemini": {
            "normal_video_capable": await _review_video(samples["normal-occlusion.mp4"]),
            "normal_frame_only": await _review_frames(samples["normal-occlusion.mp4"]),
            "black_frame_only": await _review_frames(samples["black.mp4"]),
        },
        "secrets_logged": False,
    }
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    asyncio.run(main())
