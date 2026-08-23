from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from ai_video_generator.llm.client import llm_delta_callback


def stream_llm_operation(operation: Callable[[], Awaitable[object]]) -> StreamingResponse:
    """Expose one LLM operation as a consistent delta/result/error SSE stream."""

    async def events():
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        def publish(delta: str) -> None:
            queue.put_nowait(("delta", delta))

        async def run() -> None:
            token = llm_delta_callback.set(publish)
            try:
                await queue.put(("result", await operation()))
            except Exception as exc:
                await queue.put(
                    (
                        "error",
                        {
                            "status": int(getattr(exc, "status_code", 500)),
                            "detail": getattr(exc, "detail", str(exc)),
                        },
                    )
                )
            finally:
                llm_delta_callback.reset(token)

        task = asyncio.create_task(run())
        try:
            while True:
                kind, value = await queue.get()
                payload = json.dumps(
                    jsonable_encoder(value), ensure_ascii=False, separators=(",", ":")
                )
                yield f"event: {kind}\ndata: {payload}\n\n"
                if kind in {"result", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
