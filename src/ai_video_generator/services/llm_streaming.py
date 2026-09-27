from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from ai_video_generator.llm.budget import MAX_OUTPUT_CHARACTERS
from ai_video_generator.llm.client import llm_delta_callback


@dataclass
class _Operation:
    operation_id: str
    project_id: str
    kind: str
    scope: str
    state: str = "running"
    text: str = ""
    result: object | None = None
    error: dict[str, object] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: int = 0
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "project_id": self.project_id,
            "kind": self.kind,
            "scope": self.scope,
            "state": self.state,
            "text": self.text,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


_operations: dict[str, _Operation] = {}


def list_llm_operations(
    project_id: str, *, active_only: bool = False
) -> tuple[dict[str, object], ...]:
    _prune_operations()
    return tuple(
        item.snapshot()
        for item in sorted(_operations.values(), key=lambda value: value.created_at)
        if item.project_id == project_id and (not active_only or item.state == "running")
    )


def stream_llm_operation(
    operation: Callable[[], Awaitable[object]],
    *,
    project_id: str = "",
    kind: str = "llm",
    scope: str = "",
    operation_id: str | None = None,
) -> StreamingResponse:
    """Run an LLM operation independently from any individual SSE subscriber."""
    resolved_id = operation_id or str(uuid.uuid4())
    record = _operations.get(resolved_id)
    if record is None:
        record = _Operation(
            operation_id=resolved_id,
            project_id=project_id,
            kind=kind,
            scope=scope,
        )
        _operations[resolved_id] = record
        record.task = asyncio.create_task(_run_operation(record, operation))
    elif (record.project_id, record.kind, record.scope) != (project_id, kind, scope):
        raise ValueError("LLM operation ID is already bound to another request")
    return _stream_response(record)


def resume_llm_operation(operation_id: str, project_id: str) -> StreamingResponse:
    record = _operations.get(operation_id)
    if record is None or record.project_id != project_id:
        raise KeyError(operation_id)
    return _stream_response(record)


async def cancel_llm_operation(operation_id: str, project_id: str) -> dict[str, object]:
    record = _operations.get(operation_id)
    if record is None or record.project_id != project_id:
        raise KeyError(operation_id)
    if record.state != "running" or record.task is None:
        return record.snapshot()
    record.error = {"status": 499, "detail": "用户已停止生成"}
    record.task.cancel()
    await asyncio.gather(record.task, return_exceptions=True)
    return record.snapshot()


async def _run_operation(record: _Operation, operation: Callable[[], Awaitable[object]]) -> None:
    def publish(delta: str) -> None:
        # UI replay is a bounded tail; output validation happens before commit.
        record.text = (record.text + delta)[-MAX_OUTPUT_CHARACTERS:]
        record.updated_at = datetime.now(UTC)
        record.version += 1
        _notify(record)

    token = llm_delta_callback.set(publish)
    try:
        record.result = await operation()
        record.state = "succeeded"
    except asyncio.CancelledError:
        record.state = "failed"
        if record.error is None:
            record.error = {"status": 503, "detail": "LLM operation stopped with the backend"}
        raise
    except Exception as exc:
        record.state = "failed"
        record.error = {
            "status": int(getattr(exc, "status_code", 500)),
            "detail": getattr(exc, "detail", str(exc)),
        }
    finally:
        llm_delta_callback.reset(token)
        record.updated_at = datetime.now(UTC)
        record.version += 1
        _notify(record)


def _notify(record: _Operation) -> None:
    async def notify() -> None:
        async with record.condition:
            record.condition.notify_all()

    asyncio.create_task(notify())


def _stream_response(record: _Operation) -> StreamingResponse:
    async def events():
        previous = ""
        yield _event("operation", record.snapshot())
        while True:
            if record.text != previous:
                delta = (
                    record.text[len(previous) :]
                    if record.text.startswith(previous)
                    else record.text
                )
                previous = record.text
                yield _event("delta", delta)
            if record.state == "succeeded":
                yield _event("result", record.result)
                return
            if record.state == "failed":
                yield _event("error", record.error or {"status": 500, "detail": "LLM failed"})
                return
            seen_version = record.version
            try:
                async with record.condition:
                    await asyncio.wait_for(
                        record.condition.wait_for(lambda seen=seen_version: record.version != seen),
                        timeout=15,
                    )
            except TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _event(kind: str, value: object) -> str:
    payload = json.dumps(jsonable_encoder(value), ensure_ascii=False, separators=(",", ":"))
    return f"event: {kind}\ndata: {payload}\n\n"


def _prune_operations() -> None:
    cutoff = datetime.now(UTC) - timedelta(hours=6)
    for operation_id, record in tuple(_operations.items()):
        if record.state != "running" and record.updated_at < cutoff:
            _operations.pop(operation_id, None)
