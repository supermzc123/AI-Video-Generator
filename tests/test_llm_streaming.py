import asyncio
import json

import pytest

from ai_video_generator.llm.client import llm_delta_callback
from ai_video_generator.services.llm_streaming import (
    cancel_llm_operation,
    list_llm_operations,
    resume_llm_operation,
    stream_llm_operation,
)


async def _next_event(iterator: object) -> str:
    return await iterator.__anext__()  # type: ignore[attr-defined,no-any-return]


@pytest.mark.asyncio
async def test_disconnecting_subscriber_does_not_cancel_llm_operation() -> None:
    release = asyncio.Event()

    async def operation() -> dict[str, bool]:
        callback = llm_delta_callback.get()
        assert callback is not None
        callback("partial")
        await release.wait()
        callback(" result")
        return {"saved": True}

    response = stream_llm_operation(
        operation,
        project_id="project",
        kind="project_agent",
        scope="initialize_outline",
        operation_id="detached-operation",
    )
    iterator = response.body_iterator
    assert "event: operation" in await _next_event(iterator)
    await iterator.aclose()  # type: ignore[attr-defined]

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    snapshot = next(
        item
        for item in list_llm_operations("project")
        if item["operation_id"] == "detached-operation"
    )
    assert snapshot["state"] == "succeeded"
    assert snapshot["text"] == "partial result"

    resumed = resume_llm_operation("detached-operation", "project")
    chunks = []
    async for chunk in resumed.body_iterator:
        chunks.append(chunk)
    result_block = next(
        block for block in "".join(chunks).split("\n\n") if "event: result" in block
    )
    result = json.loads(
        next(line[6:] for line in result_block.splitlines() if line.startswith("data: "))
    )
    assert result == {"saved": True}


@pytest.mark.asyncio
async def test_user_can_cancel_running_llm_operation() -> None:
    started = asyncio.Event()

    async def operation() -> None:
        started.set()
        await asyncio.Event().wait()

    stream_llm_operation(
        operation,
        project_id="cancel-project",
        kind="project_agent",
        scope="refine_outline",
        operation_id="cancel-operation",
    )
    await started.wait()

    snapshot = await cancel_llm_operation("cancel-operation", "cancel-project")

    assert snapshot["state"] == "failed"
    assert snapshot["error"] == {"status": 499, "detail": "用户已停止生成"}
    assert list_llm_operations("cancel-project", active_only=True) == ()
