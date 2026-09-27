import asyncio
import json

import httpx
import pytest
from test_h3_prompt_harness import QueueClient, library, request, valid_base_candidate
from test_llm_harness import mapping_request, valid_mapping

from ai_video_generator.llm import ChatMessage, LLMClientError, LLMHarness
from ai_video_generator.llm.budget import (
    MAX_CONTEXT_CHARACTERS,
    MAX_OUTPUT_CHARACTERS,
    llm_operation,
)
from ai_video_generator.llm.client import OpenAICompatibleClient, llm_delta_callback
from ai_video_generator.llm.h3_prompt import H3PromptHarness, H3PromptHarnessError


def response(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.mark.asyncio
async def test_transport_retry_and_business_repair_share_three_calls():
    calls = []

    async def handler(req):
        calls.append(req)
        if len(calls) in {1, 3}:
            raise httpx.ConnectError("disconnected", request=req)
        return response("not JSON")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatibleClient(base_url="http://model", model="mock", http_client=http)
        with pytest.raises(LLMClientError, match="call budget"):
            await LLMHarness(client).map_workflow(mapping_request())
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_stream_rejection_fallback_and_repair_share_budget():
    calls = []

    async def handler(req):
        calls.append(json.loads(req.content))
        if calls[-1].get("stream"):
            return httpx.Response(422, json={"error": {"message": "stream unsupported"}})
        return response("not JSON")

    token = llm_delta_callback.set(lambda _: None)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient(base_url="http://model", model="mock", http_client=http)
            with pytest.raises(LLMClientError, match="call budget"):
                await LLMHarness(client).map_workflow(mapping_request())
    finally:
        llm_delta_callback.reset(token)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_json_gateway_response_to_stream_is_collected_without_replay():
    calls = []

    async def handler(req):
        calls.append(req)
        return response(valid_mapping())

    deltas = []
    token = llm_delta_callback.set(deltas.append)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient(base_url="http://model", model="mock", http_client=http)
            await LLMHarness(client).map_workflow(mapping_request())
    finally:
        llm_delta_callback.reset(token)
    assert len(calls) == 1
    assert len(deltas) == 1


@pytest.mark.asyncio
async def test_incomplete_stream_is_not_replayed_after_content_started():
    calls = []

    async def handler(req):
        calls.append(req)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    token = llm_delta_callback.set(lambda _: None)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient(base_url="http://model", model="mock", http_client=http)
            with pytest.raises(LLMClientError, match="ended before completion"):
                await client.complete_text((ChatMessage(role="user", content="write"),))
    finally:
        llm_delta_callback.reset(token)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_one_total_deadline_bounds_repair_and_cancels_slow_provider():
    cancelled = asyncio.Event()

    async def handler(req):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatibleClient(
            base_url="http://model", model="mock", http_client=http, operation_timeout_seconds=0.01
        )
        with pytest.raises(LLMClientError, match="total deadline"):
            await client.complete_text((ChatMessage(role="user", content="write"),))
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_nested_scope_cannot_refill_budget():
    async with llm_operation(max_calls=1) as outer:
        outer.claim_call()
        async with llm_operation(max_calls=99, timeout_seconds=999) as inner:
            assert inner is outer
            with pytest.raises(LLMClientError, match="budget exhausted"):
                inner.claim_call()


@pytest.mark.asyncio
async def test_provider_concurrency_applies_across_separate_ui_clients():
    active = maximum = 0

    async def handler(req):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return response("valid text")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        clients = [
            OpenAICompatibleClient(base_url="http://model", model="mock", http_client=http)
            for _ in range(6)
        ]
        await asyncio.gather(
            *(
                client.complete_text((ChatMessage(role="user", content="write"),))
                for client in clients
            )
        )
    assert maximum == 2


@pytest.mark.asyncio
async def test_context_limit_rejects_before_network_and_output_limit_before_commit():
    calls = []

    async def handler(req):
        calls.append(req)
        return response("x" * (MAX_OUTPUT_CHARACTERS + 1))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatibleClient(base_url="http://model", model="mock", http_client=http)
        with pytest.raises(LLMClientError, match="context exceeds"):
            await client.complete_text(
                (ChatMessage(role="user", content="x" * (MAX_CONTEXT_CHARACTERS + 1)),)
            )
        assert calls == []
        with pytest.raises(LLMClientError, match="output exceeds"):
            await client.complete_text((ChatMessage(role="user", content="write"),))
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_h3_schema_repair_cannot_get_another_semantic_repair_budget():
    short = valid_base_candidate()
    short["integrated_multimodal_description"] = "[Shot 1] short"
    client = QueueClient(["invalid json", short, valid_base_candidate()])
    with pytest.raises(H3PromptHarnessError, match="one repair"):
        await H3PromptHarness(client, library()).generate(request())
    assert len(client.messages) == 2
