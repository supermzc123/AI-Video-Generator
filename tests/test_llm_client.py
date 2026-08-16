import json

import httpx
import pytest

from ai_video_generator.llm import (
    ChatMessage,
    ImageURL,
    ImageURLContentPart,
    LLMClientError,
    OpenAICompatibleClient,
    TextContentPart,
)
from ai_video_generator.llm.client import is_retryable_llm_error, llm_delta_callback


def test_retryable_llm_errors_only_include_transient_failures() -> None:
    assert is_retryable_llm_error(
        LLMClientError("OpenAI-compatible transport failed: ConnectError")
    )
    assert is_retryable_llm_error(LLMClientError("OpenAI-compatible request timed out"))
    assert is_retryable_llm_error(
        LLMClientError("OpenAI-compatible provider rejected request (status=503)")
    )
    assert not is_retryable_llm_error(
        LLMClientError("OpenAI-compatible provider rejected request (status=400)")
    )
    assert not is_retryable_llm_error(
        LLMClientError("response does not contain choices[0].message.content")
    )


@pytest.mark.asyncio
async def test_openai_compatible_client_posts_deterministic_json_request() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"bindings":[]}'}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1/",
            model="test-model",
            api_key="secret",
            http_client=http_client,
        )
        result = await client.complete_json((ChatMessage(role="user", content="map"),))

    assert result == '{"bindings":[]}'
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["body"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "map"}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


@pytest.mark.asyncio
async def test_openai_compatible_client_rejects_missing_content() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            model="test-model",
            http_client=http_client,
        )
        with pytest.raises(LLMClientError, match="choices"):
            await client.complete_json((ChatMessage(role="user", content="map"),))


@pytest.mark.asyncio
async def test_openai_compatible_client_serializes_multimodal_message() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"seen":true}'}}]},
        )

    message = ChatMessage(
        role="user",
        content=(
            TextContentPart(text="Inspect this image."),
            ImageURLContentPart(
                image_url=ImageURL(url="data:image/jpeg;base64,AAAA", detail="low")
            ),
        ),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            model="vision-model",
            http_client=http_client,
        )
        result = await client.complete_json((message,))

    assert result == '{"seen":true}'
    assert captured["body"]["messages"][0]["content"] == [
        {"type": "text", "text": "Inspect this image."},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,AAAA",
                "detail": "low",
            },
        },
    ]


@pytest.mark.asyncio
async def test_openai_compatible_client_accepts_text_content_list_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "output_text", "text": '{"ok":'},
                                {"type": "text", "text": "true}"},
                            ]
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            model="test-model",
            http_client=http_client,
        )
        result = await client.complete_json((ChatMessage(role="user", content="map"),))

    assert result == '{"ok":true}'


@pytest.mark.asyncio
async def test_openai_compatible_client_reports_safe_provider_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_image",
                    "message": "bad secret image",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            model="test-model",
            api_key="secret",
            http_client=http_client,
        )
        with pytest.raises(LLMClientError) as error:
            await client.complete_json((ChatMessage(role="user", content="map"),))

    assert "status=400" in str(error.value)
    assert "invalid_request_error" in str(error.value)
    assert "bad_image" in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_openai_compatible_client_reports_empty_length_limited_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            model="test-model",
            http_client=http_client,
        )
        with pytest.raises(LLMClientError, match="finish_reason=length"):
            await client.complete_json((ChatMessage(role="user", content="map"),))


@pytest.mark.asyncio
async def test_openai_compatible_client_lists_unique_sorted_models() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "z-model"}, {"id": "A-model"}, {"id": "z-model"}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(
            base_url="https://llm.example/v1",
            model="unused",
            http_client=http_client,
        )
        assert await client.list_models() == ("A-model", "z-model")


@pytest.mark.asyncio
async def test_openai_compatible_client_streams_json_deltas() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"true}"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    deltas: list[str] = []
    token = llm_delta_callback.set(deltas.append)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(
                base_url="https://llm.example/v1",
                model="test-model",
                http_client=http_client,
            )
            result = await client.complete_json((ChatMessage(role="user", content="map"),))
    finally:
        llm_delta_callback.reset(token)

    assert result == '{"ok":true}'
    assert deltas == ['{"ok":', "true}"]
