from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class LLMClientError(RuntimeError):
    pass


llm_delta_callback: ContextVar[Callable[[str], None] | None] = ContextVar(
    "llm_delta_callback", default=None
)


def is_retryable_llm_error(error: LLMClientError) -> bool:
    """Return whether the same request can be retried after a transient failure."""
    message = str(error).casefold()
    if "transport failed" in message or "timed out" in message:
        return True
    if "provider rejected request" not in message:
        return False
    return any(
        f"status={status}" in message
        for status in (408, 409, 425, 429, 500, 502, 503, 504)
    )


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] = "auto"


class ImageURLContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class VideoURL(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)


class VideoURLContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["video_url"] = "video_url"
    video_url: VideoURL


ChatContentPart = Annotated[
    TextContentPart | ImageURLContentPart | VideoURLContentPart,
    Field(discriminator="type"),
]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str | tuple[ChatContentPart, ...]


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30,
        proxy: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._owns_client = http_client is None
        self._api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._http = http_client or httpx.AsyncClient(
            headers=self._headers,
            timeout=timeout_seconds,
            proxy=proxy,
        )

    async def complete_json(self, messages: Sequence[ChatMessage]) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        callback = llm_delta_callback.get()
        if callback is not None:
            return await self._complete_json_stream(payload, callback)
        return await self._complete_json_nonstream(payload)

    async def _complete_json_nonstream(self, payload: dict[str, Any]) -> str:
        try:
            response = await self._http.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers,
            )
        except httpx.TimeoutException as exc:
            raise LLMClientError("OpenAI-compatible request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(
                f"OpenAI-compatible transport failed: {type(exc).__name__}"
            ) from exc

        if response.is_error:
            raise LLMClientError(self._format_provider_error(response))

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError("OpenAI-compatible response was not valid JSON") from exc

        try:
            choice = data["choices"][0]
            content = _message_text(choice["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("response does not contain choices[0].message.content") from exc
        if not content.strip():
            finish_reason = choice.get("finish_reason")
            suffix = f" (finish_reason={finish_reason})" if finish_reason else ""
            raise LLMClientError(f"response message content was empty{suffix}")
        return content

    async def _complete_json_stream(
        self, payload: dict[str, Any], on_delta: Callable[[str], None]
    ) -> str:
        chunks: list[str] = []
        try:
            async with self._http.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json={**payload, "stream": True},
                headers=self._headers,
            ) as response:
                if response.is_error:
                    await response.aread()
                    if response.status_code in {400, 404, 422}:
                        return await self._complete_json_nonstream(payload)
                    raise LLMClientError(self._format_provider_error(response))
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                        choice = event["choices"][0]
                        content = choice.get("delta", {}).get("content")
                        text = _message_text(content) if content is not None else ""
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if text:
                        chunks.append(text)
                        on_delta(text)
        except LLMClientError:
            raise
        except httpx.TimeoutException as exc:
            raise LLMClientError("OpenAI-compatible request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(
                f"OpenAI-compatible transport failed: {type(exc).__name__}"
            ) from exc
        result = "".join(chunks)
        if not result.strip():
            # Some OpenAI-compatible gateways accept `stream: true` but return
            # a regular JSON response. Retry once without streaming so those
            # providers remain usable; the UI already received an activity
            # state and will display the validated final result.
            return await self._complete_json_nonstream(payload)
        return result

    async def list_models(self) -> tuple[str, ...]:
        try:
            response = await self._http.get(
                f"{self._base_url}/models",
                headers=self._headers,
            )
        except httpx.TimeoutException as exc:
            raise LLMClientError("OpenAI-compatible model listing timed out") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(
                f"OpenAI-compatible model listing failed: {type(exc).__name__}"
            ) from exc
        if response.is_error:
            raise LLMClientError(self._format_provider_error(response))
        try:
            data = response.json()["data"]
            model_ids = {
                item["id"]
                for item in data
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"].strip()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMClientError("model listing does not contain a valid data array") from exc
        return tuple(sorted(model_ids, key=str.casefold))

    def _format_provider_error(self, response: httpx.Response) -> str:
        error_type: object = None
        error_code: object = None
        error_message: object = None
        try:
            error = response.json().get("error", {})
            if isinstance(error, dict):
                error_type = error.get("type")
                error_code = error.get("code")
                error_message = error.get("message")
        except (ValueError, AttributeError):
            pass

        fields = [f"status={response.status_code}"]
        if error_type:
            fields.append(f"type={error_type}")
        if error_code:
            fields.append(f"code={error_code}")
        if isinstance(error_message, str) and error_message.strip():
            message = " ".join(error_message.split())[:300]
            if self._api_key:
                message = message.replace(self._api_key, "[REDACTED]")
            fields.append(f"message={message}")
        return f"OpenAI-compatible provider rejected request ({', '.join(fields)})"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
            and isinstance(item.get("text"), str)
        ]
        return "".join(text_parts)
    raise TypeError("message content must be a string or text content list")
