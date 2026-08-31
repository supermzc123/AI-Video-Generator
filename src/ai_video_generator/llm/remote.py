from __future__ import annotations

from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ai_video_generator.config import Settings

from .client import ChatMessage, OpenAICompatibleClient


@dataclass(frozen=True)
class LLMRemoteConfig:
    """Immutable connection settings shared by every remote LLM operation."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0
    first_token_timeout_seconds: float | None = None
    stream_idle_timeout_seconds: float = 600.0
    proxy: str | None = None


def remote_config(settings: Settings, *, model: str | None = None) -> LLMRemoteConfig:
    return LLMRemoteConfig(
        base_url=settings.llm_base_url or "",
        model=model or settings.llm_model or "",
        api_key=(settings.llm_api_key.get_secret_value() if settings.llm_api_key else None),
        timeout_seconds=settings.llm_timeout_seconds,
        first_token_timeout_seconds=settings.llm_first_token_timeout_seconds,
        stream_idle_timeout_seconds=settings.llm_stream_idle_timeout_seconds,
        proxy=settings.network_proxy,
    )


def exception_messages(error: Exception) -> tuple[str, ...]:
    """Normalize library exceptions without assuming errors is data or a method."""
    errors = getattr(error, "errors", None)
    if callable(errors):
        errors = errors()
    if isinstance(errors, (list, tuple)):
        messages: list[str] = []
        for item in errors:
            if isinstance(item, dict):
                location = ".".join(str(part) for part in item.get("loc", ()))
                message = str(item.get("msg") or item)
                messages.append(f"{location}: {message}" if location else message)
            else:
                messages.append(str(item))
        if messages:
            return tuple(messages)
    return (str(error),)


def _client(config: LLMRemoteConfig) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
        first_token_timeout_seconds=config.first_token_timeout_seconds,
        stream_idle_timeout_seconds=config.stream_idle_timeout_seconds,
        proxy=config.proxy,
    )


@asynccontextmanager
async def open_client(config: LLMRemoteConfig):
    async with _client(config) as client:
        yield client


async def complete_text(config: LLMRemoteConfig, messages: Sequence[ChatMessage]) -> str:
    async with _client(config) as client:
        return await client.complete_text(messages)


async def complete_json(config: LLMRemoteConfig, messages: Sequence[ChatMessage]) -> str:
    async with _client(config) as client:
        return await client.complete_json(messages)


async def list_models(config: LLMRemoteConfig) -> tuple[str, ...]:
    async with _client(config) as client:
        return await client.list_models()
