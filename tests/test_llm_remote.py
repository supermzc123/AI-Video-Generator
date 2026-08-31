import pytest
from pydantic import BaseModel, Field, SecretStr, ValidationError

from ai_video_generator.config import Settings
from ai_video_generator.llm import ChatMessage
from ai_video_generator.llm.remote import (
    LLMRemoteConfig,
    complete_json,
    exception_messages,
    remote_config,
)


def test_remote_config_maps_all_runtime_connection_settings() -> None:
    settings = Settings(
        _env_file=None,
        llm_base_url="https://llm.example/v1",
        llm_model="model-a",
        llm_api_key=SecretStr("secret"),
        llm_timeout_seconds=45,
        llm_first_token_timeout_seconds=12,
        llm_stream_idle_timeout_seconds=720,
        network_proxy="http://127.0.0.1:7890",
    )

    assert remote_config(settings) == LLMRemoteConfig(
        base_url="https://llm.example/v1",
        model="model-a",
        api_key="secret",
        timeout_seconds=45,
        first_token_timeout_seconds=12,
        stream_idle_timeout_seconds=720,
        proxy="http://127.0.0.1:7890",
    )


@pytest.mark.asyncio
async def test_remote_completion_uses_the_shared_client(monkeypatch) -> None:
    calls: list[tuple[LLMRemoteConfig, tuple[ChatMessage, ...]]] = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def complete_json(self, messages):
            calls.append((config, tuple(messages)))
            return '{"ok":true}'

    config = LLMRemoteConfig(base_url="https://llm.example/v1", model="model-a")
    monkeypatch.setattr("ai_video_generator.llm.remote._client", lambda value: FakeClient())
    messages = (ChatMessage(role="user", content="test"),)

    assert await complete_json(config, messages) == '{"ok":true}'
    assert calls == [(config, messages)]


def test_exception_messages_calls_pydantic_errors_method() -> None:
    class Payload(BaseModel):
        duration: float = Field(ge=4)

    try:
        Payload(duration=1)
    except ValidationError as error:
        messages = exception_messages(error)
    else:
        raise AssertionError("validation should fail")

    assert len(messages) == 1
    assert messages[0].startswith("duration:")
    assert "greater than or equal to 4" in messages[0]
