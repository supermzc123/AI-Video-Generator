import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from platformdirs import user_data_path
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_video_generator.credential_store import load_secret, store_secret


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AIVIDEO_",
        extra="ignore",
    )

    app_name: str = "AI Video Generator"
    environment: str = "development"
    comfyui_root: Path | None = None
    comfyui_base_url: str = "http://127.0.0.1:8188"
    request_timeout_seconds: float = Field(default=3.0, gt=0, le=60)
    data_root: Path = Field(
        default_factory=lambda: user_data_path("AI Video Generator", "supermzc123") / "data"
    )
    api_port: int = Field(default=8000, ge=1, le=65535)
    instance_nonce: str = "development"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_first_token_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_stream_idle_timeout_seconds: float = Field(default=600.0, gt=0, le=3600)
    llm_video_capable: bool = False
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    worker_auth_token: SecretStr | None = None
    remote_lease_seconds: int = Field(default=60, ge=15, le=3600)
    max_artifact_upload_bytes: int = Field(default=20 * 1024**3, ge=1)
    network_proxy: str | None = None
    h3_diffusion_model: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    h3_text_encoder: str = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    h3_video_vae: str = "minimax_h3_video_vae_fp16.safetensors"
    h3_audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"
    h3_turbo_lora: str = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
    h3_turbo_enabled: bool = True
    # Legacy persisted key retained so existing installations keep their toggle value.
    h3_sage_attention_enabled: bool = False
    h3_low_vram: bool = True
    h3_steps: int = Field(default=6, ge=4, le=50)
    h3_conditioning_workflow_template_id: str | None = None
    h3_conditioning_workflow_revision: int | None = Field(default=None, ge=1)
    h3_diffusion_workflow_template_id: str | None = None
    h3_diffusion_workflow_revision: int | None = Field(default=None, ge=1)

    @field_validator("comfyui_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("ComfyUI base URL must use http:// or https://")
        return value

    @field_validator("llm_base_url")
    @classmethod
    def normalize_optional_llm_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("LLM base URL must use http:// or https://")
        return value

    @field_validator("network_proxy")
    @classmethod
    def normalize_optional_proxy(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if value.lower().startswith("mixed:"):
            value = f"http://127.0.0.1:{value.split(':', 1)[1]}"
        if not value.startswith(("http://", "https://", "socks5://")):
            raise ValueError("network proxy must use http://, https://, socks5://, or mixed:PORT")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


RUNTIME_SETTING_FIELDS = (
    "comfyui_root",
    "comfyui_base_url",
    "request_timeout_seconds",
    "llm_base_url",
    "llm_model",
    "llm_api_key",
    "llm_timeout_seconds",
    "llm_first_token_timeout_seconds",
    "llm_stream_idle_timeout_seconds",
    "llm_video_capable",
    "network_proxy",
    "h3_diffusion_model",
    "h3_text_encoder",
    "h3_video_vae",
    "h3_audio_vae",
    "h3_turbo_lora",
    "h3_turbo_enabled",
    "h3_sage_attention_enabled",
    "h3_low_vram",
    "h3_steps",
    "h3_conditioning_workflow_template_id",
    "h3_conditioning_workflow_revision",
    "h3_diffusion_workflow_template_id",
    "h3_diffusion_workflow_revision",
)


def runtime_settings_path(settings: Settings) -> Path:
    return Path(settings.data_root) / "runtime-settings.json"


def load_runtime_settings(settings: Settings) -> Settings:
    path = runtime_settings_path(settings)
    if not path.is_file():
        return settings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    if not isinstance(payload, dict):
        return settings
    values = settings.model_dump()
    values.update({key: payload[key] for key in RUNTIME_SETTING_FIELDS if key in payload})
    plaintext = payload.get("llm_api_key")
    if isinstance(plaintext, str) and plaintext:
        store_secret(plaintext)
        payload.pop("llm_api_key", None)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if settings.llm_api_key is None:
        stored = load_secret()
        if stored:
            values["llm_api_key"] = stored
    return Settings(_env_file=None, **values)


def save_runtime_settings(settings: Settings) -> None:
    path = runtime_settings_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for field in RUNTIME_SETTING_FIELDS:
        value = getattr(settings, field)
        if field == "llm_api_key":
            if isinstance(value, SecretStr) and value.get_secret_value():
                store_secret(value.get_secret_value())
            continue
        if isinstance(value, Path):
            value = str(value)
        payload[field] = value
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)
