from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @field_validator("comfyui_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("ComfyUI base URL must use http:// or https://")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
