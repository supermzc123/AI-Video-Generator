from fastapi import FastAPI

from ai_video_generator import __version__
from ai_video_generator.config import Settings, get_settings
from ai_video_generator.workers import ComfyUIAdapter, ComfyUICapabilities


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="Local-first MiniMax H3 control plane",
    )

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "build_stage": "p0-foundation",
        }

    @app.get("/api/v1/workers/local/capabilities")
    async def local_worker_capabilities() -> ComfyUICapabilities:
        adapter = ComfyUIAdapter(
            root=resolved_settings.comfyui_root,
            base_url=resolved_settings.comfyui_base_url,
            timeout_seconds=resolved_settings.request_timeout_seconds,
        )
        return await adapter.capabilities()

    return app


app = create_app()
