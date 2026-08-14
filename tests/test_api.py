import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings


@pytest.mark.asyncio
async def test_health_endpoint_has_no_worker_side_effects() -> None:
    app = create_app(Settings(_env_file=None, comfyui_root=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "build_stage": "p0-foundation",
    }
