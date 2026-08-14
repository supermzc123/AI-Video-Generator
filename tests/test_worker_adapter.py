import json
from pathlib import Path

import httpx
import pytest

from ai_video_generator.workers import ComfyUIAdapter


def build_fake_comfyui(root: Path) -> None:
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (root / ".git" / "refs" / "heads" / "master").write_text(
        "a" * 40 + "\n",
        encoding="utf-8",
    )
    (root / "comfyui_version.py").write_text('__version__ = "0.30.0"\n', encoding="utf-8")
    (root / "custom_nodes" / "ComfyUI-MiniMax-H3-Motion-Director").mkdir(parents=True)
    (root / "models" / "text_encoders").mkdir(parents=True)
    (root / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3.safetensors").touch()


def test_local_inventory_does_not_invoke_git(tmp_path: Path) -> None:
    build_fake_comfyui(tmp_path)
    adapter = ComfyUIAdapter(tmp_path, "http://127.0.0.1:8188")

    inventory = adapter.inspect_local()

    assert inventory.root_exists
    assert inventory.version == "0.30.0"
    assert inventory.commit == "a" * 40
    assert inventory.motion_director_installed
    assert inventory.h3_model_files == (
        "text_encoders/qwen3vl_32b_minimax_h3.safetensors",
    )


@pytest.mark.asyncio
async def test_capabilities_only_uses_read_endpoints(tmp_path: Path) -> None:
    build_fake_comfyui(tmp_path)
    requested: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append((request.method, request.url.path))
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "nt"}})
        if request.url.path == "/object_info":
            return httpx.Response(
                200,
                json={"MiniMaxH3ReferenceToVideo": {}, "SaveImage": {}},
            )
        return httpx.Response(404, content=json.dumps({"error": "not found"}))

    adapter = ComfyUIAdapter(
        tmp_path,
        "http://test",
        transport=httpx.MockTransport(handler),
    )
    capabilities = await adapter.capabilities()

    assert capabilities.server_online
    assert capabilities.h3_node_ids == ("MiniMaxH3ReferenceToVideo",)
    assert requested == [("GET", "/system_stats"), ("GET", "/object_info")]


@pytest.mark.asyncio
async def test_offline_worker_returns_structured_warning(tmp_path: Path) -> None:
    adapter = ComfyUIAdapter(
        tmp_path / "missing",
        "http://test",
        transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
    )

    capabilities = await adapter.capabilities()

    assert not capabilities.server_online
    assert any("root does not exist" in warning for warning in capabilities.warnings)
    assert any("server is unavailable" in warning for warning in capabilities.warnings)
