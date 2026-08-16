import json
from pathlib import Path

import httpx
import pytest

from ai_video_generator.workers import (
    MOTION_CONTEXT_NODE_TYPES,
    MOTION_CONTEXT_REPOSITORY,
    PINNED_MOTION_CONTEXT_COMMIT,
    ComfyUIAdapter,
)
from ai_video_generator.workers.h3_policy import PINNED_H3_TURBO_COMMIT


def build_fake_comfyui(root: Path) -> None:
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (root / ".git" / "refs" / "heads" / "master").write_text(
        "a" * 40 + "\n",
        encoding="utf-8",
    )
    (root / "comfyui_version.py").write_text('__version__ = "0.30.0"\n', encoding="utf-8")
    motion = root / "custom_nodes" / "ComfyUI-H3-Motion-Context"
    (motion / ".git").mkdir(parents=True)
    (motion / "__init__.py").write_text("from .nodes import *\n", encoding="utf-8")
    (motion / "nodes.py").write_text(" ".join(MOTION_CONTEXT_NODE_TYPES), encoding="utf-8")
    (motion / "probe_node.py").write_text("", encoding="utf-8")
    (motion / ".git" / "HEAD").write_text(PINNED_MOTION_CONTEXT_COMMIT, encoding="utf-8")
    (motion / ".git" / "config").write_text(
        f'[remote "origin"]\n    url = {MOTION_CONTEXT_REPOSITORY}\n',
        encoding="utf-8",
    )
    (root / "models" / "text_encoders").mkdir(parents=True)
    (root / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3.safetensors").touch()


def test_local_inventory_does_not_invoke_git(tmp_path: Path) -> None:
    build_fake_comfyui(tmp_path)
    adapter = ComfyUIAdapter(tmp_path, "http://127.0.0.1:8188")

    inventory = adapter.inspect_local()

    assert inventory.root_exists
    assert inventory.version == "0.30.0"
    assert inventory.commit == "a" * 40
    assert inventory.motion_context_verified
    assert inventory.motion_context_commit == PINNED_MOTION_CONTEXT_COMMIT
    assert inventory.h3_model_files == ("text_encoders/qwen3vl_32b_minimax_h3.safetensors",)


def test_local_inventory_verifies_pinned_official_turbo_package(tmp_path: Path) -> None:
    build_fake_comfyui(tmp_path)
    package = tmp_path / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo-official"
    (package / ".git").mkdir(parents=True)
    (package / "__init__.py").write_text(
        "MiniMaxH3TurboLoRA MiniMaxH3TurboSampler",
        encoding="utf-8",
    )
    (package / ".git" / "HEAD").write_text(PINNED_H3_TURBO_COMMIT, encoding="utf-8")
    (package / ".git" / "config").write_text(
        '[remote "origin"]\n    url = https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git\n',
        encoding="utf-8",
    )

    inventory = ComfyUIAdapter(tmp_path, "http://test").inspect_local()

    assert inventory.official_h3_turbo_verified
    assert inventory.official_h3_turbo_commit == PINNED_H3_TURBO_COMMIT
    assert not inventory.unverified_h3_turbo_nodes


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
                json={
                    **{
                        node_type: {"python_module": "custom_nodes.ComfyUI-H3-Motion-Context"}
                        for node_type in MOTION_CONTEXT_NODE_TYPES
                    },
                    "MiniMaxH3ReferenceToVideo": {},
                    "MiniMaxH3TurboLoRA": {
                        "python_module": "custom_nodes.ComfyUI-MiniMax-H3-Turbo-official"
                    },
                    "MiniMaxH3TurboSampler": {
                        "python_module": "custom_nodes.ComfyUI-MiniMax-H3-Turbo-official"
                    },
                    "SaveImage": {},
                },
            )
        return httpx.Response(404, content=json.dumps({"error": "not found"}))

    adapter = ComfyUIAdapter(
        tmp_path,
        "http://test",
        transport=httpx.MockTransport(handler),
    )
    capabilities = await adapter.capabilities()

    assert capabilities.server_online
    assert capabilities.h3_node_ids == tuple(
        sorted(
            (
                *MOTION_CONTEXT_NODE_TYPES,
                "MiniMaxH3ReferenceToVideo",
                "MiniMaxH3TurboLoRA",
                "MiniMaxH3TurboSampler",
            )
        )
    )
    assert capabilities.h3_turbo_provider_modules == (
        "custom_nodes.ComfyUI-MiniMax-H3-Turbo-official",
    )
    assert capabilities.motion_context_provider_modules == (
        "custom_nodes.ComfyUI-H3-Motion-Context",
    )
    assert capabilities.motion_context_runtime_verified
    assert requested == [("GET", "/system_stats"), ("GET", "/object_info")]


@pytest.mark.asyncio
async def test_motion_context_on_disk_is_not_runtime_capability_before_restart(
    tmp_path: Path,
) -> None:
    build_fake_comfyui(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "nt"}})
        if request.url.path == "/object_info":
            return httpx.Response(200, json={"MiniMaxH3ReferenceToVideo": {}})
        return httpx.Response(404)

    capabilities = await ComfyUIAdapter(
        tmp_path,
        "http://test",
        transport=httpx.MockTransport(handler),
    ).capabilities()

    assert capabilities.inventory.motion_context_verified
    assert not capabilities.motion_context_runtime_verified
    assert any("present on disk" in warning for warning in capabilities.warnings)


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
