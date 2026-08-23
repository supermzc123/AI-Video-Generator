import hashlib
import json
from pathlib import Path

import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.api_models import ComfyNodeInstallResult, ComfyNodeInstallStep
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    ChainSpec,
    ConditioningStack,
    GenerationMode,
    GenerationSegment,
)
from ai_video_generator.llm import (
    HarnessValidationError,
    LLMHarness,
    StructuredOperationResponse,
)
from ai_video_generator.workers import PINNED_MOTION_CONTEXT_COMMIT


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
        "version": "0.2.0-beta.1",
        "build_stage": "public-beta",
        "service_id": "io.github.supermzc123.aivideogenerator.control-plane",
        "instance_nonce": "development",
    }


@pytest.mark.asyncio
async def test_dry_run_endpoint_never_returns_a_submittable_plan(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, comfyui_root=tmp_path))
    segment = GenerationSegment(
        segment_id="S001.C01",
        ordinal=1,
        prompt_revision_id="prompt-1@1",
        normalized_prompt="A static dry run",
        generation_mode=GenerationMode.REF2VA,
        width=1024,
        height=608,
        sample_frames=277,
        visible_frames=277,
    )
    chain = ChainSpec(
        project_id="project-1",
        run_id="run-1",
        shot_revision_id="shot-1@1",
        segments=(segment,),
    )
    stack = ConditioningStack(
        text_encoder_sha256="a" * 64,
        h3_model_sha256="b" * 64,
        video_vae_sha256="c" * 64,
        audio_vae_sha256="d" * 64,
        comfyui_commit="344b43989e8c56b5bb4a66cf028c834192ab59dd",
        worker_engine_commit=PINNED_MOTION_CONTEXT_COMMIT,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/plans/dry-run",
            json={
                "chain": chain.model_dump(mode="json"),
                "conditioning_stack": stack.model_dump(mode="json"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ready_for_submission"] is False
    assert body["tasks"][-1]["task_type"] == "generate_segment"


@pytest.mark.asyncio
async def test_runtime_settings_are_persisted_and_api_key_is_redacted(tmp_path: Path) -> None:
    stored_secret: dict[str, str] = {}
    import ai_video_generator.config as config_module

    original_store = config_module.store_secret
    original_load = config_module.load_secret
    config_module.store_secret = lambda value: stored_secret.__setitem__("value", value)
    config_module.load_secret = lambda: stored_secret.get("value")
    settings = Settings(_env_file=None, data_root=tmp_path)
    try:
        app = create_app(settings)
        payload = {
            "comfyui_root": "D:/ComfyUI",
            "comfyui_base_url": "http://127.0.0.1:8288",
            "request_timeout_seconds": 5,
            "llm_base_url": "https://llm.example/v1",
            "llm_model": "example-model",
            "llm_api_key": "secret-value",
            "clear_llm_api_key": False,
            "llm_timeout_seconds": 45,
            "llm_first_token_timeout_seconds": 12,
            "network_proxy": "mixed:10808",
        }

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            updated = await client.put("/api/v1/settings", json=payload)
            fetched = await client.get("/api/v1/settings")

        assert updated.status_code == 200
        assert fetched.status_code == 200
        assert fetched.json()["comfyui_base_url"] == "http://127.0.0.1:8288"
        assert fetched.json()["llm_api_key_configured"] is True
        assert fetched.json()["llm_first_token_timeout_seconds"] == 12
        assert "secret-value" not in fetched.text

        persisted = (tmp_path / "runtime-settings.json").read_text(encoding="utf-8")
        assert "secret-value" not in persisted
        assert stored_secret["value"] == "secret-value"
        restarted = create_app(settings)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted), base_url="http://test"
        ) as client:
            after_restart = await client.get("/api/v1/settings")
        assert after_restart.json()["llm_model"] == "example-model"
        assert after_restart.json()["network_proxy"] == "http://127.0.0.1:10808"
        assert after_restart.json()["llm_first_token_timeout_seconds"] == 12
    finally:
        config_module.store_secret = original_store
        config_module.load_secret = original_load


@pytest.mark.asyncio
async def test_controlled_h3_settings_allow_disabling_optional_turbo(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        current = (await client.get("/api/v1/settings")).json()
        current["h3_turbo_enabled"] = False
        current["clear_llm_api_key"] = False
        current.pop("llm_api_key_configured")
        response = await client.put("/api/v1/settings", json=current)

    assert response.status_code == 200
    assert response.json()["h3_turbo_enabled"] is False


@pytest.mark.asyncio
async def test_comfy_node_install_endpoint_uses_saved_root_and_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ai_video_generator.api as api_module

    comfyui_root = tmp_path / "ComfyUI"
    captured: dict[str, object] = {}

    async def install(**kwargs: object) -> ComfyNodeInstallResult:
        captured.update(kwargs)
        return ComfyNodeInstallResult(
            succeeded=True,
            restart_required=True,
            comfyui_root=str(comfyui_root),
            steps=(
                ComfyNodeInstallStep(
                    component="h3_core",
                    label="H3",
                    succeeded=True,
                    message="verified",
                ),
            ),
        )

    monkeypatch.setattr(api_module, "install_required_comfy_nodes", install)
    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path / "data",
            comfyui_root=comfyui_root,
            network_proxy="mixed:10808",
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/settings/comfyui-nodes/install", json={})

    assert response.status_code == 200
    assert response.json()["restart_required"] is True
    assert captured["comfyui_root"] == comfyui_root
    assert captured["proxy"] == "http://127.0.0.1:10808"


@pytest.mark.asyncio
async def test_comfy_node_install_endpoint_requires_saved_root(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path, comfyui_root=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/settings/comfyui-nodes/install", json={})

    assert response.status_code == 422
    assert "ComfyUI" in response.json()["detail"]


@pytest.mark.asyncio
async def test_workflow_llm_mapping_validation_error_is_not_reported_as_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def reject_mapping(*_args: object, **_kwargs: object) -> object:
        raise HarnessValidationError("invalid workflow mapping", ("invented node 999",))

    monkeypatch.setattr(LLMHarness, "map_workflow", reject_mapping)
    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path,
            llm_base_url="https://llm.example/v1",
            llm_model="test-model",
        )
    )
    payload = {
        "operation_id": "map:test",
        "workflow_id": "test",
        "project": {
            "project_id": "project-1",
            "revision": 1,
            "name": "Test",
            "width": 640,
            "height": 352,
            "fps": 24,
            "target_duration_seconds": 10,
            "audio_policy": "h3_native",
        },
        "raw_workflow": {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "prompt"},
            }
        },
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/llm/workflows/map", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["validation_errors"] == ["invented node 999"]


@pytest.mark.asyncio
async def test_project_agent_compacts_audit_for_workspace_larger_than_memory_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def propose_patch(_self: object, operation: object, **kwargs: object) -> object:
        captured["operation"] = operation
        captured["asset_image_urls"] = kwargs.get("asset_image_urls")
        return StructuredOperationResponse(
            operation_id="large-workspace",
            rationale="无需修改。",
        )

    monkeypatch.setattr(LLMHarness, "propose_patch", propose_patch)
    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path,
            llm_base_url="https://llm.example/v1",
            llm_model="test-model",
        )
    )
    project = {
        "project_id": "project-large",
        "revision": 1,
        "name": "Large project",
        "width": 640,
        "height": 352,
        "fps": 24,
        "target_duration_seconds": 10,
        "audio_policy": "h3_native",
    }
    workspace = {
        "revision": 1,
        "idea": "x" * 1_100_000,
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        committed = await client.post(
            "/api/v1/projects/project-large/commit",
            json={"project": project, "payload": workspace},
        )
        legacy_memory = await client.post(
            "/api/v1/projects/project-large/memory",
            json={
                "event_id": "legacy-large-input",
                "project_id": "project-large",
                "kind": "tool",
                "source": "project_agent",
                "role": "project_agent_input",
                "content": "legacy-snapshot-marker" + ("y" * 500_000),
                "created_at": "2026-08-15T00:00:00Z",
            },
        )
        response = await client.post(
            "/api/v1/projects/project-large/agent/operate",
            json={
                "operation_id": "large-workspace",
                "operation": "refine_idea",
                "instruction": "保留原意。",
                "allowed_paths": ["/idea"],
                "locked_paths": [],
                "commit": False,
            },
        )
        memory_response = await client.get(
            "/api/v1/projects/project-large/memory", params={"limit": 10}
        )

    assert committed.status_code == 201
    assert legacy_memory.status_code == 201
    assert response.status_code == 200
    assert memory_response.status_code == 200
    operation = captured["operation"]
    assert len(operation.source_document["idea"]) == 1_100_000  # type: ignore[attr-defined]
    assert "legacy-snapshot-marker" not in operation.instruction  # type: ignore[attr-defined]
    assert "legacy_payload_omitted" in operation.instruction  # type: ignore[attr-defined]
    canonical = json.dumps(
        operation.model_dump(mode="json"),  # type: ignore[attr-defined]
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert response.json()["input_sha256"] == hashlib.sha256(canonical).hexdigest()

    input_event = next(
        event for event in memory_response.json() if event["role"] == "project_agent_input"
    )
    audit = json.loads(input_event["content"])
    assert len(input_event["content"]) < 10_000
    assert audit["operation_id"] == "large-workspace"
    assert audit["project_revision"] == 1
    assert audit["workspace_revision"] == 1
    assert audit["input_sha256"] == response.json()["input_sha256"]
    assert "source_document" not in audit


@pytest.mark.asyncio
async def test_project_agent_stream_serializes_proposal_as_an_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def propose_patch(_self: object, operation: object, **_kwargs: object) -> object:
        return StructuredOperationResponse(
            operation_id=operation.operation_id,  # type: ignore[attr-defined]
            patches=(),
            rationale="大纲无需修改。",
            warnings=(),
        )

    monkeypatch.setattr(LLMHarness, "propose_patch", propose_patch)
    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path,
            llm_base_url="https://llm.example/v1",
            llm_model="test-model",
        )
    )
    project = {
        "project_id": "stream-project",
        "revision": 1,
        "name": "Stream project",
        "width": 640,
        "height": 352,
        "fps": 24,
        "target_duration_seconds": 10,
        "audio_policy": "h3_native",
    }
    workspace = {"revision": 1, "outline": [], "shots": [], "assetPlans": [], "assets": []}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        committed = await client.post(
            "/api/v1/projects/stream-project/commit",
            json={"project": project, "payload": workspace},
        )
        assert committed.status_code == 201
        response = await client.post(
            "/api/v1/projects/stream-project/agent/operate/stream",
            json={
                "operation_id": "stream-outline",
                "operation": "initialize_outline",
                "instruction": "生成大纲。",
                "display_instruction": "自动生成故事大纲初稿",
                "allowed_paths": ["/outline"],
                "locked_paths": [],
                "commit": False,
            },
        )
        memory = await client.get("/api/v1/projects/stream-project/memory")

    assert response.status_code == 200
    result_line = next(
        line.removeprefix("data: ")
        for block in response.text.split("\n\n")
        if block.startswith("event: result")
        for line in block.splitlines()
        if line.startswith("data: ")
    )
    result = json.loads(result_line)
    assert isinstance(result["proposal"], dict)
    assert result["proposal"]["patches"] == []
    assert result["proposal"]["warnings"] == []
    dialog = next(
        event
        for event in memory.json()
        if event["role"] == "dialog_initialize_outline_user"
    )
    assert dialog["content"] == "自动生成故事大纲初稿"
