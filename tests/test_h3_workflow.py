import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    H3AccelerationMode,
    H3AttentionMode,
    H3InputTarget,
    H3TurboProfile,
    WorkflowApproval,
)
from ai_video_generator.workers import (
    build_h3_profile,
    compile_h3_workflow,
    inspect_h3_workflow_profile,
)
from ai_video_generator.workers.workflow import WorkflowContractError


def object_info() -> dict[str, object]:
    return {
        "UNETLoader": {},
        "MiniMaxH3TurboLoRA": {
            "python_module": "custom_nodes.ComfyUI-MiniMax-H3-Turbo-official",
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "lora_name": [["turbo.safetensors"]],
                    "strength": ["FLOAT"],
                    "low_vram": ["BOOLEAN"],
                }
            },
        },
        "MiniMaxH3TurboSampler": {
            "python_module": "custom_nodes.ComfyUI-MiniMax-H3-Turbo-official",
            "input": {"required": {}},
        },
        "BasicScheduler": {
            "python_module": "comfy_extras.nodes_custom_sampler",
            "input": {"required": {"model": ["MODEL"]}},
        },
        "MiniMaxH3MemoryEfficientSageAttentionPatch": {
            "python_module": "custom_nodes.comfyui-kjnodes",
            "input": {"required": {"model": ["MODEL"]}},
        },
        "BasicGuider": {},
        "SamplerCustomAdvanced": {},
    }


def turbo_workflow() -> dict[str, dict[str, object]]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}},
        "2": {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "old.safetensors",
                "strength": 0.8,
                "low_vram": False,
            },
        },
        "3": {"class_type": "MiniMaxH3TurboSampler", "inputs": {}},
        "4": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["5", 0], "scheduler": "normal", "steps": 20},
        },
        "5": {
            "class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
            "inputs": {"model": ["2", 0]},
        },
        "7": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["5", 0], "conditioning": ["1", 0]},
        },
        "6": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"sampler": ["3", 0], "sigmas": ["4", 0], "guider": ["7", 0]},
        },
    }


def turbo_settings(steps: int = 6) -> H3TurboProfile:
    return H3TurboProfile(
        lora_loader_node_id="2",
        sampler_node_id="3",
        scheduler_node_id="4",
        lora_name_target=H3InputTarget(node_id="2", input_name="lora_name"),
        strength_target=H3InputTarget(node_id="2", input_name="strength"),
        low_vram_target=H3InputTarget(node_id="2", input_name="low_vram"),
        steps_target=H3InputTarget(node_id="4", input_name="steps"),
        scheduler_name_target=H3InputTarget(node_id="4", input_name="scheduler"),
        steps=steps,
    )


def test_external_h3_turbo_profile_compiles_only_declared_inputs() -> None:
    profile = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=turbo_workflow(),
        object_info=object_info(),
        acceleration=H3AccelerationMode.TURBO,
        attention=H3AttentionMode.SAGE,
        turbo=turbo_settings(),
        sage_attention_node_id="5",
    )

    inspection = inspect_h3_workflow_profile(profile, object_info())
    compiled = compile_h3_workflow(profile)

    assert inspection.compatible
    assert compiled["2"]["inputs"]["lora_name"].endswith("v4_step600_ema.safetensors")
    assert compiled["2"]["inputs"]["strength"] == 1.0
    assert compiled["4"]["inputs"]["steps"] == 6
    assert compiled["4"]["inputs"]["scheduler"] == "simple"
    assert compiled["6"] == turbo_workflow()["6"]


def test_turbo_rejects_teacache_and_warns_on_four_step_motion_risk() -> None:
    workflow = turbo_workflow()
    workflow["7"] = {
        "class_type": "TeaCache",
        "inputs": {"model": ["2", 0]},
    }
    info = {**object_info(), "TeaCache": {}}
    profile = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=workflow,
        object_info=info,
        acceleration=H3AccelerationMode.TURBO,
        turbo=turbo_settings(steps=4),
    )

    inspection = inspect_h3_workflow_profile(profile, info)

    assert not inspection.compatible
    assert any("TeaCache" in issue for issue in inspection.issues)
    assert any("fast motion" in warning for warning in inspection.warnings)


def test_standard_profile_cannot_compile_teacache() -> None:
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "TeaCacheForH3", "inputs": {"model": ["1", 0]}},
    }
    info = {"UNETLoader": {}, "TeaCacheForH3": {}}
    profile = build_h3_profile(
        profile_id="h3-standard",
        name="H3 standard",
        raw_workflow=workflow,
        object_info=info,
    )

    with pytest.raises(WorkflowContractError, match="TeaCache"):
        compile_h3_workflow(profile)


def test_same_named_experimental_turbo_provider_is_rejected() -> None:
    info = object_info()
    info["MiniMaxH3TurboLoRA"]["python_module"] = "custom_nodes.comfyui-minimax-h3-turbo"
    info["MiniMaxH3TurboSampler"]["python_module"] = "custom_nodes.comfyui-minimax-h3-turbo"
    profile = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=turbo_workflow(),
        object_info=info,
        acceleration=H3AccelerationMode.TURBO,
        attention=H3AttentionMode.SAGE,
        turbo=turbo_settings(),
        sage_attention_node_id="5",
    )

    inspection = inspect_h3_workflow_profile(profile, info)

    assert not inspection.compatible
    assert sum("pinned official plugin" in issue for issue in inspection.issues) == 2


def test_turbo_profile_records_and_validates_official_source_revision() -> None:
    turbo = turbo_settings().model_copy(update={"source_commit": "0" * 40})
    profile = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=turbo_workflow(),
        object_info=object_info(),
        acceleration=H3AccelerationMode.TURBO,
        attention=H3AttentionMode.SAGE,
        turbo=turbo,
        sage_attention_node_id="5",
    )

    inspection = inspect_h3_workflow_profile(profile, object_info())

    assert not inspection.compatible
    assert any("source commit" in issue for issue in inspection.issues)


@pytest.mark.asyncio
async def test_h3_profile_api_inspects_and_compiles_without_submitting_gpu(tmp_path) -> None:
    info = object_info()
    profile = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=turbo_workflow(),
        object_info=info,
        acceleration=H3AccelerationMode.TURBO,
        attention=H3AttentionMode.SAGE,
        turbo=turbo_settings(),
        sage_attention_node_id="5",
    )
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        inspected = await client.post(
            "/api/v1/h3/workflows/inspect",
            json={"profile": profile.model_dump(mode="json"), "object_info": info},
        )
        compiled = await client.post(
            "/api/v1/h3/workflows/compile",
            json={
                "profile": profile.model_dump(mode="json"),
                "object_info": info,
            },
        )

    assert inspected.status_code == 200
    assert inspected.json()["compatible"] is True
    assert compiled.status_code == 200
    assert compiled.json()["4"]["inputs"]["steps"] == 6


@pytest.mark.asyncio
async def test_h3_profile_registration_uses_current_schema_and_keeps_revisions(
    tmp_path,
) -> None:
    current_info = object_info()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info"
        return httpx.Response(200, json=current_info)

    profile_v1 = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=turbo_workflow(),
        object_info=current_info,
        acceleration=H3AccelerationMode.TURBO,
        attention=H3AttentionMode.SAGE,
        turbo=turbo_settings(),
        sage_attention_node_id="5",
        approval=WorkflowApproval.APPROVED,
    )
    app = create_app(
        Settings(_env_file=None, data_root=tmp_path),
        comfyui_transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        registered_v1 = await client.post(
            "/api/v1/h3/workflows/profiles",
            json=profile_v1.model_dump(mode="json"),
        )

        # A Worker schema change is part of the profile fingerprint and therefore
        # must be registered as a separate immutable revision.
        current_info["BasicScheduler"]["display_name"] = "Basic Scheduler"
        profile_v2 = build_h3_profile(
            profile_id="h3-turbo",
            revision=2,
            name="H3 Turbo",
            raw_workflow=turbo_workflow(),
            object_info=current_info,
            acceleration=H3AccelerationMode.TURBO,
            attention=H3AttentionMode.SAGE,
            turbo=turbo_settings(),
            sage_attention_node_id="5",
            approval=WorkflowApproval.APPROVED,
        )
        registered_v2 = await client.post(
            "/api/v1/h3/workflows/profiles",
            json=profile_v2.model_dump(mode="json"),
        )
        listed = await client.get(
            "/api/v1/h3/workflows/profiles", params={"profile_id": "h3-turbo"}
        )
        fetched_v1 = await client.get("/api/v1/h3/workflows/profiles/h3-turbo/1")

    assert registered_v1.status_code == 201
    assert registered_v2.status_code == 201
    assert [item["revision"] for item in listed.json()] == [1, 2]
    assert fetched_v1.status_code == 200
    assert fetched_v1.json()["node_schema_sha256"] == profile_v1.node_schema_sha256


@pytest.mark.asyncio
async def test_h3_profile_registration_requires_approval_and_current_schema(
    tmp_path,
) -> None:
    current_info = object_info()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=current_info)

    draft = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=turbo_workflow(),
        object_info=current_info,
        acceleration=H3AccelerationMode.TURBO,
        attention=H3AttentionMode.SAGE,
        turbo=turbo_settings(),
        sage_attention_node_id="5",
    )
    stale_approved = draft.model_copy(update={"approval": WorkflowApproval.APPROVED})
    current_info["BasicScheduler"]["display_name"] = "changed after inspection"
    app = create_app(
        Settings(_env_file=None, data_root=tmp_path),
        comfyui_transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        draft_response = await client.post(
            "/api/v1/h3/workflows/profiles",
            json=draft.model_dump(mode="json"),
        )
        stale_response = await client.post(
            "/api/v1/h3/workflows/profiles",
            json=stale_approved.model_dump(mode="json"),
        )

    assert draft_response.status_code == 409
    assert stale_response.status_code == 409
    assert "node schema SHA-256" in str(stale_response.json())


@pytest.mark.asyncio
async def test_h3_profile_revision_is_immutable(tmp_path) -> None:
    info = object_info()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=info)

    profile = build_h3_profile(
        profile_id="h3-turbo",
        name="H3 Turbo",
        raw_workflow=turbo_workflow(),
        object_info=info,
        acceleration=H3AccelerationMode.TURBO,
        attention=H3AttentionMode.SAGE,
        turbo=turbo_settings(),
        sage_attention_node_id="5",
        approval=WorkflowApproval.APPROVED,
    )
    changed = profile.model_copy(update={"name": "Changed in place"})
    app = create_app(
        Settings(_env_file=None, data_root=tmp_path),
        comfyui_transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post(
            "/api/v1/h3/workflows/profiles",
            json=profile.model_dump(mode="json"),
        )
        conflict = await client.post(
            "/api/v1/h3/workflows/profiles",
            json=changed.model_dump(mode="json"),
        )

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert "immutable" in conflict.json()["detail"]
