import httpx
import pytest

from ai_video_generator.workers import (
    ComfyUIAdapter,
    compile_conditioning_encode_workflow,
    compile_conditioning_load_workflow,
    compile_h3_static_encode_workflow,
    compile_h3_static_load_workflow,
)


def base_workflow() -> dict[str, object]:
    return {
        "1": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"prompt": "shot"}},
        "2": {"class_type": "H3Sampler", "inputs": {"positive": ["1", 0]}},
    }


def test_encode_and_diffusion_workflows_are_separate() -> None:
    fingerprint = "a" * 64
    encoded = compile_conditioning_encode_workflow(
        base_workflow(),
        conditioning_node_id="1",
        conditioning_output_index=0,
        fingerprint=fingerprint,
    )
    diffusion = compile_conditioning_load_workflow(
        base_workflow(),
        target_node_id="2",
        target_input_name="positive",
        fingerprint=fingerprint,
    )

    save_node = encoded["avg-save-conditioning"]
    load_node = diffusion["avg-load-conditioning"]
    assert save_node["inputs"]["conditioning"] == ["1", 0]
    assert diffusion["2"]["inputs"]["positive"] == ["avg-load-conditioning", 0]
    assert load_node["inputs"]["fingerprint"] == fingerprint


@pytest.mark.asyncio
async def test_free_models_uses_comfyui_free_endpoint() -> None:
    requests: list[tuple[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, request.read()))
        return httpx.Response(200, json={})

    adapter = ComfyUIAdapter(
        None,
        "http://test",
        transport=httpx.MockTransport(handler),
    )
    await adapter.free_models()

    assert requests[0][0] == "/free"
    assert b'"unload_models":true' in requests[0][1]


def h3_workflow() -> dict[str, object]:
    return {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "h3.safetensors"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "h3-vae.safetensors"}},
        "static": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["vae", 0],
                "prompt": "shot",
                "width": 512,
                "height": 288,
                "length": 22,
            },
        },
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3-unet.safetensors"}},
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["unet", 0], "conditioning": ["static", 0]},
        },
        "sampler": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"guider": ["guider", 0], "latent_image": ["static", 1]},
        },
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["sampler", 0]}},
    }


def test_h3_static_workflows_isolate_encoder_and_diffusion_models() -> None:
    fingerprint = "b" * 64
    encoded = compile_h3_static_encode_workflow(
        h3_workflow(), conditioning_node_id="static", fingerprint=fingerprint
    )
    diffusion = compile_h3_static_load_workflow(
        h3_workflow(),
        conditioning_target_node_id="guider",
        conditioning_target_input_name="conditioning",
        latent_target_node_id="sampler",
        latent_target_input_name="latent_image",
        output_node_ids=("save",),
        fingerprint=fingerprint,
    )

    assert set(encoded) == {"clip", "vae", "static", "avg-save-h3-static"}
    assert "unet" not in encoded
    assert set(diffusion) == {"unet", "guider", "sampler", "save", "avg-load-h3-static"}
    assert "clip" not in diffusion
    assert "vae" not in diffusion
    assert "static" not in diffusion
    assert diffusion["guider"]["inputs"]["conditioning"] == ["avg-load-h3-static", 0]
    assert diffusion["sampler"]["inputs"]["latent_image"] == ["avg-load-h3-static", 1]


def test_h3_static_diffusion_requires_explicit_output_nodes() -> None:
    with pytest.raises(ValueError, match="diffusion output"):
        compile_h3_static_load_workflow(
            h3_workflow(),
            conditioning_target_node_id="guider",
            conditioning_target_input_name="conditioning",
            latent_target_node_id="sampler",
            latent_target_input_name="latent_image",
            output_node_ids=(),
            fingerprint="c" * 64,
        )
