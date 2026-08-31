import httpx
import pytest

from ai_video_generator.workers import ComfyUIAdapter, compile_h3_static_encode_workflow


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


def test_h3_static_workflow_isolates_encoder_models() -> None:
    fingerprint = "b" * 64
    encoded = compile_h3_static_encode_workflow(
        h3_workflow(), conditioning_node_id="static", fingerprint=fingerprint
    )

    assert set(encoded) == {"clip", "vae", "static", "avg-save-h3-static"}
    assert "unet" not in encoded
    assert "sampler" not in encoded
    assert encoded["avg-save-h3-static"]["inputs"]["conditioning"] == ["static", 0]
    assert encoded["avg-save-h3-static"]["inputs"]["latent"] == ["static", 1]
    assert encoded["avg-save-h3-static"]["inputs"]["fingerprint"] == fingerprint


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
