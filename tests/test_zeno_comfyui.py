import httpx
import pytest

from ai_video_generator.workers import ComfyUIAdapter


@pytest.mark.asyncio
async def test_object_info_and_prompt_lifecycle_are_typed() -> None:
    requests: list[tuple[str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        requests.append((request.method, request.url.path, body))
        if request.url.path == "/object_info/CLIPTextEncode":
            return httpx.Response(
                200,
                json={
                    "CLIPTextEncode": {
                        "input": {"required": {"text": ["STRING", {}]}},
                        "output": ["CONDITIONING"],
                    }
                },
            )
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "prompt-1", "number": 7})
        if request.url.path == "/history/prompt-1":
            return httpx.Response(200, json={"prompt-1": {"outputs": {}}})
        if request.url.path == "/view":
            return httpx.Response(200, content=b"image-bytes")
        if request.url.path == "/queue" and request.method == "GET":
            return httpx.Response(
                200,
                json={"queue_running": [], "queue_pending": [[8, "prompt-1", {}, {}, []]]},
            )
        if request.url.path == "/queue" and request.method == "POST":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    adapter = ComfyUIAdapter(
        None,
        "http://test",
        transport=httpx.MockTransport(handler),
    )
    info = await adapter.get_object_info("CLIPTextEncode")
    submission = await adapter.submit_prompt(
        {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}},
        client_id="client-1",
    )
    history = await adapter.get_history(submission.prompt_id)
    output = await adapter.get_output_image("result.png", subfolder="project")
    cancelled = await adapter.cancel_prompt(submission.prompt_id)

    assert len(info.node_schema_sha256) == 64
    assert tuple(info.nodes) == ("CLIPTextEncode",)
    assert history == {"outputs": {}}
    assert output == b"image-bytes"
    assert cancelled.was_pending and cancelled.deleted
    assert not cancelled.interrupted
    assert [(method, path) for method, path, _body in requests] == [
        ("GET", "/object_info/CLIPTextEncode"),
        ("POST", "/prompt"),
        ("GET", "/history/prompt-1"),
        ("GET", "/view"),
        ("GET", "/queue"),
        ("POST", "/queue"),
    ]


@pytest.mark.asyncio
async def test_cancel_only_interrupts_when_target_is_running() -> None:
    requested_paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append((request.method, request.url.path))
        if request.url.path == "/queue":
            return httpx.Response(
                200,
                json={"queue_running": [[1, "target", {}, {}, []]], "queue_pending": []},
            )
        if request.url.path == "/interrupt":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    adapter = ComfyUIAdapter(None, "http://test", transport=httpx.MockTransport(handler))
    result = await adapter.cancel_prompt("target")

    assert result.was_running and result.interrupted
    assert requested_paths == [("GET", "/queue"), ("POST", "/interrupt")]


@pytest.mark.asyncio
async def test_output_collection_retries_transient_transport_failure() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("temporary output read timeout", request=request)
        return httpx.Response(200, content=b"completed-video")

    adapter = ComfyUIAdapter(None, "http://test", transport=httpx.MockTransport(handler))

    output = await adapter.get_output_image("completed.mp4", subfolder="project")

    assert output == b"completed-video"
    assert attempts == 2


@pytest.mark.asyncio
async def test_history_poll_retries_transient_read_timeout() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("ComfyUI is busy", request=request)
        return httpx.Response(200, json={"prompt-1": {"outputs": {}}})

    adapter = ComfyUIAdapter(None, "http://test", transport=httpx.MockTransport(handler))

    history = await adapter.get_history("prompt-1")

    assert history == {"outputs": {}}
    assert attempts == 2


@pytest.mark.asyncio
async def test_object_info_retries_transient_read_timeout() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadTimeout("ComfyUI is busy", request=request)
        return httpx.Response(
            200,
            json={"CLIPTextEncode": {"input": {"required": {}}, "output": []}},
        )

    adapter = ComfyUIAdapter(None, "http://test", transport=httpx.MockTransport(handler))
    info = await adapter.get_object_info()

    assert tuple(info.nodes) == ("CLIPTextEncode",)
    assert attempts == 3


@pytest.mark.asyncio
async def test_prompt_validation_error_preserves_comfyui_node_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "prompt_outputs_failed_validation",
                    "message": "validation failed",
                },
                "node_errors": {
                    "2": {
                        "errors": [
                            {
                                "message": "Value not in list",
                                "details": "clip_name: missing.safetensors",
                            }
                        ]
                    }
                },
            },
        )

    adapter = ComfyUIAdapter(None, "http://test", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="node 2: Value not in list") as error:
        await adapter.submit_prompt(
            {"2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "missing"}}}
        )
    assert "clip_name: missing.safetensors" in str(error.value)
