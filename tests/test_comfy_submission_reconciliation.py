import pytest

from ai_video_generator.workers.comfyui import ComfyUIAdapter


@pytest.mark.asyncio
async def test_find_submission_rejects_malformed_queue_response(monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"queue_running": "bad", "queue_pending": []}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(ComfyUIAdapter, "_client", lambda self: Client())
    with pytest.raises(ValueError, match="queue_running"):
        await ComfyUIAdapter(None, "http://comfy.invalid").find_submission("token")


@pytest.mark.asyncio
async def test_find_submission_rejects_ambiguous_token_matches(monkeypatch) -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, path, **kwargs):
            if path == "/queue":
                return Response({"queue_running": [["x", "one", 0, {"avg_submission_token": "t"}]],
                                 "queue_pending": [["x", "two", 0, {"avg_submission_token": "t"}]]})
            return Response({})

    monkeypatch.setattr(ComfyUIAdapter, "_client", lambda self: Client())
    with pytest.raises(ValueError, match="multiple"):
        await ComfyUIAdapter(None, "http://comfy.invalid").find_submission("t")
