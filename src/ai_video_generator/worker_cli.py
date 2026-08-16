from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
from pathlib import Path

from ai_video_generator.domain import TaskSpec, WorkerCapabilities
from ai_video_generator.services.remote import WorkerResultStatus
from ai_video_generator.workers.comfyui import ComfyUIAdapter
from ai_video_generator.workers.remote_worker import (
    RemoteWorkerClient,
    RemoteWorkerRuntime,
    WorkerExecutionOutcome,
)


def _credential(name: str) -> str | None:
    directory = os.getenv("CREDENTIALS_DIRECTORY")
    if directory:
        path = Path(directory) / name
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return os.getenv(f"AIVIDEO_{name.upper().replace('-', '_')}")


async def _unsupported(task: TaskSpec) -> WorkerExecutionOutcome:
    return WorkerExecutionOutcome(
        status=WorkerResultStatus.FAILED,
        error_code="legacy_task_unsupported",
        error_message=f"experimental Worker only accepts workload manifests ({task.kind.value})",
    )


async def _run(args: argparse.Namespace) -> None:
    token = _credential("worker-token")
    if not token:
        raise SystemExit("worker token is required via systemd credential or AIVIDEO_WORKER_TOKEN")
    adapter = ComfyUIAdapter(Path(args.comfyui_root), args.comfyui_url, args.timeout)
    object_info = await adapter.get_object_info()
    local = await adapter.capabilities()
    if not local.server_online:
        raise SystemExit("local ComfyUI is offline")
    capabilities = WorkerCapabilities(
        worker_id=args.worker_id,
        platform="linux",
        gpu_names=tuple(
            str(item.get("name") or "GPU")
            for item in (local.system_stats or {}).get("devices", [])
            if isinstance(item, dict)
        ),
        comfyui_version=local.inventory.version,
        comfyui_commit=local.inventory.commit,
        node_schema_sha256=object_info.node_schema_sha256,
        node_types=tuple(sorted(object_info.nodes)),
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for event in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(event, stop.set)
    async with RemoteWorkerClient(
        control_plane_url=args.control_plane,
        token=token,
        timeout_seconds=args.timeout,
        allow_insecure_http=args.allow_insecure_http,
    ) as client:
        runtime = RemoteWorkerRuntime(
            client=client,
            capabilities=capabilities,
            executor=_unsupported,
        )
        await runtime.run(stop_event=stop)


def run() -> None:
    parser = argparse.ArgumentParser(description="Experimental AI Video Generator Worker")
    parser.add_argument("--control-plane", required=True)
    parser.add_argument("--worker-id", default=socket.gethostname())
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--allow-insecure-http", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    run()
