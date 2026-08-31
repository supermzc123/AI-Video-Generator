from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
from pathlib import Path
from typing import Any

from platformdirs import user_data_path

from ai_video_generator.domain import TaskSpec, TaskWorkloadManifest, WorkerCapabilities
from ai_video_generator.services.remote import WorkerResultStatus
from ai_video_generator.workers.comfyui import ComfyUIAdapter
from ai_video_generator.workers.remote_worker import (
    ProducedArtifact,
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
        error_code="workload_manifest_required",
        error_message=f"worker requires a workload manifest ({task.kind.value})",
    )


def _history_output(history: dict[str, Any], node_id: str) -> dict[str, str] | None:
    outputs = history.get("outputs")
    node = outputs.get(node_id) if isinstance(outputs, dict) else None
    if not isinstance(node, dict):
        return None
    for key in ("gifs", "videos", "images", "audio"):
        values = node.get(key)
        if isinstance(values, list) and values and isinstance(values[0], dict):
            item = values[0]
            filename = item.get("filename")
            if isinstance(filename, str) and filename:
                return {
                    "filename": filename,
                    "subfolder": str(item.get("subfolder") or ""),
                    "type": str(item.get("type") or "output"),
                }
    return None


def _workload_executor(
    *, client: RemoteWorkerClient, adapter: ComfyUIAdapter, comfyui_root: Path,
    work_root: Path,
):
    async def execute(
        task: TaskSpec, manifest: TaskWorkloadManifest
    ) -> WorkerExecutionOutcome:
        input_root = (comfyui_root / "input").resolve()
        for blob in manifest.input_blobs:
            target = (input_root / Path(blob.mount_path)).resolve()
            if input_root not in target.parents:
                raise ValueError("workload input escapes ComfyUI input directory")
            await client.download_artifact(blob.sha256, target)
        submission = await adapter.submit_prompt(manifest.prompt)
        if submission.node_errors:
            raise RuntimeError(f"ComfyUI rejected workload: {submission.node_errors}")
        deadline = asyncio.get_running_loop().time() + 4 * 60 * 60
        try:
            while asyncio.get_running_loop().time() < deadline:
                history = await adapter.get_history(submission.prompt_id)
                if history is None:
                    await asyncio.sleep(0.75)
                    continue
                status = history.get("status")
                status_text = str(
                    status.get("status_str") if isinstance(status, dict) else status or ""
                ).lower()
                if status_text in {"error", "failed"}:
                    raise RuntimeError("ComfyUI workload failed")
                artifacts: list[ProducedArtifact] = []
                missing_outputs: list[str] = []
                for output in manifest.outputs:
                    media = _history_output(history, output.node_id)
                    if media is None:
                        missing_outputs.append(output.node_id)
                        continue
                    content = await adapter.get_output_image(
                        media["filename"],
                        subfolder=media["subfolder"],
                        storage_type=media["type"],
                    )
                    directory = work_root / task.task_id.replace(":", "_")
                    directory.mkdir(parents=True, exist_ok=True)
                    filename = f"{output.node_id}-{Path(media['filename']).name}"
                    target = directory / filename
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    temporary.write_bytes(content)
                    temporary.replace(target)
                    artifacts.append(
                        ProducedArtifact(path=target, media_type=output.media_type)
                    )
                if not missing_outputs:
                    return WorkerExecutionOutcome(
                        status=WorkerResultStatus.SUCCEEDED, artifacts=tuple(artifacts)
                    )
                if isinstance(status, dict) and status.get("completed") is True:
                    missing = ", ".join(missing_outputs)
                    raise RuntimeError(
                        f"ComfyUI completed without declared output nodes: {missing}"
                    )
                await asyncio.sleep(0.75)
            await adapter.cancel_prompt(submission.prompt_id)
            raise TimeoutError("ComfyUI workload exceeded four hours")
        except asyncio.CancelledError:
            await adapter.cancel_prompt(submission.prompt_id)
            raise

    return execute


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
            workload_executor=_workload_executor(
                client=client,
                adapter=adapter,
                comfyui_root=Path(args.comfyui_root).resolve(),
                work_root=Path(args.work_root).resolve(),
            ),
        )
        await runtime.run(stop_event=stop)


def run() -> None:
    parser = argparse.ArgumentParser(description="Experimental AI Video Generator Worker")
    parser.add_argument("--control-plane", required=True)
    parser.add_argument("--worker-id", default=socket.gethostname())
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--work-root",
        default=str(user_data_path("ai-video-generator-worker", "supermzc123") / "work"),
    )
    parser.add_argument("--allow-insecure-http", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    run()
