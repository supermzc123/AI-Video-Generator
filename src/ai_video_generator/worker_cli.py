from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from platformdirs import user_data_path

from ai_video_generator.domain import TaskSpec, TaskState, TaskWorkloadManifest, WorkerCapabilities
from ai_video_generator.services.remote import WorkerResultStatus
from ai_video_generator.services.remote_execution import RemoteExecutionJournal, validated_queue_ids
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
    *,
    client: RemoteWorkerClient,
    adapter: ComfyUIAdapter,
    comfyui_root: Path,
    work_root: Path,
    poll_seconds: float = 0.75,
):
    journal = RemoteExecutionJournal(work_root)

    async def execute(task: TaskSpec, manifest: TaskWorkloadManifest) -> WorkerExecutionOutcome:
        record = journal.prepare(task, manifest)
        prompt_id = record["prompt_id"]
        token = record["submission_token"]

        def outcome(status, *, code=None, message=None, evidence=None, artifacts=()):
            return WorkerExecutionOutcome(
                status=status,
                artifacts=artifacts,
                error_code=code,
                error_message=message,
                stop_evidence=evidence,
                submission_token=token,
                external_prompt_id=prompt_id,
            )

        async def stop_owned_prompt():
            nonlocal prompt_id
            current = journal.get(task)
            if current["phase"] == "prepared":
                return "submission was never started"
            if prompt_id is None:
                prompt_id = await adapter.find_submission(token)
                if prompt_id is None:
                    return None
                journal.update(task, phase="cancelling", prompt_id=prompt_id)
            await adapter.cancel_prompt(prompt_id)
            # The POST only requests interruption. A later validated queue observation
            # is the required evidence before the Worker reports cancellation.
            for _ in range(3):
                if prompt_id not in await validated_queue_ids(adapter):
                    return f"ComfyUI queue confirms prompt {prompt_id} is absent after stop request"
                await asyncio.sleep(poll_seconds)
            return None

        try:
            if task.state == TaskState.CANCELLING:
                raise asyncio.CancelledError
            if record["phase"] == "completed":
                outputs = journal.verified_outputs(task)
                return outcome(
                    WorkerResultStatus.SUCCEEDED,
                    artifacts=tuple(
                        ProducedArtifact(Path(item["path"]), item["media_type"]) for item in outputs
                    ),
                )
            if record["phase"] == "prepared":
                if datetime.now(UTC) >= task.deadline_at:
                    journal.update(
                        task, phase="stopped", detail="deadline expired before submission"
                    )
                    return outcome(
                        WorkerResultStatus.FAILED,
                        code="deadline_exceeded",
                        message="original execution deadline expired before submission",
                        evidence="submission was never started",
                    )
                input_root = (comfyui_root / "input").resolve()
                for blob in manifest.input_blobs:
                    target = (input_root / Path(blob.mount_path)).resolve()
                    if input_root not in target.parents:
                        raise ValueError("workload input escapes ComfyUI input directory")
                    await client.download_artifact(blob.sha256, target)
                if datetime.now(UTC) >= task.deadline_at:
                    journal.update(task, phase="stopped", detail="deadline expired during inputs")
                    return outcome(
                        WorkerResultStatus.FAILED,
                        code="deadline_exceeded",
                        message="original deadline expired while preparing inputs",
                        evidence="submission was never started",
                    )
                if journal.claim_submission(task):
                    try:
                        submission = await adapter.submit_prompt(
                            manifest.prompt,
                            client_id=f"avg-worker-{task.attempt_id}",
                            extra_data={
                                "avg_task_id": task.task_id,
                                "avg_attempt_id": task.attempt_id,
                                "avg_submission_token": token,
                            },
                        )
                        prompt_id = submission.prompt_id
                        journal.update(task, phase="submitted", prompt_id=prompt_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # An exception can follow server acceptance. Intent remains
                        # durable and no path may submit the same attempt again.
                        prompt_id = await adapter.find_submission(token)
                        if prompt_id:
                            journal.update(task, phase="submitted", prompt_id=prompt_id)
            if prompt_id is None:
                prompt_id = await adapter.find_submission(token)
                if prompt_id:
                    journal.update(task, phase="submitted", prompt_id=prompt_id)
            if prompt_id is None:
                journal.update(
                    task, phase="unknown", detail="submission result cannot be reconciled"
                )
                return outcome(
                    WorkerResultStatus.NEEDS_ATTENTION,
                    code="submission_unknown",
                    message="submission outcome is unknown; automatic replay is forbidden",
                )
            while True:
                history = await adapter.get_history(prompt_id)
                status = history.get("status") if history else None
                status_text = str(
                    status.get("status_str") if isinstance(status, dict) else status or ""
                ).lower()
                if status_text in {"error", "failed"}:
                    journal.update(task, phase="stopped", detail="ComfyUI history confirms failure")
                    return outcome(
                        WorkerResultStatus.FAILED,
                        code="comfy_execution_failed",
                        message="ComfyUI workload failed",
                        evidence="ComfyUI history confirms execution failure",
                    )
                complete = isinstance(status, dict) and status.get("completed") is True
                if complete:
                    outputs = []
                    for output in manifest.outputs:
                        media = _history_output(history, output.node_id)
                        if media is None:
                            raise ValueError(
                                f"completed workload is missing output {output.node_id}"
                            )
                        content = await adapter.get_output_image(
                            media["filename"],
                            subfolder=media["subfolder"],
                            storage_type=media["type"],
                        )
                        outputs.append(
                            journal.publish(
                                task, output.node_id, media["filename"], content, output.media_type
                            )
                        )
                    journal.update(task, phase="completed", outputs=outputs)
                    return outcome(
                        WorkerResultStatus.SUCCEEDED,
                        artifacts=tuple(
                            ProducedArtifact(Path(item["path"]), item["media_type"])
                            for item in outputs
                        ),
                    )
                if datetime.now(UTC) >= task.deadline_at:
                    evidence = await stop_owned_prompt()
                    journal.update(
                        task,
                        phase="stopped" if evidence else "unknown",
                        detail=evidence or "deadline stop was not confirmed",
                    )
                    return outcome(
                        WorkerResultStatus.FAILED
                        if evidence
                        else WorkerResultStatus.NEEDS_ATTENTION,
                        code="deadline_exceeded" if evidence else "deadline_stop_unconfirmed",
                        message="original absolute execution deadline expired",
                        evidence=evidence,
                    )
                await asyncio.sleep(poll_seconds)
        except asyncio.CancelledError:
            try:
                async with asyncio.timeout(45):
                    evidence = await stop_owned_prompt()
            except Exception as exc:
                evidence = None
                journal.update(task, phase="unknown", detail=f"stop check failed: {exc}"[:2000])
            else:
                journal.update(
                    task,
                    phase="stopped" if evidence else "unknown",
                    detail=evidence or "stop was not confirmed",
                )
            if evidence:
                return outcome(WorkerResultStatus.CANCELLED, evidence=evidence)
            return outcome(
                WorkerResultStatus.NEEDS_ATTENTION,
                code="cancellation_unconfirmed",
                message="Worker could not confirm external execution stopped",
            )
        except Exception as exc:
            journal.update(task, phase="unknown", detail=str(exc)[:2000])
            return outcome(
                WorkerResultStatus.NEEDS_ATTENTION,
                code="remote_execution_unknown",
                message=str(exc)[:2000],
            )

    execute.journal = journal
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
