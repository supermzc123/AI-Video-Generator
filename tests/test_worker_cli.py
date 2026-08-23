from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_video_generator.domain import (
    ComfyUIOutput,
    ExecutionTarget,
    TaskKind,
    TaskSpec,
    TaskState,
    TaskWorkloadManifest,
    WorkloadBlob,
)
from ai_video_generator.services.remote import WorkerResultStatus
from ai_video_generator.worker_cli import _history_output, _workload_executor


def _manifest(*, output_nodes: tuple[str, ...] = ("2",)) -> TaskWorkloadManifest:
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "inputs/a.png"}},
        **{
            node_id: {"class_type": "SaveVideo", "inputs": {"images": ["1", 0]}}
            for node_id in output_nodes
        },
    }
    return TaskWorkloadManifest(
        task_kind=TaskKind.H3_GENERATION,
        workflow_sha256="1" * 64,
        node_schema_sha256="2" * 64,
        prompt=prompt,
        input_blobs=(
            WorkloadBlob(
                sha256="3" * 64,
                media_type="image/png",
                mount_path="inputs/a.png",
                role="reference_image",
            ),
        ),
        outputs=tuple(
            ComfyUIOutput(node_id=node_id, media_type="video/mp4")
            for node_id in output_nodes
        ),
    )


def _task(manifest: TaskWorkloadManifest) -> TaskSpec:
    return TaskSpec(
        task_id="remote:h3:1",
        project_id="project-1",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.RUNNING,
        idempotency_key="4" * 64,
        input_fingerprint="5" * 64,
        workload_manifest_sha256=manifest.sha256,
        execution_target=ExecutionTarget.REMOTE,
        worker_id="ubuntu-1",
    )


def test_history_output_reads_supported_comfyui_media() -> None:
    history = {
        "outputs": {
            "7": {
                "videos": [
                    {"filename": "clip.mp4", "subfolder": "run", "type": "output"}
                ]
            }
        }
    }
    assert _history_output(history, "7") == {
        "filename": "clip.mp4",
        "subfolder": "run",
        "type": "output",
    }
    assert _history_output(history, "missing") is None


@pytest.mark.asyncio
async def test_workload_executor_materializes_inputs_and_collects_all_outputs(
    tmp_path: Path,
) -> None:
    manifest = _manifest(output_nodes=("2", "3"))
    downloads: list[tuple[str, Path]] = []

    class Client:
        async def download_artifact(self, digest: str, destination: Path) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"input")
            downloads.append((digest, destination))

    class Adapter:
        async def submit_prompt(self, prompt: object) -> object:
            assert prompt == manifest.prompt
            return SimpleNamespace(prompt_id="prompt-1", node_errors={})

        async def get_history(self, prompt_id: str) -> dict[str, object]:
            assert prompt_id == "prompt-1"
            return {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {
                    node_id: {
                        "videos": [
                            {"filename": "clip.mp4", "subfolder": node_id, "type": "output"}
                        ]
                    }
                    for node_id in ("2", "3")
                },
            }

        async def get_output_image(self, filename: str, **kwargs: str) -> bytes:
            return f"{kwargs['subfolder']}:{filename}".encode()

    comfyui_root = tmp_path / "ComfyUI"
    executor = _workload_executor(
        client=Client(),
        adapter=Adapter(),
        comfyui_root=comfyui_root,
        work_root=tmp_path / "work",
    )
    outcome = await executor(_task(manifest), manifest)

    assert downloads == [("3" * 64, comfyui_root / "input" / "inputs" / "a.png")]
    assert outcome.status is WorkerResultStatus.SUCCEEDED
    assert [artifact.path.name for artifact in outcome.artifacts] == [
        "2-clip.mp4",
        "3-clip.mp4",
    ]
    assert [artifact.path.read_bytes() for artifact in outcome.artifacts] == [
        b"2:clip.mp4",
        b"3:clip.mp4",
    ]
