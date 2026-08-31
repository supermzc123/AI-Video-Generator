from ai_video_generator.domain import (
    ComfyUIOutput,
    TaskKind,
    TaskWorkloadManifest,
    WorkerCapabilities,
    worker_can_execute_manifest,
)


def manifest(*, required_node_types: tuple[str, ...] = ("LoadImage",)) -> TaskWorkloadManifest:
    return TaskWorkloadManifest(
        task_kind=TaskKind.H3_GENERATION,
        workflow_sha256="a" * 64,
        node_schema_sha256="a" * 64,
        prompt={"1": {"class_type": "LoadImage", "inputs": {}}},
        outputs=(ComfyUIOutput(node_id="1", media_type="image/png"),),
        required_node_types=required_node_types,
        required_model_sha256_values=("b" * 64,),
    )


def test_remote_worker_requires_exact_nodes_and_models() -> None:
    worker = WorkerCapabilities(
        worker_id="ubuntu-1",
        platform="linux",
        node_schema_sha256="a" * 64,
        node_types=("LoadImage", "SaveImage"),
        model_sha256_values=("b" * 64,),
    )

    assert worker_can_execute_manifest(worker, manifest())
    assert not worker_can_execute_manifest(
        worker, manifest(required_node_types=("UnknownNode",))
    )


def test_remote_worker_requires_matching_node_schema() -> None:
    worker = WorkerCapabilities(
        worker_id="ubuntu-1",
        platform="linux",
        node_schema_sha256="a" * 64,
        node_types=("LoadImage",),
        model_sha256_values=("b" * 64,),
    )

    assert not worker_can_execute_manifest(
        worker,
        manifest().model_copy(update={"node_schema_sha256": "d" * 64}),
    )
