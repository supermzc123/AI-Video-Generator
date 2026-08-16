from ai_video_generator.domain import WorkerCapabilities
from ai_video_generator.services.remote import worker_can_run


def test_remote_worker_requires_exact_nodes_and_models() -> None:
    worker = WorkerCapabilities(
        worker_id="ubuntu-1",
        platform="linux",
        node_schema_sha256="a" * 64,
        node_types=("LoadImage", "SaveImage"),
        model_sha256_values=("b" * 64,),
    )

    assert worker_can_run(
        worker,
        required_node_types={"LoadImage"},
        required_model_sha256_values={"b" * 64},
    )
    assert not worker_can_run(
        worker,
        required_node_types={"UnknownNode"},
        required_model_sha256_values={"b" * 64},
    )
