from datetime import timedelta

import pytest

from ai_video_generator.domain import ExecutionTarget, TaskKind, TaskSpec, TaskState
from ai_video_generator.persistence import LeaseError, SQLiteTaskStore
from ai_video_generator.services.remote import TaskResultReport, WorkerResultStatus


@pytest.mark.parametrize("confirmed", [True, False])
def test_remote_cancel_renewal_and_attempt_scoped_acknowledgement(tmp_path, confirmed):
    store = SQLiteTaskStore(tmp_path / "cancel.db")
    store.add_task(
        TaskSpec(
            task_id="remote",
            project_id="project",
            kind=TaskKind.H3_GENERATION,
            state=TaskState.READY,
            idempotency_key="a" * 64,
            input_fingerprint="b" * 64,
            execution_target=ExecutionTarget.REMOTE,
            worker_id="worker",
        )
    )
    store.transition_task("remote", TaskState.QUEUED)
    claimed = store.claim_next(
        "worker",
        execution_target=ExecutionTarget.REMOTE,
        lease_duration=timedelta(seconds=120),
    )
    assert claimed is not None
    store.request_task_cancellation("remote")
    renewed = store.renew_lease(
        "remote",
        "worker",
        attempt_id=claimed.attempt_id,
        lease_duration=timedelta(seconds=120),
    )
    assert renewed.state == TaskState.CANCELLING
    report = TaskResultReport(
        report_id="c" * 64,
        task_id="remote",
        worker_id="worker",
        attempt=claimed.attempt,
        attempt_id=claimed.attempt_id,
        status=WorkerResultStatus.CANCELLED if confirmed else WorkerResultStatus.NEEDS_ATTENTION,
        stop_evidence="verified queue absence after interruption" if confirmed else None,
        error_code=None if confirmed else "cancellation_unconfirmed",
        submission_token="token",
        external_prompt_id="prompt",
    )
    with pytest.raises(LeaseError):
        store.submit_remote_result(report.model_copy(update={"attempt_id": "obsolete"}))
    receipt = store.submit_remote_result(report)
    assert receipt.state == ("cancelled" if confirmed else "cancelling")
    assert store.submit_remote_result(report) == receipt
    assert store.get_task("remote").comfyui_prompt_id == "prompt"
    assert store.inspect_submission("remote")["submission_token"] == "token"
