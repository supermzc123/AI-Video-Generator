from datetime import UTC, datetime, timedelta

import pytest

from ai_video_generator.domain import TaskKind, TaskSpec, TaskState
from ai_video_generator.persistence import LeaseError, SQLiteTaskStore, StoreConflictError
from ai_video_generator.services.remote import (
    TaskResultReport,
    WorkerResultStatus,
)


def test_result_report_is_durable_idempotent_and_attempt_scoped(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    task = TaskSpec(
        task_id="segment-1",
        project_id="project-1",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.READY,
        idempotency_key="a" * 64,
        input_fingerprint="b" * 64,
    )
    store.add_task(task)
    store.transition_task(task.task_id, TaskState.QUEUED)
    now = datetime(2026, 8, 15, tzinfo=UTC)
    claimed = store.claim_next("ubuntu-1", lease_duration=timedelta(minutes=1), now=now)
    assert claimed is not None
    report = TaskResultReport(
        report_id="c" * 64,
        task_id=claimed.task_id,
        worker_id="ubuntu-1",
        attempt=claimed.attempt,
        attempt_id=claimed.attempt_id,
        status=WorkerResultStatus.SUCCEEDED,
    )

    receipt = store.submit_remote_result(report, now=now)
    assert store.submit_remote_result(report, now=now + timedelta(seconds=1)) == receipt
    assert SQLiteTaskStore(tmp_path / "tasks.db").submit_remote_result(report) == receipt
    assert store.get_task(task.task_id).state == TaskState.SUCCEEDED

    conflicting = report.model_copy(update={"status": WorkerResultStatus.NEEDS_REVIEW})
    with pytest.raises(StoreConflictError):
        store.submit_remote_result(conflicting)


def test_result_from_wrong_attempt_is_rejected(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    store.add_task(
        TaskSpec(
            task_id="segment-1",
            project_id="project-1",
            kind=TaskKind.H3_GENERATION,
            state=TaskState.READY,
            idempotency_key="a" * 64,
            input_fingerprint="b" * 64,
        )
    )
    store.transition_task("segment-1", TaskState.QUEUED)
    claimed = store.claim_next("ubuntu-1", lease_duration=timedelta(minutes=1))
    assert claimed is not None
    report = TaskResultReport(
        report_id="d" * 64,
        task_id=claimed.task_id,
        worker_id="ubuntu-1",
        attempt=claimed.attempt + 1,
        status=WorkerResultStatus.FAILED,
        error_code="worker_error",
    )

    with pytest.raises(LeaseError, match="attempt"):
        store.submit_remote_result(report)
