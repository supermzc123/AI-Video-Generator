from datetime import UTC, datetime, timedelta

import pytest

from ai_video_generator.domain import TaskKind, TaskSpec, TaskState
from ai_video_generator.persistence import SQLiteTaskStore


def task(task_id: str, *, state: TaskState = TaskState.READY) -> TaskSpec:
    fingerprint = ("b" if task_id == "t1" else "c") * 64
    idempotency = ("a" if task_id in {"t1", "parent", "a"} else "d") * 64
    return TaskSpec(
        task_id=task_id,
        project_id="p1",
        kind=TaskKind.H3_GENERATION,
        state=state,
        idempotency_key=idempotency,
        input_fingerprint=fingerprint,
    )


def test_expired_dispatcher_fences_publication_and_recovery(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control.db", local_lease_seconds=10)
    store.add_task(task("t1"))
    store.transition_task("t1", TaskState.QUEUED)
    now = datetime(2026, 9, 24, tzinfo=UTC)
    assert store.acquire_dispatcher("d1", now=now)
    claimed = store.claim_local_task("t1", "d1", now=now)
    assert claimed is not None and claimed.attempt == 1
    assert store.recover_local_attempts(now=now + timedelta(seconds=11)) == ("t1",)
    assert store.get_task("t1").state == TaskState.NEEDS_ATTENTION
    with pytest.raises(ValueError, match="running attempt"):
        store.ensure_submission_intent("t1")


def test_orchestration_yield_clears_ownership_without_new_attempt(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control.db")
    parent = task("parent")
    child = task("child")
    store.add_task(parent)
    store.add_task(child)
    store.transition_task("parent", TaskState.QUEUED)
    store.transition_task("child", TaskState.QUEUED)
    now = datetime(2026, 9, 24, tzinfo=UTC)
    assert store.acquire_dispatcher("d1", now=now)
    claimed = store.claim_local_task("parent", "d1", now=now)
    assert claimed is not None
    assert store.defer_orchestration_if_child_pending("parent", "child")
    yielded = store.get_task("parent")
    assert yielded.state == TaskState.BLOCKED
    assert yielded.attempt == 1
    assert yielded.lease_expires_at is None


def test_dependency_cycle_is_rejected(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control.db")
    store.add_task(task("a"))
    store.add_task(task("b"))
    store.defer_orchestration("a", ("b",))
    with pytest.raises(ValueError, match="cycle"):
        store.defer_orchestration("b", ("a",))
