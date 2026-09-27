import hashlib
from datetime import UTC, datetime, timedelta

from ai_video_generator.domain import TaskKind, TaskSpec, TaskState
from ai_video_generator.services.dispatch_policy import dispatch_order


def _task(
    task_id: str, *, kind=TaskKind.H3_GENERATION, priority=0,
    created_at=None, updated_at=None
):
    fingerprint = hashlib.sha256(task_id.encode()).hexdigest()
    return TaskSpec(
        task_id=task_id,
        project_id="p",
        kind=kind,
        state=TaskState.QUEUED,
        idempotency_key=fingerprint,
        input_fingerprint=fingerprint,
        affinity_key="model:a",
        priority=priority,
        created_at=created_at,
        updated_at=updated_at,
    )


def test_heartbeat_updates_do_not_reset_wait_age():
    now = datetime(2026, 9, 24, tzinfo=UTC)
    old = _task("old", created_at=now - timedelta(minutes=10), updated_at=now)
    fresh = _task("fresh", created_at=now - timedelta(seconds=1))

    assert dispatch_order((fresh, old), now=now)[0].task_id == "old"


def test_affinity_is_bounded_after_eight_tasks():
    now = datetime(2026, 9, 24, tzinfo=UTC)
    resident = tuple(
        _task(f"resident-{index}", created_at=now - timedelta(seconds=index))
        for index in range(8)
    )
    other = _task("other", created_at=now - timedelta(minutes=1), kind=TaskKind.IMAGE_GENERATION)

    ordered = dispatch_order(
        (*resident, other), resident_key="model:a", consecutive=8,
        window_started=now, now=now,
    )

    assert ordered[0].task_id == "other"
