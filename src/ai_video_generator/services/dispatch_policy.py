from __future__ import annotations

from datetime import UTC, datetime

from ai_video_generator.domain import TaskKind, TaskSpec, TaskState

PHASES = {
    TaskKind.LLM_PLANNING: 0,
    TaskKind.IMAGE_GENERATION: 1,
    TaskKind.CONDITIONING_ENCODING: 2,
    TaskKind.MODEL_SWITCH: 3,
    TaskKind.H3_GENERATION: 4,
    TaskKind.AI_REVIEW: 5,
}


def dispatch_order(
    tasks: tuple[TaskSpec, ...],
    *,
    resident_key: str | None = None,
    consecutive: int = 0,
    window_started: datetime | None = None,
    now: datetime | None = None,
) -> tuple[TaskSpec, ...]:
    """Dependency eligibility is handled by the store, not phase ordering."""
    now = now or datetime.now(UTC)
    reuse = consecutive < 8 and (
        window_started is None or (now - window_started).total_seconds() < 300
    )

    def key(task: TaskSpec):
        created = task.created_at or now
        age = max(0.0, (now - created).total_seconds())
        return (
            task.state != TaskState.RECOVERING,
            age < 300,
            created.timestamp() if age >= 300 else 0,
            -task.priority,
            not (reuse and resident_key and task.affinity_key == resident_key),
            PHASES.get(task.kind, 6),
            created.timestamp(),
            task.task_id,
        )

    return tuple(sorted(tasks, key=key))
