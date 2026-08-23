from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore, StoreConflictError

_BOUNDARY_KINDS: dict[str, frozenset[TaskKind]] = {
    "generation": frozenset(
        {
            TaskKind.IMAGE_GENERATION,
            TaskKind.CONDITIONING_ENCODING,
            TaskKind.MODEL_SWITCH,
            TaskKind.H3_GENERATION,
        }
    ),
    "review": frozenset({TaskKind.AI_REVIEW}),
    "delivery": frozenset(
        {
            TaskKind.SEEDVR2,
            TaskKind.RIFE,
            TaskKind.MASTER_ASSEMBLY,
            TaskKind.WHISPER,
            TaskKind.EXPORT,
        }
    ),
}
_TERMINAL_STATES = frozenset(
    {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED, TaskState.STALE}
)
_UNSUCCESSFUL_STATES = frozenset(
    {TaskState.FAILED, TaskState.CANCELLED, TaskState.STALE}
)


def resolve_batch_run(store: SQLiteTaskStore, batch: BatchRun) -> BatchRun:
    """Resolve project boundaries to an immutable, dependency-complete task set.

    Client-provided task IDs are intentionally ignored. They remain in the
    persisted response for backwards compatibility and observability only.
    """
    resolved_items: list[BatchRunItem] = []
    active_members = {
        item.project_id
        for existing in store.list_batch_runs()
        if existing.state in {BatchState.DRAFT, BatchState.RUNNING, BatchState.PAUSED}
        and existing.batch_id != batch.batch_id
        for item in existing.items
    }
    for item in batch.items:
        if item.project_id in active_members:
            raise StoreConflictError(
                f"project {item.project_id} already belongs to an active batch"
            )
        tasks = store.list_tasks(project_id=item.project_id)
        created_planning = False
        orchestration_kinds = {
            TaskKind.LLM_PLANNING,
            TaskKind.CONDITIONING_ENCODING,
            TaskKind.MODEL_SWITCH,
            TaskKind.H3_GENERATION,
            TaskKind.AI_REVIEW,
            TaskKind.MASTER_ASSEMBLY,
            TaskKind.EXPORT,
        }
        if not any(
            task.kind in orchestration_kinds and task.state not in _TERMINAL_STATES
            for task in tasks
        ):
            try:
                run_state = store.get_project_run_state(item.project_id)
                workspace = store.get_latest_project_workspace(item.project_id)
            except KeyError as exc:
                raise StoreConflictError(
                    f"project {item.project_id} has no saved workspace"
                ) from exc
            if not run_state.outline_approved:
                raise StoreConflictError(
                    f"project {item.project_id} must approve its outline before batch admission"
                )
            fingerprint = hashlib.sha256(
                (
                    f"batch-orchestration-v1:{batch.batch_id}:"
                    f"{item.project_id}:{workspace.revision}"
                ).encode()
            ).hexdigest()
            dependencies = tuple(
                task.task_id
                for task in tasks
                if task.state not in _TERMINAL_STATES
            )
            planning = store.add_task(
                TaskSpec(
                    task_id=f"batch-plan:{item.project_id}:{fingerprint[:16]}",
                    project_id=item.project_id,
                    kind=TaskKind.LLM_PLANNING,
                    state=TaskState.BLOCKED if dependencies else TaskState.READY,
                    idempotency_key=hashlib.sha256(
                        f"batch-plan:{fingerprint}".encode()
                    ).hexdigest(),
                    input_fingerprint=fingerprint,
                    affinity_key="llm:project-orchestration",
                    depends_on=dependencies,
                    priority=item.priority,
                    max_attempts=3,
                )
            )
            tasks = (*tasks, planning)
            created_planning = True
        selected = (
            {planning.task_id}
            if created_planning
            else _resolve_item_tasks(tasks, item.start_boundary)
        )
        if not selected:
            raise StoreConflictError(
                f"project {item.project_id} has no unfinished tasks at {item.start_boundary}"
            )
        task_ids = tuple(task.task_id for task in tasks if task.task_id in selected)
        for task_id in task_ids:
            store.set_task_priority(task_id, item.priority)
        resolved_items.append(item.model_copy(update={"task_ids": task_ids}))
    return batch.model_copy(update={"items": tuple(resolved_items)})


def batch_project_tasks(
    store: SQLiteTaskStore, batch: BatchRun, project_id: str
) -> tuple[TaskSpec, ...]:
    """Return exactly the frozen tasks belonging to one batch member."""
    item = next((item for item in batch.items if item.project_id == project_id), None)
    if item is None:
        raise StoreConflictError(f"project {project_id} is not a member of batch {batch.batch_id}")
    by_id = {task.task_id: task for task in store.list_tasks(project_id=project_id)}
    missing = tuple(task_id for task_id in item.task_ids if task_id not in by_id)
    if missing:
        raise StoreConflictError(
            f"batch {batch.batch_id} references missing tasks: {', '.join(missing)}"
        )
    return tuple(by_id[task_id] for task_id in item.task_ids)


def resolve_batch_task_ids(tasks: tuple[TaskSpec, ...], boundary: str) -> tuple[str, ...]:
    """Resolve a compiled project DAG using the same boundary contract as admission."""
    selected = _resolve_item_tasks(tasks, boundary)
    return tuple(task.task_id for task in tasks if task.task_id in selected)


def reconcile_batch_runs(
    store: SQLiteTaskStore, *, now: datetime | None = None
) -> tuple[BatchRun, ...]:
    """Settle dependency failures and complete batches whose members are terminal."""
    changed_at = now or datetime.now(UTC)
    results: list[BatchRun] = []
    for batch in store.list_batch_runs():
        if batch.state != BatchState.RUNNING:
            continue
        selected_ids = {task_id for item in batch.items for task_id in item.task_ids}
        tasks = {
            task.task_id: task
            for item in batch.items
            for task in store.list_tasks(project_id=item.project_id)
            if task.task_id in selected_ids
        }
        changed = True
        while changed:
            changed = False
            for task in tuple(tasks.values()):
                if task.state != TaskState.BLOCKED:
                    continue
                dependencies = [tasks.get(task_id) for task_id in task.depends_on]
                if any(
                    dependency is not None and dependency.state in _UNSUCCESSFUL_STATES
                    for dependency in dependencies
                ):
                    updated = store.transition_task(
                        task.task_id,
                        TaskState.CANCELLED,
                        error_code="dependency_failed",
                        error_message="cancelled because a required batch task did not succeed",
                        now=changed_at,
                    )
                    tasks[task.task_id] = updated
                    changed = True
        if tasks and all(task.state in _TERMINAL_STATES for task in tasks.values()):
            batch = batch.model_copy(
                update={"state": BatchState.COMPLETED, "updated_at": changed_at}
            )
            store.put_batch_run(batch)
        results.append(batch)
    return tuple(results)


def _resolve_item_tasks(tasks: tuple[TaskSpec, ...], boundary: str) -> set[str]:
    by_id = {task.task_id: task for task in tasks}
    children: dict[str, set[str]] = {task.task_id: set() for task in tasks}
    for task in tasks:
        for dependency in task.depends_on:
            if dependency in children:
                children[dependency].add(task.task_id)

    if boundary == "next_ready":
        selected = {task.task_id for task in tasks if task.state not in _TERMINAL_STATES}
    else:
        kinds = _BOUNDARY_KINDS.get(boundary)
        if kinds is None:
            raise StoreConflictError(f"unknown batch start boundary: {boundary}")
        seeds = {task.task_id for task in tasks if task.kind in kinds}
        selected = _walk(seeds, children)
        selected = {
            task_id for task_id in selected if by_id[task_id].state not in _TERMINAL_STATES
        }

    # Every unfinished ancestor is part of the batch. This is what prevents a
    # blocked review or export task from being selected without its H3 work.
    pending = list(selected)
    while pending:
        task = by_id[pending.pop()]
        for dependency_id in task.depends_on:
            dependency = by_id.get(dependency_id)
            if (
                dependency is not None
                and dependency.state != TaskState.SUCCEEDED
                and dependency_id not in selected
            ):
                selected.add(dependency_id)
                pending.append(dependency_id)
    return selected


def _walk(seeds: set[str], edges: dict[str, set[str]]) -> set[str]:
    found = set(seeds)
    pending = list(seeds)
    while pending:
        for child in edges.get(pending.pop(), ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found
