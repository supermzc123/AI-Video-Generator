"""Independent repair contract probes; failures are not expected/softened."""

from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from random import Random

import pytest

from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ProjectRunState,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import (
    InvalidTaskTransitionError,
    SQLiteTaskStore,
    StoreConflictError,
    execution_runtime,
    task_store,
)
from ai_video_generator.persistence.execution_runtime import execution_guard

START = datetime(2026, 9, 24, tzinfo=UTC)
SEED = 20260924


def task(
    task_id,
    *,
    state=TaskState.QUEUED,
    kind=TaskKind.H3_GENERATION,
    project_id="project",
    depends_on=(),
):
    return TaskSpec(
        task_id=task_id,
        project_id=project_id,
        kind=kind,
        state=state,
        depends_on=depends_on,
        idempotency_key=sha256(task_id.encode()).hexdigest(),
        input_fingerprint=sha256(f"input:{task_id}".encode()).hexdigest(),
    )


@contextmanager
def guarded(claimed):
    token = execution_guard.set((claimed.task_id, claimed.attempt_id))
    try:
        yield
    finally:
        execution_guard.reset(token)


@pytest.fixture
def store(tmp_path, monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return START if tz is not None else START.replace(tzinfo=None)

    # All implicit store/guard reads share the explicit deterministic test clock.
    monkeypatch.setattr(execution_runtime, "datetime", FrozenDateTime)
    monkeypatch.setattr(task_store, "datetime", FrozenDateTime)
    result = SQLiteTaskStore(tmp_path / "invariants.db")
    assert result.acquire_dispatcher("owner", now=START)
    return result


def claim(store, task_id="work", **kwargs):
    store.add_task(task(task_id, **kwargs))
    claimed = store.claim_local_task(task_id, "owner", now=START)
    assert claimed is not None
    return claimed


def test_seeded_10000_sqlite_state_operations(store):
    """Independent oracle checks every operation against real SQLite state."""
    random = Random(SEED)
    originals = [task(f"work-{index}") for index in range(12)]
    states = {item.task_id: TaskState.QUEUED for item in originals}
    for item in originals:
        store.add_task(item)
    paused = False
    operations = ("claim", "pause", "complete", "cancel", "duplicate", "reopen")
    counts = dict.fromkeys(operations, 0)
    for index in range(10_000):
        operation = operations[index % len(operations)]
        counts[operation] += 1
        original = random.choice(originals)
        task_id = original.task_id
        before = states[task_id]
        if operation == "claim":
            owner = random.choice(("owner", "intruder"))
            expected = (
                owner == "owner"
                and not paused
                and before == TaskState.QUEUED
                and TaskState.RUNNING not in states.values()
            )
            claimed = store.claim_local_task(task_id, owner, now=START)
            assert (claimed is not None) == expected, (SEED, index, operation, states)
            if claimed is not None:
                states[task_id] = TaskState.RUNNING
        elif operation == "pause":
            paused = not paused
            store.put_project_run_state(
                ProjectRunState(
                    project_id="project",
                    paused=paused,
                    updated_at=START,
                )
            )
        elif operation == "complete" and before == TaskState.RUNNING:
            store.transition_task(task_id, TaskState.SUCCEEDED, now=START)
            states[task_id] = TaskState.SUCCEEDED
        elif operation == "cancel" and before == TaskState.QUEUED:
            store.transition_task(task_id, TaskState.CANCELLED, now=START)
            states[task_id] = TaskState.CANCELLED
        elif operation == "duplicate":
            assert store.add_task(original).task_id == task_id
        elif operation == "reopen":
            store = SQLiteTaskStore(store.database_path)
        actual = {item.task_id: item.state for item in store.list_tasks()}
        assert actual == states, (SEED, index, operation, actual, states)
        assert sum(state == TaskState.RUNNING for state in actual.values()) <= 1
        if index % 30 == 29:
            new_task = task(f"new-{index}")
            store.add_task(new_task)
            originals.append(new_task)
            states[new_task.task_id] = TaskState.QUEUED
    assert sum(counts.values()) == 10_000
    assert min(counts.values()) >= 1666


@pytest.mark.parametrize(
    "parent_state", [TaskState.FAILED, TaskState.CANCELLED, TaskState.RUNNING, TaskState.BLOCKED]
)
def test_dependency_failure_never_claims_child_or_blocks_other_project(store, parent_state):
    store.add_task(task("parent", state=parent_state, kind=TaskKind.LLM_PLANNING))
    store.add_task(task("child", depends_on=("parent",)))
    assert store.claim_local_task("child", "owner", now=START) is None
    assert claim(store, "independent", project_id="another") is not None


def test_claim_pause_and_dispatcher_takeover_are_durable(store):
    store.add_task(task("work"))
    store.put_project_run_state(
        ProjectRunState(
            project_id="project",
            paused=True,
            updated_at=START,
        )
    )
    reopened = SQLiteTaskStore(store.database_path)
    assert reopened.claim_local_task("work", "owner", now=START) is None
    assert not reopened.acquire_dispatcher("intruder", now=START)
    reopened.put_project_run_state(
        ProjectRunState(
            project_id="project",
            paused=False,
            updated_at=START,
        )
    )
    later = START + timedelta(seconds=121)
    assert reopened.acquire_dispatcher("new-owner", now=later)
    assert reopened.claim_local_task("work", "owner", now=later) is None
    assert reopened.claim_local_task("work", "new-owner", now=later) is not None


def test_two_llm_slots_and_one_gpu_slot(store):
    claim(store, "llm-1", kind=TaskKind.LLM_PLANNING)
    claim(store, "llm-2", kind=TaskKind.AI_REVIEW)
    store.add_task(task("llm-3", kind=TaskKind.LLM_PLANNING))
    assert store.claim_local_task("llm-3", "owner", now=START) is None
    claim(store, "gpu-1")
    store.add_task(task("gpu-2"))
    assert store.claim_local_task("gpu-2", "owner", now=START) is None


@pytest.mark.parametrize("terminal", [TaskState.CANCELLED, TaskState.SUCCEEDED])
def test_delayed_retry_cannot_revive_terminal_task(store, terminal):
    claim(store)
    store.transition_task("work", terminal, now=START)
    with suppress(ValueError, InvalidTaskTransitionError, StoreConflictError):
        store.schedule_task_retry("work", "late network failure", now=START)
    store.promote_due_retries(now=START + timedelta(minutes=1))
    assert store.get_task("work").state == terminal


@pytest.mark.parametrize("boundary", ["restart", "owner_takeover"])
def test_previous_owner_cannot_publish_after_recovery_boundary(store, boundary):
    old = claim(store, kind=TaskKind.LLM_PLANNING)
    if boundary == "restart":
        store = SQLiteTaskStore(store.database_path)
        store.recover_local_attempts(now=START + timedelta(seconds=121))
    else:
        assert store.acquire_dispatcher("new-owner", now=START + timedelta(seconds=121))
    before = store.get_task("work")
    with guarded(old), suppress(ValueError, InvalidTaskTransitionError, StoreConflictError):
        store.transition_task("work", TaskState.SUCCEEDED, now=START + timedelta(seconds=122))
    assert store.get_task("work").state == before.state


def test_stale_attempt_after_new_claim_cannot_publish(store):
    old = claim(store, kind=TaskKind.LLM_PLANNING)
    with guarded(old):
        store.defer_orchestration("work")
    store.transition_task("work", TaskState.QUEUED, now=START)
    current = store.claim_local_task("work", "owner", now=START)
    assert current is not None and current.attempt_id != old.attempt_id
    assert current.attempt == old.attempt
    with guarded(old), pytest.raises((ValueError, InvalidTaskTransitionError, StoreConflictError)):
        store.transition_task("work", TaskState.SUCCEEDED, now=START)
    assert store.get_task("work").attempt_id == current.attempt_id


def test_unknown_submission_reserves_gpu_and_reuses_intent(store):
    current = claim(store)
    with guarded(current):
        submission, fresh = store.ensure_submission_intent("work")
    assert fresh
    store = SQLiteTaskStore(store.database_path)
    store.recover_local_attempts(now=START)
    assert store.inspect_submission("work")["submission_token"] == submission
    store.transition_task("work", TaskState.NEEDS_ATTENTION, now=START)
    store.add_task(task("next"))
    assert store.claim_local_task("next", "owner", now=START) is None


def test_multiple_recovered_external_jobs_can_each_be_observed(store):
    for task_id in ("old-a", "old-b"):
        store.add_task(
            task(task_id, state=TaskState.RUNNING).model_copy(
                update={
                    "comfyui_prompt_id": f"external-{task_id}",
                    "attempt": 1,
                }
            )
        )
    store.recover_local_attempts(now=START)
    observations = [
        store.claim_local_task(task_id, "owner", now=START) for task_id in ("old-a", "old-b")
    ]
    assert all(item is not None for item in observations), "recovered IDs mutually block"
    assert all(item.attempt == 1 for item in observations)
    store.add_task(task("fresh"))
    assert store.claim_local_task("fresh", "owner", now=START) is None


def test_cancelling_retains_gpu_until_stop_confirmation(store):
    claim(store)
    store.record_comfyui_prompt("work", "external")
    store.transition_task("work", TaskState.CANCELLING, now=START)
    store.add_task(task("next"))
    assert store.claim_local_task("next", "owner", now=START) is None
    store = SQLiteTaskStore(store.database_path)
    store.recover_local_attempts(now=START)
    assert store.get_task("work").state == TaskState.CANCELLING
    assert store.claim_local_task("next", "owner", now=START) is None
    store.transition_task("work", TaskState.CANCELLED, now=START)
    assert store.claim_local_task("next", "owner", now=START) is not None


def test_retry_deadline_and_attempt_survive_restart(store):
    current = claim(store, kind=TaskKind.LLM_PLANNING)
    store.schedule_task_retry("work", "network unavailable", now=START)
    due = store.get_task("work").next_retry_at
    assert due is not None
    store = SQLiteTaskStore(store.database_path)
    store.promote_due_retries(now=due - timedelta(microseconds=1))
    assert store.get_task("work").state == TaskState.RETRY_WAIT
    store.promote_due_retries(now=due)
    retried = store.claim_local_task("work", "owner", now=due)
    assert retried is not None and retried.attempt == current.attempt + 1
    assert retried.deadline_at == current.deadline_at, "absolute deadline reset by retry"


def test_parent_yield_preserves_attempt_and_cancelled_child(store):
    parent = claim(store, "parent", kind=TaskKind.LLM_PLANNING)
    store.add_task(task("child", state=TaskState.CANCELLED))
    with guarded(parent):
        assert store.defer_orchestration_if_child_pending("parent", "child")
    store.recover_local_attempts(now=START)
    store.promote_due_retries(now=START + timedelta(seconds=10))
    assert store.get_task("parent").state == TaskState.BLOCKED
    assert store.get_task("child").state == TaskState.CANCELLED
    assert claim(store, "other", kind=TaskKind.LLM_PLANNING).attempt == 1


def test_parent_resumption_does_not_burn_generation_attempt(store):
    parent = claim(store, "parent", kind=TaskKind.LLM_PLANNING)
    store.add_task(task("child"))
    with guarded(parent):
        assert store.defer_orchestration_if_child_pending("parent", "child")
    assert store.claim_local_task("child", "owner", now=START) is not None
    store.transition_task("child", TaskState.SUCCEEDED, now=START)
    assert store.get_task("parent").state == TaskState.READY
    store.transition_task("parent", TaskState.QUEUED, now=START)
    resumed = store.claim_local_task("parent", "owner", now=START)
    assert resumed is not None
    assert resumed.attempt == parent.attempt
    assert resumed.attempt_id != parent.attempt_id


def test_stale_parent_child_insertion_rolls_back(store):
    parent = claim(store, "parent", kind=TaskKind.LLM_PLANNING)
    with guarded(parent):
        store.defer_orchestration("parent")
    store.transition_task("parent", TaskState.QUEUED, now=START)
    assert store.claim_local_task("parent", "owner", now=START) is not None
    with (
        guarded(parent),
        pytest.raises((ValueError, InvalidTaskTransitionError, StoreConflictError)),
    ):
        store.add_task(task("orphan-child"))
    assert tuple(item.task_id for item in store.list_tasks()) == ("parent",)


def test_paused_batch_child_membership_is_atomic_and_idempotent(store):
    parent = claim(store, "parent", kind=TaskKind.LLM_PLANNING)
    store.put_batch_run(
        BatchRun(
            batch_id="batch",
            name="batch",
            state=BatchState.PAUSED,
            items=(BatchRunItem(project_id="project", task_ids=("parent",)),),
            created_at=START,
            updated_at=START,
        )
    )
    child = task("child")
    with guarded(parent):
        store.add_task(child)
        store.add_task(child.model_copy(update={"task_id": "duplicate-request"}))
    assert store.list_batch_runs()[0].items[0].task_ids == ("parent", "child")
    before = store.list_batch_runs()
    with pytest.raises(StoreConflictError):
        store.expand_running_batch_project_tasks(
            batch_id="batch",
            project_id="project",
            orchestration_task_id="parent",
            task_ids=("missing",),
            now=START,
        )
    assert store.list_batch_runs() == before
    assert len(store.list_tasks()) == 2
