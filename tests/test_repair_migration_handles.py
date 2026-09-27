"""Upgrade preservation and crash-safe checkpoint/connection regressions."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from ai_video_generator.domain import TaskKind, TaskSpec, TaskState
from ai_video_generator.domain.orchestration import TaskCheckpoint
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.persistence.sqlite_connection import ClosingConnection
from ai_video_generator.persistence.task_store import StoreConflictError, TaskStoreError


def seed_task(store):
    task = TaskSpec(
        task_id="preserved",
        project_id="project",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.READY,
        idempotency_key="a" * 64,
        input_fingerprint="b" * 64,
    )
    store.add_task(task)
    store.transition_task(task.task_id, TaskState.QUEUED)
    store.acquire_dispatcher("test-owner")
    assert store.claim_local_task(task.task_id, "test-owner") is not None
    store.record_comfyui_prompt(task.task_id, "external-prompt")
    return task


@pytest.mark.parametrize("version", ["6", "7"])
def test_upgrade_backs_up_and_preserves_existing_external_identity(tmp_path, version):
    path = tmp_path / "control.db"
    original = SQLiteTaskStore(path)
    seed_task(original)
    checkpoint = original.append_task_checkpoint("preserved", "output_collected", {"sha": "c"})
    with sqlite3.connect(path, factory=ClosingConnection) as conn:
        conn.execute("UPDATE schema_metadata SET value=? WHERE key='schema_version'", (version,))
        for table in ("task_events", "task_runtime", "dispatcher_lease", "control_commands"):
            conn.execute(f"DROP TABLE {table}")
    upgraded = SQLiteTaskStore(path)
    assert upgraded.get_task("preserved").comfyui_prompt_id == "external-prompt"
    assert upgraded.put_task_checkpoint(checkpoint) == checkpoint
    with sqlite3.connect(path, factory=ClosingConnection) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT value FROM schema_metadata").fetchone()[0] == "8"
    with sqlite3.connect(path.with_suffix(".pre-v8.bak"), factory=ClosingConnection) as conn:
        assert conn.execute("SELECT value FROM schema_metadata").fetchone()[0] == version
        prompt = conn.execute("SELECT comfyui_prompt_id FROM tasks").fetchone()[0]
        assert prompt == "external-prompt"


def test_future_schema_refused_without_modification(tmp_path):
    path = tmp_path / "future.db"
    seed_task(SQLiteTaskStore(path))
    with sqlite3.connect(path, factory=ClosingConnection) as conn:
        conn.execute("UPDATE schema_metadata SET value='999'")
    before = path.read_bytes()
    with pytest.raises(TaskStoreError, match="unsupported"):
        SQLiteTaskStore(path)
    assert path.read_bytes() == before


def test_checkpoint_replay_preserves_timestamp_but_rejects_changed_result(tmp_path):
    store = SQLiteTaskStore(tmp_path / "checkpoint.db")
    seed_task(store)
    first = TaskCheckpoint(
        checkpoint_id="output",
        task_id="preserved",
        sequence=1,
        phase="output_collected",
        payload={"sha256": "c" * 64},
        created_at=datetime.now(UTC),
    )
    store.put_task_checkpoint(first)
    replay = first.model_copy(update={"created_at": first.created_at + timedelta(seconds=20)})
    assert store.put_task_checkpoint(replay) == first
    with pytest.raises(StoreConflictError, match="immutable"):
        store.put_task_checkpoint(replay.model_copy(update={"payload": {"sha256": "d" * 64}}))


def test_clear_history_cannot_erase_unconfirmed_external_cancellation(tmp_path):
    store = SQLiteTaskStore(tmp_path / "cancel.db")
    seed_task(store)
    store.request_task_cancellation("preserved")
    with pytest.raises(StoreConflictError):
        store.clear_project_tasks("project")
    assert store.get_task("preserved").comfyui_prompt_id == "external-prompt"


@pytest.mark.parametrize("fail", [False, True])
def test_connection_context_closes_handle_and_keeps_transaction_semantics(tmp_path, fail):
    path = tmp_path / "handles.db"
    with sqlite3.connect(path, factory=ClosingConnection) as setup:
        setup.execute("CREATE TABLE example(value INTEGER)")
    connection = sqlite3.connect(path, factory=ClosingConnection)
    try:
        with connection:
            connection.execute("INSERT INTO example VALUES(1)")
            if fail:
                raise RuntimeError("fault")
    except RuntimeError:
        pass
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with sqlite3.connect(path, factory=ClosingConnection) as probe:
        assert probe.execute("SELECT COUNT(*) FROM example").fetchone()[0] == (0 if fail else 1)
    path.unlink()  # Windows refuses this while any SQLite handle is open.
