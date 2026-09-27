"""Worker-local durable ComfyUI submission and artifact recovery journal."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from ai_video_generator.domain import TaskSpec, TaskWorkloadManifest


class RemoteExecutionJournal:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "execution-journal.sqlite3"
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS attempts ("
                "task_id TEXT NOT NULL, attempt_id TEXT NOT NULL, submission_token TEXT NOT NULL,"
                "manifest_sha256 TEXT NOT NULL, task_json TEXT NOT NULL, deadline_at TEXT NOT NULL,"
                "prompt_id TEXT, phase TEXT NOT NULL, detail TEXT, outputs_json TEXT NOT NULL,"
                "updated_at TEXT NOT NULL, PRIMARY KEY(task_id,attempt_id))"
            )

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def prepare(self, task: TaskSpec, manifest: TaskWorkloadManifest) -> dict:
        if not task.attempt_id or task.deadline_at is None:
            raise ValueError("remote execution requires an attempt identity and absolute deadline")
        if task.workload_manifest_sha256 != manifest.sha256:
            raise ValueError("task does not reference the supplied workload manifest")
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM attempts WHERE task_id=? AND attempt_id=?",
                (task.task_id, task.attempt_id),
            ).fetchone()
            if row:
                if row["manifest_sha256"] != manifest.sha256:
                    raise ValueError("one attempt cannot execute different workload inputs")
                if row["deadline_at"] != task.deadline_at.isoformat():
                    raise ValueError("reattachment cannot replace the original deadline")
            else:
                conn.execute(
                    "INSERT INTO attempts VALUES(?,?,?,?,?,?,NULL,'prepared',NULL,'[]',?)",
                    (
                        task.task_id,
                        task.attempt_id,
                        str(uuid.uuid4()),
                        manifest.sha256,
                        task.model_dump_json(),
                        task.deadline_at.isoformat(),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            conn.commit()
        return self.get(task)

    def get(self, task: TaskSpec) -> dict:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM attempts WHERE task_id=? AND attempt_id=?",
                (task.task_id, task.attempt_id),
            ).fetchone()
        if not row:
            raise ValueError("unknown journal attempt")
        return dict(row)

    def update(self, task: TaskSpec, *, phase: str, prompt_id=None, detail=None, outputs=None):
        with closing(self._connect()) as conn:
            result = conn.execute(
                "UPDATE attempts SET phase=?,prompt_id=COALESCE(?,prompt_id),detail=?,"
                "outputs_json=COALESCE(?,outputs_json),updated_at=? "
                "WHERE task_id=? AND attempt_id=?",
                (
                    phase,
                    prompt_id,
                    detail,
                    json.dumps(outputs) if outputs is not None else None,
                    datetime.now(UTC).isoformat(),
                    task.task_id,
                    task.attempt_id,
                ),
            )
            if result.rowcount != 1:
                raise ValueError("journal attempt disappeared")

    def claim_submission(self, task: TaskSpec) -> bool:
        """Exactly one local invocation may pass the before-submit boundary."""
        with closing(self._connect()) as conn:
            result = conn.execute(
                "UPDATE attempts SET phase='intent',updated_at=? WHERE task_id=? AND attempt_id=? "
                "AND phase='prepared'",
                (datetime.now(UTC).isoformat(), task.task_id, task.attempt_id),
            )
        return result.rowcount == 1

    def pending_tasks(self) -> tuple[TaskSpec, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT task_json FROM attempts WHERE phase NOT IN ('reported','fenced') "
                "ORDER BY updated_at"
            ).fetchall()
        return tuple(TaskSpec.model_validate_json(row["task_json"]) for row in rows)

    def publish(self, task: TaskSpec, node_id: str, filename: str, content: bytes, media_type: str):
        if not content:
            raise ValueError("empty output cannot be published")
        digest = hashlib.sha256(content).hexdigest()
        identity = hashlib.sha256(f"{task.task_id}\0{task.attempt_id}".encode()).hexdigest()
        node = hashlib.sha256(node_id.encode()).hexdigest()[:12]
        directory = self.root / "outputs" / identity
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{node}-{digest}-{Path(filename).name}"
        descriptor, temporary = tempfile.mkstemp(prefix=".output-", suffix=".part", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("existing output blob failed integrity verification")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {
            "path": str(target),
            "sha256": digest,
            "byte_size": len(content),
            "media_type": media_type,
        }

    def verified_outputs(self, task: TaskSpec) -> list[dict]:
        outputs = json.loads(self.get(task)["outputs_json"])
        for item in outputs:
            path = Path(item["path"]).resolve()
            if self.root not in path.parents:
                raise ValueError("journal output path escapes Worker root")
            data = path.read_bytes()
            if len(data) != item["byte_size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("journal output blob failed integrity verification")
        return outputs


async def validated_queue_ids(adapter) -> set[str]:
    """Malformed queue responses cannot be interpreted as proof of cancellation."""
    async with adapter._client() as client:
        response = await client.get("/queue")
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("ComfyUI queue response must be an object")
    ids = set()
    for key in ("queue_running", "queue_pending"):
        entries = payload.get(key)
        if not isinstance(entries, list):
            raise ValueError(f"ComfyUI {key} must be a list")
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not str(entry[1]).strip():
                raise ValueError("malformed ComfyUI queue entry")
            ids.add(str(entry[1]))
    return ids
