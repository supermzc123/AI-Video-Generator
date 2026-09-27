"""Durable local execution ownership, checkpoints and submission reconciliation."""

from __future__ import annotations

import json
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

from ai_video_generator.domain.tasks import TaskKind, TaskSpec, TaskState

execution_guard: ContextVar[tuple[str, str] | None] = ContextVar("execution_guard", default=None)
GPU_KINDS = frozenset(
    {
        TaskKind.IMAGE_GENERATION,
        TaskKind.CONDITIONING_ENCODING,
        TaskKind.H3_GENERATION,
        TaskKind.MODEL_SWITCH,
        TaskKind.SEEDVR2,
        TaskKind.RIFE,
    }
)


class ExecutionRuntimeMixin:
    def _check_dependency_cycle(self, conn, task_id: str, dependency_id: str) -> None:
        cycle = conn.execute(
            "WITH RECURSIVE ancestors(task_id) AS (SELECT ? UNION "
            "SELECT d.dependency_task_id FROM task_dependencies d JOIN ancestors a "
            "ON d.task_id=a.task_id) SELECT 1 FROM ancestors WHERE task_id=?",
            (dependency_id, task_id),
        ).fetchone()
        if cycle:
            raise ValueError("dependency would create a cycle")

    def dispatcher_health(self, *, now=None) -> dict:
        ts = (now or datetime.now(UTC)).timestamp()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM dispatcher_lease WHERE lease_key='local'").fetchone()
            return {
                "healthy": bool(row and row["expires_at"] > ts),
                "lease": dict(row) if row else None,
                "lease_seconds": self.local_lease_seconds,
                "heartbeat_seconds": 20,
            }

    def inspect_submission(self, task_id: str) -> dict:
        with self._connect() as conn:
            row = self._require_task_row(conn, task_id)
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            return {
                **(dict(runtime) if runtime else {}),
                "task_id": task_id,
                "prompt_id": row["comfyui_prompt_id"],
                "state": row["state"],
                "attempt": row["attempt"],
                "lease_expires_at": row["lease_expires_at"],
            }

    def request_task_cancellation(self, task_id: str, *, now=None) -> TaskSpec:
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            if row["state"] in {"succeeded", "cancelled", "stale"}:
                return self._task_from_row(conn, row)
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            uncertain = row["state"] in {"running", "recovering", "cancelling", "needs_attention"}
            uncertain = uncertain or bool(
                row["comfyui_prompt_id"]
                or (
                    runtime
                    and runtime["submission_token"]
                    and runtime["submission_state"] != "stopped"
                )
            )
            state = "cancelling" if uncertain else "cancelled"
            conn.execute(
                "UPDATE tasks SET state=?,updated_at=?,lease_expires_at="
                "CASE WHEN ?='cancelled' THEN NULL ELSE lease_expires_at END WHERE task_id=?",
                (state, ts, state, task_id),
            )
            self._execution_event(conn, task_id, "cancel_requested", {"state": state}, ts)
            return self._task_from_row(conn, self._require_task_row(conn, task_id))

    def mark_reconciliation_outcome(
        self,
        task_id: str,
        outcome: str,
        *,
        expected_attempt_id: str | None,
        expected_submission_token: str | None,
        prompt_id: str | None = None,
        evidence: str = "",
        now=None,
    ) -> TaskSpec:
        """CAS reconciliation; absent/unknown never authorizes another submission."""
        if outcome not in {"running", "completed", "unknown", "stopped"}:
            raise ValueError("unsupported reconciliation outcome")
        if outcome == "stopped" and not evidence.strip():
            raise ValueError("confirmed stop requires evidence")
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            identity = dict(runtime) if runtime else {}
            if (
                identity.get("attempt_id") != expected_attempt_id
                or identity.get("submission_token") != expected_submission_token
            ):
                raise ValueError("reconciliation identity has been superseded")
            allowed_failed = row["state"] == "failed" and bool(
                row["comfyui_prompt_id"] or identity.get("submission_token")
            )
            if not allowed_failed and row["state"] not in {
                "running",
                "recovering",
                "needs_attention",
                "cancelling",
            }:
                raise ValueError("task is not at a reconciliation boundary")
            if prompt_id is not None:
                if not prompt_id.strip() or row["comfyui_prompt_id"] not in {None, prompt_id}:
                    raise ValueError("conflicting external prompt identity")
                conn.execute(
                    "INSERT INTO comfyui_prompts(task_id,prompt_id,submitted_at) VALUES(?,?,?) "
                    "ON CONFLICT(task_id) DO NOTHING",
                    (task_id, prompt_id, ts),
                )
                conn.execute(
                    "UPDATE tasks SET comfyui_prompt_id=? WHERE task_id=?", (prompt_id, task_id)
                )
            cancelling = row["state"] == "cancelling"
            if outcome == "stopped":
                state = "cancelled" if cancelling else "failed"
            elif cancelling:
                state = "cancelling"
            elif outcome == "unknown":
                state = "needs_attention"
            else:
                state = "recovering"
            conn.execute(
                "UPDATE tasks SET state=?,lease_expires_at=NULL,updated_at=? WHERE task_id=?",
                (state, ts, task_id),
            )
            conn.execute(
                "INSERT INTO task_runtime(task_id,submission_state) VALUES(?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET submission_state=excluded.submission_state,"
                "attempt_id=NULL,owner_id=NULL,phase='reconciling'",
                (task_id, outcome),
            )
            self._execution_event(
                conn,
                task_id,
                "reconciliation",
                {
                    "outcome": outcome,
                    "prompt_id": prompt_id,
                    "evidence": evidence[:2000],
                    "submission_token": expected_submission_token,
                },
                ts,
            )
            return self._task_from_row(conn, self._require_task_row(conn, task_id))

    def confirm_task_stopped(self, task_id: str, **kwargs) -> TaskSpec:
        return self.mark_reconciliation_outcome(task_id, "stopped", **kwargs)

    def safe_explicit_retry(self, task_id: str, *, now=None) -> TaskSpec:
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            if row["state"] not in {"failed", "needs_attention"}:
                raise ValueError("task is not retryable; cancellation cannot be revived")
            if row["attempt"] >= row["max_attempts"]:
                raise ValueError("task has exhausted its attempt budget")
            if runtime and runtime["deadline_at"] and runtime["deadline_at"] <= ts:
                raise ValueError("task absolute deadline has expired")
            if (
                row["state"] == "needs_attention"
                or row["comfyui_prompt_id"]
                or (runtime and runtime["submission_token"])
            ) and (not runtime or runtime["submission_state"] != "stopped"):
                raise ValueError("external stop must be confirmed before retry")
            self._execution_event(
                conn,
                task_id,
                "explicit_retry",
                {
                    "prompt_id": row["comfyui_prompt_id"],
                    "submission_token": runtime["submission_token"] if runtime else None,
                },
                ts,
            )
            conn.execute("DELETE FROM comfyui_prompts WHERE task_id=?", (task_id,))
            conn.execute(
                "UPDATE task_runtime SET attempt_id=NULL,owner_id=NULL,submission_token=NULL,"
                "submission_state=NULL,retry_at=NULL,resume_pending=0 WHERE task_id=?",
                (task_id,),
            )
            state = "ready" if self._dependencies_succeeded(conn, task_id) else "blocked"
            conn.execute(
                "UPDATE tasks SET state=?,comfyui_prompt_id=NULL,lease_expires_at=NULL,"
                "worker_id=CASE WHEN execution_target='local' THEN NULL ELSE worker_id END,"
                "error_code=NULL,error_message=NULL,updated_at=? WHERE task_id=?",
                (state, ts, task_id),
            )
            return self._task_from_row(conn, self._require_task_row(conn, task_id))

    def confirm_ambiguous_retry(
        self,
        task_id: str,
        *,
        expected_attempt_id: str | None,
        expected_submission_token: str | None,
        now=None,
    ) -> TaskSpec:
        """Record explicit duplicate-risk consent without asserting that old work stopped."""
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            identity = dict(runtime) if runtime else {}
            if (
                identity.get("attempt_id") != expected_attempt_id
                or identity.get("submission_token") != expected_submission_token
            ):
                raise ValueError("retry identity has been superseded")
            if row["state"] != "needs_attention":
                raise ValueError("explicit ambiguous retry requires a needs_attention task")
            if row["attempt"] >= row["max_attempts"]:
                raise ValueError("attempt budget exhausted; create a new generation explicitly")
            if identity.get("deadline_at") and identity["deadline_at"] <= ts:
                raise ValueError("absolute deadline expired; create a new generation explicitly")
            self._execution_event(
                conn,
                task_id,
                "duplicate_execution_risk_accepted",
                {
                    "attempt_id": expected_attempt_id,
                    "submission_token": expected_submission_token,
                    "prompt_id": row["comfyui_prompt_id"],
                    "worker_id": row["worker_id"],
                    "previous_submission_state": identity.get("submission_state"),
                    "external_stop_confirmed": False,
                },
                ts,
            )
            conn.execute("DELETE FROM comfyui_prompts WHERE task_id=?", (task_id,))
            conn.execute(
                "UPDATE task_runtime SET attempt_id=NULL,owner_id=NULL,submission_token=NULL,"
                "submission_state=NULL,retry_at=NULL,resume_pending=0,phase='retry_confirmed' "
                "WHERE task_id=?",
                (task_id,),
            )
            state = "ready" if self._dependencies_succeeded(conn, task_id) else "blocked"
            conn.execute(
                "UPDATE tasks SET state=?,comfyui_prompt_id=NULL,lease_expires_at=NULL,"
                "worker_id=CASE WHEN execution_target='local' THEN NULL ELSE worker_id END,"
                "error_code=NULL,error_message=NULL,updated_at=? WHERE task_id=?",
                (state, ts, task_id),
            )
            return self._task_from_row(conn, self._require_task_row(conn, task_id))

    def attach_created_task_to_batch(self, conn, task: TaskSpec) -> None:
        """Attach dynamically created work in the same insertion transaction."""
        from ai_video_generator.domain.orchestration import BatchRun

        guard = execution_guard.get()
        if not guard or guard[0] == task.task_id:
            return
        parent = self._require_task_row(conn, guard[0])
        self.check_execution_guard(conn, parent)
        if parent["kind"] != "llm_planning" or parent["project_id"] != task.project_id:
            return
        rows = conn.execute(
            "SELECT * FROM batch_runs WHERE state IN ('running','paused')"
        ).fetchall()
        for row in rows:
            batch = BatchRun.model_validate_json(row["payload_json"])
            members = []
            changed = False
            for item in batch.items:
                if item.project_id == task.project_id and guard[0] in item.task_ids:
                    item = item.model_copy(
                        update={"task_ids": tuple(dict.fromkeys((*item.task_ids, task.task_id)))}
                    )
                    changed = True
                members.append(item)
            if changed:
                updated = batch.model_copy(
                    update={"items": tuple(members), "updated_at": datetime.now(UTC)}
                )
                conn.execute(
                    "UPDATE batch_runs SET payload_json=?,updated_at=? WHERE batch_id=?",
                    (updated.model_dump_json(), updated.updated_at.timestamp(), batch.batch_id),
                )

    def initialize_execution_runtime(self) -> None:
        with self._transaction(immediate=True) as conn:
            for statement in (
                """CREATE TABLE IF NOT EXISTS task_runtime (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
                    attempt_id TEXT, owner_id TEXT, deadline_at REAL, retry_at REAL,
                    resume_pending INTEGER NOT NULL DEFAULT 0, submission_token TEXT,
                    submission_state TEXT, phase TEXT, last_activity_at REAL)""",
                """CREATE TABLE IF NOT EXISTS task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL, payload_json TEXT NOT NULL, created_at REAL NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS dispatcher_lease (
                    lease_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    expires_at REAL NOT NULL, heartbeat_at REAL NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS control_commands (
                    idempotency_key TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
                    owner_id TEXT NOT NULL, status TEXT NOT NULL,
                    response_json TEXT NOT NULL, updated_at REAL NOT NULL)""",
            ):
                conn.execute(statement)
            conn.execute("UPDATE schema_metadata SET value='8' WHERE key='schema_version'")

    def acquire_dispatcher(self, owner_id: str, *, now: datetime | None = None) -> bool:
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            row = conn.execute("SELECT * FROM dispatcher_lease WHERE lease_key='local'").fetchone()
            if row and row["owner_id"] != owner_id and row["expires_at"] > ts:
                return False
            conn.execute(
                "INSERT INTO dispatcher_lease VALUES('local',?,?,?) "
                "ON CONFLICT(lease_key) DO UPDATE SET owner_id=excluded.owner_id, "
                "expires_at=excluded.expires_at, heartbeat_at=excluded.heartbeat_at",
                (owner_id, ts + self.local_lease_seconds, ts),
            )
            return True

    def release_dispatcher(self, owner_id: str) -> None:
        with self._transaction(immediate=True) as conn:
            conn.execute("DELETE FROM dispatcher_lease WHERE owner_id=?", (owner_id,))

    def _execution_event(self, conn, task_id: str, kind: str, payload: dict, ts: float) -> None:
        conn.execute(
            "INSERT INTO task_events(task_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
            (task_id, kind, json.dumps(payload, ensure_ascii=False), ts),
        )

    def list_execution_events(self, task_id: str, *, limit: int = 100) -> tuple[dict, ...]:
        with self._connect() as conn:
            self._require_task_row(conn, task_id)
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id=? ORDER BY event_id DESC LIMIT ?",
                (task_id, max(1, min(limit, 1000))),
            ).fetchall()
        return tuple({**dict(row), "payload": json.loads(row["payload_json"])} for row in rows)

    def check_execution_guard(self, conn, row, *, now=None) -> None:
        guard = execution_guard.get()
        if not guard:
            return
        if guard[0] != row["task_id"]:
            row = self._require_task_row(conn, guard[0])
        runtime = conn.execute("SELECT * FROM task_runtime WHERE task_id=?", (guard[0],)).fetchone()
        if not runtime or runtime["attempt_id"] != guard[1]:
            raise ValueError("execution attempt has been superseded")
        if row["state"] != "running":
            raise ValueError("execution is no longer allowed to publish results")
        ts = (now or datetime.now(UTC)).timestamp()
        if row["execution_target"] == "remote":
            if (
                runtime["owner_id"] != row["worker_id"]
                or not row["lease_expires_at"]
                or row["lease_expires_at"] <= ts
            ):
                raise ValueError("execution ownership has expired")
            return
        leader = conn.execute("SELECT * FROM dispatcher_lease WHERE lease_key='local'").fetchone()
        if (
            not leader
            or leader["owner_id"] != runtime["owner_id"]
            or leader["expires_at"] <= ts
            or not row["lease_expires_at"]
            or row["lease_expires_at"] <= ts
        ):
            raise ValueError("execution ownership has expired")

    def _batch_paused(self, conn, task_id: str) -> bool:
        for row in conn.execute(
            "SELECT state,payload_json FROM batch_runs WHERE state IN ('running','paused')"
        ):
            payload = json.loads(row["payload_json"])
            for item in payload.get("items", ()):
                if task_id in item.get("task_ids", ()) and (
                    row["state"] == "paused" or item.get("paused", False)
                ):
                    return True
        return False

    def claim_local_task(self, task_id: str, owner_id: str, *, now=None) -> TaskSpec | None:
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            leader = conn.execute(
                "SELECT * FROM dispatcher_lease WHERE lease_key='local'"
            ).fetchone()
            if not leader or leader["owner_id"] != owner_id or leader["expires_at"] <= ts:
                return None
            row = self._require_task_row(conn, task_id)
            if row["execution_target"] != "local" or row["state"] not in {"queued", "recovering"}:
                return None
            resume = row["state"] == "recovering"
            paused = conn.execute(
                "SELECT paused FROM project_run_states WHERE project_id=?", (row["project_id"],)
            ).fetchone()
            if not resume and paused and paused["paused"]:
                return None
            if not self._dependencies_succeeded(conn, task_id):
                return None
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            continuing = resume or bool(runtime and runtime["resume_pending"])
            observing = resume and bool(
                row["comfyui_prompt_id"] or (runtime and runtime["submission_token"])
            )
            if not observing and (
                (paused and paused["paused"]) or self._batch_paused(conn, task_id)
            ):
                return None
            if not continuing and row["attempt"] >= row["max_attempts"]:
                return None
            kind = TaskKind(row["kind"])
            gpu = kind in GPU_KINDS or (
                kind == TaskKind.WHISPER and row["workload_manifest_sha256"]
            )
            others = conn.execute(
                "SELECT * FROM tasks WHERE execution_target='local' AND task_id<>? "
                "AND state IN ('running','recovering','cancelling','needs_attention')",
                (task_id,),
            ).fetchall()
            if (
                gpu
                and not observing
                and any(
                    (
                        TaskKind(x["kind"]) in GPU_KINDS
                        or (x["kind"] == "whisper" and x["workload_manifest_sha256"])
                    )
                    and (
                        x["state"] != "needs_attention"
                        or x["comfyui_prompt_id"]
                        or x["attempt"] > 0
                        or conn.execute(
                            "SELECT 1 FROM task_runtime WHERE task_id=? "
                            "AND submission_token IS NOT NULL",
                            (x["task_id"],),
                        ).fetchone()
                    )
                    for x in others
                )
            ):
                return None
            if (
                kind in {TaskKind.LLM_PLANNING, TaskKind.AI_REVIEW}
                and sum(
                    x["state"] == "running" and x["kind"] in {"llm_planning", "ai_review"}
                    for x in others
                )
                >= self.llm_slots
            ):
                return None
            attempt_id = str(uuid.uuid4())
            deadline = runtime["deadline_at"] if runtime else None
            deadline = deadline or ts + (14400 if gpu else self.llm_timeout_seconds)
            if deadline <= ts and not observing:
                conn.execute(
                    "UPDATE tasks SET state='failed',error_code='deadline_exceeded',"
                    "lease_expires_at=NULL,updated_at=? WHERE task_id=?",
                    (ts, task_id),
                )
                return None
            conn.execute(
                "UPDATE tasks SET state='running',worker_id=?,attempt=attempt+?,"
                "lease_expires_at=?,updated_at=? WHERE task_id=?",
                (owner_id, int(not continuing), ts + self.local_lease_seconds, ts, task_id),
            )
            conn.execute(
                "INSERT INTO task_runtime(task_id,attempt_id,owner_id,deadline_at,"
                "phase,last_activity_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
                "attempt_id=excluded.attempt_id,owner_id=excluded.owner_id,"
                "deadline_at=excluded.deadline_at,resume_pending=0,phase=excluded.phase,"
                "last_activity_at=excluded.last_activity_at",
                (
                    task_id,
                    attempt_id,
                    owner_id,
                    deadline,
                    "reconciling" if observing else "executing",
                    ts,
                ),
            )
            self._execution_event(conn, task_id, "claimed", {"attempt_id": attempt_id}, ts)
            return self._task_from_row(conn, self._require_task_row(conn, task_id))

    def renew_local_attempt(self, task_id: str, attempt_id: str, owner_id: str) -> bool:
        ts = datetime.now(UTC).timestamp()
        with self._transaction(immediate=True) as conn:
            leader = conn.execute(
                "SELECT * FROM dispatcher_lease WHERE lease_key='local'"
            ).fetchone()
            if not leader or leader["owner_id"] != owner_id or leader["expires_at"] <= ts:
                return False
            row = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=? AND attempt_id=? AND owner_id=?",
                (task_id, attempt_id, owner_id),
            ).fetchone()
            if not row:
                return False
            updated = conn.execute(
                "UPDATE tasks SET lease_expires_at=? WHERE task_id=? "
                "AND state IN ('running','cancelling') AND lease_expires_at>?",
                (ts + self.local_lease_seconds, task_id, ts),
            )
            conn.execute(
                "UPDATE task_runtime SET last_activity_at=? WHERE task_id=?", (ts, task_id)
            )
            return updated.rowcount == 1

    def recover_local_attempts(self, *, now=None) -> tuple[str, ...]:
        ts = (now or datetime.now(UTC)).timestamp()
        recovered = []
        with self._transaction(immediate=True) as conn:
            rows = conn.execute(
                "SELECT t.*,r.submission_token FROM tasks t LEFT JOIN task_runtime r "
                "ON r.task_id=t.task_id WHERE t.execution_target='local' AND t.state='running' "
                "AND (t.lease_expires_at IS NULL OR t.lease_expires_at<=? OR NOT EXISTS "
                "(SELECT 1 FROM dispatcher_lease d WHERE d.lease_key='local' "
                "AND d.owner_id=r.owner_id AND d.expires_at>?))",
                (ts, ts),
            ).fetchall()
            for row in rows:
                known = bool(row["comfyui_prompt_id"] or row["submission_token"])
                state = (
                    "recovering" if known or row["kind"] == "llm_planning" else "needs_attention"
                )
                conn.execute(
                    "UPDATE task_runtime SET attempt_id=NULL,owner_id=NULL WHERE task_id=?",
                    (row["task_id"],),
                )
                conn.execute(
                    "UPDATE tasks SET state=?,lease_expires_at=NULL,"
                    "error_code='process_interrupted',"
                    "error_message=?,updated_at=? WHERE task_id=?",
                    (
                        state,
                        "控制服务已重启，正在核对执行记录"
                        if state == "recovering"
                        else "执行结果不明确，请核对后决定是否重新执行",
                        ts,
                        row["task_id"],
                    ),
                )
                self._execution_event(
                    conn, row["task_id"], "restart_recovery", {"state": state}, ts
                )
                recovered.append(row["task_id"])
        return tuple(recovered)

    def ensure_submission_intent(self, task_id: str) -> tuple[str, bool]:
        ts = datetime.now(UTC).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            self.check_execution_guard(conn, row)
            if row["state"] != "running":
                raise ValueError("submission requires a running attempt")
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            if runtime and runtime["submission_token"]:
                return runtime["submission_token"], False
            if row["comfyui_prompt_id"] or (runtime and runtime["phase"] == "reconciling"):
                raise ValueError("observation cannot submit new work")
            token = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO task_runtime(task_id,submission_token,submission_state,"
                "last_activity_at) "
                "VALUES(?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
                "submission_token=excluded.submission_token,submission_state='intent',"
                "last_activity_at=excluded.last_activity_at",
                (task_id, token, "intent", ts),
            )
            self._execution_event(conn, task_id, "submission_intent", {"token": token}, ts)
            return token, True

    def defer_orchestration(self, task_id: str, dependencies: tuple[str, ...] = ()) -> None:
        ts = datetime.now(UTC).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            self.check_execution_guard(conn, row)
            for dep in dependencies:
                parent = self._require_task_row(conn, dep)
                if parent["project_id"] != row["project_id"] or dep == task_id:
                    raise ValueError("invalid orchestration dependency")
                self._check_dependency_cycle(conn, task_id, dep)
                conn.execute(
                    "INSERT OR IGNORE INTO task_dependencies(task_id,dependency_task_id) "
                    "VALUES(?,?)",
                    (task_id, dep),
                )
            state = "ready" if self._dependencies_succeeded(conn, task_id) else "blocked"
            conn.execute(
                "UPDATE tasks SET state=?,lease_expires_at=NULL,worker_id=NULL,updated_at=? "
                "WHERE task_id=?",
                (state, ts, task_id),
            )
            conn.execute(
                "UPDATE task_runtime SET resume_pending=1,attempt_id=NULL,owner_id=NULL,"
                "phase='waiting_dependencies',deadline_at=CASE WHEN "
                "(SELECT kind FROM tasks WHERE task_id=task_runtime.task_id)='llm_planning' "
                "THEN NULL ELSE deadline_at END "
                "WHERE task_id=?",
                (task_id,),
            )
            self._execution_event(
                conn, task_id, "orchestration_yield", {"dependencies": dependencies}, ts
            )

    def schedule_task_retry(self, task_id: str, error: str, *, now=None) -> None:
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            self.check_execution_guard(conn, row)
            if row["state"] != "running":
                raise ValueError("only a running attempt can schedule retry")
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            if row["comfyui_prompt_id"] or (runtime and runtime["submission_token"]):
                state, retry = "needs_attention", None
            elif row["attempt"] >= row["max_attempts"] or (
                runtime and runtime["deadline_at"] and runtime["deadline_at"] <= ts
            ):
                state, retry = "failed", None
            else:
                state, retry = "retry_wait", ts + min(60, 2 ** max(1, row["attempt"]))
            conn.execute(
                "UPDATE tasks SET state=?,lease_expires_at=NULL,error_code='transient_failure',"
                "error_message=?,updated_at=? WHERE task_id=?",
                (state, error[:2000], ts, task_id),
            )
            conn.execute(
                "INSERT INTO task_runtime(task_id,retry_at) VALUES(?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET retry_at=excluded.retry_at,"
                "attempt_id=NULL,owner_id=NULL",
                (task_id, retry),
            )
            self._execution_event(conn, task_id, state, {"retry_at": retry}, ts)

    def finish_attempt_failure(
        self,
        task_id: str,
        *,
        expected_attempt_id: str | None,
        expected_submission_token: str | None,
        error_code: str,
        error_message: str,
        retryable: bool = False,
        stopped_evidence: str = "",
        now=None,
    ) -> TaskSpec:
        """Atomically fence a failed attempt and decide its durable recovery action."""
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            row = self._require_task_row(conn, task_id)
            runtime = conn.execute(
                "SELECT * FROM task_runtime WHERE task_id=?", (task_id,)
            ).fetchone()
            identity = dict(runtime) if runtime else {}
            if (
                not expected_attempt_id
                or identity.get("attempt_id") != expected_attempt_id
                or identity.get("submission_token") != expected_submission_token
            ):
                raise ValueError("failure attempt identity has been superseded")
            if row["state"] not in {"running", "cancelling"}:
                raise ValueError("failure can only finish an active attempt")
            external = bool(row["comfyui_prompt_id"] or identity.get("submission_token"))
            confirmed = bool(stopped_evidence.strip())
            retry_at = None
            if row["state"] == "cancelling":
                state = "cancelled" if confirmed else "cancelling"
            elif external and not confirmed:
                state = "needs_attention"
            elif (
                retryable
                and row["attempt"] < row["max_attempts"]
                and (not identity.get("deadline_at") or identity["deadline_at"] > ts)
            ):
                state = "retry_wait"
                retry_at = ts + min(60, 2 ** max(1, row["attempt"]))
                if identity.get("deadline_at") and retry_at >= identity["deadline_at"]:
                    state, retry_at = "failed", None
            else:
                state = "failed"
            self._execution_event(
                conn,
                task_id,
                "attempt_failed",
                {
                    "attempt_id": expected_attempt_id,
                    "submission_token": expected_submission_token,
                    "prompt_id": row["comfyui_prompt_id"],
                    "error_code": error_code,
                    "error_message": error_message[:2000],
                    "retryable": retryable,
                    "stopped_evidence": stopped_evidence[:2000],
                    "state": state,
                    "retry_at": retry_at,
                },
                ts,
            )
            clear_submission = state == "retry_wait"
            if clear_submission:
                conn.execute("DELETE FROM comfyui_prompts WHERE task_id=?", (task_id,))
            conn.execute(
                "UPDATE tasks SET state=?,lease_expires_at=NULL,error_code=?,error_message=?,"
                "comfyui_prompt_id=CASE WHEN ? THEN NULL ELSE comfyui_prompt_id END,"
                "updated_at=? WHERE task_id=?",
                (state, error_code, error_message[:2000], clear_submission, ts, task_id),
            )
            conn.execute(
                "UPDATE task_runtime SET attempt_id=NULL,owner_id=NULL,retry_at=?,"
                "submission_token=CASE WHEN ? THEN NULL ELSE submission_token END,"
                "submission_state=?,phase=?,last_activity_at=? WHERE task_id=?",
                (
                    retry_at,
                    clear_submission,
                    None
                    if clear_submission
                    else ("stopped" if confirmed else identity.get("submission_state")),
                    state,
                    ts,
                    task_id,
                ),
            )
            return self._task_from_row(conn, self._require_task_row(conn, task_id))

    def promote_due_retries(self, *, now=None) -> None:
        ts = (now or datetime.now(UTC)).timestamp()
        with self._transaction(immediate=True) as conn:
            rows = conn.execute(
                "SELECT t.task_id,r.deadline_at FROM tasks t "
                "JOIN task_runtime r ON r.task_id=t.task_id "
                "WHERE t.state='retry_wait' AND r.retry_at<=? AND t.attempt<t.max_attempts",
                (ts,),
            ).fetchall()
            for row in rows:
                if row["deadline_at"] and row["deadline_at"] <= ts:
                    conn.execute(
                        "UPDATE tasks SET state='failed',error_code='deadline_exceeded',"
                        "updated_at=? WHERE task_id=?",
                        (ts, row["task_id"]),
                    )
                    continue
                if self._dependencies_succeeded(conn, row["task_id"]):
                    conn.execute(
                        "UPDATE tasks SET state='queued',updated_at=? WHERE task_id=?",
                        (ts, row["task_id"]),
                    )

    def execution_metadata(self, conn, row) -> dict:
        runtime = conn.execute(
            "SELECT * FROM task_runtime WHERE task_id=?", (row["task_id"],)
        ).fetchone()
        data = dict(runtime) if runtime else {}

        def date(value):
            return datetime.fromtimestamp(value, UTC) if value is not None else None

        actions = {
            "ready": ("pause", "cancel"),
            "queued": ("pause", "cancel"),
            "blocked": ("cancel",),
            "running": ("cancel",),
            "recovering": ("reconcile", "cancel"),
            "retry_wait": ("cancel",),
            "failed": ("retry", "cancel"),
            "paused": ("resume", "cancel"),
            "needs_attention": ("reconcile", "confirm_retry", "cancel"),
            "cancelling": ("reconcile",),
        }.get(row["state"], ())
        if row["attempt"] >= row["max_attempts"] or (
            data.get("deadline_at") and data["deadline_at"] <= datetime.now(UTC).timestamp()
        ):
            actions = tuple(
                action for action in actions if action not in {"retry", "confirm_retry"}
            )
        if (
            row["state"] == "failed"
            and (row["comfyui_prompt_id"] or data.get("submission_token"))
            and data.get("submission_state") != "stopped"
        ):
            actions = ("reconcile", "cancel")
        if (
            row["state"] == "failed"
            and row["error_code"] == "local_output_collection_failed"
            and row["comfyui_prompt_id"]
        ):
            actions = ("retry", "reconcile", "cancel")
        if (
            row["state"] == "failed"
            and row["kind"] == "llm_planning"
            and row["error_code"] == "local_control_task_failed"
            and not row["comfyui_prompt_id"]
            and not data.get("submission_token")
            and "retry" not in actions
        ):
            batches = [json.loads(item[0]) for item in conn.execute(
                "SELECT payload_json FROM batch_runs WHERE state<>'cancelled'"
            ).fetchall()]
            memberships = [batch for batch in batches if any(
                row["task_id"] in item.get("task_ids", []) for item in batch.get("items", [])
            )]
            if len(memberships) == 1 and not conn.execute(
                "SELECT 1 FROM task_dependencies WHERE dependency_task_id=?", (row["task_id"],)
            ).fetchone():
                actions = (*actions, "restart")
        blocked = None
        if row["state"] == "failed" and "retry" not in actions:
            if "reconcile" in actions:
                blocked = "外部执行结果尚未确认，请先重新对账"
            elif "restart" in actions:
                blocked = "本次执行预算已结束；修复错误详情中的配置或输入后，可重新执行编排"
            elif row["attempt"] >= row["max_attempts"]:
                blocked = "此任务的执行次数已用尽；需要继续生成时请显式新建生成任务"
            else:
                blocked = "此任务的执行截止时间已过；需要继续生成时请显式新建生成任务"
        if row["state"] == "blocked":
            deps = conn.execute(
                "SELECT t.task_id,t.kind,t.state FROM task_dependencies d JOIN tasks t "
                "ON t.task_id=d.dependency_task_id WHERE d.task_id=? AND t.state<>'succeeded'",
                (row["task_id"],),
            ).fetchall()
            labels = {
                "image_generation": "参考图",
                "conditioning_encoding": "视频条件编码",
                "h3_generation": "视频生成",
                "ai_review": "审核",
            }
            blocked = "; ".join(
                f"等待{labels.get(d['kind'], d['kind'])} {d['task_id']}（{d['state']}）"
                for d in deps
            )
        if row["state"] in {"ready", "queued", "blocked", "retry_wait", "paused"}:
            project_pause = conn.execute(
                "SELECT paused FROM project_run_states WHERE project_id=?", (row["project_id"],)
            ).fetchone()
            if project_pause and project_pause["paused"]:
                blocked = "项目已暂停派发；恢复项目后继续"
            elif self._batch_paused(conn, row["task_id"]):
                blocked = "所在批次或批次成员已暂停派发；恢复相应范围后继续"
            elif row["state"] == "paused":
                blocked = "此任务已暂停，等待恢复"
        if row["state"] in {"needs_attention", "cancelling", "recovering"}:
            blocked = (
                row["error_message"]
                or {
                    "needs_attention": "执行结果需要核对；先重新对账，再决定是否重新执行",
                    "cancelling": "等待执行者确认终止；仍保留资源占用",
                    "recovering": "正在核对原执行及产物；不会重复提交",
                }[row["state"]]
            )
        checkpoints = conn.execute(
            "SELECT payload_json FROM task_checkpoints WHERE task_id=? "
            "ORDER BY sequence DESC LIMIT 1",
            (row["task_id"],),
        ).fetchone()
        checkpoint = json.loads(checkpoints[0]) if checkpoints else {}
        phase = checkpoint.get("phase") or data.get("phase") or row["state"]
        scope = checkpoint.get("payload", {}).get("scope")
        if row["kind"] == "llm_planning" and scope and phase.startswith("batch_"):
            phase = f"{phase}:{scope}"
        return {
            "attempt_id": data.get("attempt_id"),
            "created_at": date(row["created_at"]),
            "updated_at": date(row["updated_at"]),
            "last_activity_at": date(data.get("last_activity_at") or row["updated_at"]),
            "next_retry_at": date(data.get("retry_at")),
            "deadline_at": date(data.get("deadline_at")),
            "current_phase": phase,
            "blocked_reason": blocked,
            "available_actions": actions,
        }

    def defer_orchestration_if_child_pending(self, task_id: str, child_task_id: str) -> bool:
        """Yield a parent execution slot until its persisted child succeeds."""
        if task_id == child_task_id:
            raise ValueError("An orchestration task cannot depend on itself.")
        child = self.get_task(child_task_id)
        if child.state == TaskState.SUCCEEDED:
            return False
        # Failed and cancelled children remain explicit dependencies. Retrying a
        # failed child can unblock its parent; cancellation is never undone here.
        self.defer_orchestration(task_id, dependencies=(child_task_id,))
        return True
