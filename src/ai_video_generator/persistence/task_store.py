from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_video_generator.domain.artifacts import ConditioningArtifact
from ai_video_generator.domain.generation import (
    GenerationBatch,
    ReworkMarker,
    SegmentGenerationVersion,
)
from ai_video_generator.domain.h3_workflow import H3WorkflowProfile
from ai_video_generator.domain.orchestration import (
    BatchRun,
    BatchState,
    DecisionLedger,
    ExecutionMode,
    HarnessBundle,
    HarnessRevision,
    ModelResidencyEvent,
    PerformanceEstimate,
    PerformanceSignature,
    ProjectMemoryEvent,
    ProjectRunState,
    ProjectWorkspaceRevision,
    ReviewDeadline,
    ReviewMode,
    ReviewPolicy,
    TaskCheckpoint,
)
from ai_video_generator.domain.project import ProjectSpec
from ai_video_generator.domain.prompting import (
    AssetPlan,
    H3PromptRevision,
    ImagePromptRevision,
    ProjectAsset,
    PromptSet,
    StageGenerationCheckpoint,
)
from ai_video_generator.domain.review import ReviewDecision, ReworkRequest
from ai_video_generator.domain.tasks import ExecutionTarget, TaskKind, TaskSpec, TaskState
from ai_video_generator.domain.workflow import WorkflowTemplate
from ai_video_generator.domain.workload import (
    TaskWorkloadManifest,
    WorkloadManifestRecord,
    canonical_workload_manifest_bytes,
)
from ai_video_generator.services.remote import (
    TaskResultReceipt,
    TaskResultReport,
    WorkerResultStatus,
)


class TaskStoreError(RuntimeError):
    pass


class TaskNotFoundError(TaskStoreError):
    pass


class InvalidTaskTransitionError(TaskStoreError):
    pass


class IdempotencyConflictError(TaskStoreError):
    pass


class StoreConflictError(TaskStoreError):
    pass


class LeaseError(TaskStoreError):
    pass


@dataclass(frozen=True, slots=True)
class ComfyPromptRecord:
    task_id: str
    prompt_id: str
    client_id: str | None
    submitted_at: datetime
    reconciled_at: datetime | None
    status: str | None


_ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.BLOCKED: frozenset({TaskState.READY, TaskState.CANCELLED, TaskState.STALE}),
    TaskState.READY: frozenset(
        {
            TaskState.QUEUED,
            TaskState.PAUSED,
            TaskState.CANCELLED,
            TaskState.STALE,
        }
    ),
    TaskState.QUEUED: frozenset(
        {TaskState.READY, TaskState.PAUSED, TaskState.RUNNING, TaskState.CANCELLED, TaskState.STALE}
    ),
    TaskState.PAUSED: frozenset({TaskState.READY, TaskState.CANCELLED, TaskState.STALE}),
    TaskState.RUNNING: frozenset(
        {
            TaskState.READY,
            TaskState.NEEDS_REVIEW,
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.STALE,
        }
    ),
    TaskState.NEEDS_REVIEW: frozenset(
        {
            TaskState.READY,
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.STALE,
        }
    ),
    TaskState.SUCCEEDED: frozenset({TaskState.STALE}),
    TaskState.FAILED: frozenset({TaskState.READY, TaskState.CANCELLED, TaskState.STALE}),
    TaskState.CANCELLED: frozenset(),
    TaskState.STALE: frozenset({TaskState.READY, TaskState.CANCELLED}),
}


class SQLiteTaskStore:
    """Durable, local-first task store backed by one SQLite database file."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        if str(database_path) == ":memory:":
            raise ValueError("SQLiteTaskStore requires a file-backed database")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO schema_metadata(key, value) VALUES ('schema_version', '1');

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    workload_manifest_sha256 TEXT,
                    execution_target TEXT NOT NULL,
                    worker_id TEXT,
                    affinity_key TEXT,
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_expires_at REAL,
                    comfyui_prompt_id TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    schema_version TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(project_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_claim
                    ON tasks(state, execution_target, attempt, created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_lease
                    ON tasks(state, lease_expires_at);

                CREATE TABLE IF NOT EXISTS project_revisions (
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(project_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_project_revisions_latest
                    ON project_revisions(project_id, revision DESC);

                CREATE TABLE IF NOT EXISTS task_reviews (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    decision TEXT NOT NULL CHECK(decision IN ('accepted', 'rejected')),
                    feedback TEXT,
                    reviewed_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_decisions (
                    decision_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    project_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_review_decisions_task
                    ON review_decisions(task_id, created_at);

                CREATE TABLE IF NOT EXISTS rework_requests (
                    request_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    source_h3_task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    review_task_id TEXT REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    replacement_task_id TEXT REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rework_requests_segment
                    ON rework_requests(project_id, segment_id, created_at);

                CREATE TABLE IF NOT EXISTS workload_manifests (
                    sha256 TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_result_reports (
                    report_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    worker_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    resulting_state TEXT NOT NULL,
                    accepted_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_results_task
                    ON task_result_reports(task_id);

                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    dependency_task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    PRIMARY KEY(task_id, dependency_task_id),
                    CHECK(task_id <> dependency_task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_dependencies_parent
                    ON task_dependencies(dependency_task_id);

                CREATE TABLE IF NOT EXISTS comfyui_prompts (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
                    prompt_id TEXT NOT NULL UNIQUE,
                    client_id TEXT,
                    submitted_at REAL NOT NULL,
                    reconciled_at REAL,
                    status TEXT
                );

                CREATE TABLE IF NOT EXISTS workflow_revisions (
                    template_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    workflow_sha256 TEXT NOT NULL,
                    node_schema_sha256 TEXT NOT NULL,
                    approval TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(template_id, revision)
                );

                CREATE TABLE IF NOT EXISTS h3_workflow_profile_revisions (
                    profile_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    workflow_sha256 TEXT NOT NULL,
                    node_schema_sha256 TEXT NOT NULL,
                    approval TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(profile_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_h3_profiles_list
                    ON h3_workflow_profile_revisions(profile_id, revision);

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    producer_task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_fingerprint ON artifacts(fingerprint);
                CREATE INDEX IF NOT EXISTS idx_artifacts_producer ON artifacts(producer_task_id);

                CREATE TABLE IF NOT EXISTS project_workspace_revisions (
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(project_id, revision)
                );
                CREATE TABLE IF NOT EXISTS project_run_states (
                    project_id TEXT PRIMARY KEY,
                    execution_mode TEXT NOT NULL DEFAULT 'guided',
                    paused INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_memory_events (
                    event_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_project_time
                    ON project_memory_events(project_id, created_at DESC);
                CREATE VIRTUAL TABLE IF NOT EXISTS project_memory_fts USING fts5(
                    event_id UNINDEXED, project_id UNINDEXED, content, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS decision_ledger (
                    decision_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    decision_key TEXT NOT NULL,
                    locked INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_decisions_project_key
                    ON decision_ledger(project_id, decision_key, created_at DESC);
                CREATE TABLE IF NOT EXISTS harness_bundles (
                    harness_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS harness_revisions (
                    harness_id TEXT NOT NULL
                        REFERENCES harness_bundles(harness_id) ON DELETE RESTRICT,
                    revision INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    approval TEXT NOT NULL,
                    workflow_template_id TEXT,
                    workflow_revision INTEGER,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(harness_id, revision)
                );
                CREATE TABLE IF NOT EXISTS batch_runs (
                    batch_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_deadlines (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
                    project_id TEXT NOT NULL,
                    deadline_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(task_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS model_residency_events (
                    event_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    task_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_model_residency_worker_time
                    ON model_residency_events(worker_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS performance_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature_json TEXT NOT NULL,
                    signature_key TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_performance_signature
                    ON performance_samples(signature_key, created_at);

                CREATE TABLE IF NOT EXISTS project_asset_revisions (
                    asset_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    normalized_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    blob_sha256 TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(asset_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_project_assets_project_revision
                    ON project_asset_revisions(project_id, asset_id, revision DESC);
                CREATE INDEX IF NOT EXISTS idx_project_assets_blob
                    ON project_asset_revisions(blob_sha256);

                CREATE TABLE IF NOT EXISTS asset_plan_revisions (
                    plan_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(plan_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_asset_plans_project_revision
                    ON asset_plan_revisions(project_id, plan_id, revision DESC);

                CREATE TABLE IF NOT EXISTS image_prompt_revisions (
                    prompt_revision_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    asset_plan_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(prompt_revision_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_image_prompts_project_revision
                    ON image_prompt_revisions(project_id, prompt_revision_id, revision DESC);
                CREATE INDEX IF NOT EXISTS idx_image_prompts_asset_plan
                    ON image_prompt_revisions(asset_plan_id, revision DESC);

                CREATE TABLE IF NOT EXISTS h3_prompt_revisions (
                    prompt_revision_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    segment_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    execution_ready INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(prompt_revision_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_h3_prompts_project_revision
                    ON h3_prompt_revisions(project_id, prompt_revision_id, revision DESC);
                CREATE INDEX IF NOT EXISTS idx_h3_prompts_segment
                    ON h3_prompt_revisions(project_id, segment_id, revision DESC);

                CREATE TABLE IF NOT EXISTS h3_prompt_translations (
                    prompt_revision_id TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    language TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(prompt_revision_id, prompt_sha256, language)
                );

                CREATE TABLE IF NOT EXISTS prompt_set_revisions (
                    prompt_set_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(prompt_set_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_prompt_sets_project_revision
                    ON prompt_set_revisions(project_id, prompt_set_id, revision DESC);

                CREATE TABLE IF NOT EXISTS stage_generation_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    UNIQUE(project_id, stage, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_stage_checkpoints_project_time
                    ON stage_generation_checkpoints(project_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS generation_batches (
                    batch_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    generation_number INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_generation_batches_project
                    ON generation_batches(project_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS segment_generation_versions (
                    version_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    generation_number INTEGER NOT NULL,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_segment_active_version
                    ON segment_generation_versions(project_id, segment_id)
                    WHERE state = 'active';
                CREATE INDEX IF NOT EXISTS idx_segment_versions_project
                    ON segment_generation_versions(project_id, segment_id, generation_number DESC);
                CREATE TABLE IF NOT EXISTS rework_markers (
                    marker_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    shot_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    batch_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rework_markers_project
                    ON rework_markers(project_id, state, created_at);
                """
            )
            task_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "workload_manifest_sha256" not in task_columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN workload_manifest_sha256 TEXT")
            if "affinity_key" not in task_columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN affinity_key TEXT")
            if "priority" not in task_columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
                )
            run_state_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(project_run_states)").fetchall()
            }
            if "execution_mode" not in run_state_columns:
                connection.execute(
                    "ALTER TABLE project_run_states ADD COLUMN execution_mode "
                    "TEXT NOT NULL DEFAULT 'guided'"
                )
            if "paused" not in run_state_columns:
                connection.execute(
                    "ALTER TABLE project_run_states ADD COLUMN paused INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "UPDATE schema_metadata SET value = '5' WHERE key = 'schema_version'"
            )

    def put_generation_batch(self, batch: GenerationBatch) -> GenerationBatch:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO generation_batches(
                    batch_id, project_id, state, generation_number, payload_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET state=excluded.state,
                    payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
                (
                    batch.batch_id,
                    batch.project_id,
                    batch.state.value,
                    batch.generation_number,
                    batch.model_dump_json(),
                    _timestamp(batch.created_at),
                    _timestamp(batch.updated_at),
                ),
            )
        return batch

    def list_generation_batches(self, project_id: str) -> tuple[GenerationBatch, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM generation_batches "
                "WHERE project_id=? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return tuple(GenerationBatch.model_validate_json(row[0]) for row in rows)

    def list_generation_batches_all(self) -> tuple[GenerationBatch, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM generation_batches ORDER BY created_at"
            ).fetchall()
        return tuple(GenerationBatch.model_validate_json(row[0]) for row in rows)

    def put_segment_generation_version(
        self, version: SegmentGenerationVersion
    ) -> SegmentGenerationVersion:
        with self._transaction(immediate=True) as connection:
            if version.state.value == "active":
                connection.execute(
                    """UPDATE segment_generation_versions SET state='superseded',
                       payload_json=json_set(payload_json, '$.state', 'superseded')
                       WHERE project_id=? AND segment_id=? AND state='active'""",
                    (version.project_id, version.segment_id),
                )
            connection.execute(
                """INSERT INTO segment_generation_versions(
                    version_id, project_id, segment_id, generation_number, task_id,
                    state, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id) DO UPDATE SET state=excluded.state,
                    payload_json=excluded.payload_json""",
                (
                    version.version_id,
                    version.project_id,
                    version.segment_id,
                    version.generation_number,
                    version.task_id,
                    version.state.value,
                    version.model_dump_json(),
                    _timestamp(version.created_at),
                ),
            )
        return version

    def list_segment_generation_versions(
        self, project_id: str
    ) -> tuple[SegmentGenerationVersion, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT payload_json FROM segment_generation_versions
                   WHERE project_id=? ORDER BY created_at""",
                (project_id,),
            ).fetchall()
        return tuple(SegmentGenerationVersion.model_validate_json(row[0]) for row in rows)

    def put_rework_marker(self, marker: ReworkMarker) -> ReworkMarker:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO rework_markers(
                    marker_id, project_id, shot_id, segment_id, state, batch_id,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(marker_id) DO UPDATE SET state=excluded.state,
                    batch_id=excluded.batch_id, payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at""",
                (
                    marker.marker_id,
                    marker.project_id,
                    marker.shot_id,
                    marker.segment_id,
                    marker.state.value,
                    marker.batch_id,
                    marker.model_dump_json(),
                    _timestamp(marker.created_at),
                    _timestamp(marker.updated_at),
                ),
            )
        return marker

    def list_rework_markers(self, project_id: str) -> tuple[ReworkMarker, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM rework_markers WHERE project_id=? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return tuple(ReworkMarker.model_validate_json(row[0]) for row in rows)

    def journal_mode(self) -> str:
        with self._connect() as connection:
            row = connection.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        return str(row[0]).lower()

    def add_task(self, task: TaskSpec) -> TaskSpec:
        """Insert a task or return the existing logical task for its idempotency key."""
        now = _timestamp(_utc_now())
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM tasks WHERE project_id = ? AND idempotency_key = ?",
                (task.project_id, task.idempotency_key),
            ).fetchone()
            if existing is not None:
                stored = self._task_from_row(connection, existing)
                if not _same_logical_task(stored, task):
                    raise IdempotencyConflictError(
                        "idempotency key is already associated with different task inputs"
                    )
                return stored

            same_id = connection.execute(
                "SELECT 1 FROM tasks WHERE task_id = ?", (task.task_id,)
            ).fetchone()
            if same_id is not None:
                raise StoreConflictError(f"task ID already exists: {task.task_id}")

            missing = [
                dependency
                for dependency in task.depends_on
                if connection.execute(
                    "SELECT 1 FROM tasks WHERE task_id = ?", (dependency,)
                ).fetchone()
                is None
            ]
            if missing:
                raise TaskNotFoundError(f"unknown task dependencies: {', '.join(missing)}")
            if len(task.depends_on) != len(set(task.depends_on)):
                raise StoreConflictError("task dependencies must be unique")
            if task.workload_manifest_sha256 is not None:
                manifest_row = connection.execute(
                    "SELECT payload_json FROM workload_manifests WHERE sha256 = ?",
                    (task.workload_manifest_sha256,),
                ).fetchone()
                if manifest_row is None:
                    raise StoreConflictError("task references an unknown workload manifest")
                manifest = TaskWorkloadManifest.model_validate_json(manifest_row["payload_json"])
                if manifest.task_kind != task.kind:
                    raise StoreConflictError("task kind does not match workload manifest")

            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, project_id, kind, state, idempotency_key, input_fingerprint,
                    workload_manifest_sha256, execution_target, worker_id,
                    affinity_key, priority, attempt,
                    max_attempts, lease_expires_at,
                    comfyui_prompt_id, error_code, error_message, schema_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.project_id,
                    task.kind.value,
                    task.state.value,
                    task.idempotency_key,
                    task.input_fingerprint,
                    task.workload_manifest_sha256,
                    task.execution_target.value,
                    task.worker_id,
                    task.affinity_key,
                    task.priority,
                    task.attempt,
                    task.max_attempts,
                    _optional_timestamp(task.lease_expires_at),
                    task.comfyui_prompt_id,
                    task.error_code,
                    task.error_message,
                    task.schema_version,
                    now,
                    now,
                ),
            )
            connection.executemany(
                "INSERT INTO task_dependencies(task_id, dependency_task_id) VALUES (?, ?)",
                ((task.task_id, dependency) for dependency in task.depends_on),
            )
            if task.comfyui_prompt_id is not None:
                connection.execute(
                    """
                    INSERT INTO comfyui_prompts(task_id, prompt_id, submitted_at)
                    VALUES (?, ?, ?)
                    """,
                    (task.task_id, task.comfyui_prompt_id, now),
                )
            row = self._require_task_row(connection, task.task_id)
            return self._task_from_row(connection, row)

    def get_task(self, task_id: str) -> TaskSpec:
        with self._connect() as connection:
            row = self._require_task_row(connection, task_id)
            return self._task_from_row(connection, row)

    def put_workload_manifest(self, manifest: TaskWorkloadManifest) -> WorkloadManifestRecord:
        payload = canonical_workload_manifest_bytes(manifest)
        record = WorkloadManifestRecord(
            sha256=manifest.sha256,
            byte_size=len(payload),
            manifest=manifest,
        )
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT payload_json, byte_size FROM workload_manifests WHERE sha256 = ?",
                (record.sha256,),
            ).fetchone()
            if existing is not None:
                stored = TaskWorkloadManifest.model_validate_json(existing["payload_json"])
                if stored != manifest or int(existing["byte_size"]) != record.byte_size:
                    raise StoreConflictError("workload manifest digest collision")
                return record
            connection.execute(
                """
                INSERT INTO workload_manifests(sha256, payload_json, byte_size, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.sha256,
                    payload.decode("utf-8"),
                    record.byte_size,
                    _timestamp(_utc_now()),
                ),
            )
        return record

    def get_workload_manifest(self, sha256_value: str) -> WorkloadManifestRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, byte_size FROM workload_manifests WHERE sha256 = ?",
                (sha256_value,),
            ).fetchone()
        if row is None:
            raise KeyError(sha256_value)
        return WorkloadManifestRecord(
            sha256=sha256_value,
            byte_size=int(row["byte_size"]),
            manifest=TaskWorkloadManifest.model_validate_json(row["payload_json"]),
        )

    def list_tasks(self, *, project_id: str | None = None) -> tuple[TaskSpec, ...]:
        with self._connect() as connection:
            if project_id is None:
                rows = connection.execute(
                    "SELECT * FROM tasks ORDER BY created_at, task_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at, task_id",
                    (project_id,),
                ).fetchall()
            return tuple(self._task_from_row(connection, row) for row in rows)

    def clear_project_tasks(self, project_id: str) -> int:
        """Remove execution state while retaining project content and media."""
        with self._transaction(immediate=True) as connection:
            running = connection.execute(
                "SELECT 1 FROM tasks WHERE project_id=? AND state IN ('queued','running') LIMIT 1",
                (project_id,),
            ).fetchone()
            if running is not None:
                raise StoreConflictError("项目仍有正在执行的任务，请先取消后再清除")
            self._detach_project_from_active_batches(connection, project_id)
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE project_id=?", (project_id,)
                ).fetchone()[0]
            )
            connection.execute("DELETE FROM review_decisions WHERE project_id=?", (project_id,))
            connection.execute("DELETE FROM rework_requests WHERE project_id=?", (project_id,))
            connection.execute("DELETE FROM rework_markers WHERE project_id=?", (project_id,))
            connection.execute("DELETE FROM generation_batches WHERE project_id=?", (project_id,))
            connection.execute(
                "DELETE FROM segment_generation_versions WHERE project_id=?", (project_id,)
            )
            connection.execute("DELETE FROM review_deadlines WHERE project_id=?", (project_id,))
            connection.execute(
                "DELETE FROM task_result_reports WHERE task_id IN "
                "(SELECT task_id FROM tasks WHERE project_id=?)",
                (project_id,),
            )
            connection.execute(
                "DELETE FROM task_reviews WHERE task_id IN "
                "(SELECT task_id FROM tasks WHERE project_id=?)",
                (project_id,),
            )
            connection.execute(
                "UPDATE artifacts SET producer_task_id=NULL WHERE producer_task_id IN "
                "(SELECT task_id FROM tasks WHERE project_id=?)",
                (project_id,),
            )
            connection.execute(
                "DELETE FROM task_dependencies WHERE task_id IN "
                "(SELECT task_id FROM tasks WHERE project_id=?) OR dependency_task_id IN "
                "(SELECT task_id FROM tasks WHERE project_id=?)",
                (project_id, project_id),
            )
            connection.execute("DELETE FROM tasks WHERE project_id=?", (project_id,))
            return count

    def _detach_project_from_active_batches(
        self, connection: sqlite3.Connection, project_id: str
    ) -> None:
        """Detach a project while clearing tasks, inside the same transaction."""
        rows = connection.execute(
            "SELECT batch_id, payload_json, state FROM batch_runs "
            "WHERE state IN ('draft', 'running', 'paused')"
        ).fetchall()
        now = _timestamp(_utc_now())
        for row in rows:
            batch = BatchRun.model_validate_json(row["payload_json"])
            if not any(item.project_id == project_id for item in batch.items):
                continue
            remaining = tuple(item for item in batch.items if item.project_id != project_id)
            if remaining:
                updated = batch.model_copy(update={"items": remaining, "updated_at": _utc_now()})
            else:
                # Keep one empty audit member because BatchRun requires a non-empty tuple.
                detached = next(item for item in batch.items if item.project_id == project_id)
                updated = batch.model_copy(
                    update={
                        "state": BatchState.CANCELLED,
                        "items": (detached.model_copy(update={"task_ids": ()}),),
                        "updated_at": _utc_now(),
                    }
                )
            connection.execute(
                "UPDATE batch_runs SET state=?, payload_json=?, updated_at=? WHERE batch_id=?",
                (
                    updated.state.value,
                    updated.model_dump_json(),
                    now,
                    row["batch_id"],
                ),
            )

    def delete_project(self, project_id: str) -> int:
        """Delete project records; content-addressed media files remain recoverable."""
        removed = self.clear_project_tasks(project_id)
        with self._transaction(immediate=True) as connection:
            event_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT event_id FROM project_memory_events WHERE project_id=?", (project_id,)
                ).fetchall()
            ]
            prompt_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT prompt_revision_id FROM h3_prompt_revisions WHERE project_id=?",
                    (project_id,),
                ).fetchall()
            ]
            if event_ids:
                connection.executemany(
                    "DELETE FROM project_memory_fts WHERE event_id=?",
                    ((item,) for item in event_ids),
                )
            if prompt_ids:
                connection.executemany(
                    "DELETE FROM h3_prompt_translations WHERE prompt_revision_id=?",
                    ((item,) for item in prompt_ids),
                )
            for table in (
                "asset_candidate_acceptance_claims",
                "asset_generation_candidates",
                "stage_generation_checkpoints",
                "prompt_set_revisions",
                "h3_prompt_revisions",
                "image_prompt_revisions",
                "asset_plan_revisions",
                "project_asset_revisions",
                "decision_ledger",
                "project_memory_events",
                "project_run_states",
                "project_workspace_revisions",
                "project_revisions",
            ):
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()
                if exists is None:
                    continue
                if table == "asset_candidate_acceptance_claims":
                    connection.execute(
                        "DELETE FROM asset_candidate_acceptance_claims WHERE candidate_id IN "
                        "(SELECT candidate_id FROM asset_generation_candidates WHERE project_id=?)",
                        (project_id,),
                    )
                else:
                    connection.execute(f"DELETE FROM {table} WHERE project_id=?", (project_id,))
        return removed

    def set_task_priority(self, task_id: str, priority: int) -> TaskSpec:
        if priority < -100 or priority > 100:
            raise ValueError("task priority must be between -100 and 100")
        changed_at = _utc_now()
        with self._transaction(immediate=True) as connection:
            self._require_task_row(connection, task_id)
            connection.execute(
                "UPDATE tasks SET priority = ?, updated_at = ? WHERE task_id = ?",
                (priority, _timestamp(changed_at), task_id),
            )
            return self._task_from_row(connection, self._require_task_row(connection, task_id))

    def review_task(
        self,
        task_id: str,
        *,
        accepted: bool,
        feedback: str | None = None,
        now: datetime | None = None,
    ) -> TaskSpec:
        """Atomically record a human review and apply its terminal task state."""
        decision = "accepted" if accepted else "rejected"
        changed_at = _ensure_utc(now or _utc_now())
        normalized_feedback = feedback.strip() if feedback and feedback.strip() else None
        with self._transaction(immediate=True) as connection:
            row = self._require_task_row(connection, task_id)
            existing = connection.execute(
                "SELECT decision, feedback FROM task_reviews WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if existing is not None:
                if existing["decision"] != decision or existing["feedback"] != normalized_feedback:
                    raise StoreConflictError("task already has a different review decision")
                # Clean up deadlines left by versions that did not resolve them
                # atomically with the human decision.
                connection.execute("DELETE FROM review_deadlines WHERE task_id = ?", (task_id,))
                return self._task_from_row(connection, row)

            current = TaskState(row["state"])
            if current != TaskState.NEEDS_REVIEW:
                raise InvalidTaskTransitionError(f"task {task_id} is not awaiting review")
            target = TaskState.SUCCEEDED if accepted else TaskState.FAILED
            if target == TaskState.SUCCEEDED and not self._dependencies_succeeded(
                connection, task_id
            ):
                raise InvalidTaskTransitionError("a task cannot succeed before its dependencies")

            connection.execute(
                """
                INSERT INTO task_reviews(task_id, decision, feedback, reviewed_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, decision, normalized_feedback, _timestamp(changed_at)),
            )
            connection.execute(
                """
                UPDATE tasks
                SET state = ?, lease_expires_at = NULL, error_code = ?,
                    error_message = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    target.value,
                    None if accepted else "review_rejected",
                    None if accepted else (normalized_feedback or "rejected during review"),
                    _timestamp(changed_at),
                    task_id,
                ),
            )
            connection.execute("DELETE FROM review_deadlines WHERE task_id = ?", (task_id,))
            if accepted:
                self._promote_ready_tasks(connection, changed_at)
            updated = self._require_task_row(connection, task_id)
            return self._task_from_row(connection, updated)

    def put_review_decision(self, decision: ReviewDecision) -> ReviewDecision:
        with self._transaction(immediate=True) as connection:
            task = self._require_task_row(connection, decision.task_id)
            if str(task["project_id"]) != decision.project_id:
                raise StoreConflictError("review decision project does not match task")
            existing = connection.execute(
                "SELECT payload_json FROM review_decisions WHERE decision_id = ?",
                (decision.decision_id,),
            ).fetchone()
            if existing is not None:
                stored = ReviewDecision.model_validate_json(existing["payload_json"])
                if stored != decision:
                    raise StoreConflictError("review decision ID is immutable")
                return stored
            connection.execute(
                "INSERT INTO review_decisions("
                "decision_id, task_id, project_id, segment_id, payload_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.task_id,
                    decision.project_id,
                    decision.segment_id,
                    decision.model_dump_json(),
                    _timestamp(decision.created_at),
                ),
            )
        return decision

    def list_review_decisions(self, task_id: str) -> tuple[ReviewDecision, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM review_decisions WHERE task_id = ? "
                "ORDER BY created_at, decision_id",
                (task_id,),
            ).fetchall()
        return tuple(ReviewDecision.model_validate_json(row["payload_json"]) for row in rows)

    def put_rework_request(self, request: ReworkRequest) -> ReworkRequest:
        with self._transaction(immediate=True) as connection:
            source = self._require_task_row(connection, request.source_h3_task_id)
            if str(source["project_id"]) != request.project_id:
                raise StoreConflictError("rework source project does not match request")
            if request.review_task_id:
                review = self._require_task_row(connection, request.review_task_id)
                if str(review["project_id"]) != request.project_id:
                    raise StoreConflictError("rework review project does not match request")
            existing = connection.execute(
                "SELECT payload_json FROM rework_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing is not None:
                stored = ReworkRequest.model_validate_json(existing["payload_json"])
                if stored != request:
                    raise StoreConflictError("rework request ID is immutable")
                return stored
            connection.execute(
                "INSERT INTO rework_requests("
                "request_id, project_id, segment_id, source_h3_task_id, review_task_id, "
                "replacement_task_id, payload_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request.request_id,
                    request.project_id,
                    request.segment_id,
                    request.source_h3_task_id,
                    request.review_task_id,
                    request.replacement_task_id,
                    request.model_dump_json(),
                    _timestamp(request.created_at),
                ),
            )
        return request

    def list_rework_requests(
        self, *, project_id: str, segment_id: str | None = None
    ) -> tuple[ReworkRequest, ...]:
        query = "SELECT payload_json FROM rework_requests WHERE project_id = ?"
        parameters: list[object] = [project_id]
        if segment_id is not None:
            query += " AND segment_id = ?"
            parameters.append(segment_id)
        query += " ORDER BY created_at, request_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(ReworkRequest.model_validate_json(row["payload_json"]) for row in rows)

    def transition_task(
        self,
        task_id: str,
        state: TaskState,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        now: datetime | None = None,
    ) -> TaskSpec:
        changed_at = _ensure_utc(now or _utc_now())
        with self._transaction(immediate=True) as connection:
            row = self._require_task_row(connection, task_id)
            current = TaskState(row["state"])
            if state == current:
                return self._task_from_row(connection, row)
            if state not in _ALLOWED_TRANSITIONS[current]:
                raise InvalidTaskTransitionError(
                    f"cannot transition task {task_id} from {current.value} to {state.value}"
                )
            if state == TaskState.SUCCEEDED and not self._dependencies_succeeded(
                connection, task_id
            ):
                raise InvalidTaskTransitionError("a task cannot succeed before its dependencies")
            if state == TaskState.READY:
                self._assert_can_be_ready(connection, row)
            if state == TaskState.RUNNING and int(row["attempt"]) >= int(row["max_attempts"]):
                raise InvalidTaskTransitionError("task has exhausted its attempts")

            clear_lease = state != TaskState.RUNNING
            increment_attempt = int(state == TaskState.RUNNING)
            connection.execute(
                """
                UPDATE tasks
                SET state = ?, lease_expires_at = CASE WHEN ? THEN NULL ELSE lease_expires_at END,
                    attempt = attempt + ?, error_code = ?, error_message = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    state.value,
                    int(clear_lease),
                    increment_attempt,
                    error_code,
                    error_message,
                    _timestamp(changed_at),
                    task_id,
                ),
            )
            if state == TaskState.SUCCEEDED:
                self._promote_ready_tasks(connection, changed_at)
            return self._task_from_row(connection, self._require_task_row(connection, task_id))

    def prepare_task_retry(self, task_id: str, *, now: datetime | None = None) -> TaskSpec:
        """Reset a failed task, retaining a completed job when only collection failed."""
        changed_at = _ensure_utc(now or _utc_now())
        with self._transaction(immediate=True) as connection:
            row = self._require_task_row(connection, task_id)
            current = TaskState(row["state"])
            if current not in {TaskState.FAILED, TaskState.STALE, TaskState.PAUSED}:
                raise InvalidTaskTransitionError(f"task {task_id} is not at a retryable boundary")
            if not self._dependencies_succeeded(connection, task_id):
                raise InvalidTaskTransitionError("task dependencies have not succeeded")

            # Collection is a resumable boundary: ComfyUI has already completed
            # successfully, so retrying must read that job instead of running it again.
            transport_timeout = str(row["error_message"] or "").strip() in {
                "ReadTimeout",
                "ConnectTimeout",
                "PoolTimeout",
                "WriteTimeout",
            }
            retain_comfyui_job = row["error_code"] == "local_output_collection_failed" or (
                bool(row["comfyui_prompt_id"]) and transport_timeout
            )
            attempt = min(int(row["attempt"]), int(row["max_attempts"]) - 1)
            if not retain_comfyui_job:
                connection.execute("DELETE FROM comfyui_prompts WHERE task_id = ?", (task_id,))
            connection.execute(
                """
                UPDATE tasks
                SET state = 'ready', attempt = ?, lease_expires_at = NULL,
                    comfyui_prompt_id = CASE WHEN ? THEN comfyui_prompt_id ELSE NULL END,
                    error_code = NULL, error_message = NULL,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (attempt, int(retain_comfyui_job), _timestamp(changed_at), task_id),
            )
            return self._task_from_row(connection, self._require_task_row(connection, task_id))

    def refresh_ready_tasks(self, *, now: datetime | None = None) -> tuple[str, ...]:
        changed_at = _ensure_utc(now or _utc_now())
        with self._transaction(immediate=True) as connection:
            return self._promote_ready_tasks(connection, changed_at)

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_duration: timedelta,
        execution_target: ExecutionTarget | None = None,
        resident_affinity_key: str | None = None,
        now: datetime | None = None,
    ) -> TaskSpec | None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        claimed_at = _ensure_utc(now or _utc_now())
        lease_expires_at = claimed_at + lease_duration
        with self._transaction(immediate=True) as connection:
            self._promote_ready_tasks(connection, claimed_at)
            target_clause = "" if execution_target is None else "AND execution_target = ?"
            params: list[object] = [worker_id]
            if execution_target is not None:
                params.append(execution_target.value)
            params.append(resident_affinity_key or "")
            row = connection.execute(
                f"""
                SELECT task.* FROM tasks AS task
                LEFT JOIN project_run_states AS run ON run.project_id = task.project_id
                WHERE task.state = 'queued'
                  AND task.attempt < task.max_attempts
                  AND COALESCE(run.paused, 0) = 0
                  AND (task.worker_id IS NULL OR task.worker_id = ?)
                  {target_clause}
                ORDER BY (task.priority
                          + CASE WHEN task.affinity_key = ? THEN 12 ELSE 0 END
                          + MIN(100,
                            CAST((? - task.created_at) / 300 AS INTEGER))) DESC,
                         task.created_at, task.task_id
                LIMIT 1
                """,  # noqa: S608 - the optional clause is a constant selected above
                (*params, _timestamp(claimed_at)),
            ).fetchone()
            if row is None:
                return None
            task_id = str(row["task_id"])
            updated = connection.execute(
                """
                UPDATE tasks
                SET state = ?, worker_id = ?, attempt = attempt + 1,
                    lease_expires_at = ?, error_code = NULL, error_message = NULL,
                    updated_at = ?
                WHERE task_id = ? AND state = 'queued' AND attempt < max_attempts
                """,
                (
                    TaskState.RUNNING.value,
                    worker_id,
                    _timestamp(lease_expires_at),
                    _timestamp(claimed_at),
                    task_id,
                ),
            )
            if updated.rowcount != 1:
                return None
            return self._task_from_row(connection, self._require_task_row(connection, task_id))

    def renew_lease(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> TaskSpec:
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        renewed_at = _ensure_utc(now or _utc_now())
        with self._transaction(immediate=True) as connection:
            row = self._require_task_row(connection, task_id)
            if row["state"] != TaskState.RUNNING.value or row["worker_id"] != worker_id:
                raise LeaseError("only the worker holding a running task may renew its lease")
            lease_timestamp = row["lease_expires_at"]
            if lease_timestamp is None or float(lease_timestamp) <= _timestamp(renewed_at):
                raise LeaseError("expired leases cannot be renewed")
            connection.execute(
                "UPDATE tasks SET lease_expires_at = ?, updated_at = ? WHERE task_id = ?",
                (
                    _timestamp(renewed_at + lease_duration),
                    _timestamp(renewed_at),
                    task_id,
                ),
            )
            return self._task_from_row(connection, self._require_task_row(connection, task_id))

    def active_lease_for_worker(
        self, worker_id: str, *, now: datetime | None = None
    ) -> TaskSpec | None:
        checked_at = _ensure_utc(now or _utc_now())
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM tasks
                WHERE worker_id = ? AND state = 'running'
                  AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                ORDER BY updated_at, task_id LIMIT 1
                """,
                (worker_id, _timestamp(checked_at)),
            ).fetchone()
            return None if row is None else self._task_from_row(connection, row)

    def submit_remote_result(
        self, report: TaskResultReport, *, now: datetime | None = None
    ) -> TaskResultReceipt:
        """Commit a leased Worker result exactly once by immutable report ID."""
        accepted_at = _ensure_utc(now or _utc_now())
        payload = report.model_dump_json()
        target_state = TaskState(report.status.value)
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM task_result_reports WHERE report_id = ?",
                (report.report_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload:
                    raise StoreConflictError(
                        "result report ID is already associated with different data"
                    )
                return TaskResultReceipt(
                    report_id=report.report_id,
                    task_id=str(existing["task_id"]),
                    state=str(existing["resulting_state"]),
                    accepted_at=_datetime(float(existing["accepted_at"])),
                )

            row = self._require_task_row(connection, report.task_id)
            if row["state"] != TaskState.RUNNING.value or row["worker_id"] != report.worker_id:
                raise LeaseError("only the Worker holding a running task may report its result")
            if int(row["attempt"]) != report.attempt:
                raise LeaseError("result attempt does not match the active lease")
            if target_state not in _ALLOWED_TRANSITIONS[TaskState.RUNNING]:
                raise InvalidTaskTransitionError(
                    f"invalid remote result state: {target_state.value}"
                )

            connection.execute(
                """
                UPDATE tasks SET state = ?, lease_expires_at = NULL,
                    error_code = ?, error_message = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    target_state.value,
                    report.error_code,
                    report.error_message,
                    _timestamp(accepted_at),
                    report.task_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO task_result_reports(
                    report_id, task_id, worker_id, payload_json, resulting_state, accepted_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.task_id,
                    report.worker_id,
                    payload,
                    target_state.value,
                    _timestamp(accepted_at),
                ),
            )
            if report.status == WorkerResultStatus.SUCCEEDED:
                self._promote_ready_tasks(connection, accepted_at)
            return TaskResultReceipt(
                report_id=report.report_id,
                task_id=report.task_id,
                state=target_state.value,
                accepted_at=accepted_at,
            )

    def recover_expired_leases(self, *, now: datetime | None = None) -> tuple[str, ...]:
        """Recover expired work, preserving submitted Comfy prompts for reconciliation."""
        recovered_at = _ensure_utc(now or _utc_now())
        recovered: list[str] = []
        with self._transaction(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE state = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
                ORDER BY created_at, task_id
                """,
                (_timestamp(recovered_at),),
            ).fetchall()
            for row in rows:
                task_id = str(row["task_id"])
                if row["comfyui_prompt_id"] is not None:
                    connection.execute(
                        """
                        UPDATE tasks SET lease_expires_at = NULL, updated_at = ?
                        WHERE task_id = ?
                        """,
                        (_timestamp(recovered_at), task_id),
                    )
                elif int(row["attempt"]) < int(row["max_attempts"]):
                    connection.execute(
                        """
                        UPDATE tasks
                        SET state = 'ready', lease_expires_at = NULL,
                            error_code = 'lease_expired',
                            error_message = 'worker lease expired before completion', updated_at = ?
                        WHERE task_id = ?
                        """,
                        (_timestamp(recovered_at), task_id),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE tasks
                        SET state = 'failed', lease_expires_at = NULL,
                            error_code = 'lease_expired',
                            error_message = 'worker lease expired on the final attempt',
                            updated_at = ?
                        WHERE task_id = ?
                        """,
                        (_timestamp(recovered_at), task_id),
                    )
                recovered.append(task_id)
        return tuple(recovered)

    def record_comfyui_prompt(
        self,
        task_id: str,
        prompt_id: str,
        *,
        client_id: str | None = None,
        submitted_at: datetime | None = None,
    ) -> ComfyPromptRecord:
        if not prompt_id:
            raise ValueError("prompt_id must not be empty")
        submitted = _ensure_utc(submitted_at or _utc_now())
        with self._transaction(immediate=True) as connection:
            row = self._require_task_row(connection, task_id)
            if row["state"] != TaskState.RUNNING.value:
                raise StoreConflictError("ComfyUI prompts may only be recorded for running tasks")
            existing = connection.execute(
                "SELECT * FROM comfyui_prompts WHERE task_id = ?", (task_id,)
            ).fetchone()
            if existing is not None:
                if existing["prompt_id"] != prompt_id or existing["client_id"] != client_id:
                    raise StoreConflictError("task already has a different ComfyUI prompt")
                return _prompt_from_row(existing)
            try:
                connection.execute(
                    """
                    INSERT INTO comfyui_prompts(task_id, prompt_id, client_id, submitted_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, prompt_id, client_id, _timestamp(submitted)),
                )
            except sqlite3.IntegrityError as exc:
                raise StoreConflictError(f"ComfyUI prompt is already tracked: {prompt_id}") from exc
            connection.execute(
                "UPDATE tasks SET comfyui_prompt_id = ?, updated_at = ? WHERE task_id = ?",
                (prompt_id, _timestamp(submitted), task_id),
            )
            prompt_row = connection.execute(
                "SELECT * FROM comfyui_prompts WHERE task_id = ?", (task_id,)
            ).fetchone()
            assert prompt_row is not None
            return _prompt_from_row(prompt_row)

    def reconcile_comfyui_prompt(
        self,
        task_id: str,
        status: str,
        *,
        reconciled_at: datetime | None = None,
    ) -> ComfyPromptRecord:
        if not status:
            raise ValueError("status must not be empty")
        reconciled = _ensure_utc(reconciled_at or _utc_now())
        with self._transaction(immediate=True) as connection:
            updated = connection.execute(
                """
                UPDATE comfyui_prompts SET status = ?, reconciled_at = ? WHERE task_id = ?
                """,
                (status, _timestamp(reconciled), task_id),
            )
            if updated.rowcount != 1:
                self._require_task_row(connection, task_id)
                raise StoreConflictError("task has no ComfyUI prompt to reconcile")
            row = connection.execute(
                "SELECT * FROM comfyui_prompts WHERE task_id = ?", (task_id,)
            ).fetchone()
            assert row is not None
            return _prompt_from_row(row)

    def list_unreconciled_comfyui_prompts(self) -> tuple[ComfyPromptRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.* FROM comfyui_prompts p
                JOIN tasks t ON t.task_id = p.task_id
                WHERE p.reconciled_at IS NULL AND t.state = 'running'
                ORDER BY p.submitted_at, p.task_id
                """
            ).fetchall()
            return tuple(_prompt_from_row(row) for row in rows)

    def put_workflow_revision(self, workflow: WorkflowTemplate) -> WorkflowTemplate:
        payload = workflow.model_dump_json()
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT payload_json FROM workflow_revisions
                WHERE template_id = ? AND revision = ?
                """,
                (workflow.template_id, workflow.revision),
            ).fetchone()
            if existing is not None:
                stored = WorkflowTemplate.model_validate_json(existing["payload_json"])
                if stored != workflow:
                    raise StoreConflictError("workflow revision is immutable")
                return stored
            connection.execute(
                """
                INSERT INTO workflow_revisions(
                    template_id, revision, workflow_sha256, node_schema_sha256,
                    approval, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow.template_id,
                    workflow.revision,
                    workflow.workflow_sha256,
                    workflow.node_schema_sha256,
                    workflow.approval.value,
                    payload,
                    _timestamp(_utc_now()),
                ),
            )
            return workflow

    def put_project_revision(self, project: ProjectSpec) -> ProjectSpec:
        payload = project.model_dump_json()
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT payload_json FROM project_revisions
                WHERE project_id = ? AND revision = ?
                """,
                (project.project_id, project.revision),
            ).fetchone()
            if existing is not None:
                stored = ProjectSpec.model_validate_json(existing["payload_json"])
                if stored != project:
                    raise StoreConflictError("project revision is immutable")
                return stored
            connection.execute(
                """
                INSERT INTO project_revisions(project_id, revision, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.revision,
                    payload,
                    _timestamp(_utc_now()),
                ),
            )
            return project

    def put_project_and_workspace_revision(
        self,
        project: ProjectSpec,
        workspace: ProjectWorkspaceRevision,
    ) -> ProjectWorkspaceRevision:
        """Atomically append the paired public spec and workspace revision."""
        if project.project_id != workspace.project_id or project.revision != workspace.revision:
            raise ValueError("project and workspace revision identities must match")
        project_payload = project.model_dump_json()
        workspace_payload = workspace.model_dump_json()
        with self._transaction(immediate=True) as connection:
            latest = connection.execute(
                "SELECT MAX(revision) AS revision FROM project_revisions WHERE project_id = ?",
                (project.project_id,),
            ).fetchone()
            latest_revision = int(latest["revision"]) if latest and latest["revision"] else 0
            if project.revision != latest_revision + 1:
                raise StoreConflictError(
                    "project revision conflict: expected "
                    f"{latest_revision + 1}, got {project.revision}"
                )
            connection.execute(
                "INSERT INTO project_revisions(project_id, revision, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    project.project_id,
                    project.revision,
                    project_payload,
                    _timestamp(_utc_now()),
                ),
            )
            connection.execute(
                "INSERT INTO project_workspace_revisions("
                "project_id, revision, payload_sha256, payload_json, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    workspace.project_id,
                    workspace.revision,
                    workspace.payload_sha256,
                    workspace_payload,
                    _timestamp(workspace.created_at),
                ),
            )
        return workspace

    def get_project_revision(self, project_id: str, revision: int) -> ProjectSpec:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM project_revisions
                WHERE project_id = ? AND revision = ?
                """,
                (project_id, revision),
            ).fetchone()
        if row is None:
            raise KeyError((project_id, revision))
        return ProjectSpec.model_validate_json(row["payload_json"])

    def get_latest_project_revision(self, project_id: str) -> ProjectSpec:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM project_revisions
                WHERE project_id = ? ORDER BY revision DESC LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        return ProjectSpec.model_validate_json(row["payload_json"])

    def list_project_revisions(self, project_id: str) -> tuple[ProjectSpec, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM project_revisions
                WHERE project_id = ? ORDER BY revision
                """,
                (project_id,),
            ).fetchall()
        return tuple(ProjectSpec.model_validate_json(row["payload_json"]) for row in rows)

    def list_projects(self) -> tuple[ProjectSpec, ...]:
        """Return the latest immutable revision for every project."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT revision.payload_json
                FROM project_revisions AS revision
                JOIN (
                    SELECT project_id, MAX(revision) AS revision
                    FROM project_revisions GROUP BY project_id
                ) AS latest
                  ON latest.project_id = revision.project_id
                 AND latest.revision = revision.revision
                ORDER BY revision.created_at DESC, revision.project_id
                """
            ).fetchall()
        return tuple(ProjectSpec.model_validate_json(row["payload_json"]) for row in rows)

    def list_project_summaries(self) -> tuple[dict[str, object], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT revision.payload_json, revision.created_at
                FROM project_revisions AS revision
                JOIN (
                    SELECT project_id, MAX(revision) AS revision
                    FROM project_revisions GROUP BY project_id
                ) AS latest
                  ON latest.project_id = revision.project_id
                 AND latest.revision = revision.revision
                ORDER BY revision.created_at DESC, revision.project_id
                """
            ).fetchall()
        return tuple(
            {
                **ProjectSpec.model_validate_json(row["payload_json"]).model_dump(mode="json"),
                "updated_at": _datetime(float(row["created_at"])).isoformat(),
            }
            for row in rows
        )

    def get_workflow_revision(self, template_id: str, revision: int) -> WorkflowTemplate:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM workflow_revisions
                WHERE template_id = ? AND revision = ?
                """,
                (template_id, revision),
            ).fetchone()
        if row is None:
            raise KeyError((template_id, revision))
        return WorkflowTemplate.model_validate_json(row["payload_json"])

    def list_workflow_revisions(
        self, template_id: str | None = None, *, latest_only: bool = True
    ) -> tuple[WorkflowTemplate, ...]:
        parameters: tuple[object, ...] = ()
        if latest_only:
            query = """
                SELECT stored.payload_json
                FROM workflow_revisions AS stored
                JOIN (
                    SELECT template_id, MAX(revision) AS revision
                    FROM workflow_revisions GROUP BY template_id
                ) AS latest
                  ON latest.template_id = stored.template_id
                 AND latest.revision = stored.revision
            """
            if template_id is not None:
                query += " WHERE stored.template_id = ?"
                parameters = (template_id,)
            query += " ORDER BY stored.template_id"
        else:
            query = "SELECT payload_json FROM workflow_revisions"
            if template_id is not None:
                query += " WHERE template_id = ?"
                parameters = (template_id,)
            query += " ORDER BY template_id, revision"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(WorkflowTemplate.model_validate_json(row["payload_json"]) for row in rows)

    def delete_workflow_template(self, template_id: str) -> int:
        """Delete every revision and the template-owned prompting harness."""
        with self._transaction(immediate=True) as connection:
            rows = connection.execute(
                "SELECT payload_json FROM workflow_revisions WHERE template_id=?",
                (template_id,),
            ).fetchall()
            if not rows:
                raise KeyError(template_id)
            harness_ids = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT harness_id FROM harness_revisions "
                    "WHERE workflow_template_id=?",
                    (template_id,),
                ).fetchall()
            )
            connection.execute(
                "DELETE FROM harness_revisions WHERE workflow_template_id=?", (template_id,)
            )
            for harness_id in harness_ids:
                remaining = connection.execute(
                    "SELECT 1 FROM harness_revisions WHERE harness_id=? LIMIT 1",
                    (harness_id,),
                ).fetchone()
                if remaining is None:
                    connection.execute(
                        "DELETE FROM harness_bundles WHERE harness_id=?", (harness_id,)
                    )
            connection.execute(
                "DELETE FROM workflow_revisions WHERE template_id=?", (template_id,)
            )
            return len(rows)

    def put_h3_workflow_profile_revision(
        self,
        profile: H3WorkflowProfile,
    ) -> H3WorkflowProfile:
        payload = profile.model_dump_json()
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT payload_json FROM h3_workflow_profile_revisions
                WHERE profile_id = ? AND revision = ?
                """,
                (profile.profile_id, profile.revision),
            ).fetchone()
            if existing is not None:
                stored = H3WorkflowProfile.model_validate_json(existing["payload_json"])
                if stored != profile:
                    raise StoreConflictError("H3 workflow profile revision is immutable")
                return stored
            connection.execute(
                """
                INSERT INTO h3_workflow_profile_revisions(
                    profile_id, revision, workflow_sha256, node_schema_sha256,
                    approval, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.profile_id,
                    profile.revision,
                    profile.workflow_sha256,
                    profile.node_schema_sha256,
                    profile.approval.value,
                    payload,
                    _timestamp(_utc_now()),
                ),
            )
            return profile

    def get_h3_workflow_profile_revision(
        self,
        profile_id: str,
        revision: int,
    ) -> H3WorkflowProfile:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM h3_workflow_profile_revisions
                WHERE profile_id = ? AND revision = ?
                """,
                (profile_id, revision),
            ).fetchone()
        if row is None:
            raise KeyError((profile_id, revision))
        return H3WorkflowProfile.model_validate_json(row["payload_json"])

    def list_h3_workflow_profile_revisions(
        self,
        profile_id: str | None = None,
    ) -> tuple[H3WorkflowProfile, ...]:
        query = "SELECT payload_json FROM h3_workflow_profile_revisions"
        parameters: tuple[str, ...] = ()
        if profile_id is not None:
            query += " WHERE profile_id = ?"
            parameters = (profile_id,)
        query += " ORDER BY profile_id, revision"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(H3WorkflowProfile.model_validate_json(row["payload_json"]) for row in rows)

    def put_artifact(
        self,
        artifact: ConditioningArtifact,
        *,
        producer_task_id: str | None = None,
    ) -> ConditioningArtifact:
        payload = artifact.model_dump_json()
        now = _timestamp(_utc_now())
        with self._transaction(immediate=True) as connection:
            if producer_task_id is not None:
                self._require_task_row(connection, producer_task_id)
            existing = connection.execute(
                "SELECT payload_json, producer_task_id FROM artifacts WHERE artifact_id = ?",
                (artifact.artifact_id,),
            ).fetchone()
            if existing is not None:
                stored = ConditioningArtifact.model_validate_json(existing["payload_json"])
                if stored != artifact or existing["producer_task_id"] != producer_task_id:
                    raise StoreConflictError(
                        "artifact ID is already associated with different data"
                    )
                return stored
            connection.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, fingerprint, state, producer_task_id,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.fingerprint,
                    artifact.state.value,
                    producer_task_id,
                    payload,
                    now,
                    now,
                ),
            )
            return artifact

    def get_artifact(self, artifact_id: str) -> ConditioningArtifact:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return ConditioningArtifact.model_validate_json(row["payload_json"])

    def mark_stale(
        self, task_id: str, *, propagate: bool = True, now: datetime | None = None
    ) -> tuple[str, ...]:
        changed_at = _ensure_utc(now or _utc_now())
        with self._transaction(immediate=True) as connection:
            self._require_task_row(connection, task_id)
            if propagate:
                rows = connection.execute(
                    """
                    WITH RECURSIVE descendants(task_id) AS (
                        SELECT ?
                        UNION
                        SELECT d.task_id
                        FROM task_dependencies d
                        JOIN descendants p ON d.dependency_task_id = p.task_id
                    )
                    SELECT task_id FROM descendants
                    """,
                    (task_id,),
                ).fetchall()
            else:
                rows = [(task_id,)]
            task_ids = tuple(str(row[0]) for row in rows)
            placeholders = ",".join("?" for _ in task_ids)
            connection.execute(
                f"""
                UPDATE tasks
                SET state = 'stale', lease_expires_at = NULL,
                    error_code = NULL, error_message = NULL, updated_at = ?
                WHERE task_id IN ({placeholders}) AND state <> 'cancelled'
                """,  # noqa: S608 - placeholders are generated, values remain bound parameters
                (_timestamp(changed_at), *task_ids),
            )
            return task_ids

    def put_project_workspace_revision(
        self, revision: ProjectWorkspaceRevision
    ) -> ProjectWorkspaceRevision:
        payload = revision.model_dump_json()
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT payload_json FROM project_workspace_revisions "
                "WHERE project_id = ? AND revision = ?",
                (revision.project_id, revision.revision),
            ).fetchone()
            if existing is not None:
                stored = ProjectWorkspaceRevision.model_validate_json(existing["payload_json"])
                if stored != revision:
                    raise StoreConflictError("project workspace revision is immutable")
                return stored
            connection.execute(
                """
                INSERT INTO project_workspace_revisions(
                    project_id, revision, payload_sha256, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    revision.project_id,
                    revision.revision,
                    revision.payload_sha256,
                    payload,
                    _timestamp(revision.created_at),
                ),
            )
        return revision

    def get_latest_project_workspace(self, project_id: str) -> ProjectWorkspaceRevision:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM project_workspace_revisions "
                "WHERE project_id = ? ORDER BY revision DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        return ProjectWorkspaceRevision.model_validate_json(row["payload_json"])

    def list_project_workspace_revisions(
        self, project_id: str
    ) -> tuple[ProjectWorkspaceRevision, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM project_workspace_revisions "
                "WHERE project_id = ? ORDER BY revision DESC",
                (project_id,),
            ).fetchall()
        return tuple(
            ProjectWorkspaceRevision.model_validate_json(row["payload_json"]) for row in rows
        )

    def put_project_run_state(self, state: ProjectRunState) -> ProjectRunState:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO project_run_states(
                    project_id, execution_mode, paused, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    execution_mode = excluded.execution_mode,
                    paused = excluded.paused,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    state.project_id,
                    state.execution_mode.value,
                    int(state.paused),
                    state.model_dump_json(),
                    _timestamp(state.updated_at),
                ),
            )
        return state

    def get_project_run_state(self, project_id: str) -> ProjectRunState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM project_run_states WHERE project_id = ?", (project_id,)
            ).fetchone()
            workspace_row = connection.execute(
                "SELECT payload_json FROM project_workspace_revisions "
                "WHERE project_id = ? ORDER BY revision DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        state = ProjectRunState.model_validate_json(row["payload_json"])
        if workspace_row is None:
            return state
        workspace = ProjectWorkspaceRevision.model_validate_json(
            workspace_row["payload_json"]
        ).payload
        approvals = workspace.get("stageApprovals")
        active_stage = workspace.get("activeStage")
        update: dict[str, object] = {}
        if isinstance(approvals, dict):
            update["outline_approved"] = "outline" in approvals
        if isinstance(active_stage, str) and active_stage:
            update["current_stage"] = (
                "generation" if active_stage == "review" else active_stage
            )
        return state.model_copy(update=update) if update else state

    def request_project_mode(
        self, project_id: str, mode: ExecutionMode, *, now: datetime | None = None
    ) -> ProjectRunState:
        changed_at = _ensure_utc(now or _utc_now())
        state = self.get_project_run_state(project_id)
        with self._connect() as connection:
            running = connection.execute(
                "SELECT 1 FROM tasks WHERE project_id = ? AND state = 'running' LIMIT 1",
                (project_id,),
            ).fetchone()
        if running is None:
            update = {"execution_mode": mode, "pending_mode": None}
        else:
            update = {"pending_mode": mode}
        return self.put_project_run_state(
            state.model_copy(update={**update, "updated_at": changed_at})
        )

    def apply_pending_project_mode(
        self, project_id: str, *, now: datetime | None = None
    ) -> ProjectRunState:
        state = self.get_project_run_state(project_id)
        if state.pending_mode is None:
            return state
        with self._connect() as connection:
            running = connection.execute(
                "SELECT 1 FROM tasks WHERE project_id = ? AND state = 'running' LIMIT 1",
                (project_id,),
            ).fetchone()
        if running is not None:
            return state
        return self.put_project_run_state(
            state.model_copy(
                update={
                    "execution_mode": state.pending_mode,
                    "pending_mode": None,
                    "updated_at": _ensure_utc(now or _utc_now()),
                }
            )
        )

    def add_memory_event(self, event: ProjectMemoryEvent) -> ProjectMemoryEvent:
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT payload_json FROM project_memory_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if existing is not None:
                stored = ProjectMemoryEvent.model_validate_json(existing["payload_json"])
                if stored != event:
                    raise StoreConflictError("memory event ID is immutable")
                return stored
            connection.execute(
                """
                INSERT INTO project_memory_events(
                    event_id, project_id, kind, source, role, content, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.project_id,
                    event.kind.value,
                    event.source.value,
                    event.role,
                    event.content,
                    event.model_dump_json(),
                    _timestamp(event.created_at),
                ),
            )
            connection.execute(
                "INSERT INTO project_memory_fts(event_id, project_id, content) VALUES (?, ?, ?)",
                (event.event_id, event.project_id, event.content),
            )
        return event

    def list_memory_events(
        self, project_id: str, *, query: str | None = None, limit: int = 100
    ) -> tuple[ProjectMemoryEvent, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("memory event limit must be between 1 and 1000")
        with self._connect() as connection:
            if query and query.strip():
                try:
                    rows = connection.execute(
                        """
                        SELECT event.payload_json FROM project_memory_fts AS search
                        JOIN project_memory_events AS event ON event.event_id = search.event_id
                        WHERE search.project_id = ? AND project_memory_fts MATCH ?
                        ORDER BY bm25(project_memory_fts), event.created_at DESC LIMIT ?
                        """,
                        (project_id, query.strip(), limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = connection.execute(
                        "SELECT payload_json FROM project_memory_events "
                        "WHERE project_id = ? AND content LIKE ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (project_id, f"%{query.strip()}%", limit),
                    ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT payload_json FROM project_memory_events WHERE project_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (project_id, limit),
                ).fetchall()
        return tuple(ProjectMemoryEvent.model_validate_json(row["payload_json"]) for row in rows)

    def add_decision(self, decision: DecisionLedger) -> DecisionLedger:
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT payload_json FROM decision_ledger WHERE decision_id = ?",
                (decision.decision_id,),
            ).fetchone()
            if existing is not None:
                stored = DecisionLedger.model_validate_json(existing["payload_json"])
                if stored != decision:
                    raise StoreConflictError("decision ID is immutable")
                return stored
            locked = connection.execute(
                "SELECT 1 FROM decision_ledger WHERE project_id = ? "
                "AND decision_key = ? AND locked = 1 LIMIT 1",
                (decision.project_id, decision.key),
            ).fetchone()
            if locked is not None:
                raise StoreConflictError("a locked project decision cannot be overwritten")
            connection.execute(
                "INSERT INTO decision_ledger("
                "decision_id, project_id, decision_key, locked, payload_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.project_id,
                    decision.key,
                    int(decision.locked),
                    decision.model_dump_json(),
                    _timestamp(decision.created_at),
                ),
            )
        return decision

    def list_decisions(self, project_id: str) -> tuple[DecisionLedger, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM decision_ledger WHERE project_id = ? "
                "ORDER BY created_at, decision_id",
                (project_id,),
            ).fetchall()
        return tuple(DecisionLedger.model_validate_json(row["payload_json"]) for row in rows)

    def put_harness_bundle(self, bundle: HarnessBundle) -> HarnessBundle:
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT payload_json FROM harness_bundles WHERE harness_id = ?",
                (bundle.harness_id,),
            ).fetchone()
            if existing is not None:
                stored = HarnessBundle.model_validate_json(existing["payload_json"])
                if stored != bundle:
                    raise StoreConflictError("harness bundle identity is immutable")
                return stored
            connection.execute(
                "INSERT INTO harness_bundles(harness_id, payload_json, created_at) "
                "VALUES (?, ?, ?)",
                (bundle.harness_id, bundle.model_dump_json(), _timestamp(_utc_now())),
            )
        return bundle

    def list_harness_bundles(self) -> tuple[HarnessBundle, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM harness_bundles ORDER BY harness_id"
            ).fetchall()
        return tuple(HarnessBundle.model_validate_json(row["payload_json"]) for row in rows)

    def put_harness_revision(self, revision: HarnessRevision) -> HarnessRevision:
        with self._transaction(immediate=True) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM harness_bundles WHERE harness_id = ?", (revision.harness_id,)
                ).fetchone()
                is None
            ):
                raise StoreConflictError("harness bundle must be registered first")
            existing = connection.execute(
                "SELECT payload_json FROM harness_revisions WHERE harness_id = ? AND revision = ?",
                (revision.harness_id, revision.revision),
            ).fetchone()
            if existing is not None:
                stored = HarnessRevision.model_validate_json(existing["payload_json"])
                if stored != revision:
                    raise StoreConflictError("harness revision is immutable")
                return stored
            connection.execute(
                """
                INSERT INTO harness_revisions(
                    harness_id, revision, content_sha256, approval, workflow_template_id,
                    workflow_revision, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.harness_id,
                    revision.revision,
                    revision.content_sha256,
                    revision.approval.value,
                    revision.workflow_template_id,
                    revision.workflow_revision,
                    revision.model_dump_json(),
                    _timestamp(revision.created_at),
                ),
            )
        return revision

    def list_harness_revisions(self, harness_id: str) -> tuple[HarnessRevision, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM harness_revisions WHERE harness_id = ? ORDER BY revision",
                (harness_id,),
            ).fetchall()
        return tuple(HarnessRevision.model_validate_json(row["payload_json"]) for row in rows)

    def put_batch_run(self, batch: BatchRun) -> BatchRun:
        allowed = {
            BatchState.DRAFT: {BatchState.RUNNING, BatchState.CANCELLED},
            BatchState.RUNNING: {BatchState.PAUSED, BatchState.COMPLETED, BatchState.CANCELLED},
            BatchState.PAUSED: {BatchState.RUNNING, BatchState.CANCELLED},
            BatchState.COMPLETED: set(),
            BatchState.CANCELLED: set(),
        }
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT payload_json FROM batch_runs WHERE batch_id = ?", (batch.batch_id,)
            ).fetchone()
            if existing is not None:
                stored = BatchRun.model_validate_json(existing["payload_json"])
                if batch.state != stored.state and batch.state not in allowed[stored.state]:
                    raise StoreConflictError(
                        f"cannot transition batch from {stored.state.value} to {batch.state.value}"
                    )
                if batch.created_at != stored.created_at or batch.items != stored.items:
                    raise StoreConflictError("batch membership is immutable after creation")
                connection.execute(
                    "UPDATE batch_runs SET state = ?, payload_json = ?, updated_at = ? "
                    "WHERE batch_id = ?",
                    (
                        batch.state.value,
                        batch.model_dump_json(),
                        _timestamp(batch.updated_at),
                        batch.batch_id,
                    ),
                )
            else:
                connection.execute(
                    "INSERT INTO batch_runs("
                    "batch_id, state, payload_json, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        batch.batch_id,
                        batch.state.value,
                        batch.model_dump_json(),
                        _timestamp(batch.created_at),
                        _timestamp(batch.updated_at),
                    ),
                )
        return batch

    def list_batch_runs(self) -> tuple[BatchRun, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM batch_runs ORDER BY created_at DESC"
            ).fetchall()
        return tuple(BatchRun.model_validate_json(row["payload_json"]) for row in rows)

    def expand_running_batch_project_tasks(
        self,
        *,
        batch_id: str,
        project_id: str,
        orchestration_task_id: str,
        task_ids: tuple[str, ...],
        now: datetime | None = None,
    ) -> BatchRun:
        """Atomically attach a DAG produced by a batch orchestration task."""
        changed_at = _ensure_utc(now or _utc_now())
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM batch_runs WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise StoreConflictError(f"batch {batch_id} does not exist")
            batch = BatchRun.model_validate_json(row["payload_json"])
            if batch.state != BatchState.RUNNING:
                raise StoreConflictError("only a running batch can expand compiled tasks")
            item = next((value for value in batch.items if value.project_id == project_id), None)
            if item is None or orchestration_task_id not in item.task_ids:
                raise StoreConflictError("batch project does not contain the orchestration task")
            expanded_ids = tuple(dict.fromkeys((orchestration_task_id, *task_ids)))
            if expanded_ids:
                placeholders = ",".join("?" for _ in expanded_ids)
                rows = connection.execute(
                    f"SELECT task_id, project_id FROM tasks WHERE task_id IN ({placeholders})",
                    expanded_ids,
                ).fetchall()
                found = {str(value["task_id"]): str(value["project_id"]) for value in rows}
                missing = set(expanded_ids) - set(found)
                if missing:
                    raise StoreConflictError(
                        f"compiled batch tasks do not exist: {', '.join(sorted(missing))}"
                    )
                if any(value != project_id for value in found.values()):
                    raise StoreConflictError("compiled batch tasks belong to another project")
            items = tuple(
                value.model_copy(update={"task_ids": expanded_ids})
                if value.project_id == project_id
                else value
                for value in batch.items
            )
            updated = batch.model_copy(update={"items": items, "updated_at": changed_at})
            connection.execute(
                "UPDATE batch_runs SET payload_json = ?, updated_at = ? WHERE batch_id = ?",
                (updated.model_dump_json(), _timestamp(changed_at), batch_id),
            )
        return updated

    def replace_active_batch_tasks_for_rework(
        self,
        *,
        project_id: str,
        removed_task_ids: tuple[str, ...],
        replacement_task_ids: tuple[str, ...],
        now: datetime | None = None,
    ) -> tuple[BatchRun, ...]:
        """Atomically swap a failed review branch in active batch membership."""
        removed = set(removed_task_ids)
        if not removed:
            return ()
        changed_at = _ensure_utc(now or _utc_now())
        updated_batches: list[BatchRun] = []
        with self._transaction(immediate=True) as connection:
            rows = connection.execute(
                "SELECT payload_json FROM batch_runs WHERE state IN (?, ?)",
                (BatchState.RUNNING.value, BatchState.PAUSED.value),
            ).fetchall()
            for row in rows:
                batch = BatchRun.model_validate_json(row["payload_json"])
                changed = False
                items = []
                for item in batch.items:
                    if item.project_id != project_id or not removed.intersection(item.task_ids):
                        items.append(item)
                        continue
                    task_ids = [value for value in item.task_ids if value not in removed]
                    task_ids.extend(
                        value for value in replacement_task_ids if value not in task_ids
                    )
                    items.append(item.model_copy(update={"task_ids": tuple(task_ids)}))
                    changed = True
                if not changed:
                    continue
                updated = batch.model_copy(update={"items": tuple(items), "updated_at": changed_at})
                connection.execute(
                    "UPDATE batch_runs SET payload_json = ?, updated_at = ? WHERE batch_id = ?",
                    (updated.model_dump_json(), _timestamp(changed_at), updated.batch_id),
                )
                updated_batches.append(updated)
        return tuple(updated_batches)

    def put_review_deadline(self, deadline: ReviewDeadline) -> ReviewDeadline:
        with self._transaction(immediate=True) as connection:
            task = self._require_task_row(connection, deadline.task_id)
            if task["project_id"] != deadline.project_id:
                raise StoreConflictError("review deadline project does not match task")
            connection.execute(
                """
                INSERT INTO review_deadlines(task_id, project_id, deadline_at, payload_json)
                VALUES (?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET
                project_id = excluded.project_id, deadline_at = excluded.deadline_at,
                payload_json = excluded.payload_json
                """,
                (
                    deadline.task_id,
                    deadline.project_id,
                    _timestamp(deadline.deadline_at),
                    deadline.model_dump_json(),
                ),
            )
        return deadline

    def apply_expired_review_deadlines(
        self, *, now: datetime | None = None
    ) -> tuple[ProjectRunState, ...]:
        changed_at = _ensure_utc(now or _utc_now())
        changed: list[ProjectRunState] = []
        with self._transaction(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT deadline.project_id, deadline.task_id, task.kind, task.state
                FROM review_deadlines AS deadline
                JOIN tasks AS task ON task.task_id = deadline.task_id
                WHERE deadline.deadline_at <= ?
                """,
                (_timestamp(changed_at),),
            ).fetchall()
            for row in rows:
                connection.execute(
                    "DELETE FROM review_deadlines WHERE task_id = ?", (row["task_id"],)
                )
                if (
                    row["kind"] != TaskKind.AI_REVIEW.value
                    or row["state"] != TaskState.NEEDS_REVIEW.value
                ):
                    continue
                state_row = connection.execute(
                    "SELECT payload_json FROM project_run_states WHERE project_id = ?",
                    (row["project_id"],),
                ).fetchone()
                if state_row is None:
                    continue
                state = ProjectRunState.model_validate_json(state_row["payload_json"])
                policy = state.review_policy
                if (
                    policy.configured_mode != ReviewMode.HUMAN_AI
                    or policy.effective_mode != ReviewMode.HUMAN_AI
                ):
                    continue
                policy = ReviewPolicy(
                    configured_mode=policy.configured_mode,
                    effective_mode=ReviewMode.AI_ONLY,
                    human_timeout_seconds=policy.human_timeout_seconds,
                    ai_takeover_at=changed_at,
                    takeover_reason=f"human review timed out at task {row['task_id']}",
                )
                state = state.model_copy(update={"review_policy": policy, "updated_at": changed_at})
                connection.execute(
                    """
                    UPDATE project_run_states
                    SET execution_mode = ?, paused = ?, payload_json = ?, updated_at = ?
                    WHERE project_id = ?
                    """,
                    (
                        state.execution_mode.value,
                        int(state.paused),
                        state.model_dump_json(),
                        _timestamp(state.updated_at),
                        state.project_id,
                    ),
                )
                changed.append(state)
        return tuple(changed)

    def put_task_checkpoint(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        with self._transaction(immediate=True) as connection:
            self._require_task_row(connection, checkpoint.task_id)
            existing = connection.execute(
                "SELECT payload_json FROM task_checkpoints WHERE checkpoint_id = ?",
                (checkpoint.checkpoint_id,),
            ).fetchone()
            if existing is not None:
                stored = TaskCheckpoint.model_validate_json(existing["payload_json"])
                if stored != checkpoint:
                    raise StoreConflictError("checkpoint ID is immutable")
                return stored
            connection.execute(
                "INSERT INTO task_checkpoints("
                "checkpoint_id, task_id, sequence, payload_json, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    checkpoint.checkpoint_id,
                    checkpoint.task_id,
                    checkpoint.sequence,
                    checkpoint.model_dump_json(),
                    _timestamp(checkpoint.created_at),
                ),
            )
        return checkpoint

    def list_task_checkpoints(self, task_id: str) -> tuple[TaskCheckpoint, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM task_checkpoints WHERE task_id = ? "
                "ORDER BY sequence, created_at",
                (task_id,),
            ).fetchall()
        return tuple(TaskCheckpoint.model_validate_json(row["payload_json"]) for row in rows)

    def add_model_residency_event(self, event: ModelResidencyEvent) -> ModelResidencyEvent:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO model_residency_events("
                "event_id, worker_id, model_key, action, task_id, payload_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.worker_id,
                    event.model_key,
                    event.action,
                    event.task_id,
                    event.model_dump_json(),
                    _timestamp(event.created_at),
                ),
            )
        return event

    def list_model_residency_events(
        self, *, worker_id: str | None = None, limit: int = 200
    ) -> tuple[ModelResidencyEvent, ...]:
        query = "SELECT payload_json FROM model_residency_events"
        parameters: list[object] = []
        if worker_id is not None:
            query += " WHERE worker_id = ?"
            parameters.append(worker_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(ModelResidencyEvent.model_validate_json(row["payload_json"]) for row in rows)

    def add_performance_sample(
        self, signature: PerformanceSignature, duration_seconds: float
    ) -> PerformanceEstimate:
        if duration_seconds <= 0:
            raise ValueError("performance duration must be positive")
        key = _stable_json_key(signature.model_dump(mode="json"))
        with self._transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO performance_samples("
                "signature_json, signature_key, duration_seconds, created_at"
                ") VALUES (?, ?, ?, ?)",
                (signature.model_dump_json(), key, duration_seconds, _timestamp(_utc_now())),
            )
        return self.get_performance_estimate(signature)

    def get_performance_estimate(
        self, signature: PerformanceSignature, *, projected_tasks: int | None = None
    ) -> PerformanceEstimate:
        key = _stable_json_key(signature.model_dump(mode="json"))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT duration_seconds FROM performance_samples "
                "WHERE signature_key = ? ORDER BY duration_seconds",
                (key,),
            ).fetchall()
        if not rows:
            raise KeyError(key)
        values = [float(row["duration_seconds"]) for row in rows]
        p90 = values[max(0, math.ceil(len(values) * 0.9) - 1)]
        return PerformanceEstimate(
            signature=signature,
            sample_count=len(values),
            mean_seconds=sum(values) / len(values),
            p90_seconds=p90,
            projected_project_seconds=p90 * projected_tasks if projected_tasks else None,
        )

    def put_project_asset(self, asset: ProjectAsset) -> ProjectAsset:
        """Append an immutable asset revision while keeping project names unambiguous."""
        payload = asset.model_dump_json()
        normalized_name = asset.name.casefold()
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT payload_json FROM project_asset_revisions "
                "WHERE asset_id = ? AND revision = ?",
                (asset.asset_id, asset.revision),
            ).fetchone()
            if existing is not None:
                stored = ProjectAsset.model_validate_json(existing["payload_json"])
                if stored != asset:
                    raise StoreConflictError("project asset revision is immutable")
                return stored
            latest = connection.execute(
                "SELECT project_id, MAX(revision) AS revision "
                "FROM project_asset_revisions WHERE asset_id = ?",
                (asset.asset_id,),
            ).fetchone()
            if latest is not None and latest["revision"] is not None:
                if str(latest["project_id"]) != asset.project_id:
                    raise StoreConflictError("project asset ID belongs to another project")
                if asset.revision != int(latest["revision"]) + 1:
                    raise StoreConflictError("project asset revisions must be consecutive")
            elif asset.revision != 1:
                raise StoreConflictError("the first project asset revision must be 1")
            collision = connection.execute(
                """
                WITH latest AS (
                    SELECT asset_id, MAX(revision) AS revision
                    FROM project_asset_revisions
                    WHERE project_id = ? GROUP BY asset_id
                )
                SELECT revision.asset_id
                FROM project_asset_revisions AS revision
                JOIN latest
                  ON latest.asset_id = revision.asset_id
                 AND latest.revision = revision.revision
                WHERE revision.normalized_name = ?
                  AND revision.state <> 'retired'
                  AND revision.asset_id <> ?
                LIMIT 1
                """,
                (asset.project_id, normalized_name, asset.asset_id),
            ).fetchone()
            if collision is not None:
                raise StoreConflictError("project asset name is already in use")
            connection.execute(
                """
                INSERT INTO project_asset_revisions(
                    asset_id, project_id, revision, normalized_name, state,
                    blob_sha256, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset.asset_id,
                    asset.project_id,
                    asset.revision,
                    normalized_name,
                    asset.state.value,
                    asset.sha256,
                    payload,
                    _timestamp(asset.created_at),
                ),
            )
        return asset

    def get_project_asset(self, asset_id: str, revision: int | None = None) -> ProjectAsset:
        query = "SELECT payload_json FROM project_asset_revisions WHERE asset_id = ?"
        parameters: tuple[object, ...] = (asset_id,)
        if revision is None:
            query += " ORDER BY revision DESC LIMIT 1"
        else:
            query += " AND revision = ?"
            parameters += (revision,)
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise KeyError((asset_id, revision) if revision is not None else asset_id)
        return ProjectAsset.model_validate_json(row["payload_json"])

    def list_project_assets(
        self,
        project_id: str,
        *,
        include_history: bool = False,
        include_retired: bool = False,
    ) -> tuple[ProjectAsset, ...]:
        if include_history:
            query = "SELECT payload_json FROM project_asset_revisions WHERE project_id = ?"
            parameters: list[object] = [project_id]
            if not include_retired:
                query += " AND state <> 'retired'"
            query += " ORDER BY asset_id, revision"
        else:
            query = """
                WITH latest AS (
                    SELECT asset_id, MAX(revision) AS revision
                    FROM project_asset_revisions
                    WHERE project_id = ? GROUP BY asset_id
                )
                SELECT stored.payload_json
                FROM project_asset_revisions AS stored
                JOIN latest
                  ON latest.asset_id = stored.asset_id
                 AND latest.revision = stored.revision
            """
            parameters = [project_id]
            if not include_retired:
                query += " WHERE stored.state <> 'retired'"
            query += " ORDER BY stored.normalized_name, stored.asset_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(ProjectAsset.model_validate_json(row["payload_json"]) for row in rows)

    def put_asset_plan(self, plan: AssetPlan) -> AssetPlan:
        return self._put_immutable_revision(
            table="asset_plan_revisions",
            identity_column="plan_id",
            identity=plan.plan_id,
            project_id=plan.project_id,
            revision=plan.revision,
            payload_json=plan.model_dump_json(),
            created_at=plan.created_at,
            extra_columns=("state",),
            extra_values=(plan.state.value,),
            model=AssetPlan,
            conflict_label="asset plan",
        )

    def list_asset_plans(
        self, project_id: str, *, include_history: bool = False
    ) -> tuple[AssetPlan, ...]:
        return self._list_revisions(
            table="asset_plan_revisions",
            identity_column="plan_id",
            project_id=project_id,
            include_history=include_history,
            model=AssetPlan,
        )

    def get_asset_plan(self, plan_id: str, revision: int | None = None) -> AssetPlan:
        return self._get_immutable_revision(
            table="asset_plan_revisions",
            identity_column="plan_id",
            identity=plan_id,
            revision=revision,
            model=AssetPlan,
        )

    def put_image_prompt_revision(self, revision: ImagePromptRevision) -> ImagePromptRevision:
        return self._put_immutable_revision(
            table="image_prompt_revisions",
            identity_column="prompt_revision_id",
            identity=revision.prompt_revision_id,
            project_id=revision.project_id,
            revision=revision.revision,
            payload_json=revision.model_dump_json(),
            created_at=revision.created_at,
            extra_columns=("asset_plan_id", "state"),
            extra_values=(revision.asset_plan_id, revision.state.value),
            model=ImagePromptRevision,
            conflict_label="image prompt",
        )

    def list_image_prompt_revisions(
        self, project_id: str, *, include_history: bool = False
    ) -> tuple[ImagePromptRevision, ...]:
        return self._list_revisions(
            table="image_prompt_revisions",
            identity_column="prompt_revision_id",
            project_id=project_id,
            include_history=include_history,
            model=ImagePromptRevision,
        )

    def get_image_prompt_revision(
        self, prompt_revision_id: str, revision: int | None = None
    ) -> ImagePromptRevision:
        return self._get_immutable_revision(
            table="image_prompt_revisions",
            identity_column="prompt_revision_id",
            identity=prompt_revision_id,
            revision=revision,
            model=ImagePromptRevision,
        )

    def put_h3_prompt_revision(self, revision: H3PromptRevision) -> H3PromptRevision:
        return self._put_immutable_revision(
            table="h3_prompt_revisions",
            identity_column="prompt_revision_id",
            identity=revision.prompt_revision_id,
            project_id=revision.project_id,
            revision=revision.revision,
            payload_json=revision.model_dump_json(),
            created_at=revision.created_at,
            extra_columns=("segment_id", "state", "execution_ready"),
            extra_values=(
                revision.segment_id,
                revision.state.value,
                int(revision.review.execution_ready),
            ),
            model=H3PromptRevision,
            conflict_label="H3 prompt",
        )

    def list_h3_prompt_revisions(
        self,
        project_id: str,
        *,
        segment_id: str | None = None,
        include_history: bool = False,
    ) -> tuple[H3PromptRevision, ...]:
        values = self._list_revisions(
            table="h3_prompt_revisions",
            identity_column="prompt_revision_id",
            project_id=project_id,
            include_history=include_history,
            model=H3PromptRevision,
        )
        if segment_id is not None:
            values = tuple(value for value in values if value.segment_id == segment_id)
        return values

    def get_h3_prompt_revision(
        self, prompt_revision_id: str, revision: int | None = None
    ) -> H3PromptRevision:
        return self._get_immutable_revision(
            table="h3_prompt_revisions",
            identity_column="prompt_revision_id",
            identity=prompt_revision_id,
            revision=revision,
            model=H3PromptRevision,
        )

    def get_h3_prompt_translation(
        self, prompt_revision_id: str, prompt_sha256: str, language: str = "zh-CN"
    ) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT translation FROM h3_prompt_translations "
                "WHERE prompt_revision_id = ? AND prompt_sha256 = ? AND language = ?",
                (prompt_revision_id, prompt_sha256, language),
            ).fetchone()
        return str(row["translation"]) if row is not None else None

    def put_h3_prompt_translation(
        self,
        prompt_revision_id: str,
        prompt_sha256: str,
        translation: str,
        language: str = "zh-CN",
    ) -> str:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO h3_prompt_translations("
                "prompt_revision_id, prompt_sha256, language, translation, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    prompt_revision_id,
                    prompt_sha256,
                    language,
                    translation,
                    _timestamp(datetime.now(UTC)),
                ),
            )
            row = connection.execute(
                "SELECT translation FROM h3_prompt_translations "
                "WHERE prompt_revision_id = ? AND prompt_sha256 = ? AND language = ?",
                (prompt_revision_id, prompt_sha256, language),
            ).fetchone()
        return str(row["translation"])

    def put_prompt_set(self, prompt_set: PromptSet) -> PromptSet:
        return self._put_immutable_revision(
            table="prompt_set_revisions",
            identity_column="prompt_set_id",
            identity=prompt_set.prompt_set_id,
            project_id=prompt_set.project_id,
            revision=prompt_set.revision,
            payload_json=prompt_set.model_dump_json(),
            created_at=prompt_set.created_at,
            extra_columns=(),
            extra_values=(),
            model=PromptSet,
            conflict_label="prompt set",
        )

    def list_prompt_sets(
        self, project_id: str, *, include_history: bool = False
    ) -> tuple[PromptSet, ...]:
        return self._list_revisions(
            table="prompt_set_revisions",
            identity_column="prompt_set_id",
            project_id=project_id,
            include_history=include_history,
            model=PromptSet,
        )

    def get_prompt_set(self, prompt_set_id: str, revision: int | None = None) -> PromptSet:
        return self._get_immutable_revision(
            table="prompt_set_revisions",
            identity_column="prompt_set_id",
            identity=prompt_set_id,
            revision=revision,
            model=PromptSet,
        )

    def put_stage_generation_checkpoint(
        self, checkpoint: StageGenerationCheckpoint
    ) -> StageGenerationCheckpoint:
        payload = checkpoint.model_dump_json()
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM stage_generation_checkpoints "
                "WHERE project_id = ? AND stage = ? AND idempotency_key = ?",
                (checkpoint.project_id, checkpoint.stage, checkpoint.idempotency_key),
            ).fetchone()
            if row is not None:
                stored = StageGenerationCheckpoint.model_validate_json(row["payload_json"])
                if stored == checkpoint:
                    return stored
                if stored.checkpoint_id != checkpoint.checkpoint_id:
                    raise StoreConflictError("stage idempotency key belongs to another checkpoint")
                if stored.state.value != "started" or checkpoint.state.value == "started":
                    raise StoreConflictError("completed stage checkpoint is immutable")
                connection.execute(
                    """
                    UPDATE stage_generation_checkpoints
                    SET state = ?, payload_json = ?, completed_at = ?
                    WHERE checkpoint_id = ?
                    """,
                    (
                        checkpoint.state.value,
                        payload,
                        _optional_timestamp(checkpoint.completed_at),
                        checkpoint.checkpoint_id,
                    ),
                )
                return checkpoint
            connection.execute(
                """
                INSERT INTO stage_generation_checkpoints(
                    checkpoint_id, project_id, stage, idempotency_key, state,
                    payload_json, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.project_id,
                    checkpoint.stage,
                    checkpoint.idempotency_key,
                    checkpoint.state.value,
                    payload,
                    _timestamp(checkpoint.created_at),
                    _optional_timestamp(checkpoint.completed_at),
                ),
            )
        return checkpoint

    def get_stage_generation_checkpoint(
        self, project_id: str, stage: str, idempotency_key: str
    ) -> StageGenerationCheckpoint:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM stage_generation_checkpoints "
                "WHERE project_id = ? AND stage = ? AND idempotency_key = ?",
                (project_id, stage, idempotency_key),
            ).fetchone()
        if row is None:
            raise KeyError((project_id, stage, idempotency_key))
        return StageGenerationCheckpoint.model_validate_json(row["payload_json"])

    def list_stage_generation_checkpoints(
        self, project_id: str, *, stage: str | None = None
    ) -> tuple[StageGenerationCheckpoint, ...]:
        query = "SELECT payload_json FROM stage_generation_checkpoints WHERE project_id = ?"
        parameters: list[object] = [project_id]
        if stage is not None:
            query += " AND stage = ?"
            parameters.append(stage)
        query += " ORDER BY created_at, checkpoint_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            StageGenerationCheckpoint.model_validate_json(row["payload_json"]) for row in rows
        )

    def _get_immutable_revision(
        self,
        *,
        table: str,
        identity_column: str,
        identity: str,
        revision: int | None,
        model: type[AssetPlan]
        | type[ImagePromptRevision]
        | type[H3PromptRevision]
        | type[PromptSet],
    ) -> AssetPlan | ImagePromptRevision | H3PromptRevision | PromptSet:
        query = (
            f"SELECT payload_json FROM {table} "  # noqa: S608
            f"WHERE {identity_column} = ?"  # noqa: S608
        )
        parameters: tuple[object, ...] = (identity,)
        if revision is None:
            query += " ORDER BY revision DESC LIMIT 1"
        else:
            query += " AND revision = ?"
            parameters += (revision,)
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise KeyError((identity, revision) if revision is not None else identity)
        return model.model_validate_json(row["payload_json"])

    def _put_immutable_revision(
        self,
        *,
        table: str,
        identity_column: str,
        identity: str,
        project_id: str,
        revision: int,
        payload_json: str,
        created_at: datetime,
        extra_columns: tuple[str, ...],
        extra_values: tuple[object, ...],
        model: type[AssetPlan]
        | type[ImagePromptRevision]
        | type[H3PromptRevision]
        | type[PromptSet],
        conflict_label: str,
    ) -> AssetPlan | ImagePromptRevision | H3PromptRevision | PromptSet:
        # Table and column names are internal constants supplied by the methods above.
        with self._transaction(immediate=True) as connection:
            existing = connection.execute(
                f"SELECT payload_json FROM {table} "  # noqa: S608
                f"WHERE {identity_column} = ? AND revision = ?",  # noqa: S608
                (identity, revision),
            ).fetchone()
            value = model.model_validate_json(payload_json)
            if existing is not None:
                stored = model.model_validate_json(existing["payload_json"])
                if stored != value:
                    raise StoreConflictError(f"{conflict_label} revision is immutable")
                return stored
            latest = connection.execute(
                f"SELECT project_id, MAX(revision) AS revision FROM {table} "  # noqa: S608
                f"WHERE {identity_column} = ?",  # noqa: S608
                (identity,),
            ).fetchone()
            if latest is not None and latest["revision"] is not None:
                if str(latest["project_id"]) != project_id:
                    raise StoreConflictError(f"{conflict_label} ID belongs to another project")
                if revision != int(latest["revision"]) + 1:
                    raise StoreConflictError(f"{conflict_label} revisions must be consecutive")
            elif revision != 1:
                raise StoreConflictError(f"the first {conflict_label} revision must be 1")
            columns = (identity_column, "project_id", "revision", *extra_columns)
            placeholders = ", ".join("?" for _ in range(len(columns) + 2))
            connection.execute(
                f"INSERT INTO {table} ({', '.join(columns)}, payload_json, created_at) "  # noqa: S608
                f"VALUES ({placeholders})",  # noqa: S608
                (
                    identity,
                    project_id,
                    revision,
                    *extra_values,
                    payload_json,
                    _timestamp(created_at),
                ),
            )
        return value

    def _list_revisions(
        self,
        *,
        table: str,
        identity_column: str,
        project_id: str,
        include_history: bool,
        model: type[AssetPlan]
        | type[ImagePromptRevision]
        | type[H3PromptRevision]
        | type[PromptSet],
    ) -> tuple:
        if include_history:
            query = (
                f"SELECT payload_json FROM {table} WHERE project_id = ? "  # noqa: S608
                f"ORDER BY {identity_column}, revision"  # noqa: S608
            )
        else:
            query = f"""
                WITH latest AS (
                    SELECT {identity_column} AS identity_value, MAX(revision) AS revision
                    FROM {table} WHERE project_id = ? GROUP BY {identity_column}
                )
                SELECT stored.payload_json FROM {table} AS stored
                JOIN latest
                  ON latest.identity_value = stored.{identity_column}
                 AND latest.revision = stored.revision
                ORDER BY stored.{identity_column}
            """  # noqa: S608
        with self._connect() as connection:
            rows = connection.execute(query, (project_id,)).fetchall()
        return tuple(model.model_validate_json(row["payload_json"]) for row in rows)

    def _require_task_row(self, connection: sqlite3.Connection, task_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise TaskNotFoundError(f"unknown task: {task_id}")
        return row

    def _task_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> TaskSpec:
        dependencies = connection.execute(
            """
            SELECT dependency_task_id FROM task_dependencies
            WHERE task_id = ? ORDER BY rowid
            """,
            (row["task_id"],),
        ).fetchall()
        return TaskSpec(
            schema_version=row["schema_version"],
            task_id=row["task_id"],
            project_id=row["project_id"],
            kind=row["kind"],
            state=row["state"],
            idempotency_key=row["idempotency_key"],
            input_fingerprint=row["input_fingerprint"],
            workload_manifest_sha256=row["workload_manifest_sha256"],
            depends_on=tuple(item["dependency_task_id"] for item in dependencies),
            execution_target=row["execution_target"],
            worker_id=row["worker_id"],
            affinity_key=row["affinity_key"],
            priority=row["priority"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            lease_expires_at=_optional_datetime(row["lease_expires_at"]),
            comfyui_prompt_id=row["comfyui_prompt_id"],
            error_code=row["error_code"],
            error_message=row["error_message"],
        )

    def _dependencies_succeeded(self, connection: sqlite3.Connection, task_id: str) -> bool:
        row = connection.execute(
            """
            SELECT COUNT(*) AS incomplete
            FROM task_dependencies d
            JOIN tasks parent ON parent.task_id = d.dependency_task_id
            WHERE d.task_id = ? AND parent.state <> 'succeeded'
            """,
            (task_id,),
        ).fetchone()
        assert row is not None
        return int(row["incomplete"]) == 0

    def _assert_can_be_ready(self, connection: sqlite3.Connection, row: sqlite3.Row) -> None:
        if int(row["attempt"]) >= int(row["max_attempts"]):
            raise InvalidTaskTransitionError("task has exhausted its attempts")
        if not self._dependencies_succeeded(connection, str(row["task_id"])):
            raise InvalidTaskTransitionError("task dependencies have not succeeded")

    def _promote_ready_tasks(
        self, connection: sqlite3.Connection, changed_at: datetime
    ) -> tuple[str, ...]:
        rows = connection.execute(
            """
            SELECT task.task_id
            FROM tasks task
            WHERE task.state = 'blocked'
              AND task.attempt < task.max_attempts
              AND NOT EXISTS (
                  SELECT 1
                  FROM task_dependencies d
                  JOIN tasks parent ON parent.task_id = d.dependency_task_id
                  WHERE d.task_id = task.task_id AND parent.state <> 'succeeded'
              )
            ORDER BY task.created_at, task.task_id
            """
        ).fetchall()
        task_ids = tuple(str(row["task_id"]) for row in rows)
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            connection.execute(
                f"UPDATE tasks SET state = 'ready', updated_at = ? "
                f"WHERE task_id IN ({placeholders})",  # noqa: S608
                (_timestamp(changed_at), *task_ids),
            )
        return task_ids


def _same_logical_task(left: TaskSpec, right: TaskSpec) -> bool:
    return (
        left.project_id == right.project_id
        and left.kind == right.kind
        and left.idempotency_key == right.idempotency_key
        and left.input_fingerprint == right.input_fingerprint
        and left.workload_manifest_sha256 == right.workload_manifest_sha256
        and left.depends_on == right.depends_on
        and left.execution_target == right.execution_target
        and left.affinity_key == right.affinity_key
        and left.priority == right.priority
        and left.max_attempts == right.max_attempts
    )


def _prompt_from_row(row: sqlite3.Row) -> ComfyPromptRecord:
    return ComfyPromptRecord(
        task_id=row["task_id"],
        prompt_id=row["prompt_id"],
        client_id=row["client_id"],
        submitted_at=_datetime(float(row["submitted_at"])),
        reconciled_at=_optional_datetime(row["reconciled_at"]),
        status=row["status"],
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> float:
    return _ensure_utc(value).timestamp()


def _optional_timestamp(value: datetime | None) -> float | None:
    return None if value is None else _timestamp(value)


def _datetime(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)


def _optional_datetime(value: float | None) -> datetime | None:
    return None if value is None else _datetime(float(value))


def _stable_json_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
