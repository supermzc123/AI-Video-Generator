import hashlib
import io

import pytest
from PIL import Image

from ai_video_generator.domain import ProjectSpec, TaskKind, TaskSpec, TaskState
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.persistence.execution_runtime import execution_guard
from ai_video_generator.persistence.project_assets import ProjectAssetStore, ProjectAssetStoreError


def _fixture(tmp_path):
    tasks = SQLiteTaskStore(tmp_path / "control.db")
    tasks.put_project_revision(
        ProjectSpec(project_id="p", name="Images", target_duration_seconds=4)
    )
    fingerprint = hashlib.sha256(b"image").hexdigest()
    tasks.add_task(
        TaskSpec(
            task_id="image",
            project_id="p",
            kind=TaskKind.IMAGE_GENERATION,
            state=TaskState.QUEUED,
            idempotency_key=fingerprint,
            input_fingerprint=fingerprint,
        )
    )
    assert tasks.acquire_dispatcher("owner")
    attempt = tasks.claim_local_task("image", "owner")
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(output, format="PNG")
    assets = ProjectAssetStore(tmp_path / "control.db", tmp_path / "assets")
    return tasks, assets, attempt, output.getvalue()


def test_recollecting_same_image_reuses_immutable_asset_revision(tmp_path):
    _, assets, attempt, content = _fixture(tmp_path)
    token = execution_guard.set((attempt.task_id, attempt.attempt_id))
    try:
        first = assets.add_generated(
            project_id="p", name="Reference", content=content, source_task_id="image"
        )
        second = assets.add_generated(
            project_id="p",
            name="Reference",
            content=content,
            source_task_id="image",
            replace_asset_id=first.asset_id,
        )
    finally:
        execution_guard.reset(token)
    assert second == first
    with assets._connect() as connection:
        assert connection.execute("SELECT count(*) FROM project_asset_revisions").fetchone()[0] == 1


def test_recollecting_image_candidate_does_not_duplicate_approval_choices(tmp_path):
    _, assets, attempt, content = _fixture(tmp_path)
    token = execution_guard.set((attempt.task_id, attempt.attempt_id))
    try:
        kwargs = dict(
            project_id="p",
            asset_plan_id="plan",
            source_task_id="image",
            name="Reference",
            content=content,
            current_asset_id=None,
        )
        assert assets.add_generation_candidate(**kwargs) == assets.add_generation_candidate(
            **kwargs
        )
    finally:
        execution_guard.reset(token)


def test_cancelled_attempt_cannot_publish_a_reference_asset(tmp_path):
    tasks, assets, attempt, content = _fixture(tmp_path)
    tasks.transition_task("image", TaskState.CANCELLED)
    token = execution_guard.set((attempt.task_id, attempt.attempt_id))
    try:
        with pytest.raises(ProjectAssetStoreError, match="no longer owns"):
            assets.add_generated(
                project_id="p", name="Reference", content=content, source_task_id="image"
            )
    finally:
        execution_guard.reset(token)


def test_existing_corrupt_blob_does_not_get_registered_as_valid(tmp_path):
    _, assets, attempt, content = _fixture(tmp_path)
    path = assets.blob_path(hashlib.sha256(content).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"truncated")
    token = execution_guard.set((attempt.task_id, attempt.attempt_id))
    try:
        with pytest.raises(ProjectAssetStoreError, match="integrity"):
            assets.add_generated(
                project_id="p", name="Reference", content=content, source_task_id="image"
            )
    finally:
        execution_guard.reset(token)
    with assets._connect() as connection:
        assert connection.execute("SELECT count(*) FROM project_asset_revisions").fetchone()[0] == 0
