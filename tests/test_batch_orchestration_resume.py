import hashlib

from ai_video_generator.domain import TaskKind, TaskSpec, TaskState
from ai_video_generator.services.batch_orchestration import (
    reusable_h3_prompt,
    reusable_image_prompt,
    stable_operation_id,
)


def test_image_prompt_reuse_requires_actual_workspace_schema():
    valid = {
        "id": "image-1", "assetPlanId": "plan-1", "prompt": "a clear subject",
        "negativePrompt": "blur", "workflowTemplateId": "workflow-1",
        "referenceAssetIds": [], "locked": True, "revision": 2,
        "harnessRevision": 1,
    }
    assert reusable_image_prompt(valid, "plan-1")
    assert not reusable_image_prompt({**valid, "assetPlanId": "other"}, "plan-1")
    assert not reusable_image_prompt({**valid, "referenceAssetIds": ["", 3]}, "plan-1")


def test_h3_prompt_reuse_requires_expected_segment_identity_and_duration():
    segment = {
        "segmentId": "shot-1.C01", "shotId": "shot-1", "segmentIndex": 0,
        "continuationOf": None, "durationSeconds": 10.0,
    }
    valid = {
        "id": "h3-1", "segmentId": "shot-1.C01", "shotId": "shot-1",
        "segmentIndex": 0, "continuationOf": None, "durationSeconds": 10.0,
        "inputMode": "t2va", "prompt": "camera moves slowly", "assetIds": [],
        "seed": 42, "locked": True, "revision": 1, "harnessRevision": 1,
    }
    assert reusable_h3_prompt(valid, segment)
    assert not reusable_h3_prompt({**valid, "segmentId": "shot-1.C02"}, segment)
    assert not reusable_h3_prompt({**valid, "durationSeconds": 11.0}, segment)


def test_operation_ids_are_stable():
    fingerprint = hashlib.sha256(b"task").hexdigest()
    task = TaskSpec(
        task_id="batch-task", project_id="project-1", kind=TaskKind.LLM_PLANNING,
        state=TaskState.READY, idempotency_key=fingerprint,
        input_fingerprint=fingerprint,
    )
    assert stable_operation_id(task, "storyboard", {"revision": 1}) == stable_operation_id(
        task, "storyboard", {"revision": 1}
    )
