"""Real image-registration routes preserve frozen inputs and explicit retry policy."""

import hashlib
import io
import json
from datetime import UTC, datetime

import httpx
import pytest
from PIL import Image

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BindingSemantic,
    BindingValueType,
    ProjectSpec,
    ProjectWorkspaceRevision,
    TaskState,
    WorkflowApproval,
    WorkflowBinding,
    WorkflowOutput,
    WorkflowTemplate,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.persistence.project_assets import ProjectAssetStore
from ai_video_generator.workers.workflow import canonical_json_sha256


def save(store, payload):
    try:
        revision = store.get_latest_project_workspace("p").revision + 1
    except KeyError:
        revision = 1
    return store.put_project_workspace_revision(
        ProjectWorkspaceRevision(
            project_id="p",
            revision=revision,
            payload=payload,
            payload_sha256=hashlib.sha256(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            created_at=datetime.now(UTC),
        )
    )


def image_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def image_project(tmp_path):
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    store.put_project_revision(
        ProjectSpec(project_id="p", name="Images", target_duration_seconds=4)
    )
    assets = ProjectAssetStore(tmp_path / "control-plane.db", tmp_path / "project-assets")
    reference = assets.add_generated(
        project_id="p",
        name="Reference",
        content=image_bytes("red"),
        source_task_id="seed",
    )
    raw = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "portrait"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "reference.png"}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }
    store.put_workflow_revision(
        WorkflowTemplate(
            template_id="image",
            name="Image",
            workflow_sha256=canonical_json_sha256(raw),
            node_schema_sha256="a" * 64,
            raw_workflow=raw,
            bindings=(
                WorkflowBinding(
                    binding_id="prompt",
                    semantic=BindingSemantic.PROMPT,
                    node_id="1",
                    input_name="text",
                    value_type=BindingValueType.STRING,
                    title="Prompt",
                ),
                WorkflowBinding(
                    binding_id="ref",
                    semantic=BindingSemantic.REFERENCE_IMAGE,
                    node_id="2",
                    input_name="image",
                    value_type=BindingValueType.IMAGE_PATH,
                    reference_index=1,
                    title="Reference",
                ),
            ),
            outputs=(WorkflowOutput(output_id="image", node_id="3", title="Image"),),
            required_node_types=("CLIPTextEncode", "LoadImage", "SaveImage"),
            approval=WorkflowApproval.APPROVED,
        )
    )
    payload = {
        "assetPlans": [{"id": "plan", "kind": "character", "width": 1024, "height": 1600}],
        "prompts": {
            "imagePrompts": [
                {
                    "id": "ip",
                    "assetPlanId": "plan",
                    "prompt": "portrait",
                    "negativePrompt": "blur",
                    "workflowTemplateId": "image",
                    "referenceAssetIds": [reference.asset_id],
                    "revision": 1,
                    "locked": True,
                }
            ]
        },
    }
    workspace = save(store, payload)
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return store, assets, reference, workspace, client


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [TaskState.FAILED, TaskState.PAUSED, TaskState.CANCELLED])
async def test_image_registration_never_implicitly_restarts_existing_attempt(image_project, state):
    store, _, _, workspace, client = image_project
    async with client:
        request = {"expected_workspace_sha256": workspace.payload_sha256}
        created = await client.post("/api/v1/projects/p/image-prompts/plan/run", json=request)
        assert created.status_code == 202, created.text
        task_id = created.json()["task_id"]
        if state == TaskState.FAILED:
            store.transition_task(task_id, TaskState.RUNNING)
        store.transition_task(task_id, state)
        before = store.get_task(task_id)
        repeated = await client.post("/api/v1/projects/p/image-prompts/plan/run", json=request)
        assert repeated.status_code == 202, repeated.text
        assert repeated.json()["state"] == state.value
        assert store.get_task(task_id).attempt == before.attempt
        assert len(store.list_tasks()) == 1


@pytest.mark.asyncio
async def test_reference_revision_changes_exact_image_task_identity(image_project):
    store, assets, reference, _, client = image_project
    async with client:
        first = await client.post("/api/v1/projects/p/image-prompts/plan/run", json={})
        assert first.status_code == 202, first.text
        assets.add_generated(
            project_id="p",
            name="Reference",
            content=image_bytes("blue"),
            source_task_id="replacement",
            replace_asset_id=reference.asset_id,
        )
        second = await client.post("/api/v1/projects/p/image-prompts/plan/run", json={})
        assert second.status_code == 202, second.text
        assert first.json()["task_id"] != second.json()["task_id"]
        assert first.json()["input_fingerprint"] != second.json()["input_fingerprint"]
        assert len(store.list_tasks()) == 2


@pytest.mark.asyncio
async def test_stale_workspace_image_registration_has_no_task_side_effect(image_project):
    store, _, _, workspace, client = image_project
    save(store, {**workspace.payload, "updatedAt": "changed"})
    async with client:
        response = await client.post(
            "/api/v1/projects/p/image-prompts/plan/run",
            json={
                "expected_workspace_sha256": workspace.payload_sha256,
            },
        )
    assert response.status_code == 409
    assert store.list_tasks() == ()
