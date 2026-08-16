import hashlib
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    BindingSemantic,
    BindingValueType,
    ExecutionTarget,
    ProjectSpec,
    TaskKind,
    TaskSpec,
    TaskState,
    WorkerCapabilities,
    WorkflowApproval,
    WorkflowBindingDraft,
    WorkflowInvocation,
)
from ai_video_generator.llm import WorkflowMappingRequest
from ai_video_generator.services.remote import WorkerHeartbeat
from ai_video_generator.workers import (
    inspect_api_workflow,
    workflow_template_from_inspection,
)


def object_info() -> dict[str, object]:
    return {
        "CLIPTextEncode": {
            "input": {"required": {"text": ["STRING"]}},
            "output": ["CONDITIONING"],
        },
        "SaveImage": {
            "input": {"required": {"images": ["CONDITIONING"]}},
            "output": [],
            "output_node": True,
        },
    }


def workflow() -> dict[str, object]:
    return {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "default"},
            "_meta": {"title": "正面提示词"},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0]},
            "_meta": {"title": "保存图片"},
        },
    }


@pytest.mark.asyncio
async def test_workflow_inspect_register_and_compile(tmp_path: Path) -> None:
    def comfy_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=object_info())
        return httpx.Response(404)

    app = create_app(
        Settings(_env_file=None, data_root=tmp_path, comfyui_base_url="http://comfy"),
        comfyui_transport=httpx.MockTransport(comfy_handler),
    )
    transport = httpx.ASGITransport(app=app)
    inspection = inspect_api_workflow(workflow(), object_info()).model_copy(
        update={
            "bindings": (
                WorkflowBindingDraft(
                    binding_id="prompt",
                    semantic=BindingSemantic.PROMPT,
                    node_id="1",
                    input_name="text",
                    value_type=BindingValueType.STRING,
                    title="正面提示词",
                    default_value="default",
                ),
            )
        }
    )
    template = workflow_template_from_inspection(
        inspection,
        template_id="image:test",
        name="Image test",
        approval=WorkflowApproval.APPROVED,
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        inspected = await client.post(
            "/api/v1/workflows/inspect",
            json={"raw_workflow": workflow(), "object_info": object_info()},
        )
        registered = await client.post(
            "/api/v1/workflows/templates",
            json=template.model_dump(mode="json"),
        )
        compiled = await client.post(
            "/api/v1/workflows/compile",
            json={
                "template": template.model_dump(mode="json"),
                "invocation": WorkflowInvocation(
                    template_id="image:test",
                    template_revision=1,
                    values={"prompt": "new prompt"},
                ).model_dump(mode="json"),
            },
        )

    assert inspected.status_code == 200
    assert inspected.json()["bindings"] == []
    assert inspected.json()["outputs"][0]["node_id"] == "2"
    assert registered.status_code == 201
    assert compiled.status_code == 200
    assert compiled.json()["workflow"]["1"]["inputs"]["text"] == "new prompt"


@pytest.mark.asyncio
async def test_draft_workflow_cannot_be_registered(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    inspection = inspect_api_workflow(workflow(), object_info())
    template = workflow_template_from_inspection(
        inspection,
        template_id="image:draft",
        name="Draft",
        approval=WorkflowApproval.DRAFT,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/workflows/templates",
            json=template.model_dump(mode="json"),
        )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_llm_mapping_requires_configuration(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    request = WorkflowMappingRequest(
        operation_id="mapping-1",
        workflow_id="image:test",
        project=ProjectSpec(
            project_id="project-1",
            name="Project",
            target_duration_seconds=60,
        ),
        raw_workflow=workflow(),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/llm/workflows/map",
            json=request.model_dump(mode="json"),
        )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_remote_worker_claims_a_leased_task(tmp_path: Path) -> None:
    token = "test-worker-token"
    app = create_app(Settings(_env_file=None, data_root=tmp_path, worker_auth_token=token))
    headers = {"Authorization": f"Bearer {token}"}
    worker = WorkerCapabilities(
        worker_id="ubuntu-1",
        platform="linux",
        node_schema_sha256="a" * 64,
    )
    task = TaskSpec(
        task_id="remote-image-1",
        project_id="project-1",
        kind=TaskKind.IMAGE_GENERATION,
        state=TaskState.READY,
        idempotency_key="b" * 64,
        input_fingerprint="c" * 64,
        execution_target=ExecutionTarget.REMOTE,
        worker_id="ubuntu-1",
    )
    heartbeat = WorkerHeartbeat(
        worker_id="ubuntu-1",
        sent_at=datetime.now(UTC),
        available_gpu_slots=1,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/api/v1/tasks", json=task.model_dump(mode="json"))
        registered = await client.post(
            "/api/v1/workers/register",
            headers=headers,
            json={"capabilities": worker.model_dump(mode="json")},
        )
        queued = await client.post("/api/v1/tasks/remote-image-1/run")
        claimed = await client.post(
            "/api/v1/workers/ubuntu-1/leases/claim",
            headers=headers,
            json={"heartbeat": heartbeat.model_dump(mode="json")},
        )

    assert created.status_code == 201
    assert registered.status_code == 200
    assert queued.status_code == 200
    assert claimed.status_code == 200
    assert claimed.json()["task_id"] == "remote-image-1"
    assert claimed.json()["state"] == "running"


@pytest.mark.asyncio
async def test_artifact_transfer_verifies_content_hash(tmp_path: Path) -> None:
    token = "test-worker-token"
    app = create_app(Settings(_env_file=None, data_root=tmp_path, worker_auth_token=token))
    headers = {"Authorization": f"Bearer {token}"}
    content = b"content-addressed artifact"
    digest = hashlib.sha256(content).hexdigest()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        uploaded = await client.put(
            f"/api/v1/artifacts/blobs/{digest}", headers=headers, content=content
        )
        downloaded = await client.get(f"/api/v1/artifacts/blobs/{digest}", headers=headers)
        rejected = await client.put(
            f"/api/v1/artifacts/blobs/{'0' * 64}", headers=headers, content=content
        )

    assert uploaded.status_code == 200
    assert uploaded.json()["byte_size"] == len(content)
    assert downloaded.content == content
    assert rejected.status_code == 422
