from pathlib import Path

import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import ProjectSpec, TaskKind, TaskSpec, TaskState


def project(revision: int = 1, name: str = "Project") -> ProjectSpec:
    return ProjectSpec(
        project_id="project-1",
        revision=revision,
        name=name,
        target_duration_seconds=60,
    )


def review_task(task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        project_id="project-1",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.NEEDS_REVIEW,
        idempotency_key=("a" if task_id == "accept" else "b") * 64,
        input_fingerprint=("c" if task_id == "accept" else "d") * 64,
    )


@pytest.mark.asyncio
async def test_project_revision_endpoints_preserve_history(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    first = project()
    second = project(2, "Revised")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/api/v1/projects", json=first.model_dump(mode="json"))
        revised = await client.post(
            "/api/v1/projects/project-1/revisions",
            json=second.model_dump(mode="json"),
        )
        latest = await client.get("/api/v1/projects/project-1")
        history = await client.get("/api/v1/projects/project-1/revisions")
        exact = await client.get("/api/v1/projects/project-1/revisions/1")
        projects = await client.get("/api/v1/projects")
        conflict = await client.post(
            "/api/v1/projects/project-1/revisions",
            json=first.model_copy(update={"name": "Overwrite"}).model_dump(mode="json"),
        )

    assert created.status_code == 201
    assert revised.status_code == 201
    assert latest.json()["revision"] == 2
    assert [item["revision"] for item in history.json()] == [1, 2]
    assert exact.json()["name"] == "Project"
    assert [item["project_id"] for item in projects.json()] == ["project-1"]
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_review_accept_and_reject_endpoints(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for task_id in ("accept", "reject"):
            response = await client.post(
                "/api/v1/tasks", json=review_task(task_id).model_dump(mode="json")
            )
            assert response.status_code == 201
        accepted = await client.post("/api/v1/tasks/accept/review/accept")
        accepted_retry = await client.post("/api/v1/tasks/accept/review/accept")
        rejected = await client.post(
            "/api/v1/tasks/reject/review/reject", json={"feedback": "Fix motion"}
        )
        reverse = await client.post("/api/v1/tasks/accept/review/reject", json={})

    assert accepted.status_code == 200
    assert accepted.json()["state"] == "succeeded"
    assert accepted_retry.json() == accepted.json()
    assert rejected.status_code == 200
    assert rejected.json()["state"] == "failed"
    assert rejected.json()["error_code"] == "review_rejected"
    assert rejected.json()["error_message"] == "Fix motion"
    assert reverse.status_code == 409
