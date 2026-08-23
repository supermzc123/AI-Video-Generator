import io
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from ai_video_generator.api import _project_asset_llm_context
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    AssetGenerationCandidateState,
    AssetScope,
    ProjectAsset,
    ProjectAssetPurpose,
    ProjectAssetSource,
    ProjectAssetState,
    ProjectSpec,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.persistence.project_assets import ProjectAssetStore
from ai_video_generator.project_assets_api import create_project_assets_router


def _png_bytes(color: tuple[int, int, int] = (20, 80, 140)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


def _app(tmp_path: Path) -> FastAPI:
    settings = Settings(_env_file=None, data_root=tmp_path)
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    store.put_project_revision(
        ProjectSpec(
            project_id="project-1",
            name="Asset API test",
            target_duration_seconds=30,
        )
    )
    app = FastAPI()
    app.include_router(create_project_assets_router(settings))
    return app


async def _upload(
    client: httpx.AsyncClient,
    *,
    name: str,
    filename: str = "reference.png",
    content: bytes | None = None,
    kind: str = "character",
    scope: str = "common",
    shot_id: str | None = None,
) -> httpx.Response:
    data = {"name": name, "kind": kind, "scope": scope}
    if shot_id is not None:
        data["shot_id"] = shot_id
    return await client.post(
        "/api/v1/projects/project-1/assets",
        data=data,
        files={"file": (filename, content if content is not None else _png_bytes(), "text/plain")},
    )


@pytest.mark.asyncio
async def test_upload_lists_real_image_metadata_and_serves_preview(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        uploaded = await _upload(client, name="主角")
        listed = await client.get("/api/v1/projects/project-1/assets")
        preview = await client.get(
            f"/api/v1/projects/project-1/assets/{uploaded.json()['asset_id']}/preview"
        )

    assert uploaded.status_code == 201
    body = uploaded.json()
    assert body["name"] == "主角"
    assert body["kind"] == "character"
    assert body["mime_type"] == "image/png"
    assert body["width"] == 32
    assert body["height"] == 24
    assert len(body["sha256"]) == 64
    assert listed.json() == [body]
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(preview.content)) as image:
        assert image.size == (32, 24)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content", "metadata", "expected_kind", "expected_mime"),
    [
        (
            "reference.mp4",
            b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64,
            {
                "width": 1280,
                "height": 720,
                "duration_seconds": 4.0,
                "frame_rate": 24.0,
                "has_audio": True,
            },
            "video",
            "video/mp4",
        ),
        (
            "reference.wav",
            b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 64,
            {
                "width": None,
                "height": None,
                "duration_seconds": 6.0,
                "frame_rate": None,
                "has_audio": True,
            },
            "audio",
            "audio/wav",
        ),
    ],
)
async def test_upload_serves_video_and_audio_reference_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content: bytes,
    metadata: dict[str, object],
    expected_kind: str,
    expected_mime: str,
) -> None:
    monkeypatch.setattr(ProjectAssetStore, "_probe_av", lambda *args: metadata)
    app = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        uploaded = await _upload(
            client, name=f"{expected_kind} reference", filename=filename, content=content
        )
        body = uploaded.json()
        media = await client.get(
            f"/api/v1/projects/project-1/assets/{body['asset_id']}/media"
        )
        preview = await client.get(
            f"/api/v1/projects/project-1/assets/{body['asset_id']}/preview"
        )

    assert uploaded.status_code == 201
    assert body["media_kind"] == expected_kind
    assert body["mime_type"] == expected_mime
    assert body["duration_seconds"] == metadata["duration_seconds"]
    assert media.status_code == 200
    assert media.content == content
    assert preview.status_code == 404


@pytest.mark.asyncio
async def test_upload_rejects_fake_images_and_case_insensitive_duplicate_names(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await _upload(client, name="Hero")
        duplicate = await _upload(client, name="hero", content=_png_bytes((1, 2, 3)))
        fake = await _upload(client, name="Not an image", content=b"not a png")
        missing_project = await client.post(
            "/api/v1/projects/missing/assets",
            data={"name": "orphan", "kind": "reference", "scope": "common"},
            files={"file": ("orphan.png", _png_bytes(), "image/png")},
        )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert fake.status_code == 422
    assert missing_project.status_code == 404


@pytest.mark.asyncio
async def test_patch_creates_revision_and_delete_keeps_content_addressed_blob(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        uploaded = await _upload(client, name="Hero")
        asset = uploaded.json()
        updated = await client.patch(
            f"/api/v1/projects/project-1/assets/{asset['asset_id']}",
            json={"name": "Lead", "kind": "style"},
        )
        deleted = await client.delete(f"/api/v1/projects/project-1/assets/{asset['asset_id']}")
        listed = await client.get("/api/v1/projects/project-1/assets")
        missing_preview = await client.get(
            f"/api/v1/projects/project-1/assets/{asset['asset_id']}/preview"
        )

    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["name"] == "Lead"
    assert updated.json()["kind"] == "style"
    assert deleted.status_code == 204
    assert listed.json() == []
    assert missing_preview.status_code == 404
    blobs = list((tmp_path / "project-assets" / "blobs").rglob(asset["sha256"]))
    assert len(blobs) == 1


@pytest.mark.asyncio
async def test_common_asset_switches_to_shot_scope_atomically(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        uploaded = await _upload(client, name="Hero")
        switched = await client.patch(
            f"/api/v1/projects/project-1/assets/{uploaded.json()['asset_id']}",
            json={"scope": "shot", "shot_id": "shot-1"},
        )

    assert switched.status_code == 200
    assert switched.json()["scope"] == "shot"
    assert switched.json()["shot_id"] == "shot-1"


@pytest.mark.asyncio
async def test_asset_shot_bindings_allow_multiple_or_no_shots(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        uploaded = await _upload(client, name="Hero")
        asset_url = f"/api/v1/projects/project-1/assets/{uploaded.json()['asset_id']}"
        assigned = await client.patch(asset_url, json={"shot_ids": ["shot-1", "shot-2"]})
        unassigned = await client.patch(asset_url, json={"shot_ids": []})

    assert assigned.status_code == 200
    assert assigned.json()["shot_ids"] == ["shot-1", "shot-2"]
    assert unassigned.status_code == 200
    assert unassigned.json()["shot_ids"] == []


def test_missing_blob_is_text_context_without_multimodal_preview(tmp_path: Path) -> None:
    _app(tmp_path)
    task_store = SQLiteTaskStore(tmp_path / "control-plane.db")
    task_store.put_project_asset(
        ProjectAsset(
            asset_id="legacy-1",
            project_id="project-1",
            name="旧主角参考",
            original_name="missing.png",
            state=ProjectAssetState.MISSING_BLOB,
            kind=ProjectAssetPurpose.CHARACTER,
            scope=AssetScope.COMMON,
            source=ProjectAssetSource.LEGACY,
            created_at=datetime.now(UTC),
        )
    )
    context, image_urls = _project_asset_llm_context(
        ProjectAssetStore(tmp_path / "control-plane.db", tmp_path / "project-assets"),
        "project-1",
    )

    assert image_urls == ()
    assert context[0]["asset_id"] == "legacy-1"
    assert context[0]["state"] == "missing_blob"
    assert context[0]["multimodal_preview_available"] is False


def test_generated_asset_can_be_replaced_as_an_immutable_revision(tmp_path: Path) -> None:
    _app(tmp_path)
    store = ProjectAssetStore(tmp_path / "control-plane.db", tmp_path / "project-assets")

    first = store.add_generated(
        project_id="project-1",
        name="主角定妆",
        content=_png_bytes((10, 20, 30)),
        source_task_id="image-task-1",
        kind=ProjectAssetPurpose.CHARACTER,
        shot_ids=("shot-1", "shot-2"),
    )
    second = store.add_generated(
        project_id="project-1",
        name="主角定妆",
        content=_png_bytes((30, 20, 10)),
        source_task_id="image-task-2",
        kind=ProjectAssetPurpose.CHARACTER,
        replace_asset_id=first.asset_id,
    )

    assert second.asset_id == first.asset_id
    assert second.revision == 2
    assert second.sha256 != first.sha256
    assert second.source == ProjectAssetSource.GENERATED
    assert second.source_task_id == "image-task-2"
    assert second.shot_ids == ("shot-1", "shot-2")


def test_generated_candidate_can_revise_an_uploaded_asset(tmp_path: Path) -> None:
    _app(tmp_path)
    store = ProjectAssetStore(tmp_path / "control-plane.db", tmp_path / "project-assets")
    uploaded = store.add_upload(
        project_id="project-1",
        name="主角定妆",
        original_name="hero.png",
        content=_png_bytes((10, 20, 30)),
        kind=ProjectAssetPurpose.CHARACTER,
    )

    generated = store.add_generated(
        project_id="project-1",
        name=uploaded.name,
        content=_png_bytes((30, 20, 10)),
        source_task_id="image-task-2",
        kind=ProjectAssetPurpose.CHARACTER,
        replace_asset_id=uploaded.asset_id,
    )

    assert generated.asset_id == uploaded.asset_id
    assert generated.revision == 2
    assert generated.source == ProjectAssetSource.GENERATED
    assert store.get_asset("project-1", uploaded.asset_id) == generated


def test_generated_candidate_does_not_replace_asset_until_accepted(tmp_path: Path) -> None:
    _app(tmp_path)
    store = ProjectAssetStore(tmp_path / "control-plane.db", tmp_path / "project-assets")
    current = store.add_generated(
        project_id="project-1",
        name="主角定妆",
        content=_png_bytes((10, 20, 30)),
        source_task_id="image-task-1",
        kind=ProjectAssetPurpose.CHARACTER,
    )

    candidate = store.add_generation_candidate(
        project_id="project-1",
        asset_plan_id="hero",
        source_task_id="image-task-2",
        name=current.name,
        content=_png_bytes((30, 20, 10)),
        current_asset_id=current.asset_id,
        kind=ProjectAssetPurpose.CHARACTER,
    )

    assert candidate.state == AssetGenerationCandidateState.PENDING
    assert store.get_asset("project-1", current.asset_id).revision == 1
    assert store.generation_candidate_content(candidate.candidate_id)
    discarded = store.resolve_generation_candidate(candidate.candidate_id)
    assert discarded.state == AssetGenerationCandidateState.DISCARDED
    assert store.get_asset("project-1", current.asset_id).revision == 1


def test_candidate_acceptance_claim_is_exclusive_and_recovers_when_stale(
    tmp_path: Path,
) -> None:
    _app(tmp_path)
    first = ProjectAssetStore(tmp_path / "control-plane.db", tmp_path / "project-assets")
    second = ProjectAssetStore(tmp_path / "control-plane.db", tmp_path / "project-assets")
    current = first.add_generated(
        project_id="project-1",
        name="主角定妆",
        content=_png_bytes((10, 20, 30)),
        source_task_id="image-task-1",
    )
    candidate = first.add_generation_candidate(
        project_id="project-1",
        asset_plan_id="hero",
        source_task_id="image-task-2",
        name=current.name,
        content=_png_bytes((30, 20, 10)),
        current_asset_id=current.asset_id,
    )
    now = datetime(2026, 8, 16, tzinfo=UTC)

    assert first.claim_generation_candidate_acceptance(candidate.candidate_id, now=now)
    with pytest.raises(ValueError, match="already in progress"):
        second.claim_generation_candidate_acceptance(candidate.candidate_id, now=now)
    recovered = second.claim_generation_candidate_acceptance(
        candidate.candidate_id,
        now=now + timedelta(minutes=11),
    )

    assert recovered.candidate_id == candidate.candidate_id
    second.release_generation_candidate_acceptance(candidate.candidate_id)


@pytest.mark.asyncio
async def test_delete_allows_historical_asset_mentions(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        uploaded = await _upload(client, name="Hero")
        asset_id = uploaded.json()["asset_id"]
        with sqlite3.connect(tmp_path / "control-plane.db") as connection:
            payload = (
                '{"idea":{"nodes":[{"kind":"asset_mention","asset_id":"'
                + asset_id
                + '","display_name":"Hero"}]}}'
            )
            connection.execute(
                "INSERT INTO project_workspace_revisions("
                "project_id, revision, payload_sha256, payload_json, created_at"
                ") VALUES (?, 1, ?, ?, 0)",
                ("project-1", "a" * 64, payload),
            )
        deleted = await client.delete(f"/api/v1/projects/project-1/assets/{asset_id}")
        listed = await client.get("/api/v1/projects/project-1/assets")

    assert deleted.status_code == 204
    assert listed.json() == []


@pytest.mark.asyncio
async def test_legacy_workspace_asset_is_visible_and_can_be_relinked(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with sqlite3.connect(tmp_path / "control-plane.db") as connection:
        payload = (
            '{"payload":{"assets":[{"id":"legacy-asset","name":"旧参考图",'
            '"fileName":"lost.png","kind":"character","scope":"public",'
            '"status":"missing_blob"}]}}'
        )
        connection.execute(
            "INSERT INTO project_workspace_revisions("
            "project_id, revision, payload_sha256, payload_json, created_at"
            ") VALUES (?, 1, ?, ?, 0)",
            ("project-1", "b" * 64, payload),
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        listed = await client.get("/api/v1/projects/project-1/assets")
        relinked = await client.post(
            "/api/v1/projects/project-1/assets/legacy-asset/relink",
            files={"file": ("restored.png", _png_bytes(), "image/png")},
        )
    assert listed.status_code == 200
    assert listed.json()[0]["state"] == "missing_blob"
    assert relinked.status_code == 200
    assert relinked.json()["asset_id"] == "legacy-asset"
    assert relinked.json()["state"] == "available"
