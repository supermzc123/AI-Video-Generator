from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ai_video_generator.config import Settings
from ai_video_generator.domain import AssetScope, ProjectAsset, ProjectAssetPurpose
from ai_video_generator.persistence.project_assets import (
    MAX_PROJECT_ASSET_BYTES,
    DuplicateProjectAssetNameError,
    InvalidProjectImageError,
    ProjectAssetDependencyError,
    ProjectAssetNotFoundError,
    ProjectAssetStore,
    ProjectAssetTooLargeError,
    ProjectNotFoundError,
)


class ProjectAssetUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    kind: ProjectAssetPurpose | None = None
    scope: AssetScope | None = None
    shot_id: str | None = Field(default=None, min_length=1, max_length=200)


async def _read_upload(upload: UploadFile) -> bytes:
    content = bytearray()
    while chunk := await upload.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > MAX_PROJECT_ASSET_BYTES:
            raise ProjectAssetTooLargeError(
                f"media exceeds {MAX_PROJECT_ASSET_BYTES} byte upload limit"
            )
    return bytes(content)


def create_project_assets_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/v1/projects/{project_id}/assets", tags=["project-assets"])
    store: ProjectAssetStore | None = None

    def get_store() -> ProjectAssetStore:
        nonlocal store
        if store is None:
            data_root = Path(settings.data_root)
            store = ProjectAssetStore(
                data_root / "control-plane.db",
                data_root / "project-assets",
                ffprobe_binary=settings.ffprobe_binary,
            )
        return store

    @router.post("", response_model=ProjectAsset, status_code=201)
    async def upload_project_asset(
        project_id: str,
        file: Annotated[UploadFile, File()],
        name: Annotated[str, Form(min_length=1, max_length=200)],
        kind: Annotated[ProjectAssetPurpose, Form()] = ProjectAssetPurpose.REFERENCE,
        scope: Annotated[AssetScope, Form()] = AssetScope.COMMON,
        shot_id: Annotated[str | None, Form(max_length=200)] = None,
    ) -> ProjectAsset:
        try:
            content = await _read_upload(file)
            return get_store().add_upload(
                project_id=project_id,
                name=name,
                original_name=file.filename or "upload",
                content=content,
                kind=kind,
                scope=scope,
                shot_id=shot_id,
            )
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        except DuplicateProjectAssetNameError as exc:
            raise HTTPException(
                status_code=409, detail="asset name already exists in this project"
            ) from exc
        except ProjectAssetTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (InvalidProjectImageError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            await file.close()

    @router.get("", response_model=tuple[ProjectAsset, ...])
    async def list_project_assets(project_id: str) -> tuple[ProjectAsset, ...]:
        try:
            return get_store().list_assets(project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @router.get("/{asset_id}/preview", response_class=FileResponse)
    async def get_project_asset_preview(project_id: str, asset_id: str) -> FileResponse:
        try:
            path = get_store().preview_for(project_id, asset_id)
        except ProjectAssetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project asset not found") from exc
        return FileResponse(path, media_type="image/jpeg", filename=f"{asset_id}-preview.jpg")

    @router.get("/{asset_id}/media", response_class=FileResponse)
    async def get_project_asset_media(project_id: str, asset_id: str) -> FileResponse:
        try:
            asset = get_store().get_asset(project_id, asset_id)
            path = get_store().media_for(project_id, asset_id)
        except ProjectAssetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project asset not found") from exc
        return FileResponse(
            path,
            media_type=asset.mime_type or "application/octet-stream",
            filename=asset.original_name,
        )

    @router.post("/{asset_id}/relink", response_model=ProjectAsset)
    async def relink_project_asset(
        project_id: str,
        asset_id: str,
        file: Annotated[UploadFile, File()],
    ) -> ProjectAsset:
        try:
            content = await _read_upload(file)
            return get_store().relink_upload(
                project_id,
                asset_id,
                original_name=file.filename or "upload",
                content=content,
            )
        except ProjectAssetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project asset not found") from exc
        except ProjectAssetTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (InvalidProjectImageError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            await file.close()

    @router.patch("/{asset_id}", response_model=ProjectAsset)
    async def update_project_asset(
        project_id: str, asset_id: str, request: ProjectAssetUpdateRequest
    ) -> ProjectAsset:
        if not request.model_fields_set:
            raise HTTPException(status_code=422, detail="at least one field must be supplied")
        try:
            return get_store().update_asset(
                project_id,
                asset_id,
                name=request.name,
                kind=request.kind,
                scope=request.scope,
                shot_id=request.shot_id,
                shot_id_was_set="shot_id" in request.model_fields_set,
            )
        except ProjectAssetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project asset not found") from exc
        except DuplicateProjectAssetNameError as exc:
            raise HTTPException(
                status_code=409, detail="asset name already exists in this project"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/{asset_id}", status_code=204)
    async def delete_project_asset(project_id: str, asset_id: str) -> Response:
        try:
            get_store().retire_asset(project_id, asset_id)
        except ProjectAssetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project asset not found") from exc
        except ProjectAssetDependencyError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "project asset is still referenced",
                    "references": exc.references,
                },
            ) from exc
        return Response(status_code=204)

    return router
