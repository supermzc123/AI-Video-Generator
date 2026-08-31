from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import sqlite3
import subprocess
import tempfile
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ai_video_generator.domain import (
    AssetGenerationCandidate,
    AssetGenerationCandidateState,
    AssetScope,
    ProjectAsset,
    ProjectAssetMediaKind,
    ProjectAssetPurpose,
    ProjectAssetSource,
    ProjectAssetState,
)

MAX_PROJECT_ASSET_BYTES = 200 * 1024 * 1024
MAX_IMAGE_DIMENSION = 32_768
MAX_IMAGE_PIXELS = 100_000_000
PREVIEW_MAX_SIZE = (768, 768)
SUPPORTED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
H3_REFERENCE_VIDEO_MIN_SECONDS = 2.0
H3_REFERENCE_VIDEO_MAX_SECONDS = 15.0
H3_REFERENCE_VIDEO_FPS = 24.0


class ProjectAssetStoreError(RuntimeError):
    pass


class ProjectAssetNotFoundError(ProjectAssetStoreError):
    pass


class ProjectNotFoundError(ProjectAssetStoreError):
    pass


class DuplicateProjectAssetNameError(ProjectAssetStoreError):
    pass


class InvalidProjectImageError(ProjectAssetStoreError):
    pass


class ProjectAssetTooLargeError(ProjectAssetStoreError):
    pass


def _normalized_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name.strip()).casefold()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class ProjectAssetStore:
    """Immutable project-asset metadata plus content-addressed image blobs."""

    def __init__(
        self,
        database_path: str | Path,
        storage_root: str | Path,
        *,
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        self.database_path = Path(database_path)
        self.storage_root = Path(storage_root)
        self.blob_root = self.storage_root / "blobs"
        self.preview_root = self.storage_root / "previews"
        self.ffprobe_binary = ffprobe_binary
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.preview_root.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS asset_generation_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    asset_plan_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_asset_candidates_plan
                    ON asset_generation_candidates(project_id, asset_plan_id, created_at);
                CREATE TABLE IF NOT EXISTS asset_candidate_acceptance_claims (
                    candidate_id TEXT PRIMARY KEY,
                    claimed_at REAL NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES asset_generation_candidates(candidate_id)
                        ON DELETE CASCADE
                );
                """
            )

    def project_exists(self, project_id: str) -> bool:
        with self._connect() as connection:
            for table in ("project_revisions", "project_workspace_revisions"):
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
                ).fetchone()
                if (
                    exists
                    and connection.execute(
                        f"SELECT 1 FROM {table} WHERE project_id = ? LIMIT 1", (project_id,)
                    ).fetchone()
                ):
                    return True
        return False

    def add_upload(
        self,
        *,
        project_id: str,
        name: str,
        original_name: str,
        content: bytes,
        kind: ProjectAssetPurpose = ProjectAssetPurpose.REFERENCE,
        scope: AssetScope = AssetScope.COMMON,
        shot_id: str | None = None,
    ) -> ProjectAsset:
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        if len(content) > MAX_PROJECT_ASSET_BYTES:
            raise ProjectAssetTooLargeError(
                f"media exceeds {MAX_PROJECT_ASSET_BYTES} byte upload limit"
            )
        (
            media_kind,
            mime_type,
            width,
            height,
            duration_seconds,
            frame_rate,
            has_audio,
            preview_content,
        ) = self._validate_upload(content, original_name)
        blob_sha256 = hashlib.sha256(content).hexdigest()
        preview_sha256 = (
            hashlib.sha256(preview_content).hexdigest() if preview_content is not None else None
        )
        asset = ProjectAsset(
            asset_id=str(uuid.uuid4()),
            project_id=project_id,
            revision=1,
            name=name,
            original_name=original_name or "upload",
            state=ProjectAssetState.AVAILABLE,
            sha256=blob_sha256,
            preview_sha256=preview_sha256,
            mime_type=mime_type,
            media_kind=media_kind,
            byte_size=len(content),
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            frame_rate=frame_rate,
            has_audio=has_audio,
            kind=kind,
            scope=scope,
            shot_id=shot_id,
            shot_ids=(shot_id,) if shot_id else (),
            source=ProjectAssetSource.UPLOAD,
            created_at=datetime.now(UTC),
        )
        with self._transaction(immediate=True) as connection:
            self._ensure_unique_name(connection, project_id, asset.name)
            self._write_blob_once(self.blob_path(blob_sha256), content)
            if preview_sha256 is not None and preview_content is not None:
                self._write_blob_once(self.preview_path(preview_sha256), preview_content)
            self._insert(connection, asset)
        return asset

    def add_generated(
        self,
        *,
        project_id: str,
        name: str,
        content: bytes,
        source_task_id: str,
        kind: ProjectAssetPurpose = ProjectAssetPurpose.REFERENCE,
        scope: AssetScope = AssetScope.COMMON,
        shot_id: str | None = None,
        shot_ids: tuple[str, ...] = (),
        replace_asset_id: str | None = None,
    ) -> ProjectAsset:
        """Register a generated image, optionally revising an existing project asset."""
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        if len(content) > MAX_PROJECT_ASSET_BYTES:
            raise ProjectAssetTooLargeError(
                f"media exceeds {MAX_PROJECT_ASSET_BYTES} byte upload limit"
            )
        image, mime_type, width, height = self._validate_image(content)
        blob_sha256 = hashlib.sha256(content).hexdigest()
        preview_content = self._make_preview(image)
        preview_sha256 = hashlib.sha256(preview_content).hexdigest()
        with self._transaction(immediate=True) as connection:
            current = (
                self._get_current(connection, project_id, replace_asset_id)
                if replace_asset_id
                else None
            )
            asset_id = current.asset_id if current else str(uuid.uuid4())
            effective_shot_ids = (
                shot_ids
                or (current.shot_ids if current else ())
                or ((shot_id,) if shot_id else ())
            )
            self._ensure_unique_name(
                connection,
                project_id,
                name,
                excluding_asset_id=asset_id if current else None,
            )
            self._write_blob_once(self.blob_path(blob_sha256), content)
            if preview_sha256 is not None and preview_content is not None:
                self._write_blob_once(self.preview_path(preview_sha256), preview_content)
            asset = ProjectAsset(
                asset_id=asset_id,
                project_id=project_id,
                revision=current.revision + 1 if current else 1,
                name=name,
                original_name=f"generated-{source_task_id}.png",
                state=ProjectAssetState.AVAILABLE,
                sha256=blob_sha256,
                preview_sha256=preview_sha256,
                mime_type=mime_type,
                byte_size=len(content),
                width=width,
                height=height,
                kind=kind,
                scope=scope,
                shot_id=shot_id,
                shot_ids=effective_shot_ids,
                source=ProjectAssetSource.GENERATED,
                source_task_id=source_task_id,
                created_at=datetime.now(UTC),
            )
            self._insert(connection, asset)
        return asset

    def add_generation_candidate(
        self,
        *,
        project_id: str,
        asset_plan_id: str,
        source_task_id: str,
        name: str,
        content: bytes,
        current_asset_id: str | None,
        kind: ProjectAssetPurpose = ProjectAssetPurpose.REFERENCE,
        scope: AssetScope = AssetScope.COMMON,
        shot_id: str | None = None,
    ) -> AssetGenerationCandidate:
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        if len(content) > MAX_PROJECT_ASSET_BYTES:
            raise ProjectAssetTooLargeError(
                f"image exceeds {MAX_PROJECT_ASSET_BYTES} byte upload limit"
            )
        image, mime_type, width, height = self._validate_image(content)
        blob_sha256 = hashlib.sha256(content).hexdigest()
        preview_content = self._make_preview(image)
        preview_sha256 = hashlib.sha256(preview_content).hexdigest()
        candidate = AssetGenerationCandidate(
            candidate_id=str(uuid.uuid4()),
            project_id=project_id,
            asset_plan_id=asset_plan_id,
            source_task_id=source_task_id,
            current_asset_id=current_asset_id,
            name=name,
            kind=kind,
            scope=scope,
            shot_id=shot_id,
            sha256=blob_sha256,
            preview_sha256=preview_sha256,
            mime_type=mime_type,
            byte_size=len(content),
            width=width,
            height=height,
            created_at=datetime.now(UTC),
        )
        with self._transaction(immediate=True) as connection:
            self._write_blob_once(self.blob_path(blob_sha256), content)
            if preview_sha256 is not None and preview_content is not None:
                self._write_blob_once(self.preview_path(preview_sha256), preview_content)
            connection.execute(
                "INSERT INTO asset_generation_candidates("
                "candidate_id, project_id, asset_plan_id, state, payload_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    candidate.candidate_id,
                    candidate.project_id,
                    candidate.asset_plan_id,
                    candidate.state.value,
                    _canonical_json(candidate.model_dump(mode="json")),
                    candidate.created_at.timestamp(),
                ),
            )
        return candidate

    def get_generation_candidate(self, candidate_id: str) -> AssetGenerationCandidate:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM asset_generation_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise ProjectAssetNotFoundError(candidate_id)
        return AssetGenerationCandidate.model_validate_json(row["payload_json"])

    def list_generation_candidates(
        self, project_id: str, asset_plan_id: str | None = None
    ) -> tuple[AssetGenerationCandidate, ...]:
        query = "SELECT payload_json FROM asset_generation_candidates WHERE project_id = ?"
        parameters: list[object] = [project_id]
        if asset_plan_id is not None:
            query += " AND asset_plan_id = ?"
            parameters.append(asset_plan_id)
        query += " ORDER BY created_at, candidate_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            AssetGenerationCandidate.model_validate_json(row["payload_json"])
            for row in rows
        )

    def generation_candidate_preview(self, candidate_id: str) -> Path:
        candidate = self.get_generation_candidate(candidate_id)
        path = self.preview_path(candidate.preview_sha256)
        if not path.is_file():
            raise ProjectAssetNotFoundError(candidate_id)
        return path

    def generation_candidate_content(self, candidate_id: str) -> bytes:
        candidate = self.get_generation_candidate(candidate_id)
        path = self.blob_path(candidate.sha256)
        if not path.is_file():
            raise ProjectAssetNotFoundError(candidate_id)
        return path.read_bytes()

    def claim_generation_candidate_acceptance(
        self,
        candidate_id: str,
        *,
        now: datetime | None = None,
        stale_after: timedelta = timedelta(minutes=10),
    ) -> AssetGenerationCandidate:
        """Serialize candidate acceptance across control-plane processes."""
        claimed_at = now or datetime.now(UTC)
        if claimed_at.tzinfo is None:
            claimed_at = claimed_at.replace(tzinfo=UTC)
        if stale_after <= timedelta(0):
            raise ValueError("candidate acceptance claim lifetime must be positive")
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM asset_generation_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ProjectAssetNotFoundError(candidate_id)
            candidate = AssetGenerationCandidate.model_validate_json(row["payload_json"])
            if candidate.state != AssetGenerationCandidateState.PENDING:
                raise ValueError("asset candidate has already been resolved")
            connection.execute(
                "DELETE FROM asset_candidate_acceptance_claims "
                "WHERE candidate_id = ? AND claimed_at <= ?",
                (candidate_id, (claimed_at - stale_after).timestamp()),
            )
            try:
                connection.execute(
                    "INSERT INTO asset_candidate_acceptance_claims(candidate_id, claimed_at) "
                    "VALUES (?, ?)",
                    (candidate_id, claimed_at.timestamp()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("asset candidate acceptance is already in progress") from exc
        return candidate

    def release_generation_candidate_acceptance(self, candidate_id: str) -> None:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM asset_candidate_acceptance_claims WHERE candidate_id = ?",
                (candidate_id,),
            )

    def resolve_generation_candidate(
        self, candidate_id: str, *, accepted_asset_id: str | None = None
    ) -> AssetGenerationCandidate:
        with self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM asset_generation_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ProjectAssetNotFoundError(candidate_id)
            current = AssetGenerationCandidate.model_validate_json(row["payload_json"])
            target_state = (
                AssetGenerationCandidateState.ACCEPTED
                if accepted_asset_id
                else AssetGenerationCandidateState.DISCARDED
            )
            if current.state != AssetGenerationCandidateState.PENDING:
                if (
                    current.state == target_state
                    and current.accepted_asset_id == accepted_asset_id
                ):
                    return current
                raise ValueError("asset candidate has already been resolved")
            updated = current.model_copy(
                update={"state": target_state, "accepted_asset_id": accepted_asset_id}
            )
            connection.execute(
                "UPDATE asset_generation_candidates SET state = ?, payload_json = ? "
                "WHERE candidate_id = ?",
                (
                    updated.state.value,
                    _canonical_json(updated.model_dump(mode="json")),
                    candidate_id,
                ),
            )
        return updated

    def list_assets(self, project_id: str) -> tuple[ProjectAsset, ...]:
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT current.payload_json
                FROM project_asset_revisions current
                WHERE current.project_id = ?
                  AND current.revision = (
                    SELECT MAX(candidate.revision)
                    FROM project_asset_revisions candidate
                    WHERE candidate.asset_id = current.asset_id
                  )
                  AND current.state != ?
                ORDER BY current.created_at, current.asset_id
                """,
                (project_id, ProjectAssetState.RETIRED.value),
            ).fetchall()
        return tuple(ProjectAsset.model_validate_json(row["payload_json"]) for row in rows)

    def relink_upload(
        self,
        project_id: str,
        asset_id: str,
        *,
        original_name: str,
        content: bytes,
    ) -> ProjectAsset:
        if len(content) > MAX_PROJECT_ASSET_BYTES:
            raise ProjectAssetTooLargeError(
                f"image exceeds {MAX_PROJECT_ASSET_BYTES} byte upload limit"
            )
        (
            media_kind,
            mime_type,
            width,
            height,
            duration_seconds,
            frame_rate,
            has_audio,
            preview_content,
        ) = self._validate_upload(content, original_name)
        blob_sha256 = hashlib.sha256(content).hexdigest()
        preview_sha256 = (
            hashlib.sha256(preview_content).hexdigest() if preview_content is not None else None
        )
        with self._transaction(immediate=True) as connection:
            current = self._get_current(connection, project_id, asset_id)
            if current.state != ProjectAssetState.MISSING_BLOB:
                raise ValueError("only missing_blob assets can be relinked")
            self._write_blob_once(self.blob_path(blob_sha256), content)
            if preview_sha256 is not None and preview_content is not None:
                self._write_blob_once(self.preview_path(preview_sha256), preview_content)
            updated = ProjectAsset.model_validate(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "original_name": original_name or current.original_name,
                        "state": ProjectAssetState.AVAILABLE,
                        "sha256": blob_sha256,
                        "preview_sha256": preview_sha256,
                        "mime_type": mime_type,
                        "media_kind": media_kind,
                        "byte_size": len(content),
                        "width": width,
                        "height": height,
                        "duration_seconds": duration_seconds,
                        "frame_rate": frame_rate,
                        "has_audio": has_audio,
                        "source": ProjectAssetSource.UPLOAD,
                        "created_at": datetime.now(UTC),
                    }
                ).model_dump(mode="python")
            )
            self._insert(connection, updated)
        return updated

    def get_asset(self, project_id: str, asset_id: str) -> ProjectAsset:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM project_asset_revisions
                WHERE project_id = ? AND asset_id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (project_id, asset_id),
            ).fetchone()
        if row is None:
            raise ProjectAssetNotFoundError(asset_id)
        asset = ProjectAsset.model_validate_json(row["payload_json"])
        if asset.state == ProjectAssetState.RETIRED:
            raise ProjectAssetNotFoundError(asset_id)
        return asset

    def update_asset(
        self,
        project_id: str,
        asset_id: str,
        *,
        name: str | None = None,
        kind: ProjectAssetPurpose | None = None,
        scope: AssetScope | None = None,
        shot_id: str | None = None,
        shot_id_was_set: bool = False,
        shot_ids: tuple[str, ...] | None = None,
    ) -> ProjectAsset:
        with self._transaction(immediate=True) as connection:
            current = self._get_current(connection, project_id, asset_id)
            new_name = current.name if name is None else name
            self._ensure_unique_name(connection, project_id, new_name, excluding_asset_id=asset_id)
            values: dict[str, Any] = {
                "revision": current.revision + 1,
                "name": new_name,
                "kind": current.kind if kind is None else kind,
                "scope": current.scope if scope is None else scope,
                "created_at": datetime.now(UTC),
            }
            if shot_id_was_set:
                values["shot_id"] = shot_id
            if shot_ids is not None:
                values["shot_ids"] = shot_ids
            updated = current.model_copy(update=values)
            # model_copy does not rerun Pydantic validators.
            updated = ProjectAsset.model_validate(updated.model_dump(mode="python"))
            self._insert(connection, updated)
        return updated

    def retire_asset(self, project_id: str, asset_id: str) -> None:
        with self._transaction(immediate=True) as connection:
            current = self._get_current(connection, project_id, asset_id)
            retired = current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "state": ProjectAssetState.RETIRED,
                    "created_at": datetime.now(UTC),
                }
            )
            self._insert(connection, retired)

    def preview_for(self, project_id: str, asset_id: str) -> Path:
        asset = self.get_asset(project_id, asset_id)
        if asset.preview_sha256 is None:
            raise ProjectAssetNotFoundError(asset_id)
        path = self.preview_path(asset.preview_sha256)
        if not path.is_file():
            raise ProjectAssetNotFoundError(asset_id)
        return path

    def media_for(self, project_id: str, asset_id: str) -> Path:
        asset = self.get_asset(project_id, asset_id)
        if asset.sha256 is None:
            raise ProjectAssetNotFoundError(asset_id)
        path = self.blob_path(asset.sha256)
        if not path.is_file():
            raise ProjectAssetNotFoundError(asset_id)
        return path

    def blob_path(self, sha256: str) -> Path:
        return self.blob_root / sha256[:2] / sha256

    def preview_path(self, sha256: str) -> Path:
        return self.preview_root / sha256[:2] / f"{sha256}.jpg"

    @staticmethod
    def _validate_image(content: bytes) -> tuple[Image.Image, str, int, int]:
        if not content:
            raise InvalidProjectImageError("uploaded image is empty")
        try:
            with Image.open(io.BytesIO(content)) as probe:
                image_format = probe.format
                width, height = probe.size
                probe.verify()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidProjectImageError("supported image formats are JPEG, PNG, and WebP")
            if (
                width < 1
                or height < 1
                or width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise InvalidProjectImageError("image dimensions exceed the safety limit")
            image = Image.open(io.BytesIO(content))
            image.seek(0)
            image.load()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise InvalidProjectImageError("file content is not a valid supported image") from exc
        return image, SUPPORTED_IMAGE_FORMATS[image_format], width, height

    def _validate_upload(
        self, content: bytes, original_name: str
    ) -> tuple[
        ProjectAssetMediaKind,
        str,
        int | None,
        int | None,
        float | None,
        float | None,
        bool | None,
        bytes | None,
    ]:
        try:
            image, mime_type, width, height = self._validate_image(content)
        except InvalidProjectImageError:
            mime_type = self._detect_av_mime(content, original_name)
            media_kind = (
                ProjectAssetMediaKind.VIDEO
                if mime_type.startswith("video/")
                else ProjectAssetMediaKind.AUDIO
            )
            metadata = self._probe_av(content, original_name, media_kind)
            return (
                media_kind,
                mime_type,
                metadata["width"],
                metadata["height"],
                metadata["duration_seconds"],
                metadata["frame_rate"],
                metadata["has_audio"],
                None,
            )
        return (
            ProjectAssetMediaKind.IMAGE,
            mime_type,
            width,
            height,
            None,
            None,
            None,
            self._make_preview(image),
        )

    def _probe_av(
        self,
        content: bytes,
        original_name: str,
        media_kind: ProjectAssetMediaKind,
    ) -> dict[str, Any]:
        suffix = Path(original_name).suffix or ".bin"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(content)
                temp_path = Path(handle.name)
            completed = subprocess.run(
                [
                    self.ffprobe_binary,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type,width,height,avg_frame_rate",
                    "-of",
                    "json",
                    str(temp_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InvalidProjectImageError(
                f"ffprobe could not inspect reference media: {exc}"
            ) from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "invalid media container"
            raise InvalidProjectImageError(f"reference media could not be decoded: {detail}")
        try:
            payload = json.loads(completed.stdout)
            streams = payload.get("streams", [])
            duration = float(payload.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InvalidProjectImageError("ffprobe returned invalid reference metadata") from exc
        if duration <= 0:
            raise InvalidProjectImageError("reference media must have a positive duration")
        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"), None
        )
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        if media_kind == ProjectAssetMediaKind.AUDIO:
            if not has_audio:
                raise InvalidProjectImageError("audio reference has no decodable audio stream")
            return {
                "width": None,
                "height": None,
                "duration_seconds": duration,
                "frame_rate": None,
                "has_audio": True,
            }
        if video_stream is None:
            raise InvalidProjectImageError("video reference has no decodable video stream")
        if not H3_REFERENCE_VIDEO_MIN_SECONDS <= duration <= H3_REFERENCE_VIDEO_MAX_SECONDS:
            raise InvalidProjectImageError("H3 reference videos must be between 2 and 15 seconds")
        numerator, separator, denominator = str(
            video_stream.get("avg_frame_rate") or "0/1"
        ).partition("/")
        try:
            frame_rate = float(numerator) / float(denominator) if separator else float(numerator)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise InvalidProjectImageError("video reference frame rate is invalid") from exc
        if abs(frame_rate - H3_REFERENCE_VIDEO_FPS) > 0.01:
            raise InvalidProjectImageError("H3 reference videos must be exactly 24 fps")
        width = int(video_stream.get("width") or 0) or None
        height = int(video_stream.get("height") or 0) or None
        if width is None or height is None:
            raise InvalidProjectImageError("video reference dimensions are invalid")
        return {
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "frame_rate": frame_rate,
            "has_audio": has_audio,
        }

    @staticmethod
    def _detect_av_mime(content: bytes, original_name: str) -> str:
        if not content:
            raise InvalidProjectImageError("uploaded media is empty")
        guessed = mimetypes.guess_type(original_name)[0] or ""
        header = content[:64]
        if header.startswith(b"\x1aE\xdf\xa3") and guessed in {"video/webm", "audio/webm"}:
            return guessed
        if len(header) >= 12 and header[4:8] == b"ftyp" and guessed in {
            "video/mp4",
            "video/quicktime",
            "audio/mp4",
            "audio/x-m4a",
        }:
            return guessed
        if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
            return "audio/wav"
        if header.startswith((b"ID3", b"fLaC", b"OggS")) or header[:2] in {
            b"\xff\xfb",
            b"\xff\xf3",
            b"\xff\xf2",
        }:
            return {
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
            }.get(Path(original_name).suffix.casefold(), "audio/mpeg")
        raise InvalidProjectImageError(
            "file must be a supported image, MP4/MOV/WebM video, or WAV/MP3/FLAC/OGG audio"
        )

    @staticmethod
    def _make_preview(image: Image.Image) -> bytes:
        preview = image.convert("RGB")
        preview.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
        target = io.BytesIO()
        preview.save(target, format="JPEG", quality=85, optimize=True)
        return target.getvalue()

    @staticmethod
    def _write_blob_once(path: Path, content: bytes) -> None:
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            except OSError:
                if not path.exists():
                    temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _insert(connection: sqlite3.Connection, asset: ProjectAsset) -> None:
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
                _normalized_name(asset.name),
                asset.state.value,
                asset.sha256,
                _canonical_json(asset.model_dump(mode="json")),
                asset.created_at.timestamp(),
            ),
        )

    def _get_current(
        self, connection: sqlite3.Connection, project_id: str, asset_id: str
    ) -> ProjectAsset:
        row = connection.execute(
            """
            SELECT payload_json FROM project_asset_revisions
            WHERE project_id = ? AND asset_id = ?
            ORDER BY revision DESC LIMIT 1
            """,
            (project_id, asset_id),
        ).fetchone()
        if row is None:
            raise ProjectAssetNotFoundError(asset_id)
        asset = ProjectAsset.model_validate_json(row["payload_json"])
        if asset.state == ProjectAssetState.RETIRED:
            raise ProjectAssetNotFoundError(asset_id)
        return asset

    @staticmethod
    def _ensure_unique_name(
        connection: sqlite3.Connection,
        project_id: str,
        name: str,
        *,
        excluding_asset_id: str | None = None,
    ) -> None:
        normalized = _normalized_name(name)
        if not normalized:
            raise ValueError("asset name must be non-empty")
        row = connection.execute(
            """
            SELECT current.asset_id
            FROM project_asset_revisions current
            WHERE current.project_id = ? AND current.normalized_name = ?
              AND current.state != ?
              AND current.revision = (
                SELECT MAX(candidate.revision)
                FROM project_asset_revisions candidate
                WHERE candidate.asset_id = current.asset_id
              )
              AND (? IS NULL OR current.asset_id != ?)
            LIMIT 1
            """,
            (
                project_id,
                normalized,
                ProjectAssetState.RETIRED.value,
                excluding_asset_id,
                excluding_asset_id,
            ),
        ).fetchone()
        if row is not None:
            raise DuplicateProjectAssetNameError(name)
