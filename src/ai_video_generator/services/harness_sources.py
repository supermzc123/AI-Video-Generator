from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ai_video_generator.domain import (
    H3_COMMUNITY_SKILLS_COMMIT,
    H3_OFFICIAL_SKILL_COMMIT,
)


class HarnessSourceInstallError(RuntimeError):
    pass


class InstalledHarnessSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    repository_url: str
    commit: str
    archive_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    files: tuple[str, ...]
    file_sha256: dict[str, str] = Field(default_factory=dict)
    install_root: str
    redistribution_allowed: bool


_SOURCES = {
    "community": {
        "owner": "unknowlei",
        "repository": "minimax-h3-opencode-skills",
        "commit": H3_COMMUNITY_SKILLS_COMMIT,
        "redistribution_allowed": True,
    },
    "official": {
        "owner": "MiniMax-AI",
        "repository": "MiniMax-H3",
        "commit": H3_OFFICIAL_SKILL_COMMIT,
        "redistribution_allowed": False,
    },
}
_ALLOWED_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".txt"}
_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 100 * 1024 * 1024
_OFFICIAL_SKILL_ROOT = PurePosixPath("skills/h3-prompt-writing")
_OFFICIAL_REQUIRED_FILES = (
    "skills/h3-prompt-writing/SKILL.md",
    "skills/h3-prompt-writing/references/base-en.txt",
    "skills/h3-prompt-writing/references/ref-en.txt",
)


async def install_h3_harness_source(
    source_id: str,
    data_root: str | Path,
    *,
    proxy: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> InstalledHarnessSource:
    source = _SOURCES.get(source_id)
    if source is None:
        raise HarnessSourceInstallError(f"unknown H3 harness source: {source_id}")
    owner = str(source["owner"])
    repository = str(source["repository"])
    commit = str(source["commit"])
    if source_id == "official":
        return await _install_official_skill_documents(
            source_id,
            owner,
            repository,
            commit,
            data_root,
            proxy=proxy,
            transport=transport,
        )
    url = f"https://codeload.github.com/{owner}/{repository}/zip/{commit}"
    try:
        async with httpx.AsyncClient(
            proxy=proxy, transport=transport, timeout=60, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HarnessSourceInstallError(f"failed to download pinned source: {exc}") from exc
    archive = response.content
    if not archive or len(archive) > _MAX_ARCHIVE_BYTES:
        raise HarnessSourceInstallError("pinned source archive is empty or exceeds size limit")

    destination = Path(data_root) / "harness-sources" / source_id / commit
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    extracted_bytes = 0
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                path = PurePosixPath(member.filename)
                relative_parts = path.parts[1:]
                if (
                    not relative_parts
                    or any(part in {"", ".", ".."} for part in relative_parts)
                    or path.suffix.lower() not in _ALLOWED_SUFFIXES
                ):
                    continue
                extracted_bytes += member.file_size
                if extracted_bytes > _MAX_EXTRACTED_BYTES:
                    raise HarnessSourceInstallError("extracted harness source exceeds size limit")
                relative = Path(*relative_parts)
                target = (destination / relative).resolve()
                if destination.resolve() not in target.parents:
                    raise HarnessSourceInstallError("archive contains an unsafe path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(member))
                extracted.append(relative.as_posix())
    except zipfile.BadZipFile as exc:
        raise HarnessSourceInstallError("pinned source archive is not a valid ZIP") from exc
    if not any(Path(item).name.lower() == "skill.md" for item in extracted):
        raise HarnessSourceInstallError("pinned source contains no SKILL.md")

    installed = InstalledHarnessSource(
        source_id=source_id,
        repository_url=f"https://github.com/{owner}/{repository}",
        commit=commit,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        files=tuple(sorted(extracted)),
        file_sha256={
            path: hashlib.sha256(
                (destination / Path(*PurePosixPath(path).parts)).read_bytes()
            ).hexdigest()
            for path in sorted(extracted)
        },
        install_root=str(destination),
        redistribution_allowed=bool(source["redistribution_allowed"]),
    )
    (destination / "install-manifest.json").write_text(
        json.dumps(installed.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return installed


async def _install_official_skill_documents(
    source_id: str,
    owner: str,
    repository: str,
    commit: str,
    data_root: str | Path,
    *,
    proxy: str | None,
    transport: httpx.AsyncBaseTransport | None,
) -> InstalledHarnessSource:
    try:
        async with httpx.AsyncClient(
            proxy=proxy,
            transport=transport,
            timeout=60,
            follow_redirects=True,
            headers={"Accept": "application/vnd.github+json"},
        ) as client:
            paths = sorted(_OFFICIAL_REQUIRED_FILES)
            documents: dict[str, bytes] = {}
            for path in paths:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repository}/{commit}/{path}"
                response = await client.get(raw_url)
                response.raise_for_status()
                documents[path] = response.content
    except httpx.HTTPError as exc:
        raise HarnessSourceInstallError(
            f"failed to download pinned official skill documents: {exc}"
        ) from exc

    destination = Path(data_root) / "harness-sources" / source_id / commit
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    for path, content in documents.items():
        total += len(content)
        if total > _MAX_EXTRACTED_BYTES:
            raise HarnessSourceInstallError("official skill documents exceed size limit")
        relative = Path(*PurePosixPath(path).parts)
        target = (destination / relative).resolve()
        if destination.resolve() not in target.parents:
            raise HarnessSourceInstallError("official tree contains an unsafe path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)

    installed = InstalledHarnessSource(
        source_id=source_id,
        repository_url=f"https://github.com/{owner}/{repository}",
        commit=commit,
        archive_sha256=digest.hexdigest(),
        files=tuple(documents),
        file_sha256={
            path: hashlib.sha256(content).hexdigest() for path, content in documents.items()
        },
        install_root=str(destination),
        redistribution_allowed=False,
    )
    (destination / "install-manifest.json").write_text(
        json.dumps(installed.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return installed


def validate_h3_harness_source(
    install_root: str | Path,
    *,
    source_id: str | None = None,
) -> InstalledHarnessSource:
    """Validate a persisted manifest and every installed source document."""
    root = Path(install_root)
    manifest_path = root / "install-manifest.json"
    try:
        installed = InstalledHarnessSource.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise HarnessSourceInstallError("harness install manifest is missing or invalid") from exc
    if Path(installed.install_root).resolve() != root.resolve():
        raise HarnessSourceInstallError("harness install root does not match its manifest")
    if source_id is not None and installed.source_id != source_id:
        raise HarnessSourceInstallError("harness source ID does not match its manifest")
    if installed.source_id == "official":
        if installed.commit != H3_OFFICIAL_SKILL_COMMIT:
            raise HarnessSourceInstallError("official harness commit is not approved")
        missing = sorted(set(_OFFICIAL_REQUIRED_FILES) - set(installed.files))
        if missing:
            raise HarnessSourceInstallError(
                "official harness source is incomplete: " + ", ".join(missing)
            )
    elif installed.source_id == "community" and installed.commit != H3_COMMUNITY_SKILLS_COMMIT:
        raise HarnessSourceInstallError("community harness commit is not approved")

    if not installed.file_sha256 or set(installed.file_sha256) != set(installed.files):
        raise HarnessSourceInstallError("harness manifest does not contain complete file hashes")
    for relative_name in installed.files:
        relative = PurePosixPath(relative_name)
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise HarnessSourceInstallError("harness manifest contains an unsafe path")
        target = (root / Path(*relative.parts)).resolve()
        if root.resolve() not in target.parents:
            raise HarnessSourceInstallError("harness manifest points outside its install root")
        try:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise HarnessSourceInstallError(
                f"harness document is missing: {relative_name}"
            ) from exc
        if digest != installed.file_sha256[relative_name]:
            raise HarnessSourceInstallError(f"harness document hash mismatch: {relative_name}")
    return installed


def _is_official_h3_document(path_value: str) -> bool:
    path = PurePosixPath(path_value)
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        return False
    try:
        path.relative_to(_OFFICIAL_SKILL_ROOT)
    except ValueError:
        return False
    return True
