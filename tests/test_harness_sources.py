from __future__ import annotations

import httpx
import pytest

from ai_video_generator.services.harness_sources import (
    HarnessSourceInstallError,
    install_h3_harness_source,
    validate_h3_harness_source,
)


@pytest.mark.asyncio
async def test_official_install_fetches_only_complete_h3_skill_subtree(tmp_path) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if "git/trees" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {"path": "skills/h3-prompt-writing/SKILL.md", "type": "blob"},
                        {
                            "path": "skills/h3-prompt-writing/references/base-en.txt",
                            "type": "blob",
                        },
                        {
                            "path": "skills/h3-prompt-writing/references/ref-en.txt",
                            "type": "blob",
                        },
                        {"path": "skills/other/SKILL.md", "type": "blob"},
                        {"path": "skills/h3-prompt-writing/tool.py", "type": "blob"},
                    ]
                },
            )
        return httpx.Response(200, content=b"pinned document")

    installed = await install_h3_harness_source(
        "official", tmp_path, transport=httpx.MockTransport(handler)
    )

    assert installed.files == (
        "skills/h3-prompt-writing/SKILL.md",
        "skills/h3-prompt-writing/references/base-en.txt",
        "skills/h3-prompt-writing/references/ref-en.txt",
    )
    assert len(requested) == 3
    assert not any("git/trees" in value for value in requested)
    assert set(installed.file_sha256) == set(installed.files)
    assert validate_h3_harness_source(installed.install_root) == installed


@pytest.mark.asyncio
async def test_official_install_rejects_incomplete_pinned_subtree(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/SKILL.md"):
            return httpx.Response(200, content=b"# Skill")
        return httpx.Response(404)

    with pytest.raises(HarnessSourceInstallError, match="failed to download"):
        await install_h3_harness_source(
            "official", tmp_path, transport=httpx.MockTransport(handler)
        )


@pytest.mark.asyncio
async def test_manifest_validation_rejects_modified_document(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "git/trees" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "tree": [
                        {"path": "skills/h3-prompt-writing/SKILL.md", "type": "blob"},
                        {
                            "path": "skills/h3-prompt-writing/references/base-en.txt",
                            "type": "blob",
                        },
                        {
                            "path": "skills/h3-prompt-writing/references/ref-en.txt",
                            "type": "blob",
                        },
                    ]
                },
            )
        return httpx.Response(200, content=b"pinned document")

    installed = await install_h3_harness_source(
        "official", tmp_path, transport=httpx.MockTransport(handler)
    )
    skill = (
        tmp_path
        / "harness-sources"
        / "official"
        / installed.commit
        / "skills"
        / "h3-prompt-writing"
        / "SKILL.md"
    )
    skill.write_text("modified", encoding="utf-8")

    with pytest.raises(HarnessSourceInstallError, match="hash mismatch"):
        validate_h3_harness_source(installed.install_root)
