from pathlib import Path

import pytest

from ai_video_generator.services.comfy_node_installer import (
    ComfyNodeInstallValidationError,
    install_required_comfy_nodes,
    validate_comfyui_installation,
)


def _comfyui_root(tmp_path: Path, *, official_h3: bool = True) -> Path:
    root = tmp_path / "ComfyUI"
    (root / "custom_nodes").mkdir(parents=True)
    (root / "comfy_extras").mkdir()
    (root / "main.py").write_text("# ComfyUI", encoding="utf-8")
    if official_h3:
        (root / "comfy_extras" / "nodes_minimax_h3.py").write_text(
            "# official MiniMax H3 nodes", encoding="utf-8"
        )
    return root


@pytest.mark.asyncio
async def test_installer_runs_fixed_components_without_sage(
    tmp_path: Path,
) -> None:
    root = _comfyui_root(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in (
        "install-avg-comfyui-nodes.ps1",
        "install-h3-motion-context.ps1",
        "install-h3-turbo-nodes.ps1",
    ):
        (scripts / name).write_text("Write-Output installed", encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    async def runner(command: tuple[str, ...]) -> tuple[int, str, str]:
        commands.append(command)
        return 0, "installed and verified", ""

    result = await install_required_comfy_nodes(
        comfyui_root=root,
        scripts_root=scripts,
        proxy="http://127.0.0.1:10808",
        runner=runner,
    )

    assert result.succeeded
    assert result.restart_required
    assert [step.component for step in result.steps] == [
        "h3_core",
        "avg",
        "motion_context",
        "turbo",
    ]
    assert len(commands) == 3
    assert "-Proxy" not in commands[0]
    assert commands[1][-2:] == ("-Proxy", "http://127.0.0.1:10808")
    assert commands[2][-2:] == ("-Proxy", "http://127.0.0.1:10808")
    assert all("sage" not in " ".join(command).casefold() for command in commands)


@pytest.mark.asyncio
async def test_installer_returns_partial_failure_without_hiding_success(
    tmp_path: Path,
) -> None:
    root = _comfyui_root(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in (
        "install-avg-comfyui-nodes.ps1",
        "install-h3-motion-context.ps1",
        "install-h3-turbo-nodes.ps1",
    ):
        (scripts / name).write_text("", encoding="utf-8")
    calls = 0

    async def runner(_: tuple[str, ...]) -> tuple[int, str, str]:
        nonlocal calls
        calls += 1
        return (1, "", "clone failed") if calls == 2 else (0, "verified", "")

    result = await install_required_comfy_nodes(
        comfyui_root=root,
        scripts_root=scripts,
        proxy=None,
        runner=runner,
    )

    assert not result.succeeded
    assert result.steps[1].succeeded
    assert not result.steps[2].succeeded
    assert result.steps[2].message == "clone failed"
    assert result.steps[3].succeeded


def test_installer_requires_official_h3_nodes_in_comfyui_core(tmp_path: Path) -> None:
    root = _comfyui_root(tmp_path, official_h3=False)

    with pytest.raises(ComfyNodeInstallValidationError, match="官方MiniMax H3"):
        validate_comfyui_installation(root)
