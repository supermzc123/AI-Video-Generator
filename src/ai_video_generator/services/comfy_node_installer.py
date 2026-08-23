from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path

from ai_video_generator.api_models import ComfyNodeInstallResult, ComfyNodeInstallStep

CommandRunner = Callable[[tuple[str, ...]], Awaitable[tuple[int, str, str]]]

INSTALL_COMPONENTS = (
    ("avg", "项目专属节点", "install-avg-comfyui-nodes.ps1"),
    ("motion_context", "H3 Motion Context", "install-h3-motion-context.ps1"),
    ("turbo", "MiniMax H3 Turbo", "install-h3-turbo-nodes.ps1"),
)


class ComfyNodeInstallValidationError(ValueError):
    pass


def validate_comfyui_installation(root: Path) -> Path:
    try:
        resolved = root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ComfyNodeInstallValidationError(f"ComfyUI文件夹不存在：{root}") from exc
    if not resolved.is_dir():
        raise ComfyNodeInstallValidationError(f"ComfyUI路径不是文件夹：{resolved}")
    if not (resolved / "main.py").is_file() or not (resolved / "custom_nodes").is_dir():
        raise ComfyNodeInstallValidationError(
            "所选文件夹不是ComfyUI根目录；应包含main.py和custom_nodes"
        )
    official_h3 = resolved / "comfy_extras" / "nodes_minimax_h3.py"
    if not official_h3.is_file():
        raise ComfyNodeInstallValidationError(
            "当前ComfyUI版本没有官方MiniMax H3节点，请先更新ComfyUI本体"
        )
    return resolved


async def _run_command(command: tuple[str, ...]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=(0x08000000 if os.name == "nt" else 0),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, "", "节点安装超过10分钟，已停止安装进程"
    return (
        int(process.returncode or 0),
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _summary(stdout: str, stderr: str, *, succeeded: bool) -> str:
    lines = [line.strip() for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    if not lines:
        return "安装完成" if succeeded else "安装命令失败且没有返回详情"
    return lines[-1][-2000:]


async def install_required_comfy_nodes(
    *,
    comfyui_root: Path,
    scripts_root: Path,
    proxy: str | None,
    runner: CommandRunner = _run_command,
) -> ComfyNodeInstallResult:
    resolved_root = validate_comfyui_installation(comfyui_root)
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        raise ComfyNodeInstallValidationError("未找到PowerShell，无法运行Windows节点安装器")

    steps: list[ComfyNodeInstallStep] = [
        ComfyNodeInstallStep(
            component="h3_core",
            label="ComfyUI官方MiniMax H3节点",
            succeeded=True,
            message="已在ComfyUI本体中检测到官方H3节点",
        )
    ]
    for component, label, filename in INSTALL_COMPONENTS:
        script = (scripts_root / filename).resolve()
        if not script.is_file():
            steps.append(
                ComfyNodeInstallStep(
                    component=component,
                    label=label,
                    succeeded=False,
                    message=f"安装器文件缺失：{filename}",
                )
            )
            continue
        command = (
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-ComfyUIRoot",
            str(resolved_root),
        )
        if component in {"motion_context", "turbo"} and proxy:
            command += ("-Proxy", proxy)
        return_code, stdout, stderr = await runner(command)
        succeeded = return_code == 0
        steps.append(
            ComfyNodeInstallStep(
                component=component,
                label=label,
                succeeded=succeeded,
                message=_summary(stdout, stderr, succeeded=succeeded),
            )
        )
    return ComfyNodeInstallResult(
        succeeded=all(step.succeeded for step in steps),
        restart_required=any(step.succeeded for step in steps if step.component != "h3_core"),
        comfyui_root=str(resolved_root),
        steps=tuple(steps),
    )
