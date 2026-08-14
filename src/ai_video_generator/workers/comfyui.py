from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

H3_NODE_PATTERN = re.compile(r"minimax|h3", re.IGNORECASE)
H3_MODEL_PATTERN = re.compile(r"minimax|h3|qwen3vl", re.IGNORECASE)
MODEL_DIRECTORIES = ("diffusion_models", "unet", "text_encoders", "clip", "vae", "loras")


class ComfyUIInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configured_root: str | None
    root_exists: bool
    version: str | None = None
    commit: str | None = None
    custom_nodes: tuple[str, ...] = ()
    h3_custom_nodes: tuple[str, ...] = ()
    h3_model_files: tuple[str, ...] = ()
    motion_director_installed: bool = False
    conflicting_context_nodes: tuple[str, ...] = ()


class ComfyUICapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_online: bool
    base_url: str
    inventory: ComfyUIInventory
    h3_node_ids: tuple[str, ...] = ()
    system_stats: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


class ComfyUIAdapter:
    """Read-only ComfyUI adapter used during the initial build stage."""

    def __init__(
        self,
        root: Path | None,
        base_url: str,
        timeout_seconds: float = 3.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.root = root
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def inspect_local(self) -> ComfyUIInventory:
        root = self.root
        if root is None:
            return ComfyUIInventory(configured_root=None, root_exists=False)

        root = root.expanduser()
        if not root.is_dir():
            return ComfyUIInventory(configured_root=str(root), root_exists=False)

        custom_nodes_root = root / "custom_nodes"
        custom_nodes = tuple(
            sorted(
                path.name
                for path in custom_nodes_root.iterdir()
                if path.is_dir() and not path.name.startswith((".", "__"))
            )
        ) if custom_nodes_root.is_dir() else ()
        lowered = {name: name.casefold() for name in custom_nodes}
        h3_custom_nodes = tuple(
            name for name in custom_nodes if H3_NODE_PATTERN.search(lowered[name])
        )
        motion_director_installed = any(
            "motion-director" in value or "motion_director" in value
            for value in lowered.values()
        )
        conflicting_context_nodes = tuple(
            name
            for name, value in lowered.items()
            if "h3-motion-context" in value or "contex-loop" in value
        )

        return ComfyUIInventory(
            configured_root=str(root.resolve()),
            root_exists=True,
            version=_read_comfyui_version(root),
            commit=_read_git_commit(root / ".git"),
            custom_nodes=custom_nodes,
            h3_custom_nodes=h3_custom_nodes,
            h3_model_files=_find_h3_model_files(root / "models"),
            motion_director_installed=motion_director_installed,
            conflicting_context_nodes=conflicting_context_nodes,
        )

    async def capabilities(self) -> ComfyUICapabilities:
        inventory = self.inspect_local()
        warnings = _inventory_warnings(inventory)
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
                trust_env=False,
            ) as client:
                stats_response = await client.get("/system_stats")
                stats_response.raise_for_status()
                nodes_response = await client.get("/object_info")
                nodes_response.raise_for_status()
                system_stats = stats_response.json()
                object_info = nodes_response.json()
        except (httpx.HTTPError, ValueError) as exc:
            warnings.append(f"ComfyUI server is unavailable: {exc}")
            return ComfyUICapabilities(
                server_online=False,
                base_url=self.base_url,
                inventory=inventory,
                warnings=tuple(warnings),
            )

        h3_node_ids = tuple(
            sorted(node_id for node_id in object_info if H3_NODE_PATTERN.search(node_id))
        )
        return ComfyUICapabilities(
            server_online=True,
            base_url=self.base_url,
            inventory=inventory,
            h3_node_ids=h3_node_ids,
            system_stats=system_stats,
            warnings=tuple(warnings),
        )


def _read_comfyui_version(root: Path) -> str | None:
    version_file = root / "comfyui_version.py"
    if not version_file.is_file():
        return None
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        version_file.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _read_git_commit(git_dir: Path) -> str | None:
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None

    ref = head.removeprefix("ref: ")
    loose_ref = git_dir.joinpath(*ref.split("/"))
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip() or None

    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        suffix = f" {ref}"
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.endswith(suffix):
                return line.split(" ", 1)[0]
    return None


def _find_h3_model_files(models_root: Path) -> tuple[str, ...]:
    matches: list[str] = []
    for directory in MODEL_DIRECTORIES:
        candidate = models_root / directory
        if not candidate.is_dir():
            continue
        for path in candidate.rglob("*"):
            if path.is_file() and H3_MODEL_PATTERN.search(path.name):
                matches.append(path.relative_to(models_root).as_posix())
    return tuple(sorted(matches))


def _inventory_warnings(inventory: ComfyUIInventory) -> list[str]:
    warnings: list[str] = []
    if not inventory.root_exists:
        warnings.append("Configured ComfyUI root does not exist")
        return warnings
    if not inventory.motion_director_installed:
        warnings.append("Motion Director is not installed")
    if inventory.conflicting_context_nodes:
        warnings.append(
            "Conflicting Motion Context nodes detected: "
            + ", ".join(inventory.conflicting_context_nodes)
        )
    if not inventory.h3_model_files:
        warnings.append("No H3-related model files were detected")
    return warnings
