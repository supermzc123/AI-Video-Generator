from __future__ import annotations

import asyncio
import hashlib
import json
import re
from configparser import ConfigParser
from configparser import Error as ConfigParserError
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .h3_policy import (
    OFFICIAL_H3_TURBO_MODULES,
    OFFICIAL_H3_TURBO_REPOSITORY,
    PINNED_H3_TURBO_COMMIT,
    TURBO_LORA_NODE_TYPE,
    TURBO_SAMPLER_NODE_TYPE,
)
from .motion_director import (
    MOTION_CONTEXT_NODE_TYPES,
    MOTION_CONTEXT_REPOSITORY,
    PINNED_MOTION_CONTEXT_COMMIT,
)

H3_NODE_PATTERN = re.compile(r"minimax|h3", re.IGNORECASE)
H3_MODEL_PATTERN = re.compile(r"minimax|h3|qwen3vl", re.IGNORECASE)
MODEL_DIRECTORIES = ("diffusion_models", "unet", "text_encoders", "clip", "vae", "loras")
MODEL_FILE_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}


class ComfyUIInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configured_root: str | None
    root_exists: bool
    version: str | None = None
    commit: str | None = None
    custom_nodes: tuple[str, ...] = ()
    h3_custom_nodes: tuple[str, ...] = ()
    h3_model_files: tuple[str, ...] = ()
    motion_context_path: str | None = None
    motion_context_commit: str | None = None
    motion_context_verified: bool = False
    unverified_motion_context_nodes: tuple[str, ...] = ()
    conflicting_context_nodes: tuple[str, ...] = ()
    official_h3_turbo_path: str | None = None
    official_h3_turbo_commit: str | None = None
    official_h3_turbo_verified: bool = False
    unverified_h3_turbo_nodes: tuple[str, ...] = ()


class ComfyUICapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_online: bool
    base_url: str
    inventory: ComfyUIInventory
    h3_node_ids: tuple[str, ...] = ()
    h3_turbo_provider_modules: tuple[str, ...] = ()
    motion_context_provider_modules: tuple[str, ...] = ()
    motion_context_runtime_verified: bool = False
    system_stats: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    full_pipeline_ready: bool = False
    execution_blockers: tuple[str, ...] = ()


class ComfyUIObjectInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_schema_sha256: str
    nodes: dict[str, dict[str, Any]]


class ComfyUIPromptSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    number: int | float | None = None
    node_errors: dict[str, Any] = Field(default_factory=dict)


class ComfyUICancelResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    was_running: bool = False
    was_pending: bool = False
    interrupted: bool = False
    deleted: bool = False


class ComfyUIAdapter:
    """Small, typed adapter around the ComfyUI HTTP execution surface."""

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

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
            trust_env=False,
        )

    def inspect_local(self) -> ComfyUIInventory:
        root = self.root
        if root is None:
            return ComfyUIInventory(configured_root=None, root_exists=False)

        root = root.expanduser()
        if not root.is_dir():
            return ComfyUIInventory(configured_root=str(root), root_exists=False)

        custom_nodes_root = root / "custom_nodes"
        custom_nodes = (
            tuple(
                sorted(
                    path.name
                    for path in custom_nodes_root.iterdir()
                    if path.is_dir() and not path.name.startswith((".", "__"))
                )
            )
            if custom_nodes_root.is_dir()
            else ()
        )
        lowered = {name: name.casefold() for name in custom_nodes}
        h3_custom_nodes = tuple(
            name for name in custom_nodes if H3_NODE_PATTERN.search(lowered[name])
        )
        (
            motion_context_path,
            motion_context_commit,
            unverified_motion_context_nodes,
        ) = _inspect_motion_context_packages(custom_nodes_root)
        conflicting_context_nodes = tuple(
            sorted(
                set(unverified_motion_context_nodes)
                | {
                    name
                    for name, value in lowered.items()
                    if "contex-loop" in value or "motion-director" in value
                }
            )
        )
        (
            official_h3_turbo_path,
            official_h3_turbo_commit,
            unverified_h3_turbo_nodes,
        ) = _inspect_h3_turbo_packages(custom_nodes_root)

        return ComfyUIInventory(
            configured_root=str(root.resolve()),
            root_exists=True,
            version=_read_comfyui_version(root),
            commit=_read_git_commit(root / ".git"),
            custom_nodes=custom_nodes,
            h3_custom_nodes=h3_custom_nodes,
            h3_model_files=_find_h3_model_files(root / "models"),
            motion_context_path=motion_context_path,
            motion_context_commit=motion_context_commit,
            motion_context_verified=(motion_context_commit == PINNED_MOTION_CONTEXT_COMMIT),
            unverified_motion_context_nodes=unverified_motion_context_nodes,
            conflicting_context_nodes=conflicting_context_nodes,
            official_h3_turbo_path=official_h3_turbo_path,
            official_h3_turbo_commit=official_h3_turbo_commit,
            official_h3_turbo_verified=(official_h3_turbo_commit == PINNED_H3_TURBO_COMMIT),
            unverified_h3_turbo_nodes=unverified_h3_turbo_nodes,
        )

    def list_local_models(self, *directories: str) -> tuple[str, ...]:
        if self.root is None:
            return ()
        models_root = self.root.expanduser() / "models"
        matches: set[str] = set()
        for directory in directories:
            candidate = models_root / directory
            if not candidate.is_dir():
                continue
            for path in candidate.rglob("*"):
                if path.is_file() and path.suffix.casefold() in MODEL_FILE_SUFFIXES:
                    matches.add(path.relative_to(candidate).as_posix())
        return tuple(sorted(matches, key=str.casefold))

    async def capabilities(self) -> ComfyUICapabilities:
        inventory = self.inspect_local()
        warnings = _inventory_warnings(inventory)
        try:
            async with self._client() as client:
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
        turbo_provider_modules = tuple(
            sorted(
                {
                    str(object_info[node_type].get("python_module") or "unknown")
                    for node_type in (TURBO_LORA_NODE_TYPE, TURBO_SAMPLER_NODE_TYPE)
                    if isinstance(object_info.get(node_type), dict)
                }
            )
        )
        if any(module not in OFFICIAL_H3_TURBO_MODULES for module in turbo_provider_modules):
            warnings.append(
                "The running ComfyUI process exposes an unverified MiniMax H3 Turbo provider: "
                + ", ".join(turbo_provider_modules)
            )
        motion_context_provider_modules = tuple(
            sorted(
                {
                    str(object_info[node_type].get("python_module") or "unknown")
                    for node_type in MOTION_CONTEXT_NODE_TYPES
                    if isinstance(object_info.get(node_type), dict)
                }
            )
        )
        expected_motion_module = (
            f"custom_nodes.{inventory.motion_context_path}"
            if inventory.motion_context_path
            else None
        )
        motion_context_runtime_verified = (
            inventory.motion_context_verified
            and all(node_type in object_info for node_type in MOTION_CONTEXT_NODE_TYPES)
            and motion_context_provider_modules == (expected_motion_module,)
        )
        if inventory.motion_context_verified and not motion_context_runtime_verified:
            warnings.append(
                "Pinned H3 Motion Context is present on disk but the running ComfyUI "
                "process does not expose its verified five-node contract"
            )
        return ComfyUICapabilities(
            server_online=True,
            base_url=self.base_url,
            inventory=inventory,
            h3_node_ids=h3_node_ids,
            h3_turbo_provider_modules=turbo_provider_modules,
            motion_context_provider_modules=motion_context_provider_modules,
            motion_context_runtime_verified=motion_context_runtime_verified,
            system_stats=system_stats,
            warnings=tuple(warnings),
        )

    async def get_object_info(self, node_type: str | None = None) -> ComfyUIObjectInfo:
        """Fetch and validate the node schema used to type-check API workflows."""
        path = "/object_info" if node_type is None else f"/object_info/{node_type}"
        response: httpx.Response | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with self._client() as client:
                    response = await client.get(path)
                    response.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        if response is None:
            if isinstance(last_error, httpx.TimeoutException):
                raise httpx.ReadTimeout(
                    f"ComfyUI {path} did not respond after 3 attempts"
                ) from last_error
            if last_error is not None:
                raise last_error
            raise httpx.NetworkError(f"ComfyUI {path} request failed")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("ComfyUI object_info response must be an object")
        nodes = _normalize_object_info(payload, requested_node_type=node_type)
        return ComfyUIObjectInfo(
            node_schema_sha256=_canonical_sha256(nodes),
            nodes=nodes,
        )

    async def submit_prompt(
        self,
        workflow: dict[str, dict[str, Any]],
        *,
        client_id: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> ComfyUIPromptSubmission:
        if not workflow:
            raise ValueError("ComfyUI workflow must not be empty")
        body: dict[str, Any] = {"prompt": workflow}
        if client_id:
            body["client_id"] = client_id
        if extra_data:
            body["extra_data"] = extra_data
        async with self._client() as client:
            response = await client.post("/prompt", json=body)
            if response.is_error:
                detail = _comfyui_error_detail(response)
                raise ValueError(
                    f"ComfyUI rejected the workflow ({response.status_code}): {detail}"
                )
        payload = response.json()
        if not isinstance(payload, dict) or not str(payload.get("prompt_id") or "").strip():
            raise ValueError("ComfyUI prompt response is missing prompt_id")
        node_errors = payload.get("node_errors") or {}
        if not isinstance(node_errors, dict):
            raise ValueError("ComfyUI prompt response node_errors must be an object")
        return ComfyUIPromptSubmission(
            prompt_id=str(payload["prompt_id"]),
            number=payload.get("number"),
            node_errors=node_errors,
        )

    async def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        if not prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                async with self._client() as client:
                    response = await client.get(f"/history/{prompt_id}")
                    response.raise_for_status()
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        assert response is not None
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("ComfyUI history response must be an object")
        record = payload.get(prompt_id)
        if record is None and ("outputs" in payload or "status" in payload):
            record = payload
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError("ComfyUI history record must be an object")
        return record

    async def get_output_image(
        self,
        filename: str,
        *,
        subfolder: str = "",
        storage_type: str = "output",
    ) -> bytes:
        if not filename.strip():
            raise ValueError("output filename must not be empty")
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                async with self._client() as client:
                    response = await client.get(
                        "/view",
                        params={
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": storage_type,
                        },
                    )
                    response.raise_for_status()
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        assert response is not None
        if not response.content:
            raise ValueError("ComfyUI returned an empty image output")
        return response.content

    async def free_models(self, *, unload_models: bool = True) -> None:
        """Ask ComfyUI to release resident models between encoding and diffusion phases."""
        async with self._client() as client:
            response = await client.post(
                "/free",
                json={"unload_models": unload_models, "free_memory": True},
            )
            response.raise_for_status()

    async def cancel_prompt(self, prompt_id: str) -> ComfyUICancelResult:
        """Cancel one prompt without blindly interrupting an unrelated running job."""
        if not prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        async with self._client() as client:
            queue_response = await client.get("/queue")
            queue_response.raise_for_status()
            queue = queue_response.json()
            if not isinstance(queue, dict):
                raise ValueError("ComfyUI queue response must be an object")

            running_ids = _queue_prompt_ids(queue.get("queue_running"))
            pending_ids = _queue_prompt_ids(queue.get("queue_pending"))
            was_running = prompt_id in running_ids
            was_pending = prompt_id in pending_ids
            deleted = False
            interrupted = False

            if was_pending:
                delete_response = await client.post("/queue", json={"delete": [prompt_id]})
                delete_response.raise_for_status()
                deleted = True
            if was_running:
                interrupt_response = await client.post("/interrupt", json={})
                interrupt_response.raise_for_status()
                interrupted = True

        return ComfyUICancelResult(
            prompt_id=prompt_id,
            was_running=was_running,
            was_pending=was_pending,
            interrupted=interrupted,
            deleted=deleted,
        )


def _comfyui_error_detail(response: httpx.Response) -> str:
    """Return bounded validation details without reflecting the submitted prompt."""
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:2000] if text else response.reason_phrase
    if not isinstance(payload, dict):
        return str(payload)[:2000]
    parts: list[str] = []
    error = payload.get("error")
    if isinstance(error, dict):
        error_type = str(error.get("type") or "").strip()
        message = str(error.get("message") or error.get("details") or "").strip()
        if error_type or message:
            parts.append(": ".join(value for value in (error_type, message) if value))
    elif error:
        parts.append(str(error))
    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict):
        for node_id, node_error in list(node_errors.items())[:20]:
            if not isinstance(node_error, dict):
                parts.append(f"node {node_id}: {node_error}")
                continue
            messages: list[str] = []
            for item in node_error.get("errors", ()):
                if not isinstance(item, dict):
                    continue
                message = str(item.get("message") or item.get("type") or "").strip()
                details = str(item.get("details") or "").strip()
                if message or details:
                    messages.append(": ".join(value for value in (message, details) if value))
            if messages:
                parts.append(f"node {node_id}: {'; '.join(messages)}")
    return " | ".join(parts)[:4000] or "unknown ComfyUI validation error"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_object_info(
    payload: dict[str, Any],
    *,
    requested_node_type: str | None,
) -> dict[str, dict[str, Any]]:
    if requested_node_type and requested_node_type not in payload:
        # Some ComfyUI-compatible servers return the schema directly for
        # /object_info/{node_type}; normalize that variant as well.
        if "input" in payload or "output" in payload:
            payload = {requested_node_type: payload}
        else:
            raise ValueError(f"ComfyUI does not expose node type {requested_node_type!r}")
    nodes: dict[str, dict[str, Any]] = {}
    for node_type, schema in payload.items():
        if not isinstance(node_type, str) or not node_type:
            raise ValueError("ComfyUI object_info contains an invalid node type")
        if not isinstance(schema, dict):
            raise ValueError(f"ComfyUI schema for {node_type!r} must be an object")
        input_block = schema.get("input", {})
        if input_block is not None and not isinstance(input_block, dict):
            raise ValueError(f"ComfyUI schema input block for {node_type!r} must be an object")
        nodes[node_type] = schema
    return dict(sorted(nodes.items()))


def _queue_prompt_ids(entries: Any) -> set[str]:
    prompt_ids: set[str] = set()
    if not isinstance(entries, list):
        return prompt_ids
    for entry in entries:
        if isinstance(entry, (list, tuple)) and len(entry) > 1:
            prompt_ids.add(str(entry[1]))
        elif isinstance(entry, dict):
            prompt_id = entry.get("prompt_id")
            if prompt_id is not None:
                prompt_ids.add(str(prompt_id))
    return prompt_ids


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


def _read_git_origin(git_dir: Path) -> str | None:
    config_file = git_dir / "config"
    if not config_file.is_file():
        return None
    parser = ConfigParser()
    try:
        parser.read(config_file, encoding="utf-8")
        return parser.get('remote "origin"', "url", fallback=None)
    except (ConfigParserError, OSError, UnicodeError):
        return None


def _inspect_h3_turbo_packages(
    custom_nodes_root: Path,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    if not custom_nodes_root.is_dir():
        return None, None, ()
    official_path: str | None = None
    official_commit: str | None = None
    unverified: list[str] = []
    official_repository = _normalize_git_url(OFFICIAL_H3_TURBO_REPOSITORY)
    for path in sorted(custom_nodes_root.iterdir(), key=lambda item: item.name.casefold()):
        init_file = path / "__init__.py"
        if not path.is_dir() or not init_file.is_file():
            continue
        try:
            source = init_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if not all(
            node_type in source for node_type in (TURBO_LORA_NODE_TYPE, TURBO_SAMPLER_NODE_TYPE)
        ):
            continue
        commit = _read_git_commit(path / ".git")
        origin = _normalize_git_url(_read_git_origin(path / ".git"))
        if origin == official_repository and commit == PINNED_H3_TURBO_COMMIT:
            official_path = path.name
            official_commit = commit
        else:
            unverified.append(path.name)
    return official_path, official_commit, tuple(unverified)


def _inspect_motion_context_packages(
    custom_nodes_root: Path,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    if not custom_nodes_root.is_dir():
        return None, None, ()
    verified_candidates: list[tuple[str, str]] = []
    unverified: list[str] = []
    expected_repository = _normalize_git_url(MOTION_CONTEXT_REPOSITORY)
    for path in sorted(custom_nodes_root.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_dir():
            continue
        source = ""
        for source_name in ("__init__.py", "nodes.py", "probe_node.py"):
            source_path = path / source_name
            if not source_path.is_file():
                continue
            try:
                source += source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                source = ""
                break
        if not all(node_type in source for node_type in MOTION_CONTEXT_NODE_TYPES):
            continue
        commit = _read_git_commit(path / ".git")
        origin = _normalize_git_url(_read_git_origin(path / ".git"))
        if origin == expected_repository and commit == PINNED_MOTION_CONTEXT_COMMIT:
            verified_candidates.append((path.name, commit))
        else:
            unverified.append(path.name)
    if len(verified_candidates) == 1:
        verified_path, verified_commit = verified_candidates[0]
        return verified_path, verified_commit, tuple(unverified)
    if verified_candidates:
        unverified.extend(path for path, _commit in verified_candidates)
    return None, None, tuple(sorted(unverified))


def _normalize_git_url(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().replace("\\", "/").casefold()
    return normalized.removesuffix(".git").removesuffix("/")


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
    if not inventory.motion_context_verified:
        warnings.append("Pinned H3 Motion Context provider is not installed")
    if inventory.conflicting_context_nodes:
        warnings.append(
            "Conflicting Motion Context nodes detected: "
            + ", ".join(inventory.conflicting_context_nodes)
        )
    if inventory.unverified_h3_turbo_nodes:
        warnings.append(
            "Unverified MiniMax H3 Turbo node packages detected: "
            + ", ".join(inventory.unverified_h3_turbo_nodes)
        )
    if not inventory.official_h3_turbo_verified:
        warnings.append("Pinned official MiniMax H3 Turbo nodes are not installed")
    if not inventory.h3_model_files:
        warnings.append("No H3-related model files were detected")
    return warnings
