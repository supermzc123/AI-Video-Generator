"""Deterministic context selection; source documents are never rewritten to fit."""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from contextlib import asynccontextmanager
from typing import Any

_prompt_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@asynccontextmanager
async def prompt_generation_lock(scope: tuple[str, ...]):
    """Deduplicate concurrent UI generation within the single control process."""
    locks = _prompt_locks.setdefault(asyncio.get_running_loop(), {})
    lock = locks.setdefault(scope, asyncio.Lock())
    async with lock:
        yield


def content_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def structured_source_context(
    source: dict[str, Any], allowed_paths: tuple[str, ...], operation: str
) -> dict[str, Any]:
    """Keep editable arrays intact so JSON-pointer indexes retain their meaning.

    A whole-array operation needs the whole array. Shot-specific endpoints should
    pass a single item path; unrelated histories and completed prompts are omitted.
    """
    if "" in allowed_paths:
        return source
    roots = {path.split("/")[1] for path in allowed_paths if path.startswith("/")}
    roots.update(("name", "idea", "highestInstruction"))
    if "shots" in roots:
        roots.add("outline")
    if "asset" in operation:
        roots.update(("shots", "assetPlans"))
    if "prompts" in roots:
        roots.update(("shots", "assetPlans"))
    context = {key: value for key, value in source.items() if key in roots}
    if isinstance(context.get("prompts"), dict):
        prompt_paths = {
            path.split("/")[2]
            for path in allowed_paths
            if path.count("/") >= 2 and path.startswith("/prompts/")
        }
        if "/prompts" not in allowed_paths:
            context["prompts"] = {
                key: value for key, value in context["prompts"].items() if key in prompt_paths
            }
    return context
