"""Durable, at-most-once control-command receipts for scoped user operations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from ai_video_generator.persistence import SQLiteTaskStore, StoreConflictError


def initialize_control_commands(store: SQLiteTaskStore) -> None:
    with store._transaction(immediate=True) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS control_commands ("
            "idempotency_key TEXT PRIMARY KEY, request_hash TEXT NOT NULL, "
            "owner_id TEXT NOT NULL, status TEXT NOT NULL, response_json TEXT NOT NULL, "
            "updated_at REAL NOT NULL)"
        )


def begin_control_command(
    store: SQLiteTaskStore, payload: dict, *, owner_id: str
) -> tuple[bool, dict]:
    """Record intent before any side effect; duplicates never replay a partial command."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    key = payload["idempotency_key"]
    with store._transaction(immediate=True) as connection:
        row = connection.execute(
            "SELECT * FROM control_commands WHERE idempotency_key=?", (key,)
        ).fetchone()
        if row:
            if row["request_hash"] != fingerprint:
                raise StoreConflictError("idempotency key was already used for another command")
            result = json.loads(row["response_json"])
            if row["status"] != "completed":
                result["status"] = "needs_attention" if row["owner_id"] != owner_id else "pending"
                result["message"] = "命令尚无完整回执；请先对账。重复请求不会再次执行已提交的操作。"
            return False, result
        result = {
            "idempotency_key": key,
            "scope": payload["scope"],
            "scope_id": payload["scope_id"],
            "action": payload["action"],
            "status": "pending",
            "results": [],
        }
        connection.execute(
            "INSERT INTO control_commands VALUES(?,?,?,?,?,?)",
            (
                key,
                fingerprint,
                owner_id,
                "pending",
                json.dumps(result),
                datetime.now(UTC).timestamp(),
            ),
        )
        return True, result


def save_control_command(
    store: SQLiteTaskStore, result: dict, *, owner_id: str, completed: bool = False
) -> dict:
    result = {**result, "status": "completed" if completed else "pending"}
    with store._transaction(immediate=True) as connection:
        updated = connection.execute(
            "UPDATE control_commands SET response_json=?,status=?,updated_at=? "
            "WHERE idempotency_key=? AND owner_id=?",
            (
                json.dumps(result, ensure_ascii=False),
                result["status"],
                datetime.now(UTC).timestamp(),
                result["idempotency_key"],
                owner_id,
            ),
        )
        if updated.rowcount != 1:
            raise StoreConflictError("control command ownership changed")
    return result
