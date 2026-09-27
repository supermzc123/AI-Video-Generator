"""Isolated SQLite + mock HTTP fault soak; never starts the application or GPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import tempfile
import time
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import httpx

from ai_video_generator.domain import TaskKind, TaskSpec, TaskState
from ai_video_generator.persistence import (
    InvalidTaskTransitionError,
    SQLiteTaskStore,
    StoreConflictError,
)
from ai_video_generator.persistence.execution_runtime import execution_guard

START = datetime(2026, 9, 24, tzinfo=UTC)


class MockWorker:
    def __init__(self, faults):
        self.faults = faults
        self.jobs = {}
        self.calls = Counter()

    def handle(self, request):
        path = request.url.path
        self.calls[path] += 1
        if path == "/submit":
            if self.faults["network"]:
                raise httpx.ConnectError("injected pre-submit disconnect", request=request)
            submission = json.loads(request.content)["submission"]
            self.jobs[submission] = "running"
            if self.faults["submit"]:
                raise httpx.ReadTimeout("accepted but response lost", request=request)
            return httpx.Response(200, json={"prompt_id": submission})
        submission = request.url.params["submission"]
        if path == "/cancel":
            if self.faults["network"]:
                raise httpx.ConnectError("injected cancel disconnect", request=request)
            self.jobs[submission] = "cancelled"
        return httpx.Response(200, json={"state": self.jobs.get(submission, "unknown")})


def make_task(task_id):
    digest = sha256(task_id.encode()).hexdigest()
    return TaskSpec(
        task_id=task_id,
        project_id="isolated-soak",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.QUEUED,
        idempotency_key=digest,
        input_fingerprint=digest,
    )


def run_cycle(directory, cycle, faults):
    now = START + timedelta(seconds=cycle * 300)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is not None else now.replace(tzinfo=None)

    with (
        patch("ai_video_generator.persistence.execution_runtime.datetime", FrozenDateTime),
        patch("ai_video_generator.persistence.task_store.datetime", FrozenDateTime),
    ):
        return _run_cycle(directory, cycle, faults)


def _run_cycle(directory, cycle, faults):
    store = SQLiteTaskStore(directory / "test-only.db")
    worker = MockWorker(faults)
    trace = []
    violations = []

    def check(condition, name):
        if not condition:
            violations.append(name)

    def event(name):
        trace.append(name)

    now = START + timedelta(seconds=cycle * 300)
    owner = "owner"
    store.add_task(make_task("work"))
    store.add_task(make_task("next"))
    assert store.acquire_dispatcher(owner, now=now)
    claimed = store.claim_local_task("work", owner, now=now)
    assert claimed is not None
    event("claim")
    guard = execution_guard.set((claimed.task_id, claimed.attempt_id))
    try:
        submission, fresh = store.ensure_submission_intent("work")
        assert fresh
        event("submission_intent")
        with httpx.Client(
            transport=httpx.MockTransport(worker.handle), base_url="http://mock-worker.invalid"
        ) as client:
            try:
                response = client.post("/submit", json={"submission": submission})
                store.record_comfyui_prompt("work", response.json()["prompt_id"])
                event("submit_acknowledged")
            except httpx.HTTPError:
                event("submit_unknown")
                store.transition_task("work", TaskState.NEEDS_ATTENTION, now=now)
    finally:
        execution_guard.reset(guard)
    if faults["restart"]:
        store = SQLiteTaskStore(store.database_path)
        store.release_dispatcher(owner)
        owner = "restarted-owner"
        assert store.acquire_dispatcher(owner, now=now)
        store.recover_local_attempts(now=now)
        event("restart_recovery")
        check(
            store.inspect_submission("work")["submission_token"] == submission,
            "restart_lost_submission_identity",
        )
        before = store.get_task("work").state
        guard = execution_guard.set((claimed.task_id, claimed.attempt_id))
        try:
            with suppress(ValueError, InvalidTaskTransitionError, StoreConflictError):
                store.transition_task("work", TaskState.SUCCEEDED, now=now)
        finally:
            execution_guard.reset(guard)
        event("old_attempt_completion")
        check(store.get_task("work").state == before, "stale_completion_after_restart")
    current = store.get_task("work")
    if faults["cancel"] and current.state in {
        TaskState.RUNNING,
        TaskState.RECOVERING,
        TaskState.NEEDS_ATTENTION,
    }:
        store.transition_task("work", TaskState.CANCELLING, now=now)
        event("cancel_requested")
        check(store.claim_local_task("next", owner, now=now) is None, "cancelling_released_gpu")
        with httpx.Client(
            transport=httpx.MockTransport(worker.handle), base_url="http://mock-worker.invalid"
        ) as client:
            try:
                client.post("/cancel", params={"submission": submission}).raise_for_status()
                store.transition_task("work", TaskState.CANCELLED, now=now)
                event("stop_confirmed")
            except httpx.HTTPError:
                event("stop_unconfirmed")
    current = store.get_task("work")
    if current.state == TaskState.RUNNING:
        store.transition_task("work", TaskState.SUCCEEDED, now=now)
        event("completion")
    current = store.get_task("work")
    if current.state in {TaskState.SUCCEEDED, TaskState.CANCELLED} and faults["cancel"]:
        with suppress(ValueError, InvalidTaskTransitionError, StoreConflictError):
            store.schedule_task_retry("work", "delayed network callback", now=now)
        store.promote_due_retries(now=now + timedelta(seconds=10))
        event("delayed_retry")
        check(store.get_task("work").state == current.state, "terminal_revived_by_retry")
    if store.get_task("work").state in {TaskState.NEEDS_ATTENTION, TaskState.CANCELLING}:
        check(
            store.claim_local_task("next", owner, now=now) is None,
            "unknown_external_work_released_gpu",
        )
        event("check_reserved_slot")
    check(worker.calls["/submit"] == 1, "duplicate_external_submission")
    return {
        "cycle": cycle,
        "faults": faults,
        "trace": trace,
        "violations": violations,
        "mock_requests": dict(worker.calls),
        "operations": len(trace),
        "state": store.get_task("work").state.value,
        "execution_events": store.list_execution_events("work"),
    }


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-hours", type=positive, default=0.01)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--interval-seconds", type=positive, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    for fault in ("network", "submit", "restart", "cancel"):
        parser.add_argument(f"--{fault}-fault-rate", type=probability, default=0.25)
    args = parser.parse_args(argv)
    if args.max_cycles is not None and args.max_cycles <= 0:
        parser.error("--max-cycles must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    randomizer = random.Random(args.seed)
    started = time.monotonic()
    deadline = started + args.duration_hours * 3600
    violations = Counter()
    injections = Counter()
    cycles = operations = errors = 0
    stop_reason = "duration_elapsed"
    try:
        with (args.output_dir / "events.jsonl").open("w", encoding="utf-8") as events:
            while time.monotonic() < deadline:
                if args.max_cycles is not None and cycles >= args.max_cycles:
                    stop_reason = "max_cycles"
                    break
                faults = {
                    name: randomizer.random() < getattr(args, f"{name}_fault_rate")
                    for name in ("network", "submit", "restart", "cancel")
                }
                injections.update(name for name, enabled in faults.items() if enabled)
                with tempfile.TemporaryDirectory(prefix="sqlite-", dir=args.output_dir) as temp:
                    try:
                        result = run_cycle(Path(temp), cycles, faults)
                    except Exception as exc:
                        errors += 1
                        result = {
                            "cycle": cycles,
                            "faults": faults,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    finally:
                        gc.collect()
                events.write(json.dumps(result, ensure_ascii=True) + "\n")
                events.flush()
                violations.update(result.get("violations", []))
                operations += result.get("operations", 0)
                cycles += 1
                time.sleep(min(args.interval_seconds, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    elapsed = time.monotonic() - started
    summary = {
        "evidence": "real isolated SQLite; modeled worker via httpx.MockTransport",
        "seed": args.seed,
        "requested_duration_hours": args.duration_hours,
        "elapsed_seconds": elapsed,
        "duration_completed": elapsed >= args.duration_hours * 3600,
        "stop_reason": stop_reason,
        "cycles": cycles,
        "operations": operations,
        "fault_injections": dict(injections),
        "violations": dict(violations),
        "errors": errors,
        "hardware_acceptance": False,
        "eight_hour_run": elapsed >= 8 * 3600,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if violations or errors or stop_reason == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
