# Independent repair validation — 2026-09-24

## Scope and evidence

Role C, baseline `56de60c05f721ef00603214b7ac8805eb5fe49c0`. No business implementation
changed. Tests are strict contract assertions, not xfails or source-text checks.
Initial delivery precedes A/B integration; this is **not repair acceptance**.

Owned files:
- `tests/test_repair_fault_invariants.py`: real temporary SQLite and independent
  fixed-seed state oracle; exactly 10,000 modeled operations, plus minimal fault
  sequences using store methods and execution guards.
- `tests/test_repair_api_contract.py`: existing FastAPI routes through ASGITransport,
  isolated SQLite, offline MockTransport, network socket guard, no app lifespan.
  Settings and secret loading are replaced before lazy API import because the
  API module constructs a global app and opens stores at import time.
- `scripts/repair_soak.py`: configurable isolated SQLite/mock-worker fault runner.
- This report. Detailed raw evidence and delivery notes remain uncommitted in `.tmp`.

The 10,000-operation oracle alternates claim, project pause, completion, queued
cancellation, idempotent insertion, and store reopening, with seed 20260924. It
checks persisted states and GPU exclusivity after every operation. Claim owner
and task selection are seeded; denied/irrelevant operations still count as modeled
events. It adds fresh tasks so the exercise does not become entirely terminal.
This is serial deterministic interleaving, **not** a concurrent stress proof.

Minimal probes additionally cover failed/cancelled dependencies, independent
projects, two LLM slots, one GPU slot, durable pause/dispatcher takeover,
cancel/completion/retry ordering, stale attempts, unknown submission identity,
cancel ownership across restart, retry deadlines, orchestration yielding,
paused-batch child insertion/membership, duplicate insertion, and rollback of
invalid expansion. HTTP tests cover idempotence, terminal-state protection,
missing tasks, offline cancellation, metadata and batch cancellation scope.

## Confirmed baseline findings

| ID | Owner | Minimal sequence | Observed violation / required invariant |
| --- | --- | --- | --- |
| F1 | B | Claim; mark cancelled or succeeded; delayed schedule_task_retry; promote due retries | Terminal task becomes queued. Late retries must reject/no-op without reviving terminal work. |
| F2 | B | Claim LLM; reopen store; recover_local_attempts; old guarded success before new claim | Recovering becomes succeeded. Recovery must invalidate the old writer immediately. |
| F3 | B | Claim as owner; new-owner acquires dispatcher at +121 seconds; old guarded success | Old owner publishes success. Fence by current owner as well as attempt. |
| F4 | B + A observation integration | Restore two running GPU tasks with distinct external IDs; recover both; claim observations | Both claims return None. Existing external-job observation must not compete as new GPU execution. |
| F5 | B; root deadline semantics | Claim LLM at 00:00:00 UTC; retry; reopen; promote/claim at 00:00:02 | Absolute deadline changes from 00:05:00 to 00:05:02. Preserve durable task deadline rather than restarting the budget. |
| F6 | root + A scope integration | Batch contains P; POST batch/project Q/cancel where Q is not a member | HTTP 200 cancels Q's task. Validate scope before any mutation. |
| F7 | root + A scope integration | Batch selects only P/member; POST batch cancel or batch/P/cancel | P/outside is also cancelled. Use persisted member task IDs, not all project tasks. |

Exact pytest repro selectors (prefix with the relevant owned test file):

```text
test_delayed_retry_cannot_revive_terminal_task[cancelled]
test_delayed_retry_cannot_revive_terminal_task[succeeded]
test_previous_owner_cannot_publish_after_recovery_boundary[restart]
test_previous_owner_cannot_publish_after_recovery_boundary[owner_takeover]
test_multiple_recovered_external_jobs_can_each_be_observed
test_retry_deadline_and_attempt_survive_restart
test_batch_cancel_rejects_nonmember_project_without_side_effects
test_batch_cancel_does_not_cancel_unselected_project_tasks
```

Two additional observations are not included as independent asserted regressions:
batch-wide cancellation without a project run-state silently leaves members ready;
Windows mock-soak cleanup initially found open SQLite read handles (WinError 32).
The runner collects unreachable connections before deleting **only its generated
temporary database directory**; B should audit read-connection lifetime separately.

## Commands and results

Run from this worktree. Use a new basetemp/output directory each time so existing
evidence is preserved. No shared/production database is a valid input.

```powershell
$env:PYTHONPATH = 'H:\video-repair-worktrees\20260924-verification\src'
$python = 'H:\视频生成工具\.venv\Scripts\python.exe'
& $python -m pytest tests/test_repair_fault_invariants.py tests/test_repair_api_contract.py --basetemp .tmp/pytest-full --tb=short --durations=5
& $python -m pytest tests/test_repair_fault_invariants.py tests/test_repair_api_contract.py --basetemp .tmp/pytest-final --tb=short --durations=3
& $python -m ruff check src/ai_video_generator tests
& $python -m ruff check tests/test_repair_fault_invariants.py tests/test_repair_api_contract.py scripts/repair_soak.py
& $python scripts/repair_soak.py --duration-hours 0.003 --max-cycles 40 --interval-seconds 0.001 --network-fault-rate 0.4 --submit-fault-rate 0.4 --restart-fault-rate 0.6 --cancel-fault-rate 0.6 --output-dir .tmp/soak-fault-smoke-v2
& $python scripts/repair_soak.py --duration-hours 0.001 --max-cycles 5 --network-fault-rate 0 --submit-fault-rate 0 --restart-fault-rate 0 --cancel-fault-rate 0 --output-dir .tmp/soak-clean-smoke-v2
```

- Initial complete suite: **21 passed, 9 failed**, 64.84 seconds; 52 existing
  FastAPI lifespan-deprecation warnings. Seeded 10,000-operation test passed in
  60.58 seconds. Failures correspond to F1–F7 above, not harness failures.
- Final suite with two additional parent-resumption/atomic-rollback probes:
  **23 passed, 9 failed**, 62.83 seconds; 10,000-operation probe 58.97 seconds.
  Evidence: `.tmp/final-tests.txt`. Same nine strict contract failures remain.
- Owned-file Ruff check passes after formatting/suppress-style cleanup.
- Fault smoke: **40 cycles, 242 modeled events, 3.390 seconds**, 0 runner errors;
  sampled faults: network 14, submit 20, restart 28, cancel 21. F2 reproduced 11
  times; F1 reproduced 13 times; exit 1 correctly signals invariant failures.
- Fault-free smoke: **5 cycles, 20 modeled events, 0.641 seconds**, 0 violations,
  0 errors, exit 0. Both runs stopped at max-cycles, not requested duration.

## Soak semantics

`--duration-hours 8` uses actual monotonic wall-clock time, not accelerated time.
For a later authorized run (not executed during this delivery):

```powershell
& $python scripts/repair_soak.py --duration-hours 8 --seed 20260924 --output-dir .tmp/soak-8h-new
```

The output directory must not exist. There is deliberately no database-path,
worker URL, credential, GPU, or paid-service option. A new temporary SQLite DB
is used each cycle, with a fake timestamp and an in-memory HTTP worker. Fault
rates and interval are configurable. Network faults fail before submission;
submit faults accept a job and lose the response. Restart reopens the same
cycle DB, cancel faults inject cancellation and a delayed retry callback.
Selected fault flags may mask another selected fault (e.g. network failure
prevents submit-response loss); the JSONL trace records what actually happened.

`events.jsonl` is flushed per cycle and records sampled faults, operation trace,
store events, worker request counts and violations. `summary.json` distinguishes
requested duration, actual elapsed time, stopping reason, modeled operations,
errors, violations and hardware acceptance. Exit 1 means violations, errors or
interruption; exit 0 can be a bounded smoke and is not an eight-hour certificate.
This tests real store behavior with a **test driver**, not the production
scheduler, ComfyUI adapter, process crash recovery, real network or GPU.

## Integration note: hooks required from owners

- **B**: transactional current-owner/attempt fencing on all late mutations,
  including retry, result publication and dynamic child registration. Recovery
  must fence immediately, preserve submission identity/deadline, and reserve
  unknown external work until explicit reconciliation/stop confirmation.
- **A + B**: observation-only recovery path for multiple existing external IDs;
  it must not count as a new local GPU submission. Preserve parent yield and
  atomic paused-batch child membership when integrating orchestration.
- **root**: route batch actions through membership-scoped service methods; reject
  unknown project/task members before setting pause or cancelling anything.
  Bind existing actual routes tested here; no invented route names are asserted.
- **root new interfaces**: scheduler health, task events, reconcile, explicit
  confirm-retry, and bulk scoped commands are required by the repair contract.
  Baseline has no dedicated routes for these surfaces. Freeze verb/path/schema
  before C adds HTTP assertions. Reconcile must not submit work. Confirm-retry
  must require explicit user authority. Bulk commands need scope, task IDs,
  idempotency key, transactional membership validation and itemized outcomes.
- Freeze deadline scope (task vs generation-attempt) explicitly if changing the
  contract. F5 currently enforces the documented persistent absolute task budget.

## Acceptance still open

A/B delivered diffs have **not** been independently reviewed after integration;
this initial worktree is still at the baseline. Primary must integrate owned
files and rerun these repros, then return the integrated commit/diffs for C review.
Do not consider failing tests an accepted implementation or silently xfail them.

No eight-hour run, full root regression suite, desktop build, real hardware,
ComfyUI/GPU execution, paid LLM calls, UX/media review, packaging or publishing
was performed. Full-suite collection in the baseline can initialize the global
API app; root must supply isolated settings/secret mocks before running it.
Long-lived database growth, concurrent processes, OS termination, remote workers,
transport adapter reconciliation and new API contracts remain separate gates.
