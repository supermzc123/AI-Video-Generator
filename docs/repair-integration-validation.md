# Independent integration validation — September 24, 2026

Role C initially tested snapshot `7cad79657ce26a7da2a1b5bfd848c5d6b96876a8`, then
independently retested integrated core `687dc73ece863f239c40556d079ca492c0de886d`.
The existing API/store probes now pass; new production-lifespan tests expose two
additional recovery/scheduling defects. This is not completion acceptance.
No application implementation was changed by C.

## Integrated core rerun

On `687dc73`, all **41 API/store/invariant tests passed** in 180.70 seconds,
including the fixed-seed 10,000-operation sequence (155.92 seconds). No xfail or
relaxed invariants were used. The five defects in the initial table below are
resolved by the integrated A2/B2/API changes under these probes. Evidence:
`.tmp/core-final-tests.txt`.

A separate 200-cycle mock fault smoke completed **1,235 modeled events in
155.859 seconds**, with zero invariant violations and zero runner errors.
Selected faults: network 70, response loss 78, restart 108, cancellation 118.
It stopped at max-cycles, before the requested 0.25-hour duration, and is not an
eight-hour certificate. Evidence: `.tmp/soak-core-200/summary.json` and
`events.jsonl`.

```powershell
& $python -u -m pytest tests/test_repair_fault_invariants.py tests/test_repair_api_contract.py tests/test_repair_integration_contract.py --basetemp .tmp/pytest-c-core-final --tb=short --durations=5
& $python -u scripts/repair_soak.py --duration-hours 0.25 --max-cycles 200 --interval-seconds 0.001 --network-fault-rate 0.4 --submit-fault-rate 0.4 --restart-fault-rate 0.6 --cancel-fault-rate 0.6 --output-dir .tmp/soak-core-200
```

## New production dispatcher probes

`tests/test_repair_actual_dispatch.py` starts and stops the actual application
lifespan. It exercises the real asynchronous dispatch loop, ownership wrapper,
executor, SQLite transitions and output-checkpoint registration. Only external
ComfyUI transport, media probing and the explicit interruption injection are
replaced. There are no real GPU/FFmpeg/paid LLM calls.

| Probe | Result on `687dc73` | Minimal defect |
| --- | --- | --- |
| Paused batch member before independent project in GPU ordering | Failed | Independent task stays queued with no claim event |
| Queued task with failed dependency before independent project | Failed | Same starvation; rejected claim reserves in-memory slot for the entire scan |
| First actual executor fails manifest validation; next project proceeds | Passed | Deterministic failure releases slot normally |
| Restart after video checkpoint but before success publication | Failed | Recollection uses same checkpoint ID with different timestamp; immutable conflict produces needs_attention |

Starvation occurs because the dispatch loop inserts a GPU task into the in-memory
active set before its asynchronously scheduled claim executes. A nonclaimable
first candidate therefore prevents subsequent independent tasks every scan.
The required fix is to claim before reserving the in-memory slot, preserving the
database transaction as the eligibility authority.

The output recovery probe writes a video checkpoint, injects `CancelledError`
at the subsequent success transition, stops the app and starts a second app.
ComfyUI submission count remains one. Recovery nevertheless fails with
`checkpoint ID is immutable`, because the same logical checkpoint is recreated
with a later `created_at`. Reusing a validated existing checkpoint must complete
the original task without a new expensive submission. Other fixed checkpoint ID
paths (conditioning, Motion Context, subtitles, export, candidate registration)
need the same idempotence rule.

The three initial lifespan probes returned **1 passed, 2 failed** in 15.55
seconds; the added output-recovery probe failed in 1.54 seconds. Evidence:
`.tmp/actual-dispatch.txt`, `.tmp/output-recovery.txt`. Strict tests were committed
as `98e63b0`; the root owns business fixes and their subsequent rerun.

## Test harness corrections

- Store and execution guard implicit clock reads now share the explicit fake
  clock. A lease claimed at midnight must not be rejected only because the test
  executes later that day.
- Submission identity after cancellation/recovery is inspected through the
  read-only `inspect_submission`; `ensure_submission_intent` is a write operation
  requiring a running attempt.
- Old callback isolation tests use a valid orchestration yield, requeue and new
  claim. Calling lease recovery on a current unexpired attempt must not forcibly
  replace its owner.
- The mock soak freezes implicit clock reads and transfers dispatcher ownership
  when simulating a restart. It no longer mistakes a still-current live attempt
  for a superseded writer.
- Cancellation with an unknown submission must retain `cancelling`, preventing
  automatic recovery from reviving a user's cancelled intent.

## New HTTP behavior tests

`tests/test_repair_integration_contract.py` uses ASGITransport, real isolated
SQLite databases, mock HTTP worker responses and a real network socket guard.
There is no application lifespan/scheduler execution, GPU call or paid LLM call.

| Probe | Initial `7cad796` result |
| --- | --- |
| Lost submission response, cancel, token reconciliation, no resubmission | Failed: cancellation remains needs_attention |
| Old reconciliation response arrives after a replacement claim | Failed: replacement RUNNING becomes RECOVERING |
| Pause member while same-project independent task remains claimable | Failed: `put_batch_run` rejects paused membership update; route also globally pauses project |
| Reject cross-batch selected task before any mutation | Passed |
| Mixed success/failure command receipt survives restart without replay | Passed |
| Explicit duplicate-risk confirmation cannot reset exhausted attempt budget | Failed: confirmation returns success and resets budget |
| Ordinary public retry preserves attempt count and absolute deadline | Failed: deadline advances by retry delay |
| Failed/cancelled/stale batch members are terminal but not successful | Passed |
| Health/events reads preserve state; events bounded and newest first | Passed |

The late reconciliation sequence deliberately changes persisted attempt identity
inside the mocked old history response. It uses public store reconciliation and
retry APIs, not direct SQL. The task must stay owned by the replacement even when
the old request subsequently returns a completed history record.

## Evidence

Run from `H:\video-repair-worktrees\20260924-integration-verification`:

```powershell
$env:PYTHONPATH = 'H:\video-repair-worktrees\20260924-integration-verification\src'
$env:AIVIDEO_DATA_ROOT = 'H:\video-repair-worktrees\20260924-integration-verification\.tmp\import-data'
$python = 'H:\视频生成工具\.venv\Scripts\python.exe'
& $python -m pytest tests/test_repair_integration_contract.py tests/test_repair_fault_invariants.py tests/test_repair_api_contract.py -k 'not seeded' --basetemp .tmp/pytest-c-contract3 --tb=short
& $python -m ruff check tests/test_repair_integration_contract.py tests/test_repair_fault_invariants.py tests/test_repair_api_contract.py scripts/repair_soak.py
& $python scripts/repair_soak.py --duration-hours 0.003 --max-cycles 40 --interval-seconds 0.001 --network-fault-rate 0.4 --submit-fault-rate 0.4 --restart-fault-rate 0.6 --cancel-fault-rate 0.6 --output-dir .tmp/soak-contract-smoke
```

- Public integration + corrected prior probes: **34 passed, 6 failed, 1
  deselected**, 6.74 seconds. The six failures represent five defects above
  (unknown-submission cancellation is covered twice). Raw evidence:
  `.tmp/contract3.txt`.
- Owned-file Ruff: passed.
- Mock fault soak smoke: **40 cycles, 252 modeled events, 3.938 seconds**, zero
  invariant violations and zero runner errors. Fault selections: network 14,
  submit response loss 20, restart 28, cancel 21. Stopped at max-cycles, not the
  requested duration. Evidence: `.tmp/soak-contract-smoke/summary.json` and
  `events.jsonl`.
- The previously passing 10,000-operation sequence was deliberately deselected
  pending B2 integration, to avoid repeatedly running an unchanged expensive
  probe. Root must run it once after the integrated final changes.

## Acceptance still open

The initial six failing assertions now pass on `687dc73`. The new production
lifespan probes remain strict assertions awaiting the root's scheduling and
checkpoint fixes. No xfail or relaxed acceptance was added. Passing only the
store/API suite is insufficient evidence of end-to-end recovery.

The smoke uses the store plus a modeled worker driver, not the production
scheduler. Passing it does not establish an eight-hour run or transport/hardware
recovery correctness. No eight-hour wall-clock soak, real GPU generation,
remote hardware, four-hour GPU observation, UI browser workflow, human visual
or listening review, full repository regression, packaging or publishing was
performed by C in this delivery.
