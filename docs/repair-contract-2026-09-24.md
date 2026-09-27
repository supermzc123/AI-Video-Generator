# Recovery and batch repair contract — 2026-09-24

This is the implementation contract for the approved repair plan, not a claim of completion.

## Ownership and integration

- Primary agent owns `api.py`, `api_models.py`, `config.py`, shared domain types and final integration.
- Agent A owns batch/scheduling services and their new module tests. Extract orchestration from the large API closure into a service; supply the primary agent with the exact integration call. Do not edit the API or persistence schema.
- Agent B owns persistence/runtime, ComfyUI/remote adapters and recovery services, with focused tests. Keep existing public signatures compatible or report additions before integration. Do not edit shared API/domain files.
- Agent C owns independent fault/invariant tests and validation reports; no business-code edits.
- All agents start from the same saved snapshot in separate worktrees. Maximum three children active. Integrate only owned changes; preserve original user modifications.
- After core integration and regression pass, reassign available roles to D (LLM contracts) and E (frontend). C continues independent validation.

## State and execution invariants

- Keep existing states and add `recovering`, `retry_wait`, `cancelling`, `needs_attention`.
- `TaskSpec` is a frozen Pydantic model. Existing added metadata names are authoritative: `attempt_id`, `created_at`, `updated_at`, `last_activity_at`, `next_retry_at`, `deadline_at`, `current_phase`, `blocked_reason`, `available_actions`.
- SQLite is the source of truth. Claim and pause checks share a transaction; claims require dependencies, approval/queued eligibility and current dispatcher ownership.
- One local GPU execution slot; two LLM slots by default. Re-observing existing external work is distinct from submitting new work. Multiple recovered external IDs must not deadlock each other's observation.
- Expired leases never prove external work stopped. Fence late writes by attempt and owner. Unknown submissions reserve safety and become `needs_attention`; do not blindly replay.
- Attempts, absolute deadlines, retries, external submission identity and wait explanations survive restart. Resume an orchestration step without burning a new generation attempt.
- Cancelling retains resource ownership until stop is confirmed. User-cancelled tasks cannot be revived by recovery/retry of ancestors.
- Failed descendants remain blocked and do not obstruct runnable work in other branches/projects.
- Parent orchestrators persist a checkpoint/dependency and yield; they never occupy a slot polling children.
- Register dynamic image/DAG children and batch membership atomically. A paused batch can still receive results/membership from already-running orchestration.
- Preserve review policy, confirmed prompts, asset versions and completed work. Full image workflows remain one GPU operation.

## Integration interfaces

- Existing store methods: `acquire_dispatcher`, `release_dispatcher`, `claim_local_task`, `renew_local_attempt`, `recover_local_attempts`, `ensure_submission_intent`, `defer_orchestration`, `defer_orchestration_if_child_pending`, `schedule_task_retry`, `promote_due_retries`, `list_execution_events`.
- Batch service extraction should accept `TaskSpec`, store, an internal HTTP client, the existing project-operation callback and batch settings; return `True` only after DAG registration, otherwise persist a yield and return `False`.
- Root adds scheduler health, task events, reconcile and explicit confirm-retry APIs. Reconcile never implies permission to resubmit. Bulk commands require a scope, IDs, idempotency key, membership validation and itemized outcomes.
- Frontend additions consume backend facts/actions rather than independently inventing terminal states. Project/batch controls stay scoped.

## Validation and authorization

- Use isolated temporary SQLite databases, fake clocks, recorded/model responses and mock ComfyUI transports. Never operate production tasks for fault injection.
- Run `python -m ruff check src/ai_video_generator tests`, `python -m pytest`, and `npm --prefix desktop run build` from the relevant worktree. Set PYTHONPATH to that worktree's `src` when using the shared virtualenv.
- C adds deterministic event-sequence invariants covering at least 10,000 operations and a configurable soak runner. Do not represent accelerated simulation as an 8-hour wall-clock soak.
- Real GPU, paid LLM, long-running resource occupation, remote hardware and subjective media/UX validation remain explicitly unaccepted unless actually executed with the required authorization.
- Do not reset databases, remove media/models, upgrade ComfyUI nodes or publish/package.
- Deliver changed files/commit, minimal reproductions, exact commands/results and remaining gaps. Root reruns integration tests instead of accepting reports alone.
