"""Exercise durable orchestration against SQLite, with only external calls mocked."""

import copy
import hashlib
import json
from datetime import UTC, datetime

import httpx
import pytest

from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ProjectRunState,
    ProjectWorkspaceRevision,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.persistence.execution_runtime import execution_guard
from ai_video_generator.services.batch_orchestration import (
    OrchestrationNeedsAttention,
    run_batch_project_orchestration,
)
from ai_video_generator.services.batch_runs import (
    batch_dispatch_frontier,
    reconcile_batch_runs,
)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _task(task_id, kind, *, state=TaskState.READY, project_id="p", depends_on=()):
    return TaskSpec(
        task_id=task_id,
        project_id=project_id,
        kind=kind,
        state=state,
        depends_on=depends_on,
        input_fingerprint=_hash(task_id),
        idempotency_key=_hash(task_id),
    )


def _image_prompt(text="a character portrait"):
    return {
        "id": "image-prompt",
        "assetPlanId": "character",
        "revision": 1,
        "locked": True,
        "prompt": text,
        "negativePrompt": "blur",
        "workflowTemplateId": "image-workflow",
        "referenceAssetIds": [],
        "harnessRevision": 1,
    }


def _h3_prompt():
    return {
        "id": "h3-prompt",
        "segmentId": "shot.C01",
        "shotId": "shot",
        "segmentIndex": 0,
        "continuationOf": None,
        "durationSeconds": 4.0,
        "inputMode": "t2va",
        "prompt": "a slow camera pan",
        "assetIds": [],
        "seed": 1,
        "locked": True,
        "revision": 1,
    }


def _workspace(*, prepared=False, needs_image=False):
    return {
        "idea": {"concept": "arrival"},
        "outline": [{"id": "beat", "title": "Arrival"}],
        "shots": [{"id": "shot", "durationSeconds": 4, "summary": "arrival"}] if prepared else [],
        "referenceAssetMode": "planned" if needs_image or not prepared else "none",
        "assetPlans": [{"id": "character", "kind": "character", "width": 1024, "height": 1600}]
        if needs_image
        else [],
        "prompts": {"imagePrompts": [], "h3Prompts": [_h3_prompt()] if prepared else []},
        "assets": [],
    }


class Scenario:
    def __init__(self, path, payload, *, other_tasks=()):
        self.path = path
        self.store = SQLiteTaskStore(path)
        self.calls = []
        self.operations = []
        self.pause_during_image_registration = False
        self.members_at_insertion = []
        self.store.put_project_run_state(
            ProjectRunState(
                project_id="p",
                outline_approved=True,
                updated_at=datetime.now(UTC),
            )
        )
        self.save(payload)
        self.store.add_task(_task("parent", TaskKind.LLM_PLANNING))
        for other in other_tasks:
            self.store.add_task(other)
        self.store.put_batch_run(
            BatchRun(
                batch_id="batch",
                name="Batch",
                state=BatchState.RUNNING,
                items=(
                    BatchRunItem(project_id="p", task_ids=("parent",)),
                    *(
                        BatchRunItem(project_id=other.project_id, task_ids=(other.task_id,))
                        for other in other_tasks
                    ),
                ),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    def save(self, payload):
        try:
            revision = self.store.get_latest_project_workspace("p").revision + 1
        except KeyError:
            revision = 1
        self.store.put_project_workspace_revision(
            ProjectWorkspaceRevision(
                project_id="p",
                revision=revision,
                payload=copy.deepcopy(payload),
                payload_sha256=_hash(payload),
                created_at=datetime.now(UTC),
            )
        )

    def payload(self):
        return copy.deepcopy(self.store.get_latest_project_workspace("p").payload)

    def restart(self):
        self.store = SQLiteTaskStore(self.path)

    async def operate(self, project_id, request):
        assert project_id == "p"
        self.operations.append(request.operation)
        payload = self.payload()
        if request.operation == "initialize_storyboard":
            payload["shots"] = [{"id": "shot", "durationSeconds": 4, "summary": "arrival"}]
        elif request.operation == "initialize_assets":
            payload["referenceAssetMode"] = "none"
        else:
            raise AssertionError(request.operation)
        self.save(payload)

    async def http(self, request):
        self.calls.append(request.url.path)
        payload = self.payload()
        if request.url.path.endswith("/prompts/images/character/generate"):
            payload["prompts"]["imagePrompts"] = [_image_prompt()]
            self.save(payload)
            return httpx.Response(
                200, json=self.store.get_latest_project_workspace("p").model_dump(mode="json")
            )
        if request.url.path.endswith("/image-prompts/character/run"):
            if self.pause_during_image_registration:
                batch = self.store.list_batch_runs()[0]
                self.store.put_batch_run(batch.model_copy(update={"state": BatchState.PAUSED}))
            entry = payload["prompts"]["imagePrompts"][0]
            task_id = (
                "image:p:"
                + hashlib.sha256(b"character").hexdigest()[:12]
                + ":"
                + _hash([entry, payload["assetPlans"][0]])[:16]
            )
            task = self.store.add_task(
                _task(task_id, TaskKind.IMAGE_GENERATION, state=TaskState.QUEUED)
            )
            # Inspect before the HTTP response and before explicit _register().
            self.members_at_insertion.append(self.store.list_batch_runs()[0].items[0].task_ids)
            assert task.task_id in self.members_at_insertion[-1]
            return httpx.Response(202, json=task.model_dump(mode="json"))
        if request.url.path.endswith("/prompts/h3/shot.C01/regenerate"):
            payload["prompts"]["h3Prompts"] = [_h3_prompt()]
            self.save(payload)
            return httpx.Response(
                200, json=self.store.get_latest_project_workspace("p").model_dump(mode="json")
            )
        if request.url.path.endswith("/tasks/compile"):
            tasks = tuple(
                self.store.add_task(task)
                for task in (
                    _task("encode", TaskKind.CONDITIONING_ENCODING),
                    _task(
                        "diffuse",
                        TaskKind.H3_GENERATION,
                        state=TaskState.BLOCKED,
                        depends_on=("encode",),
                    ),
                    _task(
                        "export", TaskKind.EXPORT, state=TaskState.BLOCKED, depends_on=("diffuse",)
                    ),
                )
            )
            # Actual API envelope: compile_project_tasks returns TaskSpec[] under tasks.
            return httpx.Response(
                201,
                json={
                    "fingerprint": _hash(payload),
                    "workspace_revision": self.store.get_latest_project_workspace("p").revision,
                    "h3_execution_profile": None,
                    "generation_batch": None,
                    "tasks": [task.model_dump(mode="json") for task in tasks],
                },
            )
        raise AssertionError(request.url.path)

    async def advance(self, *, finish=True, operation=None):
        parent = self.store.get_task("parent")
        if parent.state == TaskState.READY:
            self.store.transition_task("parent", TaskState.QUEUED)
        assert self.store.acquire_dispatcher("dispatcher")
        claimed = self.store.claim_local_task("parent", "dispatcher")
        assert claimed is not None
        token = execution_guard.set((claimed.task_id, claimed.attempt_id))
        try:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(self.http), base_url="http://internal"
            ) as internal:
                done = await run_batch_project_orchestration(
                    claimed,
                    store=self.store,
                    internal=internal,
                    operate_project_with_agent=operation or self.operate,
                )
            if done and finish:
                self.store.transition_task("parent", TaskState.SUCCEEDED)
            return done
        finally:
            execution_guard.reset(token)


@pytest.mark.asyncio
async def test_approved_outline_advances_one_semantic_step_per_claim_and_preserves_review(tmp_path):
    scenario = Scenario(tmp_path / "flow.db", _workspace())
    original_policy = scenario.store.get_project_run_state("p").review_policy
    assert await scenario.advance() is False
    assert scenario.operations == ["initialize_storyboard"]
    assert scenario.store.get_task("parent").state == TaskState.READY
    assert scenario.store.get_task("parent").lease_expires_at is None
    assert scenario.calls == []

    scenario.restart()
    assert await scenario.advance() is False
    assert scenario.operations == ["initialize_storyboard", "initialize_assets"]
    scenario.restart()
    assert await scenario.advance() is False
    assert scenario.calls == ["/api/v1/projects/p/prompts/h3/shot.C01/regenerate"]
    scenario.restart()
    assert await scenario.advance() is True
    assert scenario.store.get_task("parent").attempt == 1
    assert scenario.store.get_project_run_state("p").review_policy == original_policy
    assert set(scenario.store.list_batch_runs()[0].items[0].task_ids) == {
        "parent",
        "encode",
        "diffuse",
        "export",
    }
    assert scenario.store.get_task("diffuse").depends_on == ("encode",)


@pytest.mark.asyncio
async def test_prepared_inputs_compile_without_model_call_and_recover_registered_dag(tmp_path):
    payload = _workspace(prepared=True, needs_image=True)
    payload["assetPlans"][0]["fulfilledByAssetId"] = "existing-asset"
    scenario = Scenario(tmp_path / "flow.db", payload)
    assert await scenario.advance(finish=False) is True
    assert scenario.operations == []
    assert scenario.calls == ["/api/v1/projects/p/tasks/compile"]
    # Crash after registration but before parent completion: read the persisted checkpoint.
    payload = scenario.payload()
    payload["updatedAt"] = "2026-09-24T12:00:00Z"
    payload["activeStage"] = "tasks"
    scenario.save(payload)
    scenario.restart()
    # Simulate the old control service relinquishing its unique dispatcher lease.
    scenario.store.release_dispatcher("dispatcher")
    scenario.store.recover_local_attempts()
    assert await scenario.advance() is True
    assert scenario.calls == ["/api/v1/projects/p/tasks/compile"]
    assert len(scenario.store.list_tasks()) == 4
    assert scenario.store.get_task("parent").attempt == 1


@pytest.mark.asyncio
async def test_execution_input_changed_after_compile_requires_explicit_recompile(tmp_path):
    scenario = Scenario(tmp_path / "flow.db", _workspace(prepared=True))
    assert await scenario.advance(finish=False) is True
    payload = scenario.payload()
    payload["prompts"]["h3Prompts"][0]["prompt"] = "a revised camera motion"
    scenario.save(payload)
    scenario.restart()
    # Simulate the old control service relinquishing its unique dispatcher lease.
    scenario.store.release_dispatcher("dispatcher")
    scenario.store.recover_local_attempts()
    with pytest.raises(OrchestrationNeedsAttention, match="Workspace changed"):
        await scenario.advance()
    assert scenario.calls == ["/api/v1/projects/p/tasks/compile"]


@pytest.mark.asyncio
async def test_image_children_join_paused_batch_before_response_and_free_llm_slot(tmp_path):
    scenario = Scenario(tmp_path / "flow.db", _workspace(prepared=True, needs_image=True))
    scenario.pause_during_image_registration = True
    assert await scenario.advance() is False
    parent = scenario.store.get_task("parent")
    assert parent.state == TaskState.BLOCKED
    assert len(parent.depends_on) == 1
    assert parent.lease_expires_at is None
    assert scenario.store.list_batch_runs()[0].state == BatchState.PAUSED
    child_id = parent.depends_on[0]
    assert child_id in scenario.members_at_insertion[0]
    assert not any("/prompts/h3/" in call for call in scenario.calls)
    # With one unrelated LLM already running, a second can still claim after this yield.
    for index in range(2):
        independent = _task(
            f"independent-{index}",
            TaskKind.LLM_PLANNING,
            project_id=f"other-{index}",
            state=TaskState.QUEUED,
        )
        scenario.store.add_task(independent)
        assert scenario.store.claim_local_task(independent.task_id, "dispatcher") is not None


@pytest.mark.asyncio
async def test_restart_after_image_binding_reuses_image_without_new_generation(tmp_path):
    payload = _workspace(prepared=True, needs_image=True)
    payload["prompts"]["imagePrompts"] = [_image_prompt()]
    payload["prompts"]["h3Prompts"] = []
    scenario = Scenario(tmp_path / "flow.db", payload)
    assert await scenario.advance() is False
    child_id = scenario.store.get_task("parent").depends_on[0]
    scenario.store.transition_task(child_id, TaskState.RUNNING)
    scenario.store.transition_task(child_id, TaskState.SUCCEEDED)
    payload = scenario.payload()
    payload["assetPlans"][0]["fulfilledByAssetId"] = "generated-asset"
    scenario.save(payload)
    scenario.store.transition_task("parent", TaskState.READY)
    scenario.restart()
    assert await scenario.advance() is False
    assert await scenario.advance() is True
    assert scenario.operations == []
    assert sum(call.endswith("/image-prompts/character/run") for call in scenario.calls) == 1
    assert sum("/prompts/images/" in call for call in scenario.calls) == 0
    assert sum("/prompts/h3/" in call for call in scenario.calls) == 1


@pytest.mark.asyncio
async def test_old_image_for_same_plan_cannot_replace_current_prompt_inputs(tmp_path):
    payload = _workspace(prepared=True, needs_image=True)
    payload["prompts"]["imagePrompts"] = [_image_prompt("the revised character")]
    scenario = Scenario(tmp_path / "flow.db", payload)
    old_id = "image:p:" + hashlib.sha256(b"character").hexdigest()[:12] + ":old-version"
    scenario.store.add_task(_task(old_id, TaskKind.IMAGE_GENERATION, state=TaskState.SUCCEEDED))
    scenario.store.append_task_checkpoint(
        "parent",
        "batch_image_child",
        {
            "scope": "image:character",
            "task_id": old_id,
            "input_fingerprint": "old",
        },
    )
    assert await scenario.advance() is False
    parent = scenario.store.get_task("parent")
    assert len(parent.depends_on) == 1 and old_id not in parent.depends_on
    assert scenario.calls == ["/api/v1/projects/p/image-prompts/character/run"]
    assert old_id not in scenario.store.list_batch_runs()[0].items[0].task_ids


@pytest.mark.asyncio
async def test_uncertain_call_requires_same_input_reconciliation_but_allows_correction(tmp_path):
    scenario = Scenario(tmp_path / "flow.db", _workspace())

    async def interrupted(project_id, request):
        raise TimeoutError("response lost")

    with pytest.raises(TimeoutError):
        await scenario.advance(operation=interrupted)
    scenario.restart()
    # Simulate the old control service relinquishing its unique dispatcher lease.
    scenario.store.release_dispatcher("dispatcher")
    scenario.store.recover_local_attempts()
    # Updating a UI timestamp is not a semantic correction or replay authorization.
    payload = scenario.payload()
    payload["updatedAt"] = "2026-09-24T12:00:00Z"
    scenario.save(payload)
    with pytest.raises(OrchestrationNeedsAttention, match="no validated output"):
        await scenario.advance()
    assert scenario.operations == []
    # Simulate the old control service relinquishing its unique dispatcher lease.
    scenario.store.release_dispatcher("dispatcher")
    scenario.store.recover_local_attempts()
    payload["outline"][0]["title"] = "Corrected arrival"
    scenario.save(payload)
    assert await scenario.advance() is False
    assert scenario.operations == ["initialize_storyboard"]
    started = [
        checkpoint
        for checkpoint in scenario.store.list_task_checkpoints("parent")
        if checkpoint.phase == "batch_operation_started"
    ]
    assert len(started) == 2
    assert started[0].payload["operation_id"] != started[1].payload["operation_id"]


@pytest.mark.asyncio
async def test_restart_after_image_registration_before_response_reuses_exact_child(tmp_path):
    payload = _workspace(prepared=True, needs_image=True)
    payload["prompts"]["imagePrompts"] = [_image_prompt()]
    scenario = Scenario(tmp_path / "flow.db", payload)
    original_http = scenario.http

    async def interrupted(request):
        await original_http(request)
        raise httpx.ReadTimeout("response lost after durable child insertion")

    scenario.http = interrupted
    with pytest.raises(httpx.ReadTimeout):
        await scenario.advance()
    children = [
        task for task in scenario.store.list_tasks() if task.kind == TaskKind.IMAGE_GENERATION
    ]
    assert len(children) == 1
    assert children[0].task_id in scenario.members_at_insertion[0]
    assert not any(
        checkpoint.phase == "batch_image_child"
        for checkpoint in scenario.store.list_task_checkpoints("parent")
    )
    scenario.http = original_http
    scenario.restart()
    # Simulate the old control service relinquishing its unique dispatcher lease.
    scenario.store.release_dispatcher("dispatcher")
    scenario.store.recover_local_attempts()
    assert await scenario.advance() is False
    children_after = [
        task for task in scenario.store.list_tasks() if task.kind == TaskKind.IMAGE_GENERATION
    ]
    assert [child.task_id for child in children_after] == [children[0].task_id]
    assert children_after[0].attempt == 0
    assert scenario.store.get_task("parent").depends_on == (children[0].task_id,)


@pytest.mark.asyncio
async def test_failed_exact_image_input_is_registered_without_implicit_retry(tmp_path):
    payload = _workspace(prepared=True, needs_image=True)
    payload["prompts"]["imagePrompts"] = [_image_prompt()]
    scenario = Scenario(tmp_path / "flow.db", payload)
    child_id = (
        "image:p:"
        + hashlib.sha256(b"character").hexdigest()[:12]
        + ":"
        + _hash([payload["prompts"]["imagePrompts"][0], payload["assetPlans"][0]])[:16]
    )
    scenario.store.add_task(_task(child_id, TaskKind.IMAGE_GENERATION, state=TaskState.FAILED))
    assert await scenario.advance() is False
    assert scenario.store.get_task(child_id).state == TaskState.FAILED
    assert scenario.store.get_task(child_id).attempt == 0
    assert scenario.store.get_task("parent").depends_on == (child_id,)
    assert scenario.store.get_task("parent").state == TaskState.BLOCKED


@pytest.mark.asyncio
async def test_failed_image_blocks_only_its_parent_and_other_project_can_claim(tmp_path):
    independent = _task("independent-video", TaskKind.H3_GENERATION, project_id="other")
    scenario = Scenario(
        tmp_path / "flow.db",
        _workspace(prepared=True, needs_image=True),
        other_tasks=(independent,),
    )
    assert await scenario.advance() is False
    child_id = scenario.store.get_task("parent").depends_on[0]
    scenario.store.transition_task(child_id, TaskState.RUNNING)
    scenario.store.transition_task(child_id, TaskState.FAILED)
    assert reconcile_batch_runs(scenario.store)[0].state == BatchState.RUNNING
    assert scenario.store.get_task("parent").state == TaskState.BLOCKED
    frontier = batch_dispatch_frontier(scenario.store.list_tasks())
    assert tuple(task.task_id for task in frontier) == (independent.task_id,)
    scenario.store.transition_task(independent.task_id, TaskState.QUEUED)
    assert scenario.store.claim_local_task(independent.task_id, "dispatcher") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [{}, {"tasks": ["wrong"]}, {"tasks": []}, {"tasks": [{"task_id": "parent"}]}]
)
async def test_malformed_compiler_response_does_not_finish_orchestrator(tmp_path, result):
    scenario = Scenario(tmp_path / "flow.db", _workspace(prepared=True))

    async def malformed(request):
        return httpx.Response(201, json=result)

    scenario.http = malformed
    with pytest.raises(OrchestrationNeedsAttention, match="Compiler"):
        await scenario.advance()
    assert scenario.store.get_task("parent").state == TaskState.RUNNING
    assert scenario.store.list_batch_runs()[0].state == BatchState.RUNNING
