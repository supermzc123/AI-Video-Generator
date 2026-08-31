import hashlib
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings, load_runtime_settings, save_runtime_settings
from ai_video_generator.domain import (
    BatchRun,
    BatchRunItem,
    BatchState,
    ComfyUIOutput,
    ReviewDecision,
    ReviewDisposition,
    ReviewInputMode,
    ReviewIssue,
    ReviewIssueCategory,
    ReviewIssueSeverity,
    ReviewMode,
    TaskCheckpoint,
    TaskKind,
    TaskSpec,
    TaskState,
    TaskWorkloadManifest,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.services.batch_runs import reconcile_batch_runs
from ai_video_generator.services.media_review import (
    _parse_filter_findings,
    automatic_rework_policy,
    classify_automatic_rework,
)
from ai_video_generator.services.project_tasks import compile_project_task_plan


def manifest() -> TaskWorkloadManifest:
    return TaskWorkloadManifest(
        task_kind=TaskKind.H3_GENERATION,
        workflow_sha256="a" * 64,
        node_schema_sha256="b" * 64,
        prompt={
            "1": {"class_type": "Sampler", "inputs": {"seed": 12}},
            "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}},
        },
        outputs=(ComfyUIOutput(node_id="2", media_type="video/mp4"),),
        context={"project_id": "project-1", "segment_id": "segment-1"},
    )


def task(task_id: str, kind: TaskKind, state: TaskState, *, depends_on=()) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        project_id="project-1",
        kind=kind,
        state=state,
        idempotency_key=("c" if kind == TaskKind.H3_GENERATION else "d") * 64,
        input_fingerprint=("e" if kind == TaskKind.H3_GENERATION else "f") * 64,
        workload_manifest_sha256=manifest().sha256 if kind == TaskKind.H3_GENERATION else None,
        depends_on=depends_on,
    )


def chain_task(
    task_id: str, kind: TaskKind, state: TaskState, *, depends_on: tuple[str, ...] = ()
) -> TaskSpec:
    identity = hashlib.sha256(task_id.encode()).hexdigest()
    return TaskSpec(
        task_id=task_id,
        project_id="project-1",
        kind=kind,
        state=state,
        idempotency_key=identity,
        input_fingerprint=hashlib.sha256(f"input:{task_id}".encode()).hexdigest(),
        workload_manifest_sha256=manifest().sha256 if kind == TaskKind.H3_GENERATION else None,
        depends_on=depends_on,
    )


def test_review_decision_enforces_confidence_and_persists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confidence"):
        ReviewDecision(
            decision_id="bad",
            task_id="review",
            project_id="project-1",
            segment_id="segment-1",
            disposition=ReviewDisposition.ACCEPTED,
            confidence=0.84,
            input_mode=ReviewInputMode.FRAMES,
            created_at=datetime.now(UTC),
        )
    store = SQLiteTaskStore(tmp_path / "tasks.db")
    store.put_workload_manifest(manifest())
    store.add_task(task("h3", TaskKind.H3_GENERATION, TaskState.SUCCEEDED))
    store.add_task(task("review", TaskKind.AI_REVIEW, TaskState.FAILED, depends_on=("h3",)))
    decision = ReviewDecision(
        decision_id="review:decision",
        task_id="review",
        project_id="project-1",
        segment_id="segment-1",
        disposition=ReviewDisposition.REJECTED,
        confidence=0.96,
        issues=(
            ReviewIssue(
                category=ReviewIssueCategory.CONTINUITY,
                severity=ReviewIssueSeverity.ERROR,
                message="接缝处身份漂移",
                start_seconds=2,
                end_seconds=3,
            ),
        ),
        input_mode=ReviewInputMode.VIDEO,
        deterministic_checks_passed=True,
        created_at=datetime.now(UTC),
    )
    assert store.put_review_decision(decision) == decision
    assert store.list_review_decisions("review") == (decision,)


def test_filter_findings_detects_long_black_and_freeze() -> None:
    findings = _parse_filter_findings(
        "black_start:1 black_end:2.5 black_duration:1.5\n"
        "[Parsed_freezedetect_0] lavfi.freezedetect.freeze_start: 3",
        5,
    )
    assert findings[0].category == ReviewIssueCategory.BLACK_FRAME
    assert findings[0].severity == ReviewIssueSeverity.ERROR
    assert findings[1].category == ReviewIssueCategory.FREEZE


def test_automatic_rework_classification_is_conservative() -> None:
    media = ReviewIssue(
        category=ReviewIssueCategory.MEDIA,
        severity=ReviewIssueSeverity.ERROR,
        message="容器损坏",
    )
    artifact = ReviewIssue(
        category=ReviewIssueCategory.ARTIFACT,
        severity=ReviewIssueSeverity.ERROR,
        message="一帧闪烁",
    )
    continuity = ReviewIssue(
        category=ReviewIssueCategory.CONTINUITY,
        severity=ReviewIssueSeverity.ERROR,
        message="角色跨接缝漂移",
    )
    assert classify_automatic_rework((media,))[0].value == "retry"
    assert classify_automatic_rework((artifact,))[0].value == "change_seed"
    action, reason = classify_automatic_rework((continuity,))
    assert action is not None and action.value == "revise_prompt"
    assert "等待人工处理" in reason
    blocked_by_human, human_reason = automatic_rework_policy(
        (artifact,), effective_mode=ReviewMode.HUMAN_AI, previous_reworks=0
    )
    blocked_by_limit, limit_reason = automatic_rework_policy(
        (artifact,), effective_mode=ReviewMode.AI_ONLY, previous_reworks=2
    )
    assert blocked_by_human is None and "人工审核" in human_reason
    assert blocked_by_limit is None and "最多2次" in limit_reason


def test_video_capability_setting_round_trips(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_root=tmp_path, llm_video_capable=True)
    save_runtime_settings(settings)
    loaded = load_runtime_settings(Settings(_env_file=None, data_root=tmp_path))
    assert loaded.llm_video_capable is True


@pytest.mark.asyncio
async def test_segment_rework_uses_explicit_project_and_segment_identity(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    source_manifest = manifest()
    source = task("source-h3", TaskKind.H3_GENERATION, TaskState.SUCCEEDED)
    review = task(
        "source-review", TaskKind.AI_REVIEW, TaskState.FAILED, depends_on=("source-h3",)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.post(
                "/api/v1/workload-manifests",
                json=source_manifest.model_dump(mode="json"),
            )
        ).status_code == 201
        for value in (source, review):
            assert (
                await client.post("/api/v1/tasks", json=value.model_dump(mode="json"))
            ).status_code == 201
        created = await client.post(
            "/api/v1/projects/project-1/segments/segment-1/rework",
            json={
                "source_h3_task_id": "source-h3",
                "review_task_id": "source-review",
                "action": "change_seed",
                "feedback": "短暂伪影，换 seed 重试",
                "replacement_seed": 99,
            },
        )
        repeated = await client.post(
            "/api/v1/projects/project-1/segments/segment-1/rework",
            json={
                "source_h3_task_id": "source-h3",
                "review_task_id": "source-review",
                "action": "change_seed",
                "feedback": "短暂伪影，换 seed 重试",
                "replacement_seed": 99,
            },
        )
        mismatch = await client.post(
            "/api/v1/projects/wrong-project/segments/segment-1/rework",
            json={
                "source_h3_task_id": "source-h3",
                "action": "retry",
                "feedback": "重试",
            },
        )
        listed = await client.get("/api/v1/projects/project-1/reworks")
        replacement = await client.get(
            f"/api/v1/tasks/{created.json()['replacement_task_id']}"
        )
        prompt_revision = await client.post(
            "/api/v1/projects/project-1/segments/segment-1/rework",
            json={
                "source_h3_task_id": "source-h3",
                "review_task_id": "source-review",
                "action": "revise_prompt",
                "feedback": "连续性问题需要携带审核反馈重写提示词",
            },
        )

    assert created.status_code == 201
    assert created.json()["project_id"] == "project-1"
    assert created.json()["segment_id"] == "segment-1"
    assert created.json()["state"] == "queued"
    assert created.json()["replacement_task_id"].endswith(":h3")
    assert repeated.json() == created.json()
    assert replacement.json()["state"] == "queued"
    stored = SQLiteTaskStore(tmp_path / "control-plane.db")
    replacement_task = stored.get_task(created.json()["replacement_task_id"])
    replacement_manifest = stored.get_workload_manifest(
        replacement_task.workload_manifest_sha256 or ""
    ).manifest
    assert replacement_manifest.prompt["1"]["inputs"]["seed"] == 99
    assert prompt_revision.json()["state"] == "needs_prompt_revision"
    assert prompt_revision.json()["replacement_task_id"] is None
    assert mismatch.status_code == 409
    assert listed.json() == [created.json()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "post_kinds",
    [
        (),
        (TaskKind.SEEDVR2, TaskKind.RIFE, TaskKind.WHISPER),
    ],
)
async def test_rework_clones_delivery_chain_onto_replacement_video(
    tmp_path: Path, post_kinds: tuple[TaskKind, ...]
) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    source = chain_task("source-h3", TaskKind.H3_GENERATION, TaskState.SUCCEEDED)
    review = chain_task(
        "source-review", TaskKind.AI_REVIEW, TaskState.FAILED, depends_on=(source.task_id,)
    )
    original_downstream: list[TaskSpec] = []
    dependency = review.task_id
    for index, kind in enumerate((*post_kinds, TaskKind.EXPORT)):
        value = chain_task(
            f"old-{index}-{kind.value}",
            kind,
            TaskState.BLOCKED,
            depends_on=(dependency,),
        )
        original_downstream.append(value)
        dependency = value.task_id

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/workload-manifests", json=manifest().model_dump(mode="json")
        )
        for value in (source, review, *original_downstream):
            response = await client.post(
                "/api/v1/tasks", json=value.model_dump(mode="json")
            )
            assert response.status_code == 201
        created = await client.post(
            "/api/v1/projects/project-1/segments/segment-1/rework",
            json={
                "source_h3_task_id": source.task_id,
                "review_task_id": review.task_id,
                "action": "retry",
                "feedback": "局部重跑后继续原交付链",
            },
        )

    assert created.status_code == 201
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    all_tasks = store.list_tasks(project_id="project-1")
    replacement_h3 = store.get_task(created.json()["replacement_task_id"])
    replacement_review = next(
        value
        for value in all_tasks
        if value.task_id.startswith("rework:") and value.kind == TaskKind.AI_REVIEW
    )
    clones = [value for value in all_tasks if ":downstream:" in value.task_id]
    assert [value.kind for value in clones] == [*post_kinds, TaskKind.EXPORT]
    assert replacement_review.depends_on == (replacement_h3.task_id,)
    assert all(
        store.get_task(value.task_id).state == TaskState.CANCELLED
        for value in original_downstream
    )

    by_id = {value.task_id: value for value in all_tasks}
    cloned_export = next(value for value in clones if value.kind == TaskKind.EXPORT)
    reachable = set(cloned_export.depends_on)
    pending = list(reachable)
    while pending:
        current = by_id[pending.pop()]
        for dependency_id in current.depends_on:
            if dependency_id not in reachable:
                reachable.add(dependency_id)
                pending.append(dependency_id)
    assert replacement_h3.task_id in reachable
    assert review.task_id not in reachable
    assert source.task_id not in reachable

    store.transition_task(replacement_h3.task_id, TaskState.RUNNING)
    store.transition_task(replacement_h3.task_id, TaskState.SUCCEEDED)
    assert store.get_task(replacement_review.task_id).state == TaskState.READY
    store.transition_task(replacement_review.task_id, TaskState.QUEUED)
    store.transition_task(replacement_review.task_id, TaskState.RUNNING)
    store.transition_task(replacement_review.task_id, TaskState.SUCCEEDED)
    for clone in clones[:-1]:
        assert store.get_task(clone.task_id).state == TaskState.READY
        store.transition_task(clone.task_id, TaskState.QUEUED)
        store.transition_task(clone.task_id, TaskState.RUNNING)
        store.transition_task(clone.task_id, TaskState.SUCCEEDED)
    assert store.get_task(cloned_export.task_id).state == TaskState.READY


@pytest.mark.asyncio
async def removed_execution_status_uses_latest_rework_review_without_blocking_other_segments(
    tmp_path: Path,
) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    payload = {
        "width": 320,
        "height": 480,
        "referenceAssetMode": "none",
        "prompts": {
            "imagePrompts": [],
            "h3Prompts": [
                {
                    "segmentId": "segment-1",
                    "continuationOf": None,
                    "prompt": "integrated_multimodal_description: 测试\n"
                    "overall_soundscape: 环境声\nnon_diegetic_music: N/A",
                    "review": {"ready": True},
                },
                {
                    "segmentId": "segment-2",
                    "continuationOf": None,
                    "prompt": "integrated_multimodal_description: 后续片段\n"
                    "overall_soundscape: 环境声\nnon_diegetic_music: N/A",
                    "review": {"ready": True},
                },
            ],
        },
        "postProcessing": {
            "seedvr": {"enabled": False},
            "rife": {"enabled": False},
            "whisper": {"enabled": False},
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/projects",
            json={
                "project_id": "project-1",
                "name": "Review continuation",
                "width": 320,
                "height": 480,
                "target_duration_seconds": 8,
            },
        )
        await client.post(
            "/api/v1/projects/project-1/workspace",
            json={"revision": 1, "payload": payload},
        )

        store = SQLiteTaskStore(tmp_path / "control-plane.db")
        run_state = store.get_project_run_state("project-1")
        plan = compile_project_task_plan(
            project_id="project-1",
            workspace_revision=1,
            payload=payload,
            run_state=run_state,
            approved_image_workflows=set(),
            approved_image_harnesses={},
        )
        manifests: dict[str, str] = {}
        for segment_id in ("segment-1", "segment-2"):
            record = store.put_workload_manifest(
                manifest().model_copy(
                    update={"context": {"project_id": "project-1", "segment_id": segment_id}}
                )
            )
            manifests[segment_id] = record.sha256

        h3_index = 0
        failed_review_assigned = False
        for planned in plan.tasks:
            if planned.kind == TaskKind.H3_GENERATION:
                segment_id = f"segment-{h3_index + 1}"
                h3_index += 1
                stored = planned.model_copy(
                    update={
                        "state": TaskState.SUCCEEDED,
                        "workload_manifest_sha256": manifests[segment_id],
                    }
                )
            elif planned.kind == TaskKind.AI_REVIEW:
                stored = planned.model_copy(
                    update={
                        "state": (
                            TaskState.FAILED
                            if not failed_review_assigned
                            else TaskState.SUCCEEDED
                        )
                    }
                )
                failed_review_assigned = True
            elif planned.kind == TaskKind.EXPORT:
                stored = planned.model_copy(update={"state": TaskState.BLOCKED})
            else:
                stored = planned.model_copy(update={"state": TaskState.SUCCEEDED})
            store.add_task(stored)

        source_h3 = next(
            value
            for value in store.list_tasks(project_id="project-1")
            if value.kind == TaskKind.H3_GENERATION
            and store.get_workload_manifest(value.workload_manifest_sha256 or "")
            .manifest.context["segment_id"]
            == "segment-1"
        )
        source_review = next(
            value
            for value in store.list_tasks(project_id="project-1")
            if value.kind == TaskKind.AI_REVIEW
            and value.depends_on == (source_h3.task_id,)
        )
        created = await client.post(
            "/api/v1/projects/project-1/segments/segment-1/rework",
            json={
                "source_h3_task_id": source_h3.task_id,
                "review_task_id": source_review.task_id,
                "action": "retry",
                "feedback": "第一段返工时第二段审核仍须继续",
            },
        )
        initial = await client.get("/api/v1/projects/project-1/execution-status")
        replacement_h3 = store.get_task(created.json()["replacement_task_id"])
        replacement_review_id = f"{created.json()['request_id']}:review"
        store.transition_task(replacement_h3.task_id, TaskState.RUNNING)
        store.transition_task(replacement_h3.task_id, TaskState.SUCCEEDED)
        store.transition_task(replacement_review_id, TaskState.QUEUED)
        store.transition_task(replacement_review_id, TaskState.RUNNING)
        store.transition_task(replacement_review_id, TaskState.SUCCEEDED)
        reviewed = await client.get("/api/v1/projects/project-1/execution-status")
        cloned_export = next(
            value
            for value in store.list_tasks(project_id="project-1")
            if value.task_id.startswith(created.json()["request_id"])
            and value.kind == TaskKind.EXPORT
        )
        cloned_master = next(
            value
            for value in store.list_tasks(project_id="project-1")
            if value.task_id.startswith(created.json()["request_id"])
            and value.kind == TaskKind.MASTER_ASSEMBLY
        )
        store.transition_task(cloned_master.task_id, TaskState.QUEUED)
        store.transition_task(cloned_master.task_id, TaskState.RUNNING)
        store.transition_task(cloned_master.task_id, TaskState.SUCCEEDED)
        store.transition_task(cloned_export.task_id, TaskState.QUEUED)
        store.transition_task(cloned_export.task_id, TaskState.RUNNING)
        store.transition_task(cloned_export.task_id, TaskState.SUCCEEDED)
        delivered = await client.get("/api/v1/projects/project-1/execution-status")

    initial_ids = {value["task_id"] for value in initial.json()["tasks"]}
    assert replacement_h3.task_id in initial_ids
    assert replacement_review_id in initial_ids
    assert initial.json()["review_complete"] is False
    assert reviewed.json()["review_complete"] is True
    assert delivered.json()["delivery_complete"] is True


@pytest.mark.asyncio
async def test_batch_membership_is_replaced_before_rework_branch_runs(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    source = chain_task("source-h3", TaskKind.H3_GENERATION, TaskState.SUCCEEDED)
    review = chain_task(
        "source-review", TaskKind.AI_REVIEW, TaskState.FAILED, depends_on=(source.task_id,)
    )
    export = chain_task(
        "old-export", TaskKind.EXPORT, TaskState.BLOCKED, depends_on=(review.task_id,)
    )
    now = datetime(2026, 8, 16, tzinfo=UTC)
    batch = BatchRun(
        batch_id="rework-batch",
        name="返工批次",
        state=BatchState.RUNNING,
        items=(
            BatchRunItem(
                project_id="project-1",
                task_ids=(source.task_id, review.task_id, export.task_id),
            ),
        ),
        created_at=now,
        updated_at=now,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/workload-manifests", json=manifest().model_dump(mode="json")
        )
        for value in (source, review, export):
            await client.post("/api/v1/tasks", json=value.model_dump(mode="json"))
        SQLiteTaskStore(tmp_path / "control-plane.db").put_batch_run(batch)
        created = await client.post(
            "/api/v1/projects/project-1/segments/segment-1/rework",
            json={
                "source_h3_task_id": source.task_id,
                "review_task_id": review.task_id,
                "action": "retry",
                "feedback": "批次内自动返工",
            },
        )

    assert created.status_code == 201
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    updated = store.list_batch_runs()[0]
    member_ids = updated.items[0].task_ids
    assert review.task_id not in member_ids
    assert export.task_id not in member_ids
    assert created.json()["replacement_task_id"] in member_ids
    assert any(":downstream:export:" in value for value in member_ids)
    reconciled = reconcile_batch_runs(store, now=now)
    assert reconciled[0].state == BatchState.RUNNING


@pytest.mark.asyncio
async def test_continuation_review_only_matches_its_direct_h3_dependency(
    tmp_path: Path,
) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    previous_manifest = manifest().model_copy(
        update={"context": {"project_id": "project-1", "segment_id": "segment-previous"}}
    )
    continuation_manifest = manifest().model_copy(
        update={"context": {"project_id": "project-1", "segment_id": "segment-current"}}
    )
    previous = task("previous-h3", TaskKind.H3_GENERATION, TaskState.SUCCEEDED).model_copy(
        update={
            "idempotency_key": "1" * 64,
            "input_fingerprint": "2" * 64,
            "workload_manifest_sha256": previous_manifest.sha256,
        }
    )
    continuation = task(
        "continuation-h3",
        TaskKind.H3_GENERATION,
        TaskState.SUCCEEDED,
        depends_on=(previous.task_id,),
    ).model_copy(
        update={
            "idempotency_key": "3" * 64,
            "input_fingerprint": "4" * 64,
            "workload_manifest_sha256": continuation_manifest.sha256,
        }
    )
    review = task(
        "continuation-review",
        TaskKind.AI_REVIEW,
        TaskState.FAILED,
        depends_on=(continuation.task_id,),
    ).model_copy(
        update={"idempotency_key": "5" * 64, "input_fingerprint": "6" * 64}
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for value in (previous_manifest, continuation_manifest):
            assert (
                await client.post(
                    "/api/v1/workload-manifests", json=value.model_dump(mode="json")
                )
            ).status_code == 201
        for value in (previous, continuation, review):
            assert (
                await client.post("/api/v1/tasks", json=value.model_dump(mode="json"))
            ).status_code == 201

        wrong_source = await client.post(
            "/api/v1/projects/project-1/segments/segment-previous/rework",
            json={
                "source_h3_task_id": previous.task_id,
                "review_task_id": review.task_id,
                "action": "retry",
                "feedback": "不能把续段审核归到前段",
            },
        )
        direct_source = await client.post(
            "/api/v1/projects/project-1/segments/segment-current/rework",
            json={
                "source_h3_task_id": continuation.task_id,
                "review_task_id": review.task_id,
                "action": "retry",
                "feedback": "只重跑当前续段",
            },
        )

    assert wrong_source.status_code == 409
    assert wrong_source.json()["detail"] == "review task does not review source task"
    assert direct_source.status_code == 201
    assert direct_source.json()["segment_id"] == "segment-current"


@pytest.mark.asyncio
async def test_artifact_api_streams_only_registered_checkpoint_media(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path))
    media_path = tmp_path / "task-outputs" / "segment.mp4"
    media_path.parent.mkdir(parents=True)
    content = b"0123456789abcdef"
    media_path.write_bytes(content)
    source = task("artifact-h3", TaskKind.H3_GENERATION, TaskState.SUCCEEDED)
    checkpoint = TaskCheckpoint(
        checkpoint_id="artifact-h3:video",
        task_id=source.task_id,
        sequence=1,
        phase="video_saved",
        payload={
            "path": str(media_path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "segment_id": "segment-1",
        },
        created_at=datetime.now(UTC),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.post(
                "/api/v1/workload-manifests", json=manifest().model_dump(mode="json")
            )
        ).status_code == 201
        assert (
            await client.post("/api/v1/tasks", json=source.model_dump(mode="json"))
        ).status_code == 201
        assert (
            await client.post(
                f"/api/v1/tasks/{source.task_id}/checkpoints",
                json=checkpoint.model_dump(mode="json"),
            )
        ).status_code == 201
        artifacts = await client.get(f"/api/v1/tasks/{source.task_id}/artifacts")
        artifact_id = artifacts.json()[0]["artifact_id"]
        partial = await client.get(
            f"/api/v1/artifacts/{artifact_id}/media", headers={"Range": "bytes=2-5"}
        )
        unknown = await client.get(f"/api/v1/artifacts/{'0' * 64}/media")

    assert artifacts.status_code == 200
    assert artifacts.json()[0]["segment_id"] == "segment-1"
    assert artifacts.json()[0]["byte_size"] == len(content)
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["accept-ranges"] == "bytes"
    assert unknown.status_code == 404


@pytest.mark.asyncio
async def test_artifact_api_rejects_checkpoint_path_outside_data_root(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_root=tmp_path / "data"))
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not public")
    source = task("outside-h3", TaskKind.H3_GENERATION, TaskState.SUCCEEDED)
    checkpoint = TaskCheckpoint(
        checkpoint_id="outside-h3:video",
        task_id=source.task_id,
        sequence=1,
        phase="video_saved",
        payload={"path": str(outside), "segment_id": "segment-1"},
        created_at=datetime.now(UTC),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/v1/workload-manifests", json=manifest().model_dump(mode="json")
        )
        await client.post("/api/v1/tasks", json=source.model_dump(mode="json"))
        await client.post(
            f"/api/v1/tasks/{source.task_id}/checkpoints",
            json=checkpoint.model_dump(mode="json"),
        )
        artifacts = await client.get(f"/api/v1/tasks/{source.task_id}/artifacts")
    assert artifacts.json() == []
