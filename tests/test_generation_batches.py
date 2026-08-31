import hashlib
from datetime import UTC, datetime

from ai_video_generator.domain import (
    GenerationBatch,
    GenerationBatchKind,
    GenerationBatchState,
    ReworkAction,
    ReworkMarker,
    ReworkMarkerState,
    SegmentGenerationVersion,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.services.generation_batches import (
    cancel_rework_marker,
    create_rework_marker,
    frozen_segment_ids,
    motion_context_segments,
    reconcile_generation_batches,
    rework_closure,
)


def _version() -> SegmentGenerationVersion:
    return SegmentGenerationVersion(
        version_id="version-1",
        project_id="p",
        shot_id="s1",
        segment_id="s1.C01",
        segment_index=0,
        generation_number=1,
        batch_id="batch-1",
        task_id="task-1",
        seed=7,
        created_at=datetime.now(UTC),
    )


def test_technical_rework_does_not_require_user_feedback(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control-plane.db")

    marker = create_rework_marker(
        store,
        project_id="p",
        version=_version(),
        action=ReworkAction.RETRY,
        feedback=None,
        replacement_seed=None,
        source="human",
    )

    assert marker.feedback == "技术性失败，保持原提示词和 Seed 重新生成"


def test_seed_rework_does_not_require_user_feedback(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control-plane.db")

    marker = create_rework_marker(
        store,
        project_id="p",
        version=_version(),
        action=ReworkAction.CHANGE_SEED,
        feedback="",
        replacement_seed=8,
        source="human",
    )

    assert marker.feedback == "更换 Seed 重新生成"


def test_prompt_rework_still_requires_an_instruction(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control-plane.db")

    try:
        create_rework_marker(
            store,
            project_id="p",
            version=_version(),
            action=ReworkAction.REVISE_PROMPT,
            feedback=None,
            replacement_seed=None,
            source="human",
        )
    except ValueError as exc:
        assert "必须提供修改要求" in str(exc)
    else:
        raise AssertionError("prompt rework without an instruction must fail")


def _task(task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        project_id="p",
        kind=TaskKind.H3_GENERATION,
        state=TaskState.READY,
        idempotency_key=hashlib.sha256(f"key-{task_id}".encode()).hexdigest(),
        input_fingerprint=hashlib.sha256(f"fingerprint-{task_id}".encode()).hexdigest(),
    )


def _segments() -> list[dict[str, object]]:
    return [
        {"segmentId": "s1.C01", "shotId": "s1", "segmentIndex": 0},
        {"segmentId": "s1.C02", "shotId": "s1", "segmentIndex": 1},
        {"segmentId": "s2.C01", "shotId": "s2", "segmentIndex": 0},
    ]


def _marker(segment_id: str, index: int) -> ReworkMarker:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return ReworkMarker(
        marker_id=f"m-{segment_id}",
        project_id="p",
        shot_id=segment_id.split(".")[0],
        segment_id=segment_id,
        segment_index=index,
        source_version_id="v",
        action=ReworkAction.RETRY,
        feedback="bad frame",
        source="human",
        created_at=now,
        updated_at=now,
    )


def test_rework_freezes_from_marked_segment_without_freezing_other_sequences() -> None:
    marker = _marker("s1.C02", 1)
    assert frozen_segment_ids(_segments(), [marker]) == {"s1.C02"}
    assert {item["segmentId"] for item in rework_closure(_segments(), [marker])} == {"s1.C02"}


def test_cancelled_marker_does_not_freeze_sequence() -> None:
    marker = _marker("s1.C01", 0).model_copy(update={"state": ReworkMarkerState.CANCELLED})
    assert frozen_segment_ids(_segments(), [marker]) == set()


def test_sealed_rework_marker_can_be_cancelled_by_human(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    now = datetime.now(UTC)
    store.add_task(_task("replacement"))
    store.put_generation_batch(
        GenerationBatch(
            batch_id="generation:rework",
            project_id="p",
            kind=GenerationBatchKind.REWORK,
            state=GenerationBatchState.SEALED,
            generation_number=2,
            segment_ids=("s1.C01",),
            task_ids=("replacement",),
            dispatch_requested=True,
            created_at=now,
            updated_at=now,
        )
    )
    marker = store.put_rework_marker(
        _marker("s1.C01", 0).model_copy(
            update={"state": ReworkMarkerState.SEALED, "batch_id": "generation:rework"}
        )
    )

    cancelled = cancel_rework_marker(store, marker)

    assert cancelled.state == ReworkMarkerState.CANCELLED
    assert store.get_task("replacement").state == TaskState.CANCELLED
    batch = store.list_generation_batches("p")[0]
    assert batch.state == GenerationBatchState.CANCELLED
    assert batch.dispatch_requested is False


def test_motion_context_order_is_depth_first() -> None:
    payload = {"prompts": {"h3Prompts": _segments()}}
    assert [item["segmentId"] for item in motion_context_segments(payload)] == [
        "s1.C01",
        "s2.C01",
        "s1.C02",
    ]


def test_motion_context_edges_are_derived_from_present_shot_segments() -> None:
    payload = {
        "prompts": {
            "h3Prompts": [
                {
                    "shotId": "shot-7",
                    "segmentId": "shot-7-seg-1",
                    "segmentIndex": 0,
                    "continuationOf": "missing-segment",
                },
                {
                    "shotId": "shot-7",
                    "segmentId": "shot-7-seg-2",
                    "segmentIndex": 1,
                    "continuationOf": "another-missing-segment",
                },
            ]
        }
    }

    segments = motion_context_segments(payload)

    assert segments[0]["continuationOf"] is None
    assert segments[1]["continuationOf"] == "shot-7-seg-1"


def test_reconcile_cancels_orphan_batch_without_stopping_dispatch(tmp_path) -> None:
    store = SQLiteTaskStore(tmp_path / "control-plane.db")
    now = datetime.now(UTC)
    store.put_generation_batch(
        GenerationBatch(
            batch_id="generation:orphan",
            project_id="p",
            kind=GenerationBatchKind.INITIAL,
            generation_number=1,
            segment_ids=("s1.C01",),
            task_ids=("missing-h3-task",),
            encoding_task_ids=("missing-encode-task",),
            model_switch_task_id="missing-switch-task",
            dispatch_requested=True,
            created_at=now,
            updated_at=now,
        )
    )

    reconcile_generation_batches(store)

    batch = store.list_generation_batches("p")[0]
    assert batch.state == GenerationBatchState.CANCELLED
    assert batch.dispatch_requested is False
