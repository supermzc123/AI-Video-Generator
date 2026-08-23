from datetime import UTC, datetime

from ai_video_generator.domain import (
    GenerationBatch,
    GenerationBatchKind,
    GenerationBatchState,
    ReworkAction,
    ReworkMarker,
    ReworkMarkerState,
)
from ai_video_generator.persistence import SQLiteTaskStore
from ai_video_generator.services.generation_batches import (
    frozen_segment_ids,
    motion_context_segments,
    reconcile_generation_batches,
    rework_closure,
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


def test_motion_context_order_is_depth_first() -> None:
    payload = {"prompts": {"h3Prompts": _segments()}}
    assert [item["segmentId"] for item in motion_context_segments(payload)] == [
        "s1.C01",
        "s2.C01",
        "s1.C02",
    ]


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
