from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ai_video_generator.domain import (
    GenerationBatch,
    GenerationBatchKind,
    GenerationBatchState,
    ReworkAction,
    ReworkMarker,
    ReworkMarkerState,
    SegmentGenerationVersion,
    SegmentVersionState,
    TaskKind,
    TaskSpec,
    TaskState,
)
from ai_video_generator.persistence import SQLiteTaskStore, TaskNotFoundError

OPEN_MARKER_STATES = {
    ReworkMarkerState.DRAFT,
    ReworkMarkerState.PREPARING,
    ReworkMarkerState.SEALED,
}


def motion_context_segments(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    prompts = payload.get("prompts")
    values = prompts.get("h3Prompts") if isinstance(prompts, dict) else None
    if not isinstance(values, list):
        return ()
    indexed = [
        (position, item)
        for position, item in enumerate(values)
        if isinstance(item, dict) and str(item.get("segmentId") or "")
    ]
    ordered = tuple(
        item
        for _, item in sorted(
            indexed,
            key=lambda pair: (
                int(pair[1].get("segmentIndex") or 0),
                pair[0],
            ),
        )
    )
    return canonical_motion_context_chain(ordered)


def canonical_motion_context_chain(
    segments: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Derive continuation edges from the current shot topology."""
    values = [dict(item) for item in segments]
    by_id = {str(item.get("segmentId") or ""): item for item in values}

    def shot_identity(segment: dict[str, Any]) -> str:
        cursor = segment
        seen: set[str] = set()
        while True:
            explicit = str(cursor.get("shotId") or "")
            if explicit:
                return explicit
            segment_id = str(cursor.get("segmentId") or "")
            if ".C" in segment_id:
                return segment_id.split(".C", 1)[0]
            if "-seg-" in segment_id:
                return segment_id.rsplit("-seg-", 1)[0]
            if segment_id in seen:
                return min(seen)
            seen.add(segment_id)
            predecessor = str(cursor.get("continuationOf") or "")
            if not predecessor or predecessor not in by_id:
                return segment_id
            cursor = by_id[predecessor]

    by_shot: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for position, segment in enumerate(values):
        shot_id = shot_identity(segment)
        segment["shotId"] = shot_id
        by_shot[shot_id].append((position, segment))
    for shot in by_shot.values():
        previous: str | None = None
        for _, segment in sorted(
            shot,
            key=lambda item: (int(item[1].get("segmentIndex") or 0), item[0]),
        ):
            segment["continuationOf"] = previous
            previous = str(segment["segmentId"])
    return tuple(values)


def frozen_segment_ids(
    segments: Iterable[dict[str, Any]], markers: Iterable[ReworkMarker]
) -> frozenset[str]:
    earliest: dict[str, int] = {}
    for marker in markers:
        if marker.state not in OPEN_MARKER_STATES:
            continue
        earliest[marker.shot_id] = min(
            marker.segment_index, earliest.get(marker.shot_id, marker.segment_index)
        )
    return frozenset(
        str(segment.get("segmentId") or "")
        for segment in segments
        if str(segment.get("shotId") or "") in earliest
        and int(segment.get("segmentIndex") or 0) >= earliest[str(segment.get("shotId") or "")]
    )


def rework_closure(
    segments: Iterable[dict[str, Any]], markers: Iterable[ReworkMarker]
) -> tuple[dict[str, Any], ...]:
    earliest: dict[str, int] = {}
    for marker in markers:
        if marker.state != ReworkMarkerState.DRAFT:
            continue
        earliest[marker.shot_id] = min(
            marker.segment_index, earliest.get(marker.shot_id, marker.segment_index)
        )
    return tuple(
        segment
        for segment in segments
        if str(segment.get("shotId") or "") in earliest
        and int(segment.get("segmentIndex") or 0) >= earliest[str(segment.get("shotId") or "")]
    )


def ensure_initial_generation_batch(
    store: SQLiteTaskStore,
    *,
    project_id: str,
    payload: dict[str, Any],
    tasks: Iterable[TaskSpec],
) -> GenerationBatch:
    existing = store.list_generation_batches(project_id)
    segments = motion_context_segments(payload)
    h3_tasks = [task for task in tasks if task.kind == TaskKind.H3_GENERATION]
    encode_tasks = [task for task in tasks if task.kind == TaskKind.CONDITIONING_ENCODING]
    switch = next((task for task in tasks if task.kind == TaskKind.MODEL_SWITCH), None)
    if len(segments) != len(h3_tasks) or switch is None:
        raise ValueError("generation batch does not match compiled H3 task plan")
    task_ids = tuple(task.task_id for task in h3_tasks)
    matching = next((item for item in existing if item.task_ids == task_ids), None)
    generation_number = (
        matching.generation_number
        if matching is not None
        else max((item.generation_number for item in existing), default=0) + 1
    )
    active = {
        item.segment_id: item
        for item in store.list_segment_generation_versions(project_id)
        if item.state == SegmentVersionState.ACTIVE
    }
    now = datetime.now(UTC)
    batch = matching or GenerationBatch(
        batch_id=f"generation:{project_id}:{uuid4().hex}",
        project_id=project_id,
        kind=GenerationBatchKind.INITIAL,
        state=GenerationBatchState.PREPARING,
        generation_number=generation_number,
        segment_ids=tuple(str(item["segmentId"]) for item in segments),
        task_ids=tuple(task.task_id for task in h3_tasks),
        encoding_task_ids=tuple(task.task_id for task in encode_tasks),
        model_switch_task_id=switch.task_id,
        created_at=now,
        updated_at=now,
    )
    if matching is None:
        store.put_generation_batch(batch)
    versions = {
        item.segment_id: item
        for item in store.list_segment_generation_versions(project_id)
        if item.batch_id == batch.batch_id
    }
    for task, segment in zip(h3_tasks, segments, strict=True):
        segment_id = str(segment["segmentId"])
        if segment_id in versions:
            continue
        prior_id = str(segment.get("continuationOf") or "")
        version = SegmentGenerationVersion(
            version_id=f"segment-version:{uuid4().hex}",
            project_id=project_id,
            shot_id=str(segment.get("shotId") or segment_id.split(".", 1)[0]),
            segment_id=segment_id,
            runtime_segment_id=segment_id,
            segment_index=int(segment.get("segmentIndex") or 0),
            generation_number=generation_number,
            batch_id=batch.batch_id,
            task_id=task.task_id,
            predecessor_version_id=(
                versions[prior_id].version_id if prior_id in versions else None
            ),
            parent_version_id=(active[segment_id].version_id if segment_id in active else None),
            prompt_revision_id=str(segment.get("id") or "") or None,
            seed=int(segment.get("seed") or 0),
            created_at=now,
        )
        versions[segment_id] = version
        store.put_segment_generation_version(version)
    return batch


def create_rework_marker(
    store: SQLiteTaskStore,
    *,
    project_id: str,
    version: SegmentGenerationVersion,
    action: ReworkAction,
    feedback: str | None,
    replacement_seed: int | None,
    source: str,
) -> ReworkMarker:
    now = datetime.now(UTC)
    normalized_feedback = (feedback or "").strip()
    if action == ReworkAction.REVISE_PROMPT and not normalized_feedback:
        raise ValueError("修改提示词返工必须提供修改要求")
    if not normalized_feedback:
        normalized_feedback = (
            "技术性失败，保持原提示词和 Seed 重新生成"
            if action == ReworkAction.RETRY
            else "更换 Seed 重新生成"
        )
    marker = ReworkMarker(
        marker_id=f"rework-marker:{uuid4().hex}",
        project_id=project_id,
        shot_id=version.shot_id,
        segment_id=version.segment_id,
        segment_index=version.segment_index,
        source_version_id=version.version_id,
        action=action,
        feedback=normalized_feedback,
        replacement_seed=replacement_seed,
        source=source,
        created_at=now,
        updated_at=now,
    )
    stored = store.put_rework_marker(marker)
    for candidate in store.list_segment_generation_versions(project_id):
        if (
            candidate.shot_id == version.shot_id
            and candidate.segment_index >= version.segment_index
            and candidate.state == SegmentVersionState.ACTIVE
        ):
            store.put_segment_generation_version(
                candidate.model_copy(
                    update={
                        "state": SegmentVersionState.SUPERSEDED,
                        "discard_reason": f"rework_marker:{stored.marker_id}",
                    }
                )
            )
    return stored


def cancel_rework_marker(store: SQLiteTaskStore, marker: ReworkMarker) -> ReworkMarker:
    if marker.state in {ReworkMarkerState.RESOLVED, ReworkMarkerState.CANCELLED}:
        return marker
    if marker.state in {ReworkMarkerState.PREPARING, ReworkMarkerState.SEALED} and marker.batch_id:
        batch = next(
            (
                item
                for item in store.list_generation_batches(marker.project_id)
                if item.batch_id == marker.batch_id
            ),
            None,
        )
        if batch and batch.state in {
            GenerationBatchState.PREPARING,
            GenerationBatchState.SEALED,
            GenerationBatchState.RUNNING,
        }:
            batch_task_ids = (
                *batch.task_ids,
                *((batch.model_switch_task_id,) if batch.model_switch_task_id else ()),
            )
            for task_id in batch_task_ids:
                task = store.get_task(task_id)
                if task.state not in {
                    TaskState.SUCCEEDED,
                    TaskState.FAILED,
                    TaskState.CANCELLED,
                    TaskState.STALE,
                }:
                    store.transition_task(task_id, TaskState.CANCELLED)
            store.put_generation_batch(
                batch.model_copy(
                    update={
                        "state": GenerationBatchState.CANCELLED,
                        "dispatch_requested": False,
                        "updated_at": datetime.now(UTC),
                    }
                )
            )
            for sibling in store.list_rework_markers(marker.project_id):
                if sibling.batch_id == batch.batch_id and sibling.marker_id != marker.marker_id:
                    store.put_rework_marker(
                        sibling.model_copy(
                            update={
                                "state": ReworkMarkerState.DRAFT,
                                "batch_id": None,
                                "updated_at": datetime.now(UTC),
                            }
                        )
                    )
    cancelled = store.put_rework_marker(
        marker.model_copy(
            update={"state": ReworkMarkerState.CANCELLED, "updated_at": datetime.now(UTC)}
        )
    )
    candidates = sorted(
        (
            item
            for item in store.list_segment_generation_versions(marker.project_id)
            if item.shot_id == marker.shot_id
            and item.segment_index >= marker.segment_index
            and item.discard_reason == f"rework_marker:{marker.marker_id}"
        ),
        key=lambda item: item.segment_index,
    )
    now = datetime.now(UTC)
    for candidate in candidates:
        store.put_segment_generation_version(
            candidate.model_copy(
                update={
                    "state": SegmentVersionState.ACTIVE,
                    "discard_reason": None,
                    "activated_at": now,
                }
            )
        )
    # Removing the marker releases the sequence immediately. Wake the latest
    # active generation batch so the scheduler recomputes ready work instead
    # of leaving the previously frozen segment idle.
    active_batch = next(
        (
            item
            for item in reversed(store.list_generation_batches(marker.project_id))
            if item.state
            in {
                GenerationBatchState.PREPARING,
                GenerationBatchState.SEALED,
                GenerationBatchState.RUNNING,
            }
        ),
        None,
    )
    if active_batch is not None:
        store.put_generation_batch(
            active_batch.model_copy(update={"dispatch_requested": True, "updated_at": now})
        )
    return cancelled


def _clone_manifest_for_version(
    store: SQLiteTaskStore,
    source: TaskSpec,
    *,
    runtime_segment_id: str,
    predecessor_runtime_segment_id: str | None,
    replacement_seed: int | None,
) -> str:
    if not source.workload_manifest_sha256:
        raise ValueError("source generation task has no workload manifest")
    manifest = store.get_workload_manifest(source.workload_manifest_sha256).manifest
    prompt = json.loads(json.dumps(manifest.prompt))
    for node in prompt.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        if replacement_seed is not None:
            for key in ("seed", "noise_seed"):
                if isinstance(inputs.get(key), int):
                    inputs[key] = replacement_seed
    for node_id in ("mc-save-1", "mc-save-2"):
        node = prompt.get(node_id)
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
            node["inputs"]["filename_prefix"] = (
                f"ai-video-generator/{source.project_id}/motion/{runtime_segment_id}"
            )
    load = prompt.get("mc-load-1")
    if (
        predecessor_runtime_segment_id
        and isinstance(load, dict)
        and isinstance(load.get("inputs"), dict)
    ):
        load["inputs"]["latent_path"] = (
            f"ai-video-generator/{source.project_id}/motion/"
            f"{predecessor_runtime_segment_id}_00001.safetensors"
        )
    workflow_sha = hashlib.sha256(
        json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return store.put_workload_manifest(
        manifest.model_copy(
            update={
                "prompt": prompt,
                "workflow_sha256": workflow_sha,
                "context": {
                    **manifest.context,
                    "runtime_segment_id": runtime_segment_id,
                    "predecessor_runtime_segment_id": predecessor_runtime_segment_id or "",
                },
            }
        )
    ).sha256


def confirm_rework_markers(
    store: SQLiteTaskStore,
    *,
    project_id: str,
    payload: dict[str, Any],
) -> GenerationBatch:
    markers = [
        item
        for item in store.list_rework_markers(project_id)
        if item.state == ReworkMarkerState.DRAFT
    ]
    if not markers:
        preparing = next(
            (
                item
                for item in reversed(store.list_generation_batches(project_id))
                if item.kind == GenerationBatchKind.REWORK
                and item.state == GenerationBatchState.PREPARING
            ),
            None,
        )
        if preparing is None:
            raise ValueError("no draft rework markers to confirm")
        return preparing
    segments = rework_closure(motion_context_segments(payload), markers)
    if not segments:
        raise ValueError("rework markers do not match current Motion Context segments")
    versions = store.list_segment_generation_versions(project_id)
    active = {
        item.segment_id: item for item in versions if item.state == SegmentVersionState.ACTIVE
    }
    latest = max((item.generation_number for item in versions), default=0) + 1
    now = datetime.now(UTC)
    batch_id = f"generation:{project_id}:{uuid4().hex}"
    affected_ids = {str(item["segmentId"]) for item in segments}
    source_by_segment: dict[str, SegmentGenerationVersion] = {}
    source_affinities: set[str] = set()
    all_by_segment: dict[str, list[SegmentGenerationVersion]] = defaultdict(list)
    for version in versions:
        all_by_segment[version.segment_id].append(version)
    for segment_id in affected_ids:
        source = active.get(segment_id) or max(
            all_by_segment[segment_id], key=lambda item: item.generation_number
        )
        source_by_segment[segment_id] = source
        source_affinity = store.get_task(source.task_id).affinity_key
        if source_affinity:
            source_affinities.add(source_affinity)
    if len(source_affinities) > 1:
        raise ValueError("rework source chain contains mixed H3 model stacks")
    encoding_ids: set[str] = set()
    for source_version in source_by_segment.values():
        source_task = store.get_task(source_version.task_id)
        for dependency_id in source_task.depends_on:
            dependency = store.get_task(dependency_id)
            if dependency.kind == TaskKind.CONDITIONING_ENCODING:
                encoding_ids.add(dependency_id)
    switch_id = f"{batch_id}:model-switch"
    h3_affinity = next(iter(source_affinities), "h3:diffusion:default")
    switch_fingerprint = hashlib.sha256(
        json.dumps(
            {"encoding_ids": sorted(encoding_ids), "affinity": h3_affinity},
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    switch = TaskSpec(
        task_id=switch_id,
        project_id=project_id,
        kind=TaskKind.MODEL_SWITCH,
        state=TaskState.READY
        if all(store.get_task(value).state == TaskState.SUCCEEDED for value in encoding_ids)
        else TaskState.BLOCKED,
        idempotency_key=hashlib.sha256(f"{switch_id}:{switch_fingerprint}".encode()).hexdigest(),
        input_fingerprint=switch_fingerprint,
        depends_on=tuple(sorted(encoding_ids)),
        affinity_key=h3_affinity,
        max_attempts=2,
    )
    store.add_task(switch)
    marker_by_segment = {marker.segment_id: marker for marker in markers}
    new_versions: dict[str, SegmentGenerationVersion] = {}
    new_tasks: list[TaskSpec] = []
    for stable_index, segment in enumerate(segments):
        segment_id = str(segment["segmentId"])
        source_version = source_by_segment[segment_id]
        source_task = store.get_task(source_version.task_id)
        prior_segment_id = str(segment.get("continuationOf") or "")
        predecessor = new_versions.get(prior_segment_id) or active.get(prior_segment_id)
        runtime_id = uuid4().hex
        predecessor_runtime_id = (
            predecessor.runtime_segment_id or predecessor.segment_id if predecessor else None
        )
        marker = marker_by_segment.get(segment_id)
        replacement_seed = marker.replacement_seed if marker else None
        manifest_sha = _clone_manifest_for_version(
            store,
            source_task,
            runtime_segment_id=runtime_id,
            predecessor_runtime_segment_id=predecessor_runtime_id,
            replacement_seed=replacement_seed,
        )
        task_id = f"{batch_id}:h3:{runtime_id}"
        dependencies = [switch_id, *sorted(encoding_ids)]
        if predecessor and predecessor.task_id not in dependencies:
            dependencies.append(predecessor.task_id)
        fingerprint = hashlib.sha256(
            f"{manifest_sha}:{'|'.join(dependencies)}".encode()
        ).hexdigest()
        task = TaskSpec(
            task_id=task_id,
            project_id=project_id,
            kind=TaskKind.H3_GENERATION,
            state=TaskState.BLOCKED,
            idempotency_key=hashlib.sha256(f"{task_id}:{fingerprint}".encode()).hexdigest(),
            input_fingerprint=fingerprint,
            workload_manifest_sha256=manifest_sha,
            depends_on=tuple(dependencies),
            affinity_key=h3_affinity,
            priority=100 - min(99, int(segment.get("segmentIndex") or 0) * 10 + stable_index),
        )
        store.add_task(task)
        version = SegmentGenerationVersion(
            version_id=f"segment-version:{runtime_id}",
            project_id=project_id,
            shot_id=str(segment.get("shotId") or source_version.shot_id),
            segment_id=segment_id,
            runtime_segment_id=runtime_id,
            segment_index=int(segment.get("segmentIndex") or 0),
            generation_number=source_version.generation_number + 1,
            batch_id=batch_id,
            task_id=task_id,
            parent_version_id=source_version.version_id,
            predecessor_version_id=predecessor.version_id if predecessor else None,
            prompt_revision_id=str(segment.get("id") or "") or None,
            seed=replacement_seed if replacement_seed is not None else source_version.seed,
            created_at=now,
        )
        store.put_segment_generation_version(version)
        new_versions[segment_id] = version
        new_tasks.append(task)
    batch = GenerationBatch(
        batch_id=batch_id,
        project_id=project_id,
        kind=GenerationBatchKind.REWORK,
        generation_number=latest,
        segment_ids=tuple(str(item["segmentId"]) for item in segments),
        task_ids=tuple(item.task_id for item in new_tasks),
        encoding_task_ids=tuple(sorted(encoding_ids)),
        model_switch_task_id=switch_id,
        dispatch_requested=True,
        created_at=now,
        updated_at=now,
    )
    store.put_generation_batch(batch)
    for marker in markers:
        store.put_rework_marker(
            marker.model_copy(
                update={
                    "state": ReworkMarkerState.PREPARING,
                    "batch_id": batch_id,
                    "updated_at": now,
                }
            )
        )
    return batch


def reconcile_generation_batches(store: SQLiteTaskStore, *, batch_id: str | None = None) -> None:
    for batch in store.list_generation_batches_all():
        if batch_id is not None and batch.batch_id != batch_id:
            continue
        now = datetime.now(UTC)
        referenced_task_ids = {
            *batch.encoding_task_ids,
            *batch.task_ids,
            *((batch.model_switch_task_id,) if batch.model_switch_task_id else ()),
        }
        try:
            for task_id in referenced_task_ids:
                store.get_task(task_id)
        except TaskNotFoundError:
            store.put_generation_batch(
                batch.model_copy(
                    update={
                        "state": GenerationBatchState.CANCELLED,
                        "dispatch_requested": False,
                        "updated_at": now,
                    }
                )
            )
            continue
        switch = store.get_task(batch.model_switch_task_id) if batch.model_switch_task_id else None
        state = batch.state
        if (
            switch
            and switch.state in {TaskState.QUEUED, TaskState.RUNNING, TaskState.SUCCEEDED}
            and state == GenerationBatchState.PREPARING
        ):
            state = (
                GenerationBatchState.RUNNING
                if switch.state == TaskState.SUCCEEDED
                else GenerationBatchState.SEALED
            )
            for marker in store.list_rework_markers(batch.project_id):
                if (
                    marker.batch_id == batch.batch_id
                    and marker.state == ReworkMarkerState.PREPARING
                ):
                    store.put_rework_marker(
                        marker.model_copy(
                            update={"state": ReworkMarkerState.SEALED, "updated_at": now}
                        )
                    )
        versions = {
            item.task_id: item
            for item in store.list_segment_generation_versions(batch.project_id)
            if item.batch_id == batch.batch_id
        }
        open_markers = store.list_rework_markers(batch.project_id)
        frozen = frozen_segment_ids(
            motion_context_segments(store.get_latest_project_workspace(batch.project_id).payload),
            (marker for marker in open_markers if marker.batch_id != batch.batch_id),
        )
        for task_id, version in versions.items():
            task = store.get_task(task_id)
            if task.state == TaskState.RUNNING and version.state == SegmentVersionState.PLANNED:
                store.put_segment_generation_version(
                    version.model_copy(update={"state": SegmentVersionState.RUNNING})
                )
            if task.state != TaskState.SUCCEEDED or version.state in {
                SegmentVersionState.ACTIVE,
                SegmentVersionState.DISCARDED,
            }:
                continue
            predecessor_ok = True
            if version.predecessor_version_id:
                predecessor_ok = any(
                    item.version_id == version.predecessor_version_id
                    and item.state == SegmentVersionState.ACTIVE
                    for item in store.list_segment_generation_versions(batch.project_id)
                )
            discard = version.segment_id in frozen or not predecessor_ok
            store.put_segment_generation_version(
                version.model_copy(
                    update={
                        "state": (
                            SegmentVersionState.DISCARDED if discard else SegmentVersionState.ACTIVE
                        ),
                        "artifact_id": f"{task_id}:video",
                        "discard_reason": (
                            "sequence_frozen"
                            if version.segment_id in frozen
                            else "predecessor_version_inactive"
                            if not predecessor_ok
                            else None
                        ),
                        "activated_at": None if discard else now,
                    }
                )
            )
        terminal = {
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.STALE,
        }
        tasks = [store.get_task(value) for value in batch.task_ids]
        if tasks and all(task.state in terminal for task in tasks):
            state = GenerationBatchState.COMPLETED
            for marker in store.list_rework_markers(batch.project_id):
                if marker.batch_id == batch.batch_id and marker.state == ReworkMarkerState.SEALED:
                    replacement = next(
                        (
                            item
                            for item in versions.values()
                            if item.segment_id == marker.segment_id
                        ),
                        None,
                    )
                    if (
                        replacement
                        and store.get_task(replacement.task_id).state == TaskState.SUCCEEDED
                    ):
                        store.put_rework_marker(
                            marker.model_copy(
                                update={"state": ReworkMarkerState.RESOLVED, "updated_at": now}
                            )
                        )
        if state != batch.state:
            store.put_generation_batch(
                batch.model_copy(
                    update={
                        "state": state,
                        "sealed_at": now
                        if state == GenerationBatchState.SEALED
                        else batch.sealed_at,
                        "updated_at": now,
                    }
                )
            )
