from collections import OrderedDict

from ai_video_generator.domain import (
    ChainSpec,
    ConditioningArtifact,
    ConditioningStack,
    ContextMode,
    DryRunPlan,
    GenerationMode,
    ModelResidency,
    PlannedTask,
    TaskType,
)
from ai_video_generator.services import conditioning_fingerprint

PINNED_MOTION_DIRECTOR_COMMIT = "a58f28271a9db8af2a802533a1078b5890c9c4b9"
SUPPORTED_MODES = {
    GenerationMode.T2VA,
    GenerationMode.I2VA,
    GenerationMode.FL2VA,
    GenerationMode.REF2VA,
    GenerationMode.V2VA,
    GenerationMode.RV2VA,
}


def compile_dry_run(
    chain: ChainSpec,
    stack: ConditioningStack,
    *,
    motion_director_installed: bool,
) -> DryRunPlan:
    fingerprints_by_segment = {
        segment.segment_id: conditioning_fingerprint(segment, stack)
        for segment in chain.segments
    }
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for segment in chain.segments:
        fingerprint = fingerprints_by_segment[segment.segment_id]
        grouped.setdefault(fingerprint, []).append(segment.segment_id)

    artifacts = tuple(
        ConditioningArtifact(
            artifact_id=f"conditioning:{fingerprint}",
            fingerprint=fingerprint,
            tensor_path=f"conditioning/blobs/{fingerprint}.safetensors",
            manifest_path=f"conditioning/manifests/{fingerprint}.json",
            segment_ids=tuple(segment_ids),
        )
        for fingerprint, segment_ids in grouped.items()
    )

    tasks: list[PlannedTask] = []
    encode_task_by_fingerprint: dict[str, str] = {}
    for artifact in artifacts:
        task_id = f"encode:{artifact.fingerprint}"
        encode_task_by_fingerprint[artifact.fingerprint] = task_id
        tasks.append(
            PlannedTask(
                task_id=task_id,
                task_type=TaskType.ENCODE_CONDITIONING,
                segment_ids=artifact.segment_ids,
                conditioning_fingerprint=artifact.fingerprint,
                required_residency=ModelResidency.CONDITIONING,
                resulting_residency=ModelResidency.CONDITIONING,
            )
        )

    unload_task_id = "models:unload-conditioning"
    tasks.append(
        PlannedTask(
            task_id=unload_task_id,
            task_type=TaskType.UNLOAD_CONDITIONING,
            depends_on=tuple(encode_task_by_fingerprint.values()),
            required_residency=ModelResidency.CONDITIONING,
            resulting_residency=ModelResidency.UNLOADED,
        )
    )
    load_task_id = "models:load-diffusion"
    tasks.append(
        PlannedTask(
            task_id=load_task_id,
            task_type=TaskType.LOAD_DIFFUSION,
            depends_on=(unload_task_id,),
            required_residency=ModelResidency.UNLOADED,
            resulting_residency=ModelResidency.DIFFUSION,
        )
    )

    generation_task_by_segment: dict[str, str] = {}
    for segment in chain.segments:
        fingerprint = fingerprints_by_segment[segment.segment_id]
        dependencies = [load_task_id, encode_task_by_fingerprint[fingerprint]]
        incoming = segment.incoming_context
        if incoming.mode == ContextMode.MOTION_CONTEXT:
            predecessor_task = generation_task_by_segment[incoming.predecessor_segment_id]
            dependencies.append(predecessor_task)

        task_id = f"generate:{segment.segment_id}"
        generation_task_by_segment[segment.segment_id] = task_id
        tasks.append(
            PlannedTask(
                task_id=task_id,
                task_type=TaskType.GENERATE_SEGMENT,
                segment_ids=(segment.segment_id,),
                conditioning_fingerprint=fingerprint,
                depends_on=tuple(dependencies),
                required_residency=ModelResidency.DIFFUSION,
                resulting_residency=ModelResidency.DIFFUSION,
            )
        )

    warnings: list[str] = []
    blockers: list[str] = [
        "Dry-run plans cannot be submitted; the Worker execution endpoint is not implemented"
    ]
    if not motion_director_installed:
        blockers.append("Motion Director is not installed in the configured ComfyUI Worker")
    if stack.worker_engine_commit != PINNED_MOTION_DIRECTOR_COMMIT:
        blockers.append(
            "Conditioning stack does not match the pinned Motion Director commit "
            f"{PINNED_MOTION_DIRECTOR_COMMIT}"
        )

    unsupported = sorted(
        {
            segment.generation_mode.value
            for segment in chain.segments
            if segment.generation_mode not in SUPPORTED_MODES
        }
    )
    if unsupported:
        blockers.append(
            "Motion Director adapter mapping is not defined for modes: " + ", ".join(unsupported)
        )

    duplicate_count = len(chain.segments) - len(artifacts)
    if duplicate_count:
        warnings.append(
            f"{duplicate_count} segment conditioning job(s) were deduplicated by fingerprint"
        )

    return DryRunPlan(
        project_id=chain.project_id,
        run_id=chain.run_id,
        shot_revision_id=chain.shot_revision_id,
        engine_commit=PINNED_MOTION_DIRECTOR_COMMIT,
        conditioning_artifacts=artifacts,
        tasks=tuple(tasks),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
    )
