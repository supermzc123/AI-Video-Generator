from collections import OrderedDict

from ai_video_generator.domain import (
    ChainSpec,
    ConditioningArtifact,
    ConditioningStack,
    ContextMode,
    DryRunPlan,
    GenerationMode,
    GenerationSegment,
    ModelResidency,
    MotionContextHandoff,
    MotionContextProfile,
    PlannedTask,
    TaskType,
)
from ai_video_generator.services import conditioning_fingerprint

MOTION_CONTEXT_REPOSITORY = "https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git"
PINNED_MOTION_CONTEXT_COMMIT = "725a731e644c669601799da1eb63f4e7497c628f"
MOTION_CONTEXT_NODE_TYPES = (
    "MiniMaxH3MotionContext",
    "MiniMaxH3MotionContextLoadLatent",
    "MiniMaxH3MotionContextSaveLatent",
    "MiniMaxH3MotionContextSeamProbe",
    "MiniMaxH3MotionContextTrim",
)
PINNED_MOTION_CONTEXT_PROFILE = MotionContextProfile(
    provider_id="niko-h3-motion-context-v0.3.1",
    source_repository=MOTION_CONTEXT_REPOSITORY,
    source_commit=PINNED_MOTION_CONTEXT_COMMIT,
    node_types=MOTION_CONTEXT_NODE_TYPES,
    native_fps=24,
    context_lengths=(5, 22, 39, 56),
    handoff=MotionContextHandoff.AV_LATENT,
)
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
    motion_context_profile: MotionContextProfile | None,
) -> DryRunPlan:
    fingerprints_by_segment = {
        segment.segment_id: conditioning_fingerprint(segment, stack) for segment in chain.segments
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
    motion_segments = tuple(
        segment
        for segment in chain.segments
        if segment.incoming_context.mode == ContextMode.MOTION_CONTEXT
    )
    if motion_segments:
        if motion_context_profile is None:
            blockers.append(
                "A verified Motion Context provider is not installed in the configured "
                "ComfyUI Worker"
            )
        else:
            blockers.extend(
                _validate_motion_context_segments(motion_segments, motion_context_profile)
            )
            if stack.worker_engine_commit != motion_context_profile.source_commit:
                blockers.append(
                    "Conditioning stack does not match Motion Context provider commit "
                    f"{motion_context_profile.source_commit}"
                )

    unsupported = sorted(
        {
            segment.generation_mode.value
            for segment in motion_segments
            if segment.generation_mode not in SUPPORTED_MODES
        }
    )
    if unsupported:
        blockers.append(
            "Motion Context adapter mapping is not defined for modes: " + ", ".join(unsupported)
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
        engine_commit=(
            motion_context_profile.source_commit
            if motion_context_profile is not None
            else PINNED_MOTION_CONTEXT_COMMIT
        ),
        conditioning_artifacts=artifacts,
        tasks=tuple(tasks),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
    )


def _validate_motion_context_segments(
    segments: tuple[GenerationSegment, ...],
    profile: MotionContextProfile,
) -> list[str]:
    blockers: list[str] = []
    for segment in segments:
        incoming = segment.incoming_context
        prefix = f"Segment {segment.segment_id}"
        if segment.fps != profile.native_fps:
            blockers.append(
                f"{prefix} uses {segment.fps}fps; {profile.provider_id} requires "
                f"{profile.native_fps}fps"
            )
        if incoming.context_frames not in profile.context_lengths:
            allowed = ", ".join(str(value) for value in profile.context_lengths)
            blockers.append(
                f"{prefix} context_length must be one of {allowed}; got {incoming.context_frames}"
            )
        if profile.handoff == MotionContextHandoff.AV_LATENT and not (
            incoming.visual and incoming.audio
        ):
            blockers.append(
                f"{prefix} must bind both visual and audio context for AV latent handoff"
            )
        if incoming.trim_head_frames != incoming.context_frames:
            blockers.append(f"{prefix} trim_head_frames must equal the bound context_length")
        unused = segment.sample_frames - incoming.trim_head_frames - segment.visible_frames
        if unused < 0 or unused >= 17:
            blockers.append(
                f"{prefix} sample/visible/trim values are inconsistent with the H3 frame grid"
            )
    return blockers
