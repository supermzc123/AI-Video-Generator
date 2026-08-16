from ai_video_generator.domain import (
    ChainSpec,
    ConditioningStack,
    ContextMode,
    GenerationMode,
    GenerationSegment,
    IncomingContext,
    TaskType,
)
from ai_video_generator.workers import (
    PINNED_MOTION_CONTEXT_COMMIT,
    PINNED_MOTION_CONTEXT_PROFILE,
    compile_dry_run,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def make_stack(engine_commit: str = PINNED_MOTION_CONTEXT_COMMIT) -> ConditioningStack:
    return ConditioningStack(
        text_encoder_sha256=HASH_A,
        h3_model_sha256=HASH_B,
        video_vae_sha256=HASH_C,
        audio_vae_sha256=HASH_D,
        comfyui_commit="344b43989e8c56b5bb4a66cf028c834192ab59dd",
        worker_engine_commit=engine_commit,
    )


def make_segment(
    segment_id: str,
    ordinal: int,
    prompt: str,
    incoming_context: IncomingContext | None = None,
) -> GenerationSegment:
    return GenerationSegment(
        segment_id=segment_id,
        ordinal=ordinal,
        prompt_revision_id=f"prompt-{ordinal}@1",
        normalized_prompt=prompt,
        generation_mode=GenerationMode.REF2VA,
        width=1024,
        height=608,
        sample_frames=277,
        visible_frames=255 if incoming_context else 277,
        incoming_context=incoming_context or IncomingContext(),
    )


def make_chain(second_prompt: str = "Second shot") -> ChainSpec:
    return ChainSpec(
        project_id="project-1",
        run_id="run-1",
        shot_revision_id="shot-1@1",
        segments=(
            make_segment("S001.C01", 1, "First shot"),
            make_segment(
                "S001.C02",
                2,
                second_prompt,
                IncomingContext(
                    mode=ContextMode.MOTION_CONTEXT,
                    predecessor_segment_id="S001.C01",
                    context_frames=22,
                    trim_head_frames=22,
                ),
            ),
        ),
    )


def test_dry_run_places_all_encoding_before_diffusion_load() -> None:
    plan = compile_dry_run(
        make_chain(),
        make_stack(),
        motion_context_profile=PINNED_MOTION_CONTEXT_PROFILE,
    )

    encode_tasks = [task for task in plan.tasks if task.task_type == TaskType.ENCODE_CONDITIONING]
    unload = next(task for task in plan.tasks if task.task_type == TaskType.UNLOAD_CONDITIONING)
    load = next(task for task in plan.tasks if task.task_type == TaskType.LOAD_DIFFUSION)
    generations = [task for task in plan.tasks if task.task_type == TaskType.GENERATE_SEGMENT]

    assert len(encode_tasks) == 2
    assert unload.depends_on == tuple(task.task_id for task in encode_tasks)
    assert load.depends_on == (unload.task_id,)
    assert generations[0].task_id in generations[1].depends_on
    assert not plan.ready_for_submission
    assert "execution endpoint is not implemented" in plan.blockers[0]


def test_identical_static_conditioning_is_encoded_once() -> None:
    chain = make_chain(second_prompt="First shot")

    plan = compile_dry_run(
        chain,
        make_stack(),
        motion_context_profile=PINNED_MOTION_CONTEXT_PROFILE,
    )

    encode_tasks = [task for task in plan.tasks if task.task_type == TaskType.ENCODE_CONDITIONING]
    assert len(encode_tasks) == 1
    assert encode_tasks[0].segment_ids == ("S001.C01", "S001.C02")
    assert plan.warnings == ("1 segment conditioning job(s) were deduplicated by fingerprint",)


def test_missing_motion_context_provider_is_a_blocker() -> None:
    plan = compile_dry_run(
        make_chain(),
        make_stack(engine_commit="f" * 40),
        motion_context_profile=None,
    )

    assert any("not installed" in blocker for blocker in plan.blockers)
    assert not any("provider commit" in blocker for blocker in plan.blockers)


def test_provider_rejects_non_native_fps_context_length_and_split_av_handoff() -> None:
    invalid = GenerationSegment(
        segment_id="S001.C02",
        ordinal=2,
        prompt_revision_id="prompt-2@1",
        normalized_prompt="Second shot",
        generation_mode=GenerationMode.REF2VA,
        fps=30,
        width=1024,
        height=608,
        sample_frames=277,
        visible_frames=260,
        incoming_context=IncomingContext(
            mode=ContextMode.MOTION_CONTEXT,
            predecessor_segment_id="S001.C01",
            visual=True,
            audio=False,
            context_frames=17,
            trim_head_frames=17,
        ),
    )
    chain = ChainSpec(
        project_id="project-1",
        run_id="run-1",
        shot_revision_id="shot-1@1",
        segments=(make_segment("S001.C01", 1, "First shot"), invalid),
    )

    plan = compile_dry_run(
        chain,
        make_stack(),
        motion_context_profile=PINNED_MOTION_CONTEXT_PROFILE,
    )

    assert any("requires 24fps" in blocker for blocker in plan.blockers)
    assert any(
        "context_length must be one of 5, 22, 39, 56" in blocker for blocker in plan.blockers
    )
    assert any("both visual and audio" in blocker for blocker in plan.blockers)


def test_provider_commit_mismatch_is_a_blocker() -> None:
    plan = compile_dry_run(
        make_chain(),
        make_stack(engine_commit="f" * 40),
        motion_context_profile=PINNED_MOTION_CONTEXT_PROFILE,
    )

    assert any("provider commit" in blocker for blocker in plan.blockers)
