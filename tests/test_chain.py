import pytest
from pydantic import ValidationError

from ai_video_generator.domain import (
    ChainSpec,
    ContextMode,
    GenerationMode,
    GenerationSegment,
    IncomingContext,
)


def segment(
    segment_id: str,
    ordinal: int,
    incoming_context: IncomingContext | None = None,
) -> GenerationSegment:
    return GenerationSegment(
        segment_id=segment_id,
        ordinal=ordinal,
        prompt_revision_id=f"prompt-{ordinal}@1",
        normalized_prompt="A deliberate tracking shot",
        generation_mode=GenerationMode.REF2VA,
        width=1024,
        height=608,
        sample_frames=277,
        visible_frames=255 if incoming_context is not None else 277,
        incoming_context=incoming_context or IncomingContext(),
    )


def test_chain_accepts_adjacent_motion_context() -> None:
    first = segment("S001.C01", 1)
    second = segment(
        "S001.C02",
        2,
        IncomingContext(
            mode=ContextMode.MOTION_CONTEXT,
            predecessor_segment_id="S001.C01",
            context_frames=22,
            trim_head_frames=22,
        ),
    )

    chain = ChainSpec(
        project_id="project-1",
        run_id="run-1",
        shot_revision_id="shot-1@1",
        segments=(first, second),
    )

    assert chain.segments[1].incoming_context.predecessor_segment_id == "S001.C01"


def test_segment_rejects_more_than_fifteen_seconds() -> None:
    with pytest.raises(ValidationError, match="15 seconds"):
        GenerationSegment(
            segment_id="S001.C01",
            ordinal=1,
            prompt_revision_id="prompt-1@1",
            normalized_prompt="A shot",
            generation_mode=GenerationMode.T2VA,
            width=1024,
            height=608,
            sample_frames=362,
            visible_frames=362,
        )


def test_chain_rejects_non_adjacent_context() -> None:
    with pytest.raises(ValidationError, match="immediately preceding"):
        ChainSpec(
            project_id="project-1",
            run_id="run-1",
            shot_revision_id="shot-1@1",
            segments=(
                segment("S001.C01", 1),
                segment("S001.C02", 2),
                segment(
                    "S001.C03",
                    3,
                    IncomingContext(
                        mode=ContextMode.MOTION_CONTEXT,
                        predecessor_segment_id="S001.C01",
                        context_frames=22,
                        trim_head_frames=22,
                    ),
                ),
            ),
        )


def test_segment_rejects_motion_context_trim_that_does_not_match_context() -> None:
    with pytest.raises(ValidationError, match="must equal context_frames"):
        segment(
            "S001.C02",
            2,
            IncomingContext(
                mode=ContextMode.MOTION_CONTEXT,
                predecessor_segment_id="S001.C01",
                context_frames=22,
                trim_head_frames=5,
            ),
        )


def test_segment_rejects_unexplained_full_h3_grid_of_sampled_tail() -> None:
    with pytest.raises(ValidationError, match="smaller than one H3 grid interval"):
        GenerationSegment(
            segment_id="S001.C01",
            ordinal=1,
            prompt_revision_id="prompt-1@1",
            normalized_prompt="A shot",
            generation_mode=GenerationMode.T2VA,
            width=1024,
            height=608,
            sample_frames=277,
            visible_frames=260,
        )
