from ai_video_generator.domain import ContextMode, ProjectSpec, ShotSpec
from ai_video_generator.services.shot_compiler import ShotCompilationRequest, compile_shot


def make_request(duration: float) -> ShotCompilationRequest:
    project = ProjectSpec(
        project_id="project-1",
        name="Project",
        target_duration_seconds=60,
        fps=24,
    )
    shot = ShotSpec(
        shot_revision_id="shot-1@1",
        project_id="project-1",
        ordinal=1,
        title="Long take",
        description="A continuous tracking shot",
        target_duration_seconds=duration,
    )
    return ShotCompilationRequest(project=project, shot=shot, run_id="run-1")


def test_fourteen_second_shot_stays_one_segment() -> None:
    chain = compile_shot(make_request(14))

    assert len(chain.segments) == 1
    assert chain.segments[0].visible_frames == 14 * 24
    assert chain.segments[0].sample_frames <= 15 * 24


def test_thirty_one_second_shot_has_exact_visible_duration_and_context() -> None:
    chain = compile_shot(make_request(31))

    assert len(chain.segments) == 3
    assert sum(segment.visible_frames for segment in chain.segments) == 31 * 24
    assert all(segment.sample_frames <= 15 * 24 for segment in chain.segments)
    assert all((segment.sample_frames - 5) % 17 == 0 for segment in chain.segments)
    assert chain.segments[1].incoming_context.mode == ContextMode.MOTION_CONTEXT
    assert chain.segments[1].incoming_context.trim_head_frames == 56
    assert chain.segments[1].incoming_context.trim_head_frames / 24 >= 2
    assert chain.segments[2].incoming_context.predecessor_segment_id == chain.segments[1].segment_id
