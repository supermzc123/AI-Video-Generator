from __future__ import annotations

import math

from pydantic import Field, model_validator

from ai_video_generator.domain import (
    AssetBinding,
    ChainSpec,
    ContextMode,
    GenerationMode,
    GenerationSegment,
    IncomingContext,
    ProjectSpec,
    ShotSpec,
)
from ai_video_generator.domain.chain import FrozenModel


class ShotCompilationRequest(FrozenModel):
    project: ProjectSpec
    shot: ShotSpec
    run_id: str = Field(min_length=1)
    generation_mode: GenerationMode = GenerationMode.REF2VA
    prompts: tuple[str, ...] = ()
    prompt_revision_ids: tuple[str, ...] = ()
    common_assets: tuple[AssetBinding, ...] = ()
    local_assets: tuple[AssetBinding, ...] = ()
    preferred_segment_seconds: float = Field(default=12, gt=0, le=15)
    # 56 is the first supported Motion Context window that reserves at least
    # two seconds at H3's native 24 fps (2.33 s). The inherited head is part of
    # the model's 15-second sampling budget and is trimmed by the ComfyUI node.
    context_frames: int = Field(default=56, ge=0)
    audio_context_video_frames: int = Field(default=24, ge=0)
    inherit_visual: bool = True
    inherit_audio: bool = True

    @model_validator(mode="after")
    def validate_request(self) -> ShotCompilationRequest:
        if self.shot.project_id != self.project.project_id:
            raise ValueError("shot and project IDs must match")
        if self.prompts and any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("segment prompts must not be blank")
        if self.prompt_revision_ids and len(self.prompt_revision_ids) != len(self.prompts):
            raise ValueError("prompt revision IDs must align with prompts")
        if not (self.inherit_visual or self.inherit_audio):
            raise ValueError("continuation must inherit visual or audio context")
        if self.inherit_visual and self.context_frames == 0:
            raise ValueError("visual continuation requires context_frames")
        if self.inherit_audio and self.audio_context_video_frames == 0:
            raise ValueError("audio continuation requires an audio context window")
        return self


def compile_shot(request: ShotCompilationRequest) -> ChainSpec:
    fps = request.project.fps
    target_frames = max(1, round(request.shot.target_duration_seconds * fps))
    max_sample_frames = _floor_h3_grid(fps * 15)
    preferred_visible_frames = max(1, round(request.preferred_segment_seconds * fps))

    if target_frames <= max_sample_frames:
        visible_chunks = [target_frames]
    else:
        visible_chunks = []
        remaining = target_frames
        ordinal = 1
        while remaining:
            trim_head = 0 if ordinal == 1 else request.context_frames
            max_visible = max_sample_frames - trim_head
            chunk = min(remaining, preferred_visible_frames, max_visible)
            visible_chunks.append(chunk)
            remaining -= chunk
            ordinal += 1

    if len(request.prompts) not in (0, 1, len(visible_chunks)):
        raise ValueError("provide zero, one, or exactly one prompt per compiled segment")

    segments: list[GenerationSegment] = []
    for index, visible_frames in enumerate(visible_chunks):
        ordinal = index + 1
        segment_id = f"{request.shot.shot_revision_id}.C{ordinal:02d}"
        trim_head = 0 if index == 0 else request.context_frames
        sample_frames = _ceil_h3_grid(visible_frames + trim_head)
        if sample_frames > max_sample_frames:
            raise ValueError("compiled segment exceeds the H3 15-second frame budget")

        if len(request.prompts) == len(visible_chunks):
            prompt = request.prompts[index]
        elif request.prompts:
            prompt = request.prompts[0]
        else:
            prompt = request.shot.description
        prompt_revision_id = (
            request.prompt_revision_ids[index]
            if len(request.prompt_revision_ids) == len(visible_chunks)
            else (
                request.prompt_revision_ids[0]
                if request.prompt_revision_ids
                else f"{request.shot.shot_revision_id}.prompt.{ordinal}@1"
            )
        )
        incoming = IncomingContext()
        if index:
            incoming = IncomingContext(
                mode=ContextMode.MOTION_CONTEXT,
                predecessor_segment_id=segments[-1].segment_id,
                visual=request.inherit_visual,
                audio=request.inherit_audio,
                context_frames=request.context_frames if request.inherit_visual else 0,
                audio_context_video_frames=(
                    request.audio_context_video_frames if request.inherit_audio else 0
                ),
                trim_head_frames=trim_head,
            )
        segments.append(
            GenerationSegment(
                segment_id=segment_id,
                ordinal=ordinal,
                prompt_revision_id=prompt_revision_id,
                normalized_prompt=" ".join(prompt.split()),
                generation_mode=request.generation_mode,
                fps=fps,
                width=request.project.width,
                height=request.project.height,
                sample_frames=sample_frames,
                visible_frames=visible_frames,
                common_assets=request.common_assets,
                local_assets=request.local_assets,
                incoming_context=incoming,
            )
        )

    return ChainSpec(
        project_id=request.project.project_id,
        run_id=request.run_id,
        shot_revision_id=request.shot.shot_revision_id,
        segments=tuple(segments),
    )


def _ceil_h3_grid(frames: int) -> int:
    return max(5, math.ceil((frames - 5) / 17) * 17 + 5)


def _floor_h3_grid(frames: int) -> int:
    return max(5, math.floor((frames - 5) / 17) * 17 + 5)
