from pathlib import Path

import httpx
import pytest

from ai_video_generator.api import create_app
from ai_video_generator.config import Settings
from ai_video_generator.domain import (
    ChainSpec,
    ConditioningStack,
    GenerationMode,
    GenerationSegment,
)
from ai_video_generator.workers import PINNED_MOTION_DIRECTOR_COMMIT


@pytest.mark.asyncio
async def test_health_endpoint_has_no_worker_side_effects() -> None:
    app = create_app(Settings(_env_file=None, comfyui_root=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "build_stage": "p0-dry-run",
    }


@pytest.mark.asyncio
async def test_dry_run_endpoint_never_returns_a_submittable_plan(tmp_path: Path) -> None:
    (tmp_path / "custom_nodes" / "ComfyUI-MiniMax-H3-Motion-Director").mkdir(parents=True)
    app = create_app(Settings(_env_file=None, comfyui_root=tmp_path))
    segment = GenerationSegment(
        segment_id="S001.C01",
        ordinal=1,
        prompt_revision_id="prompt-1@1",
        normalized_prompt="A static dry run",
        generation_mode=GenerationMode.REF2VA,
        width=1024,
        height=608,
        sample_frames=277,
        visible_frames=277,
    )
    chain = ChainSpec(
        project_id="project-1",
        run_id="run-1",
        shot_revision_id="shot-1@1",
        segments=(segment,),
    )
    stack = ConditioningStack(
        text_encoder_sha256="a" * 64,
        h3_model_sha256="b" * 64,
        video_vae_sha256="c" * 64,
        audio_vae_sha256="d" * 64,
        comfyui_commit="344b43989e8c56b5bb4a66cf028c834192ab59dd",
        worker_engine_commit=PINNED_MOTION_DIRECTOR_COMMIT,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/plans/dry-run",
            json={
                "chain": chain.model_dump(mode="json"),
                "conditioning_stack": stack.model_dump(mode="json"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ready_for_submission"] is False
    assert body["tasks"][-1]["task_type"] == "generate_segment"
