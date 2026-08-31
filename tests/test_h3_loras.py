from datetime import UTC, datetime

import pytest

from ai_video_generator.domain import ExecutionMode, ProjectRunState, ReviewPolicy
from ai_video_generator.services.h3_loras import (
    diffusion_affinity,
    model_stack_sha256,
    normalize_project_loras,
)
from ai_video_generator.services.project_tasks import compile_project_task_plan


def test_normalize_project_loras_preserves_order_and_ignores_disabled() -> None:
    assert normalize_project_loras(
        {
            "h3Loras": [
                {"name": "a.safetensors", "strength": 0.7, "enabled": True},
                {"name": "off.safetensors", "strength": 1, "enabled": False},
                {"name": "b.safetensors", "strength": 1.1, "enabled": True},
                {"name": "a.safetensors", "strength": 2, "enabled": True},
            ]
        }
    ) == (
        {"name": "a.safetensors", "strength": 0.7},
        {"name": "b.safetensors", "strength": 1.1},
    )


def test_project_lora_stack_changes_diffusion_affinity() -> None:
    base = {"diffusion_model": "h3.safetensors", "project_loras": []}
    configured = {
        "diffusion_model": "h3.safetensors",
        "project_loras": [{"name": "character.safetensors", "strength": 1.0}],
    }
    base["model_stack_sha256"] = model_stack_sha256(base)
    configured["model_stack_sha256"] = model_stack_sha256(configured)
    assert diffusion_affinity(base) != diffusion_affinity(configured)


def test_invalid_project_lora_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="strength"):
        normalize_project_loras(
            {"h3Loras": [{"name": "bad.safetensors", "strength": 5, "enabled": True}]}
        )


def test_project_task_affinity_is_isolated_by_model_stack() -> None:
    payload = {
        "width": 320,
        "height": 480,
        "prompts": {
            "imagePrompts": [],
            "h3Prompts": [{
                "segmentId": "segment-1",
                "segmentIndex": 0,
                "durationSeconds": 4,
                "prompt": "A cinematic shot",
                "continuationOf": None,
            }],
        },
    }
    run_state = ProjectRunState(
        project_id="project",
        execution_mode=ExecutionMode.GUIDED,
        review_policy=ReviewPolicy(),
        updated_at=datetime.now(UTC),
    )
    profiles = []
    for name in ("a.safetensors", "b.safetensors"):
        profile = {
            "diffusion_model": "h3.safetensors",
            "project_loras": [{"name": name, "strength": 1.0}],
        }
        profile["model_stack_sha256"] = model_stack_sha256(profile)
        profiles.append(profile)
    plans = [
        compile_project_task_plan(
            project_id="project",
            workspace_revision=1,
            payload=payload,
            run_state=run_state,
            approved_image_workflows=set(),
            approved_image_harnesses={},
            h3_execution_profile=profile,
        )
        for profile in profiles
    ]
    affinities = [
        next(task.affinity_key for task in plan.tasks if task.kind.value == "h3_generation")
        for plan in plans
    ]
    assert affinities[0] != affinities[1]
    assert plans[0].fingerprint != plans[1].fingerprint
