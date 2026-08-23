import pytest

from ai_video_generator.domain import (
    ConditioningArtifact,
    DryRunPlan,
    ModelResidency,
    PlannedTask,
    TaskType,
)

FINGERPRINT = "a" * 64


def valid_payload() -> dict[str, object]:
    encode = PlannedTask(
        task_id="encode:1",
        task_type=TaskType.ENCODE_CONDITIONING,
        conditioning_fingerprint=FINGERPRINT,
        required_residency=ModelResidency.CONDITIONING,
        resulting_residency=ModelResidency.CONDITIONING,
    )
    unload = PlannedTask(
        task_id="unload",
        task_type=TaskType.UNLOAD_CONDITIONING,
        depends_on=(encode.task_id,),
        required_residency=ModelResidency.CONDITIONING,
        resulting_residency=ModelResidency.UNLOADED,
    )
    load = PlannedTask(
        task_id="load",
        task_type=TaskType.LOAD_DIFFUSION,
        depends_on=(unload.task_id,),
        required_residency=ModelResidency.UNLOADED,
        resulting_residency=ModelResidency.DIFFUSION,
    )
    generation = PlannedTask(
        task_id="generate:1",
        task_type=TaskType.GENERATE_SEGMENT,
        conditioning_fingerprint=FINGERPRINT,
        depends_on=(load.task_id, encode.task_id),
        required_residency=ModelResidency.DIFFUSION,
        resulting_residency=ModelResidency.DIFFUSION,
    )
    plan = DryRunPlan(
        project_id="project-1",
        run_id="run-1",
        shot_revision_id="shot-1@1",
        engine_commit="b" * 40,
        conditioning_artifacts=(
            ConditioningArtifact(
                artifact_id="conditioning:1",
                fingerprint=FINGERPRINT,
                tensor_path="conditioning/1.safetensors",
                manifest_path="conditioning/1.json",
                segment_ids=("segment-1",),
            ),
        ),
        tasks=(encode, unload, load, generation),
        blockers=("dry run",),
    )
    return plan.model_dump()


def test_plan_contract_rejects_incomplete_encode_barrier() -> None:
    payload = valid_payload()
    payload["tasks"][1]["depends_on"] = ()

    with pytest.raises(ValueError, match="every encoding task"):
        DryRunPlan.model_validate(payload)


def test_plan_contract_rejects_generation_that_skips_diffusion_load() -> None:
    payload = valid_payload()
    payload["tasks"][3]["depends_on"] = ("encode:1",)

    with pytest.raises(ValueError, match="must depend on diffusion load"):
        DryRunPlan.model_validate(payload)
