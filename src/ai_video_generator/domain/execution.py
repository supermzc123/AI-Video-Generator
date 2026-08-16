from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .artifacts import ConditioningArtifact
from .chain import FrozenModel


class ModelResidency(StrEnum):
    UNLOADED = "unloaded"
    CONDITIONING = "conditioning"
    DIFFUSION = "diffusion"


class TaskType(StrEnum):
    ENCODE_CONDITIONING = "encode_conditioning"
    UNLOAD_CONDITIONING = "unload_conditioning"
    LOAD_DIFFUSION = "load_diffusion"
    GENERATE_SEGMENT = "generate_segment"


class PlannedTask(FrozenModel):
    task_id: str = Field(min_length=1)
    task_type: TaskType
    segment_ids: tuple[str, ...] = ()
    conditioning_fingerprint: str | None = None
    depends_on: tuple[str, ...] = ()
    required_residency: ModelResidency
    resulting_residency: ModelResidency


class DryRunPlan(FrozenModel):
    schema_version: str = "1.0"
    project_id: str
    run_id: str
    shot_revision_id: str
    engine: str = "h3-motion-context"
    engine_commit: str
    ready_for_submission: Literal[False] = False
    conditioning_artifacts: tuple[ConditioningArtifact, ...]
    tasks: tuple[PlannedTask, ...]
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_conditioning_phase_barrier(self) -> "DryRunPlan":
        tasks_by_id = {task.task_id: task for task in self.tasks}
        if len(tasks_by_id) != len(self.tasks):
            raise ValueError("planned task IDs must be unique")
        for task in self.tasks:
            missing = tuple(
                dependency for dependency in task.depends_on if dependency not in tasks_by_id
            )
            if missing:
                raise ValueError(
                    f"planned task {task.task_id!r} has unknown dependencies: " + ", ".join(missing)
                )

        encode_tasks = tuple(
            task for task in self.tasks if task.task_type == TaskType.ENCODE_CONDITIONING
        )
        unload_tasks = tuple(
            task for task in self.tasks if task.task_type == TaskType.UNLOAD_CONDITIONING
        )
        load_tasks = tuple(task for task in self.tasks if task.task_type == TaskType.LOAD_DIFFUSION)
        generation_tasks = tuple(
            task for task in self.tasks if task.task_type == TaskType.GENERATE_SEGMENT
        )
        if encode_tasks or generation_tasks:
            if len(unload_tasks) != 1 or len(load_tasks) != 1:
                raise ValueError(
                    "conditioning execution requires exactly one unload and diffusion-load task"
                )
            unload = unload_tasks[0]
            load = load_tasks[0]
            encode_ids = {task.task_id for task in encode_tasks}
            if not encode_ids.issubset(unload.depends_on):
                raise ValueError("conditioning unload must depend on every encoding task")
            if unload.task_id not in load.depends_on:
                raise ValueError("diffusion load must depend on conditioning unload")
            for generation in generation_tasks:
                if load.task_id not in generation.depends_on:
                    raise ValueError(
                        f"generation task {generation.task_id!r} must depend on diffusion load"
                    )
                matching_encodes = {
                    task.task_id
                    for task in encode_tasks
                    if task.conditioning_fingerprint == generation.conditioning_fingerprint
                }
                if not matching_encodes.intersection(generation.depends_on):
                    raise ValueError(
                        f"generation task {generation.task_id!r} must depend on its encoding task"
                    )
        return self
