from __future__ import annotations

from typing import Any

from ai_video_generator.domain import (
    BindingSemantic,
    ComfyUIOutput,
    TaskKind,
    TaskWorkloadManifest,
    WorkflowInvocation,
    WorkflowOutputType,
    WorkflowTemplate,
)
from ai_video_generator.workers.workflow import WorkflowContractError, compile_workflow


def compile_user_video_workflow_manifest(
    *,
    template: WorkflowTemplate,
    task_kind: TaskKind,
    expected_kind: str,
    project_id: str,
    segment_id: str,
    source_task_id: str,
    input_mount_path: str,
    model: str | None = None,
    factor: float | None = None,
    language: str | None = None,
) -> TaskWorkloadManifest:
    """Compile an approved user video template without any Harness dependency."""
    if task_kind not in {TaskKind.SEEDVR2, TaskKind.RIFE, TaskKind.WHISPER}:
        raise WorkflowContractError("user video workflows only support post-processing tasks")
    if template.kind != expected_kind:
        raise WorkflowContractError(
            f"workflow {template.template_id} is {template.kind}, expected {expected_kind}"
        )
    expected_output = (
        WorkflowOutputType.SUBTITLE
        if expected_kind == "transcription"
        else WorkflowOutputType.VIDEO
    )
    declared_outputs = tuple(
        output for output in template.outputs if output.output_type == expected_output
    )
    if not declared_outputs:
        raise WorkflowContractError(
            f"user {expected_kind} workflow has no declared {expected_output.value} output"
        )

    values: dict[str, Any] = {}
    factor_semantic = (
        BindingSemantic.INTERPOLATION_FACTOR
        if expected_kind == "interpolation"
        else BindingSemantic.UPSCALE_FACTOR
    )
    for binding in template.bindings:
        if binding.semantic == BindingSemantic.SOURCE_VIDEO:
            values[binding.binding_id] = input_mount_path
        elif binding.semantic == BindingSemantic.MODEL and model:
            values[binding.binding_id] = model
        elif binding.semantic == factor_semantic and factor is not None:
            values[binding.binding_id] = factor
        elif binding.semantic == BindingSemantic.LANGUAGE and language:
            values[binding.binding_id] = language

    compiled = compile_workflow(
        template,
        WorkflowInvocation(
            template_id=template.template_id,
            template_revision=template.revision,
            values=values,
        ),
    )
    return TaskWorkloadManifest(
        task_kind=task_kind,
        workflow_sha256=compiled.workflow_sha256,
        node_schema_sha256=template.node_schema_sha256,
        prompt=compiled.workflow,
        outputs=tuple(
            ComfyUIOutput(
                node_id=output.node_id,
                media_type="text/srt" if expected_kind == "transcription" else "video/mp4",
            )
            for output in declared_outputs
        ),
        required_node_types=template.required_node_types,
        workflow_template_id=template.template_id,
        context={
            "project_id": project_id,
            "segment_id": segment_id,
            "source_task_id": source_task_id,
            "input_mount_path": input_mount_path,
            "profile_id": template.template_id,
            "profile_revision": str(template.revision),
            "model_id": model or "",
            "language": language or "",
        },
    )
