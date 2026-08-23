from __future__ import annotations

import pytest

from ai_video_generator.domain import (
    BindingSemantic,
    BindingValueType,
    TaskKind,
    WorkflowApproval,
    WorkflowBinding,
    WorkflowOutput,
    WorkflowOutputType,
    WorkflowTemplate,
)
from ai_video_generator.services.user_video_workflows import (
    compile_user_video_workflow_manifest,
)
from ai_video_generator.workers.workflow import (
    WorkflowContractError,
    canonical_json_sha256,
)


def video_template(kind: str) -> WorkflowTemplate:
    factor_semantic = (
        BindingSemantic.INTERPOLATION_FACTOR
        if kind == "interpolation"
        else BindingSemantic.UPSCALE_FACTOR
    )
    raw = {
        "1": {
            "class_type": "LoadVideo",
            "inputs": {"video": "input.mp4"},
        },
        "2": {
            "class_type": "VideoProcessor",
            "inputs": {
                "video": ["1", 0],
                "model": "default.safetensors",
                "factor": 2.0,
            },
        },
        "3": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["2", 0]},
        },
    }
    return WorkflowTemplate(
        template_id=f"user:{kind}",
        revision=1,
        name=kind,
        kind=kind,
        workflow_sha256=canonical_json_sha256(raw),
        node_schema_sha256="a" * 64,
        raw_workflow=raw,
        bindings=(
            WorkflowBinding(
                binding_id="source",
                semantic=BindingSemantic.SOURCE_VIDEO,
                node_id="1",
                input_name="video",
                value_type=BindingValueType.VIDEO_PATH,
                title="Source",
                default_value="input.mp4",
            ),
            WorkflowBinding(
                binding_id="model",
                semantic=BindingSemantic.MODEL,
                node_id="2",
                input_name="model",
                value_type=BindingValueType.STRING,
                title="Model",
                default_value="default.safetensors",
            ),
            WorkflowBinding(
                binding_id="factor",
                semantic=factor_semantic,
                node_id="2",
                input_name="factor",
                value_type=BindingValueType.NUMBER,
                title="Factor",
                default_value=2.0,
            ),
        ),
        outputs=(
            WorkflowOutput(
                output_id="video:3",
                node_id="3",
                output_type=WorkflowOutputType.VIDEO,
                title="Output",
            ),
        ),
        required_node_types=("LoadVideo", "SaveVideo", "VideoProcessor"),
        approval=WorkflowApproval.APPROVED,
    )


@pytest.mark.parametrize(
    ("kind", "task_kind"),
    (
        ("interpolation", TaskKind.RIFE),
        ("restoration", TaskKind.SEEDVR2),
    ),
)
def test_user_video_workflow_compiler_overrides_only_runtime_bindings(
    kind: str,
    task_kind: TaskKind,
) -> None:
    manifest = compile_user_video_workflow_manifest(
        template=video_template(kind),
        task_kind=task_kind,
        expected_kind=kind,
        project_id="project",
        segment_id="segment-1",
        source_task_id="source-task",
        input_mount_path="post/project/segment-1.mp4",
        model="selected.safetensors",
        factor=3.0,
    )

    assert manifest.prompt["1"]["inputs"]["video"] == "post/project/segment-1.mp4"
    assert manifest.prompt["2"]["inputs"]["model"] == "selected.safetensors"
    assert manifest.prompt["2"]["inputs"]["factor"] == 3.0
    assert manifest.workflow_template_id == f"user:{kind}"
    assert manifest.context["source_task_id"] == "source-task"


def test_user_video_workflow_compiler_rejects_cross_purpose_template() -> None:
    with pytest.raises(WorkflowContractError, match="expected restoration"):
        compile_user_video_workflow_manifest(
            template=video_template("interpolation"),
            task_kind=TaskKind.SEEDVR2,
            expected_kind="restoration",
            project_id="project",
            segment_id="segment-1",
            source_task_id="source-task",
            input_mount_path="post/project/segment-1.mp4",
        )


def test_user_transcription_workflow_compiles_video_and_language_bindings() -> None:
    raw = {
        "1": {
            "class_type": "TranscribeVideo",
            "inputs": {"video": "input.mp4", "model": "large-v3", "language": "auto"},
        },
        "2": {"class_type": "SaveSubtitle", "inputs": {"subtitle": ["1", 0]}},
    }
    template = WorkflowTemplate(
        template_id="user:transcription",
        revision=1,
        name="Transcription",
        kind="transcription",
        workflow_sha256=canonical_json_sha256(raw),
        node_schema_sha256="b" * 64,
        raw_workflow=raw,
        bindings=(
            WorkflowBinding(
                binding_id="source",
                semantic=BindingSemantic.SOURCE_VIDEO,
                node_id="1",
                input_name="video",
                value_type=BindingValueType.VIDEO_PATH,
                title="Source",
            ),
            WorkflowBinding(
                binding_id="model",
                semantic=BindingSemantic.MODEL,
                node_id="1",
                input_name="model",
                value_type=BindingValueType.STRING,
                title="Model",
            ),
            WorkflowBinding(
                binding_id="language",
                semantic=BindingSemantic.LANGUAGE,
                node_id="1",
                input_name="language",
                value_type=BindingValueType.STRING,
                title="Language",
            ),
        ),
        outputs=(
            WorkflowOutput(
                output_id="subtitle:2",
                node_id="2",
                output_type=WorkflowOutputType.SUBTITLE,
                title="Subtitle",
            ),
        ),
        required_node_types=("SaveSubtitle", "TranscribeVideo"),
        approval=WorkflowApproval.APPROVED,
    )

    manifest = compile_user_video_workflow_manifest(
        template=template,
        task_kind=TaskKind.WHISPER,
        expected_kind="transcription",
        project_id="project",
        segment_id="master",
        source_task_id="master-task",
        input_mount_path="post/project/master.mp4",
        model="small",
        language="zh",
    )

    assert manifest.prompt["1"]["inputs"] == {
        "video": "post/project/master.mp4",
        "model": "small",
        "language": "zh",
    }
    assert manifest.outputs[0].media_type == "text/srt"
