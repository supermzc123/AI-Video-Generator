import pytest
from pydantic import ValidationError

from ai_video_generator.domain import (
    BindingSemantic,
    BindingValueType,
    WorkflowApproval,
    WorkflowBinding,
    WorkflowOutput,
    WorkflowTemplate,
)
from ai_video_generator.workers.workflow import canonical_json_sha256


def make_workflow() -> dict[str, dict[str, object]]:
    return {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a portrait"},
            "_meta": {"title": "AVG_PROMPT"},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0]},
            "_meta": {"title": "AVG_OUTPUT_IMAGE"},
        },
    }


def test_approved_workflow_contract_references_real_nodes() -> None:
    workflow = make_workflow()
    template = WorkflowTemplate(
        template_id="portrait",
        name="Portrait",
        workflow_sha256=canonical_json_sha256(workflow),
        node_schema_sha256="b" * 64,
        raw_workflow=workflow,
        bindings=(
            WorkflowBinding(
                binding_id="prompt",
                semantic=BindingSemantic.PROMPT,
                node_id="1",
                input_name="text",
                value_type=BindingValueType.STRING,
                title="Prompt",
            ),
        ),
        outputs=(WorkflowOutput(output_id="image", node_id="2", title="Image"),),
        required_node_types=("CLIPTextEncode", "SaveImage"),
        approval=WorkflowApproval.APPROVED,
    )

    assert template.bindings[0].node_id == "1"


def test_unknown_nodes_prevent_workflow_approval() -> None:
    with pytest.raises(ValidationError, match="unknown node types"):
        WorkflowTemplate(
            template_id="unsafe",
            name="Unsafe",
            workflow_sha256=canonical_json_sha256(make_workflow()),
            node_schema_sha256="b" * 64,
            raw_workflow=make_workflow(),
            outputs=(WorkflowOutput(output_id="image", node_id="2", title="Image"),),
            required_node_types=("UnknownNode",),
            unknown_node_types=("UnknownNode",),
            approval=WorkflowApproval.APPROVED,
        )


def test_binding_semantic_enforces_value_type() -> None:
    with pytest.raises(ValidationError, match="width requires integer"):
        WorkflowBinding(
            binding_id="width",
            semantic=BindingSemantic.WIDTH,
            node_id="1",
            input_name="width",
            value_type=BindingValueType.STRING,
            title="Width",
        )
