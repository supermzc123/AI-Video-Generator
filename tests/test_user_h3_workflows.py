from ai_video_generator.domain import (
    BindingSemantic,
    BindingValueType,
    WorkflowApproval,
    WorkflowBinding,
    WorkflowOutput,
    WorkflowOutputType,
    WorkflowTemplate,
)
from ai_video_generator.services.user_h3_workflows import (
    compile_user_h3_segment_manifests,
    validate_h3_template_contract,
)
from ai_video_generator.workers.workflow import canonical_json_sha256


def template(kind: str) -> WorkflowTemplate:
    conditioning = kind == "h3_conditioning"
    workflow = {
        "1": {
            "class_type": "UserH3",
            "inputs": (
                {"prompt": "", "width": 1, "height": 1, "frames": 1, "key": ""}
                if conditioning
                else {"seed": 0, "key": "", "prefix": "", "context": "", "context_out": ""}
            ),
        }
    }
    definitions = (
        [
            (BindingSemantic.PROMPT, "prompt", BindingValueType.STRING),
            (BindingSemantic.WIDTH, "width", BindingValueType.INTEGER),
            (BindingSemantic.HEIGHT, "height", BindingValueType.INTEGER),
            (BindingSemantic.FRAME_COUNT, "frames", BindingValueType.INTEGER),
            (BindingSemantic.CONDITIONING_FINGERPRINT, "key", BindingValueType.STRING),
        ]
        if conditioning
        else [
            (BindingSemantic.SEED, "seed", BindingValueType.INTEGER),
            (BindingSemantic.CONDITIONING_FINGERPRINT, "key", BindingValueType.STRING),
            (BindingSemantic.OUTPUT_PREFIX, "prefix", BindingValueType.STRING),
            (BindingSemantic.MOTION_CONTEXT_INPUT, "context", BindingValueType.STRING),
            (
                BindingSemantic.MOTION_CONTEXT_OUTPUT_PREFIX,
                "context_out",
                BindingValueType.STRING,
            ),
        ]
    )
    return WorkflowTemplate(
        template_id=f"user:{kind}",
        name=kind,
        kind=kind,
        workflow_sha256=canonical_json_sha256(workflow),
        node_schema_sha256="a" * 64,
        raw_workflow=workflow,
        bindings=tuple(
            WorkflowBinding(
                binding_id=semantic.value,
                semantic=semantic,
                node_id="1",
                input_name=input_name,
                value_type=value_type,
                title=semantic.value,
            )
            for semantic, input_name, value_type in definitions
        ),
        outputs=(
            WorkflowOutput(
                output_id=f"{kind}:1",
                node_id="1",
                output_type=(
                    WorkflowOutputType.CONDITIONING
                    if conditioning
                    else WorkflowOutputType.VIDEO
                ),
                title="output",
            ),
        ),
        required_node_types=("UserH3",),
        approval=WorkflowApproval.APPROVED,
    )


def test_custom_h3_pair_compiles_with_shared_conditioning_key() -> None:
    encoding = template("h3_conditioning")
    diffusion = template("h3_diffusion")
    assert validate_h3_template_contract(encoding) == ()
    assert validate_h3_template_contract(diffusion) == ()

    encode_manifest, diffusion_manifest = compile_user_h3_segment_manifests(
        conditioning_template=encoding,
        diffusion_template=diffusion,
        project_id="project",
        prompt={
            "segmentId": "segment-1",
            "durationSeconds": 4,
            "prompt": "H3 prompt",
            "seed": 42,
            "continuationOf": None,
        },
        width=640,
        height=352,
        node_schema_sha256="b" * 64,
    )

    assert encode_manifest.prompt["1"]["inputs"]["prompt"] == "H3 prompt"
    assert encode_manifest.prompt["1"]["inputs"]["frames"] == 107
    assert (
        encode_manifest.prompt["1"]["inputs"]["key"]
        == diffusion_manifest.prompt["1"]["inputs"]["key"]
    )
    assert diffusion_manifest.prompt["1"]["inputs"]["seed"] == 42
    assert diffusion_manifest.workflow_template_id == "user:h3_diffusion"
