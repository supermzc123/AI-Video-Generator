import pytest

from ai_video_generator.domain import (
    BindingSemantic,
    BindingValueType,
    WorkflowApproval,
    WorkflowBindingDraft,
    WorkflowInvocation,
    WorkflowOutput,
)
from ai_video_generator.workers import (
    WorkflowContractError,
    compile_workflow,
    extract_workflow_outputs,
    inspect_api_workflow,
    parse_api_workflow,
    validate_workflow_template,
    workflow_template_from_inspection,
)


def object_info() -> dict[str, object]:
    return {
        "CLIPTextEncode": {
            "input": {"required": {"clip": ["CLIP"], "text": ["STRING", {"multiline": True}]}},
            "output": ["CONDITIONING"],
        },
        "SaveImage": {
            "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}},
            "output": [],
            "output_node": True,
        },
    }


def workflow() -> dict[str, object]:
    return {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 0], "text": "default"},
            "_meta": {"title": "Positive prompt"},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["4", 0], "filename_prefix": "test"},
            "_meta": {"title": "Save result"},
        },
        "3": {"class_type": "UnknownClip", "inputs": {}, "_meta": {"title": "clip"}},
        "4": {"class_type": "UnknownImage", "inputs": {}, "_meta": {"title": "image"}},
    }


def test_parse_rejects_ui_format_and_dangling_links() -> None:
    with pytest.raises(WorkflowContractError, match="UI workflow"):
        parse_api_workflow({"nodes": []})
    with pytest.raises(WorkflowContractError, match="unknown node"):
        parse_api_workflow({"1": {"class_type": "SaveImage", "inputs": {"images": ["9", 0]}}})


def configured_inspection():
    inspection = inspect_api_workflow(workflow(), object_info())
    return inspection.model_copy(
        update={
            "bindings": (
                WorkflowBindingDraft(
                    binding_id="prompt",
                    semantic=BindingSemantic.PROMPT,
                    node_id="1",
                    input_name="text",
                    value_type=BindingValueType.STRING,
                    title="Positive prompt",
                    default_value="default",
                ),
            ),
            "unknown_node_types": (),
            "issues": (),
        }
    )


def test_manual_mapping_compilation_and_output_extraction() -> None:
    inspection = inspect_api_workflow(workflow(), object_info())
    assert inspection.bindings == ()
    assert inspection.outputs[0].node_id == "2"
    assert inspection.unknown_node_types == ("UnknownClip", "UnknownImage")

    # Confirming unknown custom nodes is a separate approval action; compilation
    # itself remains deterministic once a frozen approved template exists.
    approved = workflow_template_from_inspection(
        configured_inspection(),
        template_id="image:test",
        name="test",
        approval=WorkflowApproval.APPROVED,
    )
    compiled = compile_workflow(
        approved,
        WorkflowInvocation(
            template_id="image:test",
            template_revision=1,
            values={"prompt": "new prompt"},
        ),
    )
    assert compiled.workflow["1"]["inputs"]["text"] == "new prompt"
    assert compiled.workflow_sha256 != approved.workflow_sha256

    outputs = extract_workflow_outputs(
        {
            "outputs": {
                "2": {"images": [{"filename": "result.png", "subfolder": "test", "type": "output"}]}
            }
        },
        approved,
    )
    assert outputs[0].filename == "result.png"
    assert outputs[0].subfolder == "test"


def test_compiler_rejects_unknown_binding_values() -> None:
    template = workflow_template_from_inspection(
        configured_inspection(),
        template_id="image:test",
        name="test",
        approval=WorkflowApproval.APPROVED,
    )
    with pytest.raises(WorkflowContractError, match="unknown workflow binding"):
        compile_workflow(template, {"not_a_binding": 1})


def test_manual_output_selection_does_not_require_avg_title_marker() -> None:
    info = {
        **object_info(),
        "UnknownImage": {"input": {"required": {}}, "output": ["IMAGE"]},
    }
    source = {
        "1": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "test"},
            "_meta": {"title": "保存最终图片"},
        },
        "2": {"class_type": "UnknownImage", "inputs": {}},
    }
    inspection = inspect_api_workflow(source, info)
    selected = inspection.model_copy(
        update={
            "outputs": (
                WorkflowOutput(
                    output_id="image:1",
                    node_id="1",
                    title="保存最终图片",
                ),
            ),
            "unknown_node_types": (),
            "issues": (),
        }
    )
    template = workflow_template_from_inspection(
        selected,
        template_id="image:manual-output",
        name="manual output",
        approval=WorkflowApproval.APPROVED,
    )

    assert validate_workflow_template(template, info) == ()


def test_unrelated_worker_schema_change_does_not_invalidate_template() -> None:
    info = {
        **object_info(),
        "UnknownClip": {"input": {"required": {}}, "output": ["CLIP"]},
        "UnknownImage": {"input": {"required": {}}, "output": ["IMAGE"]},
    }
    inspection = configured_inspection().model_copy(
        update={"node_schema_sha256": "0" * 64}
    )
    template = workflow_template_from_inspection(
        inspection,
        template_id="image:schema-change",
        name="schema change",
        approval=WorkflowApproval.APPROVED,
    )
    changed_info = {
        **info,
        "UnrelatedCustomNode": {
            "input": {"required": {"value": ["FLOAT", {"default": 1.0}]}},
            "output": ["FLOAT"],
        },
    }

    assert validate_workflow_template(template, changed_info) == ()


def test_node_titles_do_not_create_bindings() -> None:
    info = {
        "LoadImage": {
            "input": {"required": {"image": ["STRING"]}},
            "output": ["IMAGE"],
        },
        "SaveImage": object_info()["SaveImage"],
    }
    source = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "reference.png"},
            "_meta": {"title": "参考图一"},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0], "filename_prefix": "test"},
            "_meta": {"title": "保存结果"},
        },
    }

    inspection = inspect_api_workflow(source, info)

    assert inspection.bindings == ()
    assert inspection.outputs[0].node_id == "2"


def test_comfy_v3_autogrow_dotted_inputs_are_validated_as_typed_links() -> None:
    info = {
        "ReferenceNode": {
            "input": {
                "required": {},
                "optional": {
                    "ref_images": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"ref_image": ["IMAGE", {}]}},
                                "prefix": "ref_image_",
                                "min": 0,
                                "max": 2,
                            }
                        },
                    ]
                },
            },
            "output": ["IMAGE"],
        },
        "LoadImage": {
            "input": {"required": {"image": ["STRING"]}},
            "output": ["IMAGE"],
        },
        "SaveImage": object_info()["SaveImage"],
    }
    source = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "one.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "two.png"}},
        "3": {
            "class_type": "ReferenceNode",
            "inputs": {
                "ref_images.ref_image_0": ["1", 0],
                "ref_images.ref_image_1": ["2", 0],
            },
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "test"},
            "_meta": {"title": "保存图片"},
        },
    }

    inspection = inspect_api_workflow(source, info)

    assert inspection.compatible


def test_comfy_v3_autogrow_rejects_unknown_or_out_of_range_slots() -> None:
    info = {
        "ReferenceNode": {
            "input": {
                "optional": {
                    "ref_images": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"ref_image": ["IMAGE", {}]}},
                                "prefix": "ref_image_",
                                "min": 0,
                                "max": 1,
                            }
                        },
                    ]
                }
            },
            "output": ["IMAGE"],
        },
        "LoadImage": {
            "input": {"required": {"image": ["STRING"]}},
            "output": ["IMAGE"],
        },
        "SaveImage": object_info()["SaveImage"],
    }
    source = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "one.png"}},
        "2": {
            "class_type": "ReferenceNode",
            "inputs": {"ref_images.ref_image_1": ["1", 0]},
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "test"},
            "_meta": {"title": "保存图片"},
        },
    }

    inspection = inspect_api_workflow(source, info)

    assert any("not exposed by object_info" in issue for issue in inspection.issues)


def test_dynamic_combo_and_matchtype_are_valid_workflow_contracts() -> None:
    info = {
        "LoadImage": {
            "input": {"required": {"image": ["STRING"]}},
            "output": ["IMAGE"],
        },
        "ModelLoader": {
            "input": {"required": {"model": ["COMBO", {"options": ["model.safetensors"]}]}},
            "output": ["MODEL"],
        },
        "ResizeImageMaskNode": {
            "input": {
                "required": {
                    "input": ["COMFY_MATCHTYPE_V3"],
                    "resize_type": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {
                                    "key": "scale by multiplier",
                                    "inputs": {
                                        "required": {
                                            "multiplier": [
                                                "FLOAT",
                                                {"min": 0.01, "max": 8.0},
                                            ]
                                        }
                                    },
                                }
                            ]
                        },
                    ],
                }
            },
            "output": ["COMFY_MATCHTYPE_V3"],
        },
        "SaveImage": object_info()["SaveImage"],
    }
    source = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
        "2": {"class_type": "ModelLoader", "inputs": {"model": "model.safetensors"}},
        "3": {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["1", 0],
                "resize_type": "scale by multiplier",
                "resize_type.multiplier": 2.0,
            },
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "test"},
        },
    }

    inspection = inspect_api_workflow(source, info).model_copy(
        update={
            "bindings": (
                WorkflowBindingDraft(
                    binding_id="model",
                    semantic=BindingSemantic.MODEL,
                    node_id="2",
                    input_name="model",
                    value_type=BindingValueType.STRING,
                    title="Model",
                ),
                WorkflowBindingDraft(
                    binding_id="upscale",
                    semantic=BindingSemantic.UPSCALE_FACTOR,
                    node_id="3",
                    input_name="resize_type.multiplier",
                    value_type=BindingValueType.NUMBER,
                    title="Upscale factor",
                ),
            )
        }
    )
    template = workflow_template_from_inspection(
        inspection,
        template_id="restoration:test",
        name="Restoration",
        approval=WorkflowApproval.APPROVED,
    )

    assert inspection.compatible
    assert validate_workflow_template(template, info) == ()
