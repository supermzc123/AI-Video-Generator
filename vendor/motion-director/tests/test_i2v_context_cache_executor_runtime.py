from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from _minimax_h3_motion_director_testpkg.director import context_cache


def _segment(index: int, prompt: str, source_clip=None):
    return SimpleNamespace(
        index=index,
        ui_index=None,
        timeline_index=index,
        start_frame=index * 10,
        end_frame=(index + 1) * 10,
        frame_count=10,
        prompt=prompt,
        negative_prompt="",
        task_type="i2v",
        task_key="i2v",
        source_clip=source_clip,
        refs=[],
        ref_audios=[],
        ref_videos=[],
        ref_video_audios=[],
        reference_video_meta={},
        reference_video_start_frame=0,
        reference_tags={},
    )


def _plan(segments, *, run_indices=None):
    return SimpleNamespace(
        frame_rate=24.0,
        total_frames=len(segments) * 10,
        width=32,
        height=32,
        ref_max_size=32,
        output_mode="fixed",
        source_width=32,
        source_height=32,
        global_task_type="i2v",
        global_task_key="i2v",
        global_prompt="",
        global_refs=[],
        global_ref_audios=[],
        segments=list(segments),
        segment_count=len(segments),
        source_video=torch.full((len(segments), 16, 16, 3), 0.5),
        edit_mode="prompt_batch",
        raw={"timelineMode": "gen_blank", "output": {"audioMode": "mute"}},
        source_total_frames=len(segments),
        export_max_frames=0,
        export_mode="segments",
        run_indices=run_indices,
        run_select_enabled=run_indices is not None,
        continuity_enabled=False,
        continuity_overlap_frames=0,
        source_overlap_frames=0,
        color_reanchor_enabled=False,
        spatial_stride=32,
    )


def test_selection_run_new_i2v_segment_samples_only_new_segment_with_cached_tail(
    monkeypatch, tmp_path
):
    """Exercise the real executor/cache boundary; only GPU/model work is replaced."""
    try:
        executor = importlib.import_module(
            "_minimax_h3_motion_director_testpkg.director.executor_core"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"full ComfyUI runtime is unavailable: {exc}")

    s1 = _segment(0, "one", torch.ones((1, 32, 32, 3)))
    s2 = _segment(1, "two")
    s3 = _segment(2, "three")
    initial = _plan([s1, s2, s3])
    s4 = _segment(3, "four")
    selected = _plan([s1, s2, s3, s4], run_indices=frozenset({3}))
    monkeypatch.setattr(context_cache, "_cache_root", lambda _node: tmp_path)

    cache_written = False

    def load_real_cached_tail(node_id, seg, plan, *, settings, strict=False):
        nonlocal cache_written
        if not cache_written:
            assert context_cache.save_motion_context_cache(
                node_id,
                s3,
                initial,
                frames=torch.zeros((10, 32, 32, 3)),
                audio=None,
                settings=settings,
            )
            cache_written = True
        return context_cache.load_motion_context_cache(
            node_id, seg, plan, settings=settings, strict=strict
        )

    conditioning_prompts = []
    sample_calls = []

    def fake_conditioning(**kwargs):
        conditioning_prompts.append(kwargs["prompt"])
        length = int(kwargs["length"])
        return [[torch.zeros(1), {}]], [], {"requested_length": length}, "mock"

    def fake_sample(**kwargs):
        sample_calls.append(kwargs["seed"])
        return kwargs["latent"]

    def fake_decode(samples, *_args, **_kwargs):
        count = int(samples["requested_length"])
        return torch.zeros((count, 32, 32, 3)), None

    monkeypatch.setattr(executor, "plan_summary", lambda _plan: "mock plan")
    monkeypatch.setattr(executor, "run_minimax_conditioning", fake_conditioning)
    monkeypatch.setattr(executor, "sample_single_stage", fake_sample)
    monkeypatch.setattr(executor, "_decode_av_latent", fake_decode)
    monkeypatch.setattr(executor, "load_motion_context_cache", load_real_cached_tail)
    monkeypatch.setattr(executor, "load_latent_context_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(executor, "save_motion_context_cache", lambda *_a, **_k: True)
    monkeypatch.setattr(executor, "save_latent_context_cache", lambda *_a, **_k: True)
    monkeypatch.setattr(executor, "prepare_latent_context_tail", lambda latent, handoff: (latent, handoff))
    motion_info = SimpleNamespace(
        visual_source="cached pixels",
        audio_source="off",
        context_frames=5,
        audio_seconds=0.0,
        removed_start_anchors=0,
        preserved_last_anchors=0,
        color_reanchor_status="OFF",
    )
    monkeypatch.setattr(
        executor,
        "apply_exported_motion_context",
        lambda positive, **_k: (positive, motion_info),
    )
    monkeypatch.setattr(executor, "motion_context_patch_status", lambda: (True, "ok"))
    monkeypatch.setattr(executor, "cleanup_segment_vram", lambda **_k: None)
    monkeypatch.setattr(executor, "report_director_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(executor, "report_director_finish", lambda *_a, **_k: None)
    monkeypatch.setattr(executor, "report_director_segment_preview", lambda *_a, **_k: None)
    monkeypatch.setattr(executor, "tensor_frame_to_jpeg_b64", lambda _frame: "jpeg")

    model = SimpleNamespace(model=SimpleNamespace(), model_options={})
    combined, outputs, _audios, report = executor.execute_director_plan_core(
        plan=selected,
        node_id="i2v-selection-runtime",
        model=model,
        vae=object(),
        audio_vae=object(),
        clip=object(),
        motion_context_enabled=True,
        context_length=5,
        source_overlap_frames=0,
        audio_context_enabled=False,
        clear_vram_between_segments=False,
    )

    assert cache_written
    assert conditioning_prompts == ["four"]
    assert len(sample_calls) == 1
    assert len(outputs) == 1
    assert int(combined.shape[0]) == 10
    assert "Run selection: 1/4" in report
