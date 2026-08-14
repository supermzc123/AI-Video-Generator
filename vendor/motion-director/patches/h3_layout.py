# Portions derived from ComfyUI-H3-Motion-Context
# Copyright (C) 2026 NikoDemon80 and contributors
# Modified for MiniMax H3 Motion Director, 2026-08-09
# Licensed under GNU GPL v3.0. See LICENSE and NOTICE.

"""Lift MiniMax H3's first/last-only keyframe anchor restriction.

Stock ComfyUI builds keyframe conditioning rows at one of two time
coordinates and rejects everything else:

    if pixel_index == 0:
        cond_t = float(text_len)
    elif frame_count is not None and pixel_index == frame_count - 1:
        cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
    else:
        raise ValueError("only first/last keyframe anchors are supported")

Both branches are the same expression. Each video token spans
FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] and covers FRAME_PER_TOKEN[k % 5]
pixel frames, so the cumulative time at pixel frame p is exactly
FRAME_RESCALE * p, for every p. Substituting p = frame_count - 1
reproduces the second branch identically:

    text_len + FRAME_RESCALE * (frame_count - 1)
      == text_len + FRAME_RESCALE * frame_count - FRAME_RESCALE
      == text_len + sum(_video_t_spans(latent_t)) - FRAME_RESCALE

So the general position is:

    cond_t = text_len + FRAME_RESCALE * pixel_index

We do NOT rewrite the source of PackedLayout.__init__. Instead every
keyframe is handed to stock code with resolved_frame_index = 0, which is
always legal, and the real index rides along under MC_KEY. After the
stock constructor returns we rewrite the time column of each cond
segment's rows in position_ids. RoPE is built at forward time from
position_ids, so this lands before anything reads it.

That keeps the patch surface to one attribute we can verify rather than a
copy of a 90-line constructor that would rot on the next ComfyUI change.
"""

import logging

import torch

import comfy.ldm.minimax.model as mm

from .markers import MC_AUDIO_KEY, MC_KEY

_LOG = logging.getLogger("h3_motion_context")

_orig_init = None
_applied = False
_failure_reason = None


REF_SEGMENT_KINDS = ("ref_img", "ref_audio")


def _target_origin(layout):
    """Read the target clip's real start coordinate from PackedLayout."""
    if not layout.segments:
        raise RuntimeError(
            "h3_motion_context: PackedLayout has no segments. Upstream layout "
            "change; refusing to rewrite positions."
        )
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "h3_motion_context: expected the target video rows to be the last "
            "layout segment, found %r spanning %d rows. Upstream layout "
            "change; refusing to rewrite positions." % (kind, b - a)
        )
    return float(layout.position_ids[a, 0])


def _expected_ref_segments(blk):
    """Return the actual segment kinds one stock H3 reference should emit."""
    kind = blk.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",)
    if kind == "video":
        return ("ref_img",)
    if kind == "video_audio":
        return ("ref_audio", "ref_img")
    raise RuntimeError(
        "h3_motion_context: unknown reference kind %r; cannot tell which "
        "layout rows belong to it." % (kind,)
    )


def _ref_segment_map(layout, refs):
    """Map each reference list index to the exact rows PackedLayout emitted."""
    ref_segments = [
        (a, b, kind)
        for a, b, kind in layout.segments
        if kind in REF_SEGMENT_KINDS
    ]
    expected = [
        (index, kind)
        for index, block in enumerate(refs or [])
        for kind in _expected_ref_segments(block)
    ]
    if len(expected) != len(ref_segments):
        raise RuntimeError(
            "h3_motion_context: %d reference blocks should produce %d "
            "reference layout segments, but PackedLayout contains %d. "
            "ComfyUI MiniMax H3 layout behavior changed; refusing to move "
            "rows."
            % (len(refs or []), len(expected), len(ref_segments))
        )
    mapped = {}
    for (index, wanted), (a, b, actual) in zip(expected, ref_segments):
        if actual != wanted:
            raise RuntimeError(
                "h3_motion_context: reference block %d (%r) should emit a "
                "%s segment, but PackedLayout contains %s at that position. "
                "ComfyUI MiniMax H3 layout behavior changed; refusing to "
                "move rows."
                % (index, refs[index].get("kind"), wanted, actual)
            )
        mapped.setdefault(index, {})[wanted] = (a, b)
    return mapped


def _cond_t(text_len, latent_t, frame_count, p):
    """Time coordinate for a keyframe anchored at pixel frame p.

    The endpoints reuse stock's exact expressions rather than the general
    formula. They are mathematically identical, but stock accumulates
    latent_t float additions where the general form does one multiply, and
    those differ in the last bits (about 7e-15). Matching stock bit for bit
    means an existing first/last graph builds byte-identical positions
    after this patch is applied, and lets the self-test stay strict.
    """
    if p == 0:
        return float(text_len)
    if frame_count is not None and p == frame_count - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _fixup(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    """Rewrite cond rows relative to the target origin PackedLayout built."""
    offset = _target_origin(layout) - float(text_len)
    if offset and any(kf.get(MC_KEY) is None for kf in keyframes):
        # keyframes without MC_KEY are left exactly as stock built them,
        # which means they do NOT get the ref cursor compensation. Mixing
        # them with MC keyframes under a ref would slide the stock anchors
        # relative to ours and to the target. Nothing produces this today;
        # refuse loudly in case something ever does.
        raise RuntimeError(
            "h3_motion_context: stock and motion-context keyframes mixed in "
            "one graph alongside a ref; their coordinates would disagree. "
            "Give every keyframe a %s entry or remove the refs." % MC_KEY)
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "h3_motion_context: expected %d cond segments, layout has %d. "
            "Refusing to rewrite positions."
            % (len(keyframes), len(cond_spans)))
    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(MC_KEY)
        if p is None:
            continue
        layout.position_ids[a:b, 0] = _cond_t(text_len, latent_t, frame_count, p) + offset


def _fixup_audio(layout, text_len, refs):
    """Move exactly the marked audio ref segment onto the target timeline.

    Refs and keyframes carry identical row machinery; what makes the model
    read a ref as "a separate clip to imitate" rather than "this clip,
    continued" is that its coordinates sit in a span before the target.
    That distinction decided continuation vs reproduction for video, and
    seam analysis showed the audio ref producing phase-unlocked imitation.
    So: keep the audio on the ref path for construction and payload (rows
    built, latents filled, all stock code untouched) and TRANSLATE its
    time coordinates so the window END lands at target frame
    MC_AUDIO_KEY -- the same instant the pinned video ends.

    Translation, not per-row assignment: new = old + shift preserves
    whatever intra-block structure stock built (row order, the rows-per-
    step factor, fractional offsets), so nothing about the block's
    internals is assumed.

    The ref still advances the layout cursor, so its old coordinate slot
    [text_len, text_len + rt) is left VACANT after the move. An audio
    window longer than the video window therefore spills backwards into
    empty coordinate space rather than onto the text rows -- the collision
    that made `before` mode fail for video does not arise here.

    Row selection comes from PackedLayout's actual segment table.  This is
    exact even when Picture, Video, paired Video Audio, normal Audio, and
    Motion Audio references coexist or are reordered.
    """
    marked = [
        index
        for index, block in enumerate(refs or [])
        if block.get(MC_AUDIO_KEY) is not None
    ]
    if len(marked) != 1:
        raise RuntimeError(
            "h3_motion_context: expected exactly one marked Motion Audio "
            "Context ref, found %d among %d refs."
            % (len(marked), len(refs or []))
        )
    index = marked[0]
    blk = refs[index]
    if blk.get("kind") != "audio":
        raise RuntimeError(
            "h3_motion_context: %s is set on a %r ref; only a standalone "
            "audio ref may be moved onto the target timeline."
            % (MC_AUDIO_KEY, blk.get("kind"))
        )
    rt = int(blk.get("ref_audio_t", 0))
    if rt <= 0:
        raise RuntimeError(
            "h3_motion_context: marked Motion Audio Context has no latent steps."
        )

    segment = _ref_segment_map(layout, refs).get(index, {}).get("ref_audio")
    if segment is None:
        raise RuntimeError(
            "h3_motion_context: the marked Motion Audio Context produced no "
            "ref_audio segment. ComfyUI MiniMax H3 layout behavior changed; "
            "refusing to move rows."
        )
    a, b = segment
    expected_rows = rt * 2
    if b - a != expected_rows:
        raise RuntimeError(
            "h3_motion_context: the marked Motion Audio Context has %d rows "
            "for %d latent steps, expected exactly %d. ComfyUI MiniMax H3 "
            "layout behavior changed; refusing to move rows."
            % (b - a, rt, expected_rows)
        )

    end_frame = float(blk[MC_AUDIO_KEY])
    target_origin = _target_origin(layout)
    old_start = float(layout.position_ids[a, 0])
    # window end at target time FRAME_RESCALE * end_frame, width rt steps
    desired_start = target_origin + mm.FRAME_RESCALE * end_frame - float(rt)
    layout.position_ids[a:b, 0] = (
        layout.position_ids[a:b, 0] + (desired_start - old_start)
    )


def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, frame_count=None):
    _orig_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
               keyframes=keyframes, refs=refs, frame_count=frame_count)
    has_mc_kf = bool(keyframes) and any(
        kf.get(MC_KEY) is not None for kf in keyframes)
    has_mc_audio = bool(refs) and any(
        r.get(MC_AUDIO_KEY) is not None for r in refs)
    if has_mc_kf:
        _fixup(self, text_len, latent_t, frame_count, keyframes, refs)
    if has_mc_audio:
        _fixup_audio(self, text_len, refs)
    # neither marked: stock graph, leave it exactly as built


def _self_test():
    """Prove the rewrite reproduces stock positions before committing.

    Builds the two anchors stock code already supports, once the stock way
    and once through our mechanism, and requires the position tensors to
    match exactly. If ComfyUI changes the position maths underneath us this
    fails and the patch is not applied.
    """
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    stock_kf = [{"resolved_frame_index": 0},
                {"resolved_frame_index": frame_count - 1}]
    ours_kf = [{"resolved_frame_index": 0, MC_KEY: 0},
               {"resolved_frame_index": 0, MC_KEY: frame_count - 1}]

    a = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(a, text_len, latent_t, lh, lw, audio_t,
               keyframes=stock_kf, frame_count=frame_count)

    b = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(b, text_len, latent_t, lh, lw, audio_t,
               keyframes=ours_kf, frame_count=frame_count)
    _fixup(b, text_len, latent_t, frame_count, ours_kf)

    if a.position_ids.shape != b.position_ids.shape:
        raise RuntimeError("position_ids shape mismatch in self-test")
    if not torch.equal(a.position_ids, b.position_ids):
        bad = (a.position_ids != b.position_ids).any(dim=1).nonzero().flatten()
        raise RuntimeError("position mismatch at rows %s" % bad[:8].tolist())

    # a consecutive run must land on strictly increasing coordinates inside
    # the span the two endpoints define
    run = [{"resolved_frame_index": 0, MC_KEY: i} for i in range(4)]
    c = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(c, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, frame_count=frame_count)
    _fixup(c, text_len, latent_t, frame_count, run)
    ts = [float(c.position_ids[s, 0]) for s, _, k in c.segments if k == "cond"]
    if len(ts) != len(run):
        raise RuntimeError("expected %d cond segments, got %d" % (len(run), len(ts)))
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise RuntimeError("consecutive anchors not strictly increasing: %s" % ts)
    t_last = float(text_len) + mm.FRAME_RESCALE * (frame_count - 1)
    if not (ts[0] == float(text_len) and ts[-1] < t_last):
        raise RuntimeError("run %s escapes the [%.4f, %.4f] span"
                           % (ts, float(text_len), t_last))

    # Adding a ref must keep every anchor at the same offset from the real
    # target origin.  The origin is read directly from the final video
    # segment rather than reconstructed from reference metadata.
    ref = [{"kind": "audio", "ref_audio_t": 8}]
    d = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(d, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=ref, frame_count=frame_count)
    _fixup(d, text_len, latent_t, frame_count, run, refs=ref)
    ts_ref = [float(d.position_ids[s, 0]) for s, _, k in d.segments if k == "cond"]
    if len(ts_ref) != len(ts):
        raise RuntimeError("cond segment count changed when a ref was added")
    tol = 1e-3
    origin = _target_origin(d)
    expected_ts = [origin + mm.FRAME_RESCALE * i for i in range(len(run))]
    if any(abs(got - expected) > tol
           for got, expected in zip(ts_ref, expected_ts)):
        raise RuntimeError(
            "ref-compensated anchors %s do not follow target origin %.6f"
            % (ts_ref, origin)
        )

    # Multi-ref audio placement: Picture, Video, Motion Audio, user Audio.
    # Only the exact segment emitted by the marked block may move. The marker
    # deliberately sits before another audio block, proving lookup does not
    # depend on the Motion Audio ref being first or last.
    end_frame = 4
    rt = 8
    refs_plain = [
        {"kind": "image", "latent_h": 8, "latent_w": 12},
        {"kind": "video", "latent_t": 2, "latent_h": 8, "latent_w": 12,
         "ref_audio_t": 0},
        {"kind": "audio", "ref_audio_t": rt},
        {"kind": "audio", "ref_audio_t": 5},
    ]
    ref_mc = [dict(r) for r in refs_plain]
    ref_mc[2][MC_AUDIO_KEY] = end_frame
    d_audio = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(d_audio, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=refs_plain, frame_count=frame_count)
    _fixup(d_audio, text_len, latent_t, frame_count, run, refs=refs_plain)
    e = mm.PackedLayout.__new__(mm.PackedLayout)
    _orig_init(e, text_len, latent_t, lh, lw, audio_t,
               keyframes=run, refs=ref_mc, frame_count=frame_count)
    _fixup(e, text_len, latent_t, frame_count, run, refs=ref_mc)
    _fixup_audio(e, text_len, ref_mc)
    if e.position_ids.shape != d_audio.position_ids.shape:
        raise RuntimeError("audio move changed the layout shape")
    if not torch.equal(d_audio.position_ids[:, 1:], e.position_ids[:, 1:]):
        raise RuntimeError("audio move touched a non-time coordinate column")
    td, te = d_audio.position_ids[:, 0], e.position_ids[:, 0]
    segment_map = _ref_segment_map(d_audio, refs_plain)
    a, b = segment_map[2]["ref_audio"]
    expect_moved = set(range(a, b))
    moved = set(i for i in range(len(td)) if float(td[i]) != float(te[i]))
    if moved != expect_moved:
        raise RuntimeError(
            "audio move touched the wrong rows: %d moved, %d expected, "
            "e.g. %s" % (len(moved), len(expect_moved),
                         sorted(moved ^ expect_moved)[:8]))
    if not moved:
        raise RuntimeError("audio move moved no rows")
    deltas = [float(te[i]) - float(td[i]) for i in sorted(moved)]
    if any(abs(dd - deltas[0]) > 1e-5 for dd in deltas):
        raise RuntimeError("audio rows did not move by one uniform shift: %s"
                           % (deltas[:4],))
    desired_start = (_target_origin(e) + mm.FRAME_RESCALE * end_frame
                     - float(rt))
    if abs(float(te[a]) - desired_start) > 1e-5:
        raise RuntimeError(
            "audio segment starts at %.6f, expected %.6f from target origin"
            % (float(te[a]), desired_start)
        )


def apply_patch():
    global _orig_init, _applied, _failure_reason
    if _applied:
        return True
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        _failure_reason = "MiniMax H3 model module is missing PackedLayout/FRAME_RESCALE"
        _LOG.warning("h3_motion_context: %s; patch not applied", _failure_reason)
        return False
    _orig_init = mm.PackedLayout.__init__
    try:
        _self_test()
    except Exception as exc:
        _orig_init = None
        _failure_reason = str(exc)
        _LOG.warning("h3_motion_context: self-test failed (%s), patch not applied. "
                     "Interior keyframe anchors unavailable.", exc)
        return False
    mm.PackedLayout.__init__ = _patched_init
    _applied = True
    _failure_reason = None
    _LOG.info("h3_motion_context: interior keyframe anchors enabled")
    return True


def is_applied():
    return _applied


def failure_reason():
    return _failure_reason
