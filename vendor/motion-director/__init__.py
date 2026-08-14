# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.
# Portions derived from ComfyUI-H3-Motion-Context
# Copyright (C) 2026 NikoDemon80 and contributors
# Modified for MiniMax H3 Motion Director, 2026-08-09
# Licensed under GNU GPL v3.0. See LICENSE and NOTICE.

"""MiniMax H3 Motion Director — multi-segment Director with integrated Motion/Audio Context."""

from __future__ import annotations

import logging


# Pytest may inspect a custom-node repository's root ``__init__.py`` as a plain
# file named ``__init__`` when the checkout directory contains hyphens.  Such a
# module has no package context, so relative imports cannot work.  ComfyUI and
# the real import tests always load this file as a package; keep the plain-file
# probe inert while preserving normal runtime behavior.
if not __package__:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    WEB_DIRECTORY = "./web/js"
else:
    from .patches import apply_motion_context_patches, motion_context_patch_status

    _log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director")

    # Apply and self-test before any graph can execute. Failure is retained and
    # reported by the Director instead of silently generating misaligned layouts.
    _motion_patch_ready = apply_motion_context_patches()
    if not _motion_patch_ready:
        _log.error("Motion Context disabled: %s", motion_context_patch_status()[1])

    from .nodes.conditioning import (  # noqa: E402
        MiniMaxH3MotionDirectorConditioning,
        MiniMaxH3MotionDirectorPlannerConditioning,
    )
    from .nodes.director import MiniMaxH3MotionDirector  # noqa: E402
    from .nodes.director_groups import (  # noqa: E402
        MiniMaxH3MotionDirectorGroupImageToVideo,
        MiniMaxH3MotionDirectorGroupReferenceToVideo,
        MiniMaxH3MotionDirectorGroupsCombine,
    )

    NODE_CLASS_MAPPINGS = {
        "MiniMaxH3MotionDirector": MiniMaxH3MotionDirector,
        "MiniMaxH3MotionDirectorConditioning": MiniMaxH3MotionDirectorConditioning,
        "MiniMaxH3MotionDirectorPlannerConditioning": MiniMaxH3MotionDirectorPlannerConditioning,
        "MiniMaxH3MotionDirectorGroupImageToVideo": MiniMaxH3MotionDirectorGroupImageToVideo,
        "MiniMaxH3MotionDirectorGroupReferenceToVideo": MiniMaxH3MotionDirectorGroupReferenceToVideo,
        "MiniMaxH3MotionDirectorGroupsCombine": MiniMaxH3MotionDirectorGroupsCombine,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "MiniMaxH3MotionDirector": "MiniMax H3 Motion Director",
        "MiniMaxH3MotionDirectorConditioning": "MiniMax H3 Motion Director Conditioning",
        "MiniMaxH3MotionDirectorPlannerConditioning": "MiniMax H3 Motion Director Planner Conditioning",
        "MiniMaxH3MotionDirectorGroupImageToVideo": "MiniMax H3 Motion Director Group (Image to Video)",
        "MiniMaxH3MotionDirectorGroupReferenceToVideo": "MiniMax H3 Motion Director Group (Reference to Video)",
        "MiniMaxH3MotionDirectorGroupsCombine": "MiniMax H3 Motion Director Groups Combine",
    }

    WEB_DIRECTORY = "./web/js"

    try:
        from .director.http_routes import register_routes as _register_director_routes

        if not _register_director_routes():
            _log.warning(
                "Motion Director HTTP routes deferred because PromptServer is not ready."
            )
    except Exception as _route_exc:  # pragma: no cover - ComfyUI startup only
        _log.warning("Motion Director HTTP routes failed to load: %s", _route_exc)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
