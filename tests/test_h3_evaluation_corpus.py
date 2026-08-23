from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ai_video_generator.domain.h3_prompt import (
    H3AssetInput,
    H3AssetKind,
    H3AssetPromptRole,
    H3PromptRequest,
)
from ai_video_generator.llm.h3_prompt import route_h3_mode


def test_sanitized_h3_evaluation_corpus_has_30_routable_cases() -> None:
    path = Path(__file__).parent / "fixtures" / "h3_eval_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) >= 30
    assert len({case["id"] for case in cases}) == len(cases)
    modes = Counter(case["expected_mode"] for case in cases)
    assert all(modes[mode] >= 5 for mode in ("t2va", "i2va", "fl2va", "l2va", "ref2va"))

    for case in cases:
        ordinals = Counter()
        assets = []
        for index, raw in enumerate(case["assets"], start=1):
            kind = H3AssetKind(raw["kind"])
            ordinals[kind.value] += 1
            label_type = {"image": "Picture", "video": "Video", "audio": "Audio"}[
                kind.value
            ]
            assets.append(
                H3AssetInput(
                    asset_id=f"{case['id']}-asset-{index}",
                    label=f"<{label_type} {ordinals[kind.value]}>",
                    kind=kind,
                    role=H3AssetPromptRole(raw["role"]),
                    preservation="Preserve only the explicitly assigned reference attributes.",
                )
            )
        request = H3PromptRequest(
            operation_id=case["id"],
            segment_id=f"segment-{case['id']}",
            creative_brief=case["brief"],
            duration_seconds=4,
            assets=tuple(assets),
        )
        assert route_h3_mode(request).value == case["expected_mode"]
