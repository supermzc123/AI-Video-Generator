from ai_video_generator.services.workspace_topology import normalize_workspace_topology


def test_normalize_workspace_removes_prompts_for_deleted_shots() -> None:
    payload = {
        "shots": [
            {
                "id": "shot-1",
                "durationSeconds": 20,
                "motionSegments": [
                    {"id": "shot-1.C01", "durationSeconds": 10, "summary": "first"},
                    {"id": "shot-1.C02", "durationSeconds": 10, "summary": "second"},
                ],
            }
        ],
        "prompts": {
            "h3Prompts": [
                {"shotId": "shot-1", "segmentId": "shot-1.C01", "prompt": "one"},
                {"shotId": "shot-1", "segmentId": "shot-1.C02", "prompt": "two"},
                {"shotId": "shot-3", "segmentId": "shot-3.C01", "prompt": "stale"},
            ]
        },
    }

    normalized = normalize_workspace_topology(payload)

    assert [item["segmentId"] for item in normalized["prompts"]["h3Prompts"]] == [
        "shot-1.C01",
        "shot-1.C02",
    ]
    assert len(payload["prompts"]["h3Prompts"]) == 3


def test_normalize_workspace_removes_prompts_for_replaced_automatic_segments() -> None:
    payload = {
        "shots": [{"id": "shot-1", "durationSeconds": 8, "motionSegments": []}],
        "prompts": {
            "h3Prompts": [
                {"shotId": "shot-1", "segmentId": "shot-1.C01", "prompt": "current"},
                {"shotId": "shot-1", "segmentId": "shot-1.C02", "prompt": "old split"},
            ]
        },
    }

    normalized = normalize_workspace_topology(payload)

    assert normalized["prompts"]["h3Prompts"] == [
        {"shotId": "shot-1", "segmentId": "shot-1.C01", "prompt": "current"}
    ]
