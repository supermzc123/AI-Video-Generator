from __future__ import annotations

from types import SimpleNamespace

import pytest

from _minimax_h3_motion_director_testpkg.director.effective_refs import (
    SemanticReferenceError,
    compile_effective_references,
    compile_semantic_prompt,
    semantic_reference_token,
)


def _picture(asset_id: str):
    return SimpleNamespace(index=99, asset_id=asset_id, tensor=asset_id)


def _video(asset_id: str):
    return SimpleNamespace(index=99, asset_id=asset_id, tensor=asset_id)


def _audio(asset_id: str):
    return SimpleNamespace(index=99, asset_id=asset_id, audio={"waveform": asset_id})


def _compile(selected, *, local_pictures=(), local_videos=(), local_audios=()):
    common_pictures = [_picture(x) for x in "ABC"]
    common_videos = [_video("V-common")]
    common_audios = [_audio("A-common")]
    paired = [_audio("V-common")]
    return compile_effective_references(
        common_pictures=common_pictures,
        common_videos=common_videos,
        common_audios=common_audios,
        common_video_audios=paired,
        selected_common_asset_ids=set(selected),
        local_pictures=list(local_pictures),
        local_videos=list(local_videos),
        local_audios=list(local_audios),
        local_video_audios=[],
    )


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ({"A", "B", "C"}, ["A", "B", "C"]),
        (set(), []),
        ({"A", "B"}, ["A", "B"]),
        ({"A", "C"}, ["A", "C"]),
        ({"B", "C"}, ["B", "C"]),
    ],
)
def test_common_picture_subset_is_compiled_without_empty_slots(selected, expected):
    result = _compile(selected)
    assert [item.asset_id for item in result.pictures] == expected
    assert [item.index for item in result.pictures] == list(range(len(expected)))


def test_common_then_local_order_is_per_segment_and_local_does_not_inherit():
    seg1 = _compile({"A", "B", "C"}, local_pictures=[_picture("D")])
    seg2 = _compile(set())
    seg3 = _compile({"A", "B", "C"}, local_pictures=[_picture("E")])

    assert [x.asset_id for x in seg1.pictures] == ["A", "B", "C", "D"]
    assert [x.asset_id for x in seg2.pictures] == []
    assert [x.asset_id for x in seg3.pictures] == ["A", "B", "C", "E"]


def test_picture_tags_renumber_when_middle_common_asset_is_disabled():
    result = _compile({"A", "C"})
    assert result.tags[("picture", "A")] == "<Picture 1>"
    assert result.tags[("picture", "C")] == "<Picture 2>"


def test_video_soundtrack_precedes_standalone_audio_in_official_audio_order():
    result = _compile(
        {"V-common", "A-common"},
        local_videos=[_video("V-local")],
        local_audios=[_audio("A-local")],
    )

    assert [x.asset_id for x in result.videos] == ["V-common", "V-local"]
    assert [x.index for x in result.videos] == [0, 1]
    assert [x.asset_id for x in result.video_audios] == ["V-common"]
    assert [x.index for x in result.video_audios] == [0]
    assert [x.asset_id for x in result.audios] == ["A-common", "A-local"]
    assert [x.index for x in result.audios] == [0, 1]
    assert result.tags[("video", "V-common")] == "<Video 1>"
    assert result.tags[("video", "V-local")] == "<Video 2>"
    assert result.tags[("audio", "V-common")] == "<Audio 1>"
    assert result.tags[("audio", "A-common")] == "<Audio 2>"
    assert result.tags[("audio", "A-local")] == "<Audio 3>"


def test_semantic_prompt_compiles_asset_identity_to_current_official_tag():
    token = semantic_reference_token("picture", "C")
    before = _compile({"A", "B", "C"})
    after = _compile({"A", "C"})

    assert compile_semantic_prompt(token, before.tags) == "<Picture 3>"
    assert compile_semantic_prompt(token, after.tags) == "<Picture 2>"


def test_semantic_prompt_fails_loud_when_common_asset_is_disabled():
    token = semantic_reference_token("picture", "B")
    result = _compile({"A", "C"})
    with pytest.raises(SemanticReferenceError, match=r'Picture.*B.*disabled'):
        compile_semantic_prompt(
            f"keep {token}",
            result.tags,
            known_assets={("picture", "B"): 'Common Picture "B"'},
            segment_label="Segment 2",
        )
