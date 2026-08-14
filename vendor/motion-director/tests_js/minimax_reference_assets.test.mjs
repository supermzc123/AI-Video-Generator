import test from "node:test";
import assert from "node:assert/strict";

import {
    allKnownReferenceAssets,
    authoringReferenceAssets,
    compileSemanticPrompt,
    effectiveReferenceAssets,
    ensureR2vReferenceAssetSchema,
    ensureReferenceAssetSchema,
    hydrateOfficialReferenceTags,
    referenceAssetStates,
    moveReferenceAssetSlot,
    replaceReferenceAssetAtSlot,
    setCommonAssetEnabled,
    semanticReferenceToken,
} from "../web/js/minimax_reference_assets.mjs";


function fixture() {
    return {
        global: {
            prompt: "COMMON",
        },
        r2vCommon: {
            refs: [
                { index: 0, assetId: "A", imageFile: "a.png" },
                { index: 1, assetId: "B", imageFile: "b.png" },
                { index: 2, assetId: "C", imageFile: "c.png" },
            ],
            refVideos: [
                { index: 0, assetId: "V", videoFile: "v.mp4", pairedAudioFile: "v.wav" },
            ],
            refAudios: [{ index: 0, assetId: "AU", audioFile: "a.wav" }],
        },
        segments: [{
            id: "s1",
            useCommonAssets: true,
            excludedCommonAssetIds: ["B"],
            refs: [{ index: 0, assetId: "D", imageFile: "d.png" }],
            refVideos: [{ index: 0, assetId: "VL", videoFile: "vl.mp4" }],
            refAudios: [{ index: 0, assetId: "AL", audioFile: "al.wav" }],
        }],
    };
}


test("new segment defaults to every existing Common asset independently of Common Prompt", () => {
    const timeline = fixture();
    timeline.segments.push({ id: "s2", refs: [], refVideos: [], refAudios: [] });
    ensureR2vReferenceAssetSchema(timeline, () => "generated");
    assert.equal(timeline.segments[1].useCommonAssets, true);
    assert.deepEqual(timeline.segments[1].excludedCommonAssetIds, []);
    timeline.r2vCommon.refs.push({ index: 3, assetId: "D2", imageFile: "d2.png" });
    assert.deepEqual(
        effectiveReferenceAssets(timeline.r2vCommon, timeline.segments[1])
            .filter((x) => x.kind === "picture").map((x) => x.assetId),
        ["A", "B", "C", "D2"],
    );
    timeline.segments[1].useCommonAssets = false;
    assert.deepEqual(effectiveReferenceAssets(timeline.r2vCommon, timeline.segments[1]), []);
});

test("enabling one Common asset after select-none does not enable the whole pool", () => {
    const segment = { useCommonAssets: false, excludedCommonAssetIds: [] };
    setCommonAssetEnabled(segment, ["A", "B", "C"], "B", true);
    assert.equal(segment.useCommonAssets, true);
    assert.deepEqual(segment.excludedCommonAssetIds.sort(), ["A", "C"]);
});

test("authoring picture tags stay Common-first plus Local regardless of selection", () => {
    const timeline = fixture();
    const expected = [
        ["A", "<Picture 1>"],
        ["B", "<Picture 2>"],
        ["C", "<Picture 3>"],
        ["D", "<Picture 4>"],
    ];
    for (const segment of [
        { ...timeline.segments[0], excludedCommonAssetIds: [] },
        { ...timeline.segments[0], excludedCommonAssetIds: ["A", "B"] },
    ]) {
        assert.deepEqual(
            authoringReferenceAssets(timeline.r2vCommon, segment)
                .filter((asset) => asset.kind === "picture")
                .map((asset) => [asset.assetId, asset.authoringTag]),
            expected,
        );
    }
});

test("authoring video and audio tags include disabled Common assets in official soundtrack order", () => {
    const timeline = fixture();
    timeline.segments[0].refVideos[0].pairedAudioFile = "vl.wav";
    const assets = authoringReferenceAssets(timeline.r2vCommon, {
        ...timeline.segments[0],
        excludedCommonAssetIds: ["V", "AU"],
    });
    assert.deepEqual(
        assets.filter((asset) => asset.kind === "video")
            .map((asset) => [asset.assetId, asset.authoringTag]),
        [["V", "<Video 1>"], ["VL", "<Video 2>"]],
    );
    assert.deepEqual(
        assets.filter((asset) => asset.kind === "audio")
            .map((asset) => [asset.assetId, asset.authoringTag, asset.paired]),
        [
            ["V", "<Audio 1>", true],
            ["VL", "<Audio 2>", true],
            ["AU", "<Audio 3>", false],
            ["AL", "<Audio 4>", false],
        ],
    );
});


test("effective references are dense Common-first then Local with paired soundtrack audio order", () => {
    const timeline = fixture();
    ensureReferenceAssetSchema(timeline);
    const assets = effectiveReferenceAssets(timeline.r2vCommon, timeline.segments[0]);

    assert.deepEqual(
        assets.filter((x) => x.kind === "picture").map((x) => [x.assetId, x.effectiveTag]),
        [["A", "<Picture 1>"], ["C", "<Picture 2>"], ["D", "<Picture 3>"]],
    );
    assert.deepEqual(
        assets.filter((x) => x.kind === "video").map((x) => [x.assetId, x.effectiveTag]),
        [["V", "<Video 1>"], ["VL", "<Video 2>"]],
    );
    assert.deepEqual(
        assets.filter((x) => x.kind === "audio").map((x) => [x.assetId, x.effectiveTag, x.paired]),
        [["V", "<Audio 1>", true], ["AU", "<Audio 2>", false], ["AL", "<Audio 3>", false]],
    );
});

test("effective picture tags are dense for every selected Common subset and Local combination", () => {
    const timeline = fixture();
    const commonIds = ["A", "B", "C"];
    const commonOnly = { ...timeline.segments[0], refs: [], refVideos: [], refAudios: [] };
    const cases = [
        ["A only", ["A"], [], [["A", "<Picture 1>"]]],
        ["B only", ["B"], [], [["B", "<Picture 1>"]]],
        ["C only", ["C"], [], [["C", "<Picture 1>"]]],
        ["A+C", ["A", "C"], [], [["A", "<Picture 1>"], ["C", "<Picture 2>"]]],
        ["B+C", ["B", "C"], [], [["B", "<Picture 1>"], ["C", "<Picture 2>"]]],
        ["ABC", ["A", "B", "C"], [], [["A", "<Picture 1>"], ["B", "<Picture 2>"], ["C", "<Picture 3>"]]],
        ["ABC+D", ["A", "B", "C"], timeline.segments[0].refs, [["A", "<Picture 1>"], ["B", "<Picture 2>"], ["C", "<Picture 3>"], ["D", "<Picture 4>"]]],
        ["C+D", ["C"], timeline.segments[0].refs, [["C", "<Picture 1>"], ["D", "<Picture 2>"]]],
        ["D only", [], timeline.segments[0].refs, [["D", "<Picture 1>"]]],
    ];
    for (const [name, selected, refs, expected] of cases) {
        const selectedSet = new Set(selected);
        const segment = {
            ...commonOnly,
            refs,
            useCommonAssets: selected.length > 0,
            excludedCommonAssetIds: commonIds.filter((assetId) => !selectedSet.has(assetId)),
        };
        assert.deepEqual(
            effectiveReferenceAssets(timeline.r2vCommon, segment)
                .filter((asset) => asset.kind === "picture")
                .map((asset) => [asset.assetId, asset.effectiveTag]),
            expected,
            name,
        );
    }
});


test("semantic asset identity survives Common subset renumbering", () => {
    const timeline = fixture();
    const token = semanticReferenceToken("picture", "C");
    let assets = effectiveReferenceAssets(timeline.r2vCommon, {
        ...timeline.segments[0], excludedCommonAssetIds: [], refs: [], refVideos: [], refAudios: [],
    });
    assert.equal(compileSemanticPrompt(token, assets), "<Picture 3>");
    assets = effectiveReferenceAssets(timeline.r2vCommon, {
        ...timeline.segments[0], excludedCommonAssetIds: ["B"], refs: [], refVideos: [], refAudios: [],
    });
    assert.equal(compileSemanticPrompt(token, assets), "<Picture 2>");
});

test("semantic prompt compilation treats effectiveTag as the runtime source of truth", () => {
    assert.equal(compileSemanticPrompt(
        semanticReferenceToken("picture", "C"),
        [{
            kind: "picture",
            assetId: "C",
            status: "active",
            effectiveTag: "<Picture 1>",
            officialTag: "<Picture 9>",
        }],
    ), "<Picture 1>");
});

test("raw official picture tags hydrate through authoring tags even when Common is disabled", () => {
    const timeline = fixture();
    const segment = {
        ...timeline.segments[0],
        useCommonAssets: false,
        excludedCommonAssetIds: [],
    };
    const states = referenceAssetStates(timeline.r2vCommon, segment);
    assert.deepEqual(
        states.filter((asset) => asset.kind === "picture")
            .map((asset) => [asset.assetId, asset.authoringTag, asset.effectiveTag, asset.status]),
        [
            ["A", "<Picture 1>", "", "disabled"],
            ["B", "<Picture 2>", "", "disabled"],
            ["C", "<Picture 3>", "", "disabled"],
            ["D", "<Picture 4>", "<Picture 1>", "active"],
        ],
    );
    assert.equal(
        hydrateOfficialReferenceTags(
            "<Picture 1> <Picture 2> <Picture 3> <Picture 4>",
            states,
        ),
        ["A", "B", "C", "D"]
            .map((assetId) => semanticReferenceToken("picture", assetId))
            .join(" "),
    );
});

test("raw Video and Audio tags hydrate through their authoring mappings", () => {
    const timeline = fixture();
    const states = referenceAssetStates(timeline.r2vCommon, {
        ...timeline.segments[0],
        useCommonAssets: false,
        excludedCommonAssetIds: [],
    });
    assert.equal(
        hydrateOfficialReferenceTags(
            "<Video 1> <Video 2> | <Audio 1> <Audio 2> <Audio 3>",
            states,
        ),
        [
            semanticReferenceToken("video", "V"),
            semanticReferenceToken("video", "VL"),
            "|",
            semanticReferenceToken("audio", "V"),
            semanticReferenceToken("audio", "AU"),
            semanticReferenceToken("audio", "AL"),
        ].join(" "),
    );
});

test("raw Picture 4 hydrates only after Local D is uploaded", () => {
    const timeline = fixture();
    const segment = {
        ...timeline.segments[0],
        useCommonAssets: false,
        excludedCommonAssetIds: [],
        refs: [],
        refVideos: [],
        refAudios: [],
    };
    const raw = "Use <Picture 4> as the hero";
    assert.equal(
        hydrateOfficialReferenceTags(raw, referenceAssetStates(timeline.r2vCommon, segment)),
        raw,
    );
    segment.refs = [{ index: 0, assetId: "D", imageFile: "d.png" }];
    assert.equal(
        hydrateOfficialReferenceTags(raw, referenceAssetStates(timeline.r2vCommon, segment)),
        `Use ${semanticReferenceToken("picture", "D")} as the hero`,
    );
});

test("semantic C identity keeps authoring tag while effective chip tag follows selection", () => {
    const timeline = fixture();
    const cases = [
        [[], "<Picture 3>", "active"],
        [["B"], "<Picture 2>", "active"],
        [["A", "B"], "<Picture 1>", "active"],
        [["A", "B", "C"], "", "disabled"],
        [["A", "B"], "<Picture 1>", "active"],
    ];
    for (const [excludedCommonAssetIds, effectiveTag, status] of cases) {
        const states = referenceAssetStates(timeline.r2vCommon, {
            ...timeline.segments[0],
            refs: [],
            refVideos: [],
            refAudios: [],
            excludedCommonAssetIds,
        });
        const c = states.find((asset) => asset.kind === "picture" && asset.assetId === "C");
        assert.equal(c.assetId, "C");
        assert.equal(c.authoringTag, "<Picture 3>");
        assert.equal(c.effectiveTag, effectiveTag);
        assert.equal(c.status, status);
    }
});


test("disabled Common assets stay excluded from effective compilation and prompt preflight fails", () => {
    const timeline = fixture();
    const assets = effectiveReferenceAssets(timeline.r2vCommon, {
        ...timeline.segments[0], excludedCommonAssetIds: ["B", "C", "V", "AU"], refs: [], refVideos: [], refAudios: [],
    });
    assert.deepEqual(assets.map((x) => x.assetId), ["A"]);
    assert.throws(
        () => compileSemanticPrompt(semanticReferenceToken("picture", "B"), assets, {
            knownAssets: [{ kind: "picture", assetId: "B", label: "B" }],
            segmentLabel: "Segment 2",
        }),
        /Segment 2.*B.*disabled/,
    );
});


test("asset IDs persist through JSON workflow save and reopen", () => {
    const timeline = fixture();
    const reopened = JSON.parse(JSON.stringify(timeline));
    ensureReferenceAssetSchema(reopened);
    assert.equal(reopened.r2vCommon.refs[2].assetId, "C");
    assert.equal(reopened.segments[0].refs[0].assetId, "D");
    assert.deepEqual(reopened.segments[0].excludedCommonAssetIds, ["B"]);
});

test("legacy allow-list migrates once to exclusions and Common never copies into Local", () => {
    const timeline = {
        global: { refs: [{ index: 0, assetId: "G", imageFile: "global.png" }] },
        r2vCommon: { refs: [{ index: 0, assetId: "A", imageFile: "a.png" }, { index: 1, assetId: "B", imageFile: "b.png" }] },
        segments: [{ commonAssetIds: ["A"], refs: [] }],
    };
    ensureR2vReferenceAssetSchema(timeline);
    assert.deepEqual(timeline.segments[0].excludedCommonAssetIds, ["B"]);
    assert.equal("commonAssetIds" in timeline.segments[0], false);
    assert.deepEqual(timeline.segments[0].refs, []);
    assert.equal(timeline.global.refs[0].assetId, "G");
});

test("snake-case legacy R2V state migrates into the independent Common container", () => {
    const timeline = {
        global: {
            refs: [{ index: 0, asset_id: "A", imageFile: "a.png" }],
            ref_videos: [{ index: 0, asset_id: "V", videoFile: "v.mp4" }],
            ref_audios: [{ index: 0, asset_id: "AU", audioFile: "a.wav" }],
        },
        segments: [{
            common_asset_ids: ["A", "AU"],
            use_common_prompt: false,
            refs: [],
        }],
    };
    ensureR2vReferenceAssetSchema(timeline);
    assert.deepEqual(timeline.r2vCommon.refs.map((x) => x.assetId), ["A"]);
    assert.deepEqual(timeline.r2vCommon.refVideos.map((x) => x.assetId), ["V"]);
    assert.deepEqual(timeline.r2vCommon.refAudios.map((x) => x.assetId), ["AU"]);
    assert.deepEqual(timeline.segments[0].excludedCommonAssetIds, ["V"]);
    assert.deepEqual(timeline.global.refs, []);
    assert.deepEqual(timeline.global.refVideos, []);
    assert.deepEqual(timeline.global.refAudios, []);
    assert.equal("ref_videos" in timeline.global, false);
    assert.equal("ref_audios" in timeline.global, false);
});

test("snake-case new Common container normalizes without touching global assets", () => {
    const timeline = {
        global: { refs: [{ index: 0, assetId: "G", imageFile: "g.png" }] },
        r2v_common: { refs: [{ index: 0, assetId: "C", imageFile: "c.png" }] },
        segments: [{ use_common_assets: false, excluded_common_asset_ids: ["C"] }],
    };
    ensureR2vReferenceAssetSchema(timeline);
    assert.equal(timeline.r2vCommon.refs[0].assetId, "C");
    assert.equal(timeline.global.refs[0].assetId, "G");
    assert.equal(timeline.segments[0].useCommonAssets, false);
    assert.deepEqual(timeline.segments[0].excludedCommonAssetIds, ["C"]);
    assert.equal("r2v_common" in timeline, false);
});

test("active, disabled and missing semantic assets remain distinct", () => {
    const timeline = fixture();
    ensureR2vReferenceAssetSchema(timeline);
    const states = referenceAssetStates(timeline.r2vCommon, timeline.segments[0]);
    assert.equal(states.find((x) => x.assetId === "A").status, "active");
    assert.equal(states.find((x) => x.assetId === "B").status, "disabled");
    assert.throws(() => compileSemanticPrompt(semanticReferenceToken("picture", "B"), states, {
        knownAssets: states,
        segmentLabel: "Segment 1",
    }), /disabled/i);
    const known = allKnownReferenceAssets(timeline.r2vCommon, timeline.segments[0]);
    assert.throws(() => compileSemanticPrompt(semanticReferenceToken("picture", "MISSING"), states, {
        knownAssets: known,
        segmentLabel: "Segment 1",
    }), /unknown.*MISSING/i);
});

test("replace and move preserve stable asset IDs; delete leaves semantic prompt identity missing", () => {
    const container = {
        refs: [
            { index: 0, assetId: "stable-A", imageFile: "old.png" },
            { index: 1, assetId: "stable-B", imageFile: "b.png" },
        ],
    };
    replaceReferenceAssetAtSlot(container, "refs", 0, { imageFile: "new.png" });
    assert.equal(container.refs.find((x) => x.index === 0).assetId, "stable-A");
    assert.equal(moveReferenceAssetSlot(container, "refs", 0, 1), true);
    assert.equal(container.refs.find((x) => x.index === 1).assetId, "stable-A");
    assert.equal(container.refs.find((x) => x.index === 0).assetId, "stable-B");
    const prompt = semanticReferenceToken("picture", "stable-A");
    container.refs = container.refs.filter((x) => x.assetId !== "stable-A");
    assert.equal(prompt, "{{mmx-ref:picture:stable-A}}");
    assert.throws(() => compileSemanticPrompt(prompt, [], { knownAssets: [] }), /unknown.*stable-A/i);
});
