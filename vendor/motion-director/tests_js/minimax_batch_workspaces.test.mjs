import test from "node:test";
import assert from "node:assert/strict";

import {
    restoreBatchTaskWorkspace,
    stashBatchTaskWorkspace,
} from "../web/js/minimax_batch_workspaces.mjs";

test("R2V, I2V and their prompts keep separate prompt-batch workspaces", () => {
    const timeline = {
        global: { refs: [{ assetId: "GLOBAL" }] },
        r2vCommon: { refs: [{ assetId: "COMMON" }] },
    };
    stashBatchTaskWorkspace(timeline, "r2v", {
        segments: [{ prompt: "R2V AAA", refs: [{ assetId: "LOCAL" }] }],
        selectedIndex: 0,
    });
    stashBatchTaskWorkspace(timeline, "i2v", {
        segments: [{ prompt: "I2V BBB", refs: [] }],
        selectedIndex: 0,
    });
    assert.equal(restoreBatchTaskWorkspace(timeline, "r2v").segments[0].prompt, "R2V AAA");
    assert.equal(restoreBatchTaskWorkspace(timeline, "i2v").segments[0].prompt, "I2V BBB");
    assert.equal(timeline.r2vCommon.refs[0].assetId, "COMMON");
    assert.equal(timeline.global.refs[0].assetId, "GLOBAL");
});
