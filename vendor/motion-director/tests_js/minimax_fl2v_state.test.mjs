import test from "node:test";
import assert from "node:assert/strict";

import { resolveFl2vEndpointState } from "../web/js/minimax_fl2v_state.mjs";

test("FL2V end-only state never mirrors image1 into image0", () => {
    const end = { imageFile: "end.png", width: 640, height: 864 };
    const state = resolveFl2vEndpointState({ startImage: null, endImage: end });
    assert.equal(state.hasStart, false);
    assert.equal(state.hasEnd, true);
    assert.equal(state.endOnly, true);
    assert.equal(state.startFile, "");
    assert.equal(state.endFile, "end.png");
    assert.equal(state.badgeKey, "fl2v.badge.endOnly");
});

test("FL2V start-only and start+end states keep independent endpoints", () => {
    const start = { imageFile: "start.png" };
    const end = { imageFile: "end.png" };
    assert.deepEqual(
        resolveFl2vEndpointState({ startImage: start, endImage: null }),
        {
            hasStart: true,
            hasEnd: false,
            endOnly: false,
            startFile: "start.png",
            endFile: "",
            badgeKey: "fl2v.badge.i2v",
        },
    );
    assert.equal(
        resolveFl2vEndpointState({ startImage: start, endImage: end }).badgeKey,
        "fl2v.badge.startEnd",
    );
});
