import test from "node:test";
import assert from "node:assert/strict";

import {
    isExternalGroupPassthroughNode,
    resolveExternalGroupTerminal,
} from "../web/js/minimax_external_groups.mjs";

function graphFixture(nodes, links) {
    return {
        links,
        getNodeById(id) { return nodes[id] ?? null; },
    };
}

test("direct External Group link resolves to its source node", () => {
    const group = { id: 1, comfyClass: "MiniMaxH3I2VGroup" };
    const graph = graphFixture({ 1: group }, { 10: [10, 1, 0, 99, 0] });
    assert.equal(resolveExternalGroupTerminal(graph, 10)?.node, group);
});

test("one and two recognized Reroute nodes resolve to the original group", () => {
    const group = { id: 1, comfyClass: "MiniMaxH3R2VGroup" };
    const rerouteA = { id: 2, type: "Reroute", inputs: [{ link: 10 }] };
    const rerouteB = { id: 3, comfyClass: "Reroute (rgthree)", inputs: [{ link: 20 }] };
    const graph = graphFixture(
        { 1: group, 2: rerouteA, 3: rerouteB },
        {
            10: [10, 1, 0, 2, 0],
            20: [20, 2, 0, 3, 0],
            30: [30, 3, 0, 99, 0],
        },
    );
    assert.equal(resolveExternalGroupTerminal(graph, 20)?.node, group);
    assert.equal(resolveExternalGroupTerminal(graph, 30)?.node, group);
});

test("explicit virtual single-input passthrough resolves, ordinary semantic node does not", () => {
    const group = { id: 1, comfyClass: "MiniMaxH3I2VGroup" };
    const virtual = { id: 2, type: "Anything", isVirtualNode: true, inputs: [{ link: 10 }] };
    const semantic = { id: 3, type: "SemanticTransform", inputs: [{ link: 10 }] };
    const graph = graphFixture(
        { 1: group, 2: virtual, 3: semantic },
        {
            10: [10, 1, 0, 2, 0],
            20: [20, 2, 0, 99, 0],
            30: [30, 3, 0, 99, 0],
        },
    );
    assert.equal(isExternalGroupPassthroughNode(virtual), true);
    assert.equal(isExternalGroupPassthroughNode(semantic), false);
    assert.equal(resolveExternalGroupTerminal(graph, 20)?.node, group);
    assert.equal(resolveExternalGroupTerminal(graph, 30)?.node, semantic);
});

test("broken and cyclic Reroute chains fail closed", () => {
    const broken = { id: 2, type: "Reroute", inputs: [{ link: null }] };
    const cyclic = { id: 3, type: "Reroute", inputs: [{ link: 30 }] };
    const graph = graphFixture(
        { 2: broken, 3: cyclic },
        {
            20: [20, 2, 0, 99, 0],
            30: [30, 3, 0, 3, 0],
        },
    );
    assert.equal(resolveExternalGroupTerminal(graph, 20), null);
    assert.equal(resolveExternalGroupTerminal(graph, 30), null);
});
