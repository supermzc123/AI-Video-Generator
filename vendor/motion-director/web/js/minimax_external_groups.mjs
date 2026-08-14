// Pure LiteGraph traversal helpers for External Groups wiring.

function graphLinkRecord(graph, linkId) {
    if (linkId == null || !graph?.links) return null;
    const links = graph.links;
    let link = links[linkId];
    if (!link && typeof links.find === "function") {
        link = links.find((entry) => entry && (entry.id === linkId || entry[0] === linkId));
    }
    if (!link) return null;
    return {
        originId: link.origin_id ?? link[1],
        originSlot: link.origin_slot ?? link[2],
    };
}

export function isExternalGroupPassthroughNode(node) {
    if (!node) return false;
    const cls = String(node.comfyClass || node.type || "");
    if (/reroute/i.test(cls)) return true;
    if (node.isVirtualNode === true) {
        const linked = (node.inputs || []).filter((input) => input?.link != null);
        return linked.length === 1;
    }
    return false;
}

export function passthroughUpstreamLinkId(node) {
    if (!isExternalGroupPassthroughNode(node)) return null;
    const linked = (node?.inputs || []).filter((input) => input?.link != null);
    return linked.length === 1 ? linked[0].link : null;
}

export function resolveExternalGroupTerminal(graph, linkId, maxDepth = 16) {
    let current = linkId;
    const visited = new Set();
    for (let depth = 0; depth <= maxDepth; depth += 1) {
        if (current == null || visited.has(current)) return null;
        visited.add(current);
        const record = graphLinkRecord(graph, current);
        if (!record) return null;
        const node = graph.getNodeById?.(record.originId);
        if (!node) return null;
        if (!isExternalGroupPassthroughNode(node)) {
            return { linkId: current, record, node, depth };
        }
        current = passthroughUpstreamLinkId(node);
    }
    return null;
}
