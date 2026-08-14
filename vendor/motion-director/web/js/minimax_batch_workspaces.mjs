/** Task-keyed prompt-batch workspace isolation. */

function clone(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
}

export function stashBatchTaskWorkspace(timeline, taskKey, state) {
    const key = String(taskKey || "").trim().toLowerCase();
    if (!timeline || !key || !Array.isArray(state?.segments)) return false;
    timeline.batchTaskWorkspaces = timeline.batchTaskWorkspaces || {};
    timeline.batchTaskWorkspaces[key] = {
        segments: clone(state.segments),
        selectedIndex: Number(state.selectedIndex || 0),
        runSelectEnabled: !!state.runSelectEnabled,
        runSelection: [...(state.runSelection || [])],
    };
    return true;
}

export function restoreBatchTaskWorkspace(timeline, taskKey) {
    const key = String(taskKey || "").trim().toLowerCase();
    const workspace = timeline?.batchTaskWorkspaces?.[key];
    if (!workspace?.segments?.length) return null;
    return clone(workspace);
}
