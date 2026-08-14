/** DOM helpers for R2V Common/Local reference authoring UI. */

function requireDom(card, documentRef) {
    if (!card?.appendChild || !documentRef?.createElement) {
        throw new Error("R2V reference UI requires a card and document.");
    }
}

export function formatR2vAssetStatusLabel(asset = {}, disabledLabel = "disabled") {
    const name = String(asset.name || asset.label || asset.assetId || "");
    const state = asset.status === "active" && asset.effectiveTag
        ? asset.effectiveTag
        : disabledLabel;
    return `${name} · ${state}`;
}

export function mountR2vMediaLayout(card, {
    documentRef = globalThis.document,
} = {}) {
    requireDom(card, documentRef);
    const body = documentRef.createElement("div");
    body.className = "bd-batch-r2v-body";
    const assets = documentRef.createElement("div");
    assets.className = "bd-batch-r2v-assets";
    body.appendChild(assets);

    const main = documentRef.createElement("div");
    main.className = "bd-batch-r2v-main";
    body.appendChild(main);
    card.appendChild(body);
    return { body, assets, main };
}

export function mountR2vCommonSelection(card, {
    documentRef = globalThis.document,
    assets = [],
    labels = {},
    onSelectAll,
    onSelectNone,
    onToggle,
} = {}) {
    requireDom(card, documentRef);
    const section = documentRef.createElement("div");
    section.className = "bd-r2v-common-select";
    const head = documentRef.createElement("div");
    head.className = "bd-r2v-common-select-head";
    const title = documentRef.createElement("span");
    title.textContent = String(labels.title || "");
    head.appendChild(title);

    const actions = documentRef.createElement("div");
    actions.className = "bd-r2v-common-actions";
    for (const [action, text, callback] of [
        ["r2v-common-select-all", labels.selectAll, onSelectAll],
        ["r2v-common-select-none", labels.selectNone, onSelectNone],
    ]) {
        const button = documentRef.createElement("button");
        button.type = "button";
        button.setAttribute("data-a", action);
        button.textContent = String(text || "");
        button.onclick = (event) => {
            event.preventDefault?.();
            event.stopPropagation?.();
            callback?.();
        };
        actions.appendChild(button);
    }
    head.appendChild(actions);
    section.appendChild(head);

    const items = documentRef.createElement("div");
    items.className = "bd-r2v-common-items";
    if (!assets.length) {
        const empty = documentRef.createElement("span");
        empty.className = "bd-r2v-common-item";
        empty.textContent = String(labels.empty || "");
        items.appendChild(empty);
    }
    for (const asset of assets) {
        const label = documentRef.createElement("label");
        label.className = "bd-r2v-common-item";
        const checkbox = documentRef.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = asset.status === "active";
        checkbox.setAttribute("data-asset-id", String(asset.assetId || ""));
        checkbox.onchange = (event) => {
            event.stopPropagation?.();
            onToggle?.(asset.assetId, checkbox.checked);
        };
        const text = documentRef.createElement("span");
        text.setAttribute("data-r", "r2v-common-asset-label");
        const kind = labels.kind?.(asset.kind) || asset.kind;
        text.textContent = `${kind}: ${formatR2vAssetStatusLabel(asset, labels.disabled)}`;
        text.title = text.textContent;
        label.append(checkbox, text);
        items.appendChild(label);
    }
    section.appendChild(items);
    card.appendChild(section);
    return section;
}
