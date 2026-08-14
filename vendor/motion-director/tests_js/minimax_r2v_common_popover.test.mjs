import test from "node:test";
import assert from "node:assert/strict";

let api = {};
try {
    api = await import("../web/js/minimax_r2v_common_popover.mjs");
} catch {
    // RED: implementation is added after these behavioral tests fail.
}

const {
    buildR2vCommonSections,
    createR2vCommonPopover,
    firstFreeReferenceSlot,
    renderR2vCommonSections,
} = api;

class TestClassList {
    constructor(owner) { this.owner = owner; this.values = new Set(); }
    set(value) { this.values = new Set(String(value || "").split(/\s+/).filter(Boolean)); }
    add(...names) { names.forEach((name) => this.values.add(name)); }
    contains(name) { return this.values.has(name); }
    toggle(name, force) {
        const on = force === undefined ? !this.values.has(name) : !!force;
        if (on) this.values.add(name);
        else this.values.delete(name);
        return on;
    }
}

class TestElement {
    constructor(tagName) {
        this.tagName = String(tagName || "div").toUpperCase();
        this.children = [];
        this.parentElement = null;
        this.dataset = {};
        this.style = {};
        this.attributes = new Map();
        this.classList = new TestClassList(this);
        this._className = "";
        this.hidden = false;
        this.textContent = "";
    }
    set className(value) { this._className = String(value || ""); this.classList.set(value); }
    get className() { return this._className; }
    append(...children) { children.forEach((child) => this.appendChild(child)); }
    appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
    replaceChildren(...children) {
        this.children.forEach((child) => { child.parentElement = null; });
        this.children = [];
        this.append(...children);
    }
    remove() {
        if (!this.parentElement) return;
        const siblings = this.parentElement.children;
        siblings.splice(siblings.indexOf(this), 1);
        this.parentElement = null;
    }
    setAttribute(name, value) {
        this.attributes.set(name, String(value));
        if (name.startsWith("data-")) {
            const key = name.slice(5).replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
            this.dataset[key] = String(value);
        }
    }
    contains(target) {
        if (target === this) return true;
        return this.children.some((child) => child.contains?.(target));
    }
    getBoundingClientRect() {
        return this._rect || { left: 100, right: 180, top: 100, bottom: 130, width: 80, height: 30 };
    }
}

class TestEventTarget {
    constructor() { this.listeners = new Map(); }
    addEventListener(type, handler) {
        const handlers = this.listeners.get(type) || [];
        handlers.push(handler);
        this.listeners.set(type, handlers);
    }
    removeEventListener(type, handler) {
        this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== handler));
    }
    dispatch(type, event) {
        for (const handler of this.listeners.get(type) || []) handler(event);
    }
}

class TestDocument extends TestEventTarget {
    createElement(tagName) { return new TestElement(tagName); }
}

class TestWindow extends TestEventTarget {
    constructor() {
        super();
        this.innerWidth = 1200;
        this.innerHeight = 800;
    }
    requestAnimationFrame(callback) { callback(); return 1; }
}

test("Common section model contains existing assets plus only the first-free Add slot", () => {
    assert.equal(typeof buildR2vCommonSections, "function");
    assert.equal(firstFreeReferenceSlot([{ index: 0 }, { index: 2 }], 9), 1);
    const sections = buildR2vCommonSections({
        refs: [
            { index: 0, imageFile: "A.png", assetId: "A" },
            { index: 2, imageFile: "C.png", assetId: "C" },
            { index: 4, imageFile: "E.png", assetId: "E" },
        ],
        refVideos: [],
        refAudios: [],
    });
    const [pictures, videos, audios] = sections;

    assert.deepEqual(pictures.items.map((item) => item.slot), [0, 2, 4]);
    assert.equal(pictures.addSlot, 1);
    assert.equal(pictures.items.length, 3);
    assert.equal(videos.items.length, 0);
    assert.equal(videos.addSlot, 0);
    assert.equal(audios.items.length, 0);
    assert.equal(audios.addSlot, 0);
    assert.equal(sections.reduce((count, section) => count + section.items.length, 0), 3);
});

test("full Common section has no Add slot and never synthesizes empty asset cards", () => {
    assert.equal(typeof buildR2vCommonSections, "function");
    const refs = Array.from({ length: 9 }, (_, index) => ({
        index,
        imageFile: String(index) + ".png",
        assetId: "p" + String(index),
    }));
    const pictures = buildR2vCommonSections({ refs })[0];
    assert.equal(pictures.items.length, 9);
    assert.equal(pictures.addSlot, null);
});

test("popover DOM renders A/B/C plus Add, and empty media kinds render only Add", () => {
    assert.equal(typeof renderR2vCommonSections, "function");
    const documentRef = new TestDocument();
    const body = documentRef.createElement("div");
    const sections = buildR2vCommonSections({
        refs: [
            { index: 0, imageFile: "A.png", assetId: "A" },
            { index: 1, imageFile: "B.png", assetId: "B" },
            { index: 2, imageFile: "C.png", assetId: "C" },
        ],
    });
    renderR2vCommonSections(body, sections, {
        documentRef,
        labels: {
            section: (kind) => kind,
            add: (kind) => "+ " + kind,
        },
        renderAsset: (_section, item, card) => { card.textContent = item.assetId; },
    });

    const grids = body.children.map((section) => section.children[1]);
    assert.deepEqual(grids[0].children.map((card) => card.textContent), ["A", "B", "C", "+ picture"]);
    assert.deepEqual(grids[1].children.map((card) => card.textContent), ["+ video"]);
    assert.deepEqual(grids[2].children.map((card) => card.textContent), ["+ audio"]);
    assert.equal(
        grids.flatMap((grid) => grid.children).filter((card) => card.classList.contains("bd-r2v-common-popover-asset")).length,
        3,
    );
});

test("Common manager mounts in the overlay layer and toggling never mutates the batch list", () => {
    assert.equal(typeof createR2vCommonPopover, "function");
    const documentRef = new TestDocument();
    const windowRef = new TestWindow();
    const overlayLayer = documentRef.createElement("div");
    const batchList = documentRef.createElement("div");
    const segment = documentRef.createElement("div");
    batchList.appendChild(segment);
    const anchor = documentRef.createElement("button");
    const openStates = [];

    const manager = createR2vCommonPopover({
        anchor,
        overlayLayer,
        documentRef,
        windowRef,
        onOpenChange: (open) => openStates.push(open),
    });
    assert.equal(manager.panel.parentElement, overlayLayer);
    assert.equal(manager.panel.parentElement === batchList, false);
    assert.equal(batchList.children.length, 1);
    assert.equal(batchList.children[0], segment);

    manager.toggle();
    assert.equal(manager.isOpen, true);
    assert.equal(manager.panel.hidden, false);
    manager.toggle();
    assert.equal(manager.isOpen, false);
    assert.equal(manager.panel.hidden, true);
    assert.deepEqual(openStates, [true, false]);
    assert.equal(batchList.children.length, 1);
});

test("outside pointer and Escape close the Common popover", () => {
    assert.equal(typeof createR2vCommonPopover, "function");
    const documentRef = new TestDocument();
    const windowRef = new TestWindow();
    const overlayLayer = documentRef.createElement("div");
    const anchor = documentRef.createElement("button");
    const manager = createR2vCommonPopover({ anchor, overlayLayer, documentRef, windowRef });

    manager.open();
    documentRef.dispatch("pointerdown", { target: {} });
    assert.equal(manager.isOpen, false);

    manager.open();
    let prevented = false;
    windowRef.dispatch("keydown", {
        key: "Escape",
        preventDefault() { prevented = true; },
        stopPropagation() {},
    });
    assert.equal(manager.isOpen, false);
    assert.equal(prevented, true);
});
