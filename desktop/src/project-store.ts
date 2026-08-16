import type { PipelineStageId, ProjectDraft } from "./types";

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function positiveNumber(...values: unknown[]): number | null {
  const found = values.find((value) => typeof value === "number" && Number.isFinite(value) && value > 0);
  return typeof found === "number" ? found : null;
}

function text(...values: unknown[]): string {
  const found = values.find((value) => typeof value === "string");
  return typeof found === "string" ? found : "";
}

function defaultAssetResolution(kind: ProjectDraft["assetPlans"][number]["kind"]): [number, number] {
  if (kind === "character") return [1024, 1600];
  if (kind === "scene") return [1600, 1024];
  return [1280, 1280];
}

function imageDimension(value: unknown, fallback: number): number {
  return typeof value === "number"
    && Number.isInteger(value)
    && value >= 64
    && value <= 4096
    && value % 8 === 0
    ? value
    : fallback;
}

export function newProject(): ProjectDraft {
  return {
    projectId: crypto.randomUUID(),
    revision: 1,
    name: "未命名项目",
    width: 1024,
    height: 608,
    fps: 24,
    targetDurationSeconds: 60,
    audioPolicy: "h3_native",
    externalAudioAssetId: null,
    executionMode: "guided",
    paused: false,
    reviewPolicy: {
      configuredMode: "human_ai",
      effectiveMode: "human_ai",
      humanTimeoutSeconds: 600,
      aiTakeoverAt: null,
    },
    timeBudgetSeconds: null,
    activeStage: "config",
    stageApprovals: {},
    idea: {
      concept: "",
      conceptDocument: { nodes: [{ type: "text", text: "" }] },
      genre: "剧情短片",
      visualStyle: "电影感写实",
      audience: "",
    },
    outline: [],
    shots: [],
    assetPlans: [],
    assets: [],
    referenceAssetMode: "planned",
    prompts: { imagePrompts: [], h3Prompts: [], generatedAt: null },
    reviewNotes: "",
    postProcessing: {
      normalizeVideo: true,
      outputWidth: 1024,
      outputHeight: 608,
      crf: 18,
      normalizeAudio: true,
      seedvr: { enabled: false, profileId: null, profileRevision: null, modelId: null, modelSha256: null, vaeId: null, vaeSha256: null, steps: 4, processingWidth: 1280, processingHeight: 720, previewSeconds: 1, previewApproved: false },
      rife: { enabled: false, profileId: null, profileRevision: null, modelId: null, modelSha256: null, targetFps: 48 },
      whisper: { enabled: false, profileId: null, profileRevision: null, modelId: null, modelSha256: null, device: "auto", precision: "auto", language: "zh", burnIn: false },
    },
    updatedAt: new Date().toISOString(),
  };
}

export function hydrateProject(value: Partial<ProjectDraft>): ProjectDraft {
  const defaults = newProject();
  const raw = value as Record<string, unknown>;
  const rawOutline = Array.isArray(raw.outline) ? raw.outline : [];
  const rawShots = Array.isArray(raw.shots) ? raw.shots : [];
  const rawAssets = Array.isArray(raw.assets) ? raw.assets : [];
  const rawAssetPlans = Array.isArray(raw.assetPlans) ? raw.assetPlans : [];
  const rawPrompts = record(raw.prompts);
  const rawImagePrompts = Array.isArray(rawPrompts.imagePrompts) ? rawPrompts.imagePrompts : [];
  const rawH3Prompts = Array.isArray(rawPrompts.h3Prompts) ? rawPrompts.h3Prompts : [];
  const idea = record(raw.idea);
  const concept = text(idea.concept);
  const conceptDocument = record(idea.conceptDocument);
  const conceptNodes = Array.isArray(conceptDocument.nodes) ? conceptDocument.nodes : [];
  const hydratedConceptNodes: ProjectDraft["idea"]["conceptDocument"]["nodes"] = [];
  for (const node of conceptNodes) {
    const item = record(node);
    if (item.type === "asset_mention" && text(item.assetId, item.asset_id)) {
      hydratedConceptNodes.push({ type: "asset_mention", assetId: text(item.assetId, item.asset_id), displayName: text(item.displayName, item.display_name) });
    } else if (item.type === "text") {
      hydratedConceptNodes.push({ type: "text", text: text(item.text) });
    }
  }
  const hydrated: ProjectDraft = {
    ...defaults,
    ...value,
    idea: {
      ...defaults.idea,
      ...value.idea,
      concept,
      conceptDocument: {
        nodes: hydratedConceptNodes.length ? hydratedConceptNodes : [{ type: "text" as const, text: concept }],
      },
    },
    stageApprovals: { ...value.stageApprovals },
    outline: rawOutline.map<ProjectDraft["outline"][number]>((item, index) => {
      const beat = record(item);
      return {
        id: text(beat.id) || `beat-${crypto.randomUUID()}`,
        title: text(beat.title) || `段落 ${index + 1}`,
        summary: text(beat.summary, beat.description),
        durationSeconds: positiveNumber(beat.durationSeconds, beat.targetDurationSeconds) ?? 1,
      };
    }),
    shots: rawShots.map<ProjectDraft["shots"][number]>((item, index) => {
      const shot = record(item);
      const harnessRevision = positiveNumber(shot.harnessRevision);
      return {
        id: text(shot.id) || `shot-${crypto.randomUUID()}`,
        title: text(shot.title) || `镜头 ${index + 1}`,
        summary: text(shot.summary, shot.description),
        camera: text(shot.camera) || "固定机位",
        seed: typeof shot.seed === "number" && Number.isInteger(shot.seed) && shot.seed >= 0 ? shot.seed : 0,
        durationSeconds: positiveNumber(shot.durationSeconds, shot.targetDurationSeconds) ?? 1,
        locked: typeof shot.locked === "boolean" ? shot.locked : false,
      };
    }),
    referenceAssetMode: raw.referenceAssetMode === "none" ? "none" : "planned",
    assets: rawAssets.map<ProjectDraft["assets"][number]>((item, index) => {
      const asset = record(item);
      const kind = ["character", "scene", "prop", "style"].includes(text(asset.kind))
        ? text(asset.kind) as ProjectDraft["assets"][number]["kind"]
        : "character";
      const scope = text(asset.scope) === "shot" ? "shot" : "public";
      return {
        id: text(asset.id) || `asset-${crypto.randomUUID()}`,
        name: text(asset.name) || `素材 ${index + 1}`,
        originalFileName: text(asset.originalFileName, asset.original_file_name, asset.name),
        sha256: text(asset.sha256) || null,
        mimeType: text(asset.mimeType, asset.mime_type) || null,
        width: positiveNumber(asset.width),
        height: positiveNumber(asset.height),
        byteSize: positiveNumber(asset.byteSize, asset.byte_size),
        kind,
        scope,
        shotId: scope === "shot" ? text(asset.shotId) || null : null,
        source: text(asset.source) === "generated" ? "generated" : "upload",
        status: ["ready", "missing_blob", "uploading", "failed"].includes(text(asset.status))
          ? text(asset.status) as ProjectDraft["assets"][number]["status"]
          : text(asset.sha256) ? "ready" : "missing_blob",
        previewUrl: text(asset.previewUrl, asset.preview_url) || null,
      };
    }),
    assetPlans: rawAssetPlans.map<ProjectDraft["assetPlans"][number]>((item, index) => {
      const plan = record(item);
      const kind = ["character", "scene", "prop", "style"].includes(text(plan.kind))
        ? text(plan.kind) as ProjectDraft["assetPlans"][number]["kind"] : "character";
      const scope = text(plan.scope) === "shot" ? "shot" : "public";
      const state = ["draft", "ready", "satisfied", "stale"].includes(text(plan.state))
        ? text(plan.state) as ProjectDraft["assetPlans"][number]["state"] : "draft";
      const [defaultWidth, defaultHeight] = defaultAssetResolution(kind);
      const resolutionSource = ["default", "ai", "manual"].includes(text(plan.resolutionSource, plan.resolution_source))
        ? text(plan.resolutionSource, plan.resolution_source) as ProjectDraft["assetPlans"][number]["resolutionSource"]
        : "default";
      return {
        id: text(plan.id, plan.plan_id) || `asset-plan-${crypto.randomUUID()}`,
        name: text(plan.name) || `素材需求 ${index + 1}`,
        description: text(plan.description),
        kind,
        scope,
        shotId: scope === "shot" ? text(plan.shotId, plan.shot_id) || null : null,
        fulfilledByAssetId: text(plan.fulfilledByAssetId, plan.fulfilled_by_asset_id) || null,
        state,
        width: imageDimension(plan.width, defaultWidth),
        height: imageDimension(plan.height, defaultHeight),
        resolutionSource,
      };
    }),
    prompts: {
      imagePrompts: rawImagePrompts.map((item) => {
        const prompt = record(item);
        return {
          id: text(prompt.id) || `image-prompt-${crypto.randomUUID()}`,
          assetPlanId: text(prompt.assetPlanId, prompt.asset_plan_id),
          prompt: text(prompt.prompt),
          endState: text(prompt.endState, prompt.end_state) || null,
          negativePrompt: text(prompt.negativePrompt, prompt.negative_prompt),
          workflowTemplateId: text(prompt.workflowTemplateId, prompt.workflow_template_id) || null,
          harnessRevision: positiveNumber(prompt.harnessRevision, prompt.harness_revision),
          referenceAssetIds: Array.isArray(prompt.referenceAssetIds) ? prompt.referenceAssetIds.filter((id): id is string => typeof id === "string") : [],
          locked: prompt.locked === true,
          revision: positiveNumber(prompt.revision) ?? 1,
        };
      }),
      h3Prompts: rawH3Prompts.map((item) => {
        const prompt = record(item);
        const mode = ["t2va", "i2va", "fl2va", "l2va", "ref2va"].includes(text(prompt.inputMode, prompt.input_mode))
          ? text(prompt.inputMode, prompt.input_mode) as ProjectDraft["prompts"]["h3Prompts"][number]["inputMode"]
          : "t2va";
        const review = record(prompt.review);
        return {
          id: text(prompt.id) || `h3-prompt-${crypto.randomUUID()}`,
          shotId: text(prompt.shotId, prompt.shot_id),
          segmentId: text(prompt.segmentId, prompt.segment_id),
          segmentIndex: Math.max(0, Math.floor(positiveNumber(prompt.segmentIndex, prompt.segment_index) ?? 0)),
          durationSeconds: positiveNumber(prompt.durationSeconds, prompt.duration_seconds) ?? 1,
          continuationOf: text(prompt.continuationOf, prompt.continuation_of) || null,
          inputMode: mode,
          prompt: text(prompt.prompt),
          endState: text(prompt.endState, prompt.end_state) || null,
          assetIds: Array.isArray(prompt.assetIds) ? prompt.assetIds.filter((id): id is string => typeof id === "string") : [],
          seed: typeof prompt.seed === "number" && prompt.seed >= 0 ? Math.floor(prompt.seed) : 0,
          harnessRevision: positiveNumber(prompt.harnessRevision, prompt.harness_revision),
          locked: prompt.locked === true,
          revision: positiveNumber(prompt.revision) ?? 1,
          legacy: prompt.legacy === true,
          review: {
            ready: review.ready === true,
            issues: Array.isArray(review.issues) ? review.issues.flatMap((issue) => {
              const value = record(issue);
              return text(value.message) ? [{ severity: text(value.severity) === "warning" ? "warning" as const : "error" as const, code: text(value.code), message: text(value.message) }] : [];
            }) : [],
            reviewedAt: text(review.reviewedAt, review.reviewed_at) || null,
          },
        };
      }),
      generatedAt: text(rawPrompts.generatedAt, rawPrompts.generated_at) || null,
    },
    postProcessing: {
      ...defaults.postProcessing,
      ...value.postProcessing,
      seedvr: {
        ...defaults.postProcessing.seedvr,
        ...value.postProcessing?.seedvr,
        modelId: value.postProcessing?.seedvr?.modelId ?? ((value.postProcessing?.seedvr as unknown as { model?: string })?.model === "7b_int8" ? null : null),
      },
      rife: { ...defaults.postProcessing.rife, ...value.postProcessing?.rife },
      whisper: {
        ...defaults.postProcessing.whisper,
        ...value.postProcessing?.whisper,
        modelId: value.postProcessing?.whisper?.modelId ?? null,
      },
    },
  };
  if (!rawH3Prompts.length) {
    hydrated.prompts.h3Prompts = rawShots.flatMap((item, shotIndex) => {
      const shot = record(item);
      const legacyPrompt = text(shot.h3Prompt, shot.prompt);
      if (!legacyPrompt) return [];
      const shotId = hydrated.shots[shotIndex]?.id ?? text(shot.id);
      const duration = hydrated.shots[shotIndex]?.durationSeconds ?? 1;
      const count = duration <= 15 ? 1 : Math.max(2, Math.ceil(duration / 12));
      return Array.from({ length: count }, (_, segmentIndex) => ({
        id: `legacy-h3-${crypto.randomUUID()}`,
        shotId,
        segmentId: `${shotId}-segment-${segmentIndex + 1}`,
        segmentIndex,
        durationSeconds: Math.round((Math.max(4, duration) / count) * 1000) / 1000,
        continuationOf: segmentIndex ? `${shotId}-segment-${segmentIndex}` : null,
        inputMode: "t2va" as const,
        prompt: legacyPrompt,
        endState: null,
        assetIds: [],
        seed: hydrated.shots[shotIndex]?.seed ?? 0,
        harnessRevision: positiveNumber(shot.harnessRevision),
        locked: hydrated.shots[shotIndex]?.locked ?? false,
        revision: 1,
        legacy: true,
        review: { ready: false, issues: [{ severity: "warning" as const, code: "legacy_prompt", message: "旧版提示词需要通过 H3 Reviewer 重新校验" }], reviewedAt: null },
      }));
    });
  }
  if (hydrated.activeStage === "generation" && !hydrated.stageApprovals.prompts) {
    hydrated.activeStage = "prompts";
  }
  if (!("config" in (value.stageApprovals ?? {}))) {
    hydrated.activeStage = "config";
  }
  return hydrated;
}

export function normalizePatchedProject(value: unknown): ProjectDraft {
  const document = record(value);
  for (const key of ["outline", "shots", "assetPlans", "assets"] as const) {
    if (!Array.isArray(document[key])) throw new Error(`LLM 返回的 ${key} 必须是数组，修改未应用`);
    if (document[key].some((item) => !item || typeof item !== "object" || Array.isArray(item))) {
      throw new Error(`LLM 返回的 ${key} 包含无效项目，修改未应用`);
    }
  }
  if (!document.idea || typeof document.idea !== "object" || Array.isArray(document.idea)) {
    throw new Error("LLM 返回的创意结构无效，修改未应用");
  }
  return hydrateProject(document as Partial<ProjectDraft>);
}

export function loadProject(): ProjectDraft {
  return newProject();
}

export function persistProject(project: ProjectDraft): ProjectDraft {
  const saved = { ...project, updatedAt: new Date().toISOString() };
  return saved;
}

export function invalidateFromStage(
  project: ProjectDraft,
  stage: PipelineStageId,
  order: PipelineStageId[],
): ProjectDraft {
  const start = order.indexOf(stage);
  const stageApprovals = { ...project.stageApprovals };
  order.slice(Math.max(0, start)).forEach((id) => delete stageApprovals[id]);
  return { ...project, stageApprovals };
}

export function saveProject(project: ProjectDraft): ProjectDraft {
  const saved = { ...project, revision: project.revision + 1, updatedAt: new Date().toISOString() };
  return saved;
}
