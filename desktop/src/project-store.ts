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
    highestInstruction: "",
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
    h3Loras: [],
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
      seedvr: { enabled: false, workflowTemplateId: null, workflowRevision: null, upscaleFactor: null, profileId: null, profileRevision: null, modelId: null, modelSha256: null, vaeId: null, vaeSha256: null, steps: 4, processingWidth: 1280, processingHeight: 720, previewSeconds: 1, previewApproved: false },
      rife: { enabled: false, profileId: null, profileRevision: null, modelId: null, modelSha256: null, targetFps: 48, workflowTemplateId: null, workflowRevision: null },
      whisper: { enabled: false, workflowTemplateId: null, workflowRevision: null, profileId: null, profileRevision: null, modelId: null, modelSha256: null, device: "auto", precision: "auto", language: "zh", burnIn: false },
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
  const rawH3Loras = Array.isArray(raw.h3Loras) ? raw.h3Loras : [];
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
    highestInstruction: text(raw.highestInstruction),
    h3Loras: rawH3Loras.flatMap((value) => {
      const lora = record(value);
      const name = text(lora.name).trim();
      if (!name) return [];
      const parsedStrength = Number(lora.strength);
      return [{
        id: text(lora.id) || crypto.randomUUID(),
        name,
        strength: Number.isFinite(parsedStrength) ? Math.max(-4, Math.min(4, parsedStrength)) : 1,
        enabled: lora.enabled !== false,
      }];
    }),
    activeStage: value.activeStage ?? defaults.activeStage,
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
        motionSegments: (Array.isArray(shot.motionSegments) ? shot.motionSegments : []).map((segment, segmentIndex) => {
          const motionSegment = record(segment);
          return {
            id: text(motionSegment.id) || `motion-${segmentIndex + 1}-${crypto.randomUUID()}`,
            durationSeconds: positiveNumber(motionSegment.durationSeconds) ?? 1,
            summary: text(motionSegment.summary, motionSegment.description),
          };
        }),
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
      const assetShotId = text(asset.shotId) || null;
      return {
        id: text(asset.id) || `asset-${crypto.randomUUID()}`,
        name: text(asset.name) || `素材 ${index + 1}`,
        originalFileName: text(asset.originalFileName, asset.original_file_name, asset.name),
        sha256: text(asset.sha256) || null,
        mimeType: text(asset.mimeType, asset.mime_type) || null,
        mediaKind: ["image", "video", "audio"].includes(text(asset.mediaKind, asset.media_kind))
          ? text(asset.mediaKind, asset.media_kind) as ProjectDraft["assets"][number]["mediaKind"]
          : text(asset.mimeType, asset.mime_type).startsWith("video/") ? "video"
            : text(asset.mimeType, asset.mime_type).startsWith("audio/") ? "audio" : "image",
        width: positiveNumber(asset.width),
        height: positiveNumber(asset.height),
        durationSeconds: positiveNumber(asset.durationSeconds, asset.duration_seconds),
        frameRate: positiveNumber(asset.frameRate, asset.frame_rate),
        hasAudio: typeof asset.hasAudio === "boolean" ? asset.hasAudio
          : typeof asset.has_audio === "boolean" ? asset.has_audio : null,
        byteSize: positiveNumber(asset.byteSize, asset.byte_size),
        kind,
        scope,
        shotId: scope === "shot" ? assetShotId : null,
        shotIds: Array.isArray(asset.shotIds)
          ? asset.shotIds.map((value) => text(value)).filter(Boolean)
          : [],
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
      const planShotId = text(plan.shotId) || null;
      const shotIds = Array.isArray(plan.shotIds)
        ? plan.shotIds.map((value) => text(value)).filter(Boolean)
        : [];
      const state = ["draft", "ready", "satisfied", "stale"].includes(text(plan.state))
        ? text(plan.state) as ProjectDraft["assetPlans"][number]["state"] : "draft";
      const [defaultWidth, defaultHeight] = defaultAssetResolution(kind);
      const resolutionSource = ["default", "ai", "manual"].includes(text(plan.resolutionSource, plan.resolution_source))
        ? text(plan.resolutionSource, plan.resolution_source) as ProjectDraft["assetPlans"][number]["resolutionSource"]
        : "default";
      return {
        id: text(plan.id) || `asset-plan-${crypto.randomUUID()}`,
        name: text(plan.name) || `素材需求 ${index + 1}`,
        description: text(plan.description),
        kind,
        scope,
        shotId: scope === "shot" ? planShotId : null,
        shotIds,
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
          assetPlanId: text(prompt.assetPlanId),
          prompt: text(prompt.prompt),
          endState: text(prompt.endState) || null,
          negativePrompt: text(prompt.negativePrompt, prompt.negative_prompt),
          workflowTemplateId: text(prompt.workflowTemplateId, prompt.workflow_template_id) || null,
          harnessRevision: positiveNumber(prompt.harnessRevision),
          harnessManifestSha256: text(prompt.harnessManifestSha256, prompt.harness_manifest_sha256) || null,
          route: text(prompt.route) || null,
          assetRoles: Array.isArray(prompt.assetRoles) ? prompt.assetRoles as Array<Record<string, unknown>> : [],
          assumptions: Array.isArray(prompt.assumptions) ? prompt.assumptions.map(text) : [],
          stageTrace: Array.isArray(prompt.stageTrace) ? prompt.stageTrace as ProjectDraft["prompts"]["h3Prompts"][number]["stageTrace"] : [],
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
          shotId: text(prompt.shotId),
          segmentId: text(prompt.segmentId),
          segmentIndex: Math.max(0, Math.floor(positiveNumber(prompt.segmentIndex) ?? 0)),
          durationSeconds: positiveNumber(prompt.durationSeconds) ?? 1,
          continuationOf: text(prompt.continuationOf) || null,
          inputMode: mode,
          prompt: text(prompt.prompt),
          endState: text(prompt.endState, prompt.end_state) || null,
          assetIds: Array.isArray(prompt.assetIds) ? prompt.assetIds.filter((id): id is string => typeof id === "string") : [],
          seed: typeof prompt.seed === "number" && prompt.seed >= 0 ? Math.floor(prompt.seed) : 0,
          harnessRevision: positiveNumber(prompt.harnessRevision),
          locked: prompt.locked === true,
          revision: positiveNumber(prompt.revision) ?? 1,
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
