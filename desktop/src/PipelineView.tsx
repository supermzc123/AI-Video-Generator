import {
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Film,
  Music,
  ImagePlus,
  ListPlus,
  Languages,
  LockKeyhole,
  PackageCheck,
  Play,
  Plus,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Save,
  Pause,
  Square,
  Trash2,
  Upload,
  X,
  ZoomIn,
} from "lucide-react";
import { ChangeEvent, KeyboardEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  cancelTask,
  acceptAssetCandidate,
  assetCandidatePreviewUrl,
  compileProjectDeliveryTasks,
  compileProjectTasks,
  createReworkMarker,
  confirmProjectReworks,
  deleteProjectAsset,
  discardAssetCandidate,
  streamGenerateImagePrompt,
  getProjectWorkspace,
  getProjectExecutionStatus,
  getTask,
  listProjectAssets,
  listAssetCandidates,
  listTaskArtifacts,
  listTaskReviewDecisions,
  artifactMediaUrl,
  projectAssetPreviewUrl,
  projectAssetMediaUrl,
  regenerateH3Prompt,
  streamRegenerateH3Prompt,
  regenerateProjectAsset,
  relinkProjectAsset,
  runTask,
  runProjectImagePrompt,
  saveProjectToControlPlane,
  setProjectMode,
  setProjectPaused,
  setReviewMode,
  startProjectGeneration,
  translateH3Prompt,
  updateProjectAsset,
  uploadProjectAsset,
  withdrawReworkMarker,
} from "./api";
import { invalidateFromStage, persistProject } from "./project-store";
import { ProjectAgentPanel } from "./ProjectAgentPanel";
import { ProjectView } from "./ProjectView";
import type {
  AssetDraft,
  AssetGenerationCandidate,
  ArtifactDescriptor,
  OutlineBeat,
  PipelineStageId,
  ProjectDraft,
  ProjectExecutionStatus,
  ShotDraft,
  TaskKind,
  TaskSpec,
  ReviewDecision,
  ReworkMarker,
  SegmentGenerationVersion,
} from "./types";

function extractH3Description(stream: string): string | null {
  return stream || null;
}

function detachAsset(project: ProjectDraft, assetId: string): ProjectDraft {
  const conceptNodes = project.idea.conceptDocument.nodes.filter(
    (node) => node.type !== "asset_mention" || node.assetId !== assetId,
  );
  return {
    ...project,
    idea: {
      ...project.idea,
      concept: conceptNodes.map((node) => node.type === "asset_mention" ? `@${node.displayName}` : node.text).join(""),
      conceptDocument: { nodes: conceptNodes },
    },
    assets: project.assets.filter((asset) => asset.id !== assetId),
    assetPlans: project.assetPlans.map((plan) => plan.fulfilledByAssetId === assetId
      ? { ...plan, fulfilledByAssetId: null, state: "ready" as const }
      : plan),
    prompts: {
      ...project.prompts,
      imagePrompts: project.prompts.imagePrompts.map((prompt) => ({
        ...prompt,
        referenceAssetIds: prompt.referenceAssetIds.filter((id) => id !== assetId),
      })),
      h3Prompts: project.prompts.h3Prompts.filter((prompt) => !prompt.assetIds.includes(assetId)),
    },
  };
}

export const pipelineOrder: PipelineStageId[] = [
  "config",
  "idea",
  "outline",
  "storyboard",
  "assets",
  "prompts",
  "generation",
  "delivery",
];

const stageMeta: Record<PipelineStageId, { index: string; label: string; short: string }> = {
  config: { index: "01", label: "项目配置", short: "配置" },
  idea: { index: "02", label: "创意设定", short: "创意" },
  outline: { index: "03", label: "故事大纲", short: "大纲" },
  storyboard: { index: "04", label: "电影分镜", short: "分镜" },
  assets: { index: "05", label: "素材与参考图", short: "素材" },
  prompts: { index: "06", label: "提示词设计", short: "提示词" },
  generation: { index: "07", label: "视频生成", short: "生成" },
  delivery: { index: "08", label: "后处理与交付", short: "交付" },
};

const taskKindLabels: Record<TaskKind, string> = {
  llm_planning: "LLM 规划",
  image_generation: "图片生成",
  conditioning_encoding: "条件编码",
  h3_generation: "H3 生成",
  ai_review: "审核",
  model_switch: "准备视频生成模型",
  seedvr2: "SeedVR2",
  rife: "RIFE",
  whisper: "Whisper",
  master_assembly: "母版拼接",
  export: "FFmpeg 导出",
  asset_transfer: "素材传输",
};

const taskStateLabels: Record<TaskSpec["state"], string> = {
  blocked: "等待前置任务",
  ready: "等待派发",
  queued: "已派发",
  paused: "已暂停",
  running: "执行中",
  needs_review: "等待人工审核",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  stale: "已失效",
};

const deliveryTaskKinds = new Set<TaskKind>(["seedvr2", "rife", "whisper", "master_assembly", "export"]);

const reviewBoundaryKinds = new Set<TaskKind>([
  "image_generation",
  "conditioning_encoding",
  "model_switch",
  "h3_generation",
  "ai_review",
]);

type Props = {
  project: ProjectDraft;
  workflows: Array<{ name: string; id: string; revision: number; kind?: "image" | "interpolation" | "restoration" | "transcription" | "video" }>;
  workerOnline: boolean;
  onChange: (project: ProjectDraft) => void;
  onOpenWorkflows: () => void;
  onOpenTasks: () => void;
  onOpenBatch: () => void;
};

type AssetResolutionInputProps = {
  value: number;
  label: string;
  onCommit: (value: number) => void;
  onInvalid: () => void;
};

function AssetResolutionInput({ value, label, onCommit, onInvalid }: AssetResolutionInputProps) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => { setDraft(String(value)); }, [value]);

  const commit = () => {
    const parsed = Number(draft.trim());
    if (!Number.isInteger(parsed) || parsed < 64 || parsed > 4096 || parsed % 8 !== 0) {
      setDraft(String(value));
      onInvalid();
      return;
    }
    onCommit(parsed);
  };

  return <input
    inputMode="numeric"
    value={draft}
    aria-label={label}
    onChange={(event) => setDraft(event.target.value)}
    onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }}
    onBlur={commit}
  />;
}

function uid(prefix: string) {
  return `${prefix}-${crypto.randomUUID()}`;
}

type H3SegmentSlot = {
  shotId: string;
  segmentId: string;
  segmentIndex: number;
  segmentCount: number;
  durationSeconds: number;
  continuationOf: string | null;
  seed: number;
};

function buildH3SegmentSlots(shots: ShotDraft[]): H3SegmentSlot[] {
  return shots.flatMap((shot) => {
    const configured = shot.motionSegments;
    const durations = configured.length
      ? configured.map((segment) => segment.durationSeconds)
      : automaticMotionSegmentDurations(shot.durationSeconds);
    const segmentCount = durations.length;
    return durations.map((durationSeconds, segmentIndex) => ({
      shotId: shot.id,
      segmentId: configured[segmentIndex]?.id
        || `${shot.id}.C${String(segmentIndex + 1).padStart(2, "0")}`,
      segmentIndex,
      segmentCount,
      durationSeconds,
      continuationOf: segmentIndex
        ? configured[segmentIndex - 1]?.id
          || `${shot.id}.C${String(segmentIndex).padStart(2, "0")}`
        : null,
      seed: shot.seed,
    }));
  });
}

function automaticMotionSegmentDurations(durationSeconds: number): number[] {
  const executionDuration = Math.max(4, durationSeconds);
  const count = durationSeconds <= 15 ? 1 : Math.max(2, Math.ceil(durationSeconds / 12));
  const piece = Math.round((executionDuration / count) * 1000) / 1000;
  const durations = Array.from({ length: count }, () => piece);
  durations[count - 1] = Math.round((executionDuration - piece * (count - 1)) * 1000) / 1000;
  return durations;
}

type GenerationSegment = ProjectExecutionStatus["hierarchy"][number]["segments"][number];

function GenerationSegmentCard({ segment, project, tasks, reviewTasks, artifactsByTask, decisionsByTask, markers, actionBusy, onMark, onWithdraw }: {
  segment: GenerationSegment;
  project: ProjectDraft;
  tasks: TaskSpec[];
  reviewTasks: TaskSpec[];
  artifactsByTask: Record<string, ArtifactDescriptor[]>;
  decisionsByTask: Record<string, ReviewDecision[]>;
  markers: ReworkMarker[];
  actionBusy: boolean;
  onMark: (version: SegmentGenerationVersion) => void;
  onWithdraw: (markerId: string) => void;
}) {
  const version = segment.active_version ?? segment.versions.at(-1) ?? null;
  const task = version ? tasks.find((item) => item.task_id === version.task_id) ?? segment.task : segment.task;
  const video = task ? artifactsByTask[task.task_id]?.find((item) => item.kind === "video_segment") : null;
  const reviewTask = task ? reviewTasks.find((item) => item.depends_on.includes(task.task_id)) : null;
  const decision = reviewTask ? decisionsByTask[reviewTask.task_id]?.at(-1) : null;
  const marker = markers.find((item) => item.segment_id === segment.segment_id && ["draft", "preparing", "sealed"].includes(item.state));
  const prompt = project.prompts.h3Prompts.find((item) => item.segmentId === segment.segment_id);
  const showAi = !["manual", "none"].includes(project.reviewPolicy.effectiveMode);
  return <article className="review-card">
    <header><div><span>连接段 {segment.segment_index + 1}</span><div><strong>{segment.segment_id} · {version ? `第 ${version.generation_number} 次生成` : "尚未生成"}</strong><small>{segment.frozen ? "已冻结" : task ? taskStateLabels[task.state] : "等待计划"}</small></div></div><span className={`state state-${segment.frozen ? "paused" : task?.state ?? "blocked"}`}>{segment.frozen ? "已冻结" : task ? taskStateLabels[task.state] : "等待计划"}</span></header>
    <div className="review-media">{video ? <video controls preload="metadata" src={artifactMediaUrl(video)}>当前系统播放器不支持此视频格式。</video> : <div className="media-placeholder"><Film size={24} /><span>{task?.state === "succeeded" ? "视频产物未登记或已作废" : segment.freeze_reason ?? "等待片段生成完成"}</span></div>}</div>
    {prompt && <details className="review-prompt"><summary>查看生成提示词</summary><pre>{prompt.prompt}</pre></details>}
    {decision && showAi && <section className="review-decision"><div className="decision-summary"><strong>{decision.disposition === "accepted" ? "AI 建议通过" : "AI 建议返工"}</strong><span>置信度 {Math.round(decision.confidence * 100)}%</span></div>{decision.issues.map((issue, index) => <div className={`review-issue ${issue.severity}`} key={`${issue.category}:${index}`}><header><strong>{issue.message}</strong><span>{issue.start_seconds === null ? "未标注时间" : `${issue.start_seconds.toFixed(1)}s`}</span></header>{issue.suggested_action && <p>建议：{issue.suggested_action}</p>}</div>)}</section>}
    {marker && <div className="rework-marker"><CircleAlert size={14} /><span>{marker.state === "draft" ? "等待统一确认返工" : marker.state === "preparing" ? "返工批次准备中" : "返工批次已封存"}：{marker.feedback}</span>{marker.state !== "sealed" && <button className="secondary-button" disabled={actionBusy} onClick={() => onWithdraw(marker.marker_id)}>撤销标记</button>}</div>}
    {version && !marker && project.reviewPolicy.effectiveMode !== "none" && <footer><button className="secondary-button" disabled={actionBusy || !video} onClick={() => onMark(version)}><RotateCcw size={14} />标记返工</button></footer>}
  </article>;
}

function motionSegmentsValid(shot: ShotDraft): boolean {
  if (!shot.motionSegments.length) return true;
  const total = shot.motionSegments.reduce((sum, segment) => sum + segment.durationSeconds, 0);
  return Math.abs(total - Math.max(4, shot.durationSeconds)) < 0.01
    && shot.motionSegments.every((segment, index) => (
      segment.durationSeconds >= 4
      && segment.durationSeconds <= (index === 0 ? 15 : 12)
      && segment.summary.trim()
    ));
}

function activateFileLabel(event: KeyboardEvent<HTMLLabelElement>) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    event.currentTarget.querySelector<HTMLInputElement>('input[type="file"]')?.click();
  }
}

function Field({ label, children, wide = false }: { label: string; children: ReactNode; wide?: boolean }) {
  return <label className={`field ${wide ? "field-wide" : ""}`}><span>{label}</span>{children}</label>;
}

function buildConceptDocument(value: string, assets: AssetDraft[]): ProjectDraft["idea"]["conceptDocument"] {
  const lookup = new Map(assets.map((asset) => [asset.name.toLocaleLowerCase(), asset]));
  const nodes: ProjectDraft["idea"]["conceptDocument"]["nodes"] = [];
  const pattern = /@([^\s@，。！？；：,.!?;:]+)/g;
  let cursor = 0;
  for (const match of value.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > cursor) nodes.push({ type: "text", text: value.slice(cursor, index) });
    const asset = lookup.get(match[1].toLocaleLowerCase());
    nodes.push(asset
      ? { type: "asset_mention", assetId: asset.id, displayName: asset.name }
      : { type: "text", text: match[0] });
    cursor = index + match[0].length;
  }
  if (cursor < value.length) nodes.push({ type: "text", text: value.slice(cursor) });
  return { nodes };
}

function MentionTextarea({ value, assets, onChange }: {
  value: string;
  assets: AssetDraft[];
  onChange: (value: string, document: ProjectDraft["idea"]["conceptDocument"]) => void;
}) {
  const [caret, setCaret] = useState(value.length);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const token = value.slice(0, caret).match(/@([^\s@，。！？；：,.!?;:]*)$/)?.[1] ?? null;
  const suggestions = token === null ? [] : assets
    .filter((asset) => asset.name.toLocaleLowerCase().includes(token.toLocaleLowerCase()))
    .slice(0, 6);

  const insert = (asset: AssetDraft) => {
    const prefix = value.slice(0, caret).replace(/@([^\s@，。！？；：,.!?;:]*)$/, `@${asset.name}`);
    const next = `${prefix} ${value.slice(caret)}`;
    onChange(next, buildConceptDocument(next, assets));
    requestAnimationFrame(() => textarea.current?.focus());
  };

  return <div className="mention-editor">
    <textarea
      ref={textarea}
      rows={6}
      value={value}
      onChange={(event) => {
        setCaret(event.target.selectionStart);
        onChange(event.target.value, buildConceptDocument(event.target.value, assets));
      }}
      onClick={(event) => setCaret(event.currentTarget.selectionStart)}
      onKeyUp={(event) => setCaret(event.currentTarget.selectionStart)}
      placeholder="人物、目标、冲突和预期结局；输入 @ 引用已命名图片"
    />
    {suggestions.length > 0 && <div className="mention-suggestions">
      {suggestions.map((asset) => <button type="button" key={asset.id} onMouseDown={(event) => { event.preventDefault(); insert(asset); }}>
        <span>@{asset.name}</span><small>{asset.kind}</small>
      </button>)}
    </div>}
  </div>;
}

function AssetLightbox({ assets, activeId, projectId, onClose, onSelect }: {
  assets: AssetDraft[];
  activeId: string;
  projectId: string;
  onClose: () => void;
  onSelect: (assetId: string) => void;
}) {
  const index = Math.max(0, assets.findIndex((asset) => asset.id === activeId));
  const asset = assets[index];
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; originX: number; originY: number } | null>(null);

  const move = (direction: number) => {
    if (!assets.length) return;
    onSelect(assets[(index + direction + assets.length) % assets.length].id);
  };

  useEffect(() => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }, [activeId]);

  useEffect(() => {
    const keydown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      else if (event.key === "ArrowLeft") move(-1);
      else if (event.key === "ArrowRight") move(1);
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  });

  if (!asset) return null;
  const mediaUrl = projectAssetMediaUrl(projectId, asset.id);
  const imageUrl = asset.previewUrl || projectAssetPreviewUrl(projectId, asset.id);
  return <div className="asset-lightbox" role="dialog" aria-modal="true" aria-label={`预览 ${asset.name}`} onClick={onClose}>
    <header onClick={(event) => event.stopPropagation()}><div><strong>{asset.name}</strong><span>{asset.width && asset.height ? `${asset.width} × ${asset.height}` : "原比例预览"} · {index + 1}/{assets.length}</span></div><button className="icon-button" title="关闭预览" onClick={onClose}><X size={20} /></button></header>
    <button className="lightbox-nav previous" title="上一张" onClick={(event) => { event.stopPropagation(); move(-1); }}><ChevronLeft size={28} /></button>
    <div
      className={`lightbox-canvas ${scale > 1 ? "zoomed" : ""}`}
      onClick={(event) => event.stopPropagation()}
      onWheel={(event) => {
        event.preventDefault();
        setScale((current) => Math.min(6, Math.max(1, current * (event.deltaY < 0 ? 1.18 : .85))));
        if (event.deltaY > 0 && scale <= 1.18) setOffset({ x: 0, y: 0 });
      }}
      onPointerDown={(event) => {
        if (scale <= 1) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { x: event.clientX, y: event.clientY, originX: offset.x, originY: offset.y };
      }}
      onPointerMove={(event) => {
        if (!drag.current) return;
        setOffset({ x: drag.current.originX + event.clientX - drag.current.x, y: drag.current.originY + event.clientY - drag.current.y });
      }}
      onPointerUp={() => { drag.current = null; }}
    >
      {asset.mediaKind === "video" ? <video src={mediaUrl} controls autoPlay={false} />
        : asset.mediaKind === "audio" ? <audio src={mediaUrl} controls />
          : <img src={imageUrl} alt={asset.name} draggable={false} style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }} />}
    </div>
    <button className="lightbox-nav next" title="下一张" onClick={(event) => { event.stopPropagation(); move(1); }}><ChevronRight size={28} /></button>
    <footer onClick={(event) => event.stopPropagation()}>{asset.mediaKind === "image" ? <><ZoomIn size={15} /><span>{Math.round(scale * 100)}% · 滚轮缩放，拖动查看细节，方向键切换</span></> : <><Film size={15} /><span>{asset.mediaKind === "video" ? "视频参考" : "音频参考"} · 方向键切换素材</span></>}</footer>
  </div>;
}

export function PipelineView({ project, workflows, workerOnline, onChange, onOpenWorkflows, onOpenTasks, onOpenBatch }: Props) {
  const latestProject = useRef(project);
  latestProject.current = project;
  const [actionBusy, setActionBusy] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [promptGenerationBusy, setPromptGenerationBusy] = useState(false);
  const [promptGenerationProgress, setPromptGenerationProgress] = useState({ completed: 0, total: 0 });
  const [configFormValid, setConfigFormValid] = useState(true);
  const [promptTab, setPromptTab] = useState<"image" | "h3">("h3");
  const [promptTranslations, setPromptTranslations] = useState<Record<string, string>>({});
  const [translatingPromptId, setTranslatingPromptId] = useState<string | null>(null);
  const [pendingUploads, setPendingUploads] = useState<Array<{ file: File; planId: string | null }>>([]);
  const [pendingUploadName, setPendingUploadName] = useState("");
  const [pendingUploadKind, setPendingUploadKind] = useState<AssetDraft["kind"]>("scene");
  const [agentGenerationBusy, setAgentGenerationBusy] = useState(false);
  const [executionStatus, setExecutionStatus] = useState<ProjectExecutionStatus | null>(null);
  const [guidedRunTarget, setGuidedRunTarget] = useState<"review" | "delivery" | null>(null);
  const [activeImageTasks, setActiveImageTasks] = useState<Record<string, string>>({});
  const [candidateTaskPlans, setCandidateTaskPlans] = useState<Record<string, boolean>>({});
  const [assetCandidatesByPlan, setAssetCandidatesByPlan] = useState<Record<string, AssetGenerationCandidate[]>>({});
  const [candidatePreviewId, setCandidatePreviewId] = useState<string | null>(null);
  const [selectedWorkflowByPlan, setSelectedWorkflowByPlan] = useState<Record<string, string>>({});
  const [lightboxAssetId, setLightboxAssetId] = useState<string | null>(null);
  const [artifactsByTask, setArtifactsByTask] = useState<Record<string, ArtifactDescriptor[]>>({});
  const [reviewDecisionsByTask, setReviewDecisionsByTask] = useState<Record<string, ReviewDecision[]>>({});
  const [mediaLoadError, setMediaLoadError] = useState<string | null>(null);
  const [reworkDialog, setReworkDialog] = useState<null | {
    version: SegmentGenerationVersion;
    segmentId: string;
    action: "retry" | "change_seed" | "revise_prompt";
    feedback: string;
    replacementSeed: number;
  }>(null);
  const [imagePromptDialog, setImagePromptDialog] = useState<null | {
    planId: string;
    instruction: string;
    prompt: string;
    negativePrompt: string;
  }>(null);
  const promptGenerationStarted = useRef(new Set<string>());
  const promptGenerationAbort = useRef<AbortController | null>(null);
  const guidedRunAbort = useRef<AbortController | null>(null);
  const activeIndex = pipelineOrder.indexOf(project.activeStage);
  const shotsDuration = useMemo(
    () => project.shots.reduce((total, shot) => total + shot.durationSeconds, 0),
    [project.shots],
  );
  const h3SegmentSlots = useMemo(() => buildH3SegmentSlots(project.shots), [project.shots]);
  const promptForSlot = (slot: H3SegmentSlot) => project.prompts.h3Prompts.find((item) => (
    item.segmentId === slot.segmentId
    || (item.shotId === slot.shotId && item.segmentIndex === slot.segmentIndex)
  ));
  const segmentCount = h3SegmentSlots.length;
  const continuationCount = Math.max(0, segmentCount - project.shots.length);
  const currentPlanTasks = executionStatus?.tasks ?? [];
  const generationTasks = currentPlanTasks.filter((task) => reviewBoundaryKinds.has(task.kind));
  const reviewTasks = currentPlanTasks.filter((task) => task.kind === "ai_review");
  const deliveryTasks = currentPlanTasks.filter((task) => deliveryTaskKinds.has(task.kind));
  const imageWorkflows = workflows.filter((item) => !item.kind || item.kind === "image");
  const restorationWorkflows = workflows.filter((item) => item.kind === "restoration");
  const interpolationWorkflows = workflows.filter((item) => item.kind === "interpolation");
  const transcriptionWorkflows = workflows.filter((item) => item.kind === "transcription");
  const mediaTaskKey = currentPlanTasks
    .filter((task) => task.kind === "h3_generation" || task.kind === "ai_review" || task.kind === "export")
    .map((task) => `${task.task_id}:${task.state}:${task.attempt}`)
    .join("|");
  const handleAutomaticGenerationChange = useCallback((busy: boolean) => setAgentGenerationBusy(busy), []);
  const refreshExecutionStatus = async () => {
    try {
      const status = await getProjectExecutionStatus(project.projectId);
      setExecutionStatus(status);
      return status;
    } catch {
      setExecutionStatus(null);
      return null;
    }
  };

  useEffect(() => {
    void refreshExecutionStatus();
    if (activeIndex < pipelineOrder.indexOf("generation")) return;
    const timer = window.setInterval(() => void refreshExecutionStatus(), 2000);
    return () => window.clearInterval(timer);
  }, [project.projectId, project.revision, activeIndex]);

  useEffect(() => {
    if (project.activeStage !== "generation" && project.activeStage !== "delivery") return;
    let cancelled = false;
    const loadMedia = async () => {
      const artifactTasks = currentPlanTasks.filter((task) => task.kind === "h3_generation" || task.kind === "export");
      const decisionTasks = currentPlanTasks.filter((task) => task.kind === "ai_review");
      try {
        const [artifactEntries, decisionEntries] = await Promise.all([
          Promise.all(artifactTasks.map(async (task) => [task.task_id, await listTaskArtifacts(task.task_id)] as const)),
          Promise.all(decisionTasks.map(async (task) => [task.task_id, await listTaskReviewDecisions(task.task_id)] as const)),
        ]);
        if (cancelled) return;
        setArtifactsByTask(Object.fromEntries(artifactEntries));
        setReviewDecisionsByTask(Object.fromEntries(decisionEntries));
        setMediaLoadError(null);
      } catch (cause) {
        if (!cancelled) setMediaLoadError(cause instanceof Error ? cause.message : "无法读取审核媒体");
      }
    };
    void loadMedia();
    return () => { cancelled = true; };
  }, [project.projectId, project.activeStage, mediaTaskKey]);

  useEffect(() => () => {
    promptGenerationAbort.current?.abort();
    guidedRunAbort.current?.abort();
  }, []);

  const update = (stage: PipelineStageId, transform: (current: ProjectDraft) => ProjectDraft) => {
    const changed = transform(project);
    const invalidated = project.stageApprovals[stage]
      ? invalidateFromStage(changed, stage, pipelineOrder)
      : changed;
    onChange(persistProject(invalidated));
  };

  const availableImageWorkflow = (candidate: string | null | undefined) => (
    imageWorkflows.some((workflow) => workflow.id === candidate) ? candidate ?? null : null
  );

  const imageWorkflowForPlan = (planId: string) => {
    const existing = project.prompts.imagePrompts.find((item) => item.assetPlanId === planId);
    return availableImageWorkflow(selectedWorkflowByPlan[planId])
      ?? availableImageWorkflow(existing?.workflowTemplateId)
      ?? imageWorkflows[0]?.id
      ?? null;
  };

  const commitAssetResolution = (
    plan: ProjectDraft["assetPlans"][number],
    field: "width" | "height",
    value: number,
  ) => {
    if (value === plan[field] && plan.resolutionSource === "manual") return;
    update("assets", (current) => ({
      ...current,
      assetPlans: current.assetPlans.map((item) => item.id === plan.id
        ? { ...item, [field]: value, resolutionSource: "manual" as const }
        : item),
    }));
    const other = field === "width" ? plan.height : plan.width;
    setActionMessage(`已将“${plan.name}”设为手动分辨率，当前约 ${((value * other) / 1_000_000).toFixed(2)} MP`);
  };

  const assetResolutionControls = (plan: ProjectDraft["assetPlans"][number]) => (
    <div className="asset-resolution-control">
      <span>生成尺寸</span>
      <label><span>宽</span><AssetResolutionInput
        value={plan.width}
        label={`${plan.name}图片宽度`}
        onCommit={(value) => commitAssetResolution(plan, "width", value)}
        onInvalid={() => setActionError("图片宽高必须是 64 到 4096 之间且能被 8 整除的整数")}
      /></label>
      <i>×</i>
      <label><span>高</span><AssetResolutionInput
        value={plan.height}
        label={`${plan.name}图片高度`}
        onCommit={(value) => commitAssetResolution(plan, "height", value)}
        onInvalid={() => setActionError("图片宽高必须是 64 到 4096 之间且能被 8 整除的整数")}
      /></label>
      <em>{plan.resolutionSource === "manual" ? "手动" : plan.resolutionSource === "ai" ? "AI" : "待 AI 决定"}</em>
    </div>
  );

  const completion = useMemo<Record<PipelineStageId, { valid: boolean; message: string }>>(() => ({
    config: {
      valid: Boolean(
        project.name.trim()
        && project.width >= 64
        && project.height >= 64
        && project.width % 32 === 0
        && project.height % 32 === 0
        && configFormValid
        && project.fps > 0
        && project.targetDurationSeconds > 0
      ),
      message: "完成项目名称、画幅、帧率和目标时长配置",
    },
    idea: {
      valid: Boolean(project.idea.concept.trim() && project.idea.genre.trim()),
      message: "填写核心创意和类型",
    },
    outline: {
      valid: project.outline.length > 0 && project.outline.every((beat) => beat.title.trim() && beat.summary.trim()),
      message: "至少保留一个完整故事段落",
    },
    storyboard: {
      valid: project.shots.length > 0 && project.shots.every((shot) => shot.title.trim() && shot.summary.trim() && shot.durationSeconds > 0 && motionSegmentsValid(shot)),
      message: "至少保留一个有效电影分镜；自定义续段须合计为镜头时长，每段 4–12 秒并填写内容",
    },
    assets: {
      valid: project.referenceAssetMode === "none" || (
        project.assetPlans.length
          ? project.assetPlans.every((plan) => Boolean(plan.fulfilledByAssetId))
          : project.assets.some((asset) => asset.status === "ready")
      ),
      message: "先上传或生成并确认全部需求图片；完成后才会生成 H3 视频提示词",
    },
    prompts: {
      valid: h3SegmentSlots.length > 0
        && h3SegmentSlots.every((slot) => Boolean(promptForSlot(slot)?.prompt.trim())),
      message: "每个 H3 片段都需要一份非空视频提示词；点击下一步即确认当前文本",
    },
    generation: {
      valid: executionStatus?.review_complete === true,
      message: executionStatus?.delivery_blocked_reason
        ?? (executionStatus?.compiled ? "等待完整活动视频链和全部返工结论" : "先创建并执行本项目任务计划"),
    },
    delivery: {
      valid: executionStatus?.delivery_complete === true,
      message: "创建并成功完成 FFmpeg 导出任务",
    },
  }), [configFormValid, executionStatus, project]);

  useEffect(() => {
    let cancelled = false;
    const planIds = project.assetPlans.filter((plan) => Boolean(plan.fulfilledByAssetId)).map((plan) => plan.id);
    if (!planIds.length) {
      setAssetCandidatesByPlan({});
      return;
    }
    void Promise.all(planIds.map(async (planId) => [planId, await listAssetCandidates(project.projectId, planId)] as const))
      .then((entries) => { if (!cancelled) setAssetCandidatesByPlan(Object.fromEntries(entries)); })
      .catch((cause) => { if (!cancelled) setActionError(cause instanceof Error ? cause.message : "读取图片候选版本失败"); });
    return () => { cancelled = true; };
  }, [project.projectId, project.assetPlans.map((plan) => `${plan.id}:${plan.fulfilledByAssetId}`).join("|")]);

  useEffect(() => {
    const entries = Object.entries(activeImageTasks);
    if (!entries.length) return;
    let cancelled = false;
    const poll = async () => {
      for (const [planId, taskId] of entries) {
        try {
          const task = await getTask(taskId);
          if (cancelled) return;
          if (task.state === "succeeded") {
            if (candidateTaskPlans[planId]) {
              const candidates = await listAssetCandidates(project.projectId, planId);
              if (cancelled) return;
              setAssetCandidatesByPlan((current) => ({ ...current, [planId]: candidates }));
              setCandidateTaskPlans((current) => {
                const next = { ...current };
                delete next[planId];
                return next;
              });
              setActiveImageTasks((current) => {
                const next = { ...current };
                delete next[planId];
                return next;
              });
              setActionMessage("候选图片已生成；接受前不会替换当前素材");
              continue;
            }
            const [workspace, assets] = await Promise.all([
              getProjectWorkspace(project.projectId),
              listProjectAssets(project.projectId),
            ]);
            if (cancelled) return;
            onChange({ ...workspace, assets });
            setActiveImageTasks((current) => {
              const next = { ...current };
              delete next[planId];
              return next;
            });
            setActionMessage("图片已生成并登记为项目素材；可以查看缩略图后继续或修改");
          } else if (["failed", "cancelled"].includes(task.state)) {
            setActiveImageTasks((current) => {
              const next = { ...current };
              delete next[planId];
              return next;
            });
            setActionError(task.error_message || "图片生成任务失败");
          }
        } catch (cause) {
          if (!cancelled) setActionError(cause instanceof Error ? cause.message : "读取图片任务状态失败");
        }
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [activeImageTasks, candidateTaskPlans, project.projectId]);

  useEffect(() => {
    if (project.activeStage === "prompts") setPromptTab("h3");
  }, [project.activeStage, project.projectId]);

  async function runPromptGeneration(force: boolean) {
    const key = `${project.projectId}:${project.revision}:prompts`;
    if (!force && promptGenerationStarted.current.has(key)) return;
    const targets = h3SegmentSlots.filter((slot) => {
      const prompt = promptForSlot(slot);
      return !prompt?.prompt.trim();
    });
    if (!targets.length) {
      setActionMessage("所有视频提示词均已填写；文本框内容将直接用于视频生成");
      return;
    }
    promptGenerationStarted.current.add(key);
    const controller = new AbortController();
    promptGenerationAbort.current?.abort();
    promptGenerationAbort.current = controller;
    setPromptGenerationBusy(true);
    setPromptGenerationProgress({ completed: 0, total: targets.length });
    setActionMessage(`正在生成第 1/${targets.length} 个 H3 视频提示词...`);
    let succeeded = 0;
    let failed = 0;
    try {
      const results = await Promise.allSettled(targets.map(async (slot) => {
        let streamed = "";
        const generated = await streamRegenerateH3Prompt(
          project.projectId,
          slot.segmentId,
          (delta) => {
            streamed += delta;
            const preview = extractH3Description(streamed);
            if (preview) patchH3Prompt(slot, preview);
          },
          controller.signal,
        );
        return { slot, generated };
      }));
      const merged = new Map(project.prompts.h3Prompts.map((item) => [item.segmentId, item]));
      results.forEach((result) => {
        if (result.status === "fulfilled") {
          const prompt = result.value.generated.prompts.h3Prompts.find((item) => (
            item.segmentId === result.value.slot.segmentId
            || (item.shotId === result.value.slot.shotId && item.segmentIndex === result.value.slot.segmentIndex)
          ));
          if (prompt?.prompt.trim()) { succeeded += 1; merged.set(prompt.segmentId, prompt); } else failed += 1;
        } else if (!controller.signal.aborted) failed += 1;
      });
      if (merged.size !== project.prompts.h3Prompts.length || succeeded) {
        onChange({ ...project, prompts: { ...project.prompts, h3Prompts: [...merged.values()] } });
      }
      setPromptGenerationProgress({ completed: succeeded + failed, total: targets.length });
      if (controller.signal.aborted) {
        setActionMessage(`已停止视频提示词生成；本次已完成 ${succeeded + failed}/${targets.length} 个片段`);
      } else if (failed) {
        setActionError(`提示词仅部分完成：${succeeded} 个成功，${failed} 个失败。可再次点击补全失败片段。`);
      } else {
        setActionMessage(`视频提示词已全部生成并保存，共 ${succeeded} 个片段`);
      }
    } catch (cause) {
      if (controller.signal.aborted) {
        setActionMessage(`已停止视频提示词生成；已完成 ${succeeded + failed}/${targets.length} 个片段`);
      } else {
        setActionError(cause instanceof Error ? cause.message : "提示词自动生成失败");
      }
    } finally {
      if (promptGenerationAbort.current === controller) {
        promptGenerationAbort.current = null;
        setPromptGenerationBusy(false);
      }
    }
  }

  const stopPromptGeneration = () => {
    promptGenerationAbort.current?.abort();
    setActionMessage("正在停止视频提示词生成；已完成的片段会保留...");
  };

  const saveDraft = async (value = project) => {
    setActionBusy(true);
    setActionMessage(null);
    try {
      const saved = await saveProjectToControlPlane(value);
      await setProjectMode(saved.projectId, saved.executionMode);
      await setProjectPaused(saved.projectId, saved.paused);
      await setReviewMode(
        saved.projectId,
        saved.reviewPolicy.configuredMode,
        saved.reviewPolicy.humanTimeoutSeconds,
      );
      onChange(saved);
      setActionMessage("修订已保存");
      return saved;
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "保存失败");
      return null;
    } finally {
      setActionBusy(false);
    }
  };

  const approve = async () => {
    const current = project.activeStage;
    if (!completion[current].valid) return;
    const nextIndex = Math.min(pipelineOrder.length - 1, activeIndex + 1);
    const approved = {
      ...project,
      stageApprovals: { ...project.stageApprovals, [current]: new Date().toISOString() },
      activeStage: pipelineOrder[nextIndex],
    };
    await saveDraft(approved);
  };

  const selectStage = (stage: PipelineStageId) => {
    onChange(persistProject({ ...project, activeStage: stage }));
  };

  const addOutlineBeat = () => update("outline", (current) => ({
    ...current,
    outline: [...current.outline, {
      id: uid("beat"),
      title: `段落 ${current.outline.length + 1}`,
      summary: "",
      durationSeconds: Math.max(1, Math.round(current.targetDurationSeconds / Math.max(1, current.outline.length + 1))),
    }],
  }));

  const patchBeat = (id: string, values: Partial<OutlineBeat>) => update("outline", (current) => ({
    ...current,
    outline: current.outline.map((beat) => beat.id === id ? { ...beat, ...values } : beat),
  }));

  const addShot = () => update("storyboard", (current) => ({
    ...current,
    shots: [...current.shots, {
      id: uid("shot"),
      title: `镜头 ${current.shots.length + 1}`,
      summary: "",
      camera: "固定机位",
      seed: 0,
      durationSeconds: 8,
      motionSegments: [],
      locked: false,
    }],
  }));

  const patchShot = (id: string, values: Partial<ShotDraft>) => update("storyboard", (current) => ({
    ...current,
    shots: current.shots.map((shot) => shot.id === id ? { ...shot, ...values } : shot),
  }));

  const customizeMotionSegments = (shot: ShotDraft) => patchShot(shot.id, {
    motionSegments: shot.motionSegments.length
      ? []
      : automaticMotionSegmentDurations(shot.durationSeconds).map((durationSeconds, index) => ({
        id: uid("motion"),
        durationSeconds,
        summary: index === 0 ? shot.summary : "承接上一段结束状态并继续动作",
      })),
  });

  const addMotionSegment = (shot: ShotDraft) => {
    const source = shot.motionSegments.length ? [...shot.motionSegments] : automaticMotionSegmentDurations(shot.durationSeconds).map((durationSeconds, index) => ({ id: uid("motion"), durationSeconds, summary: index === 0 ? shot.summary : "承接上一段结束状态并继续动作" }));
    const splitIndex = source.reduce((best, segment, index) => segment.durationSeconds > source[best].durationSeconds ? index : best, 0);
    if (source[splitIndex].durationSeconds < 8) {
      setActionError("每个 H3 可见片段至少需要 4 秒；当前没有可继续拆分的片段");
      return;
    }
    const first = Math.round((source[splitIndex].durationSeconds / 2) * 1000) / 1000;
    const second = Math.round((source[splitIndex].durationSeconds - first) * 1000) / 1000;
    source.splice(splitIndex, 1,
      { ...source[splitIndex], durationSeconds: first },
      { id: uid("motion"), durationSeconds: second, summary: "承接上一段结束状态并继续动作" },
    );
    patchShot(shot.id, { motionSegments: source });
  };

  const removeMotionSegment = (shot: ShotDraft, segmentIndex: number) => {
    if (shot.motionSegments.length <= 1) {
      patchShot(shot.id, { motionSegments: [] });
      return;
    }
    const next = shot.motionSegments.filter((_, index) => index !== segmentIndex);
    const mergeIndex = Math.max(0, segmentIndex - 1);
    next[mergeIndex] = { ...next[mergeIndex], durationSeconds: Math.round((next[mergeIndex].durationSeconds + shot.motionSegments[segmentIndex].durationSeconds) * 1000) / 1000 };
    patchShot(shot.id, { motionSegments: next });
  };

  const queueAssetUploads = (event: ChangeEvent<HTMLInputElement>, planId: string | null = null) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!files.length) return;
    const queued = files.map((file) => ({ file, planId }));
    setPendingUploads(queued);
    const plan = planId ? project.assetPlans.find((item) => item.id === planId) : null;
    setPendingUploadName(plan?.name ?? files[0].name.replace(/\.[^.]+$/, ""));
    setPendingUploadKind(plan?.kind ?? "scene");
  };

  const confirmAssetUpload = async () => {
    const pending = pendingUploads[0];
    const name = pendingUploadName.trim();
    if (!pending || !name) {
      setActionError("请输入素材名称");
      return;
    }
    if (project.assets.some((asset) => asset.name.localeCompare(name, undefined, { sensitivity: "accent" }) === 0)) {
      setActionError(`素材名称“${name}”已存在，请使用唯一名称`);
      return;
    }
    setActionBusy(true);
    setActionMessage(null);
    try {
      const plan = pending.planId ? project.assetPlans.find((item) => item.id === pending.planId) : null;
      const uploadScope = plan?.scope === "shot" && plan.shotId ? "shot" : "public";
      const uploaded = await uploadProjectAsset(
        project.projectId,
        pending.file,
        name,
        pendingUploadKind,
        uploadScope,
        uploadScope === "shot" ? plan?.shotId ?? null : null,
      );
      const changed = {
        ...project,
        assets: [...project.assets, uploaded],
        assetPlans: project.assetPlans.map((item) => item.id === pending.planId
          ? { ...item, fulfilledByAssetId: uploaded.id, state: "satisfied" as const }
          : item),
        prompts: {
          ...project.prompts,
          imagePrompts: project.prompts.imagePrompts.filter((item) => item.assetPlanId !== pending.planId),
          h3Prompts: pending.planId ? [] : project.prompts.h3Prompts,
          generatedAt: pending.planId ? null : project.prompts.generatedAt,
        },
      };
      const invalidationStage = pending.planId ? "assets" : "idea";
      onChange(await saveProjectToControlPlane(invalidateFromStage(changed, invalidationStage, pipelineOrder)));
      const remaining = pendingUploads.slice(1);
      setPendingUploads(remaining);
      if (remaining.length) {
        const nextPlan = remaining[0].planId ? project.assetPlans.find((item) => item.id === remaining[0].planId) : null;
        setPendingUploadName(nextPlan?.name ?? remaining[0].file.name.replace(/\.[^.]+$/, ""));
        setPendingUploadKind(nextPlan?.kind ?? "scene");
      } else {
        setPendingUploadKind("scene");
      }
      setActionMessage(pending.planId
        ? `已上传并绑定素材“${name}”；旧视频提示词和任务计划已失效，请重新生成提示词并重新创建任务计划`
        : `已上传并登记素材“${name}”`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "图片上传失败");
    } finally {
      setActionBusy(false);
    }
  };

  const fillImagePromptWithAI = async () => {
    if (!imagePromptDialog) return;
    const plan = project.assetPlans.find((item) => item.id === imagePromptDialog.planId);
    if (!plan) return;
    const selectedWorkflow = imageWorkflowForPlan(plan.id);
    if (!selectedWorkflow) {
      setActionError("请先为该素材需求选择图片工作流");
      return;
    }
    setActionBusy(true);
    setActionMessage(`正在为“${plan.name}”编写图片提示词...`);
    try {
      const saved = await saveProjectToControlPlane(project);
      setImagePromptDialog((current) => current ? { ...current, prompt: "" } : current);
      const changed = await streamGenerateImagePrompt(
        saved.projectId,
        plan.id,
        imagePromptDialog.instruction.trim() || null,
        selectedWorkflow,
        (delta) => setImagePromptDialog((current) => current ? { ...current, prompt: `${current.prompt}${delta}` } : current),
      );
      const generated = changed.prompts.imagePrompts.find((item) => item.assetPlanId === plan.id);
      if (!generated?.prompt.trim()) throw new Error("LLM 未返回该素材需求的有效图片提示词");
      onChange(changed);
      setImagePromptDialog({
        planId: plan.id,
        instruction: "",
        prompt: generated.prompt,
        negativePrompt: generated.negativePrompt,
      });
      setActionMessage("AI 已将图片提示词写入统一编辑框，请确认或继续修改");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "图片提示词生成失败");
    } finally { setActionBusy(false); }
  };

  const generateAllImagePrompts = async () => {
    const targets = project.assetPlans.filter((plan) => Boolean(imageWorkflowForPlan(plan.id)));
    if (!targets.length) {
      setActionError("没有可用的图片素材需求或已批准图片工作流");
      return;
    }
    setActionBusy(true);
    setActionMessage(`正在并发生成 ${targets.length} 份图片提示词...`);
    try {
      const results = await Promise.allSettled(targets.map(async (plan) => {
        const generated = await streamGenerateImagePrompt(
          project.projectId,
          plan.id,
          null,
          imageWorkflowForPlan(plan.id),
          () => undefined,
        );
        return generated.prompts.imagePrompts.find((item) => item.assetPlanId === plan.id);
      }));
      const byPlan = new Map(project.prompts.imagePrompts.map((item) => [item.assetPlanId, item]));
      let completed = 0;
      results.forEach((result) => {
        if (result.status === "fulfilled" && result.value?.prompt.trim()) {
          byPlan.set(result.value.assetPlanId, result.value);
          completed += 1;
        }
      });
      onChange({ ...project, prompts: { ...project.prompts, imagePrompts: [...byPlan.values()] } });
      if (completed !== targets.length) setActionError(`图片提示词部分完成：${completed}/${targets.length}`);
      else setActionMessage(`图片提示词已全部生成，共 ${completed} 份`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "图片提示词批量生成失败");
    } finally {
      setActionBusy(false);
    }
  };

  const generateAllAssets = async () => {
    const targets = project.assetPlans.filter((plan) => (
      !plan.fulfilledByAssetId
      && Boolean(imageWorkflowForPlan(plan.id))
      && !activeImageTasks[plan.id]
    ));
    if (!targets.length) {
      setActionError("没有待生成素材，或尚未登记已批准的图片工作流");
      return;
    }
    setActionBusy(true);
    setActionMessage(`正在准备并并发生成 ${targets.length} 份素材...`);
    try {
      const results = await Promise.allSettled(targets.map(async (plan) => {
        const existing = project.prompts.imagePrompts.find((item) => item.assetPlanId === plan.id);
        if (!existing?.prompt.trim()) {
          await streamGenerateImagePrompt(
            project.projectId,
            plan.id,
            null,
            imageWorkflowForPlan(plan.id),
            () => undefined,
          );
        }
        return [plan.id, await runProjectImagePrompt(project.projectId, plan.id)] as const;
      }));
      const started = results.flatMap((result) => (
        result.status === "fulfilled" ? [result.value] : []
      ));
      if (started.length) {
        setActiveImageTasks((current) => ({
          ...current,
          ...Object.fromEntries(started.map(([planId, task]) => [planId, task.task_id])),
        }));
      }
      const failed = results.length - started.length;
      if (failed) setActionError(`素材批量任务部分创建失败：${started.length}/${results.length}`);
      else setActionMessage(`已派发 ${started.length} 份素材，完成后会自动登记并显示`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "素材批量生成失败");
    } finally {
      setActionBusy(false);
    }
  };

  const saveImagePromptAndRun = async () => {
    if (!imagePromptDialog) return;
    const plan = project.assetPlans.find((item) => item.id === imagePromptDialog.planId);
    if (!plan || !imagePromptDialog.prompt.trim()) {
      setActionError("图片提示词不能为空");
      return;
    }
    const existing = project.prompts.imagePrompts.find((item) => item.assetPlanId === plan.id);
    const defaultWorkflow = imageWorkflows.find(
      (item) => item.id === selectedWorkflowByPlan[plan.id],
    ) ?? imageWorkflows[0];
    if (!existing && !defaultWorkflow) {
      setActionError("请先登记并批准一个图片工作流");
      return;
    }
    setActionBusy(true);
    try {
      const entry = existing ? {
        ...existing,
        prompt: imagePromptDialog.prompt,
        negativePrompt: imagePromptDialog.negativePrompt,
        revision: existing.revision + 1,
      } : {
        id: `image-${crypto.randomUUID()}`,
        assetPlanId: plan.id,
        prompt: imagePromptDialog.prompt,
        negativePrompt: imagePromptDialog.negativePrompt,
        workflowTemplateId: defaultWorkflow!.id,
        harnessRevision: null,
        referenceAssetIds: [],
        locked: false,
        revision: 1,
      };
      const changed = {
        ...project,
        assetPlans: project.assetPlans.map((item) => item.id === plan.id ? { ...item, state: "ready" as const } : item),
        prompts: {
          ...project.prompts,
          imagePrompts: existing
            ? project.prompts.imagePrompts.map((item) => item.assetPlanId === plan.id ? entry : item)
            : [...project.prompts.imagePrompts, entry],
          h3Prompts: [],
          generatedAt: null,
        },
      };
      const saved = await saveProjectToControlPlane(invalidateFromStage(changed, "assets", pipelineOrder));
      onChange(saved);
      if (!workerOnline) throw new Error("提示词已保存，但 ComfyUI Worker 离线，暂时不能生成图片");
      const task = await runProjectImagePrompt(saved.projectId, plan.id);
      setActiveImageTasks((current) => ({ ...current, [plan.id]: task.task_id }));
      setImagePromptDialog(null);
      setActionMessage(`“${plan.name}”的提示词已保存，图片任务正在运行`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "保存并生成图片失败");
    } finally { setActionBusy(false); }
  };

  const openImagePromptDialog = (plan: ProjectDraft["assetPlans"][number]) => {
    const existing = project.prompts.imagePrompts.find((item) => item.assetPlanId === plan.id);
    setImagePromptDialog({
      planId: plan.id,
      instruction: "",
      prompt: existing?.prompt ?? "",
      negativePrompt: existing?.negativePrompt ?? "",
    });
  };

  const patchH3Prompt = (slot: H3SegmentSlot, promptText: string) => {
    const current = latestProject.current;
    const existingIndex = current.prompts.h3Prompts.findIndex((item) => (
      item.segmentId === slot.segmentId
      || (item.shotId === slot.shotId && item.segmentIndex === slot.segmentIndex)
    ));
    const changed = (() => {
      const next = existingIndex >= 0
        ? current.prompts.h3Prompts.map((item, index) => index === existingIndex
          ? { ...item, prompt: promptText, review: { ...item.review, ready: false } }
          : item)
        : [...current.prompts.h3Prompts, {
          id: `draft-${slot.segmentId}`,
          shotId: slot.shotId,
          segmentId: slot.segmentId,
          segmentIndex: slot.segmentIndex,
          durationSeconds: slot.durationSeconds,
          continuationOf: slot.continuationOf,
          inputMode: "t2va" as const,
          prompt: promptText,
          endState: null,
          assetIds: [],
          seed: slot.seed,
          harnessRevision: null,
          locked: false,
          revision: 1,
          legacy: false,
          review: { ready: false, issues: [], reviewedAt: null },
        }];
      return { ...current, prompts: { ...current.prompts, h3Prompts: next } };
    })();
    latestProject.current = changed;
    onChange(changed);
  };

  const generateOneH3Prompt = async (slot: H3SegmentSlot) => {
    setActionBusy(true);
    setActionMessage(`正在生成“${project.shots.find((shot) => shot.id === slot.shotId)?.title ?? slot.segmentId}”第 ${slot.segmentIndex + 1} 段提示词...`);
    try {
      const saved = await saveProjectToControlPlane(project);
      let streamed = "";
      const generated = await streamRegenerateH3Prompt(saved.projectId, slot.segmentId, (delta) => {
        streamed += delta;
        const preview = extractH3Description(streamed);
        if (preview) patchH3Prompt(slot, preview);
      });
      onChange(generated);
      const failed = generated.prompts.generationSummary?.failedSegmentIds?.includes(slot.segmentId);
      if (failed) {
        const issue = generated.prompts.h3Prompts.find((item) => item.segmentId === slot.segmentId)?.review.issues[0];
        throw new Error(issue?.message || "视频提示词生成失败，请查看该片段的问题详情");
      }
      setActionMessage("视频提示词已生成并写入文本框");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "视频提示词生成失败");
    } finally { setActionBusy(false); }
  };

  const translatePrompt = async (promptId: string) => {
    if (promptId.startsWith("draft-")) {
      setActionError("请先使用 AI 编写并保存英文执行稿，再生成中文对照");
      return;
    }
    setTranslatingPromptId(promptId);
    setActionError(null);
    try {
      const result = await translateH3Prompt(project.projectId, promptId);
      setPromptTranslations((current) => ({ ...current, [promptId]: result.translation }));
      setActionMessage("中文对照已生成；英文执行稿和审核状态未改变");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "中文对照生成失败");
    } finally {
      setTranslatingPromptId(null);
    }
  };

  const dismissAssetPlan = async (planId: string) => {
    setActionBusy(true);
    try {
      const remainingPlans = project.assetPlans.filter((item) => item.id !== planId);
      const changed = invalidateFromStage({
        ...project,
        assetPlans: remainingPlans,
        referenceAssetMode: remainingPlans.length ? project.referenceAssetMode : "none",
        prompts: { ...project.prompts, imagePrompts: project.prompts.imagePrompts.filter((item) => item.assetPlanId !== planId) },
      }, "assets", pipelineOrder);
      onChange(await saveProjectToControlPlane(changed));
      setActionMessage("已移除该素材需求，视频模型将不依赖这张参考图");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "移除素材需求失败");
    } finally { setActionBusy(false); }
  };

  const patchAsset = async (asset: AssetDraft, changes: Partial<Pick<AssetDraft, "name" | "kind" | "scope" | "shotId" | "shotIds">>) => {
    const next = { ...asset, ...changes };
    if (next.name.trim() !== asset.name && project.assets.some((item) => item.id !== asset.id && item.name.toLocaleLowerCase() === next.name.trim().toLocaleLowerCase())) {
      setActionMessage(`素材名称“${next.name}”已存在`);
      return;
    }
    setActionBusy(true);
    try {
      const saved = await updateProjectAsset(project.projectId, asset.id, { ...next, name: next.name.trim() });
      update("assets", (current) => ({
        ...current,
        idea: {
          ...current.idea,
          concept: current.idea.conceptDocument.nodes.map((node) => node.type === "asset_mention" && node.assetId === saved.id ? `@${saved.name}` : node.type === "asset_mention" ? `@${node.displayName}` : node.text).join(""),
          conceptDocument: { nodes: current.idea.conceptDocument.nodes.map((node) => node.type === "asset_mention" && node.assetId === saved.id ? { ...node, displayName: saved.name } : node) },
        },
        assets: current.assets.map((item) => item.id === asset.id ? saved : item),
      }));
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "素材修改失败");
    } finally { setActionBusy(false); }
  };

  const removeAsset = async (asset: AssetDraft) => {
    setActionBusy(true);
    try {
      const changed = invalidateFromStage(
        detachAsset(project, asset.id),
        project.activeStage === "idea" ? "idea" : "assets",
        pipelineOrder,
      );
      const saved = await saveProjectToControlPlane(changed);
      await deleteProjectAsset(project.projectId, asset.id);
      onChange(saved);
      setActionMessage(`已删除素材“${asset.name}”`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "删除素材失败");
    } finally { setActionBusy(false); }
  };

  const relinkAsset = async (asset: AssetDraft, file: File) => {
    setActionBusy(true);
    try {
      const restored = await relinkProjectAsset(project.projectId, asset.id, file);
      const changed = {
        ...project,
        assets: project.assets.map((item) => item.id === asset.id ? restored : item),
      };
      onChange(await saveProjectToControlPlane(invalidateFromStage(changed, "assets", pipelineOrder)));
      setActionMessage(`已重新关联“${asset.name}”`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "重新关联失败");
    } finally { setActionBusy(false); }
  };

  const compileDag = async () => {
    setActionBusy(true);
    setActionMessage("正在检查提示词、工作流和运行配置并创建任务计划...");
    try {
      const saved = await saveProjectToControlPlane(project);
      onChange(saved);
      const result = await compileProjectTasks(saved.projectId);
      await refreshExecutionStatus();
      setActionMessage(`已创建 ${result.tasks.length} 个任务；尚未启动 GPU`);
      return result.tasks;
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "任务计划创建失败");
      return null;
    } finally { setActionBusy(false); }
  };

  const startGeneration = async () => {
    if (!workerOnline) {
      setActionError("ComfyUI Worker 离线，无法启动视频生成");
      return;
    }
    setActionBusy(true);
    try {
      const status = await ensureCompiledDag();
      await startProjectGeneration(project.projectId);
      await refreshExecutionStatus();
      setActionMessage(status.compiled ? "后台调度已启动；关闭页面后仍会继续" : "生成批次已启动");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "启动生成失败");
    } finally {
      setActionBusy(false);
    }
  };

  const restartGeneration = async () => {
    if (!window.confirm("将取消当前 MiniMax H3 执行，重新读取全局设置，并从条件编码开始生成。已确认的素材和提示词会保留。是否继续？")) return;
    setActionBusy(true);
    setActionError(null);
    setActionMessage("正在重新读取设置并重建 MiniMax H3 执行链...");
    let restarted = false;
    try {
      const saved = await saveProjectToControlPlane(project);
      onChange(saved);
      const result = await compileProjectTasks(saved.projectId, true);
      await refreshExecutionStatus();
      const profile = result.h3_execution_profile;
      setActionMessage(profile
        ? `已读取最新设置：${profile.diffusion_model} · ${profile.turbo_enabled ? "Turbo" : "标准模式"} · ${profile.steps} 步；已重建 ${result.tasks.length} 个任务`
        : `已重新读取设置并重建 ${result.tasks.length} 个任务`);
      restarted = true;
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "重新开始 MiniMax H3 生成失败");
    } finally {
      setActionBusy(false);
    }
    if (restarted) await startProjectGeneration(project.projectId);
  };

  const ensureCompiledDag = async (target: "generation" | "delivery" = "generation") => {
    const saved = await saveProjectToControlPlane(project);
    onChange(saved);
    const compiled = target === "delivery"
      ? await compileProjectDeliveryTasks(saved.projectId)
      : await compileProjectTasks(saved.projectId);
    const status = await getProjectExecutionStatus(saved.projectId);
    setExecutionStatus(status);
    setActionMessage(`已创建 ${compiled.tasks.length} 个任务，准备从流程中执行`);
    return status;
  };

  const runImagePrompt = async (assetPlanId: string) => {
    if (!workerOnline) {
      setActionError("ComfyUI Worker 离线，无法启动图片生成");
      return;
    }
    setActionBusy(true);
    try {
      const saved = await saveProjectToControlPlane(project);
      onChange(saved);
      const task = await runProjectImagePrompt(saved.projectId, assetPlanId);
      setActiveImageTasks((current) => ({ ...current, [assetPlanId]: task.task_id }));
      setActionMessage("图片任务正在运行");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "图片任务提交失败");
    } finally { setActionBusy(false); }
  };

  const regenerateAssetPlan = async (plan: ProjectDraft["assetPlans"][number]) => {
    const existing = project.prompts.imagePrompts.find((prompt) => prompt.assetPlanId === plan.id && prompt.prompt.trim());
    if (!existing) {
      setActionError("该素材没有可复用的图片提示词，请先使用 AI 修改或手动编辑");
      return;
    }
    if (!workerOnline) {
      setActionError("ComfyUI Worker 离线，无法重新生成图片");
      return;
    }
    setActionBusy(true);
    setActionMessage("正在按原提示词生成候选版本；当前素材会保持不变...");
    try {
      // A revision bump intentionally creates a new idempotency fingerprint while preserving prompt text.
      const changed = {
        ...project,
        prompts: {
          ...project.prompts,
          imagePrompts: project.prompts.imagePrompts.map((prompt) => prompt.id === existing.id
            ? { ...prompt, revision: prompt.revision + 1 }
            : prompt),
        },
      };
      const saved = await saveProjectToControlPlane(changed);
      onChange(saved);
      const task = await regenerateProjectAsset(saved.projectId, plan.id);
      setActiveImageTasks((current) => ({ ...current, [plan.id]: task.task_id }));
      setCandidateTaskPlans((current) => ({ ...current, [plan.id]: true }));
      setActionMessage("已创建候选图片任务；生成完成后可对比、接受或放弃");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "图片重新生成失败");
    } finally {
      setActionBusy(false);
    }
  };

  const resolveAssetCandidate = async (candidate: AssetGenerationCandidate, accept: boolean) => {
    setActionBusy(true);
    try {
      if (accept) {
        await acceptAssetCandidate(candidate.candidate_id);
        const [workspace, assets] = await Promise.all([
          getProjectWorkspace(project.projectId),
          listProjectAssets(project.projectId),
        ]);
        onChange({
          ...workspace,
          assets,
          prompts: {
            ...workspace.prompts,
            h3Prompts: [],
            generatedAt: null,
          },
        });
        setActionMessage("已接受候选版本；旧视频提示词已失效，请按最新参考重新生成");
      } else {
        await discardAssetCandidate(candidate.candidate_id);
        setActionMessage("已放弃候选版本，当前素材保持不变");
      }
      const candidates = await listAssetCandidates(project.projectId, candidate.asset_plan_id);
      setAssetCandidatesByPlan((current) => ({ ...current, [candidate.asset_plan_id]: candidates }));
      setCandidatePreviewId(null);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "处理图片候选版本失败");
    } finally {
      setActionBusy(false);
    }
  };

  const runNextTask = async () => {
    setActionBusy(true);
    setActionMessage(null);
    try {
      if (!workerOnline) throw new Error("ComfyUI Worker 离线，任务不会被派发");
      const status = await ensureCompiledDag();
      const task = status.tasks.find((item) => item.state === "ready");
      if (!task) {
        setActionMessage(status.tasks.some((item) => ["queued", "running"].includes(item.state))
          ? "已有任务正在执行；完成后会解锁下一项"
          : "当前没有可执行任务，请检查失败项或等待前置任务");
        return;
      }
      await runTask(task.task_id);
      await refreshExecutionStatus();
      setActionMessage(`已从流程提交：${taskKindLabels[task.kind]}`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "任务提交失败");
    } finally {
      setActionBusy(false);
    }
  };

  const waitForGuidedPoll = (controller: AbortController) => new Promise<void>((resolve) => {
    const timer = window.setTimeout(resolve, 1500);
    controller.signal.addEventListener("abort", () => {
      window.clearTimeout(timer);
      resolve();
    }, { once: true });
  });

  const runGuidedTo = async (target: "review" | "delivery") => {
    if (guidedRunTarget) return;
    if (!workerOnline && target === "review") {
      setActionError("ComfyUI Worker 离线，无法执行视频生成链");
      return;
    }
    const controller = new AbortController();
    guidedRunAbort.current = controller;
    setGuidedRunTarget(target);
    setActionMessage(target === "review" ? "正在准备并执行到审核门..." : "正在执行后处理与导出...");
    try {
      await ensureCompiledDag(target === "delivery" ? "delivery" : "generation");
      while (!controller.signal.aborted) {
        const status = await getProjectExecutionStatus(project.projectId);
        setExecutionStatus(status);
        const relevant = target === "review"
          ? status.tasks.filter((task) => reviewBoundaryKinds.has(task.kind))
          : status.tasks.filter((task) => deliveryTaskKinds.has(task.kind));
        const failed = relevant.find((task) => task.state === "failed"
          && task.error_code !== "ai_review_rework_queued"
          && !(target === "delivery" && task.kind === "ai_review"));
        if (failed) {
          throw new Error(`${taskKindLabels[failed.kind]}失败：${failed.error_message || "请在当前流程中重试"}`);
        }
        if (target === "review" && status.review_complete) {
          setActionMessage("视频生成与审核已完成，可以进入审核阶段确认");
          break;
        } else if (target === "delivery" && status.delivery_complete) {
          setActionMessage("后处理与最终视频导出已完成");
          break;
        }

        const ready = relevant.filter((task) => task.state === "ready");
        if (ready.length) {
          for (const task of ready) {
            if (controller.signal.aborted) break;
            await runTask(task.task_id);
          }
          setActionMessage(`已派发 ${ready.length} 项；等待执行完成并自动继续...`);
        } else if (relevant.some((task) => ["queued", "running"].includes(task.state))) {
          const waitingForHuman = relevant.filter((task) => task.state === "needs_review").length;
          setActionMessage(waitingForHuman
            ? `${waitingForHuman} 个片段等待人工确认；其他片段仍在审核`
            : "正在等待已派发任务完成...");
        } else if (target === "review" && relevant.some((task) => task.state === "needs_review")) {
          const waitingForHuman = relevant.filter((task) => task.state === "needs_review").length;
          setActionMessage(`其余片段已审核，${waitingForHuman} 个片段等待人工确认`);
          break;
        } else if (!relevant.some((task) => task.state === "blocked")) {
          throw new Error("当前流程没有可继续的任务，请重新创建任务计划或检查失败原因");
        }
        await waitForGuidedPoll(controller);
      }
      if (controller.signal.aborted) setActionMessage("已停止自动派发；正在运行的原子任务会继续完成");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "流程执行失败");
    } finally {
      if (guidedRunAbort.current === controller) {
        guidedRunAbort.current = null;
        setGuidedRunTarget(null);
      }
      await refreshExecutionStatus();
    }
  };

  const stopGuidedRun = () => guidedRunAbort.current?.abort();

  const operatePipelineTask = async (task: TaskSpec, operation: "retry" | "cancel") => {
    setActionBusy(true);
    setActionError(null);
    setActionMessage(operation === "retry" ? `正在重新提交：${taskKindLabels[task.kind]}...` : null);
    try {
      if (operation === "cancel") await cancelTask(task.task_id);
      else await runTask(task.task_id);
      await refreshExecutionStatus();
      setActionMessage(operation === "cancel" ? "已取消当前任务" : "已重新提交失败任务");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "任务操作失败");
    } finally {
      setActionBusy(false);
    }
  };

  const openReworkDialog = (version: SegmentGenerationVersion) => {
    const prompt = project.prompts.h3Prompts.find((item) => item.segmentId === version.segment_id);
    setReworkDialog({
      version,
      segmentId: version.segment_id,
      action: "revise_prompt",
      feedback: "",
      replacementSeed: (prompt?.seed ?? 0) + 1,
    });
  };

  const submitRework = async () => {
    if (!reworkDialog || !reworkDialog.feedback.trim()) return;
    setActionBusy(true);
    try {
      await createReworkMarker(project.projectId, {
        version_id: reworkDialog.version.version_id,
        action: reworkDialog.action,
        feedback: reworkDialog.feedback.trim(),
        replacement_seed: reworkDialog.action === "change_seed" ? reworkDialog.replacementSeed : null,
      });
      setReworkDialog(null);
      setActionMessage("已标记返工；该 Motion Context 序列已从问题段起冻结");
      await refreshExecutionStatus();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "创建返工任务失败");
    } finally {
      setActionBusy(false);
    }
  };

  const confirmAllReworks = async () => {
    setActionBusy(true);
    try {
      await confirmProjectReworks(project.projectId);
      await refreshExecutionStatus();
      setActionMessage("返工批次已确认，后台将按统一管线执行");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "确认返工失败");
    } finally {
      setActionBusy(false);
    }
  };

  const withdrawMarker = async (markerId: string) => {
    setActionBusy(true);
    try {
      await withdrawReworkMarker(project.projectId, markerId);
      await refreshExecutionStatus();
      setActionMessage("已撤销返工标记，后台会重新计算可执行片段");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "撤销返工失败");
    } finally {
      setActionBusy(false);
    }
  };

  const switchMode = async () => {
    setActionBusy(true);
    try {
      const mode = project.executionMode === "guided" ? "batch" : "guided";
      const state = await setProjectMode(project.projectId, mode);
      onChange({ ...project, executionMode: state.execution_mode });
      setActionMessage(state.pending_mode ? "将在当前原子任务结束后切换" : "执行方式已切换");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "模式切换失败");
    } finally {
      setActionBusy(false);
    }
  };

  return <section className="pipeline-workspace">
    <div className="run-toolbar">
      <div><strong>{project.executionMode === "guided" ? "精细化模式" : "批量模式"}</strong><span>{actionMessage ?? "项目进度和任务状态会自动保存"}</span></div>
      <button className="secondary-button" disabled={actionBusy} onClick={() => void saveDraft()}><Save size={15} />保存草稿</button>
      {project.stageApprovals.outline && project.outline.length > 0 && !executionStatus?.task_count && <button className="primary-button" disabled={actionBusy} onClick={onOpenBatch}><PackageCheck size={15} />加入批量队列</button>}
      {executionStatus?.task_count ? <button className="secondary-button" disabled={actionBusy} onClick={() => void (async () => {
        setActionBusy(true);
        try { const state = await setProjectPaused(project.projectId, !project.paused); onChange({ ...project, paused: state.paused }); }
        catch (cause) { setActionError(cause instanceof Error ? cause.message : "暂停失败"); }
        finally { setActionBusy(false); }
      })()}><Pause size={15} />{project.paused ? "继续派发" : "暂停"}</button> : null}
      {executionStatus?.task_count ? <button className="primary-button" disabled={actionBusy} onClick={() => void switchMode()}>{project.executionMode === "guided" ? "转入批量工作台" : "退出批量模式"}</button> : null}
    </div>
    {actionError && <div className="global-error-banner pipeline-error-banner" role="alert"><CircleAlert size={19} /><strong>操作未完成</strong><span>{actionError}</span><button className="icon-button" title="关闭错误" onClick={() => setActionError(null)}><X size={16} /></button></div>}
    <div className="pipeline-rail" aria-label="制作阶段">
      {pipelineOrder.map((stage, index) => {
        const approved = Boolean(project.stageApprovals[stage]) && completion[stage].valid;
        const unlocked = true;
        const current = project.activeStage === stage;
        return <button
          key={stage}
          className={`pipeline-step ${current ? "current" : ""} ${approved ? "complete" : ""}`}
          disabled={!unlocked}
          onClick={() => selectStage(stage)}
          title={stageMeta[stage].label}
        >
          <span className="pipeline-step-index">{approved ? <Check size={14} /> : unlocked ? stageMeta[stage].index : <LockKeyhole size={13} />}</span>
          <span>{stageMeta[stage].short}</span>
          {index < pipelineOrder.length - 1 && <ChevronRight className="pipeline-connector" size={14} />}
        </button>;
      })}
    </div>

    <div className="pipeline-layout">
      <div className="stage-surface">
        <header className="stage-header">
          <div><span className="stage-kicker">阶段 {stageMeta[project.activeStage].index}</span><h2>{stageMeta[project.activeStage].label}</h2></div>
          {project.stageApprovals[project.activeStage] && completion[project.activeStage].valid && <span className="approval-chip"><Check size={14} />已批准</span>}
          {project.stageApprovals[project.activeStage] && !completion[project.activeStage].valid && <span className="approval-chip stale"><CircleAlert size={14} />内容已变化，需重新批准</span>}
        </header>

        {project.activeStage === "config" && <ProjectView
          embedded
          project={project}
          onValidationChange={setConfigFormValid}
          onChange={(changed) => {
            const next = project.stageApprovals.config
              ? invalidateFromStage(changed, "config", pipelineOrder)
              : changed;
            onChange(persistProject(next));
          }}
        />}

        {project.activeStage === "idea" && <div className="stage-form two-column">
          <Field label="核心创意" wide><MentionTextarea value={project.idea.concept} assets={project.assets} onChange={(concept, conceptDocument) => update("idea", (current) => ({ ...current, idea: { ...current.idea, concept, conceptDocument } }))} /></Field>
          <Field label="类型"><input value={project.idea.genre} onChange={(event) => update("idea", (current) => ({ ...current, idea: { ...current.idea, genre: event.target.value } }))} /></Field>
          <Field label="视觉风格"><input value={project.idea.visualStyle} onChange={(event) => update("idea", (current) => ({ ...current, idea: { ...current.idea, visualStyle: event.target.value } }))} /></Field>
          <Field label="目标观众"><input value={project.idea.audience} onChange={(event) => update("idea", (current) => ({ ...current, idea: { ...current.idea, audience: event.target.value } }))} placeholder="可选" /></Field>
          <div className="idea-assets field-wide">
            <div className="inline-toolbar"><span>创意参考素材 · 输入 @名称 引用</span><label className="secondary-button" role="button" aria-label="上传并命名创意参考素材" tabIndex={0} onKeyDown={activateFileLabel}><Upload size={15} />上传并命名<input type="file" accept="image/*,video/*,audio/*" multiple onChange={(event) => queueAssetUploads(event)} /></label></div>
            <div className="idea-asset-strip">{project.assets.map((asset) => <div key={asset.id} className="idea-asset-chip">
              {asset.status === "ready" ? <button className="thumbnail-button" title={`预览 ${asset.name}`} onClick={() => setLightboxAssetId(asset.id)}>{asset.mediaKind === "image" ? <img src={asset.previewUrl || projectAssetPreviewUrl(project.projectId, asset.id)} alt={asset.name} /> : asset.mediaKind === "video" ? <Film size={18} /> : <Music size={18} />}</button> : <ImagePlus size={18} />}
              <span>@{asset.name}</span>
              <button className="icon-button" disabled={actionBusy} title={`删除素材 ${asset.name}`} aria-label={`删除素材 ${asset.name}`} onClick={() => void removeAsset(asset)}><Trash2 size={14} /></button>
            </div>)}{!project.assets.length && <span className="muted-copy">上传后即可在创意和对话中按名称引用</span>}</div>
          </div>
        </div>}

        {project.activeStage === "outline" && <div className="editor-list">
          <div className="inline-toolbar"><span>{project.outline.length} 个故事段落</span><button className="secondary-button" onClick={addOutlineBeat}><ListPlus size={16} />添加段落</button></div>
          {project.outline.map((beat, index) => <div className="editor-row" key={beat.id}>
            <span className="row-number">{String(index + 1).padStart(2, "0")}</span>
            <div className="row-fields">
              <input className="title-input" value={beat.title} onChange={(event) => patchBeat(beat.id, { title: event.target.value })} />
              <textarea rows={3} value={beat.summary} onChange={(event) => patchBeat(beat.id, { summary: event.target.value })} placeholder="情节推进、人物状态和转折" />
            </div>
            <div className="row-duration"><input type="number" min="1" value={beat.durationSeconds} onChange={(event) => patchBeat(beat.id, { durationSeconds: Math.max(1, Number(event.target.value) || 1) })} /><span>秒</span></div>
            <button className="icon-button" title="删除段落" onClick={() => update("outline", (current) => ({ ...current, outline: current.outline.filter((item) => item.id !== beat.id) }))}><Trash2 size={16} /></button>
          </div>)}
          {!project.outline.length && <button className="empty-action" onClick={addOutlineBeat}><Plus size={20} />添加第一个故事段落</button>}
        </div>}

        {project.activeStage === "storyboard" && <div className="editor-list">
          <div className="inline-toolbar"><span>{project.shots.length} 个电影分镜 · {shotsDuration} 秒 · 将拆分为 {segmentCount} 个视频片段</span><button className="secondary-button" onClick={addShot}><Plus size={16} />添加分镜</button></div>
          {project.shots.map((shot, index) => <div className="shot-row" key={shot.id}>
            <div className="shot-index"><span>{String(index + 1).padStart(2, "0")}</span><Film size={17} /></div>
            <div className="shot-main">
              <div className="shot-title-line"><input className="title-input" value={shot.title} onChange={(event) => patchShot(shot.id, { title: event.target.value })} />{buildH3SegmentSlots([shot]).length > 1 && <span className="continuation-chip">{buildH3SegmentSlots([shot]).length} 段连续生成</span>}</div>
              <textarea rows={3} value={shot.summary} onChange={(event) => patchShot(shot.id, { summary: event.target.value })} placeholder="画面内容、人物动作和环境变化" />
              <div className="shot-controls">
                <input value={shot.camera} onChange={(event) => patchShot(shot.id, { camera: event.target.value })} aria-label="运镜" />
                <label>Seed <input type="number" min="0" value={shot.seed} onChange={(event) => patchShot(shot.id, { seed: Math.max(0, Number(event.target.value) || 0) })} /></label>
                <label><input type="checkbox" checked={shot.locked} onChange={(event) => patchShot(shot.id, { locked: event.target.checked })} />锁定 LLM 字段</label>
              </div>
              <div className="motion-segment-editor">
                <header><div><strong>Motion Context 分段</strong><span>{shot.motionSegments.length ? "手工与 AI 共用此分段计划" : "当前按时长自动均分，可改为明确接缝"}</span></div><div><button className="secondary-button" onClick={() => customizeMotionSegments(shot)}>{shot.motionSegments.length ? "重置均分" : "编辑分段"}</button><button className="secondary-button" onClick={() => addMotionSegment(shot)}><Plus size={13} />增加分段</button></div></header>
                {shot.motionSegments.map((segment, segmentIndex) => <div className="motion-segment-row" key={segment.id}>
                  <span>C{String(segmentIndex + 1).padStart(2, "0")}</span>
                  <input type="number" min="4" max={segmentIndex === 0 ? 15 : 12} step="0.5" value={segment.durationSeconds} aria-label={`续段 ${segmentIndex + 1} 时长`} onChange={(event) => patchShot(shot.id, { motionSegments: shot.motionSegments.map((item, index) => index === segmentIndex ? { ...item, durationSeconds: Math.max(.5, Number(event.target.value) || .5) } : item) })} />
                  <textarea rows={2} value={segment.summary} aria-label={`续段 ${segmentIndex + 1} 内容`} onChange={(event) => patchShot(shot.id, { motionSegments: shot.motionSegments.map((item, index) => index === segmentIndex ? { ...item, summary: event.target.value } : item) })} placeholder="本段动作、机位变化，以及适合交给下一段继承的结束状态" />
                  <button className="icon-button" title="移除分段" onClick={() => removeMotionSegment(shot, segmentIndex)}><Trash2 size={14} /></button>
                </div>)}
                {shot.motionSegments.length ? <footer className={motionSegmentsValid(shot) ? "valid" : "invalid"}>计划合计 {shot.motionSegments.reduce((sum, segment) => sum + segment.durationSeconds, 0)} / {Math.max(4, shot.durationSeconds)} 秒；首段最多 15 秒，续段最多 12 秒</footer> : null}
              </div>
            </div>
            <div className="row-duration"><input type="number" min="1" step="0.5" value={shot.durationSeconds} onChange={(event) => patchShot(shot.id, { durationSeconds: Math.max(.5, Number(event.target.value) || .5) })} /><span>秒</span></div>
            <button className="icon-button" title="删除分镜" onClick={() => update("storyboard", (current) => ({ ...current, shots: current.shots.filter((item) => item.id !== shot.id) }))}><Trash2 size={16} /></button>
          </div>)}
          {!project.shots.length && <button className="empty-action" onClick={addShot}><Plus size={20} />添加第一个电影分镜</button>}
        </div>}

        {project.activeStage === "assets" && <div className="asset-stage">
          <div className="inline-toolbar"><span>{project.assets.length} 个素材 · 仅按分镜中的明确勾选引用</span><div className="toolbar-actions"><button className="secondary-button" onClick={onOpenWorkflows}><SlidersHorizontal size={16} />图片工作流</button><button className="primary-button" disabled={actionBusy || !workerOnline || !project.assetPlans.some((plan) => !plan.fulfilledByAssetId && Boolean(imageWorkflowForPlan(plan.id)))} onClick={() => void generateAllAssets()}><Play size={15} />生成全部素材</button><label className="primary-button" role="button" aria-label="上传项目参考素材" tabIndex={0} onKeyDown={activateFileLabel}><Upload size={16} />上传参考素材<input type="file" accept="image/*,video/*,audio/*" multiple onChange={(event) => queueAssetUploads(event)} /></label></div></div>
          <label className="check-label"><input type="checkbox" checked={project.referenceAssetMode === "none"} onChange={(event) => update("assets", (current) => ({ ...current, referenceAssetMode: event.target.checked ? "none" : "planned" }))} />本项目不需要参考素材，交由 H3 直接生成</label>
          <div className="asset-shot-map">
            <div className="inline-toolbar"><h3>分镜素材引用</h3><span>先按分镜查看引用，再在下方集中管理素材</span></div>
            {project.shots.map((shot, index) => {
              const plans = project.assetPlans.filter((plan) =>
                plan.shotIds.includes(shot.id) || plan.shotId === shot.id,
              );
              const referencedAssets = project.assets.filter((asset) => (
                asset.shotIds.includes(shot.id)
                || plans.some((plan) => plan.fulfilledByAssetId === asset.id)
              ));
              const togglePlanShot = (plan: ProjectDraft["assetPlans"][number], checked: boolean) => {
                const currentIds = plan.shotIds.length
                  ? plan.shotIds
                  : plan.shotId ? [plan.shotId] : [];
                const nextIds = checked
                  ? [...new Set([...currentIds, shot.id])]
                  : currentIds.filter((id) => id !== shot.id);
                update("assets", (current) => ({
                  ...current,
                  assetPlans: current.assetPlans.map((item) => item.id === plan.id
                    ? { ...item, scope: "public" as const, shotId: null, shotIds: nextIds }
                    : item),
                }));
              };
              return <div className="asset-shot-map-row" key={shot.id}>
                <strong>分镜 {index + 1} · {shot.title}</strong>
                <div className="asset-shot-map-options">
                  {project.assetPlans.length ? project.assetPlans.map((plan) => {
                    const checked = plan.shotIds.length
                      ? plan.shotIds.includes(shot.id)
                      : plan.shotId === shot.id;
                    const fulfilled = project.assets.find((asset) => asset.id === plan.fulfilledByAssetId);
                    return <label key={plan.id} className="asset-shot-map-option"><input type="checkbox" checked={checked} onChange={(event) => togglePlanShot(plan, event.target.checked)} /><span>@{fulfilled?.name ?? plan.name}</span></label>;
                  }) : <span>暂无素材需求</span>}
                  {project.assets.filter((asset) => asset.status === "ready" && !project.assetPlans.some((plan) => plan.fulfilledByAssetId === asset.id)).map((asset) => {
                    const checked = asset.shotIds.includes(shot.id);
                    return <label key={asset.id} className="asset-shot-map-option asset-shot-map-uploaded">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(event) => {
                          if (event.target.checked) {
                            void patchAsset(asset, { scope: "public", shotId: null, shotIds: [...new Set([...asset.shotIds, shot.id])] });
                          } else {
                            void patchAsset(asset, { scope: "public", shotId: null, shotIds: asset.shotIds.filter((id) => id !== shot.id) });
                          }
                        }}
                      />
                      {asset.mediaKind === "video" ? <Film size={13} /> : asset.mediaKind === "audio" ? <Music size={13} /> : <ImagePlus size={13} />}
                      @{asset.name}
                    </label>;
                  })}
                  {!plans.length && project.assetPlans.length > 0 && <small>当前未引用素材</small>}
                </div>
                <div className="asset-shot-map-assets">
                  <small>已上传 / 已生成素材</small>
                  {referencedAssets.length ? referencedAssets.map((asset) => (
                    <span className="asset-shot-map-asset" key={asset.id}>
                      {asset.mediaKind === "video" ? <Film size={12} /> : asset.mediaKind === "audio" ? <Music size={12} /> : <ImagePlus size={12} />}
                      @{asset.name}
                    </span>
                  )) : <span className="asset-shot-map-empty">暂无已绑定素材</span>}
                </div>
              </div>;
            })}
            {!project.shots.length && <div className="table-empty">请先完成分镜，素材需求才能绑定到具体镜头。</div>}
          </div>
          <div className="asset-grid">
            {project.assets.map((asset) => <div className="asset-item" key={asset.id}>
              <button className="asset-preview" disabled={asset.status !== "ready"} title={asset.status === "ready" ? `预览 ${asset.name}` : "素材不可预览"} onClick={() => setLightboxAssetId(asset.id)}>{asset.status !== "ready" ? <ImagePlus size={24} /> : asset.mediaKind === "image" ? <img src={asset.previewUrl || projectAssetPreviewUrl(project.projectId, asset.id)} alt={asset.name} /> : asset.mediaKind === "video" ? <Film size={24} /> : <Music size={24} />}</button>
              <div className="asset-info"><input aria-label="素材名称" defaultValue={asset.name} onBlur={(event) => { if (event.target.value.trim() && event.target.value.trim() !== asset.name) void patchAsset(asset, { name: event.target.value }); }} /><span>{asset.status === "missing_blob" ? "缺少原始文件" : asset.source === "upload" ? "本地上传" : "工作流生成"}</span></div>
              <select value={asset.kind} onChange={(event) => void patchAsset(asset, { kind: event.target.value as AssetDraft["kind"] })}><option value="character">角色</option><option value="scene">场景</option><option value="prop">道具</option><option value="style">风格</option></select>
              <select value={asset.scope} onChange={(event) => {
                const scope = event.target.value as AssetDraft["scope"];
                if (scope === "shot") {
                  const shotId = asset.shotId ?? project.shots[0]?.id ?? null;
                  if (!shotId) { setActionError("请先创建电影分镜，再将素材设为镜头专用"); return; }
                  void patchAsset(asset, { scope, shotId, shotIds: [shotId] });
                } else {
                  void patchAsset(asset, { scope, shotId: null });
                }
              }}><option value="public">公共</option><option value="shot" disabled={!project.shots.length}>镜头专用</option></select>
              {asset.scope === "shot" && <select value={asset.shotId ?? ""} onChange={(event) => { const shotId = event.target.value || null; if (shotId) void patchAsset(asset, { shotId, shotIds: [shotId] }); }}><option value="">选择分镜</option>{project.shots.map((shot) => <option key={shot.id} value={shot.id}>{shot.title}</option>)}</select>}
              {asset.status === "missing_blob" && <label className="secondary-button" role="button" aria-label={`重新关联素材 ${asset.name}`} tabIndex={0} onKeyDown={activateFileLabel}><Upload size={14} />重新关联<input type="file" accept="image/*,video/*,audio/*" onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void relinkAsset(asset, file); }} /></label>}
              <button className="icon-button" title="移除素材" onClick={() => void removeAsset(asset)}><Trash2 size={16} /></button>
            </div>)}
            {!project.assets.length && <label className="asset-empty" role="button" aria-label="上传图片、视频或音频参考" tabIndex={0} onKeyDown={activateFileLabel}><Upload size={24} /><span>上传图片、视频或音频参考</span><input type="file" accept="image/*,video/*,audio/*" multiple onChange={(event) => queueAssetUploads(event)} /></label>}
          </div>
          {project.assetPlans.length > 0 && <div className="asset-plan-list">
            <div className="inline-toolbar"><h3>AI 素材需求</h3><button className="secondary-button" onClick={() => void (async () => { const changed = invalidateFromStage({ ...project, assetPlans: [], referenceAssetMode: "none", prompts: { ...project.prompts, imagePrompts: [] } }, "assets", pipelineOrder); onChange(await saveProjectToControlPlane(changed)); setActionMessage("已忽略全部 AI 素材需求，H3 将直接生成画面"); })()}><Trash2 size={14} />忽略全部需求</button></div>
            {project.assetPlans.some((plan) => Boolean(plan.fulfilledByAssetId)) && <div className="direct-replacement-note"><CircleAlert size={14} />重新生成会先保存为候选版本；只有接受候选后才会替换当前素材。</div>}
            {project.assetPlans.map((plan) => {
              const fulfilled = project.assets.find((asset) => asset.id === plan.fulfilledByAssetId);
              const existingPrompt = project.prompts.imagePrompts.find((prompt) => prompt.assetPlanId === plan.id);
              const hasPrompt = Boolean(existingPrompt?.prompt.trim());
              const selectedWorkflow = imageWorkflowForPlan(plan.id) ?? "";
              const generating = Boolean(activeImageTasks[plan.id]);
              const pendingCandidate = assetCandidatesByPlan[plan.id]?.find((candidate) => candidate.state === "pending");
              return <div key={plan.id} className="asset-plan-row">
                <button className="asset-plan-preview" disabled={!fulfilled} title={fulfilled ? `放大预览 ${fulfilled.name}` : "尚未生成"} onClick={() => fulfilled && setLightboxAssetId(fulfilled.id)}>{fulfilled ? <img src={fulfilled.previewUrl || projectAssetPreviewUrl(project.projectId, fulfilled.id)} alt={fulfilled.name} /> : <ImagePlus size={19} />}</button>
                <div className="asset-plan-info"><strong>{fulfilled?.name ?? plan.name}</strong><p>{plan.description}</p><small>{plan.kind} · {plan.scope === "public" ? "公共素材" : `引用分镜：${plan.shotIds.length ? plan.shotIds.map((id) => project.shots.find((shot) => shot.id === id)?.title ?? id).join("、") : plan.shotId ? (project.shots.find((shot) => shot.id === plan.shotId)?.title ?? plan.shotId) : "未指定"}`}</small>{assetResolutionControls(plan)}</div>
                <span className={`state ${fulfilled ? "state-ready" : ""}`}>{generating ? "GENERATING" : fulfilled ? "SATISFIED" : hasPrompt ? "PROMPT READY" : plan.state.toUpperCase()}</span>
                <div className="asset-plan-actions">
                  {!fulfilled && !generating && <label className="asset-plan-workflow"><span>图片工作流</span><select value={selectedWorkflow} onChange={(event) => setSelectedWorkflowByPlan((current) => ({ ...current, [plan.id]: event.target.value }))}><option value="">请选择</option>{imageWorkflows.map((workflow) => <option value={workflow.id} key={workflow.id}>{workflow.name} · R{workflow.revision}</option>)}</select></label>}
                  {!fulfilled && !generating && <label className="secondary-button" role="button" aria-label={`上传图片填充素材需求 ${plan.name}`} tabIndex={0} onKeyDown={activateFileLabel}><Upload size={14} />上传填充<input type="file" accept="image/*" onChange={(event) => queueAssetUploads(event, plan.id)} /></label>}
                  {!generating && <button className={!fulfilled ? "primary-button" : "secondary-button"} disabled={actionBusy || !selectedWorkflow} onClick={() => openImagePromptDialog(plan)}><SlidersHorizontal size={14} />编辑提示词</button>}
                  {fulfilled?.source === "generated" && hasPrompt && !generating && !pendingCandidate && <button className="secondary-button" disabled={actionBusy || !workerOnline} title="生成候选版本，不会立即替换当前素材" onClick={() => void regenerateAssetPlan(plan)}><RotateCcw size={14} />按原提示词重新生成</button>}
                  <button className="icon-button" disabled={actionBusy} title="不需要此素材，由视频模型直接生成" onClick={() => void dismissAssetPlan(plan.id)}><Trash2 size={15} /></button>
                </div>
                {pendingCandidate && <div className="asset-candidate-comparison">
                  <div><span>当前版本</span>{fulfilled && <button type="button" onClick={() => setLightboxAssetId(fulfilled.id)}><img src={fulfilled.previewUrl || projectAssetPreviewUrl(project.projectId, fulfilled.id)} alt={`${fulfilled.name} 当前版本`} /></button>}</div>
                  <div><span>候选版本 · {pendingCandidate.width}×{pendingCandidate.height}</span><button type="button" onClick={() => setCandidatePreviewId(pendingCandidate.candidate_id)}><img src={assetCandidatePreviewUrl(pendingCandidate.candidate_id)} alt={`${pendingCandidate.name} 候选版本`} /></button></div>
                  <div className="asset-candidate-actions"><button className="primary-button" disabled={actionBusy} onClick={() => void resolveAssetCandidate(pendingCandidate, true)}><Check size={14} />接受候选</button><button className="secondary-button" disabled={actionBusy} onClick={() => void resolveAssetCandidate(pendingCandidate, false)}><X size={14} />放弃候选</button></div>
                </div>}
              </div>;
            })}
          </div>}
        </div>}

        {pendingUploads.length > 0 && <div className="asset-name-dialog-backdrop">
          <form className="asset-name-dialog" onSubmit={(event) => { event.preventDefault(); void confirmAssetUpload(); }}>
            <header><div><strong>为图片命名</strong><span>{pendingUploads[0].file.name}{pendingUploads.length > 1 ? ` · 还有 ${pendingUploads.length - 1} 张` : ""}</span></div></header>
            <label className="field"><span>素材名称</span><input autoFocus value={pendingUploadName} onChange={(event) => setPendingUploadName(event.target.value)} placeholder="在创意中可用 @名称 引用" /></label>
            <label className="field"><span>素材类型</span><select value={pendingUploadKind} onChange={(event) => setPendingUploadKind(event.target.value as AssetDraft["kind"])}><option value="scene">场景</option><option value="character">角色</option><option value="prop">道具</option><option value="style">风格</option></select></label>
            <footer><button type="button" className="secondary-button" disabled={actionBusy} onClick={() => { setPendingUploads([]); setPendingUploadKind("scene"); }}>取消</button><button type="submit" className="primary-button" disabled={actionBusy || !pendingUploadName.trim()}>{actionBusy ? "正在上传..." : "上传并登记"}</button></footer>
          </form>
        </div>}

        {imagePromptDialog && <div className="asset-name-dialog-backdrop">
          <form className="asset-name-dialog image-prompt-dialog" onSubmit={(event) => {
            event.preventDefault();
            void saveImagePromptAndRun();
          }}>
            <header><div><strong>图片提示词</strong><span>{project.assetPlans.find((item) => item.id === imagePromptDialog.planId)?.name} · 手工与 AI 共用</span></div></header>
            <div className="prompt-ai-command"><label className="field"><span>交给 AI 的要求（可选）</span><textarea rows={3} value={imagePromptDialog.instruction} onChange={(event) => setImagePromptDialog({ ...imagePromptDialog, instruction: event.target.value })} placeholder="例如：保持人物身份不变，改为雨夜霓虹场景，使用低机位全身构图" /></label><button type="button" className="secondary-button" disabled={actionBusy} onClick={() => void fillImagePromptWithAI()}><Sparkles size={14} />AI 填入输入框</button></div>
            <label className="field"><span>正向提示词</span><textarea autoFocus rows={9} value={imagePromptDialog.prompt} onChange={(event) => setImagePromptDialog({ ...imagePromptDialog, prompt: event.target.value })} /></label>
            <label className="field"><span>负向提示词</span><textarea rows={4} value={imagePromptDialog.negativePrompt} onChange={(event) => setImagePromptDialog({ ...imagePromptDialog, negativePrompt: event.target.value })} /></label>
            <footer><button type="button" className="secondary-button" disabled={actionBusy} onClick={() => setImagePromptDialog(null)}>取消</button><button type="submit" className="primary-button" disabled={actionBusy || !imagePromptDialog.prompt.trim()}>{actionBusy ? "正在处理..." : "保存并生成"}</button></footer>
          </form>
        </div>}

        {candidatePreviewId && <div className="candidate-preview-backdrop" role="dialog" aria-modal="true" aria-label="候选图片预览" onClick={() => setCandidatePreviewId(null)}>
          <button className="icon-button candidate-preview-close" title="关闭预览" onClick={() => setCandidatePreviewId(null)}><X size={20} /></button>
          <img src={assetCandidatePreviewUrl(candidatePreviewId)} alt="候选图片大图预览" onClick={(event) => event.stopPropagation()} />
        </div>}

        {reworkDialog && <div className="asset-name-dialog-backdrop">
          <form className="asset-name-dialog rework-dialog" onSubmit={(event) => { event.preventDefault(); void submitRework(); }}>
            <header><div><strong>标记片段返工</strong><span>片段 {reworkDialog.segmentId} · 标记后立即冻结所属连续序列</span></div></header>
            <fieldset className="rework-actions"><legend>处理方式</legend>
              <label><input type="radio" name="rework-action" checked={reworkDialog.action === "retry"} onChange={() => setReworkDialog({ ...reworkDialog, action: "retry" })} /><span><strong>重跑原任务</strong><small>用于文件损坏、黑帧或执行异常</small></span></label>
              <label><input type="radio" name="rework-action" checked={reworkDialog.action === "change_seed"} onChange={() => setReworkDialog({ ...reworkDialog, action: "change_seed" })} /><span><strong>更换 Seed</strong><small>用于瞬态伪影、偶发形体或构图问题</small></span></label>
              <label><input type="radio" name="rework-action" checked={reworkDialog.action === "revise_prompt"} onChange={() => setReworkDialog({ ...reworkDialog, action: "revise_prompt" })} /><span><strong>修改提示词</strong><small>用于语义、身份或连续性偏差</small></span></label>
            </fieldset>
            {reworkDialog.action === "change_seed" && <label className="field"><span>新 Seed</span><input type="number" min="0" value={reworkDialog.replacementSeed} onChange={(event) => setReworkDialog({ ...reworkDialog, replacementSeed: Math.max(0, Number(event.target.value) || 0) })} /></label>}
            <label className="field"><span>驳回反馈（必填）</span><textarea autoFocus rows={6} required value={reworkDialog.feedback} onChange={(event) => setReworkDialog({ ...reworkDialog, feedback: event.target.value })} placeholder="描述问题、出现时间和期望修改，例如：2.1–3.0 秒人物身份漂移，保持服装和脸部特征不变。" /></label>
            <footer><button type="button" className="secondary-button" disabled={actionBusy} onClick={() => setReworkDialog(null)}>取消</button><button type="submit" className="primary-button" disabled={actionBusy || !reworkDialog.feedback.trim()}>{actionBusy ? "正在标记..." : "标记返工"}</button></footer>
          </form>
        </div>}

        {project.activeStage === "prompts" && <div className="prompt-stage">
          <div className="prompt-tabs" role="tablist">
            <button className={promptTab === "image" ? "active" : ""} onClick={() => setPromptTab("image")}>图片提示词 <span>{project.prompts.imagePrompts.length}</span></button>
            <button className={promptTab === "h3" ? "active" : ""} onClick={() => setPromptTab("h3")}>H3 视频提示词 <span>{project.prompts.h3Prompts.length}/{segmentCount}</span></button>
          </div>
          {promptGenerationBusy && <div className="prompt-generating" aria-live="polite"><Sparkles size={17} /><span>正在分析素材并编写提示词；已完成 {promptGenerationProgress.completed}/{promptGenerationProgress.total}，结果会逐项写入文本框。</span></div>}
          {promptTab === "image" && <div className="prompt-list">
            <div className="inline-toolbar prompt-stage-toolbar">
              <span>{project.assetPlans.length} 份图片素材需求 · AI 请求并发执行</span>
              <button className="primary-button" disabled={actionBusy || !project.assetPlans.length} onClick={() => void generateAllImagePrompts()}><Sparkles size={15} />生成全部图片提示词</button>
            </div>
            {project.prompts.imagePrompts.map((prompt, index) => <article className="prompt-card" key={prompt.id}>
              <header><div><span>图片 {String(index + 1).padStart(2, "0")}</span><strong>{project.assetPlans.find((plan) => plan.id === prompt.assetPlanId)?.name ?? "待生成素材"}</strong></div><span className="state">已编写</span></header>
              <label>正向提示词<textarea rows={5} value={prompt.prompt} onChange={(event) => update("prompts", (current) => ({ ...current, prompts: { ...current.prompts, imagePrompts: current.prompts.imagePrompts.map((item) => item.id === prompt.id ? { ...item, prompt: event.target.value } : item) } }))} /></label>
              <label>负向提示词<textarea rows={2} value={prompt.negativePrompt} onChange={(event) => update("prompts", (current) => ({ ...current, prompts: { ...current.prompts, imagePrompts: current.prompts.imagePrompts.map((item) => item.id === prompt.id ? { ...item, negativePrompt: event.target.value } : item) } }))} /></label>
              <label>图片工作流<select value={prompt.workflowTemplateId ?? ""} onChange={(event) => update("prompts", (current) => ({ ...current, prompts: { ...current.prompts, imagePrompts: current.prompts.imagePrompts.map((item) => item.id === prompt.id ? { ...item, workflowTemplateId: event.target.value || null, harnessRevision: null } : item) } }))}><option value="">选择已批准工作流</option>{imageWorkflows.map((workflow) => <option key={workflow.id} value={workflow.id}>{workflow.name} · R{workflow.revision}</option>)}</select></label>
              {(() => { const plan = project.assetPlans.find((item) => item.id === prompt.assetPlanId); return plan ? assetResolutionControls(plan) : null; })()}
              <footer><span>{prompt.referenceAssetIds.length} 张参考图</span><button className="primary-button" disabled={actionBusy || !prompt.workflowTemplateId || !workerOnline} onClick={() => void runImagePrompt(prompt.assetPlanId)}><ImagePlus size={14} />重新生成图片</button></footer>
            </article>)}
            {!project.prompts.imagePrompts.length && <div className="prompt-empty"><ImagePlus size={22} /><span>当前没有需要图片工作流生成的素材；上传素材仍可直接用于 H3 参考。</span></div>}
          </div>}
          {promptTab === "h3" && <div className="prompt-list">
            <div className="inline-toolbar prompt-stage-toolbar">
              <span>{project.shots.length} 个电影分镜 · {h3SegmentSlots.length} 份独立视频提示词</span>
              <button
                className="primary-button"
                disabled={actionBusy || !h3SegmentSlots.length}
                onClick={() => promptGenerationBusy ? stopPromptGeneration() : void runPromptGeneration(true)}
              >
                {promptGenerationBusy ? <Square size={14} /> : <Sparkles size={15} />}
                {promptGenerationBusy
                  ? `停止生成 ${promptGenerationProgress.completed}/${promptGenerationProgress.total}`
                  : "生成全部视频提示词"}
              </button>
            </div>
            {project.shots.map((shot, shotIndex) => {
              const slots = h3SegmentSlots.filter((slot) => slot.shotId === shot.id);
              return <article className="h3-shot-row" key={shot.id}>
                <header>
                  <div className="shot-index"><span>{String(shotIndex + 1).padStart(2, "0")}</span><Film size={17} /></div>
                  <div><strong>{shot.title}</strong><span>{shot.summary}</span></div>
                  <div className="h3-shot-meta"><span>{shot.camera}</span><strong>{shot.durationSeconds} 秒 · {slots.length} 个 H3 片段</strong></div>
                </header>
                <div className="h3-segment-list">{slots.map((slot) => {
                  const prompt = promptForSlot(slot);
                  return <section className="h3-segment-editor" key={slot.segmentId}>
                    <div className="h3-segment-heading">
                      <div><strong>视频提示词 {slot.segmentIndex + 1}/{slot.segmentCount}</strong><span>{slot.durationSeconds} 秒 · Seed {slot.seed}{slot.continuationOf ? " · Motion Context 续段" : slot.segmentCount > 1 ? " · 连续链首段" : " · 独立片段"}</span></div>
                      <div className="h3-segment-actions"><span className={`state ${prompt?.prompt.trim() ? "state-ready" : ""}`}>{prompt?.prompt.trim() ? "已填写" : "待编写"}</span><button className="secondary-button" disabled={actionBusy || promptGenerationBusy} onClick={() => void generateOneH3Prompt(slot)}>{prompt?.prompt.trim() ? <RotateCcw size={14} /> : <Sparkles size={14} />}{prompt?.prompt.trim() ? "AI 重新编写" : "AI 编写"}</button></div>
                    </div>
                    <textarea
                      rows={9}
                      disabled={actionBusy || promptGenerationBusy}
                      value={prompt?.prompt ?? ""}
                      onChange={(event) => patchH3Prompt(slot, event.target.value)}
                      placeholder="可在此手动编写 H3 视频提示词，或使用右上角按钮让 AI 按官方规范编写。执行描述建议使用英文；中文对白、歌词和画面文字保持原文。文本框中的最终内容会直接传给 MiniMax。"
                    />
                    <footer>
                      <span>{prompt ? `${prompt.inputMode.toUpperCase()} · 规则版本 R${prompt.harnessRevision ?? "-"} · ${prompt.assetIds.length} 个参考素材` : "尚未生成 · 将根据素材自动选择输入模式"}</span>
                      <div>
                        {prompt?.prompt.trim() && <button
                          className="secondary-button"
                          disabled={actionBusy || translatingPromptId === prompt.id || prompt.id.startsWith("draft-")}
                          onClick={() => void translatePrompt(prompt.id)}
                          title="生成不参与 H3 执行的中文对照"
                        ><Languages size={14} />{translatingPromptId === prompt.id ? "翻译中" : "生成中文对照"}</button>}
                        {prompt?.prompt.trim() && <span>点击“批准并进入下一阶段”即确认此文本</span>}
                      </div>
                    </footer>
                    {prompt && (prompt.assumptions?.length || prompt.stageTrace?.length || prompt.assetRoles?.length) ? <details className="h3-runtime-details">
                      <summary>Harness 运行记录</summary>
                      {prompt.assumptions?.length ? <div><strong>自动假设</strong><ul>{prompt.assumptions.map((item) => <li key={item}>{item}</li>)}</ul></div> : null}
                      {prompt.assetRoles?.length ? <div><strong>素材职责</strong><pre>{JSON.stringify(prompt.assetRoles, null, 2)}</pre></div> : null}
                      {prompt.stageTrace?.length ? <div><strong>阶段轨迹</strong><div className="h3-stage-trace">{prompt.stageTrace.map((item, index) => <span key={`${item.stage}-${index}`}>{item.stage} · {item.status}</span>)}</div></div> : null}
                    </details> : null}
                    {prompt && promptTranslations[prompt.id] ? <section className="h3-translation" aria-label="非执行中文对照">
                      <header><Languages size={14} /><strong>中文对照</strong><span>非执行内容</span></header>
                      <p>{promptTranslations[prompt.id]}</p>
                    </section> : null}
                  </section>;
                })}</div>
              </article>;
            })}
            {!project.shots.length && <div className="prompt-empty"><Film size={22} /><span>还没有电影分镜，请先返回分镜阶段建立镜头。</span></div>}
          </div>}
        </div>}

        {project.activeStage === "generation" && <div className="generation-stage">
          <div className="execution-summary">
            <div><span>电影分镜</span><strong>{project.shots.length}</strong></div><ArrowRight size={18} /><div><span>H3 执行片段</span><strong>{segmentCount}</strong></div><ArrowRight size={18} /><div><span>连续衔接</span><strong>{continuationCount}</strong></div>
          </div>
          <div className="generation-actions">
            <button className="primary-button" disabled={actionBusy || !workerOnline} onClick={() => void startGeneration()}><Play size={16} />{executionStatus?.compiled ? "开始 / 继续派发" : "创建计划并开始生成"}</button>
            <button className="secondary-button" disabled={actionBusy || !workerOnline} onClick={() => void restartGeneration()} title="重新读取全局 MiniMax H3 设置，保留素材和提示词并重建视频任务"><RotateCcw size={15} />重新开始</button>
            <button className="secondary-button" disabled={actionBusy} onClick={() => void compileDag()}><ListPlus size={16} />仅创建任务计划</button>
            <button className="secondary-button" onClick={onOpenTasks}><ArrowRight size={15} />查看任务进度</button>
            <button className="secondary-button" disabled={actionBusy || !executionStatus?.rework_markers.some((item) => item.state === "draft")} onClick={() => void confirmAllReworks()}><Check size={15} />统一确认返工</button>
          </div>
          <Field label="项目审核标准" wide><textarea rows={4} value={project.reviewNotes} onChange={(event) => update("generation", (current) => ({ ...current, reviewNotes: event.target.value }))} placeholder="身份一致性、动作连续性、闪烁、黑帧、声音接缝等" /></Field>
          {!workerOnline && <div className="error-banner inline-banner"><CircleAlert size={16} />ComfyUI Worker 离线。可以创建任务计划，但不能启动 GPU 生成。</div>}
          {mediaLoadError && <div className="error-banner inline-banner"><CircleAlert size={16} />视频读取失败：{mediaLoadError}</div>}
          <div className="generation-tree"><strong>{project.name}</strong>{(executionStatus?.hierarchy ?? []).map((shot) => <section key={shot.shot_id}><h3>{project.shots.find((item) => item.id === shot.shot_id)?.title ?? shot.shot_id}</h3>{shot.segments.map((segment) => <GenerationSegmentCard key={segment.segment_id} segment={segment} project={project} tasks={currentPlanTasks} reviewTasks={reviewTasks} artifactsByTask={artifactsByTask} decisionsByTask={reviewDecisionsByTask} markers={executionStatus?.rework_markers ?? []} actionBusy={actionBusy} onMark={openReworkDialog} onWithdraw={(markerId) => void withdrawMarker(markerId)} />)}</section>)}{!executionStatus?.hierarchy.length && <div className="table-empty">尚未创建任务计划。</div>}</div>
        </div>}

        {project.activeStage === "delivery" && <div className="delivery-stage">
          <div className="delivery-workflow-entry"><div><strong>用户后处理工作流</strong><span>上传 ComfyUI API 工作流，使用 LLM 协助标定输入，登记后在本页选择。</span></div><button className="secondary-button" onClick={onOpenWorkflows}><Upload size={15} />上传 / 管理工作流</button></div>
          {deliveryTasks.flatMap((task) => artifactsByTask[task.task_id] ?? []).filter((artifact) => artifact.kind === "video_export").map((artifact) => <section className="delivery-artifact" key={artifact.artifact_id}>
            <div><strong>最终导出视频</strong><span>可直接播放并核对交付文件</span></div>
            <video controls preload="metadata" src={artifactMediaUrl(artifact)}>当前系统播放器不支持此视频格式。</video>
            <dl><div><dt>文件</dt><dd>{artifact.file_name}</dd></div><div><dt>媒体地址</dt><dd>{artifact.media_url}</dd></div><div><dt>大小</dt><dd>{(artifact.byte_size / 1024 / 1024).toFixed(1)} MB</dd></div><div><dt>SHA-256</dt><dd>{artifact.sha256 ?? "未记录"}</dd></div></dl>
          </section>)}
          <div className="postprocess-item always-on">
            <div className="postprocess-icon"><Film size={19} /></div><div><strong>确定性 FFmpeg 处理</strong><span>统一分辨率、帧率、H.264/AAC 和音量</span></div><span className="approval-chip"><Check size={13} />默认</span>
            <div className="postprocess-controls"><span>输出画幅沿用项目配置：{project.width} × {project.height}</span></div>
          </div>

          <div className={`postprocess-item ${project.postProcessing.seedvr.enabled ? "enabled" : ""}`}>
            <div className="postprocess-icon"><Sparkles size={19} /></div><div><strong>视频修复与超分</strong><span>使用已登记的用户 ComfyUI 视频工作流</span></div><label className="switch"><input type="checkbox" checked={project.postProcessing.seedvr.enabled} onChange={(event) => update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, seedvr: { ...current.postProcessing.seedvr, enabled: event.target.checked, previewApproved: false } } }))} /><span /></label>
            {project.postProcessing.seedvr.enabled && <div className="postprocess-detail">
              <label>用户工作流 <select value={project.postProcessing.seedvr.workflowTemplateId ?? ""} onChange={(event) => { const selected = restorationWorkflows.find((item) => item.id === event.target.value); update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, seedvr: { ...current.postProcessing.seedvr, workflowTemplateId: selected?.id ?? null, workflowRevision: selected?.revision ?? null, profileId: selected?.id ?? null, profileRevision: selected?.revision ?? null, previewApproved: false } } })); }}><option value="">请选择已登记的修复 / 超分工作流</option>{restorationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label>
              <label>放大倍率 <input type="number" min="1" max="8" step="0.25" placeholder="使用模板默认值" value={project.postProcessing.seedvr.upscaleFactor ?? ""} onChange={(event) => update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, seedvr: { ...current.postProcessing.seedvr, upscaleFactor: event.target.value ? Number(event.target.value) : null, previewApproved: false } } }))} /></label>
              {!restorationWorkflows.length && <span className="resource-note"><CircleAlert size={15} />请先上传并登记“视频超分 / 修复”工作流</span>}
            </div>}
          </div>

          <div className={`postprocess-item ${project.postProcessing.rife.enabled ? "enabled" : ""}`}>
            <div className="postprocess-icon"><SlidersHorizontal size={19} /></div><div><strong>视频插帧</strong><span>使用已登记的用户工作流逐片段处理</span></div><label className="switch"><input type="checkbox" checked={project.postProcessing.rife.enabled} onChange={(event) => update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, rife: { ...current.postProcessing.rife, enabled: event.target.checked } } }))} /><span /></label>
            {project.postProcessing.rife.enabled && <div className="postprocess-detail"><label>用户工作流 <select value={project.postProcessing.rife.workflowTemplateId ?? ""} onChange={(event) => { const selected = interpolationWorkflows.find((item) => item.id === event.target.value); update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, rife: { ...current.postProcessing.rife, workflowTemplateId: selected?.id ?? null, workflowRevision: selected?.revision ?? null, profileId: selected?.id ?? null, profileRevision: selected?.revision ?? null } } })); }}><option value="">请选择已登记的插帧工作流</option>{interpolationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label>目标帧率 <select value={project.postProcessing.rife.targetFps} onChange={(event) => update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, rife: { ...current.postProcessing.rife, targetFps: Number(event.target.value) as 48 | 60 | 120 } } }))}>{[48, 60, 120].map((fps) => <option value={fps} key={fps}>{fps} fps</option>)}</select></label>{!interpolationWorkflows.length && <span className="resource-note"><CircleAlert size={15} />请先上传并登记“视频插帧”工作流</span>}</div>}
          </div>

          <div className={`postprocess-item ${project.postProcessing.whisper.enabled ? "enabled" : ""}`}>
            <div className="postprocess-icon"><PackageCheck size={19} /></div><div><strong>语音转写与字幕</strong><span>使用用户上传的 ComfyUI 语音识别工作流生成 SRT</span></div><label className="switch"><input type="checkbox" checked={project.postProcessing.whisper.enabled} onChange={(event) => update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, whisper: { ...current.postProcessing.whisper, enabled: event.target.checked } } }))} /><span /></label>
            {project.postProcessing.whisper.enabled && <div className="postprocess-detail"><label>用户工作流 <select value={project.postProcessing.whisper.workflowTemplateId ?? ""} onChange={(event) => { const selected = transcriptionWorkflows.find((item) => item.id === event.target.value); update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, whisper: { ...current.postProcessing.whisper, workflowTemplateId: selected?.id ?? null, workflowRevision: selected?.revision ?? null, profileId: selected?.id ?? null, profileRevision: selected?.revision ?? null } } })); }}><option value="">请选择已登记的语音识别工作流</option>{transcriptionWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label>识别语言 <select value={project.postProcessing.whisper.language} onChange={(event) => update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, whisper: { ...current.postProcessing.whisper, language: event.target.value as "zh" | "en" | "auto" } } }))}><option value="zh">中文</option><option value="en">英文</option><option value="auto">自动</option></select></label><label className="check-label"><input type="checkbox" checked={project.postProcessing.whisper.burnIn} onChange={(event) => update("delivery", (current) => ({ ...current, postProcessing: { ...current.postProcessing, whisper: { ...current.postProcessing.whisper, burnIn: event.target.checked } } }))} />烧录字幕</label>{!transcriptionWorkflows.length && <span className="resource-note"><CircleAlert size={15} />请先上传并登记“语音识别 / 字幕”工作流</span>}</div>}
          </div>
          <div className="generation-actions"><button className="primary-button" disabled={actionBusy || !executionStatus?.review_complete || (Boolean(guidedRunTarget) && guidedRunTarget !== "delivery")} onClick={() => guidedRunTarget === "delivery" ? stopGuidedRun() : void runGuidedTo("delivery")}>{guidedRunTarget === "delivery" ? <Square size={14} /> : <PackageCheck size={16} />}{guidedRunTarget === "delivery" ? "停止自动派发" : executionStatus?.delivery_complete ? "重新检查交付状态" : "开始后处理并导出"}</button><button className="secondary-button" onClick={onOpenTasks}><ArrowRight size={15} />查看只读任务进度</button></div>
          <div className="task-table pipeline-task-table">
            <div className="task-header"><span>交付执行链</span><span>类型</span><span>状态</span><span>尝试</span><span>操作</span></div>
            {deliveryTasks.map((task) => <div className="task-row" key={task.task_id}>
              <div><strong>{taskKindLabels[task.kind]}</strong><small>{task.state === "blocked" ? `等待 ${task.depends_on.length} 个前置任务` : task.affinity_key ?? "本地执行"}</small></div>
              <span>{taskKindLabels[task.kind]}</span><span className={`state state-${task.state}`}>{taskStateLabels[task.state]}</span><span>{task.attempt}/{task.max_attempts}</span>
              <div className="task-actions">
                {["failed", "stale", "paused"].includes(task.state) && <button className="secondary-button" disabled={actionBusy || Boolean(guidedRunTarget)} onClick={() => void operatePipelineTask(task, "retry")}><RotateCcw size={14} />重试</button>}
                {["queued", "running"].includes(task.state) && <button className="secondary-button" disabled={actionBusy} onClick={() => void operatePipelineTask(task, "cancel")}><Square size={13} />取消</button>}
              </div>
              {task.error_message && <div className="task-detail">{task.error_message}</div>}
            </div>)}
            {!deliveryTasks.length && <div className="table-empty">审核完成后，交付任务会在这里解锁并直接执行。</div>}
          </div>
        </div>}

        {project.activeStage !== "config" && project.activeStage !== "prompts" && <ProjectAgentPanel
          key={project.projectId}
          stage={project.activeStage}
          project={project}
          onAutomaticGenerationChange={handleAutomaticGenerationChange}
          prepareProject={() => saveDraft()}
          applyProject={async (changed) => {
            const invalidated = invalidateFromStage(changed, project.activeStage, pipelineOrder);
            const saved = await saveDraft(invalidated);
            if (!saved) throw new Error("项目修改未能保存，已保留应用前内容");
            try {
              const latest = await getProjectWorkspace(saved.projectId);
              onChange(latest);
            } catch {
              // saveDraft already applied the committed revision locally; a later refresh can reconcile it.
              onChange(saved);
              setActionMessage("修改已保存；读取最新修订失败，界面暂时显示刚保存的内容");
            }
          }}
        />}

        <footer className="stage-footer">
          <div className={agentGenerationBusy ? "stage-generating" : completion[project.activeStage].valid ? "stage-valid" : "stage-invalid"}>{agentGenerationBusy ? <Sparkles size={16} /> : completion[project.activeStage].valid ? <Check size={16} /> : <CircleAlert size={16} />}{agentGenerationBusy ? "正在生成初稿，请稍候" : completion[project.activeStage].valid ? "阶段内容可批准" : completion[project.activeStage].message}</div>
          <button className="primary-button approve-stage" disabled={actionBusy || agentGenerationBusy || !completion[project.activeStage].valid} onClick={() => void approve()}>{project.activeStage === "delivery" ? <PackageCheck size={16} /> : <Check size={16} />}{project.activeStage === "delivery" ? "确认交付完成" : project.stageApprovals[project.activeStage] ? "前往下一阶段" : "批准并进入下一阶段"}</button>
        </footer>
      </div>

      {lightboxAssetId && <AssetLightbox
        assets={project.assets.filter((asset) => asset.status === "ready")}
        activeId={lightboxAssetId}
        projectId={project.projectId}
        onClose={() => setLightboxAssetId(null)}
        onSelect={setLightboxAssetId}
      />}

      <aside className="project-summary">
        <div className="summary-heading"><div><span>活动项目</span><strong>{project.name}</strong></div><span>R{project.revision}</span></div>
        <dl><div><dt>画幅</dt><dd>{project.width} × {project.height}</dd></div><div><dt>帧率</dt><dd>{project.fps} fps</dd></div><div><dt>目标时长</dt><dd>{project.targetDurationSeconds} 秒</dd></div><div><dt>分镜</dt><dd>{project.shots.length}</dd></div><div><dt>H3 片段</dt><dd>{segmentCount}</dd></div><div><dt>公共素材</dt><dd>{project.assets.filter((asset) => asset.scope === "public").length}</dd></div></dl>
        <div className="summary-progress"><span>流程进度</span><strong>{Math.round((pipelineOrder.filter((stage) => project.stageApprovals[stage] && completion[stage].valid).length / pipelineOrder.length) * 100)}%</strong><div><i style={{ width: `${(pipelineOrder.filter((stage) => project.stageApprovals[stage] && completion[stage].valid).length / pipelineOrder.length) * 100}%` }} /></div></div>
        {continuationCount > 0 && <div className="summary-notice"><RotateCcw size={15} /><span>{continuationCount} 个续段将继承 Motion Context</span></div>}
        <div className="summary-models"><span>执行策略</span><div><i />先统一准备全部素材信息</div><div><i />单 GPU 逐段生成视频</div><div><i />已关闭额外缓存加速</div></div>
      </aside>
    </div>
  </section>;
}
