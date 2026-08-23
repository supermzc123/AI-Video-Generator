import { invoke } from "@tauri-apps/api/core";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";
import type {
  ApiWorkflow,
  AgentProposal,
  BackendInspection,
  BatchRun,
  HarnessBundle,
  HarnessRevision,
  Health,
  AssetDraft,
  ProjectDraft,
  ProjectExecutionStatus,
  ProjectMemoryEvent,
  ProjectRunState,
  TaskSpec,
  WorkflowTemplateSummary,
  ArtifactDescriptor,
  ReviewDecision,
  ReworkRequest,
  ReworkMarker,
  GenerationBatch,
  AssetGenerationCandidate,
  SetupStatus,
  PostProcessingCapabilities,
} from "./types";

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
let apiOrigin = "";

async function resolveApiOrigin(): Promise<string> {
  if (!isTauri || apiOrigin) return apiOrigin;
  const status = await invoke<DesktopControlPlaneStatus>("control_plane_status");
  if (!status.base_url) throw new ApiError(503, "桌面控制平面尚未提供本地地址", status);
  apiOrigin = status.base_url.replace(/\/$/, "");
  return apiOrigin;
}

export class ApiError extends Error {
  constructor(public readonly status: number, message: string, public readonly detail: unknown) {
    super(message);
    this.name = "ApiError";
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const origin = await resolveApiOrigin();
  const response = await (isTauri ? tauriFetch : fetch)(`${origin}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail);
    throw new ApiError(response.status, detail || `${response.status} ${response.statusText}`, body?.detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function requestForm<T>(path: string, body: FormData): Promise<T> {
  const origin = await resolveApiOrigin();
  const response = await (isTauri ? tauriFetch : fetch)(`${origin}${path}`, { method: "POST", body });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = typeof payload?.detail === "string" ? payload.detail : JSON.stringify(payload?.detail) || `${response.status} ${response.statusText}`;
    throw new ApiError(response.status, detail, payload?.detail);
  }
  return (await response.json()) as T;
}

async function requestEventStream<T>(
  path: string,
  options: {
    body?: unknown;
    signal?: AbortSignal;
    onDelta: (delta: string) => void;
    parseResult: (value: unknown) => T;
    errorMessage: string;
  },
): Promise<T> {
  const origin = await resolveApiOrigin();
  const response = await fetch(`${origin}${path}`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      ...(options.body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  });
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => null) as { detail?: unknown } | null;
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail === undefined ? "" : JSON.stringify(detail);
    throw new ApiError(response.status, message || `${options.errorMessage}：${response.status} ${response.statusText}`, detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  let result: T;
  const handleEvent = (block: string) => {
    const lines = block.replace(/\r\n/g, "\n").split("\n");
    const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
    const data = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!event || !data || data === "[DONE]") return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(data) as unknown;
    } catch {
      throw new ApiError(502, `${options.errorMessage}返回了无效 JSON`, data);
    }
    if (event === "delta" && typeof parsed === "string") options.onDelta(parsed);
    if (event === "result") {
      result = options.parseResult(parsed);
      completed = true;
    }
    if (event === "error") {
      const error = isRecord(parsed) ? parsed : {};
      const status = typeof error.status === "number" ? error.status : 500;
      const detail = error.detail;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail);
      throw new ApiError(status, message || options.errorMessage, detail);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    events.forEach(handleEvent);
    if (done) {
      if (buffer.trim()) handleEvent(buffer);
      break;
    }
  }
  if (!completed) throw new ApiError(502, `${options.errorMessage}结束但没有返回结果`, null);
  return result!;
}

type BackendProjectAsset = {
  asset_id: string;
  name: string;
  original_file_name?: string;
  original_name?: string;
  sha256?: string | null;
  blob_sha256?: string | null;
  mime_type: string | null;
  media_kind?: AssetDraft["mediaKind"];
  width: number | null;
  height: number | null;
  duration_seconds?: number | null;
  frame_rate?: number | null;
  has_audio?: boolean | null;
  byte_size: number | null;
  kind?: AssetDraft["kind"];
  purpose?: "reference" | "character" | "scene" | "prop" | "style" | "keyframe";
  scope: AssetDraft["scope"] | "common";
  shot_id: string | null;
  shot_ids?: string[];
  source: AssetDraft["source"];
  status?: AssetDraft["status"];
  state?: "available" | "missing_blob" | "retired";
  preview_url?: string | null;
};

function mapProjectAsset(asset: BackendProjectAsset): AssetDraft {
  return {
    id: asset.asset_id,
    name: asset.name,
    originalFileName: asset.original_file_name ?? asset.original_name ?? asset.name,
    sha256: asset.sha256 ?? asset.blob_sha256 ?? null,
    mimeType: asset.mime_type,
    mediaKind: asset.media_kind ?? (asset.mime_type?.startsWith("video/") ? "video" : asset.mime_type?.startsWith("audio/") ? "audio" : "image"),
    width: asset.width,
    height: asset.height,
    durationSeconds: asset.duration_seconds ?? null,
    frameRate: asset.frame_rate ?? null,
    hasAudio: asset.has_audio ?? null,
    byteSize: asset.byte_size,
    kind: asset.kind ?? (asset.purpose === "reference" || asset.purpose === "keyframe" ? "character" : asset.purpose ?? "character"),
    scope: asset.scope === "common" ? "public" : asset.scope,
    shotId: asset.shot_id,
    shotIds: asset.shot_ids ?? (asset.shot_id ? [asset.shot_id] : []),
    source: asset.source,
    status: asset.status ?? (asset.state === "available" ? "ready" : asset.state === "retired" ? "failed" : "missing_blob"),
    previewUrl: asset.preview_url ?? null,
  };
}

export function projectAssetPreviewUrl(projectId: string, assetId: string): string {
  return `${apiOrigin}/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/preview`;
}

export function projectAssetMediaUrl(projectId: string, assetId: string): string {
  return `${apiOrigin}/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/media`;
}

export function artifactMediaUrl(artifact: ArtifactDescriptor): string {
  return `${apiOrigin}${artifact.media_url}`;
}

export async function uploadProjectAsset(
  projectId: string,
  file: File,
  name: string,
  kind: AssetDraft["kind"] = "character",
  scope: AssetDraft["scope"] = "public",
  shotId: string | null = null,
): Promise<AssetDraft> {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("name", name);
  form.append("kind", kind);
  form.append("scope", scope === "public" ? "common" : scope);
  if (shotId) form.append("shot_id", shotId);
  return mapProjectAsset(await requestForm<BackendProjectAsset>(`/api/v1/projects/${encodeURIComponent(projectId)}/assets`, form));
}

export async function listProjectAssets(projectId: string): Promise<AssetDraft[]> {
  const result = await requestJson<BackendProjectAsset[]>(`/api/v1/projects/${encodeURIComponent(projectId)}/assets`);
  return result.map(mapProjectAsset);
}

export async function relinkProjectAsset(
  projectId: string,
  assetId: string,
  file: File,
): Promise<AssetDraft> {
  const form = new FormData();
  form.append("file", file);
  return mapProjectAsset(await requestForm<BackendProjectAsset>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/relink`,
    form,
  ));
}

export async function updateProjectAsset(
  projectId: string,
  assetId: string,
  changes: Pick<AssetDraft, "name" | "kind" | "scope" | "shotId" | "shotIds">,
): Promise<AssetDraft> {
  const result = await requestJson<BackendProjectAsset>(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}`, {
    method: "PATCH",
    body: JSON.stringify({ name: changes.name, kind: changes.kind, scope: changes.scope === "public" ? "common" : changes.scope, shot_id: changes.shotId, shot_ids: changes.shotIds }),
  });
  return mapProjectAsset(result);
}

export function deleteProjectAsset(projectId: string, assetId: string): Promise<void> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}`, { method: "DELETE" });
}

export async function generateProjectPromptStage(projectId: string): Promise<ProjectDraft> {
  const response = await requestJson<{ payload: ProjectDraft }>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/stages/prompts/generate`,
    { method: "POST", body: JSON.stringify({}) },
  );
  return response.payload;
}

export async function regenerateH3Prompt(
  projectId: string,
  segmentId: string,
  signal?: AbortSignal,
): Promise<ProjectDraft> {
  const response = await requestJson<{ payload: ProjectDraft }>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/prompts/h3/${encodeURIComponent(segmentId)}/regenerate`,
    { method: "POST", body: JSON.stringify({}), signal },
  );
  return response.payload;
}

export async function streamRegenerateH3Prompt(
  projectId: string,
  segmentId: string,
  onDelta: (delta: string) => void,
  signal?: AbortSignal,
): Promise<ProjectDraft> {
  return requestEventStream(
    `/api/v1/projects/${encodeURIComponent(projectId)}/prompts/h3/${encodeURIComponent(segmentId)}/regenerate/stream`,
    {
      signal,
      onDelta,
      errorMessage: "H3 流式生成失败",
      parseResult: (value) => {
        if (!isRecord(value) || !isRecord(value.payload)) throw new ApiError(502, "H3 最终结果缺少项目数据", value);
        return value.payload as ProjectDraft;
      },
    },
  );
}

export function translateH3Prompt(
  projectId: string,
  promptRevisionId: string,
): Promise<import("./types").H3PromptTranslation> {
  return requestJson(
    `/api/v1/projects/${encodeURIComponent(projectId)}/prompts/h3/${encodeURIComponent(promptRevisionId)}/translate`,
    { method: "POST" },
  );
}

export async function generateImagePrompt(
  projectId: string,
  assetPlanId: string,
  instruction: string | null = null,
  workflowTemplateId: string | null = null,
): Promise<ProjectDraft> {
  const response = await requestJson<{ payload: ProjectDraft }>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/prompts/images/${encodeURIComponent(assetPlanId)}/generate`,
    { method: "POST", body: JSON.stringify({ instruction, workflow_template_id: workflowTemplateId }) },
  );
  return response.payload;
}

export async function streamGenerateImagePrompt(
  projectId: string,
  assetPlanId: string,
  instruction: string | null,
  workflowTemplateId: string | null,
  onDelta: (delta: string) => void,
  signal?: AbortSignal,
): Promise<ProjectDraft> {
  return requestEventStream(
    `/api/v1/projects/${encodeURIComponent(projectId)}/prompts/images/${encodeURIComponent(assetPlanId)}/generate/stream`,
    {
      body: { instruction, workflow_template_id: workflowTemplateId },
      signal,
      onDelta,
      errorMessage: "图片提示词流式生成失败",
      parseResult: (value) => {
        if (!isRecord(value) || !isRecord(value.payload)) throw new ApiError(502, "图片提示词最终结果缺少项目数据", value);
        return value.payload as ProjectDraft;
      },
    },
  );
}

export function runProjectImagePrompt(projectId: string, assetPlanId: string): Promise<TaskSpec> {
  return requestJson(
    `/api/v1/projects/${encodeURIComponent(projectId)}/image-prompts/${encodeURIComponent(assetPlanId)}/run`,
    { method: "POST", body: JSON.stringify({}) },
  );
}

export function regenerateProjectAsset(projectId: string, assetPlanId: string): Promise<TaskSpec> {
  return requestJson(
    `/api/v1/projects/${encodeURIComponent(projectId)}/asset-plans/${encodeURIComponent(assetPlanId)}/regenerate`,
    { method: "POST", body: JSON.stringify({}) },
  );
}

export function listAssetCandidates(projectId: string, assetPlanId: string): Promise<AssetGenerationCandidate[]> {
  return requestJson(
    `/api/v1/projects/${encodeURIComponent(projectId)}/asset-plans/${encodeURIComponent(assetPlanId)}/candidates`,
  );
}

export function assetCandidatePreviewUrl(candidateId: string): string {
  return `${apiOrigin}/api/v1/asset-candidates/${encodeURIComponent(candidateId)}/preview`;
}

export function acceptAssetCandidate(candidateId: string): Promise<AssetGenerationCandidate> {
  return requestJson(`/api/v1/asset-candidates/${encodeURIComponent(candidateId)}/accept`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function discardAssetCandidate(candidateId: string): Promise<AssetGenerationCandidate> {
  return requestJson(`/api/v1/asset-candidates/${encodeURIComponent(candidateId)}/discard`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function getHealth(): Promise<Health> {
  return requestJson("/api/v1/health");
}

export function getSetupStatus(): Promise<SetupStatus> {
  return requestJson("/api/v1/setup/status");
}

export function installDefaultH3HarnessSource(sourceId: "community" | "official"): Promise<unknown> {
  return requestJson(`/api/v1/harnesses/h3-default/sources/${sourceId}/install`, {
    method: "POST",
  });
}

export function assembleDefaultH3Harness(): Promise<HarnessRevision> {
  return requestJson("/api/v1/harnesses/h3-default/assemble?approve=true", {
    method: "POST",
  });
}

export function getPostProcessingCapabilities(): Promise<PostProcessingCapabilities> {
  return requestJson("/api/v1/postprocessing/capabilities");
}

export function getLocalWorkerCapabilities(): Promise<{
  server_online: boolean;
  full_pipeline_ready: boolean;
  execution_blockers: string[];
  h3_node_ids: string[];
  inventory: { version: string | null };
}> {
  return requestJson("/api/v1/workers/local/capabilities");
}

export type RuntimeSettings = {
  comfyuiRoot: string;
  comfyuiBaseUrl: string;
  requestTimeoutSeconds: number;
  llmBaseUrl: string;
  llmModel: string;
  llmApiKeyConfigured: boolean;
  llmTimeoutSeconds: number;
  llmFirstTokenTimeoutSeconds: number;
  llmVideoCapable: boolean;
  networkProxy: string;
  h3DiffusionModel: string;
  h3TextEncoder: string;
  h3VideoVae: string;
  h3AudioVae: string;
  h3TurboLora: string;
  h3TurboEnabled: boolean;
  h3SageAttentionEnabled: boolean;
  h3LowVram: boolean;
  h3Steps: number;
};

export type ComfyNodeInstallResult = {
  succeeded: boolean;
  restart_required: boolean;
  comfyui_root: string;
  steps: Array<{
    component: string;
    label: string;
    succeeded: boolean;
    message: string;
  }>;
};

type BackendRuntimeSettings = {
  comfyui_root: string | null;
  comfyui_base_url: string;
  request_timeout_seconds: number;
  llm_base_url: string | null;
  llm_model: string | null;
  llm_api_key_configured: boolean;
  llm_timeout_seconds: number;
  llm_first_token_timeout_seconds: number;
  llm_video_capable: boolean;
  network_proxy: string | null;
  h3_diffusion_model: string;
  h3_text_encoder: string;
  h3_video_vae: string;
  h3_audio_vae: string;
  h3_turbo_lora: string;
  h3_turbo_enabled: boolean;
  h3_sage_attention_enabled: boolean;
  h3_low_vram: boolean;
  h3_steps: number;
};

function mapRuntimeSettings(value: BackendRuntimeSettings): RuntimeSettings {
  return {
    comfyuiRoot: value.comfyui_root ?? "",
    comfyuiBaseUrl: value.comfyui_base_url,
    requestTimeoutSeconds: value.request_timeout_seconds,
    llmBaseUrl: value.llm_base_url ?? "",
    llmModel: value.llm_model ?? "",
    llmApiKeyConfigured: value.llm_api_key_configured,
    llmTimeoutSeconds: value.llm_timeout_seconds,
    llmFirstTokenTimeoutSeconds: value.llm_first_token_timeout_seconds ?? value.llm_timeout_seconds,
    llmVideoCapable: value.llm_video_capable ?? false,
    networkProxy: value.network_proxy ?? "",
    h3DiffusionModel: value.h3_diffusion_model,
    h3TextEncoder: value.h3_text_encoder,
    h3VideoVae: value.h3_video_vae,
    h3AudioVae: value.h3_audio_vae,
    h3TurboLora: value.h3_turbo_lora,
    h3TurboEnabled: value.h3_turbo_enabled,
    h3SageAttentionEnabled: value.h3_sage_attention_enabled,
    h3LowVram: value.h3_low_vram,
    h3Steps: value.h3_steps,
  };
}

export async function getRuntimeSettings(): Promise<RuntimeSettings> {
  return mapRuntimeSettings(await requestJson<BackendRuntimeSettings>("/api/v1/settings"));
}

export function listLlmModels(): Promise<{ models: string[]; count: number }> {
  return requestJson("/api/v1/settings/llm-models");
}

export function listComfyuiModels(): Promise<{
  diffusion_models: string[]; text_encoders: string[]; vaes: string[]; loras: string[];
}> {
  return requestJson("/api/v1/settings/comfyui-models");
}

export async function updateRuntimeSettings(
  settings: RuntimeSettings,
  apiKey: string,
  clearApiKey: boolean,
): Promise<RuntimeSettings> {
  const response = await requestJson<BackendRuntimeSettings>("/api/v1/settings", {
    method: "PUT",
    body: JSON.stringify({
      comfyui_root: settings.comfyuiRoot.trim() || null,
      comfyui_base_url: settings.comfyuiBaseUrl,
      request_timeout_seconds: settings.requestTimeoutSeconds,
      llm_base_url: settings.llmBaseUrl.trim() || null,
      llm_model: settings.llmModel.trim() || null,
      ...(apiKey ? { llm_api_key: apiKey } : {}),
      clear_llm_api_key: clearApiKey,
      llm_timeout_seconds: settings.llmTimeoutSeconds,
      llm_first_token_timeout_seconds: settings.llmFirstTokenTimeoutSeconds,
      llm_video_capable: settings.llmVideoCapable,
      network_proxy: settings.networkProxy.trim() || null,
      h3_diffusion_model: settings.h3DiffusionModel,
      h3_text_encoder: settings.h3TextEncoder,
      h3_video_vae: settings.h3VideoVae,
      h3_audio_vae: settings.h3AudioVae,
      h3_turbo_lora: settings.h3TurboLora,
      h3_turbo_enabled: settings.h3TurboEnabled,
      h3_sage_attention_enabled: settings.h3SageAttentionEnabled,
      h3_low_vram: settings.h3LowVram,
      h3_steps: settings.h3Steps,
    }),
  });
  return mapRuntimeSettings(response);
}

export function installRequiredComfyNodes(): Promise<ComfyNodeInstallResult> {
  return requestJson("/api/v1/settings/comfyui-nodes/install", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export type DesktopControlPlaneStatus = {
  phase: "starting" | "online" | "offline";
  managed: boolean;
  message: string;
  base_url: string;
};

export function getDesktopControlPlaneStatus(): Promise<DesktopControlPlaneStatus | null> {
  return isTauri ? invoke("control_plane_status") : Promise.resolve(null);
}

export function listTasks(projectId?: string): Promise<TaskSpec[]> {
  const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
  return requestJson(`/api/v1/tasks${query}`);
}

export function getTask(taskId: string): Promise<TaskSpec> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}`);
}

export function createTask(task: TaskSpec): Promise<TaskSpec> {
  return requestJson("/api/v1/tasks", { method: "POST", body: JSON.stringify(task) });
}

export function compileProjectTasks(projectId: string, restartH3 = false): Promise<{
  fingerprint: string;
  workspace_revision: number;
  h3_execution_profile: {
    generation_revision: number;
    diffusion_model: string;
    turbo_enabled: boolean;
    steps: number;
  } | null;
  tasks: TaskSpec[];
}> {
  const query = restartH3 ? "?restart_h3=true" : "";
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/tasks/compile${query}`, {
    method: "POST",
  });
}

export function compileProjectDeliveryTasks(projectId: string): Promise<{
  fingerprint: string;
  workspace_revision: number;
  tasks: TaskSpec[];
}> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/delivery/tasks/compile`, { method: "POST" });
}

export function clearProjectTasks(projectId: string): Promise<{ removed_tasks: number; media_retained: boolean }> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/tasks/clear`, { method: "POST" });
}

export function deleteProject(projectId: string): Promise<{ deleted: boolean; removed_tasks: number; media_files_retained: boolean }> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" });
}

export function getProjectExecutionStatus(projectId: string): Promise<ProjectExecutionStatus> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/execution-status`);
}

export function startProjectGeneration(projectId: string): Promise<GenerationBatch> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/generation/start`, {
    method: "POST",
  });
}

export function createReworkMarker(
  projectId: string,
  request: {
    version_id: string;
    action: ReworkMarker["action"];
    feedback: string;
    replacement_seed: number | null;
    source?: "human" | "ai";
  },
): Promise<ReworkMarker> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/rework-markers`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function withdrawReworkMarker(projectId: string, markerId: string): Promise<ReworkMarker> {
  return requestJson(
    `/api/v1/projects/${encodeURIComponent(projectId)}/rework-markers/${encodeURIComponent(markerId)}`,
    { method: "DELETE" },
  );
}

export function confirmProjectReworks(projectId: string): Promise<GenerationBatch> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/rework-markers/confirm`, {
    method: "POST",
  });
}

export function cancelTask(taskId: string): Promise<TaskSpec> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/cancel`, { method: "POST" });
}

export function runTask(taskId: string): Promise<TaskSpec> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/run`, { method: "POST" });
}

export function pauseTask(taskId: string): Promise<TaskSpec> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/pause`, { method: "POST" });
}

export function resumeTask(taskId: string): Promise<TaskSpec> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/resume`, { method: "POST" });
}

export function redoTask(taskId: string): Promise<TaskSpec> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/redo`, { method: "POST" });
}

export function reviewTask(taskId: string, accepted: boolean, feedback = ""): Promise<TaskSpec> {
  const action = accepted ? "accept" : "reject";
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/review/${action}`, {
    method: "POST",
    body: JSON.stringify({ feedback: feedback || null }),
  });
}

export function listTaskArtifacts(taskId: string): Promise<ArtifactDescriptor[]> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/artifacts`);
}

export function listTaskReviewDecisions(taskId: string): Promise<ReviewDecision[]> {
  return requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/review-decisions`);
}

export function createSegmentRework(
  projectId: string,
  segmentId: string,
  request: {
    source_h3_task_id: string;
    review_task_id: string | null;
    action: ReworkRequest["action"];
    feedback: string;
    replacement_seed: number | null;
  },
): Promise<ReworkRequest> {
  return requestJson(
    `/api/v1/projects/${encodeURIComponent(projectId)}/segments/${encodeURIComponent(segmentId)}/rework`,
    { method: "POST", body: JSON.stringify(request) },
  );
}

export type BackendProjectSpec = {
  project_id: string;
  revision: number;
  name: string;
  width: number;
  height: number;
  fps: number;
  target_duration_seconds: number;
  audio_policy: ProjectDraft["audioPolicy"];
  external_audio_asset_id: string | null;
  updated_at?: string;
};

export function listProjects(): Promise<BackendProjectSpec[]> {
  return requestJson("/api/v1/project-summaries");
}

export async function getProjectWorkspace(projectId: string): Promise<ProjectDraft> {
  const response = await requestJson<{ payload: ProjectDraft }>(
    `/api/v1/projects/${encodeURIComponent(projectId)}/workspace`,
  );
  return response.payload;
}

export async function saveProjectToControlPlane(project: ProjectDraft): Promise<ProjectDraft> {
  let latest: BackendProjectSpec | null = null;
  try {
    latest = await requestJson<BackendProjectSpec>(
      `/api/v1/projects/${encodeURIComponent(project.projectId)}`,
    );
  } catch (cause) {
    if (cause instanceof ApiError && cause.status === 404) latest = null;
    else throw cause;
  }
  const revision = latest ? latest.revision + 1 : 1;
  const spec: BackendProjectSpec = {
    project_id: project.projectId,
    revision,
    name: project.name,
    width: project.width,
    height: project.height,
    fps: project.fps,
    target_duration_seconds: project.targetDurationSeconds,
    audio_policy: project.audioPolicy,
    external_audio_asset_id: project.externalAudioAssetId,
  };
  const saved = { ...project, revision, updatedAt: new Date().toISOString() };
  await requestJson(`/api/v1/projects/${encodeURIComponent(project.projectId)}/commit`, {
    method: "POST",
    body: JSON.stringify({ project: spec, payload: saved }),
  });
  return saved;
}

export function getProjectRunState(projectId: string): Promise<ProjectRunState> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/run-state`);
}

export function setProjectMode(
  projectId: string,
  mode: "guided" | "batch",
): Promise<ProjectRunState> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/mode`, {
    method: "POST",
    body: JSON.stringify({ mode }),
  });
}

export function setProjectPaused(projectId: string, paused: boolean): Promise<ProjectRunState> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/pause`, {
    method: "POST",
    body: JSON.stringify({ paused }),
  });
}

export function setReviewMode(
  projectId: string,
  mode: "human_ai" | "ai_only" | "manual" | "none",
  humanTimeoutSeconds: number,
): Promise<ProjectRunState> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/review-policy`, {
    method: "POST",
    body: JSON.stringify({ mode, human_timeout_seconds: humanTimeoutSeconds }),
  });
}

export function requestProjectAgentProposal(
  projectId: string,
  request: {
    operationId: string;
    operation: string;
    instruction: string;
    displayInstruction?: string;
    allowedPaths: string[];
    lockedPaths: string[];
  },
): Promise<{
  proposal: AgentProposal;
  input_sha256: string;
  output_sha256: string;
  committed_revision: number | null;
}> {
  return requestJson(`/api/v1/projects/${encodeURIComponent(projectId)}/agent/operate`, {
    method: "POST",
    body: JSON.stringify({
      operation_id: request.operationId,
      operation: request.operation,
      instruction: request.instruction,
      display_instruction: request.displayInstruction,
      allowed_paths: request.allowedPaths,
      locked_paths: request.lockedPaths,
      commit: false,
    }),
  });
}

export async function streamProjectAgentProposal(
  projectId: string,
  request: {
    operationId: string;
    operation: string;
    instruction: string;
    displayInstruction?: string;
    allowedPaths: string[];
    lockedPaths: string[];
  },
  onDelta: (delta: string) => void,
): Promise<{
  proposal: AgentProposal;
  input_sha256: string;
  output_sha256: string;
  committed_revision: number | null;
}> {
  return requestEventStream(
    `/api/v1/projects/${encodeURIComponent(projectId)}/agent/operate/stream`,
    {
      body: {
        operation_id: request.operationId,
        operation: request.operation,
        instruction: request.instruction,
        display_instruction: request.displayInstruction,
        allowed_paths: request.allowedPaths,
        locked_paths: request.lockedPaths,
        commit: false,
      },
      onDelta,
      errorMessage: "LLM 流式请求失败",
      parseResult: normalizeAgentStreamResult,
    },
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeAgentStreamResult(
  value: unknown,
): Awaited<ReturnType<typeof requestProjectAgentProposal>> {
  if (!isRecord(value) || !isRecord(value.proposal)) {
    throw new ApiError(502, "LLM 最终结果缺少 proposal 对象", value);
  }
  const proposal = value.proposal;
  const patches = Array.isArray(proposal.patches) ? proposal.patches : null;
  const warnings = proposal.warnings === undefined ? [] : proposal.warnings;
  if (
    !patches
    || !Array.isArray(warnings)
    || typeof proposal.operation_id !== "string"
    || typeof proposal.rationale !== "string"
  ) {
    throw new ApiError(502, "LLM 最终结果结构无效，项目未被修改", value);
  }
  return {
    proposal: {
      operation_id: proposal.operation_id,
      patches: patches as AgentProposal["patches"],
      rationale: proposal.rationale,
      warnings: warnings.filter((item): item is string => typeof item === "string"),
    },
    input_sha256: typeof value.input_sha256 === "string" ? value.input_sha256 : "",
    output_sha256: typeof value.output_sha256 === "string" ? value.output_sha256 : "",
    committed_revision: typeof value.committed_revision === "number" ? value.committed_revision : null,
  };
}

export function listProjectMemory(
  projectId: string,
  limit = 200,
): Promise<ProjectMemoryEvent[]> {
  return requestJson(
    `/api/v1/projects/${encodeURIComponent(projectId)}/memory?limit=${limit}`,
  );
}

export function listBatches(): Promise<BatchRun[]> {
  return requestJson("/api/v1/batches");
}

export function createBatch(batch: BatchRun): Promise<BatchRun> {
  return requestJson("/api/v1/batches", { method: "POST", body: JSON.stringify(batch) });
}

export function transitionBatch(batchId: string, action: string): Promise<BatchRun> {
  return requestJson(
    `/api/v1/batches/${encodeURIComponent(batchId)}/${encodeURIComponent(action)}`,
    { method: "POST" },
  );
}

export function cancelBatchProject(batchId: string, projectId: string): Promise<BatchRun> {
  return requestJson(
    `/api/v1/batches/${encodeURIComponent(batchId)}/projects/${encodeURIComponent(projectId)}/cancel`,
    { method: "POST" },
  );
}

export function listHarnesses(): Promise<HarnessBundle[]> {
  return requestJson("/api/v1/harnesses");
}

export function registerHarness(bundle: HarnessBundle): Promise<HarnessBundle> {
  return requestJson("/api/v1/harnesses", { method: "POST", body: JSON.stringify(bundle) });
}

export function registerHarnessRevision(revision: HarnessRevision): Promise<HarnessRevision> {
  return requestJson(
    `/api/v1/harnesses/${encodeURIComponent(revision.harness_id)}/revisions`,
    { method: "POST", body: JSON.stringify(revision) },
  );
}

export function listHarnessRevisions(harnessId: string): Promise<HarnessRevision[]> {
  return requestJson(
    `/api/v1/harnesses/${encodeURIComponent(harnessId)}/revisions`,
  );
}

export function inspectWorkflow(rawWorkflow: ApiWorkflow): Promise<BackendInspection> {
  return requestJson("/api/v1/workflows/inspect", {
    method: "POST",
    body: JSON.stringify({ raw_workflow: rawWorkflow }),
  });
}

export function suggestWorkflowBindings(
  workflowId: string,
  rawWorkflow: ApiWorkflow,
  project: ProjectDraft,
  requestedSemantics?: string[],
  instructions?: string,
): Promise<{ bindings: BackendInspection["bindings"] }> {
  return requestJson("/api/v1/llm/workflows/map", {
    method: "POST",
    body: JSON.stringify({
      operation_id: `map:${workflowId}`,
      workflow_id: workflowId,
      project: {
        project_id: project.projectId,
        revision: project.revision,
        name: project.name,
        width: project.width,
        height: project.height,
        fps: project.fps,
        target_duration_seconds: project.targetDurationSeconds,
        audio_policy: project.audioPolicy,
      },
      raw_workflow: rawWorkflow,
      requested_semantics: requestedSemantics,
      instructions,
    }),
  });
}

export function registerWorkflow(template: Record<string, unknown>): Promise<unknown> {
  return requestJson("/api/v1/workflows/templates", {
    method: "POST",
    body: JSON.stringify(template),
  });
}

export function listWorkflowTemplates(): Promise<WorkflowTemplateSummary[]> {
  return requestJson("/api/v1/workflows/templates");
}

export function deleteWorkflowTemplate(templateId: string): Promise<{
  deleted: boolean;
  removed_revisions: number;
}> {
  return requestJson(`/api/v1/workflows/templates/${encodeURIComponent(templateId)}`, {
    method: "DELETE",
  });
}
