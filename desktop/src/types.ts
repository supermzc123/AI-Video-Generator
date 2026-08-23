export type WorkflowNode = {
  class_type: string;
  inputs: Record<string, unknown>;
  _meta?: { title?: string };
};

export type ApiWorkflow = Record<string, WorkflowNode>;

export type PipelineStageId =
  | "config"
  | "idea"
  | "outline"
  | "storyboard"
  | "assets"
  | "prompts"
  | "generation"
  | "delivery";

export type OutlineBeat = {
  id: string;
  title: string;
  summary: string;
  durationSeconds: number;
};

export type ShotDraft = {
  id: string;
  title: string;
  summary: string;
  camera: string;
  seed: number;
  durationSeconds: number;
  motionSegments: Array<{
    id: string;
    durationSeconds: number;
    summary: string;
  }>;
  locked: boolean;
};

export type AssetKind = "character" | "scene" | "prop" | "style";
export type AssetScope = "public" | "shot";
export type AssetStatus = "ready" | "missing_blob" | "uploading" | "failed";
export type AssetMediaKind = "image" | "video" | "audio";

export type AssetDraft = {
  id: string;
  name: string;
  originalFileName: string;
  sha256: string | null;
  mimeType: string | null;
  mediaKind: AssetMediaKind;
  width: number | null;
  height: number | null;
  durationSeconds?: number | null;
  frameRate?: number | null;
  hasAudio?: boolean | null;
  byteSize: number | null;
  kind: AssetKind;
  scope: AssetScope;
  shotId: string | null;
  source: "upload" | "generated";
  status: AssetStatus;
  previewUrl: string | null;
};

export type RichTextNode =
  | { type: "text"; text: string }
  | { type: "asset_mention"; assetId: string; displayName: string };

export type RichTextDocument = { nodes: RichTextNode[] };

export type AssetPlan = {
  id: string;
  name: string;
  description: string;
  kind: AssetKind;
  scope: AssetScope;
  shotId: string | null;
  shotIds: string[];
  fulfilledByAssetId: string | null;
  state: "draft" | "ready" | "satisfied" | "stale";
  width: number;
  height: number;
  resolutionSource: "default" | "ai" | "manual";
};

export type AssetGenerationCandidate = {
  schema_version: "1.0";
  candidate_id: string;
  project_id: string;
  asset_plan_id: string;
  source_task_id: string;
  current_asset_id: string | null;
  name: string;
  kind: "reference" | "character" | "scene" | "prop" | "style" | "keyframe";
  scope: "common" | "shot";
  shot_id: string | null;
  sha256: string;
  preview_sha256: string;
  mime_type: string;
  byte_size: number;
  width: number;
  height: number;
  state: "pending" | "accepted" | "discarded";
  accepted_asset_id: string | null;
  created_at: string;
};

export type PromptReviewResult = {
  ready: boolean;
  issues: Array<{ severity: "error" | "warning"; code: string; message: string }>;
  reviewedAt: string | null;
};

export type ImagePromptRevision = {
  id: string;
  assetPlanId: string;
  prompt: string;
  negativePrompt: string;
  workflowTemplateId: string | null;
  harnessRevision: number | null;
  referenceAssetIds: string[];
  locked: boolean;
  revision: number;
};

export type H3InputMode = "t2va" | "i2va" | "fl2va" | "l2va" | "ref2va";

export type H3PromptRevision = {
  id: string;
  shotId: string;
  segmentId: string;
  segmentIndex: number;
  durationSeconds: number;
  continuationOf: string | null;
  inputMode: H3InputMode;
  prompt: string;
  endState: string | null;
  assetIds: string[];
  seed: number;
  harnessRevision: number | null;
  harnessManifestSha256?: string | null;
  route?: string | null;
  assetRoles?: Array<Record<string, unknown>>;
  assumptions?: string[];
  stageTrace?: Array<{ stage: string; status: string; attempt?: number; findings?: string[] }>;
  locked: boolean;
  revision: number;
  legacy: boolean;
  review: PromptReviewResult;
};

export type H3PromptTranslation = {
  prompt_revision_id: string;
  source_sha256: string;
  language: string;
  translation: string;
  executable: false;
};

export type PromptSet = {
  imagePrompts: ImagePromptRevision[];
  h3Prompts: H3PromptRevision[];
  generatedAt: string | null;
  generationSummary?: {
    total: number;
    succeeded: number;
    failed: number;
    failedSegmentIds: string[];
  };
};

export type PostProcessingSettings = {
  normalizeVideo: boolean;
  outputWidth: number;
  outputHeight: number;
  crf: number;
  normalizeAudio: boolean;
  seedvr: {
    enabled: boolean;
    workflowTemplateId?: string | null;
    workflowRevision?: number | null;
    upscaleFactor?: number | null;
    profileId: string | null;
    profileRevision: number | null;
    modelId: string | null;
    modelSha256: string | null;
    vaeId: string | null;
    vaeSha256: string | null;
    steps: number;
    processingWidth: number;
    processingHeight: number;
    previewSeconds: number;
    previewApproved: boolean;
  };
  rife: {
    enabled: boolean;
    profileId: string | null;
    profileRevision: number | null;
    modelId: string | null;
    modelSha256: string | null;
    targetFps: 48 | 60 | 120;
    workflowTemplateId?: string | null;
    workflowRevision?: number | null;
  };
  whisper: {
    enabled: boolean;
    workflowTemplateId?: string | null;
    workflowRevision?: number | null;
    profileId: string | null;
    profileRevision: number | null;
    modelId: string | null;
    modelSha256: string | null;
    device: "auto" | "cpu" | "cuda";
    precision: "auto" | "int8" | "float16" | "float32";
    language: "zh" | "en" | "auto";
    burnIn: boolean;
  };
};

export type ProjectDraft = {
  projectId: string;
  revision: number;
  name: string;
  width: number;
  height: number;
  fps: number;
  targetDurationSeconds: number;
  audioPolicy: "h3_native" | "h3_with_external" | "muted";
  externalAudioAssetId: string | null;
  executionMode: "guided" | "batch";
  paused: boolean;
  reviewPolicy: {
    configuredMode: "human_ai" | "ai_only" | "manual" | "none";
    effectiveMode: "human_ai" | "ai_only" | "manual" | "none";
    humanTimeoutSeconds: number;
    aiTakeoverAt: string | null;
  };
  timeBudgetSeconds: number | null;
  activeStage: PipelineStageId;
  stageApprovals: Partial<Record<PipelineStageId, string>>;
  idea: {
    concept: string;
    conceptDocument: RichTextDocument;
    genre: string;
    visualStyle: string;
    audience: string;
  };
  outline: OutlineBeat[];
  shots: ShotDraft[];
  assetPlans: AssetPlan[];
  assets: AssetDraft[];
  referenceAssetMode: "planned" | "none";
  prompts: PromptSet;
  reviewNotes: string;
  postProcessing: PostProcessingSettings;
  updatedAt: string;
};

export type AgentPatch = {
  op: "add" | "remove" | "replace";
  path: string;
  value?: unknown;
};

export type AgentProposal = {
  operation_id: string;
  patches: AgentPatch[];
  rationale: string;
  warnings: string[];
};

export type ProjectMemoryEvent = {
  schema_version: "1.0";
  event_id: string;
  project_id: string;
  kind: "message" | "tool" | "fact" | "constraint" | "approval" | "summary";
  source: "user" | "project_agent" | "subagent" | "system";
  role: string;
  content: string;
  input_sha256: string | null;
  output_sha256: string | null;
  created_at: string;
};

export type TaskKind =
  | "llm_planning"
  | "image_generation"
  | "conditioning_encoding"
  | "h3_generation"
  | "ai_review"
  | "model_switch"
  | "seedvr2"
  | "rife"
  | "whisper"
  | "master_assembly"
  | "export"
  | "asset_transfer";

export type TaskState =
  | "blocked"
  | "ready"
  | "queued"
  | "paused"
  | "running"
  | "needs_review"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "stale";

export type TaskSpec = {
  schema_version: "1.0";
  task_id: string;
  project_id: string;
  kind: TaskKind;
  state: TaskState;
  idempotency_key: string;
  input_fingerprint: string;
  depends_on: string[];
  execution_target: "local" | "remote";
  worker_id: string | null;
  affinity_key: string | null;
  priority: number;
  attempt: number;
  max_attempts: number;
  lease_expires_at: string | null;
  comfyui_prompt_id: string | null;
  error_code: string | null;
  error_message: string | null;
};

export type ProjectRunState = {
  project_id: string;
  execution_mode: "guided" | "batch";
  pending_mode: "guided" | "batch" | null;
  paused: boolean;
  outline_approved: boolean;
  generation_revision: number;
  current_stage: string;
  review_policy: {
    configured_mode: "human_ai" | "ai_only" | "manual" | "none";
    effective_mode: "human_ai" | "ai_only" | "manual" | "none";
    human_timeout_seconds: number;
    ai_takeover_at: string | null;
    takeover_reason: string | null;
  };
  time_budget: {
    total_seconds: number;
    spent_active_seconds: number;
    estimated_remaining_seconds: number | null;
    max_ai_retries: number;
  } | null;
  updated_at: string;
};

export type BatchRun = {
  schema_version: "1.0";
  batch_id: string;
  name: string;
  state: "draft" | "running" | "paused" | "completed" | "cancelled";
  items: Array<{
    project_id: string;
    task_ids: string[];
    start_boundary: string;
    priority: number;
  }>;
  created_at: string;
  updated_at: string;
};

export type HarnessBundle = {
  schema_version: "1.0";
  harness_id: string;
  name: string;
  purpose: string;
  workflow_template_id: string | null;
  sources: Array<Record<string, unknown>>;
};

export type HarnessRevision = {
  schema_version: "1.0" | "2.0";
  harness_id: string;
  revision: number;
  markdown: string;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  content_sha256: string;
  approval: "draft" | "approved" | "rejected";
  workflow_template_id: string | null;
  workflow_revision: number | null;
  created_at: string;
};

export type Health = {
  status: string;
  version: string;
  build_stage: string;
  service_id: string;
  instance_nonce: string;
};

export type SetupStatus = {
  ready: boolean;
  data_root: string;
  data_root_writable: boolean;
  ffmpeg_ready: boolean;
  ffprobe_ready: boolean;
  llm_configured: boolean;
  llm_api_key_configured: boolean;
  h3_harness_ready: boolean;
  comfyui_online: boolean;
  blockers: string[];
};

export type PostProcessProfileCapability = {
  profile: {
    profile_id: string;
    revision: number;
    name: string;
    kind: "restoration" | "interpolation" | "transcription";
    engine: string;
    supported_target_fps: Array<48 | 60 | 120>;
    workflow_file?: string | null;
  };
  available: boolean;
  models: string[];
  auxiliary_models: string[];
  blockers: string[];
};

export type PostProcessingCapabilities = {
  schema_version: "1.0";
  server_online: boolean;
  profiles: PostProcessProfileCapability[];
};

export type Binding = {
  semantic: string;
  nodeId: string;
  inputName: string;
  title: string;
  valueType: "string" | "integer" | "number" | "image_path" | "video_path";
};

export type WorkflowBindingSemantic =
  | "prompt"
  | "negative_prompt"
  | "width"
  | "height"
  | "steps"
  | "cfg"
  | "seed"
  | "batch_size"
  | "reference_image"
  | "source_video"
  | "model"
  | "interpolation_factor"
  | "upscale_factor"
  | "language";

export type WorkflowDraft = {
  fileName: string;
  nodes: ApiWorkflow;
  bindings: Binding[];
  outputNodeId: string | null;
  kind?: "image" | "interpolation" | "restoration" | "transcription";
  inspection?: BackendInspection;
};

export type BackendBinding = {
  binding_id: string;
  semantic: WorkflowBindingSemantic;
  node_id: string;
  input_name: string;
  value_type: Binding["valueType"];
  title: string;
  default_value?: unknown;
  string_template?: string | null;
  minimum?: number | null;
  maximum?: number | null;
  reference_index?: number | null;
  confidence?: number;
  rationale?: string;
};

export type BackendInspection = {
  workflow_sha256: string;
  node_schema_sha256: string;
  raw_workflow: ApiWorkflow;
  bindings: BackendBinding[];
  outputs: Array<{ output_id: string; node_id: string; output_type: "image"; title: string }>;
  required_node_types: string[];
  unknown_node_types: string[];
  issues: string[];
  compatible: boolean;
};

export type WorkflowTemplateSummary = {
  template_id: string;
  revision: number;
  name: string;
  approval: "draft" | "needs_confirmation" | "approved" | "rejected";
  built_in?: boolean;
  kind?: "image" | "interpolation" | "restoration" | "transcription";
};

export type ProjectExecutionStatus = {
  workspace_revision: number;
  compiled: boolean;
  task_count: number;
  tasks: TaskSpec[];
  generation_complete: boolean;
  review_complete: boolean;
  delivery_complete: boolean;
  batches: GenerationBatch[];
  versions: SegmentGenerationVersion[];
  rework_markers: ReworkMarker[];
  hierarchy: Array<{
    shot_id: string;
    segments: Array<{
      segment_id: string;
      segment_index: number;
      active_version: SegmentGenerationVersion | null;
      versions: SegmentGenerationVersion[];
      task: TaskSpec | null;
      frozen: boolean;
      freeze_reason: string | null;
    }>;
  }>;
  active_chain_complete: boolean;
  delivery_blocked_reason: string | null;
};

export type GenerationBatch = {
  batch_id: string;
  project_id: string;
  kind: "initial" | "rework";
  state: "preparing" | "sealed" | "running" | "completed" | "cancelled";
  generation_number: number;
  segment_ids: string[];
  task_ids: string[];
  encoding_task_ids: string[];
  model_switch_task_id: string | null;
  dispatch_requested: boolean;
  sealed_at: string | null;
  created_at: string;
  updated_at: string;
};

export type SegmentGenerationVersion = {
  version_id: string;
  project_id: string;
  shot_id: string;
  segment_id: string;
  runtime_segment_id: string;
  segment_index: number;
  generation_number: number;
  batch_id: string;
  task_id: string;
  parent_version_id: string | null;
  predecessor_version_id: string | null;
  prompt_revision_id: string | null;
  seed: number;
  state: "planned" | "running" | "active" | "superseded" | "discarded";
  artifact_id: string | null;
  discard_reason: string | null;
  created_at: string;
  activated_at: string | null;
};

export type ReworkMarker = {
  marker_id: string;
  project_id: string;
  shot_id: string;
  segment_id: string;
  segment_index: number;
  source_version_id: string;
  action: "retry" | "change_seed" | "revise_prompt";
  feedback: string;
  replacement_seed: number | null;
  state: "draft" | "preparing" | "sealed" | "resolved" | "cancelled";
  batch_id: string | null;
  source: "human" | "ai";
  created_at: string;
  updated_at: string;
};

export type ArtifactDescriptor = {
  schema_version: "1.0";
  artifact_id: string;
  task_id: string;
  project_id: string;
  kind: "video_segment" | "video_master" | "video_export" | "subtitle";
  media_type: string;
  file_name: string;
  byte_size: number;
  sha256: string | null;
  segment_id: string | null;
  media_url: string;
  created_at: string;
};

export type ReviewIssue = {
  category: "media" | "black_frame" | "freeze" | "audio" | "anatomy" | "identity" | "motion" | "continuity" | "semantic" | "composition" | "artifact" | "other";
  severity: "info" | "warning" | "error";
  message: string;
  start_seconds: number | null;
  end_seconds: number | null;
  evidence: string | null;
  suggested_action: string | null;
};

export type ReviewDecision = {
  schema_version: "1.0";
  decision_id: string;
  task_id: string;
  project_id: string;
  segment_id: string;
  disposition: "accepted" | "rejected" | "needs_human";
  confidence: number;
  issues: ReviewIssue[];
  input_mode: "video" | "frames";
  fallback_reason: string | null;
  deterministic_checks_passed: boolean;
  created_at: string;
};

export type ReworkRequest = {
  schema_version: "1.0";
  request_id: string;
  project_id: string;
  segment_id: string;
  source_h3_task_id: string;
  review_task_id: string | null;
  action: "retry" | "change_seed" | "revise_prompt";
  feedback: string;
  replacement_seed: number | null;
  state: "requested" | "queued" | "needs_prompt_revision";
  replacement_task_id: string | null;
  created_at: string;
};
