import {
  ArrowLeft,
  Check,
  ChevronRight,
  CircleAlert,
  Clapperboard,
  FolderOpen,
  FilePlus2,
  ListChecks,
  MonitorCog,
  PanelsTopLeft,
  Plus,
  Sparkles,
  Trash2,
  Upload,
  Workflow,
  Layers3,
} from "lucide-react";
import { ChangeEvent, KeyboardEvent, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  getDesktopControlPlaneStatus,
  getLocalWorkerCapabilities,
  getHealth,
  getProjectRunState,
  getProjectWorkspace,
  getSetupStatus,
  inspectWorkflow,
  listProjectAssets,
  listProjects,
  listWorkflowTemplates,
  registerWorkflow,
  saveProjectToControlPlane,
  suggestWorkflowBindings,
} from "./api";
import { BatchView } from "./BatchView";
import { HarnessEditor } from "./HarnessEditor";
import { SettingsView } from "./SettingsView";
import { SetupWizard } from "./SetupWizard";
import { PipelineView } from "./PipelineView";
import { ProjectSwitcher } from "./ProjectSwitcher";
import { hydrateProject, loadProject, newProject } from "./project-store";
import { TasksView } from "./TasksView";
import type { BackendProjectSpec } from "./api";
import type { ApiWorkflow, BackendBinding, Binding, ProjectDraft, SetupStatus, WorkflowBindingSemantic, WorkflowDraft, WorkflowNode } from "./types";

type View = "pipeline" | "workflow" | "batch" | "tasks" | "settings";

function activateFileLabel(event: KeyboardEvent<HTMLLabelElement>) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    event.currentTarget.querySelector<HTMLInputElement>('input[type="file"]')?.click();
  }
}

const nav: { id: View; label: string; icon: typeof Clapperboard }[] = [
  { id: "pipeline", label: "制作流程", icon: PanelsTopLeft },
  { id: "workflow", label: "图片工作流", icon: Workflow },
  { id: "batch", label: "批量工作台", icon: Layers3 },
  { id: "tasks", label: "任务进度", icon: ListChecks },
  { id: "settings", label: "全局设置", icon: MonitorCog },
];

const workflowSemantics: Array<{ value: WorkflowBindingSemantic; label: string; valueType: BackendBinding["value_type"] }> = [
  { value: "prompt", label: "正向提示词", valueType: "string" },
  { value: "negative_prompt", label: "负向提示词", valueType: "string" },
  { value: "width", label: "宽度", valueType: "integer" },
  { value: "height", label: "高度", valueType: "integer" },
  { value: "steps", label: "步数", valueType: "integer" },
  { value: "cfg", label: "CFG", valueType: "number" },
  { value: "seed", label: "Seed", valueType: "integer" },
  { value: "batch_size", label: "批量数", valueType: "integer" },
  { value: "reference_image", label: "参考图", valueType: "image_path" },
];

function editableInputNames(node: WorkflowNode): string[] {
  return Object.entries(node.inputs)
    .filter(([, value]) => !(Array.isArray(value) && value.length === 2 && typeof value[0] === "string" && typeof value[1] === "number"))
    .map(([name]) => name);
}

function toDisplayBindings(bindings: BackendBinding[]): Binding[] {
  return bindings.map((binding) => ({
    semantic: binding.semantic,
    nodeId: binding.node_id,
    inputName: binding.input_name,
    title: binding.title,
    valueType: binding.value_type,
  }));
}

function analyzeWorkflow(fileName: string, nodes: ApiWorkflow): WorkflowDraft {
  return { fileName, nodes, bindings: [], outputNodeId: null };
}

function App() {
  const [view, setView] = useState<View>("pipeline");
  const [project, setProject] = useState(loadProject);
  const [controlPlaneOnline, setControlPlaneOnline] = useState(false);
  const [controlPlaneLabel, setControlPlaneLabel] = useState("正在连接控制平面");
  const [draft, setDraft] = useState<WorkflowDraft | null>(null);
  const [step, setStep] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [registered, setRegistered] = useState<Array<{ name: string; id: string; revision: number; builtIn: boolean }>>([]);
  const [workerOnline, setWorkerOnline] = useState(false);
  const [workerBlockers, setWorkerBlockers] = useState<string[]>([]);
  const [harnessTarget, setHarnessTarget] = useState<{ name: string; id: string; revision: number } | null>(null);
  const [unknownConfirmed, setUnknownConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [projectDialog, setProjectDialog] = useState<"open" | "new" | null>(null);
  const [projects, setProjects] = useState<BackendProjectSpec[]>([]);
  const [projectBusy, setProjectBusy] = useState(false);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [startupLoading, setStartupLoading] = useState(true);
  const [workflowNotice, setWorkflowNotice] = useState<string | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [setupDismissed, setSetupDismissed] = useState(false);

  const nodeRows = useMemo(() => (draft ? Object.entries(draft.nodes) : []), [draft]);
  const outputNodeRows = useMemo(() => {
    if (!draft) return [];
    const detected = new Set(draft.inspection?.outputs.map((output) => output.node_id) ?? []);
    return nodeRows.filter(([id]) => detected.has(id));
  }, [draft, nodeRows]);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        await getHealth();
        if (active) {
          setControlPlaneOnline(true);
          setControlPlaneLabel("控制平面在线");
          const status = await getSetupStatus().catch(() => null);
          if (active && status) setSetupStatus(status);
        }
      } catch {
        const desktopStatus = await getDesktopControlPlaneStatus().catch(() => null);
        if (active) { setControlPlaneOnline(false); setControlPlaneLabel(desktopStatus?.message ?? "控制平面离线"); }
      }
    };
    void check();
    const timer = window.setInterval(() => void check(), 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const [templates, capabilities] = await Promise.all([
          listWorkflowTemplates(),
          getLocalWorkerCapabilities(),
        ]);
        if (!active) return;
        setRegistered(templates.map((item) => ({
          name: item.name,
          id: item.template_id,
          revision: item.revision,
          builtIn: item.built_in,
        })));
        setWorkerOnline(capabilities.full_pipeline_ready);
        setWorkerBlockers(capabilities.execution_blockers);
      } catch {
        if (!active) return;
        setWorkerOnline(false);
        setWorkerBlockers(["完整执行能力探测失败"]);
        try {
          const templates = await listWorkflowTemplates();
          if (active) setRegistered(templates.map((item) => ({
            name: item.name,
            id: item.template_id,
            revision: item.revision,
            builtIn: item.built_in,
          })));
        } catch { /* the global control-plane banner already reports this */ }
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 10_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const projects = await listProjects();
        if (active) setProjects(projects);
        const latest = projects[0];
        if (!latest) {
          if (active) setProjectDialog("new");
          return;
        }
        const workspace = hydrateProject(await getProjectWorkspace(latest.project_id));
        let assets = workspace.assets;
        try { assets = await listProjectAssets(latest.project_id); }
        catch (cause) { if (active) setStartupError(`素材同步失败，正在显示项目工作区副本：${cause instanceof Error ? cause.message : "未知错误"}`); }
        let run = null;
        try { run = await getProjectRunState(latest.project_id); }
        catch (cause) { if (active) setStartupError(`项目运行状态读取失败：${cause instanceof Error ? cause.message : "未知错误"}`); }
        if (active) setProject({
          ...workspace,
          assets,
          executionMode: run?.execution_mode ?? workspace.executionMode,
          paused: run?.paused ?? workspace.paused,
          reviewPolicy: run ? {
            configuredMode: run.review_policy.configured_mode,
            effectiveMode: run.review_policy.effective_mode,
            humanTimeoutSeconds: run.review_policy.human_timeout_seconds,
            aiTakeoverAt: run.review_policy.ai_takeover_at,
          } : workspace.reviewPolicy,
        });
      } catch (cause) {
        if (active) setStartupError(`项目加载失败：${cause instanceof Error ? cause.message : "未知错误"}`);
      } finally {
        if (active) setStartupLoading(false);
      }
    })();
    return () => { active = false; };
  }, []);

  async function openSavedProject(summary: BackendProjectSpec) {
    setProjectBusy(true);
    setProjectError(null);
    try {
      let workspace: ProjectDraft;
      try {
        workspace = hydrateProject(await getProjectWorkspace(summary.project_id));
        try { workspace = { ...workspace, assets: await listProjectAssets(summary.project_id) }; }
        catch (cause) {
          setStartupError(`素材同步失败，正在显示项目工作区副本：${cause instanceof Error ? cause.message : "未知错误"}`);
        }
      } catch (cause) {
        if (!(cause instanceof ApiError) || cause.status !== 404) {
          throw cause;
        }
        workspace = hydrateProject({
          ...newProject(),
          projectId: summary.project_id,
          revision: summary.revision,
          name: summary.name,
          width: summary.width,
          height: summary.height,
          fps: summary.fps,
          targetDurationSeconds: summary.target_duration_seconds,
          audioPolicy: summary.audio_policy,
          externalAudioAssetId: summary.external_audio_asset_id,
          activeStage: "config",
        });
      }
      let run = null;
      try { run = await getProjectRunState(summary.project_id); }
      catch (cause) { setStartupError(`项目运行状态读取失败：${cause instanceof Error ? cause.message : "未知错误"}`); }
      setProject({
        ...workspace,
        executionMode: run?.execution_mode ?? workspace.executionMode,
        paused: run?.paused ?? workspace.paused,
        reviewPolicy: run ? {
          configuredMode: run.review_policy.configured_mode,
          effectiveMode: run.review_policy.effective_mode,
          humanTimeoutSeconds: run.review_policy.human_timeout_seconds,
          aiTakeoverAt: run.review_policy.ai_takeover_at,
        } : workspace.reviewPolicy,
      });
      setView("pipeline");
      setProjectDialog(null);
    } catch (cause) {
      setProjectError(cause instanceof Error ? cause.message : "无法打开项目");
    } finally {
      setProjectBusy(false);
    }
  }

  async function createNewProject(input: { name: string; width: string; height: string; targetDurationSeconds: string }) {
    const name = input.name.trim();
    const rawWidth = Number(input.width);
    const rawHeight = Number(input.height);
    const width = Math.max(64, Math.round(rawWidth / 32) * 32);
    const height = Math.max(64, Math.round(rawHeight / 32) * 32);
    const duration = Number(input.targetDurationSeconds);
    if (!name) { setProjectError("请输入项目名称"); return; }
    if (
      !input.width.trim()
      || !input.height.trim()
      || !Number.isFinite(rawWidth)
      || !Number.isFinite(rawHeight)
      || rawWidth <= 0
      || rawHeight <= 0
    ) {
      setProjectError("请填写有效的宽度和高度");
      return;
    }
    if (!Number.isFinite(duration) || duration <= 0) { setProjectError("目标时长必须大于 0 秒"); return; }
    setProjectBusy(true);
    setProjectError(null);
    try {
      const base = newProject();
      const created = await saveProjectToControlPlane({
        ...base,
        name,
        width,
        height,
        targetDurationSeconds: duration,
        postProcessing: { ...base.postProcessing, outputWidth: width, outputHeight: height },
      });
      setProject(created);
      setProjects(await listProjects());
      setView("pipeline");
      setProjectDialog(null);
    } catch (cause) {
      setProjectError(cause instanceof Error ? cause.message : "无法创建项目");
    } finally {
      setProjectBusy(false);
    }
  }

  async function importWorkflow(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);
    setWorkflowNotice(null);
    try {
      const parsed: unknown = JSON.parse(await file.text());
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("文件不是ComfyUI API工作流");
      }
      const nodes = parsed as ApiWorkflow;
      if (
        !Object.keys(nodes).length ||
        Object.values(nodes).some(
          (node) => !node || typeof node.class_type !== "string" || !node.inputs,
        )
      ) {
        throw new Error("缺少API工作流所需的class_type或inputs");
      }
      const localDraft = analyzeWorkflow(file.name, nodes);
      setDraft(localDraft);
      setStep(2);
      setUnknownConfirmed(false);
      try {
        const inspection = await inspectWorkflow(nodes);
        setDraft((current) => current ? {
          ...current,
          inspection,
          bindings: toDisplayBindings(inspection.bindings),
          outputNodeId: null,
        } : current);
        setWorkflowNotice(`节点检查完成：${Object.keys(inspection.raw_workflow).length} 个节点、${inspection.outputs.length} 个可选输出；下一步由 LLM 或用户标定输入`);
      } catch (cause) {
        setError(`控制平面校验失败：${cause instanceof Error ? cause.message : "未知错误"}`);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法解析工作流");
    } finally {
      event.target.value = "";
    }
  }

  async function requestLlmMapping() {
    if (!draft) return;
    setBusy(true);
    setError(null);
    setWorkflowNotice("LLM 正在分析节点和可编辑字段…");
    try {
      const result = await suggestWorkflowBindings(draft.fileName, draft.nodes, project);
      setDraft((current) => current?.inspection ? {
        ...current,
        inspection: { ...current.inspection, bindings: result.bindings },
        bindings: toDisplayBindings(result.bindings),
      } : current);
      setWorkflowNotice(result.bindings.length
        ? `LLM 已填入 ${result.bindings.length} 条映射建议，请逐项确认`
        : "LLM 未找到可安全暴露的字段，请手动添加映射");
    } catch (cause) {
      setError(`LLM映射失败：${cause instanceof Error ? cause.message : "未知错误"}`);
      setWorkflowNotice(null);
    } finally {
      setBusy(false);
    }
  }

  function updateWorkflowBindings(transform: (bindings: BackendBinding[], current: WorkflowDraft) => BackendBinding[]) {
    setDraft((current) => {
      if (!current?.inspection) return current;
      const bindings = transform([...current.inspection.bindings], current);
      return {
        ...current,
        bindings: toDisplayBindings(bindings),
        inspection: { ...current.inspection, bindings },
      };
    });
    setWorkflowNotice("映射已修改，登记前将再次进行确定性校验");
  }

  function changeWorkflowBinding(index: number, field: "semantic" | "node" | "input", value: string) {
    updateWorkflowBindings((bindings, current) => {
      const existing = bindings[index];
      if (!existing) return bindings;
      let nodeId = existing.node_id;
      let inputName = existing.input_name;
      let semantic = existing.semantic;
      if (field === "semantic") semantic = value as WorkflowBindingSemantic;
      if (field === "node") {
        nodeId = value;
        inputName = editableInputNames(current.nodes[nodeId])[0] ?? Object.keys(current.nodes[nodeId].inputs)[0] ?? "";
      }
      if (field === "input") inputName = value;
      const semanticDefinition = workflowSemantics.find((item) => item.value === semantic) ?? workflowSemantics[0];
      const referenceIndexes = bindings
        .filter((_, bindingIndex) => bindingIndex !== index)
        .map((binding) => binding.reference_index ?? 0);
      const referenceIndex = semantic === "reference_image"
        ? existing.reference_index ?? Math.max(0, ...referenceIndexes) + 1
        : null;
      const node = current.nodes[nodeId];
      bindings[index] = {
        ...existing,
        binding_id: `${semantic}:${nodeId}:${inputName}:${index + 1}`,
        semantic,
        node_id: nodeId,
        input_name: inputName,
        value_type: semanticDefinition.valueType,
        title: node._meta?.title?.trim() || `${node.class_type} #${nodeId}`,
        default_value: node.inputs[inputName],
        string_template: semanticDefinition.valueType === "string" && typeof node.inputs[inputName] === "string"
          ? existing.string_template ?? null
          : null,
        reference_index: referenceIndex,
        confidence: 1,
        rationale: "用户在导入向导中确认",
      };
      return bindings;
    });
  }

  function addWorkflowBinding() {
    if (!draft?.inspection) return;
    const candidate = Object.entries(draft.nodes).find(([, node]) => editableInputNames(node).length);
    if (!candidate) {
      setError("工作流中没有可进行类型化赋值的输入字段");
      return;
    }
    updateWorkflowBindings((bindings) => {
      const [nodeId, node] = candidate;
      const inputName = editableInputNames(node)[0];
      const index = bindings.length;
      return [...bindings, {
        binding_id: `prompt:${nodeId}:${inputName}:${index + 1}`,
        semantic: "prompt",
        node_id: nodeId,
        input_name: inputName,
        value_type: "string",
        title: node._meta?.title?.trim() || `${node.class_type} #${nodeId}`,
        default_value: node.inputs[inputName],
        string_template: null,
        minimum: null,
        maximum: null,
        reference_index: null,
        confidence: 1,
        rationale: "用户在导入向导中添加",
      }];
    });
  }

  async function registerWorkflowTemplate() {
    if (!draft?.outputNodeId) {
      setError("请选择图片输出节点");
      return;
    }
    if (!draft.inspection) {
      setError("控制平面尚未完成节点和字段校验");
      return;
    }
    if (draft.inspection.unknown_node_types.length && !unknownConfirmed) {
      setError("请先确认工作流中的自定义节点");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const cleanBindings = draft.inspection.bindings.map(({ confidence: _, rationale: __, ...binding }) => binding);
      const outputNode = draft.nodes[draft.outputNodeId];
      const templateId = `user:${draft.inspection.workflow_sha256.slice(0, 16)}`;
      await registerWorkflow({
        schema_version: "1.0",
        template_id: templateId,
        revision: 1,
        name: draft.fileName.replace(/\.json$/i, ""),
        workflow_sha256: draft.inspection.workflow_sha256,
        node_schema_sha256: draft.inspection.node_schema_sha256,
        raw_workflow: draft.inspection.raw_workflow,
        bindings: cleanBindings,
        outputs: [{
          output_id: `image:${draft.outputNodeId}`,
          node_id: draft.outputNodeId,
          output_type: "image",
          title: outputNode._meta?.title?.trim() || `${outputNode.class_type} #${draft.outputNodeId}`,
        }],
        required_node_types: draft.inspection.required_node_types,
        unknown_node_types: [],
        approval: "approved",
        built_in: false,
      });
      const registeredWorkflow = { name: draft.fileName, id: templateId, revision: 1, builtIn: false };
      setRegistered((items) => [...items, registeredWorkflow]);
      setHarnessTarget({ name: draft.fileName, id: templateId, revision: 1 });
      setDraft(null);
      setStep(1);
      setUnknownConfirmed(false);
      setWorkflowNotice(`${draft.fileName} 已登记；请继续配置并批准图片 Harness`);
    } catch (cause) {
      setError(`登记失败：${cause instanceof Error ? cause.message : "未知错误"}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><MonitorCog size={20} /><strong>AI Video Generator</strong></div>
        <nav>
          {nav.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} title={item.label} aria-label={item.label} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}>
                <Icon size={18} /><span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className={`worker-state ${controlPlaneOnline && workerOnline ? "" : "offline"}`} title={!controlPlaneOnline ? controlPlaneLabel : workerOnline ? "完整 H3 执行链已就绪" : workerBlockers.join("；") || "完整执行链未就绪"}><span className="status-dot" />{!controlPlaneOnline ? controlPlaneLabel : workerOnline ? "H3 执行链就绪" : "H3 执行链未就绪"}</div>
      </aside>

      <main>
        <header className="topbar">
          <div><h1>{nav.find((item) => item.id === view)?.label}</h1><span>{startupLoading ? "正在读取最近项目..." : `${project.name} · 修订 ${project.revision}`}</span></div>
          <div className="project-commands">
            <button className="secondary-button" disabled={startupLoading} onClick={() => { setProjectError(null); setProjectDialog("new"); }}><FilePlus2 size={16} />新建</button>
            <button className="secondary-button" disabled={startupLoading} onClick={() => { setProjectError(null); void listProjects().then(setProjects).catch((cause) => setProjectError(cause instanceof Error ? cause.message : "无法读取项目列表")); setProjectDialog("open"); }}><FolderOpen size={16} />打开</button>
          </div>
        </header>

        {startupError && <div className="global-error-banner"><CircleAlert size={16} /><span>{startupError}</span><button className="icon-button" title="关闭错误" onClick={() => setStartupError(null)}>×</button></div>}

        {!startupLoading && projectDialog && <ProjectSwitcher mode={projectDialog} projects={projects} activeProjectId={project.projectId} busy={projectBusy} error={projectError} onClose={() => setProjectDialog(null)} onModeChange={(mode) => { setProjectError(null); setProjectDialog(mode); }} onOpen={(item) => void openSavedProject(item)} onCreate={(value) => void createNewProject(value)} />}

        {startupLoading ? <section className="startup-loading" aria-live="polite"><Sparkles size={22} /><div><strong>正在加载项目</strong><span>正在读取最近项目、素材和运行状态...</span></div></section> : view === "pipeline" ? <PipelineView project={project} workflows={registered} workerOnline={workerOnline} onChange={setProject} onOpenWorkflows={() => setView("workflow")} onOpenTasks={() => setView("tasks")} onOpenBatch={() => setView("batch")} /> : view === "settings" ? <SettingsView /> : view === "batch" ? <BatchView project={project} projects={projects} /> : view === "tasks" ? <TasksView project={project} /> : (
          <section className="workspace">
            <div className="section-heading">
              <div><h2>图片工作流模板</h2><span>{registered.length} 个已登记模板 · H3 工作流由软件提供</span></div>
              <label className="primary-button" role="button" aria-label="导入图片工作流 JSON" tabIndex={0} onKeyDown={activateFileLabel}><Upload size={17} />导入图片工作流<input type="file" accept="application/json,.json" onChange={importWorkflow} /></label>
            </div>

            <div className="template-table">
              {registered.map((item, index) => (
                <div className="template-row" key={`${item.name}-${index}`}>
                  <div className="template-icon"><Workflow size={18} /></div>
                  <div><strong>{item.name}</strong><span>{item.builtIn ? "内置模板 · 后端已登记" : `用户工作流 · 修订 ${item.revision}`}</span></div>
                  <span className="approved"><Check size={14} />已登记</span>
                  <button className="icon-button" title="编辑 Harness" onClick={() => setHarnessTarget({ name: item.name, id: item.id, revision: item.revision })}><ChevronRight size={18} /></button>
                </div>
              ))}
            </div>

            {harnessTarget && <HarnessEditor workflowId={harnessTarget.id} workflowRevision={harnessTarget.revision} workflowName={harnessTarget.name} />}

            <div className="import-panel">
              <div className="stepper">
                {["上传", "节点", "输入", "输出"].map((label, index) => (
                  <div className={step >= index + 1 ? "step current" : "step"} key={label}>
                    <span>{index + 1}</span>{label}
                  </div>
                ))}
              </div>

              {error && <div className="error-banner"><CircleAlert size={17} />{error}</div>}
              {workflowNotice && <div className="validation-ok workflow-notice"><Check size={15} />{workflowNotice}</div>}
              {!draft ? (
                <label className="dropzone" role="button" aria-label="选择 ComfyUI 图片生成 API 工作流 JSON" tabIndex={0} onKeyDown={activateFileLabel}><Upload size={28} /><strong>选择图片生成 API 工作流</strong><span>ComfyUI API JSON</span><input type="file" accept="application/json,.json" onChange={importWorkflow} /></label>
              ) : (
                <>
                  <div className="file-summary"><div><strong>{draft.fileName}</strong><span>{nodeRows.length} 个节点 · {draft.bindings.length} 个已选输入</span></div></div>
                  {step === 2 && <>
                    {!draft.inspection && <div className="workflow-pending">正在从 ComfyUI 读取节点接口…</div>}
                    {draft.inspection?.issues.length ? <div className="workflow-issues"><strong>需要在后续步骤确认</strong>{draft.inspection.issues.map((issue) => <span key={issue}>{issue}</span>)}</div> : null}
                    {draft.inspection?.unknown_node_types.length ? (
                      <label className="unknown-confirm">
                        <input type="checkbox" checked={unknownConfirmed} onChange={(event) => setUnknownConfirmed(event.target.checked)} />
                        已了解并确认本机自定义节点：{draft.inspection.unknown_node_types.join("、")}
                      </label>
                    ) : draft.inspection ? <div className="validation-ok"><Check size={15} />所有节点类型均存在于当前 ComfyUI</div> : null}
                    <div className="node-table">
                      <div className="node-header"><span>节点</span><span>类型</span><span>状态</span></div>
                      {nodeRows.map(([id, node]) => {
                        const outputCandidate = draft.inspection?.outputs.some((output) => output.node_id === id);
                        return <div className="node-row" key={id}><span>#{id} {node._meta?.title ?? "未命名"}</span><code>{node.class_type}</code><span>{outputCandidate ? "可选输出" : editableInputNames(node).length ? "可标定输入" : "固定连接"}</span></div>;
                      })}
                    </div>
                    <div className="panel-actions">
                      <button className="secondary-button" onClick={() => { setDraft(null); setStep(1); setWorkflowNotice(null); }}>取消导入</button>
                      <button className="primary-button" onClick={() => { setStep(3); setError(null); setWorkflowNotice("节点已确认，请检查需要暴露给项目的输入字段"); }} disabled={!draft.inspection || busy || Boolean(draft.inspection.unknown_node_types.length && !unknownConfirmed)}>确认节点并继续<ChevronRight size={16} /></button>
                    </div>
                  </>}

                  {step === 3 && draft.inspection && <>
                    <div className="workflow-step-heading"><div><strong>输入字段映射</strong><span>只会类型化修改下列字段，其他节点参数保持工作流原值</span></div><button className="secondary-button" onClick={requestLlmMapping} disabled={busy}><Sparkles size={16} />{busy ? "LLM 分析中" : "LLM 协助识别"}</button></div>
                    {!draft.inspection.bindings.length && <div className="workflow-empty">尚未暴露任何输入。使用 LLM 协助识别，或手动添加一项。</div>}
                    <div className="binding-table">
                      {draft.inspection.bindings.map((binding, index) => (
                        <div className="binding-row" key={`${binding.binding_id}:${index}`}>
                          <label><span>用途</span><select value={binding.semantic} onChange={(event) => changeWorkflowBinding(index, "semantic", event.target.value)}>{workflowSemantics.map((semantic) => <option value={semantic.value} key={semantic.value}>{semantic.label}</option>)}</select></label>
                          <label><span>节点</span><select value={binding.node_id} onChange={(event) => changeWorkflowBinding(index, "node", event.target.value)}>{nodeRows.filter(([id, node]) => id === binding.node_id || editableInputNames(node).length).map(([id, node]) => <option value={id} key={id}>#{id} {node._meta?.title || node.class_type}</option>)}</select></label>
                          <label><span>输入字段</span><select value={binding.input_name} onChange={(event) => changeWorkflowBinding(index, "input", event.target.value)}>{Array.from(new Set([binding.input_name, ...editableInputNames(draft.nodes[binding.node_id])])).map((input) => <option value={input} key={input}>{input}</option>)}</select></label>
                          <div className="binding-type"><span>类型</span><code>{binding.value_type}</code></div>
                          <button className="icon-button" title="移除此映射" onClick={() => updateWorkflowBindings((bindings) => bindings.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={16} /></button>
                          {(binding.rationale || binding.confidence !== undefined) && <small>{binding.rationale || "映射建议"}{binding.confidence !== undefined ? ` · 置信度 ${Math.round(binding.confidence * 100)}%` : ""}</small>}
                        </div>
                      ))}
                    </div>
                    <div className="panel-actions split-actions">
                      <button className="secondary-button" onClick={() => setStep(2)}><ArrowLeft size={16} />返回节点</button>
                      <span />
                      <button className="secondary-button" onClick={addWorkflowBinding}><Plus size={16} />添加映射</button>
                      <button className="primary-button" onClick={() => { setStep(4); setError(null); setWorkflowNotice("输入映射已确认，请选择从 ComfyUI history 提取的图片输出节点"); }}>确认输入并继续<ChevronRight size={16} /></button>
                    </div>
                  </>}

                  {step === 4 && draft.inspection && <>
                    <div className="workflow-output-step">
                      <label className="field"><span>图片输出节点</span><select value={draft.outputNodeId ?? ""} onChange={(event) => setDraft((current) => current ? { ...current, outputNodeId: event.target.value || null } : current)}><option value="">请选择输出节点</option>{outputNodeRows.map(([id, node]) => <option value={id} key={id}>#{id} {node._meta?.title || node.class_type}</option>)}</select></label>
                      {!outputNodeRows.length && <div className="error-banner"><CircleAlert size={17} />当前 ComfyUI 未将此工作流中的任何节点声明为输出节点，请检查工作流或节点安装。</div>}
                      <dl><div><dt>输入映射</dt><dd>{draft.inspection.bindings.length} 项</dd></div><div><dt>固定节点</dt><dd>{Math.max(0, nodeRows.length - new Set(draft.inspection.bindings.map((binding) => binding.node_id)).size)} 个</dd></div><div><dt>未知节点</dt><dd>{draft.inspection.unknown_node_types.length} 类（已确认）</dd></div></dl>
                      <p>登记只保存工作流、字段映射和输出提取规则，不会运行图片生成。</p>
                    </div>
                    <div className="panel-actions split-actions">
                      <button className="secondary-button" onClick={() => setStep(3)}><ArrowLeft size={16} />返回输入</button>
                      <span />
                      <button className="primary-button" onClick={registerWorkflowTemplate} disabled={busy || !draft.outputNodeId}><Check size={16} />{busy ? "正在校验" : "校验并登记模板"}</button>
                    </div>
                  </>}
                </>
              )}
            </div>
          </section>
        )}
      </main>
      {controlPlaneOnline && setupStatus && !setupStatus.ready && !setupDismissed && <SetupWizard
        status={setupStatus}
        onStatus={setSetupStatus}
        onOpenSettings={() => { setSetupDismissed(true); setView("settings"); }}
        onDismiss={() => setSetupDismissed(true)}
      />}
    </div>
  );
}

export default App;
