import { CircleAlert, ListFilter, Pause, Play, RefreshCw, Search, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cancelBatchProject, createBatch, getProjectRunState, listBatches, listTasks, pauseBatchProject, taskCommands, transitionBatch } from "./api";
import type { BackendProjectSpec } from "./api";
import type { BatchRun, BatchSettings, ProjectDraft, TaskSpec, TaskState } from "./types";

import { TaskStatus } from "./TaskStatus";
import { activeExecutionStates, batchMemberProgress, batchProgress, canTaskAction, taskExplanation, terminalTaskStates } from "./task-status";
import { describeCommandReceipt } from "./task-commands";

type WorkflowChoice = { id: string; name: string; revision: number; kind?: string };
type Props = { project: ProjectDraft; projects: BackendProjectSpec[]; workflows?: WorkflowChoice[] };
type Selection = { startBoundary: string; priority: number; settings: BatchSettings };
type BatchFilter = "active" | "history" | "all";

function UpscaleFactorInput({ value, placeholder, onChange }: {
  value: number | null | undefined;
  placeholder?: string;
  onChange: (value: number | null) => void;
}) {
  return <input
    type="number"
    min="0.1"
    step="any"
    value={value ?? ""}
    placeholder={placeholder}
    onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))}
  />;
}

const boundaryOptions = [
  { value: "next_ready", label: "从当前未完成任务继续" },
  { value: "generation", label: "生成、审核建议与交付" },
  { value: "delivery", label: "仅后处理与交付" },
];
const terminalStates = terminalTaskStates;
const activeBatchStates = new Set(["draft", "running", "paused"]);
const batchStateLabels: Record<BatchRun["state"], string> = {
  draft: "待启动", running: "运行中", paused: "已暂停", completed: "已结束", cancelled: "已取消",
};


export function BatchView({ project, projects, workflows = [] }: Props) {
  const [tasks, setTasks] = useState<TaskSpec[]>([]);
  const [batches, setBatches] = useState<BatchRun[]>([]);
  const [selected, setSelected] = useState<Map<string, Selection>>(new Map());
  const [batchName, setBatchName] = useState("");
  const [pendingDraft, setPendingDraft] = useState<BatchRun | null>(null);
  const [query, setQuery] = useState("");
  const [batchFilter, setBatchFilter] = useState<BatchFilter>("active");
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null);
  const [offline, setOffline] = useState(false);
  const refreshSequence = useRef(0);
  const [outlineReady, setOutlineReady] = useState<Set<string>>(new Set());
  const [defaults, setDefaults] = useState<BatchSettings>({ imageWorkflowId: null, seedvrEnabled: false, seedvrWorkflowId: null, seedvrUpscaleFactor: 2, rifeEnabled: false, rifeWorkflowId: null, rifeTargetFps: 48 });

  const allProjects = useMemo(() => {
    const values = new Map(projects.map((item) => [item.project_id, item.name]));
    values.set(project.projectId, project.name);
    return [...values.entries()];
  }, [project.name, project.projectId, projects]);

  const refresh = useCallback(async (showBusy = false) => {
    const sequence = ++refreshSequence.current;
    if (showBusy) setRefreshing(true);
    try {
      const [nextTasks, nextBatches, runStates] = await Promise.all([
        listTasks(), listBatches(),
        Promise.allSettled(allProjects.map(([projectId]) => getProjectRunState(projectId))),
      ]);
      if (sequence !== refreshSequence.current) return;
      setTasks(nextTasks);
      setBatches(nextBatches);
      setOutlineReady(new Set(runStates.flatMap((result) => result.status === "fulfilled" && result.value.outline_approved ? [result.value.project_id] : [])));
      setRefreshedAt(new Date()); setOffline(false);
    } catch {
      if (sequence === refreshSequence.current) setOffline(true);
    } finally {
      if (showBusy) setRefreshing(false);
    }
  }, [allProjects]);

  useEffect(() => {
    let stopped = false;
    let timer: number;
    const poll = async () => { await refresh(); if (!stopped) timer = window.setTimeout(() => void poll(), 3000); };
    void poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [refresh]);

  const tasksByProject = useMemo(() => {
    const values = new Map<string, TaskSpec[]>();
    for (const task of tasks) values.set(task.project_id, [...(values.get(task.project_id) ?? []), task]);
    return values;
  }, [tasks]);

  const activeProjectIds = useMemo(() => new Set(
    batches.filter((batch) => activeBatchStates.has(batch.state)).flatMap((batch) => batch.items.map((item) => item.project_id)),
  ), [batches]);
  const visibleProjects = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    return normalized ? allProjects.filter(([projectId, name]) => `${name} ${projectId}`.toLocaleLowerCase().includes(normalized)) : allProjects;
  }, [allProjects, query]);
  const eligibleProjectIds = useMemo(() => visibleProjects.flatMap(([projectId]) => {
    const unfinished = (tasksByProject.get(projectId) ?? []).some((task) => !terminalStates.has(task.state));
    return (unfinished || outlineReady.has(projectId)) && !activeProjectIds.has(projectId) ? [projectId] : [];
  }), [activeProjectIds, outlineReady, tasksByProject, visibleProjects]);
  const visibleBatches = useMemo(() => batches.filter((batch) => batchFilter === "all"
    || (batchFilter === "active" ? activeBatchStates.has(batch.state) : !activeBatchStates.has(batch.state))), [batchFilter, batches]);

  useEffect(() => {
    setSelected((current) => {
      const next = new Map(
        [...current].filter(([projectId]) => !activeProjectIds.has(projectId)),
      );
      return next.size === current.size ? current : next;
    });
  }, [activeProjectIds]);

  function toggleProject(projectId: string, checked: boolean) {
    setSelected((current) => {
      const next = new Map(current);
      if (checked) next.set(projectId, { startBoundary: "next_ready", priority: 0, settings: { ...defaults } });
      else next.delete(projectId);
      return next;
    });
  }

  function toggleAllEligible() {
    const allSelected = eligibleProjectIds.length > 0 && eligibleProjectIds.every((projectId) => selected.has(projectId));
    setSelected((current) => {
      const next = new Map(current);
      for (const projectId of eligibleProjectIds) {
        if (allSelected) next.delete(projectId);
        else if (!next.has(projectId)) next.set(projectId, { startBoundary: "next_ready", priority: 0, settings: { ...defaults } });
      }
      return next;
    });
  }

  const imageWorkflows = workflows.filter((item) => item.kind === "image");
  const restorationWorkflows = workflows.filter((item) => item.kind === "restoration");
  const interpolationWorkflows = workflows.filter((item) => item.kind === "interpolation");
  const updateDefaults = (update: Partial<BatchSettings>) => {
    setDefaults((current) => ({ ...current, ...update }));
  };

  function updateSelection(projectId: string, update: Partial<Selection>) {
    setSelected((current) => {
      const next = new Map(current);
      const existing = next.get(projectId);
      if (existing) next.set(projectId, { ...existing, ...update });
      return next;
    });
  }

  function retainDraft(batch: BatchRun) {
    setPendingDraft(batch);
    setBatches((current) => [...current.filter((item) => item.batch_id !== batch.batch_id), batch]);
    setBatchFilter("active");
  }

  async function startDraft(batch: BatchRun) {
    setBusy(true); setError(null); setNotice(null);
    retainDraft(batch);
    try {
      const started = await transitionBatch(batch.batch_id, "start");
      setBatches((current) => current.map((item) => item.batch_id === started.batch_id ? started : item));
      setPendingDraft(null); setSelected(new Map()); setBatchName("");
      setNotice(`批次“${batch.name}”已启动`);
    } catch (cause) {
      setError(`批次“${batch.name}”未确认启动：${cause instanceof Error ? cause.message : "未知错误"}。草稿已保留，可继续启动同一批次。`);
    } finally { setBusy(false); void refresh(); }
  }

  async function createAndStart() {
    if (busy) return;
    const now = new Date().toISOString();
    const draft: BatchRun = pendingDraft ?? {
      schema_version: "1.0", batch_id: crypto.randomUUID(), name: batchName.trim() || `${selected.size} 个项目批次`, state: "draft",
      settings: defaults,
      items: [...selected].map(([projectId, selection]) => ({ project_id: projectId, task_ids: [], start_boundary: selection.startBoundary, priority: selection.priority, settings: selection.settings })),
      created_at: now, updated_at: now,
    };
    // Keep the identity even if creation was accepted but its HTTP response was lost.
    setPendingDraft(draft); setBusy(true); setError(null); setNotice(null);
    try {
      const existing = (await listBatches()).find((batch) => batch.batch_id === draft.batch_id);
      const resolved = existing ?? await createBatch(draft);
      if (resolved.state !== "draft") {
        setBatches((current) => [...current.filter((item) => item.batch_id !== resolved.batch_id), resolved]);
        setPendingDraft(null); setSelected(new Map()); setBatchName("");
        setNotice(`批次“${resolved.name}”已登记，当前状态：${batchStateLabels[resolved.state]}`);
        setBusy(false); void refresh(); return;
      }
      retainDraft(resolved);
      await startDraft(resolved);
    } catch (cause) {
      setError(`批次“${draft.name}”未确认创建：${cause instanceof Error ? cause.message : "未知错误"}。再次点击将核对并复用同一批次标识。`);
      setBusy(false); void refresh();
    }
  }

  async function action(batch: BatchRun, value: "pause" | "resume" | "cancel") {
    if (value === "cancel" && !window.confirm(`取消整个批次“${batch.name}”中的 ${batch.items.length} 个项目？已提交任务需等待执行端确认停止。`)) return;
    setBusy(true); setError(null);
    try {
      await transitionBatch(batch.batch_id, value);
      setNotice(value === "pause" ? `已暂停“${batch.name}”的新派发；已提交任务继续收尾` : value === "resume" ? `已恢复“${batch.name}”的派发；失败项需单独重试` : `已请求取消“${batch.name}”；执行状态以各任务确认为准`);
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "批次操作失败"); }
    finally { setBusy(false); }
  }

  async function cancelMember(batch: BatchRun, projectId: string) {
    if (!window.confirm(`仅取消批次“${batch.name}”内项目“${allProjects.find(([id]) => id === projectId)?.[1] ?? projectId}”的任务？其他项目继续运行。`)) return;
    setBusy(true); setError(null);
    try {
      await cancelBatchProject(batch.batch_id, projectId);
      setNotice("已请求取消该项目的批次任务；等待执行端确认，其他项目继续运行");
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "取消批次项目失败"); }
    finally { setBusy(false); }
  }

  async function pauseMember(batch: BatchRun, projectId: string, paused: boolean) {
    setBusy(true); setError(null);
    try {
      await pauseBatchProject(batch.batch_id, projectId, paused);
      setNotice(paused ? "已暂停该批次成员的新派发；已提交任务继续收尾" : "已恢复该批次成员的派发；失败任务不会自动重试");
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "成员派发操作失败"); }
    finally { setBusy(false); }
  }

  async function retryFailed(batch: BatchRun, memberTasks: TaskSpec[]) {
    const selected = memberTasks.filter((task) => task.state === "failed" && canTaskAction(task, "retry"));
    if (!selected.length || !window.confirm(`重试批次“${batch.name}”内 ${selected.length} 个可重试的失败任务？成功及用户取消项不受影响。`)) return;
    setBusy(true); setError(null);
    try { setNotice(describeCommandReceipt(await taskCommands.run({ scope: "batch", scope_id: batch.batch_id }, "retry", selected))); await refresh(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "重试失败项未完成"); }
    finally { setBusy(false); }
  }

  return <section className="workspace">
    <div className="section-heading"><div><h2>批量工作台</h2><span>项目级排队、优先级、暂停与失败隔离</span></div><button className="icon-button" title="刷新批量状态" disabled={refreshing} onClick={() => void refresh(true)}><RefreshCw className={refreshing ? "spin" : ""} size={18} /></button></div>
    {offline && <div className="error-banner inline-banner" role="status">连接中断，保留最后状态（{refreshedAt?.toLocaleString() ?? "尚未同步"}）；正在自动重连。</div>}
    {error && <div className="error-banner inline-banner"><CircleAlert size={17} />{error}</div>}
    {notice && <div className="batch-notice">{notice}</div>}
    {pendingDraft && <div className="batch-notice" role="status">保留批次“{pendingDraft.name}”；再次启动会复用同一批次，不会新建重复记录。</div>}
    <div className="batch-summary">
      <div><span>活动批次</span><strong>{batches.filter((batch) => activeBatchStates.has(batch.state)).length}</strong></div>
      <div><span>运行项目</span><strong>{activeProjectIds.size}</strong></div>
      <div><span>排队/执行/恢复任务</span><strong>{tasks.filter((task) => activeExecutionStates.has(task.state)).length}</strong></div>
      <div><span>失败任务</span><strong>{tasks.filter((task) => task.state === "failed").length}</strong></div>
    </div>
    <div className="batch-toolbar batch-default-settings"><strong>默认设置</strong><label>图片工作流<select value={defaults.imageWorkflowId ?? ""} onChange={(event) => updateDefaults({ imageWorkflowId: event.target.value || null })}><option value="">项目当前设置</option>{imageWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(defaults.seedvrEnabled)} onChange={(event) => updateDefaults({ seedvrEnabled: event.target.checked })} />启用超分</label><label>放大倍数<UpscaleFactorInput value={defaults.seedvrUpscaleFactor} onChange={(value) => updateDefaults({ seedvrUpscaleFactor: value })} /></label><label>超分工作流<select value={defaults.seedvrWorkflowId ?? ""} onChange={(event) => updateDefaults({ seedvrWorkflowId: event.target.value || null })}><option value="">项目当前设置</option>{restorationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(defaults.rifeEnabled)} onChange={(event) => updateDefaults({ rifeEnabled: event.target.checked })} />启用补帧</label><label>目标帧率<select value={defaults.rifeTargetFps ?? 48} onChange={(event) => updateDefaults({ rifeTargetFps: Number(event.target.value) as 48 | 60 | 120 })}><option value={48}>48 fps</option><option value={60}>60 fps</option><option value={120}>120 fps</option></select></label><label>补帧工作流<select value={defaults.rifeWorkflowId ?? ""} onChange={(event) => updateDefaults({ rifeWorkflowId: event.target.value || null })}><option value="">项目当前设置</option>{interpolationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label></div>
    <div className="batch-layout">
      <div className="batch-selector">
        <div className="batch-toolbar batch-create-toolbar"><label className="batch-name"><span>批次名称</span><input value={batchName} maxLength={200} placeholder={`${selected.size || "所选"}个项目批次`} onChange={(event) => setBatchName(event.target.value)} /></label><button className="primary-button" disabled={busy || offline || (!selected.size && !pendingDraft)} onClick={() => void createAndStart()}><Play size={16} />{pendingDraft ? `继续启动“${pendingDraft.name}”` : `启动 ${selected.size || ""} 个项目`}</button></div>
        <div className="batch-list-tools"><label><Search size={14} /><input value={query} placeholder="搜索项目" onChange={(event) => setQuery(event.target.value)} /></label><button className="secondary-button" disabled={!eligibleProjectIds.length} onClick={toggleAllEligible}>{eligibleProjectIds.every((projectId) => selected.has(projectId)) && eligibleProjectIds.length ? "清除可见选择" : "选择全部可加入项目"}</button></div>
        {visibleProjects.map(([projectId, name]) => {
          const projectTasks = tasksByProject.get(projectId) ?? [];
          const unfinished = projectTasks.filter((task) => !terminalStates.has(task.state));
          const selection = selected.get(projectId);
          const inActiveBatch = activeProjectIds.has(projectId);
          const canEnterBatch = (unfinished.length > 0 || outlineReady.has(projectId)) && !inActiveBatch;
          const counts = projectTasks.reduce<Record<string, number>>((result, task) => { result[task.state] = (result[task.state] ?? 0) + 1; return result; }, {});
          const status = inActiveBatch ? "已在活动批次中" : projectTasks.length ? `${unfinished.length} 个未完成 · ${counts.failed ?? 0} 个失败` : outlineReady.has(projectId) ? "大纲已批准 · 可自动规划" : "需先批准故事大纲";
          return <div className={`batch-project ${inActiveBatch ? "in-active-batch" : ""}`} key={projectId}>
            <header><label className="batch-project-choice"><input type="checkbox" checked={Boolean(selection)} disabled={!canEnterBatch} onChange={(event) => toggleProject(projectId, event.target.checked)} /><strong>{name}</strong></label><span>{status}</span></header>
            {selection && <div className="batch-project-controls"><label>执行范围<select value={selection.startBoundary} onChange={(event) => updateSelection(projectId, { startBoundary: event.target.value })}>{boundaryOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label><label>优先级<input type="number" min="-100" max="100" value={selection.priority} onChange={(event) => updateSelection(projectId, { priority: Number(event.target.value) })} /></label><label>图片工作流<select value={selection.settings.imageWorkflowId ?? ""} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, imageWorkflowId: event.target.value || null } })}><option value="">使用默认设置</option>{imageWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(selection.settings.seedvrEnabled)} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, seedvrEnabled: event.target.checked } })} />启用超分</label><label>放大倍数<UpscaleFactorInput value={selection.settings.seedvrUpscaleFactor} placeholder={`默认 ${defaults.seedvrUpscaleFactor ?? 2}`} onChange={(value) => updateSelection(projectId, { settings: { ...selection.settings, seedvrUpscaleFactor: value } })} /></label><label>超分工作流<select value={selection.settings.seedvrWorkflowId ?? ""} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, seedvrWorkflowId: event.target.value || null } })}><option value="">使用默认设置</option>{restorationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(selection.settings.rifeEnabled)} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, rifeEnabled: event.target.checked } })} />启用补帧</label><label>目标帧率<select value={selection.settings.rifeTargetFps ?? defaults.rifeTargetFps ?? 48} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, rifeTargetFps: Number(event.target.value) as 48 | 60 | 120 } })}><option value={48}>48 fps</option><option value={60}>60 fps</option><option value={120}>120 fps</option></select></label><label>补帧工作流<select value={selection.settings.rifeWorkflowId ?? ""} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, rifeWorkflowId: event.target.value || null } })}><option value="">使用默认设置</option>{interpolationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label></div>}
            {projectTasks.length > 0 && <div className="batch-status-strip">{Object.entries(counts).map(([state, count]) => <span key={state}><TaskStatus state={state as TaskState} /> {count}</span>)}</div>}
          </div>;
        })}
        {!visibleProjects.length && <div className="table-empty">没有匹配的项目</div>}
      </div>
      <div className="batch-history">
        <div className="batch-history-toolbar"><strong>批次</strong><div className="batch-filter"><ListFilter size={14} />{(["active", "history", "all"] as BatchFilter[]).map((value) => <button className={batchFilter === value ? "active" : ""} key={value} onClick={() => setBatchFilter(value)}>{value === "active" ? "活动" : value === "history" ? "历史" : "全部"}</button>)}</div></div>
        {visibleBatches.map((batch) => {
          const memberTasks = batch.items.flatMap((item) => item.task_ids.map((taskId) => tasks.find((task) => task.task_id === taskId)).filter((task): task is TaskSpec => Boolean(task)));
          const completed = memberTasks.filter((task) => terminalStates.has(task.state)).length;
          const progress = batchProgress(batch, tasks);
          const stageProgress = batchMemberProgress(batch.items.flatMap((item) => item.task_ids), tasks);
          const retryable = memberTasks.filter((task) => task.state === "failed" && canTaskAction(task, "retry"));
          return <div className="batch-run" key={batch.batch_id}>
            <div className="batch-run-heading"><div><strong>{batch.name}</strong><span>{batch.items.length} 个项目 · {completed}/{memberTasks.length} 项已结束</span></div><span className={`state state-${batch.state}`}>{batchStateLabels[batch.state]}</span>{batch.state === "draft" && <button className="secondary-button" disabled={busy || offline} onClick={() => void startDraft(batch)}><Play size={14} />启动草稿</button>}{batch.state === "running" && <button className="icon-button" disabled={busy || offline} title="暂停批次" onClick={() => void action(batch, "pause")}><Pause size={16} /></button>}{batch.state === "paused" && <button className="icon-button" disabled={busy || offline} title="恢复批次" onClick={() => void action(batch, "resume")}><Play size={16} /></button>}{activeBatchStates.has(batch.state) && <button className="icon-button" disabled={busy || offline} title="取消整个批次" onClick={() => void action(batch, "cancel")}><Square size={15} /></button>}</div>
            <div className="batch-outcomes"><span>{progress.label}</span><span>成功 {progress.counts.succeeded ?? 0} · 失败 {progress.counts.failed ?? 0} · 取消 {progress.counts.cancelled ?? 0} · 失效 {progress.counts.stale ?? 0} · 需要处理 {progress.counts.needs_attention ?? 0}</span></div>
            <div className="batch-phase">当前环节：{stageProgress.phase}</div>
            <div className="batch-stage-counts">{stageProgress.stages.map((stage) => <span key={stage.key}><strong>{stage.label}</strong> {stage.succeeded}/{stage.registered} {stage.totalKnown ? "已完成" : "已登记"}</span>)}</div>
            {progress.percent !== null && <div className="batch-progress" aria-label="任务已结束比例"><i style={{ width: `${progress.percent}%` }} /><span>{progress.percent}% 已结束</span></div>}
            {retryable.length > 0 && <button className="secondary-button" disabled={busy || offline || retryable.length > 500} onClick={() => void retryFailed(batch, memberTasks)}>重试批次失败项 ({retryable.length})</button>}
            <div className="batch-members">{batch.items.map((item) => {
              const itemTasks = item.task_ids.map((taskId) => tasks.find((task) => task.task_id === taskId)).filter((task): task is TaskSpec => Boolean(task));
              const itemCompleted = itemTasks.filter((task) => terminalStates.has(task.state)).length;
              const failed = itemTasks.filter((task) => task.state === "failed");
              const running = itemTasks.filter((task) => activeExecutionStates.has(task.state)).length;
              const waiting = itemTasks.filter((task) => ["blocked", "ready", "paused", "needs_review", "needs_attention"].includes(task.state)).length;
              const latestError = failed.find((task) => task.error_message)?.error_message;
              const active = itemTasks.find((task) => !terminalStates.has(task.state));
              const itemProgress = batchMemberProgress(item.task_ids, tasks);
              return <div className="batch-member" key={item.project_id}><div><strong>{allProjects.find(([projectId]) => projectId === item.project_id)?.[1] ?? item.project_id}{item.paused ? " · 已暂停派发" : ""}</strong><span>{itemCompleted}/{itemTasks.length} 已结束 · {running} 执行/排队 · {waiting} 等待 · {failed.length} 失败</span><span className="batch-member-phase">当前环节：{itemProgress.phase}</span><div className="batch-stage-counts">{itemProgress.stages.map((stage) => <span key={stage.key}><strong>{stage.label}</strong> {stage.succeeded}/{stage.registered} {stage.totalKnown ? "已完成" : "已登记"}</span>)}</div>{active && <small>{taskExplanation(active, tasks)}</small>}{latestError && <details><summary>错误详情</summary><small>{latestError}</small></details>}<em>{boundaryOptions.find((option) => option.value === item.start_boundary)?.label ?? item.start_boundary} · 优先级 {item.priority}</em></div>{["running", "paused"].includes(batch.state) && <div className="task-control-buttons"><button className="secondary-button" disabled={busy || offline} onClick={() => void pauseMember(batch, item.project_id, !item.paused)}>{item.paused ? <Play size={13} /> : <Pause size={13} />}{item.paused ? "恢复成员派发" : "暂停成员派发"}</button>{active && <button className="secondary-button" disabled={busy || offline} onClick={() => void cancelMember(batch, item.project_id)}><Square size={13} />取消批次成员</button>}</div>}</div>;
            })}</div>
          </div>;
        })}
        {!visibleBatches.length && <div className="table-empty">{batchFilter === "active" ? "当前没有活动批次" : "没有批次记录"}</div>}
      </div>
    </div>
  </section>;
}
