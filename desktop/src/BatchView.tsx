import { CircleAlert, ListFilter, Pause, Play, RefreshCw, Search, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { cancelBatchProject, createBatch, getProjectRunState, listBatches, listTasks, transitionBatch } from "./api";
import type { BackendProjectSpec } from "./api";
import type { BatchRun, BatchSettings, ProjectDraft, TaskSpec } from "./types";

type WorkflowChoice = { id: string; name: string; revision: number; kind?: string };
type Props = { project: ProjectDraft; projects: BackendProjectSpec[]; workflows?: WorkflowChoice[] };
type Selection = { startBoundary: string; priority: number; settings: BatchSettings };
type BatchFilter = "active" | "history" | "all";

const boundaryOptions = [
  { value: "next_ready", label: "从当前未完成任务继续" },
  { value: "generation", label: "生成、审核建议与交付" },
  { value: "delivery", label: "仅后处理与交付" },
];
const terminalStates = new Set(["succeeded", "failed", "cancelled", "stale"]);
const activeBatchStates = new Set(["draft", "running", "paused"]);
const batchStateLabels: Record<BatchRun["state"], string> = {
  draft: "待启动", running: "运行中", paused: "已暂停", completed: "已结束", cancelled: "已取消",
};
const taskStateLabels: Record<string, string> = {
  blocked: "等待依赖", ready: "可执行", queued: "排队", running: "运行中", paused: "暂停",
  needs_review: "待审核", succeeded: "成功", failed: "失败", cancelled: "取消", stale: "已失效",
};

export function BatchView({ project, projects, workflows = [] }: Props) {
  const [tasks, setTasks] = useState<TaskSpec[]>([]);
  const [batches, setBatches] = useState<BatchRun[]>([]);
  const [selected, setSelected] = useState<Map<string, Selection>>(new Map());
  const [batchName, setBatchName] = useState("");
  const [query, setQuery] = useState("");
  const [batchFilter, setBatchFilter] = useState<BatchFilter>("active");
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [outlineReady, setOutlineReady] = useState<Set<string>>(new Set());
  const [defaults, setDefaults] = useState<BatchSettings>({ imageWorkflowId: null, seedvrEnabled: false, seedvrWorkflowId: null, seedvrUpscaleFactor: 2, rifeEnabled: false, rifeWorkflowId: null, rifeTargetFps: 48 });

  const allProjects = useMemo(() => {
    const values = new Map(projects.map((item) => [item.project_id, item.name]));
    values.set(project.projectId, project.name);
    return [...values.entries()];
  }, [project.name, project.projectId, projects]);

  const refresh = useCallback(async (showBusy = false) => {
    if (showBusy) setRefreshing(true);
    try {
      const [nextTasks, nextBatches, runStates] = await Promise.all([
        listTasks(), listBatches(),
        Promise.allSettled(allProjects.map(([projectId]) => getProjectRunState(projectId))),
      ]);
      setTasks(nextTasks);
      setBatches(nextBatches);
      setOutlineReady(new Set(runStates.flatMap((result) => result.status === "fulfilled" && result.value.outline_approved ? [result.value.project_id] : [])));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法加载批量工作台");
    } finally {
      if (showBusy) setRefreshing(false);
    }
  }, [allProjects]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
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

  async function createAndStart() {
    setBusy(true); setError(null); setNotice(null);
    try {
      const now = new Date().toISOString();
      const name = batchName.trim() || `${selected.size} 个项目批次`;
      const batch: BatchRun = {
        schema_version: "1.0", batch_id: crypto.randomUUID(), name, state: "draft",
        settings: defaults,
        items: [...selected].map(([projectId, selection]) => ({ project_id: projectId, task_ids: [], start_boundary: selection.startBoundary, priority: selection.priority, settings: selection.settings })),
        created_at: now, updated_at: now,
      };
      const resolved = await createBatch(batch);
      await transitionBatch(resolved.batch_id, "start");
      setSelected(new Map()); setBatchName(""); setBatchFilter("active");
      setNotice(`批次“${name}”已启动`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "创建批次失败");
    } finally { setBusy(false); }
  }

  async function action(batch: BatchRun, value: "pause" | "resume" | "cancel") {
    setBusy(true); setError(null);
    try {
      await transitionBatch(batch.batch_id, value);
      setNotice(value === "pause" ? `已暂停“${batch.name}”` : value === "resume" ? `已恢复“${batch.name}”` : `已取消“${batch.name}”`);
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "批次操作失败"); }
    finally { setBusy(false); }
  }

  async function cancelMember(batch: BatchRun, projectId: string) {
    setBusy(true); setError(null);
    try {
      await cancelBatchProject(batch.batch_id, projectId);
      setNotice("已取消该项目的批量执行；其他项目继续运行");
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "取消批次项目失败"); }
    finally { setBusy(false); }
  }

  return <section className="workspace">
    <div className="section-heading"><div><h2>批量工作台</h2><span>项目级排队、优先级、暂停与失败隔离</span></div><button className="icon-button" title="刷新批量状态" disabled={refreshing} onClick={() => void refresh(true)}><RefreshCw className={refreshing ? "spin" : ""} size={18} /></button></div>
    {error && <div className="error-banner inline-banner"><CircleAlert size={17} />{error}</div>}
    {notice && <div className="batch-notice">{notice}</div>}
    <div className="batch-summary">
      <div><span>活动批次</span><strong>{batches.filter((batch) => activeBatchStates.has(batch.state)).length}</strong></div>
      <div><span>运行项目</span><strong>{activeProjectIds.size}</strong></div>
      <div><span>排队/运行任务</span><strong>{tasks.filter((task) => ["queued", "running"].includes(task.state)).length}</strong></div>
      <div><span>失败任务</span><strong>{tasks.filter((task) => task.state === "failed").length}</strong></div>
    </div>
    <div className="batch-toolbar batch-default-settings"><strong>默认设置</strong><label>图片工作流<select value={defaults.imageWorkflowId ?? ""} onChange={(event) => updateDefaults({ imageWorkflowId: event.target.value || null })}><option value="">项目当前设置</option>{imageWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(defaults.seedvrEnabled)} onChange={(event) => updateDefaults({ seedvrEnabled: event.target.checked })} />启用超分</label><label>放大倍数<select value={defaults.seedvrUpscaleFactor ?? 2} onChange={(event) => updateDefaults({ seedvrUpscaleFactor: Number(event.target.value) })}><option value={2}>2x</option><option value={4}>4x</option></select></label><label>超分工作流<select value={defaults.seedvrWorkflowId ?? ""} onChange={(event) => updateDefaults({ seedvrWorkflowId: event.target.value || null })}><option value="">项目当前设置</option>{restorationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(defaults.rifeEnabled)} onChange={(event) => updateDefaults({ rifeEnabled: event.target.checked })} />启用补帧</label><label>目标帧率<select value={defaults.rifeTargetFps ?? 48} onChange={(event) => updateDefaults({ rifeTargetFps: Number(event.target.value) as 48 | 60 | 120 })}><option value={48}>48 fps</option><option value={60}>60 fps</option><option value={120}>120 fps</option></select></label><label>补帧工作流<select value={defaults.rifeWorkflowId ?? ""} onChange={(event) => updateDefaults({ rifeWorkflowId: event.target.value || null })}><option value="">项目当前设置</option>{interpolationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label></div>
    <div className="batch-layout">
      <div className="batch-selector">
        <div className="batch-toolbar batch-create-toolbar"><label className="batch-name"><span>批次名称</span><input value={batchName} maxLength={200} placeholder={`${selected.size || "所选"}个项目批次`} onChange={(event) => setBatchName(event.target.value)} /></label><button className="primary-button" disabled={busy || !selected.size} onClick={() => void createAndStart()}><Play size={16} />启动 {selected.size || ""} 个项目</button></div>
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
            {selection && <div className="batch-project-controls"><label>执行范围<select value={selection.startBoundary} onChange={(event) => updateSelection(projectId, { startBoundary: event.target.value })}>{boundaryOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label><label>优先级<input type="number" min="-100" max="100" value={selection.priority} onChange={(event) => updateSelection(projectId, { priority: Number(event.target.value) })} /></label><label>图片工作流<select value={selection.settings.imageWorkflowId ?? ""} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, imageWorkflowId: event.target.value || null } })}><option value="">使用默认设置</option>{imageWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(selection.settings.seedvrEnabled)} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, seedvrEnabled: event.target.checked } })} />启用超分</label><label>放大倍数<select value={selection.settings.seedvrUpscaleFactor ?? defaults.seedvrUpscaleFactor ?? 2} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, seedvrUpscaleFactor: Number(event.target.value) } })}><option value={2}>2x</option><option value={4}>4x</option></select></label><label>超分工作流<select value={selection.settings.seedvrWorkflowId ?? ""} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, seedvrWorkflowId: event.target.value || null } })}><option value="">使用默认设置</option>{restorationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label><label><input type="checkbox" checked={Boolean(selection.settings.rifeEnabled)} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, rifeEnabled: event.target.checked } })} />启用补帧</label><label>目标帧率<select value={selection.settings.rifeTargetFps ?? defaults.rifeTargetFps ?? 48} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, rifeTargetFps: Number(event.target.value) as 48 | 60 | 120 } })}><option value={48}>48 fps</option><option value={60}>60 fps</option><option value={120}>120 fps</option></select></label><label>补帧工作流<select value={selection.settings.rifeWorkflowId ?? ""} onChange={(event) => updateSelection(projectId, { settings: { ...selection.settings, rifeWorkflowId: event.target.value || null } })}><option value="">使用默认设置</option>{interpolationWorkflows.map((item) => <option key={item.id} value={item.id}>{item.name} · R{item.revision}</option>)}</select></label></div>}
            {projectTasks.length > 0 && <div className="batch-status-strip">{Object.entries(counts).map(([state, count]) => <span className={`state state-${state}`} key={state}>{taskStateLabels[state] ?? state} {count}</span>)}</div>}
          </div>;
        })}
        {!visibleProjects.length && <div className="table-empty">没有匹配的项目</div>}
      </div>
      <div className="batch-history">
        <div className="batch-history-toolbar"><strong>批次</strong><div className="batch-filter"><ListFilter size={14} />{(["active", "history", "all"] as BatchFilter[]).map((value) => <button className={batchFilter === value ? "active" : ""} key={value} onClick={() => setBatchFilter(value)}>{value === "active" ? "活动" : value === "history" ? "历史" : "全部"}</button>)}</div></div>
        {visibleBatches.map((batch) => {
          const memberTasks = batch.items.flatMap((item) => item.task_ids.map((taskId) => tasks.find((task) => task.task_id === taskId)).filter((task): task is TaskSpec => Boolean(task)));
          const completed = memberTasks.filter((task) => terminalStates.has(task.state)).length;
          const percent = memberTasks.length ? Math.round(completed / memberTasks.length * 100) : 0;
          return <div className="batch-run" key={batch.batch_id}>
            <div className="batch-run-heading"><div><strong>{batch.name}</strong><span>{batch.items.length} 个项目 · {completed}/{memberTasks.length} 项已结束</span></div><span className={`state state-${batch.state}`}>{batchStateLabels[batch.state]}</span>{batch.state === "running" && <button className="icon-button" disabled={busy} title="暂停批次" onClick={() => void action(batch, "pause")}><Pause size={16} /></button>}{batch.state === "paused" && <button className="icon-button" disabled={busy} title="恢复批次" onClick={() => void action(batch, "resume")}><Play size={16} /></button>}{activeBatchStates.has(batch.state) && <button className="icon-button" disabled={busy} title="取消整个批次" onClick={() => void action(batch, "cancel")}><Square size={15} /></button>}</div>
            <div className="batch-progress"><i style={{ width: `${percent}%` }} /><span>{percent}%</span></div>
            <div className="batch-members">{batch.items.map((item) => {
              const itemTasks = item.task_ids.map((taskId) => tasks.find((task) => task.task_id === taskId)).filter((task): task is TaskSpec => Boolean(task));
              const itemCompleted = itemTasks.filter((task) => terminalStates.has(task.state)).length;
              const failed = itemTasks.filter((task) => task.state === "failed");
              const running = itemTasks.filter((task) => ["queued", "running"].includes(task.state)).length;
              const waiting = itemTasks.filter((task) => ["blocked", "ready", "paused"].includes(task.state)).length;
              const latestError = failed.find((task) => task.error_message)?.error_message;
              return <div className="batch-member" key={item.project_id}><div><strong>{allProjects.find(([projectId]) => projectId === item.project_id)?.[1] ?? item.project_id}</strong><span>{itemCompleted}/{itemTasks.length} 已结束 · {running} 运行/排队 · {waiting} 等待 · {failed.length} 失败</span>{latestError && <small>{latestError}</small>}<em>{boundaryOptions.find((option) => option.value === item.start_boundary)?.label ?? item.start_boundary} · 优先级 {item.priority}</em></div>{batch.state === "running" && itemTasks.some((task) => !terminalStates.has(task.state)) && <button className="secondary-button" disabled={busy} onClick={() => void cancelMember(batch, item.project_id)}><Square size={13} />取消项目</button>}</div>;
            })}</div>
          </div>;
        })}
        {!visibleBatches.length && <div className="table-empty">{batchFilter === "active" ? "当前没有活动批次" : "没有批次记录"}</div>}
      </div>
    </div>
  </section>;
}
