import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, Pause, Play, RefreshCw, RotateCcw, Square, Trash2 } from "lucide-react";
import { clearProjectTasks, getProjectRunState, getSchedulerHealth, listBatches, listTasks, pauseBatchProject, setProjectPaused, taskCommands } from "./api";
import type { BackendProjectSpec } from "./api";
import type { BatchRun, ProjectDraft, SchedulerHealth, TaskCommandAction, TaskSpec } from "./types";
import { TaskActionButtons, TaskDiagnostics, TaskStatus } from "./TaskStatus";
import { canTaskAction, filterTasks, formatActivity, taskExplanation, taskKindLabels, taskPhaseLabel, taskStateLabels, terminalTaskStates } from "./task-status";
import { describeCommandReceipt } from "./task-commands";

type Props = { project: ProjectDraft; projects?: BackendProjectSpec[] };

export function TasksView({ project, projects = [] }: Props) {
  const [tasks, setTasks] = useState<TaskSpec[]>([]);
  const [batches, setBatches] = useState<BatchRun[]>([]);
  const [health, setHealth] = useState<SchedulerHealth | null>(null);
  const [scopeProjectId, setScopeProjectId] = useState(project.projectId);
  const [batchId, setBatchId] = useState("");
  const [kind, setKind] = useState("");
  const [state, setState] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const [auxiliaryErrors, setAuxiliaryErrors] = useState<Record<string, string>>({});
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null);
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState<boolean | null>(null);
  const activeScope = useRef(scopeProjectId);
  const refreshSequence = useRef(0);
  activeScope.current = scopeProjectId;
  const projectNames = useMemo(() => new Map([...projects.map((item) => [item.project_id, item.name] as const), [project.projectId, project.name]]), [projects, project.projectId, project.name]);
  const scopeName = projectNames.get(scopeProjectId) ?? scopeProjectId;
  useEffect(() => { setScopeProjectId(project.projectId); setBatchId(""); }, [project.projectId]);
  const refresh = useCallback(async () => {
    const sequence = ++refreshSequence.current;
    const current = () => activeScope.current === scopeProjectId && sequence === refreshSequence.current;
    const auxiliary = async <T,>(name: string, read: Promise<T>, apply: (value: T) => void) => {
      try {
        const value = await read;
        if (!current()) return;
        apply(value);
        setAuxiliaryErrors((errors) => { const next = { ...errors }; delete next[name]; return next; });
      } catch {
        if (current()) setAuxiliaryErrors((errors) => ({ ...errors, [name]: `${name}暂时无法更新，已保留最后状态` }));
      }
    };
    await Promise.allSettled([
      listTasks().then((nextTasks) => {
        if (!current()) return;
        setTasks(nextTasks); setRefreshedAt(new Date()); setOffline(false);
      }).catch(() => { if (current()) setOffline(true); }),
      auxiliary("批次", listBatches(), setBatches),
      auxiliary("调度健康", getSchedulerHealth(), setHealth),
      auxiliary("项目派发状态", getProjectRunState(scopeProjectId), (value) => setPaused(value.paused)),
    ]);
  }, [scopeProjectId]);
  useEffect(() => {
    let stopped = false;
    let timer: number;
    setPaused(null);
    const poll = async () => { await refresh(); if (!stopped) timer = window.setTimeout(() => void poll(), 3000); };
    void poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [refresh]);
  const selectedBatch = batches.find((batch) => batch.batch_id === batchId);
  const dispatchPaused = selectedBatch ? Boolean(selectedBatch.items.find((item) => item.project_id === scopeProjectId)?.paused) : paused;
  const projectTasks = tasks.filter((task) => task.project_id === scopeProjectId);
  const visible = filterTasks(tasks, { projectId: scopeProjectId, batch: selectedBatch, kind, state });
  const retryable = visible.filter((task) => task.state === "failed" && canTaskAction(task, "retry"));
  const cancellable = visible.filter((task) => canTaskAction(task, "cancel"));
  const scope = selectedBatch ? { scope: "batch" as const, scope_id: selectedBatch.batch_id } : { scope: "project" as const, scope_id: scopeProjectId };
  const scopeLabel = selectedBatch ? `批次“${selectedBatch.name}”中的项目“${scopeName}”` : `项目“${scopeName}”`;

  async function command(action: TaskCommandAction, selected: TaskSpec[], confirmed = false) {
    if (busy) return;
    if (selected.length > 1 && !window.confirm(`${action === "cancel" ? "取消任务" : "重试失败项"}：${scopeLabel}，当前筛选中的 ${selected.length} 个任务？`)) return;
    setBusy(true); setError(null); setNotice(null);
    try { setNotice(describeCommandReceipt(await taskCommands.run(scope, action, selected, confirmed))); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "命令未完成，请刷新状态"); }
    finally { setBusy(false); void refresh(); }
  }
  async function togglePaused() {
    if (dispatchPaused === null) return;
    setBusy(true); setError(null);
    try {
      if (selectedBatch) await pauseBatchProject(selectedBatch.batch_id, scopeProjectId, !dispatchPaused);
      else {
        const result = await setProjectPaused(scopeProjectId, !dispatchPaused);
        setPaused(result.paused);
      }
      setNotice(!dispatchPaused ? `已暂停${scopeLabel}的新派发；已经提交的任务继续观察和收尾` : `已恢复${scopeLabel}的派发；失败任务需单独重试，上层暂停仍然有效`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "暂停操作未完成"); }
    finally { setBusy(false); void refresh(); }
  }
  async function clearHistory() {
    if (!window.confirm(`清除项目“${scopeName}”的任务、批次、审核记录和生成版本索引？媒体文件和项目内容保留。此操作不能用于恢复失败任务，清除后无法在此查看历史诊断。`)) return;
    setBusy(true); setError(null);
    try { const result = await clearProjectTasks(scopeProjectId); setNotice(`已清除 ${result.removed_tasks} 条任务历史；媒体文件保留`); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "无法清除历史"); }
    finally { setBusy(false); void refresh(); }
  }

  return <section className="workspace">
    <div className="section-heading"><div><h2>任务管控</h2><span>{scopeLabel} · {visible.length} 项 · 最后同步 {refreshedAt?.toLocaleTimeString() ?? "尚未成功"}</span></div><button className="icon-button" title="刷新任务状态" onClick={() => void refresh()}><RefreshCw size={18} /></button></div>
    {offline && <div className="error-banner inline-banner" role="status"><CircleAlert size={17} />连接中断，保留最后已知状态（{refreshedAt?.toLocaleString() ?? "尚未取得数据"}）；正在自动重连。</div>}
    {Object.keys(auxiliaryErrors).length > 0 && <div className="batch-notice" role="status">{Object.values(auxiliaryErrors).join("；")}。任务控制仍可使用。</div>}
    {error && <div className="error-banner inline-banner" role="alert"><CircleAlert size={17} />{error}</div>}
    {notice && <div className="batch-notice" role="status">{notice}</div>}
    <div className="task-filters">
      <label>项目<select aria-label="筛选项目" value={scopeProjectId} onChange={(event) => { setScopeProjectId(event.target.value); setBatchId(""); setNotice(null); }} >{[...projectNames].map(([id, name]) => <option key={id} value={id}>{name}{id === project.projectId ? "（当前项目）" : ""}</option>)}</select></label>
      <label>批次<select aria-label="筛选批次" value={batchId} onChange={(event) => setBatchId(event.target.value)}><option value="">本项目全部批次</option>{batches.filter((batch) => batch.items.some((item) => item.project_id === scopeProjectId)).map((batch) => <option key={batch.batch_id} value={batch.batch_id}>{batch.name}</option>)}</select></label>
      <label>类型<select aria-label="筛选任务类型" value={kind} onChange={(event) => setKind(event.target.value)}><option value="">全部类型</option>{Object.entries(taskKindLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label>状态<select aria-label="筛选任务状态" value={state} onChange={(event) => setState(event.target.value)}><option value="">全部状态</option>{Object.entries(taskStateLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
    </div>
    <div className="task-toolbar"><button className="secondary-button" disabled={busy || offline || dispatchPaused === null || Boolean(auxiliaryErrors[selectedBatch ? "批次" : "项目派发状态"])} onClick={() => void togglePaused()}>{dispatchPaused ? <Play size={14} /> : <Pause size={14} />}{dispatchPaused ? "恢复" : "暂停"}{selectedBatch ? "批次成员" : "项目"}派发</button>
      <button className="secondary-button" disabled={busy || offline || !retryable.length || retryable.length > 500} onClick={() => void command("retry", retryable)}><RotateCcw size={14} />重试筛选内失败项 ({retryable.length})</button>
      <button className="secondary-button danger-button" disabled={busy || offline || !cancellable.length || cancellable.length > 500} onClick={() => void command("cancel", cancellable)}><Square size={14} />取消筛选内任务 ({cancellable.length})</button>
      <span className="task-scheduler-status">{health ? health.owns_dispatcher ? "调度器已取得派发权" : "当前服务未取得派发权" : "正在读取调度状态"}{health?.last_error ? ` · ${health.last_error}` : ""}</span>
    </div>
    <div className="task-table task-management-table"><div className="task-header"><span>任务与等待原因</span><span>类型</span><span>状态</span><span>尝试</span><span>操作</span></div>
      {visible.map((task) => <div className="task-row" key={task.task_id} data-task-id={task.task_id}>
        <div><strong>{task.task_id}</strong><small>{taskExplanation(task, tasks)}</small><small>最后活动：{formatActivity(task.last_activity_at ?? task.updated_at)}{task.current_phase ? ` · ${taskPhaseLabel(task.current_phase)}` : ""}</small></div>
        <span>{taskKindLabels[task.kind]}</span><TaskStatus state={task.state} /><span>{task.attempt}/{task.max_attempts}</span>
        <TaskActionButtons task={task} disabled={busy || offline} onAction={(action, confirmed) => void command(action, [task], confirmed)} />
        <TaskDiagnostics task={task} />
      </div>)}
      {!visible.length && <div className="table-empty">{refreshedAt ? "当前项目和筛选条件下没有任务" : "正在读取任务"}</div>}
    </div>
    <details className="task-history-controls"><summary>历史记录管理</summary><p>清除记录和取消执行分别处理。仅在项目所有任务已结束后清除；媒体文件保留，任务与版本索引会删除。</p><button className="secondary-button danger-button" disabled={busy || offline || !projectTasks.length || projectTasks.some((task) => !terminalTaskStates.has(task.state))} onClick={() => void clearHistory()}><Trash2 size={14} />清除“{scopeName}”的历史记录</button></details>
  </section>;
}
