import { useCallback, useEffect, useState } from "react";
import { CircleAlert, RefreshCw, Square, Trash2 } from "lucide-react";
import { cancelTask, clearProjectTasks, listTasks } from "./api";
import type { ProjectDraft, TaskKind, TaskSpec } from "./types";

const labels: Record<TaskKind, string> = {
  llm_planning: "LLM 规划",
  image_generation: "图片生成",
  conditioning_encoding: "条件编码",
  h3_generation: "H3 生成",
  ai_review: "AI 审核",
  model_switch: "模型切换",
  seedvr2: "SeedVR2",
  rife: "RIFE",
  whisper: "Whisper",
  master_assembly: "母版拼接",
  export: "导出",
  asset_transfer: "素材传输",
};
const stateLabels: Record<TaskSpec["state"], string> = {
  blocked: "等待依赖", ready: "可运行", queued: "已排队", paused: "已暂停",
  running: "运行中", needs_review: "等待人工审核", succeeded: "已完成",
  failed: "失败", cancelled: "已取消", stale: "已失效",
};

type Props = { project: ProjectDraft; reviewOnly?: boolean };

export function TasksView({ project, reviewOnly = false }: Props) {
  const [tasks, setTasks] = useState<TaskSpec[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null);
  const [clearBusy, setClearBusy] = useState(false);
  const [cancelBusy, setCancelBusy] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    try { setTasks(await listTasks(project.projectId)); setRefreshedAt(new Date()); setError(null); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "无法读取任务"); }
  }, [project.projectId]);
  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 5000); return () => window.clearInterval(timer); }, [refresh]);

  const clearTasks = useCallback(async () => {
    if (clearBusy) return;
    if (!window.confirm(`清除项目“${project.name}”的全部任务、批次、审核记录和生成版本？项目内容、提示词、素材、工作流及媒体文件会保留。`)) return;
    setClearBusy(true);
    setError(null);
    try {
      const result = await clearProjectTasks(project.projectId);
      window.alert(`已清除 ${result.removed_tasks} 个任务；项目内容和媒体文件已保留。`);
      await refresh();
    } catch (cause) {
      setError(`清除任务失败：${cause instanceof Error ? cause.message : "未知错误"}`);
    } finally {
      setClearBusy(false);
    }
  }, [clearBusy, project.name, project.projectId, refresh]);

  const cancel = useCallback(async (task: TaskSpec) => {
    if (cancelBusy) return;
    setCancelBusy(task.task_id);
    setError(null);
    try {
      await cancelTask(task.task_id);
      await refresh();
    } catch (cause) {
      setError(`取消任务失败：${cause instanceof Error ? cause.message : "未知错误"}`);
    } finally {
      setCancelBusy(null);
    }
  }, [cancelBusy, refresh]);

  const cancelActive = useCallback(async () => {
    const active = tasks.filter((task) => ["queued", "running"].includes(task.state));
    if (!active.length || cancelBusy) return;
    if (!window.confirm(`取消当前项目的 ${active.length} 个排队/运行任务？已完成任务不会受影响。`)) return;
    setCancelBusy("all");
    setError(null);
    try {
      await Promise.all(active.map((task) => cancelTask(task.task_id)));
      await refresh();
    } catch (cause) {
      setError(`批量取消失败：${cause instanceof Error ? cause.message : "未知错误"}`);
      await refresh();
    } finally {
      setCancelBusy(null);
    }
  }, [cancelBusy, refresh, tasks]);

  const visible = reviewOnly ? tasks.filter((task) => task.state === "needs_review") : tasks;
  return <section className="workspace">
    <div className="section-heading">
      <div><h2>{reviewOnly ? "待审核产物" : "任务进度"}</h2><span>{visible.length} 项 · 只读监控 · {refreshedAt ? `${refreshedAt.toLocaleTimeString()} 已刷新` : "正在加载"}</span></div>
      <div className="section-heading-actions">
        {!reviewOnly && <button className="secondary-button danger-button" disabled={cancelBusy !== null || !tasks.some((task) => ["queued", "running"].includes(task.state))} onClick={() => void cancelActive()}><Square size={14} />取消运行任务</button>}
        {!reviewOnly && <button className="secondary-button danger-button" disabled={clearBusy} onClick={() => void clearTasks()}><Trash2 size={16} />{clearBusy ? "正在清除" : "清除所有任务"}</button>}
        <button className="icon-button" title="刷新" onClick={() => void refresh()}><RefreshCw size={18} /></button>
      </div>
    </div>
    {error && <div className="error-banner inline-banner"><CircleAlert size={17} />{error}</div>}
    <div className="task-table">
      <div className="task-header"><span>任务与等待原因</span><span>类型</span><span>状态</span><span>尝试</span><span>执行位置</span></div>
      {visible.map((task) => <div className="task-row" key={task.task_id}>
        <div><strong>{task.task_id.split(":").slice(-2).join(" · ")}</strong><small>{task.state === "blocked" ? `等待 ${task.depends_on.length} 个前置任务` : task.comfyui_prompt_id ? `ComfyUI ${task.comfyui_prompt_id}` : task.affinity_key ? `模型亲和：${task.affinity_key}` : task.execution_target === "local" ? "本地控制平面" : task.worker_id}</small></div>
        <span>{labels[task.kind]}</span><span className={`state state-${task.state}`}>{stateLabels[task.state]}</span><span>{task.attempt}/{task.max_attempts}</span>
        <span className="task-row-actions">{project.executionMode === "batch" ? "批量工作台" : task.state === "needs_review" ? "流程 · 审核" : "项目流程"}{["queued", "running"].includes(task.state) && <button className="icon-button" title="取消任务" aria-label="取消任务" disabled={cancelBusy !== null} onClick={() => void cancel(task)}><Square size={13} /></button>}</span>
        {task.error_message && <div className="task-detail">{task.error_message}</div>}
      </div>)}
      {!visible.length && <div className="table-empty">{reviewOnly ? "当前没有待审核产物" : "当前项目还没有任务"}</div>}
    </div>
  </section>;
}
