import { CircleAlert, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { listTasks } from "./api";
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
  const refresh = useCallback(async () => {
    try { setTasks(await listTasks(project.projectId)); setRefreshedAt(new Date()); setError(null); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "无法读取任务"); }
  }, [project.projectId]);
  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 5000); return () => window.clearInterval(timer); }, [refresh]);

  const visible = reviewOnly ? tasks.filter((task) => task.state === "needs_review") : tasks;
  return <section className="workspace">
    <div className="section-heading">
      <div><h2>{reviewOnly ? "待审核产物" : "任务进度"}</h2><span>{visible.length} 项 · 只读监控 · {refreshedAt ? `${refreshedAt.toLocaleTimeString()} 已刷新` : "正在加载"}</span></div>
      <button className="icon-button" title="刷新" onClick={() => void refresh()}><RefreshCw size={18} /></button>
    </div>
    {error && <div className="error-banner inline-banner"><CircleAlert size={17} />{error}</div>}
    <div className="task-table">
      <div className="task-header"><span>任务与等待原因</span><span>类型</span><span>状态</span><span>尝试</span><span>执行位置</span></div>
      {visible.map((task) => <div className="task-row" key={task.task_id}>
        <div><strong>{task.task_id.split(":").slice(-2).join(" · ")}</strong><small>{task.state === "blocked" ? `等待 ${task.depends_on.length} 个前置任务` : task.comfyui_prompt_id ? `ComfyUI ${task.comfyui_prompt_id}` : task.affinity_key ? `模型亲和：${task.affinity_key}` : task.execution_target === "local" ? "本地控制平面" : task.worker_id}</small></div>
        <span>{labels[task.kind]}</span><span className={`state state-${task.state}`}>{stateLabels[task.state]}</span><span>{task.attempt}/{task.max_attempts}</span>
        <span>{project.executionMode === "batch" ? "批量工作台" : task.state === "needs_review" ? "流程 · 审核" : "项目流程"}</span>
        {task.error_message && <div className="task-detail">{task.error_message}</div>}
      </div>)}
      {!visible.length && <div className="table-empty">{reviewOnly ? "当前没有待审核产物" : "当前项目还没有任务"}</div>}
    </div>
  </section>;
}
