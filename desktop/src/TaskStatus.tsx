import { useState } from "react";
import { getTaskEvents } from "./api";
import { canTaskAction, taskActionLabels, taskStateLabels } from "./task-status";
import type { TaskCommandAction, TaskEvent, TaskSpec, TaskState } from "./types";

export function TaskStatus({ state, label }: { state: TaskState; label?: string }) {
  return <span className={`state state-${state}`}>{label ?? taskStateLabels[state]}</span>;
}

export function TaskActionButtons({ task, disabled = false, onAction }: {
  task: TaskSpec; disabled?: boolean; onAction: (action: TaskCommandAction, confirmed: boolean) => void;
}) {
  return <div className="task-control-buttons">{(Object.keys(taskActionLabels) as TaskCommandAction[]).filter((action) => canTaskAction(task, action)).map((action) => <button
    key={action} className={`secondary-button ${action === "confirm_retry" ? "danger-button" : ""}`}
    disabled={disabled} aria-label={`${taskActionLabels[action]} ${task.task_id}`}
    onClick={() => {
      if (action === "restart" && !window.confirm(`任务 ${task.task_id} 已停止。请先修复错误详情中的配置或输入问题。将新建一次编排执行，复用已有内容，保留原失败记录；后续可能产生生成费用。继续？`)) return;
      if (action === "confirm_retry" && !window.confirm(`任务 ${task.task_id} 的外部执行结果不明确。重新执行可能重复生成并消耗 GPU 或付费资源。建议先“重新对账”。确定承担重复执行风险并重新执行？`)) return;
      onAction(action, action === "confirm_retry");
    }}
  >{taskActionLabels[action]}</button>)}</div>;
}

export function TaskDiagnostics({ task }: { task: TaskSpec }) {
  const [events, setEvents] = useState<TaskEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  async function loadEvents() {
    setLoading(true);
    try { setEvents(await getTaskEvents(task.task_id)); setError(null); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "无法读取事件"); }
    finally { setLoading(false); }
  }
  return <details className="task-diagnostics"><summary>{task.error_message ? `错误：${task.error_message.split("\n")[0].slice(0, 180)}` : "任务诊断与事件"}</summary>
    <div className="task-diagnostic-actions"><button className="secondary-button" disabled={loading} onClick={() => void loadEvents()}>{loading ? "读取中" : "读取最近事件"}</button><button className="secondary-button" onClick={() => void navigator.clipboard.writeText(JSON.stringify({ task, events }, null, 2)).then(() => setCopied(true)).catch(() => setError("无法复制，请手动选择下方诊断文本"))}>{copied ? "已复制" : "复制诊断"}</button></div>
    {error && <p role="alert">{error}</p>}
    <pre>{JSON.stringify({ task_id: task.task_id, phase: task.current_phase, state: task.state, attempt: task.attempt, max_attempts: task.max_attempts, attempt_id: task.attempt_id, deadline_at: task.deadline_at, error_code: task.error_code, error_message: task.error_message, blocked_reason: task.blocked_reason, depends_on: task.depends_on, comfyui_prompt_id: task.comfyui_prompt_id, events }, null, 2)}</pre>
  </details>;
}
