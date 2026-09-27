import type { BatchRun, TaskCommandAction, TaskKind, TaskSpec, TaskState } from "./types";

export const taskKindLabels: Record<TaskKind, string> = {
  llm_planning: "项目自动推进", image_generation: "图片生成", conditioning_encoding: "条件编码",
  h3_generation: "H3 生成", ai_review: "AI 审核", model_switch: "模型切换", seedvr2: "SeedVR2 超分",
  rife: "RIFE 补帧", whisper: "语音转写", master_assembly: "母版拼接", export: "导出", asset_transfer: "素材传输",
};
export const taskStateLabels: Record<TaskState, string> = {
  blocked: "等待依赖", ready: "等待派发", queued: "已排队", paused: "已暂停", running: "执行中",
  recovering: "恢复中", retry_wait: "等待重试", cancelling: "正在取消", needs_attention: "需要处理",
  needs_review: "等待人工审核", succeeded: "已成功", failed: "失败", cancelled: "已取消", stale: "已失效",
};
export const taskActionLabels: Record<TaskCommandAction, string> = {
  pause: "暂停派发", resume: "恢复派发", cancel: "取消任务", retry: "重试失败", reconcile: "重新对账", confirm_retry: "确认重新执行", restart: "重试编排（新执行）",
};
export const terminalTaskStates = new Set<TaskState>(["succeeded", "failed", "cancelled", "stale"]);
export const activeExecutionStates = new Set<TaskState>(["queued", "running", "recovering", "retry_wait", "cancelling"]);
const phaseLabels: Record<string, string> = {
  batch_orchestration_started: "检查已有内容与审核策略", batch_operation_started: "补齐当前缺失内容",
  batch_output_ready: "已校验并复用阶段产出", batch_image_child: "等待图片素材就绪",
  batch_dag_compiled: "已编译视频执行计划", batch_dag_registered: "视频执行任务已登记",
  batch_dag_compiling: "编译视频执行计划与工作流预检",
  storyboard: "补齐分镜", asset_plan: "补齐素材规划", claimed: "已领取，准备执行",
  submitting: "正在提交工作流", submitted: "工作流已提交", collecting: "正在收集产物",
};
export function taskPhaseLabel(phase?: string | null): string {
  if (!phase) return "";
  const [event, ...scopeParts] = phase.split(":");
  const scope = scopeParts.join(":");
  if (event.startsWith("batch_") && scope) {
    if (scope === "storyboard") return "分镜生成与校验";
    if (scope === "asset_plan") return "素材规划与校验";
    if (scope.startsWith("image:")) return `图片提示词/素材 ${scope.slice(6)}`;
    if (scope.startsWith("h3:")) return `视频提示词 ${scope.slice(3)}`;
  }
  if (phase.startsWith("image:")) return "补齐图片提示词与素材";
  if (phase.startsWith("h3:")) return "补齐视频提示词";
  return phaseLabels[phase] ?? "正在执行当前步骤（详情见诊断）";
}


export function canTaskAction(task: TaskSpec, action: TaskCommandAction): boolean {
  // Missing metadata is not permission. The server can deny a retry even in failed state.
  return task.available_actions?.includes(action) === true;
}

export function formatActivity(value?: string | null): string {
  if (!value) return "尚无活动记录";
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? "活动时间未知" : parsed.toLocaleString();
}

export function taskExplanation(task: TaskSpec, tasks: TaskSpec[] = []): string {
  if (task.blocked_reason) return task.blocked_reason;
  if (task.state === "blocked") {
    const dependencies = task.depends_on.map((id) => tasks.find((candidate) => candidate.task_id === id))
      .filter((candidate): candidate is TaskSpec => Boolean(candidate && candidate.state !== "succeeded"));
    if (dependencies.length) return dependencies.map((dep) => `等待${taskKindLabels[dep.kind]} ${dep.task_id}（${taskStateLabels[dep.state]}）`).join("；");
    return task.blocked_reason || "等待依赖校验；可展开任务诊断查看依赖标识";
  }
  if (task.state === "retry_wait") return task.next_retry_at ? `将在 ${formatActivity(task.next_retry_at)} 重试` : "等待后台确认下一次重试时间";
  if (task.state === "cancelling") return "已请求取消，等待执行端确认停止；资源仍保留";
  if (task.state === "recovering") return "正在核对外部执行结果并恢复观察";
  if (task.state === "needs_attention") return "执行结果不明确；先重新对账，再决定是否重新执行";
  if (task.state === "paused") return "暂停新派发；恢复派发后继续等待依赖";
  if (task.state === "needs_review") return "等待审核结论，请在制作流程查看对应片段";
  if (task.state === "stale") return "输入版本已变更；请在制作流程检查并重新编译";
  if (task.state === "cancelled") return "任务已取消，不会自动重试";
  if (task.state === "failed") return canTaskAction(task, "retry") ? "可重试本次失败；已成功任务会保留" : "自动重试已停止；展开诊断查看错误、次数与截止时间";
  if (task.current_phase) return `当前阶段：${taskPhaseLabel(task.current_phase)}`;
  return task.comfyui_prompt_id ? `ComfyUI ${task.comfyui_prompt_id}` : task.state === "ready" || task.state === "queued" ? "等待依赖、审核与资源条件满足后领取" : task.execution_target === "remote" ? `远程执行：${task.worker_id ?? "待分配"}` : "本地执行";
}

export function filterTasks(tasks: TaskSpec[], filters: { projectId: string; batch?: BatchRun; kind?: string; state?: string }): TaskSpec[] {
  const members = filters.batch ? new Set(filters.batch.items.flatMap((item) => item.task_ids)) : null;
  return tasks.filter((task) => task.project_id === filters.projectId && (!members || members.has(task.task_id))
    && (!filters.kind || task.kind === filters.kind) && (!filters.state || task.state === filters.state));
}

export function batchProgress(batch: BatchRun, tasks: TaskSpec[]) {
  const ids = new Set(batch.items.flatMap((item) => item.task_ids));
  const members = tasks.filter((task) => ids.has(task.task_id));
  const counts = batch.task_counts ?? members.reduce<Record<string, number>>((all, task) => ({ ...all, [task.state]: (all[task.state] ?? 0) + 1 }), {});
  const dynamic = members.some((task) => task.kind === "llm_planning" && task.state !== "succeeded");
  const ended = members.filter((task) => terminalTaskStates.has(task.state)).length;
  // Missing members and open orchestration can grow the denominator.
  const percent = dynamic || members.length !== ids.size || ids.size === 0 ? null : Math.round(ended / ids.size * 100);
  const label = batch.all_tasks_succeeded ? "全部成功" : batch.all_tasks_ended ? "全部已结束，请核对失败与取消项" : dynamic ? "编排中，任务总数尚未确定" : "执行进度";
  return { counts, ended, total: ids.size, percent, label, dynamic };
}

const batchStages = [
  { key: "image", label: "图片", kinds: ["image_generation"] },
  { key: "conditioning", label: "条件编码", kinds: ["conditioning_encoding"] },
  { key: "diffusion", label: "H3 扩散", kinds: ["h3_generation"] },
  { key: "postprocess", label: "后处理/交付", kinds: ["seedvr2", "rife", "whisper", "master_assembly", "export"] },
] as const;

export function batchMemberProgress(taskIds: string[], tasks: TaskSpec[]) {
  const ids = new Set(taskIds);
  const members = tasks.filter((task) => ids.has(task.task_id));
  const orchestrators = members.filter((task) => task.kind === "llm_planning");
  const orchestrator = orchestrators.find((task) => task.state !== "succeeded") ?? orchestrators[0];
  const expanding = orchestrators.some((task) => task.state !== "succeeded");
  const stages = batchStages.map(({ key, label, kinds }) => {
    const stageTasks = members.filter((task) => (kinds as readonly string[]).includes(task.kind));
    return { key, label, succeeded: stageTasks.filter((task) => task.state === "succeeded").length, registered: stageTasks.length, totalKnown: !expanding && members.length === ids.size };
  });
  const priority: TaskState[] = ["needs_attention", "failed", "cancelling", "running", "recovering", "retry_wait", "queued", "ready", "paused", "needs_review", "blocked"];
  const current = orchestrator && ["failed", "needs_attention"].includes(orchestrator.state)
    ? orchestrator
    : priority.flatMap((state) => members.filter((task) => task.state === state))
      .find((task) => task.kind !== "llm_planning") ?? (orchestrator && !terminalTaskStates.has(orchestrator.state) ? orchestrator : undefined);
  let phase: string;
  if (current) {
    const detail = current.kind === "llm_planning"
      ? taskPhaseLabel(current.current_phase)
      : taskKindLabels[current.kind];
    phase = `${detail || "项目编排"} · ${taskStateLabels[current.state]}`;
  } else if (orchestrator?.state === "failed" || orchestrator?.state === "needs_attention") {
    phase = `${taskPhaseLabel(orchestrator.current_phase) || "项目编排"} · ${taskStateLabels[orchestrator.state]}`;
  } else if (!members.length) {
    phase = "等待登记项目任务";
  } else if (members.every((task) => task.state === "succeeded")) {
    phase = "已登记任务全部成功";
  } else {
    phase = "等待下一阶段任务登记";
  }
  return { stages, phase, expanding };
}
