import type { TaskCommandAction, TaskCommandReceipt, TaskCommandRequest, TaskCommandScope, TaskSpec } from "./types";

export class TaskCommandClient {
  private pending = new Map<string, TaskCommandRequest>();
  constructor(private post: (payload: TaskCommandRequest) => Promise<TaskCommandReceipt>, private key = () => crypto.randomUUID()) {}

  async run(scope: TaskCommandScope, action: TaskCommandAction, tasks: TaskSpec[], confirmed = false): Promise<TaskCommandReceipt> {
    if (!scope.scope_id || !tasks.length || tasks.length > 500) throw new Error("请选择作用范围内的 1–500 个任务");
    if (scope.scope === "project" && tasks.some((task) => task.project_id !== scope.scope_id)) throw new Error("任务不属于当前项目");
    if (new Set(tasks.map((task) => task.task_id)).size !== tasks.length) throw new Error("任务选择存在重复");
    if (action === "confirm_retry" && !confirmed) throw new Error("需要明确确认重复执行风险");
    const taskIds = tasks.map((task) => task.task_id).sort();
    const signature = JSON.stringify([scope.scope, scope.scope_id, action, taskIds]);
    const existing = this.pending.get(signature);
    if (!existing && tasks.some((task) => !task.available_actions?.includes(action))) throw new Error("任务状态已变更或不允许此操作，请刷新后重试");
    const request = existing ?? { ...scope, action, task_ids: taskIds, idempotency_key: this.key(), confirm_duplicate_execution: confirmed };
    this.pending.set(signature, request);
    // A lost response retains the same key. Re-clicking checks the receipt without replaying side effects.
    const receipt = await this.post(request);
    if (receipt.status === "completed") this.pending.delete(signature);
    return receipt;
  }
}

export function describeCommandReceipt(receipt: TaskCommandReceipt): string {
  const failures = receipt.results.filter((item) => !item.ok);
  if (receipt.status !== "completed") return receipt.message || "命令回执尚未完整，请刷新状态并重新对账";
  if (failures.length) return `${receipt.results.length - failures.length} 项已处理，${failures.length} 项未完成：${failures.map((item) => `${item.task_id}：${item.error ?? "未知错误"}`).join("；")}`;
  const cancelling = receipt.results.filter((item) => item.state === "cancelling").length;
  if (cancelling) return `${cancelling} 项正在取消，等待执行端确认；其余 ${receipt.results.length - cancelling} 项已处理`;
  return `${receipt.results.length} 项命令已处理，具体结果以最新任务状态为准`;
}
