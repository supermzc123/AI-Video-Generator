import { CircleAlert, Pause, Play, RefreshCw, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { cancelBatchProject, createBatch, getProjectRunState, listBatches, listTasks, transitionBatch } from "./api";
import type { BackendProjectSpec } from "./api";
import type { BatchRun, ProjectDraft, TaskSpec } from "./types";

type Props = { project: ProjectDraft; projects: BackendProjectSpec[] };
type Selection = { startBoundary: string; priority: number };

const boundaryOptions = [
  { value: "next_ready", label: "下一可执行边界" },
  { value: "generation", label: "生成阶段" },
  { value: "review", label: "审核阶段" },
  { value: "delivery", label: "交付阶段" },
];
const terminalStates = new Set(["succeeded", "failed", "cancelled", "stale"]);

export function BatchView({ project, projects }: Props) {
  const [tasks, setTasks] = useState<TaskSpec[]>([]);
  const [batches, setBatches] = useState<BatchRun[]>([]);
  const [selected, setSelected] = useState<Map<string, Selection>>(new Map());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outlineReady, setOutlineReady] = useState<Set<string>>(new Set());
  const allProjects = useMemo(() => {
    const values = new Map(projects.map((item) => [item.project_id, item.name]));
    values.set(project.projectId, project.name);
    return [...values.entries()];
  }, [project.name, project.projectId, projects]);

  const refresh = useCallback(async () => {
    try {
      const [nextTasks, nextBatches, runStates] = await Promise.all([
        listTasks(),
        listBatches(),
        Promise.allSettled(allProjects.map(([projectId]) => getProjectRunState(projectId))),
      ]);
      setTasks(nextTasks);
      setBatches(nextBatches);
      setOutlineReady(new Set(runStates.flatMap((result) => result.status === "fulfilled" && result.value.outline_approved ? [result.value.project_id] : [])));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法加载批量工作台");
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

  function toggleProject(projectId: string, checked: boolean) {
    setSelected((current) => {
      const next = new Map(current);
      if (checked) next.set(projectId, { startBoundary: "next_ready", priority: 0 });
      else next.delete(projectId);
      return next;
    });
  }

  function updateSelection(projectId: string, update: Partial<Selection>) {
    setSelected((current) => {
      const next = new Map(current);
      const existing = next.get(projectId);
      if (existing) next.set(projectId, { ...existing, ...update });
      return next;
    });
  }

  async function createAndStart() {
    setBusy(true);
    try {
      const now = new Date().toISOString();
      const batch: BatchRun = {
        schema_version: "1.0",
        batch_id: crypto.randomUUID(),
        name: `${selected.size} 个项目批次`,
        state: "draft",
        items: [...selected].map(([projectId, selection]) => ({
          project_id: projectId,
          task_ids: [],
          start_boundary: selection.startBoundary,
          priority: selection.priority,
        })),
        created_at: now,
        updated_at: now,
      };
      const resolved = await createBatch(batch);
      await transitionBatch(resolved.batch_id, "start");
      setSelected(new Map());
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "创建批次失败");
    } finally {
      setBusy(false);
    }
  }

  async function action(batchId: string, value: "pause" | "resume" | "cancel") {
    setBusy(true);
    try {
      await transitionBatch(batchId, value);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "批次操作失败");
    } finally {
      setBusy(false);
    }
  }

  async function cancelMember(batchId: string, projectId: string) {
    setBusy(true);
    try {
      await cancelBatchProject(batchId, projectId);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "取消批次项目失败");
    } finally {
      setBusy(false);
    }
  }

  return <section className="workspace">
    <div className="section-heading">
      <div><h2>批量工作台</h2><span>选择项目和起点；依赖、模型切换与执行顺序由控制平面自动处理</span></div>
      <button className="icon-button" title="刷新" onClick={() => void refresh()}><RefreshCw size={18} /></button>
    </div>
    {error && <div className="error-banner inline-banner"><CircleAlert size={17} />{error}</div>}
    <div className="batch-layout">
      <div className="batch-selector">
        <div className="batch-toolbar"><span>已选择 {selected.size} 个项目</span><button className="primary-button" disabled={busy || !selected.size} onClick={() => void createAndStart()}><Play size={16} />启动批次</button></div>
        {allProjects.map(([projectId, name]) => {
          const projectTasks = tasksByProject.get(projectId) ?? [];
          const unfinished = projectTasks.filter((task) => !terminalStates.has(task.state));
          const selection = selected.get(projectId);
          const canEnterBatch = unfinished.length > 0 || outlineReady.has(projectId);
          const counts = projectTasks.reduce<Record<string, number>>((result, task) => {
            result[task.state] = (result[task.state] ?? 0) + 1;
            return result;
          }, {});
          return <div className="batch-project" key={projectId}>
            <header><label className="batch-project-choice"><input type="checkbox" checked={Boolean(selection)} disabled={!canEnterBatch} onChange={(event) => toggleProject(projectId, event.target.checked)} /><strong>{name}</strong></label><span>{projectTasks.length ? `${unfinished.length} 个未完成 · ${counts.blocked ?? 0} 个等待依赖` : outlineReady.has(projectId) ? "大纲已批准 · 将自动规划并编译" : "需先批准故事大纲"}</span></header>
            {selection && <div className="batch-project-controls"><label>起始边界<select value={selection.startBoundary} onChange={(event) => updateSelection(projectId, { startBoundary: event.target.value })}>{boundaryOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label><label>优先级<input type="number" min="-100" max="100" value={selection.priority} onChange={(event) => updateSelection(projectId, { priority: Number(event.target.value) })} /></label></div>}
            {projectTasks.length > 0 && <div className="batch-status-strip">{Object.entries(counts).map(([state, count]) => <span className={`state state-${state}`} key={state}>{state} {count}</span>)}</div>}
          </div>;
        })}
      </div>
      <div className="batch-history">
        {batches.map((batch) => <div className="batch-run" key={batch.batch_id}>
          <div className="batch-run-heading"><div><strong>{batch.name}</strong><span>{batch.items.length} 个项目 · {batch.items.reduce((sum, item) => sum + item.task_ids.length, 0)} 项编排与执行任务</span></div><span className={`state state-${batch.state}`}>{batch.state}</span>{batch.state === "running" && <button className="icon-button" title="暂停批次" onClick={() => void action(batch.batch_id, "pause")}><Pause size={16} /></button>}{batch.state === "paused" && <button className="icon-button" title="恢复批次" onClick={() => void action(batch.batch_id, "resume")}><Play size={16} /></button>}{(["draft", "running", "paused"] as string[]).includes(batch.state) && <button className="icon-button" title="取消整个批次" onClick={() => void action(batch.batch_id, "cancel")}><Square size={15} /></button>}</div>
          <div className="batch-members">{batch.items.map((item) => {
            const memberTasks = item.task_ids.map((taskId) => tasks.find((task) => task.task_id === taskId)).filter((task): task is TaskSpec => Boolean(task));
            const unfinished = memberTasks.filter((task) => !terminalStates.has(task.state)).length;
            return <div className="batch-member" key={item.project_id}><div><strong>{allProjects.find(([projectId]) => projectId === item.project_id)?.[1] ?? item.project_id}</strong><span>{unfinished ? `${unfinished} 项进行中或等待` : "已结束"}</span></div>{batch.state === "running" && unfinished > 0 && <button className="secondary-button" disabled={busy} onClick={() => void cancelMember(batch.batch_id, item.project_id)}><Square size={13} />取消此项目</button>}</div>;
          })}</div>
        </div>)}
        {!batches.length && <div className="table-empty">尚未创建批次</div>}
      </div>
    </div>
  </section>;
}
