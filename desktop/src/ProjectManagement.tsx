import { Clock3, FilePlus2, FolderOpen, ListChecks, Search, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { BackendProjectSpec } from "./api";

type Props = {
  projects: BackendProjectSpec[];
  activeProjectId: string;
  busy: boolean;
  onOpen: (project: BackendProjectSpec) => void;
  onCreate: () => void;
  onClearTasks: (project: BackendProjectSpec) => void;
  onDelete: (project: BackendProjectSpec) => void;
};

export function ProjectManagement({ projects, activeProjectId, busy, onOpen, onCreate, onClearTasks, onDelete }: Props) {
  const [query, setQuery] = useState("");
  const visible = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    return normalized
      ? projects.filter((item) => item.name.toLocaleLowerCase().includes(normalized))
      : projects;
  }, [projects, query]);

  return <section className="workspace project-management">
    <div className="section-heading">
      <div><h2>项目管理</h2><span>{projects.length} 个项目 · 在同一处打开、清理或删除项目</span></div>
      <button className="primary-button" disabled={busy} onClick={onCreate}><FilePlus2 size={16} />新建项目</button>
    </div>
    <label className="project-management-search"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索项目名称" /></label>
    <div className="project-management-table">
      <div className="project-management-header"><span>项目</span><span>规格</span><span>最后更新</span><span>操作</span></div>
      {visible.map((item) => <div className={`project-management-row ${item.project_id === activeProjectId ? "current" : ""}`} key={item.project_id}>
        <div><FolderOpen size={18} /><span><strong>{item.name}</strong><small>R{item.revision}{item.project_id === activeProjectId ? " · 当前项目" : ""}</small></span></div>
        <span>{item.width} x {item.height} · {item.fps} FPS · {item.target_duration_seconds} 秒</span>
        <span><Clock3 size={13} />{item.updated_at ? new Date(item.updated_at).toLocaleString() : "未知"}</span>
        <div className="project-management-actions">
          <button className="secondary-button" disabled={busy} onClick={() => onOpen(item)}><FolderOpen size={14} />打开</button>
          <button className="secondary-button" disabled={busy} onClick={() => onClearTasks(item)}><ListChecks size={14} />清除任务</button>
          <button className="icon-button danger-button" title="删除项目" aria-label={`删除项目 ${item.name}`} disabled={busy} onClick={() => onDelete(item)}><Trash2 size={16} /></button>
        </div>
      </div>)}
      {!visible.length && <div className="table-empty">{projects.length ? "没有匹配的项目" : "还没有项目"}</div>}
    </div>
  </section>;
}
