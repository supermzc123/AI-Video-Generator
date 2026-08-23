import { CircleAlert, Clock3, FilePlus2, FolderOpen, Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import type { BackendProjectSpec } from "./api";

type NewProjectInput = {
  name: string;
  width: string;
  height: string;
  targetDurationSeconds: string;
};

type Props = {
  mode: "open" | "new";
  projects: BackendProjectSpec[];
  activeProjectId: string;
  busy: boolean;
  error: string | null;
  onClose: () => void;
  onModeChange: (mode: "open" | "new") => void;
  onOpen: (project: BackendProjectSpec) => void;
  onCreate: (value: NewProjectInput) => void;
};

function snapResolution(value: string): string {
  const parsed = Number(value);
  if (!value.trim() || !Number.isFinite(parsed)) return value;
  return String(Math.max(64, Math.round(parsed / 32) * 32));
}

export function ProjectSwitcher({ mode, projects, activeProjectId, busy, error, onClose, onModeChange, onOpen, onCreate }: Props) {
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState<NewProjectInput>({
    name: "",
    width: "1024",
    height: "608",
    targetDurationSeconds: "60",
  });
  const visible = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    return normalized
      ? projects.filter((project) => project.name.toLocaleLowerCase().includes(normalized))
      : projects;
  }, [projects, query]);

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="project-dialog" role="dialog" aria-modal="true" aria-labelledby="project-dialog-title">
        <header>
          <div>
            <h2 id="project-dialog-title">{mode === "open" ? "打开项目" : "新建项目"}</h2>
            <span>{mode === "open" ? "选择一个已保存项目继续制作" : "创建后从项目配置阶段开始"}</span>
          </div>
          <button className="icon-button" title="关闭" aria-label="关闭" onClick={onClose}><X size={18} /></button>
        </header>

        <div className="project-dialog-tabs" role="tablist">
          <button role="tab" aria-selected={mode === "open"} className={mode === "open" ? "active" : ""} onClick={() => onModeChange("open")}><FolderOpen size={16} />打开项目</button>
          <button role="tab" aria-selected={mode === "new"} className={mode === "new" ? "active" : ""} onClick={() => onModeChange("new")}><FilePlus2 size={16} />新建项目</button>
        </div>

        {error && <div className="error-banner project-dialog-error"><CircleAlert size={17} />{error}</div>}

        {mode === "open" ? (
          <div className="project-browser">
            <label className="project-search"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="按项目名称搜索" /></label>
            <div className="project-list">
              {visible.map((item) => (
                <button className={`project-list-item ${item.project_id === activeProjectId ? "current" : ""}`} key={item.project_id} disabled={busy} onClick={() => onOpen(item)}>
                  <span className="project-list-icon"><FolderOpen size={18} /></span>
                  <span className="project-list-copy"><strong>{item.name}</strong><small>{item.width} x {item.height} · {item.fps} FPS · {item.target_duration_seconds} 秒{item.updated_at ? ` · 更新于 ${new Date(item.updated_at).toLocaleString()}` : ""}</small></span>
                  <span className="project-list-revision">R{item.revision}{item.project_id === activeProjectId ? <small>当前</small> : null}</span>
                </button>
              ))}
              {!visible.length && <div className="project-list-empty">{projects.length ? "没有匹配的项目" : "还没有已保存项目"}</div>}
            </div>
          </div>
        ) : (
          <form className="new-project-form" onSubmit={(event) => { event.preventDefault(); const normalized = { ...draft, width: snapResolution(draft.width), height: snapResolution(draft.height) }; setDraft(normalized); onCreate(normalized); }}>
            <label className="field field-wide"><span>项目名称</span><input autoFocus value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} placeholder="例如：雨夜追踪短片" /></label>
            <label className="field"><span>宽度</span><input inputMode="numeric" value={draft.width} onChange={(event) => setDraft({ ...draft, width: event.target.value })} onBlur={() => setDraft((current) => ({ ...current, width: snapResolution(current.width) }))} placeholder="1024" /></label>
            <label className="field"><span>高度</span><input inputMode="numeric" value={draft.height} onChange={(event) => setDraft({ ...draft, height: event.target.value })} onBlur={() => setDraft((current) => ({ ...current, height: snapResolution(current.height) }))} placeholder="608" /></label>
            <label className="field field-wide"><span>目标时长（秒）</span><input inputMode="decimal" value={draft.targetDurationSeconds} onChange={(event) => setDraft({ ...draft, targetDurationSeconds: event.target.value })} placeholder="60" /></label>
            <div className="new-project-note"><Clock3 size={15} /><span>画幅可在项目配置阶段继续修改；输入完成后宽高会自动四舍五入到最接近的 32 倍数。</span></div>
            <footer>
              <button type="button" className="secondary-button" onClick={onClose}>取消</button>
              <button type="submit" className="primary-button" disabled={busy}>{busy ? "正在创建..." : "创建并打开"}</button>
            </footer>
          </form>
        )}
      </section>
    </div>
  );
}
