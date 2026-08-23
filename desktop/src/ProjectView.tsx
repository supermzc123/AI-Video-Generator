import { RotateCcw, Save } from "lucide-react";
import { useEffect, useState } from "react";
import {
  saveProjectToControlPlane,
  setProjectMode,
  setProjectPaused,
  setReviewMode,
} from "./api";
import { newProject } from "./project-store";
import type { ProjectDraft } from "./types";

type Props = {
  project: ProjectDraft;
  onChange: (project: ProjectDraft) => void;
  embedded?: boolean;
  onValidationChange?: (valid: boolean) => void;
};

export function ProjectView({ project, onChange, embedded = false, onValidationChange }: Props) {
  const [draft, setDraft] = useState(project);
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [widthInput, setWidthInput] = useState(String(project.width));
  const [heightInput, setHeightInput] = useState(String(project.height));
  const updateDraft = (transform: (current: ProjectDraft) => ProjectDraft) => {
    const changed = transform(draft);
    setDraft(changed);
    if (embedded) onChange(changed);
  };
  useEffect(() => {
    setDraft(project);
    setWidthInput(String(project.width));
    setHeightInput(String(project.height));
  }, [project]);
  const number = (key: keyof ProjectDraft, minimum: number) => (value: string) =>
    updateDraft((current) => ({ ...current, [key]: Math.max(minimum, Number(value) || minimum) }));

  function snappedResolution(value: string): number | null {
    const parsed = Number(value);
    if (!value.trim() || !Number.isFinite(parsed)) return null;
    return Math.max(64, Math.round(parsed / 32) * 32);
  }

  function resolutionValue(label: string, value: string): number | null {
    const parsed = snappedResolution(value);
    if (parsed === null) setError(`${label}必须填写有效数字`);
    return parsed;
  }

  function commitResolution(key: "width" | "height", label: string, value: string) {
    const parsed = resolutionValue(label, value);
    if (parsed === null) {
      onValidationChange?.(false);
      return;
    }
    if (key === "width") setWidthInput(String(parsed));
    else setHeightInput(String(parsed));
    updateDraft((current) => ({ ...current, [key]: parsed }));
    const otherValid = snappedResolution(key === "width" ? heightInput : widthInput) !== null;
    onValidationChange?.(otherValid);
    if (otherValid) setError(null);
  }

  return <section className={embedded ? "embedded-project-settings" : "workspace"}>
    {!embedded && <div className="section-heading"><div><h2>项目设置</h2><span>控制平面不可变修订 · R{project.revision}</span></div></div>}
    <div className="form-panel">
      {error && <div className="error-banner inline-banner field-wide">{error}</div>}
      <label className="field field-wide"><span>项目名称</span><input value={draft.name} onChange={(e) => updateDraft((current) => ({ ...current, name: e.target.value }))} /></label>
      <label className="field"><span>宽度</span><input inputMode="numeric" value={widthInput} onChange={(e) => setWidthInput(e.target.value)} onBlur={() => commitResolution("width", "宽度", widthInput)} /></label>
      <label className="field"><span>高度</span><input inputMode="numeric" value={heightInput} onChange={(e) => setHeightInput(e.target.value)} onBlur={() => commitResolution("height", "高度", heightInput)} /></label>
      <small className="field-wide resolution-hint">输入完成后自动四舍五入到最接近的 32 倍数。</small>
      <label className="field"><span>帧率</span><input type="number" min="1" max="120" value={draft.fps} onChange={(e) => number("fps", 1)(e.target.value)} /></label>
      <label className="field"><span>目标时长（秒）</span><input type="number" min="1" value={draft.targetDurationSeconds} onChange={(e) => number("targetDurationSeconds", 1)(e.target.value)} /></label>
      <label className="field"><span>音频策略</span><select value={draft.audioPolicy} onChange={(e) => updateDraft((current) => ({ ...current, audioPolicy: e.target.value as ProjectDraft["audioPolicy"], externalAudioAssetId: e.target.value === "h3_with_external" ? current.externalAudioAssetId : null }))}><option value="h3_native">H3 原生音频</option><option value="h3_with_external">H3 + 外部音轨</option><option value="muted">静音</option></select></label>
      <label className="field"><span>时间预算（分钟，留空不限）</span><input type="number" min="1" value={draft.timeBudgetSeconds ? draft.timeBudgetSeconds / 60 : ""} onChange={(e) => updateDraft((current) => ({ ...current, timeBudgetSeconds: e.target.value ? Math.max(60, Number(e.target.value) * 60) : null }))} /></label>
      {draft.audioPolicy === "h3_with_external" && <label className="field field-wide"><span>外部音频素材 ID</span><input value={draft.externalAudioAssetId ?? ""} onChange={(e) => updateDraft((current) => ({ ...current, externalAudioAssetId: e.target.value || null }))} /></label>}
      <label className="field"><span>执行方式</span><select value={draft.executionMode} onChange={(e) => updateDraft((current) => ({ ...current, executionMode: e.target.value as ProjectDraft["executionMode"] }))}><option value="guided">精细化模式</option><option value="batch">批量模式</option></select></label>
      <label className="field"><span>审核模式</span><select value={draft.reviewPolicy.configuredMode} onChange={(e) => updateDraft((current) => ({ ...current, reviewPolicy: { ...current.reviewPolicy, configuredMode: e.target.value as ProjectDraft["reviewPolicy"]["configuredMode"], effectiveMode: e.target.value as ProjectDraft["reviewPolicy"]["effectiveMode"], aiTakeoverAt: null } }))}><option value="human_ai">AI + 人工</option><option value="ai_only">AI</option><option value="manual">手动</option><option value="none">无</option></select></label>
      {draft.reviewPolicy.configuredMode === "human_ai" && <label className="field"><span>人工审核超时（分钟）</span><input type="number" min="1" value={draft.reviewPolicy.humanTimeoutSeconds / 60} onChange={(e) => updateDraft((current) => ({ ...current, reviewPolicy: { ...current.reviewPolicy, humanTimeoutSeconds: Math.max(60, Number(e.target.value) * 60) } }))} /></label>}
      {!embedded && <div className="form-actions">
        <button className="secondary-button" disabled={busy} onClick={() => { const fresh = newProject(); setDraft(fresh); setWidthInput(String(fresh.width)); setHeightInput(String(fresh.height)); onChange(fresh); }}><RotateCcw size={16} />新建</button>
        <button className="primary-button" disabled={busy || !draft.name.trim() || (draft.audioPolicy === "h3_with_external" && !draft.externalAudioAssetId)} onClick={() => void (async () => {
          const width = resolutionValue("宽度", widthInput);
          const height = resolutionValue("高度", heightInput);
          if (width === null || height === null) return;
          setBusy(true); setError(null);
          try {
            const next = await saveProjectToControlPlane({ ...draft, width, height, name: draft.name.trim() });
            await setProjectMode(next.projectId, next.executionMode);
            await setProjectPaused(next.projectId, next.paused);
            await setReviewMode(next.projectId, next.reviewPolicy.configuredMode, next.reviewPolicy.humanTimeoutSeconds);
            onChange(next); setDraft(next); setSaved(true);
            window.setTimeout(() => setSaved(false), 1600);
          } catch (cause) { setError(cause instanceof Error ? cause.message : "保存失败"); }
          finally { setBusy(false); }
        })()}><Save size={16} />{busy ? "保存中" : saved ? "已保存" : "保存修订"}</button>
      </div>}
    </div>
  </section>;
}
