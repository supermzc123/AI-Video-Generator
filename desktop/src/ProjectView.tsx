import { ArrowDown, ArrowUp, Plus, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { listComfyuiModels } from "./api";
import type { ProjectDraft } from "./types";

type Props = {
  project: ProjectDraft;
  onChange: (project: ProjectDraft) => void;
  onValidationChange?: (valid: boolean) => void;
};

export function ProjectView({ project, onChange, onValidationChange }: Props) {
  const [draft, setDraft] = useState(project);
  const [error, setError] = useState<string | null>(null);
  const [widthInput, setWidthInput] = useState(String(project.width));
  const [heightInput, setHeightInput] = useState(String(project.height));
  const [availableLoras, setAvailableLoras] = useState<string[]>([]);
  const [loraLoading, setLoraLoading] = useState(true);
  const [loraMessage, setLoraMessage] = useState<string | null>(null);
  const updateDraft = (transform: (current: ProjectDraft) => ProjectDraft) => {
    const changed = transform(draft);
    setDraft(changed);
    onChange(changed);
  };
  useEffect(() => {
    setDraft(project);
    setWidthInput(String(project.width));
    setHeightInput(String(project.height));
  }, [project]);
  async function refreshLoras() {
    setLoraLoading(true);
    setLoraMessage(null);
    try {
      const models = await listComfyuiModels();
      setAvailableLoras(models.loras);
      setLoraMessage(models.warning ?? `已读取 ${models.loras.length} 个 LoRA`);
    } catch (cause) {
      setAvailableLoras([]);
      setLoraMessage(cause instanceof Error ? cause.message : "LoRA 列表读取失败");
    } finally {
      setLoraLoading(false);
    }
  }
  useEffect(() => { void refreshLoras(); }, []);
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

  return <section className="embedded-project-settings">
    <div className="form-panel">
      {error && <div className="error-banner inline-banner field-wide">{error}</div>}
      <label className="field field-wide"><span>项目名称</span><input value={draft.name} onChange={(e) => updateDraft((current) => ({ ...current, name: e.target.value }))} /></label>
      <label className="field field-wide"><span>最高指令</span><textarea rows={4} value={draft.highestInstruction} onChange={(e) => updateDraft((current) => ({ ...current, highestInstruction: e.target.value }))} placeholder="本项目所有 AI 阶段都必须遵守的最高级项目要求" /></label>
      <label className="field"><span>宽度</span><input inputMode="numeric" value={widthInput} onChange={(e) => setWidthInput(e.target.value)} onBlur={() => commitResolution("width", "宽度", widthInput)} /></label>
      <label className="field"><span>高度</span><input inputMode="numeric" value={heightInput} onChange={(e) => setHeightInput(e.target.value)} onBlur={() => commitResolution("height", "高度", heightInput)} /></label>
      <small className="field-wide resolution-hint">输入完成后自动四舍五入到最接近的 32 倍数。</small>
      <label className="field"><span>帧率</span><input type="number" min="1" max="120" value={draft.fps} onChange={(e) => number("fps", 1)(e.target.value)} /></label>
      <label className="field"><span>目标时长（秒）</span><input type="number" min="1" value={draft.targetDurationSeconds} onChange={(e) => number("targetDurationSeconds", 1)(e.target.value)} /></label>
      <label className="field"><span>音频策略</span><select value={draft.audioPolicy} onChange={(e) => updateDraft((current) => ({ ...current, audioPolicy: e.target.value as ProjectDraft["audioPolicy"], externalAudioAssetId: e.target.value === "h3_with_external" ? current.externalAudioAssetId : null }))}><option value="h3_native">H3 原生音频</option><option value="h3_with_external">H3 + 外部音轨</option><option value="muted">静音</option></select></label>
      <label className="field"><span>时间预算（分钟，留空不限）</span><input type="number" min="1" value={draft.timeBudgetSeconds ? draft.timeBudgetSeconds / 60 : ""} onChange={(e) => updateDraft((current) => ({ ...current, timeBudgetSeconds: e.target.value ? Math.max(60, Number(e.target.value) * 60) : null }))} /></label>
      {draft.audioPolicy === "h3_with_external" && <label className="field field-wide"><span>外部音频素材 ID</span><input value={draft.externalAudioAssetId ?? ""} onChange={(e) => updateDraft((current) => ({ ...current, externalAudioAssetId: e.target.value || null }))} /></label>}
      <label className="field"><span>审核模式</span><select value={draft.reviewPolicy.configuredMode} onChange={(e) => updateDraft((current) => ({ ...current, reviewPolicy: { ...current.reviewPolicy, configuredMode: e.target.value as ProjectDraft["reviewPolicy"]["configuredMode"], effectiveMode: e.target.value as ProjectDraft["reviewPolicy"]["effectiveMode"], aiTakeoverAt: null } }))}><option value="human_ai">AI + 人工</option><option value="ai_only">AI</option><option value="manual">手动</option><option value="none">无</option></select></label>
      {draft.reviewPolicy.configuredMode === "human_ai" && <label className="field"><span>人工审核超时（分钟）</span><input type="number" min="1" value={draft.reviewPolicy.humanTimeoutSeconds / 60} onChange={(e) => updateDraft((current) => ({ ...current, reviewPolicy: { ...current.reviewPolicy, humanTimeoutSeconds: Math.max(60, Number(e.target.value) * 60) } }))} /></label>}
      <div className="field field-wide">
        <div className="inline-toolbar lora-toolbar">
          <span>项目 LoRA（按顺序应用）</span>
          <button className="icon-button" type="button" title="刷新 LoRA 列表" disabled={loraLoading} onClick={() => void refreshLoras()}><RefreshCw size={14} /></button>
          <button className="secondary-button" type="button" disabled={loraLoading || !availableLoras.some((name) => !draft.h3Loras.some((item) => item.name === name))} onClick={() => {
            const name = availableLoras.find((candidate) => !draft.h3Loras.some((item) => item.name === candidate));
            if (name) updateDraft((current) => ({ ...current, h3Loras: [...current.h3Loras, { id: crypto.randomUUID(), name, strength: 1, enabled: true }] }));
          }}><Plus size={14} />添加 LoRA</button>
        </div>
        <small>{loraLoading ? "正在读取 LoRA 列表..." : loraMessage ?? "没有找到可加载的 LoRA 文件。"}</small>
        {draft.h3Loras.length === 0 && <small>当前项目不加载额外 LoRA。</small>}
        {draft.h3Loras.map((lora, index) => <div className="inline-toolbar lora-row" key={lora.id}>
          <input type="checkbox" checked={lora.enabled} aria-label={`启用 ${lora.name}`} onChange={(event) => updateDraft((current) => ({ ...current, h3Loras: current.h3Loras.map((item) => item.id === lora.id ? { ...item, enabled: event.target.checked } : item) }))} />
          <select value={lora.name} onChange={(event) => updateDraft((current) => ({ ...current, h3Loras: current.h3Loras.map((item) => item.id === lora.id ? { ...item, name: event.target.value } : item) }))}>
            {[lora.name, ...availableLoras.filter((name) => name !== lora.name && !draft.h3Loras.some((item) => item.name === name))].map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
          <input type="number" min="-4" max="4" step="0.05" value={lora.strength} aria-label={`${lora.name} 强度`} onChange={(event) => {
            const strength = Math.max(-4, Math.min(4, Number(event.target.value) || 0));
            updateDraft((current) => ({ ...current, h3Loras: current.h3Loras.map((item) => item.id === lora.id ? { ...item, strength } : item) }));
          }} />
          <button className="icon-button" type="button" title="上移" disabled={index === 0} onClick={() => updateDraft((current) => { const items = [...current.h3Loras]; [items[index - 1], items[index]] = [items[index], items[index - 1]]; return { ...current, h3Loras: items }; })}><ArrowUp size={14} /></button>
          <button className="icon-button" type="button" title="下移" disabled={index === draft.h3Loras.length - 1} onClick={() => updateDraft((current) => { const items = [...current.h3Loras]; [items[index], items[index + 1]] = [items[index + 1], items[index]]; return { ...current, h3Loras: items }; })}><ArrowDown size={14} /></button>
          <button className="icon-button" type="button" title="删除" onClick={() => updateDraft((current) => ({ ...current, h3Loras: current.h3Loras.filter((item) => item.id !== lora.id) }))}><Trash2 size={14} /></button>
        </div>)}
      </div>
    </div>
  </section>;
}
