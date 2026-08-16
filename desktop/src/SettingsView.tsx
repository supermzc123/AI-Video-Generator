import { Blocks, Check, CircleAlert, Eye, EyeOff, PlugZap, RefreshCw, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  getRuntimeSettings,
  getLocalWorkerCapabilities,
  installRequiredComfyNodes,
  listComfyuiModels,
  listLlmModels,
  updateRuntimeSettings,
} from "./api";
import type { ComfyNodeInstallResult, RuntimeSettings } from "./api";

const fallback: RuntimeSettings = {
  comfyuiRoot: "",
  comfyuiBaseUrl: "http://127.0.0.1:8188",
  requestTimeoutSeconds: 3,
  llmBaseUrl: "",
  llmModel: "",
  llmApiKeyConfigured: false,
  llmTimeoutSeconds: 30,
  llmVideoCapable: false,
  networkProxy: "",
  h3DiffusionModel: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
  h3TextEncoder: "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
  h3VideoVae: "minimax_h3_video_vae_fp16.safetensors",
  h3AudioVae: "minimax_h3_audio_vae_fp32.safetensors",
  h3TurboLora: "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
  h3TurboEnabled: true,
  h3SageAttentionEnabled: false,
  h3LowVram: true,
  h3Steps: 6,
};

function splitComfyUrl(value: string): { origin: string; port: string } {
  try {
    const url = new URL(value);
    const port = url.port || (url.protocol === "https:" ? "443" : "80");
    return { origin: `${url.protocol}//${url.hostname}`, port };
  } catch {
    return { origin: "http://127.0.0.1", port: "8188" };
  }
}

export function SettingsView() {
  const [settings, setSettings] = useState<RuntimeSettings>(fallback);
  const [comfyOrigin, setComfyOrigin] = useState("http://127.0.0.1");
  const [comfyPort, setComfyPort] = useState("8188");
  const [apiKey, setApiKey] = useState("");
  const [clearApiKey, setClearApiKey] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [llmModels, setLlmModels] = useState<string[]>([]);
  const [comfyModels, setComfyModels] = useState({ diffusion_models: [] as string[], text_encoders: [] as string[], vaes: [] as string[], loras: [] as string[] });
  const [savedSnapshot, setSavedSnapshot] = useState("");
  const [workerBlockers, setWorkerBlockers] = useState<string[]>([]);
  const [nodeInstallResult, setNodeInstallResult] = useState<ComfyNodeInstallResult | null>(null);

  useEffect(() => {
    void getRuntimeSettings()
      .then((value) => {
        const comfy = splitComfyUrl(value.comfyuiBaseUrl);
        setSettings(value);
        setComfyOrigin(comfy.origin);
        setComfyPort(comfy.port);
        setSavedSnapshot(JSON.stringify({ value, origin: comfy.origin, port: comfy.port }));
      })
      .catch((cause) => setError(cause instanceof Error ? cause.message : "无法读取设置"));
  }, []);

  const portValid = useMemo(() => {
    const value = Number(comfyPort);
    return Number.isInteger(value) && value >= 1 && value <= 65535;
  }, [comfyPort]);
  const dirty = useMemo(
    () => Boolean(apiKey || clearApiKey || savedSnapshot !== JSON.stringify({ value: settings, origin: comfyOrigin, port: comfyPort })),
    [apiKey, clearApiKey, savedSnapshot, settings, comfyOrigin, comfyPort],
  );
  const h3Models = useMemo(() => ({
    diffusion: comfyModels.diffusion_models.filter((item) => item.toLowerCase().includes("minimax")),
    textEncoders: comfyModels.text_encoders.filter((item) => item.toLowerCase().includes("minimax")),
    vaes: comfyModels.vaes.filter((item) => item.toLowerCase().includes("minimax")),
    loras: comfyModels.loras.filter((item) => item.toLowerCase().includes("minimax_h3_turbo")),
  }), [comfyModels]);

  async function save(showSuccess = true): Promise<boolean> {
    if (!portValid) {
      setError("ComfyUI 端口必须是 1 到 65535 之间的整数");
      return false;
    }
    let baseUrl: string;
    try {
      const parsed = new URL(comfyOrigin);
      parsed.port = comfyPort;
      parsed.pathname = "";
      parsed.search = "";
      parsed.hash = "";
      baseUrl = parsed.toString().replace(/\/$/, "");
    } catch {
      setError("ComfyUI 地址必须包含 http:// 或 https://");
      return false;
    }
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const saved = await updateRuntimeSettings(
        { ...settings, comfyuiBaseUrl: baseUrl },
        apiKey,
        clearApiKey,
      );
      setSettings(saved);
      setApiKey("");
      setClearApiKey(false);
      const comfy = splitComfyUrl(saved.comfyuiBaseUrl);
      setComfyOrigin(comfy.origin);
      setComfyPort(comfy.port);
      setSavedSnapshot(JSON.stringify({ value: saved, origin: comfy.origin, port: comfy.port }));
      if (showSuccess) setMessage("全局设置已保存并立即生效");
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "保存失败");
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function installNodes() {
    if (!settings.comfyuiRoot.trim()) {
      setError("请先填写ComfyUI文件夹");
      return;
    }
    if (!await save(false)) return;
    setBusy(true);
    setError(null);
    setMessage("正在安装并校验项目节点、H3 Motion Context 和 Turbo节点；请勿关闭应用...");
    setNodeInstallResult(null);
    try {
      const result = await installRequiredComfyNodes();
      setNodeInstallResult(result);
      if (result.succeeded) {
        setMessage("所需节点已安装并校验。请重启ComfyUI，然后点击测试连接");
      } else {
        const failed = result.steps.filter((step) => !step.succeeded).map((step) => step.label);
        setMessage(null);
        setError(`部分节点安装失败：${failed.join("、")}`);
      }
    } catch (cause) {
      setMessage(null);
      setError(`节点安装失败：${cause instanceof Error ? cause.message : "未知错误"}`);
    } finally {
      setBusy(false);
    }
  }

  async function testComfyUI() {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const [capabilities, models] = await Promise.all([getLocalWorkerCapabilities(), listComfyuiModels()]);
      setComfyModels(models);
      setWorkerBlockers(capabilities.execution_blockers);
      setMessage(
        `ComfyUI ${capabilities.inventory.version ?? "未知版本"} 连接正常 · `
        + `${capabilities.h3_node_ids.length} 个 H3 节点`,
      );
    } catch (cause) {
      setError(`ComfyUI 连接失败：${cause instanceof Error ? cause.message : "未知错误"}`);
    } finally {
      setBusy(false);
    }
  }

  async function refreshLlmModels() {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await listLlmModels();
      setLlmModels(result.models);
      setMessage(`已从 LLM 服务获取 ${result.count} 个模型，可在模型输入框中搜索选择`);
    } catch (cause) {
      setError(`模型列表获取失败：${cause instanceof Error ? cause.message : "未知错误"}`);
    } finally {
      setBusy(false);
    }
  }

  return <section className="workspace settings-workspace">
    <div className="section-heading">
      <div><h2>全局设置</h2><span>连接、模型服务与本机运行环境</span></div>
    </div>
    {error && <div className="error-banner inline-banner"><CircleAlert size={17} />{error}</div>}
    {message && <div className="validation-ok settings-message"><Check size={16} />{message}</div>}

    <div className="settings-section">
      <div className="settings-section-heading"><div><strong>ComfyUI Worker</strong><span>保存后新的探测和任务立即使用此连接</span></div><button className="secondary-button" disabled={busy} onClick={() => void testComfyUI()}><PlugZap size={16} />测试连接</button></div>
      <div className="settings-grid">
        <label className="field field-wide"><span>地址</span><input value={comfyOrigin} placeholder="http://127.0.0.1" onChange={(event) => setComfyOrigin(event.target.value)} /></label>
        <label className="field"><span>端口</span><input inputMode="numeric" value={comfyPort} onChange={(event) => setComfyPort(event.target.value)} /></label>
        <label className="field"><span>请求超时（秒）</span><input type="number" min="1" max="60" value={settings.requestTimeoutSeconds} onChange={(event) => setSettings({ ...settings, requestTimeoutSeconds: Number(event.target.value) })} /></label>
        <label className="field field-wide"><span>ComfyUI 文件夹</span><input value={settings.comfyuiRoot} placeholder="D:\\Comfy_new\\ComfyUI" onChange={(event) => { setSettings({ ...settings, comfyuiRoot: event.target.value }); setNodeInstallResult(null); }} /></label>
        <div className="node-install-row field-wide"><div><strong>安装视频生成所需节点</strong><span>安装项目专属节点、H3 Motion Context和官方Turbo节点；不安装SageAttention</span></div><button className="secondary-button" disabled={busy || !settings.comfyuiRoot.trim()} onClick={() => void installNodes()}><Blocks size={16} />{busy ? "处理中" : "保存路径并安装"}</button></div>
        {nodeInstallResult && <div className="node-install-results field-wide">{nodeInstallResult.steps.map((step) => <div className={step.succeeded ? "installed" : "failed"} key={step.component}>{step.succeeded ? <Check size={15} /> : <CircleAlert size={15} />}<span><strong>{step.label}</strong><small>{step.message}</small></span></div>)}</div>}
      </div>
    </div>

    <div className="settings-section">
      <div className="settings-section-heading"><div><strong>MiniMax H3 工作流</strong><span>模型选择器只显示文件名包含 minimax 的已安装模型</span></div><button className="secondary-button" disabled={busy} onClick={() => void testComfyUI()}><RefreshCw size={16} />刷新模型</button></div>
      <div className="settings-grid">
        <label className="field field-wide"><span>扩散模型</span><select value={settings.h3DiffusionModel} onChange={(event) => setSettings({ ...settings, h3DiffusionModel: event.target.value })}><option value={settings.h3DiffusionModel}>{settings.h3DiffusionModel}</option>{h3Models.diffusion.filter((item) => item !== settings.h3DiffusionModel).map((item) => <option value={item} key={item}>{item}</option>)}</select></label>
        <label className="field field-wide"><span>文本编码器</span><select value={settings.h3TextEncoder} onChange={(event) => setSettings({ ...settings, h3TextEncoder: event.target.value })}><option value={settings.h3TextEncoder}>{settings.h3TextEncoder}</option>{h3Models.textEncoders.filter((item) => item !== settings.h3TextEncoder).map((item) => <option value={item} key={item}>{item}</option>)}</select></label>
        <label className="field"><span>视频 VAE</span><select value={settings.h3VideoVae} onChange={(event) => setSettings({ ...settings, h3VideoVae: event.target.value })}><option value={settings.h3VideoVae}>{settings.h3VideoVae}</option>{h3Models.vaes.filter((item) => item !== settings.h3VideoVae).map((item) => <option value={item} key={item}>{item}</option>)}</select></label>
        <label className="field"><span>音频 VAE</span><select value={settings.h3AudioVae} onChange={(event) => setSettings({ ...settings, h3AudioVae: event.target.value })}><option value={settings.h3AudioVae}>{settings.h3AudioVae}</option>{h3Models.vaes.filter((item) => item !== settings.h3AudioVae).map((item) => <option value={item} key={item}>{item}</option>)}</select></label>
        <label className="field field-wide"><span>Turbo LoRA</span><select disabled={!settings.h3TurboEnabled} value={settings.h3TurboLora} onChange={(event) => setSettings({ ...settings, h3TurboLora: event.target.value })}><option value={settings.h3TurboLora}>{settings.h3TurboLora}</option>{h3Models.loras.filter((item) => item !== settings.h3TurboLora).map((item) => <option value={item} key={item}>{item}</option>)}</select></label>
        <label className="field"><span>采样步数</span><input type="number" min="4" max="50" value={settings.h3Steps} onChange={(event) => setSettings({ ...settings, h3Steps: Number(event.target.value) })} /></label>
        <label className="check-field"><input type="checkbox" checked={settings.h3TurboEnabled} onChange={(event) => setSettings({ ...settings, h3TurboEnabled: event.target.checked, h3Steps: event.target.checked ? 6 : 20 })} />启用官方 Turbo（可选）</label>
        <label className="check-field"><input type="checkbox" checked={settings.h3SageAttentionEnabled} onChange={(event) => setSettings({ ...settings, h3SageAttentionEnabled: event.target.checked })} />启用 SageAttention</label>
        <label className="check-field"><input type="checkbox" checked={settings.h3LowVram} onChange={(event) => setSettings({ ...settings, h3LowVram: event.target.checked })} />低显存模式</label>
      </div>
      <div className="h3-prerequisite" role="note">
        <CircleAlert size={18} />
        <div><strong>首次运行前必须安装并重启ComfyUI</strong><span>在上方填写ComfyUI文件夹并点击“保存路径并安装”。安装器只使用固定来源和校验版本，不安装SageAttention；官方H3基础节点由ComfyUI本体提供。</span>{workerBlockers.filter((item) => item.includes("AVG")).map((item) => <small key={item}>{item}</small>)}</div>
      </div>
      <small>为避免误选其他架构，扩散模型、编码器和 VAE 列表会过滤掉文件名中不含 minimax 的项目；Turbo LoRA 只显示 minimax_h3_turbo。列表来自当前 ComfyUI `/object_info`，刷新和保存都不会提交 GPU 任务。</small>
    </div>

    <div className="settings-section">
      <div className="settings-section-heading"><div><strong>LLM 服务</strong><span>OpenAI-compatible Chat Completions 接口</span></div><button className="secondary-button" disabled={busy || !settings.llmBaseUrl || !settings.llmApiKeyConfigured} onClick={() => void refreshLlmModels()}><RefreshCw size={16} />刷新模型列表</button></div>
      <div className="settings-grid">
        <label className="field field-wide"><span>API URL</span><input value={settings.llmBaseUrl} placeholder="https://example.com/v1" onChange={(event) => setSettings({ ...settings, llmBaseUrl: event.target.value })} /></label>
        <label className="field"><span>模型</span><input list="llm-model-options" value={settings.llmModel} placeholder="输入或从列表选择" onChange={(event) => setSettings({ ...settings, llmModel: event.target.value })} /><datalist id="llm-model-options">{llmModels.map((model) => <option value={model} key={model} />)}</datalist><small>{llmModels.length ? `${llmModels.length} 个服务端模型已载入` : "保存连接后刷新模型列表"}</small></label>
        <label className="field"><span>请求超时（秒）</span><input type="number" min="1" max="300" value={settings.llmTimeoutSeconds} onChange={(event) => setSettings({ ...settings, llmTimeoutSeconds: Number(event.target.value) })} /></label>
        <label className="check-field field-wide"><input type="checkbox" checked={settings.llmVideoCapable} onChange={(event) => setSettings({ ...settings, llmVideoCapable: event.target.checked })} />当前模型支持直接读取视频</label>
        <small className="field-wide">开启后审核会优先发送压缩视频；服务明确拒绝视频输入时会记录原因并自动回退到抽帧审核。</small>
        <label className="field field-wide"><span>API Key</span><div className="secret-input"><input type={showApiKey ? "text" : "password"} value={apiKey} disabled={clearApiKey} placeholder={settings.llmApiKeyConfigured ? "已保存，留空保持不变" : "尚未配置"} onChange={(event) => setApiKey(event.target.value)} /><button className="icon-button" title={showApiKey ? "隐藏密钥" : "显示密钥"} onClick={() => setShowApiKey(!showApiKey)}>{showApiKey ? <EyeOff size={17} /> : <Eye size={17} />}</button></div></label>
        <label className="check-field field-wide"><input type="checkbox" checked={clearApiKey} onChange={(event) => setClearApiKey(event.target.checked)} />清除已保存的 API Key</label>
        <label className="field field-wide"><span>网络代理（可选）</span><input value={settings.networkProxy} placeholder="mixed:10808" onChange={(event) => setSettings({ ...settings, networkProxy: event.target.value })} /></label>
      </div>
    </div>

    <div className={`settings-actions ${dirty ? "settings-actions-dirty" : ""}`}><span>{dirty ? "存在未保存修改" : "设置已保存"}</span><button className="primary-button" disabled={busy || !portValid || !dirty} onClick={() => void save()}><Save size={16} />{busy ? "处理中" : "保存全局设置"}</button></div>
  </section>;
}
