import { Check, CircleAlert, Download, LoaderCircle, Settings2, X } from "lucide-react";
import { useState } from "react";
import {
  assembleDefaultH3Harness,
  getSetupStatus,
  installDefaultH3HarnessSource,
} from "./api";
import type { SetupStatus } from "./types";

type Props = {
  status: SetupStatus;
  onStatus: (status: SetupStatus) => void;
  onOpenSettings: () => void;
  onDismiss: () => void;
};

export function SetupWizard({ status, onStatus, onOpenSettings, onDismiss }: Props) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const installHarness = async () => {
    setBusy(true);
    setMessage("正在下载固定版本的社区与官方Harness文件...");
    try {
      await installDefaultH3HarnessSource("community");
      setMessage("社区来源已校验，正在获取官方提示词规范...");
      await installDefaultH3HarnessSource("official");
      setMessage("正在组装并批准H3 Harness...");
      await assembleDefaultH3Harness();
      const refreshed = await getSetupStatus();
      onStatus(refreshed);
      setMessage("H3 Harness已安装并批准");
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Harness安装失败");
    } finally {
      setBusy(false);
    }
  };

  const checks = [
    ["应用数据目录", status.data_root_writable, status.data_root],
    ["FFmpeg / ffprobe", status.ffmpeg_ready && status.ffprobe_ready, "随安装包提供并在执行前复检"],
    ["LLM服务", status.llm_configured, status.llm_api_key_configured ? "模型和凭据已配置" : "请配置URL、模型与凭据"],
    ["ComfyUI", status.comfyui_online, status.comfyui_online ? "服务在线" : "请启动ComfyUI并检查端口"],
    ["H3 Harness", status.h3_harness_ready, status.h3_harness_ready ? "固定来源已批准" : "必须联网安装固定来源"],
  ] as const;

  return <div className="setup-backdrop" role="dialog" aria-modal="true" aria-label="首次运行设置">
    <section className="setup-wizard">
      <header><div><strong>首次运行检查</strong><span>完成这些项目后才能提交视频生成任务</span></div><button className="icon-button" title="稍后设置" onClick={onDismiss}><X size={18} /></button></header>
      <div className="setup-checks">{checks.map(([label, ready, detail]) => <div className={ready ? "ready" : "blocked"} key={label}>{ready ? <Check size={17} /> : <CircleAlert size={17} />}<div><strong>{label}</strong><span>{detail}</span></div></div>)}</div>
      {message && <div className="setup-message" role="status">{busy && <LoaderCircle className="spin" size={16} />}{message}</div>}
      <footer><button className="secondary-button" onClick={onOpenSettings}><Settings2 size={15} />打开全局设置</button>{!status.h3_harness_ready && <button className="primary-button" disabled={busy} onClick={() => void installHarness()}><Download size={15} />安装H3 Harness</button>}<button className="secondary-button" disabled={!status.ready} onClick={onDismiss}><Check size={15} />开始使用</button></footer>
    </section>
  </div>;
}
