import { Bot, Check, CircleAlert, Maximize2, Minimize2, Send, Sparkles, UserRound, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { listProjectMemory, streamProjectAgentProposal } from "./api";
import { applyAgentPatches } from "./agent-patches";
import type { AgentProposal, PipelineStageId, ProjectDraft } from "./types";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
};

type AgentStage = Exclude<PipelineStageId, "config" | "prompts">;

const stageScopes: Record<AgentStage, string[]> = {
  idea: ["/name", "/idea", "/targetDurationSeconds"],
  outline: ["/outline"],
  storyboard: ["/shots"],
  assets: ["/assetPlans"],
  generation: [],
  review: ["/reviewNotes"],
  delivery: ["/postProcessing"],
};

const stageNames: Record<AgentStage, string> = {
  idea: "创意设定",
  outline: "故事大纲",
  storyboard: "电影分镜",
  assets: "素材规划",
  generation: "生成执行计划",
  review: "审核标准",
  delivery: "后处理与交付",
};

const automaticDraftInstructions: Partial<Record<AgentStage, string>> = {
  outline: "根据已批准的创意和项目时长，直接生成完整故事大纲。段落时长总和应接近项目目标时长。",
  storyboard: "根据已批准的大纲，直接生成完整电影分镜。每个分镜要包含标题、画面摘要、运镜和时长；需要超过15秒的连续镜头可以保留为一个电影分镜。系统会为长镜头分别编写提示词并用 Motion Context 拼接；续段的15秒模型预算中必须预留至少2秒继承上一段末尾潜空间并在输出时裁掉，所以不要机械规划15+15。30秒可以是10+10+10。请优先建议在密集信息或关键动作完成之后、人物运动与机位相对稳定且不易暴露接缝的位置分段。",
  assets: "根据已批准的创意、大纲和电影分镜，生成完整素材需求计划。先查看 project_assets 清单和随请求提供的图片：这些是已经上传并命名的真实素材，不要为已满足的职责重复创建需求；只规划仍缺少的角色、场景、道具和风格素材，不虚构素材文件。每项需求必须独立决定图片 width、height 和构图方向，不得照搬视频分辨率；宽高使用8的倍数，总像素建议控制在1280×1280（1,638,400像素）左右，并将 resolutionSource 设为 ai。",
  review: "根据项目创意、分镜、连续镜头和音频策略，直接生成可执行的项目审核标准。",
};

function needsAutomaticDraft(stage: AgentStage, project: ProjectDraft) {
  if (stage === "outline") return project.outline.length === 0;
  if (stage === "storyboard") return project.shots.length === 0;
  if (stage === "assets") return project.assetPlans.length === 0;
  if (stage === "review") return !project.reviewNotes.trim();
  return false;
}

type Props = {
  stage: AgentStage;
  project: ProjectDraft;
  prepareProject: () => Promise<ProjectDraft | null>;
  applyProject: (project: ProjectDraft) => Promise<void>;
  onAutomaticGenerationChange?: (busy: boolean) => void;
};

export function ProjectAgentPanel({ stage, project, prepareProject, applyProject, onAutomaticGenerationChange }: Props) {
  const [conversations, setConversations] = useState<Partial<Record<PipelineStageId, ChatMessage[]>>>({});
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [automaticBusy, setAutomaticBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<{ proposal: AgentProposal; base: ProjectDraft } | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [streamPreview, setStreamPreview] = useState("");
  const automaticStarted = useRef(new Set<string>());
  const messages = conversations[stage] ?? [];

  useEffect(() => {
    onAutomaticGenerationChange?.(automaticBusy);
    return () => onAutomaticGenerationChange?.(false);
  }, [automaticBusy, onAutomaticGenerationChange]);

  useEffect(() => {
    let active = true;
    const operations = [`refine_${stage}`, `initialize_${stage}`];
    void listProjectMemory(project.projectId).then((events) => {
      if (!active) return;
      const restored = events
        .filter((event) => operations.some((operation) => event.role === `dialog_${operation}_user` || event.role === `dialog_${operation}_assistant`))
        .reverse()
        .map<ChatMessage>((event) => ({
          id: event.event_id,
          role: event.role.endsWith("_user") ? "user" : "assistant",
          text: event.content,
        }));
      setConversations((current) => ({ ...current, [stage]: restored }));
      const automaticInstruction = automaticDraftInstructions[stage];
      const automaticKey = `${project.projectId}:${project.revision}:${stage}`;
      if (automaticInstruction && needsAutomaticDraft(stage, project) && !automaticStarted.current.has(automaticKey)) {
        automaticStarted.current.add(automaticKey);
        void runInstruction(automaticInstruction, {
          operation: `initialize_${stage}`,
          automatic: true,
        });
      }
    }).catch((cause) => {
      if (!active) return;
      setError(`项目记忆读取失败，将继续生成新初稿：${cause instanceof Error ? cause.message : "未知错误"}`);
      const automaticInstruction = automaticDraftInstructions[stage];
      const automaticKey = `${project.projectId}:${project.revision}:${stage}`;
      if (automaticInstruction && needsAutomaticDraft(stage, project) && !automaticStarted.current.has(automaticKey)) {
        automaticStarted.current.add(automaticKey);
        void runInstruction(automaticInstruction, { operation: `initialize_${stage}`, automatic: true });
      }
    });
    return () => { active = false; };
  }, [project.projectId, stage]);

  function addMessage(message: ChatMessage) {
    setConversations((current) => ({
      ...current,
      [stage]: [...(current[stage] ?? []), message],
    }));
  }

  async function runInstruction(
    text: string,
    options: { operation: string; automatic: boolean },
  ) {
    if (!text.trim()) return;
    setError(null);
    setStreamPreview("");
    setPending(null);
    addMessage({
      id: crypto.randomUUID(),
      role: "user",
      text: options.automatic ? `自动生成${stageNames[stage]}初稿` : text,
    });
    setBusy(true);
    if (options.automatic) setAutomaticBusy(true);
    try {
      const saved = await prepareProject();
      if (!saved) throw new Error("项目修订保存失败，无法交给 LLM 修改");
      const lockedPaths = saved.shots
        .map((shot, index) => shot.locked ? `/shots/${index}` : null)
        .filter((value): value is string => Boolean(value));
      const result = await streamProjectAgentProposal(saved.projectId, {
        operationId: crypto.randomUUID(),
        operation: options.operation,
        instruction: `你是本视频项目的专属负责人。当前阶段是“${stageNames[stage]}”。请结合完整项目记忆，${options.automatic ? "直接生成并返回将由系统校验后立即应用的完整初稿" : "根据用户消息直接修改对应内容；有效修改会由系统校验后立即应用，无需用户再次审批"}：${text}`,
        allowedPaths: stageScopes[stage],
        lockedPaths,
      }, (delta) => setStreamPreview((current) => (current + delta).slice(-1800)));
      if (!result.proposal.patches.length) {
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          text: `${result.proposal.rationale}\n\n本次没有返回可应用的字段修改，项目内容未发生变化。`,
        });
        if (options.automatic) {
          throw new Error(`LLM 未生成${stageNames[stage]}的可应用内容，请重试或补充要求`);
        }
        return;
      }

      const changed = applyAgentPatches(saved, result.proposal.patches);
      try {
        await applyProject(changed);
      } catch (cause) {
        // Keep the proposal available when persistence or refresh fails so the user can retry it.
        setPending({ proposal: result.proposal, base: saved });
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          text: "修改方案已经生成，但尚未保存到项目。请查看错误后重试应用。",
        });
        throw cause;
      }
      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        text: `${result.proposal.rationale}\n\n已应用并保存 ${result.proposal.patches.length} 项修改，界面已同步到最新项目修订。`,
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "LLM 生成失败");
    } finally {
      setStreamPreview("");
      setBusy(false);
      if (options.automatic) setAutomaticBusy(false);
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    await runInstruction(text, { operation: `refine_${stage}`, automatic: false });
  }

  async function apply() {
    if (!pending) return;
    setBusy(true);
    setError(null);
    try {
      await applyProject(applyAgentPatches(pending.base, pending.proposal.patches));
      setPending(null);
      addMessage({ id: crypto.randomUUID(), role: "assistant", text: "修改已应用并保存，界面已同步到最新项目修订。" });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "应用修改失败");
    } finally {
      setBusy(false);
    }
  }

  return <div className={`agent-panel ${expanded ? "expanded" : ""}`}>
    <div className="agent-heading"><div className="agent-avatar"><Bot size={18} /></div><div><strong>项目负责人</strong><span>已读取项目记忆 · 当前可修改：{stageNames[stage]}</span></div><span className="agent-online"><i />LLM</span><button className="icon-button" title={expanded ? "收起对话" : "展开对话"} onClick={() => setExpanded(!expanded)}>{expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}</button></div>
    <div className="agent-thread">
      {!messages.length && <div className="agent-empty"><Sparkles size={18} /><span>可以讨论方案、询问意见，或要求生成和修改本阶段内容。</span></div>}
      {messages.map((message) => <div className={`agent-message ${message.role}`} key={message.id}><span>{message.role === "user" ? <UserRound size={14} /> : <Bot size={14} />}</span><p>{message.text}</p></div>)}
      {busy && <div className="agent-message assistant streaming"><span><Bot size={14} /></span><p>{streamPreview || (automaticBusy ? `正在生成${stageNames[stage]}初稿...` : "正在结合项目记忆分析...")}</p></div>}
    </div>
    {pending && <div className="agent-proposal">
      <div><strong>待应用修改</strong><span>{pending.proposal.patches.length} 个字段操作</span></div>
      <ul>{pending.proposal.patches.map((patch, index) => <li key={`${patch.path}-${index}`}><code>{patch.op}</code><span>{patch.path}</span></li>)}</ul>
      {pending.proposal.warnings.map((warning) => <p className="agent-warning" key={warning}><CircleAlert size={14} />{warning}</p>)}
      <div className="agent-proposal-actions"><button className="secondary-button" disabled={busy} onClick={() => setPending(null)}><X size={15} />放弃</button><button className="primary-button" disabled={busy} onClick={() => void apply()}><Check size={15} />应用修改</button></div>
    </div>}
    {error && <div className="error-banner agent-error" role="alert"><CircleAlert size={16} /><strong>项目负责人未完成操作</strong><span>{error}</span></div>}
    <div className="agent-composer"><textarea rows={2} value={input} placeholder={`与项目负责人讨论${stageNames[stage]}；有效修改会自动保存`} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} /><button className="icon-button" title="发送" disabled={busy || !input.trim()} onClick={() => void send()}><Send size={17} /></button></div>
  </div>;
}
