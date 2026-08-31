import { Bot, Check, CircleAlert, Maximize2, Minimize2, Pencil, RotateCcw, Send, Sparkles, Square, UserRound, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cancelProjectLlmOperation, getProjectWorkspace, listProjectLlmOperations, listProjectMemory, resumeProjectAgentOperation, streamProjectAgentProposal } from "./api";
import { applyAgentPatches } from "./agent-patches";
import type { AgentProposal, PipelineStageId, ProjectDraft, ProjectMemoryEvent } from "./types";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  operation: string;
  automatic: boolean;
  persisted: boolean;
};

type AgentStage = Exclude<PipelineStageId, "config" | "prompts">;

const stageScopes: Record<AgentStage, string[]> = {
  idea: ["/name", "/idea", "/targetDurationSeconds"],
  outline: ["/outline"],
  storyboard: ["/shots"],
  assets: ["/assetPlans"],
  generation: [],
  delivery: ["/postProcessing"],
};

const stageNames: Record<AgentStage, string> = {
  idea: "创意设定",
  outline: "故事大纲",
  storyboard: "电影分镜",
  assets: "素材规划",
  generation: "生成执行计划",
  delivery: "后处理与交付",
};

const automaticDraftInstructions: Partial<Record<AgentStage, string>> = {
  outline: "根据已批准的创意和项目时长，直接生成完整故事大纲。段落时长总和应接近项目目标时长。",
  storyboard: "根据当前项目内容直接生成完整电影分镜。每个分镜要包含标题、画面摘要、运镜和时长。只要连续画面需要跨多次 H3 执行继承同一动作、姿态、机位、运动方向、构图、光线、环境或声音状态，就必须把它表示为同一分镜下的 motionSegments 连续链，不得拆成互不依赖的普通分镜；只有明确切镜或连续性重置才能另建分镜。分段时明确列出每段 durationSeconds 和 summary，时长之和必须等于分镜时长，每段至少4秒，首段最多15秒，续段最多12秒。不要机械规划15+15；30秒可以是10+10+10。接缝优先放在密集信息或关键动作完成之后、人物运动与机位相对稳定处，并在 summary 中写清本段内容及交给下一段继承的结束状态。",
  assets: "根据已批准的创意、大纲和电影分镜，生成完整素材需求计划。先查看 project_assets 清单和随请求提供的图片：这些是已经上传并命名的真实素材，不要为已满足的职责重复创建需求；只规划仍缺少的角色、场景、道具和风格素材，不虚构素材文件。只在同一可辨识场景或物品会被至少两个分镜复用时，为它创建素材需求；只使用一次的场景或物品必须留在对应分镜描述中，由视频模型直接生成，不得创建图片素材。角色身份和项目级视觉风格不受此限制；已经上传的素材是项目事实，即使只使用一次也不得因此删除。shotIds 只列出叙事、身份、场景、物体或风格上确实需要该素材的镜头；允许为空，绝对不要为了覆盖全部素材而把不需要该素材的镜头加入 shotIds。scope=public 只表示素材可复用，不代表所有镜头自动引用；实际使用始终以 shotIds 为准。同一素材被多个镜头引用时只创建一个集中素材需求，不要复制需求。不得填写不存在的 shotId。每项需求必须独立决定图片 width、height 和构图方向，不得照搬视频分辨率；宽高使用8的倍数，总像素建议控制在1280×1280（1,638,400像素）左右，并将 resolutionSource 设为 ai。",
};

function needsAutomaticDraft(stage: AgentStage, project: ProjectDraft) {
  if (stage === "outline") return project.outline.length === 0;
  if (stage === "storyboard") return project.shots.length === 0;
  if (stage === "assets") return project.assetPlans.length === 0;
  return false;
}

function activeDialogBranch(events: ProjectMemoryEvent[], operations: string[]) {
  const relevant = events
    .filter((event) => operations.some((operation) => event.role === `dialog_${operation}_user` || event.role === `dialog_${operation}_assistant`))
    .reverse();
  const latest = relevant.at(-1);
  if (!latest) return [];
  const byId = new Map(relevant.map((event) => [event.event_id, event]));
  const branch: ProjectMemoryEvent[] = [];
  let cursor: ProjectMemoryEvent | undefined = latest;
  while (cursor) {
    branch.push(cursor);
    cursor = cursor.parent_event_id ? byId.get(cursor.parent_event_id) : undefined;
  }
  branch.reverse();
  return branch;
}

type Props = {
  stage: AgentStage;
  project: ProjectDraft;
  prepareProject: () => Promise<ProjectDraft | null>;
  applyProject: (project: ProjectDraft) => Promise<void>;
  acceptCommittedProject: (project: ProjectDraft) => void;
  onAutomaticGenerationChange?: (busy: boolean) => void;
};

export function ProjectAgentPanel({ stage, project, prepareProject, applyProject, acceptCommittedProject, onAutomaticGenerationChange }: Props) {
  const [conversations, setConversations] = useState<Partial<Record<PipelineStageId, ChatMessage[]>>>({});
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [automaticBusy, setAutomaticBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<{ proposal: AgentProposal; base: ProjectDraft } | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [streamPreview, setStreamPreview] = useState("");
  const [activeOperationId, setActiveOperationId] = useState<string | null>(null);
  const [retryDraft, setRetryDraft] = useState<{ messageId: string; parentEventId: string | null } | null>(null);
  const stoppedByUser = useRef(false);
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
      const restored = activeDialogBranch(events, operations)
        .map<ChatMessage>((event) => ({
          id: event.event_id,
          role: event.role.endsWith("_user") ? "user" : "assistant",
          text: event.role === `dialog_initialize_${stage}_user`
            ? `自动生成${stageNames[stage]}初稿`
            : event.content,
          operation: event.role.slice("dialog_".length).replace(/_(user|assistant)$/, ""),
          automatic: event.role.includes("initialize_"),
          persisted: true,
        }));
      setConversations((current) => ({ ...current, [stage]: restored }));
    }).catch((cause) => {
      if (!active) return;
      setError(`项目记忆读取失败：${cause instanceof Error ? cause.message : "未知错误"}`);
    });
    return () => { active = false; };
  }, [project.projectId, stage]);

  useEffect(() => {
    let active = true;
    const scopes = new Set([`initialize_${stage}`, `refine_${stage}`]);
    void listProjectLlmOperations(project.projectId).then((operations) => {
      if (!active) return;
      const running = [...operations].reverse().find((operation) => (
        operation.kind === "project_agent"
        && operation.state === "running"
        && scopes.has(operation.scope)
      ));
      if (!running) {
        return getProjectWorkspace(project.projectId).then((latest) => {
          if (active && latest.revision > project.revision) acceptCommittedProject(latest);
          return undefined;
        });
      }
      const automatic = running.scope.startsWith("initialize_");
      setBusy(true);
      setAutomaticBusy(automatic);
      setActiveOperationId(running.operation_id);
      setStreamPreview(running.text);
      let resumed = false;
      return resumeProjectAgentOperation(project.projectId, running.operation_id, (delta) => {
        if (!active) return;
        if (!resumed) {
          resumed = true;
          setStreamPreview(delta.slice(-1800));
        } else {
          setStreamPreview((current) => (current + delta).slice(-1800));
        }
      });
    }).then((result) => {
      if (!active || !result) return;
      if (result.committed_workspace) acceptCommittedProject(result.committed_workspace);
      addMessage({
        id: result.assistant_event_id || crypto.randomUUID(),
        role: "assistant",
        text: `${result.proposal.rationale}\n\n后台生成已完成并保存，界面已恢复到最新项目修订。`,
        operation: `refine_${stage}`,
        automatic: false,
        persisted: true,
      });
    }).catch((cause) => {
      if (active && !stoppedByUser.current) setError(cause instanceof Error ? cause.message : "恢复 LLM 生成状态失败");
    }).finally(() => {
      if (active) {
        setBusy(false);
        setAutomaticBusy(false);
        setActiveOperationId(null);
        setStreamPreview("");
      }
    });
    return () => { active = false; };
  }, [acceptCommittedProject, project.projectId, stage]);

  function addMessage(message: ChatMessage) {
    setConversations((current) => ({
      ...current,
      [stage]: [...(current[stage] ?? []), message],
    }));
  }

  async function runInstruction(
    text: string,
    options: { operation: string; automatic: boolean; parentEventId?: string | null },
  ) {
    if (!text.trim()) return;
    setError(null);
    setStreamPreview("");
    setPending(null);
    const operationId = crypto.randomUUID();
    stoppedByUser.current = false;
    const parentEventId = options.parentEventId === undefined
      ? [...messages].reverse().find((message) => message.persisted)?.id ?? null
      : options.parentEventId;
    addMessage({
      id: `${operationId}:user`,
      role: "user",
      text: options.automatic ? `自动生成${stageNames[stage]}初稿` : text,
      operation: options.operation,
      automatic: options.automatic,
      persisted: true,
    });
    setBusy(true);
    if (options.automatic) setAutomaticBusy(true);
    try {
      const saved = await prepareProject();
      if (!saved) throw new Error("项目修订保存失败，无法交给 LLM 修改");
      const lockedPaths = saved.shots
        .map((shot, index) => shot.locked ? `/shots/${index}` : null)
        .filter((value): value is string => Boolean(value));
      setActiveOperationId(operationId);
      const result = await streamProjectAgentProposal(saved.projectId, {
        operationId,
        operation: options.operation,
        instruction: `你是本视频项目的专属负责人。当前阶段是“${stageNames[stage]}”。请结合完整项目记忆，${options.automatic ? "直接生成并返回将由系统校验后立即应用的完整初稿" : "根据用户消息直接修改对应内容；有效修改会由系统校验后立即应用，无需用户再次审批"}：${text}`,
        displayInstruction: options.automatic ? `自动生成${stageNames[stage]}初稿` : text,
        conversationParentEventId: parentEventId,
        allowedPaths: stageScopes[stage],
        lockedPaths,
      }, (delta) => setStreamPreview((current) => (current + delta).slice(-1800)));
      if (!result.proposal.patches.length) {
        addMessage({
          id: result.assistant_event_id || crypto.randomUUID(),
          role: "assistant",
          text: `${result.proposal.rationale}\n\n本次没有返回可应用的字段修改，项目内容未发生变化。`,
          operation: options.operation,
          automatic: false,
          persisted: true,
        });
        if (options.automatic) {
          throw new Error(`LLM 未生成${stageNames[stage]}的可应用内容，请重试或补充要求`);
        }
        return;
      }

      if (result.committed_workspace) {
        acceptCommittedProject(result.committed_workspace);
      } else {
        const changed = applyAgentPatches(saved, result.proposal.patches);
        try {
          await applyProject(changed);
        } catch (cause) {
          setPending({ proposal: result.proposal, base: saved });
          addMessage({
            id: result.assistant_event_id || crypto.randomUUID(),
            role: "assistant",
            text: "修改方案已经生成，但尚未保存到项目。请查看错误后重试应用。",
            operation: options.operation,
            automatic: false,
            persisted: true,
          });
          throw cause;
        }
      }
      addMessage({
        id: result.assistant_event_id || crypto.randomUUID(),
        role: "assistant",
        text: `${result.proposal.rationale}\n\n已应用并保存 ${result.proposal.patches.length} 项修改，界面已同步到最新项目修订。`,
        operation: options.operation,
        automatic: false,
        persisted: true,
      });
    } catch (cause) {
      if (!stoppedByUser.current) setError(cause instanceof Error ? cause.message : "LLM 生成失败");
    } finally {
      setStreamPreview("");
      setBusy(false);
      setActiveOperationId(null);
      if (options.automatic) setAutomaticBusy(false);
    }
  }

  async function stopGeneration() {
    if (!activeOperationId) return;
    stoppedByUser.current = true;
    setError(null);
    try {
      await cancelProjectLlmOperation(project.projectId, activeOperationId);
    } catch (cause) {
      stoppedByUser.current = false;
      setError(cause instanceof Error ? cause.message : "停止生成失败");
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    if (retryDraft) {
      const parentEventId = retryDraft.parentEventId;
      setConversations((current) => ({
        ...current,
        [stage]: (current[stage] ?? []).filter((message) => message.id !== retryDraft.messageId),
      }));
      setRetryDraft(null);
      await runInstruction(text, { operation: `refine_${stage}`, automatic: false, parentEventId });
      return;
    }
    await runInstruction(text, { operation: `refine_${stage}`, automatic: false });
  }

  function editMessage(message: ChatMessage, index: number) {
    const previous = messages.slice(0, index);
    setConversations((current) => ({ ...current, [stage]: [...previous, message] }));
    setRetryDraft({ messageId: message.id, parentEventId: [...previous].reverse().find((item) => item.persisted)?.id ?? null });
    setInput(message.automatic ? automaticDraftInstructions[stage] ?? message.text : message.text);
    setError(null);
    setPending(null);
  }

  function retryMessage(message: ChatMessage, index: number) {
    const previous = messages.slice(0, index);
    const text = message.automatic ? automaticDraftInstructions[stage] ?? message.text : message.text;
    setConversations((current) => ({ ...current, [stage]: previous }));
    setRetryDraft(null);
    setInput("");
    void runInstruction(text, {
      operation: message.operation,
      automatic: message.automatic,
      parentEventId: [...previous].reverse().find((item) => item.persisted)?.id ?? null,
    });
  }

  async function apply() {
    if (!pending) return;
    setBusy(true);
    setError(null);
    try {
      await applyProject(applyAgentPatches(pending.base, pending.proposal.patches));
      setPending(null);
      addMessage({ id: crypto.randomUUID(), role: "assistant", text: "修改已应用并保存，界面已同步到最新项目修订。", operation: `refine_${stage}`, automatic: false, persisted: false });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "应用修改失败");
    } finally {
      setBusy(false);
    }
  }

  return <div className={`agent-panel ${expanded ? "expanded" : ""}`}>
    <div className="agent-heading"><div className="agent-avatar"><Bot size={18} /></div><div><strong>项目负责人</strong><span>已读取项目记忆 · 当前可修改：{stageNames[stage]}</span></div><span className="agent-online"><i />LLM</span><button className="icon-button" title={expanded ? "收起对话" : "展开对话"} onClick={() => setExpanded(!expanded)}>{expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}</button></div>
    <div className="agent-thread">
      {!messages.length && <div className="agent-empty"><Sparkles size={18} /><span>可以讨论方案、询问意见，或要求生成和修改本阶段内容。</span>{automaticDraftInstructions[stage] && needsAutomaticDraft(stage, project) ? <button className="secondary-button" disabled={busy} onClick={() => void runInstruction(automaticDraftInstructions[stage]!, { operation: `initialize_${stage}`, automatic: true })}>生成初稿</button> : null}</div>}
      {messages.map((message, index) => <div className={`agent-message ${message.role}`} key={message.id}><span>{message.role === "user" ? <UserRound size={14} /> : <Bot size={14} />}</span><div className="agent-message-content"><p>{message.text}</p>{message.role === "user" && !busy && <div className="agent-message-actions"><button className="icon-button" title="编辑并回到此消息" onClick={() => editMessage(message, index)}><Pencil size={13} /></button><button className="icon-button" title="从此消息重新生成" onClick={() => retryMessage(message, index)}><RotateCcw size={13} /></button></div>}</div></div>)}
      {busy && <div className="agent-message assistant streaming"><span><Bot size={14} /></span><div className="agent-message-content"><p>{streamPreview || (automaticBusy ? `正在生成${stageNames[stage]}初稿...` : "正在结合项目记忆分析...")}</p><button className="secondary-button agent-stop-generation" disabled={!activeOperationId} onClick={() => void stopGeneration()}><Square size={13} />停止生成</button></div></div>}
    </div>
    {pending && <div className="agent-proposal">
      <div><strong>待应用修改</strong><span>{pending.proposal.patches.length} 个字段操作</span></div>
      <ul>{pending.proposal.patches.map((patch, index) => <li key={`${patch.path}-${index}`}><code>{patch.op}</code><span>{patch.path}</span></li>)}</ul>
      {pending.proposal.warnings.map((warning) => <p className="agent-warning" key={warning}><CircleAlert size={14} />{warning}</p>)}
      <div className="agent-proposal-actions"><button className="secondary-button" disabled={busy} onClick={() => setPending(null)}><X size={15} />放弃</button><button className="primary-button" disabled={busy} onClick={() => void apply()}><Check size={15} />应用修改</button></div>
    </div>}
    {error && <div className="error-banner agent-error" role="alert"><CircleAlert size={16} /><strong>项目负责人未完成操作</strong><span>{error}</span></div>}
    <div className="agent-composer"><textarea rows={2} value={input} placeholder={retryDraft ? "编辑消息后点击重试；编辑不会自动发送" : `与项目负责人讨论${stageNames[stage]}；有效修改会自动保存`} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} /><button className="icon-button" title={retryDraft ? "重试" : "发送"} disabled={busy || !input.trim()} onClick={() => void send()}>{retryDraft ? <RotateCcw size={17} /> : <Send size={17} />}</button></div>
  </div>;
}
