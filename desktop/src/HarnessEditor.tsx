import { Check, CircleAlert, Save } from "lucide-react";
import { useEffect, useState } from "react";
import {
  listHarnessRevisions,
  registerHarness,
  registerHarnessRevision,
} from "./api";
import type { HarnessRevision } from "./types";

type Props = {
  workflowId: string;
  workflowRevision: number;
  workflowName: string;
};

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${stable(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

async function sha256(value: string) {
  const data = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(data), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function HarnessEditor({ workflowId, workflowRevision, workflowName }: Props) {
  const harnessId = `image-harness:${workflowId}`;
  const [markdown, setMarkdown] = useState(
    "# 图片模型补充指导\n\n请结合项目主管随后提供的完整需求，编写适合此图片模型和工作流的详细中文提示词。",
  );
  const [approval, setApproval] = useState<"draft" | "approved">("draft");
  const [revision, setRevision] = useState(1);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  useEffect(() => {
    void listHarnessRevisions(harnessId)
      .then((items) => {
        const latest = items.at(-1);
        if (latest) {
          setMarkdown(latest.markdown);
          setApproval(latest.approval === "approved" ? "approved" : "draft");
          setRevision(latest.revision + 1);
        }
      })
      .catch(() => undefined);
  }, [harnessId]);

  async function save() {
    setBusy(true);
    setMessage(null);
    try {
      await registerHarness({
        schema_version: "1.0",
        harness_id: harnessId,
        name: `${workflowName} Harness`,
        purpose: "image_prompting",
        workflow_template_id: workflowId,
        sources: [],
      });
      const inputSchema = { type: "object" };
      const outputSchema = { type: "object" };
      const content = {
        markdown,
        input_schema: inputSchema,
        output_schema: outputSchema,
        workflow_template_id: workflowId,
        workflow_revision: workflowRevision,
      };
      const item: HarnessRevision = {
        schema_version: "1.0",
        harness_id: harnessId,
        revision,
        markdown,
        input_schema: inputSchema,
        output_schema: outputSchema,
        content_sha256: await sha256(stable(content)),
        approval,
        workflow_template_id: workflowId,
        workflow_revision: workflowRevision,
        created_at: new Date().toISOString(),
      };
      await registerHarnessRevision(item);
      setRevision((value) => value + 1);
      setMessage(approval === "approved" ? "已批准并绑定工作流" : "草稿已登记");
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Harness 保存失败");
    } finally {
      setBusy(false);
    }
  }

  return <div className="harness-editor">
    <div className="harness-heading">
      <div><strong>图片 Harness</strong><span>{workflowName} · 下一修订 R{revision}</span></div>
      <select value={approval} onChange={(event) => setApproval(event.target.value as "draft" | "approved")}><option value="draft">保存草稿</option><option value="approved">批准使用</option></select>
      <button className="primary-button" disabled={busy || !markdown.trim()} onClick={() => void save()}><Save size={15} />保存修订</button>
    </div>
    <textarea value={markdown} onChange={(event) => setMarkdown(event.target.value)} spellCheck={false} />
    <div className={message?.includes("失败") ? "harness-message error" : "harness-message"}>{message ? <CircleAlert size={14} /> : <Check size={14} />}{message ?? "无需填写变量；项目主管会将完整需求和参考图附在这段指导之后"}</div>
  </div>;
}
