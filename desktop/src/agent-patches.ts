import type { AgentPatch, ProjectDraft } from "./types";
import { normalizePatchedProject } from "./project-store";

function pointerParts(path: string): string[] {
  if (!path.startsWith("/")) throw new Error(`无效修改路径：${path}`);
  return path
    .slice(1)
    .split("/")
    .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"));
}

export function applyAgentPatches(project: ProjectDraft, patches: AgentPatch[]): ProjectDraft {
  const document = structuredClone(project) as unknown;
  for (const patch of patches) {
    const parts = pointerParts(patch.path);
    if (!parts.length) throw new Error("不允许替换整个项目");
    let parent = document as Record<string, unknown> | unknown[];
    for (const part of parts.slice(0, -1)) {
      const next = Array.isArray(parent) ? parent[Number(part)] : parent[part];
      if (!next || typeof next !== "object") throw new Error(`修改路径不存在：${patch.path}`);
      parent = next as Record<string, unknown> | unknown[];
    }
    const key = parts.at(-1)!;
    if (Array.isArray(parent)) {
      if (patch.op === "add" && key === "-") parent.push(patch.value);
      else {
        const index = Number(key);
        if (!Number.isInteger(index) || index < 0 || index >= parent.length) {
          throw new Error(`数组修改路径不存在：${patch.path}`);
        }
        if (patch.op === "remove") parent.splice(index, 1);
        else if (patch.op === "add") parent.splice(index, 0, patch.value);
        else parent[index] = patch.value;
      }
    } else if (patch.op === "remove") {
      if (!(key in parent)) throw new Error(`修改路径不存在：${patch.path}`);
      delete parent[key];
    } else {
      parent[key] = patch.value;
    }
  }
  return normalizePatchedProject(document);
}
