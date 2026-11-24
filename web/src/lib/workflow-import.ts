export const PLAIN_COMFYUI_WORKFLOW_NOTICE =
  '检测到普通 ComfyUI 工作流 JSON（不含 Studio manifest / object_info）。如需检测并下载所需模型，请前往「模型库 › 检测工作流模型」；创作模式工作流需导入包含 manifest、api_graph、object_info 的完整工作流包，或含元数据的 PNG 原图。';

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function isVisualWorkflow(value: unknown): boolean {
  if (!isRecord(value) || !Array.isArray(value.nodes)) return false;
  if (value.links !== undefined && !Array.isArray(value.links)) return false;
  return value.nodes.every(
    (node) =>
      isRecord(node)
      && (typeof node.id === 'number' || typeof node.id === 'string')
      && typeof node.type === 'string',
  );
}

function isApiPromptGraph(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const entries = Object.values(value);
  if (!entries.length) return false;
  return entries.every(
    (node) => isRecord(node) && typeof node.class_type === 'string' && isRecord(node.inputs),
  );
}

export type ClassifiedWorkflowJson =
  | { kind: 'studio_bundle'; bundle: Record<string, unknown> }
  | { kind: 'plain_comfyui'; message: string }
  | { kind: 'invalid'; message: string };

/** Classify a JSON value submitted to the Workflows page under the conservative import policy. */
export function classifyWorkflowJson(value: unknown): ClassifiedWorkflowJson {
  if (!isRecord(value)) {
    return { kind: 'invalid', message: 'JSON 根节点必须是对象' };
  }
  const source = isRecord(value.workflow) ? value.workflow : value;
  if (
    isRecord(source.image_extraction)
    || (isRecord(source.manifest) && isRecord(source.api_graph) && isRecord(source.object_info))
  ) {
    return { kind: 'studio_bundle', bundle: value };
  }
  if (
    isVisualWorkflow(value)
    || isApiPromptGraph(value)
    || (value.workflow !== undefined && isVisualWorkflow(value.workflow))
    || (value.prompt !== undefined && isApiPromptGraph(value.prompt))
  ) {
    return { kind: 'plain_comfyui', message: PLAIN_COMFYUI_WORKFLOW_NOTICE };
  }
  return { kind: 'invalid', message: 'JSON 必须包含 manifest、api_graph、object_info' };
}

/** Validate and clean a user-entered workflow URL before sending it to the backend. */
export function validateWorkflowUrlInput(raw: string): { ok: true; url: string } | { ok: false; error: string } {
  const trimmed = raw.trim();
  if (!trimmed) {
    return { ok: false, error: '请输入 HTTPS 工作流或图片链接' };
  }
  if (trimmed.length > 4096) {
    return { ok: false, error: '链接长度超过 4096 字符限制' };
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return { ok: false, error: '请输入有效的 HTTPS 链接（例如 https://...）' };
  }
  if (parsed.protocol !== 'https:') {
    return { ok: false, error: '仅支持 HTTPS 公网链接' };
  }
  if (parsed.username || parsed.password) {
    return { ok: false, error: '链接不能包含用户名或密码' };
  }
  if (parsed.port && parsed.port !== '443') {
    return { ok: false, error: '仅支持标准 443 端口的 HTTPS 链接' };
  }
  return { ok: true, url: trimmed };
}
