import type { WorkflowExportSummary, WorkflowVersionConflictDetail } from './gpu-api';

export const PLAIN_COMFYUI_WORKFLOW_NOTICE =
  '检测到普通 ComfyUI 工作流 JSON（不含 Studio manifest / object_info）。如需检测并下载所需模型，请前往「模型库 › 检测工作流模型」；创作模式工作流需导入包含 manifest、api_graph、object_info 的完整工作流包，或含元数据的 PNG 原图。';

const WORKFLOW_KEY_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/;
const SEMVER_RE = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;

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

export function isValidWorkflowKey(value: string): boolean {
  return WORKFLOW_KEY_RE.test(value.trim());
}

export function isValidSemver(value: string): boolean {
  return SEMVER_RE.test(value.trim());
}

/** Extract structured conflict detail from a WORKFLOW_VERSION_CONFLICT ApiError-like object. */
export function parseWorkflowVersionConflict(error: unknown): WorkflowVersionConflictDetail | null {
  if (!error || typeof error !== 'object') return null;
  const candidate = error as { code?: unknown; detail?: unknown };
  if (candidate.code !== 'WORKFLOW_VERSION_CONFLICT' || !isRecord(candidate.detail)) {
    return null;
  }
  const detail = candidate.detail;
  if (
    typeof detail.workflow_key !== 'string'
    || typeof detail.workflow_version !== 'string'
    || typeof detail.suggested_version !== 'string'
    || typeof detail.suggested_workflow_key !== 'string'
  ) {
    return null;
  }
  return {
    workflow_key: detail.workflow_key,
    workflow_version: detail.workflow_version,
    existing_state: typeof detail.existing_state === 'string' ? detail.existing_state : undefined,
    existing_versions: Array.isArray(detail.existing_versions)
      ? detail.existing_versions.filter((item): item is string => typeof item === 'string')
      : [],
    suggested_version: detail.suggested_version,
    suggested_workflow_key: detail.suggested_workflow_key,
  };
}

/** Immutably apply a user-chosen workflow_key and SemVer version onto a Studio bundle or wrapper. */
export function applyBundleIdentityOverride(
  bundle: Record<string, unknown>,
  workflowKey: string,
  version: string,
): Record<string, unknown> {
  const nextKey = workflowKey.trim();
  const nextVersion = version.trim();
  const cloned = JSON.parse(JSON.stringify(bundle)) as Record<string, unknown>;
  const target = isRecord(cloned.workflow) ? cloned.workflow : cloned;
  target.workflow_key = nextKey;
  if ('key' in target) {
    target.key = nextKey;
  }
  const manifest = isRecord(target.manifest) ? target.manifest : {};
  manifest.version = nextVersion;
  if (typeof manifest.workflow_key === 'string') {
    manifest.workflow_key = nextKey;
  }
  target.manifest = manifest;
  return cloned;
}

export function formatExportSummaryMessage(summary: WorkflowExportSummary): string {
  const parts = [`已导出「${summary.workflow_key}」v${summary.workflow_version}`];
  if (summary.civitai_weights > 0 || summary.manual_rebind_weights > 0) {
    const notes: string[] = [];
    if (summary.civitai_weights > 0) notes.push(`${summary.civitai_weights} 个 Civitai 来源`);
    if (summary.manual_rebind_weights > 0) notes.push(`${summary.manual_rebind_weights} 个需重绑模型`);
    parts.push(`含 ${notes.join('、')}（已移除个人 Drive ID 并保留模型名称与 SHA-256）`);
  } else if (summary.stripped_drive_ids > 0) {
    parts.push(`已自动还原为公开来源并移除 ${summary.stripped_drive_ids} 个个人 Drive ID`);
  }
  return parts.join('，');
}
