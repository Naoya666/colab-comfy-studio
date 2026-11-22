import { apiFetch } from './gpu-api';

export interface LibraryModel {
  id: string;
  display_name: string;
  filename: string;
  model_type: string;
  architecture: string;
  relative_path?: string;
  external_source?: { url: string; status: 'verified' | 'unverified' } | null;
  state: string;
  size?: number;
  sha256?: string;
  expected_sha256?: string;
  error?: string;
  tags?: string[];
  trigger_words?: string[];
  drive_file_id?: string;
  source?: string;
  source_url?: string | null;
  original_source_url?: string | null;
  download_url?: string | null;
  version_name?: string;
}

export interface ModelTask {
  id: string;
  model_id: string;
  state: string;
  bytes: number;
  total?: number;
  speed?: number;
  error?: string;
  attempt?: number;
  resume_offset?: number;
  resume_phase?: string;
  restart_reason?: string;
}

export interface DependencyWeight {
  source?: string;
  repo_id?: string;
  repo_type?: string;
  revision?: string;
  filename?: string;
  target?: string;
  sha256?: string;
  size?: number;
  architecture?: string;
  drive_file_id?: string;
  source_url?: string;
  download_url?: string;
  url?: string;
}

export interface Dependency {
  id: string;
  node_id: string;
  node_type: string;
  field: string;
  reference: string;
  model_type: string;
  architecture: string;
  model_id?: string | null;
  status: string;
  required: boolean;
  disabled?: boolean;
  bypassed?: boolean;
  strengths?: Record<string, number>;
  candidates?: string[];
  source_hint?: string;
  source_error?: string;
  source_conflict?: boolean;
  expected_sha256?: string;
  expected_size?: number;
  detected_via?: 'rule' | 'metadata' | 'heuristic';
  weight?: DependencyWeight;
}

export interface DependencyRecord {
  dependencies: Dependency[];
  models_ready: boolean;
  workflow_state: string;
  runtime_validated: boolean;
}

export interface SourceCandidate {
  name: string;
  url: string;
  source?: string;
}

export interface SourceSearchResponse {
  candidates: SourceCandidate[];
  warnings?: string[];
  requires_confirmation?: boolean;
}

export interface ResolvedModelFile {
  display_name?: string;
  filename: string;
  model_type?: string;
  architecture?: string;
  source?: string;
  source_url?: string;
  original_source_url?: string;
  download_url?: string;
  expected_sha256?: string | null;
  expected_size?: number | null;
  reported_size?: number | null;
  source_identity?: Record<string, unknown>;
  version_name?: string;
  trigger_words?: string[];
  [key: string]: unknown;
}

export interface WorkflowInspectSummary {
  total: number;
  ready: number;
  in_progress: number;
  needs_confirmation: number;
  unresolved: number;
  manual_review: number;
  with_source_hint: number;
  optional: number;
}

export interface WorkflowInspectResult {
  format: 'visual' | 'api' | 'both' | 'bundle';
  node_count: number;
  dependencies: Dependency[];
  models_ready: boolean;
  warnings: string[];
  summary: WorkflowInspectSummary;
}

export interface StoredInspectorState {
  open: boolean;
  fileName: string;
  result: WorkflowInspectResult | null;
  importedMap: Record<string, string>;
}

const WORKFLOW_INSPECT_SESSION_KEY = 'studio-model-library-workflow-inspect-v1';

const DEFAULT_STORED_INSPECTOR_STATE: StoredInspectorState = {
  open: false,
  fileName: '',
  result: null,
  importedMap: {},
};

export function readStoredWorkflowInspect(): StoredInspectorState {
  try {
    const raw = sessionStorage.getItem(WORKFLOW_INSPECT_SESSION_KEY);
    if (!raw) return DEFAULT_STORED_INSPECTOR_STATE;
    const parsed = JSON.parse(raw) as Partial<StoredInspectorState>;
    return {
      open: Boolean(parsed.open),
      fileName: typeof parsed.fileName === 'string' ? parsed.fileName : '',
      result: parsed.result && Array.isArray(parsed.result.dependencies) ? parsed.result : null,
      importedMap: parsed.importedMap && typeof parsed.importedMap === 'object' ? parsed.importedMap : {},
    };
  } catch {
    return DEFAULT_STORED_INSPECTOR_STATE;
  }
}

export function writeStoredWorkflowInspect(state: StoredInspectorState) {
  try {
    if (!state.result && !state.open) {
      sessionStorage.removeItem(WORKFLOW_INSPECT_SESSION_KEY);
    } else {
      sessionStorage.setItem(WORKFLOW_INSPECT_SESSION_KEY, JSON.stringify(state));
    }
  } catch {
    // Ignore storage quota or private mode restrictions.
  }
}

export interface Catalog {
  items: LibraryModel[];
  types: Record<string, string>;
  statistics: {
    count: number;
    bytes: number;
    by_type: Record<string, { count: number; bytes: number }>;
  };
}

export interface DriveUsageCategory {
  key: string;
  label: string;
  description: string;
  bytes: number;
  count: number;
}

export interface DriveStorageQuota {
  limit?: number | null;
  usage?: number | null;
  usage_in_drive?: number | null;
  usage_in_drive_trash?: number | null;
}

export interface DriveUsageReport {
  connected: boolean;
  account_id?: string | null;
  account_alias?: string | null;
  cached_at?: number;
  quota?: DriveStorageQuota | null;
  studio_total_bytes: number;
  studio_file_count: number;
  truncated: boolean;
  categories: DriveUsageCategory[];
  message?: string;
}

export const ACTIVE_MODEL_STATES = new Set(['pending', 'resolving', 'downloading', 'verifying', 'uploading']);

export const libraryApi = <T,>(path = '', body?: unknown, method = body === undefined ? 'GET' : 'POST') =>
  apiFetch<T>(`/api/gpu/model-library${path}`, { method, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });

export const getDriveUsage = (refresh = false, signal?: AbortSignal) =>
  apiFetch<DriveUsageReport>(`/api/gpu/model-library/drive-usage${refresh ? '?refresh=true' : ''}`, { cache: 'no-store', signal }, 45000);

export const dependencyApi = <T,>(workflow: string, action = '', body?: unknown) =>
  apiFetch<T>(
    `/api/gpu/workflows/${encodeURIComponent(workflow)}/model-dependencies${action ? '/' + action : ''}`,
    body === undefined && !action ? {} : { method: 'POST', body: JSON.stringify(body || {}) },
  );

export const inspectWorkflowDependencies = (payload: Record<string, unknown>) =>
  libraryApi<WorkflowInspectResult>('/inspect-workflow', payload, 'POST');

export const bytesLabel = (value?: number | null) => {
  if (!value || !Number.isFinite(value) || value <= 0) return '大小待确认';
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
};

export const modelStatus: Record<string, string> = {
  pending: '待准备',
  resolving: '解析中',
  downloading: '下载中',
  verifying: '校验中',
  uploading: '上传至 Drive',
  ready: 'Drive 已就绪',
  failed: '失败',
  missing: 'Drive 文件缺失',
  unverified: '待校验',
  cancelled: '已取消',
  matched: '已就绪',
  unresolved: '待补充来源',
  needs_confirmation: '库内有同名',
  manual_review: '节点待确认',
};

export function modelBasename(reference: string): string {
  const parts = reference.replace(/\\/g, '/').split('/').filter(Boolean);
  return parts[parts.length - 1] || reference;
}

export function defaultSearchQuery(reference: string): string {
  const base = modelBasename(reference).trim();
  const stripped = base.replace(/\.(safetensors|gguf|onnx|ckpt|pt|pth|bin)$/i, '').trim();
  return (stripped || base).slice(0, 200);
}

export function shortHash(sha?: string | null): string {
  if (!sha) return '';
  const cleaned = sha.trim();
  if (cleaned.length <= 16) return cleaned;
  return `${cleaned.slice(0, 10)}…${cleaned.slice(-6)}`;
}

export function sourceHostLabel(url: string, explicitSource?: string): string {
  const lower = (explicitSource || '').toLowerCase();
  if (lower === 'huggingface') return 'Hugging Face';
  if (lower === 'civitai') return 'Civitai';
  try {
    const host = new URL(url).hostname.toLowerCase();
    if (host === 'huggingface.co' || host.endsWith('.huggingface.co')) return 'Hugging Face';
    if (host === 'civitai.com' || host.endsWith('.civitai.com')) return 'Civitai';
    if (host === 'github.com' || host.endsWith('.github.com')) return 'GitHub';
    return host;
  } catch {
    return 'HTTPS';
  }
}

export function dependencySourceUrl(dep: Dependency): string | null {
  if (typeof dep.source_hint === 'string' && dep.source_hint.trim()) {
    return dep.source_hint.trim();
  }
  const weight = dep.weight;
  if (weight) {
    if (weight.repo_id && weight.revision && weight.filename) {
      const repoType = (weight.repo_type || 'models').toLowerCase();
      const prefix = repoType === 'datasets' ? 'datasets/' : repoType === 'spaces' ? 'spaces/' : '';
      const encodedPath = weight.filename.split('/').map(encodeURIComponent).join('/');
      return `https://huggingface.co/${prefix}${weight.repo_id}/resolve/${encodeURIComponent(weight.revision)}/${encodedPath}`;
    }
    const direct = weight.source_url || weight.download_url || weight.url;
    if (typeof direct === 'string' && direct.startsWith('https://')) {
      return direct;
    }
  }
  return null;
}

export function safeRelativePathFor(reference: string, filename: string): string | undefined {
  const normalized = reference.replace(/\\/g, '/').trim();
  if (!normalized.includes('/') || normalized.startsWith('/') || normalized.includes('..') || normalized.includes(':')) {
    return undefined;
  }
  if (modelBasename(normalized) !== filename) return undefined;
  return normalized;
}

export interface ReconciledDependency {
  dep: Dependency;
  bound: LibraryModel | undefined;
  task: ModelTask | undefined;
  candidates: LibraryModel[];
  effectiveStatus: string;
  sourceUrl: string | null;
}

export function reconcileDependencyWithCatalog(
  dep: Dependency,
  models: LibraryModel[],
  tasks: ModelTask[] = [],
  knownModelId?: string | null,
): ReconciledDependency {
  const sourceUrl = dependencySourceUrl(dep);
  const typeCompatible = (m: LibraryModel) => dep.model_type === 'unknown' || m.model_type === dep.model_type;
  const preferredId = knownModelId || dep.model_id || null;

  let bound = preferredId ? models.find((m) => m.id === preferredId) : undefined;
  if (!bound && dep.expected_sha256) {
    const targetSha = dep.expected_sha256.toLowerCase();
    const matches = models.filter(
      (m) => typeCompatible(m) && (m.sha256?.toLowerCase() === targetSha || m.expected_sha256?.toLowerCase() === targetSha),
    );
    bound = matches.find((m) => m.state === 'ready') || matches[0];
  }
  if (!bound && sourceUrl) {
    const matches = models.filter((m) => {
      if (!typeCompatible(m)) return false;
      const urls = [m.external_source?.url, m.original_source_url, m.source_url, m.download_url].filter(Boolean);
      return urls.includes(sourceUrl);
    });
    bound = matches.find((m) => m.state === 'ready') || matches[0];
  }

  const baseName = modelBasename(dep.reference);
  const candidates = bound
    ? []
    : models.filter((m) => typeCompatible(m) && m.filename === baseName);

  const task = bound
    ? tasks.find((t) => t.model_id === bound.id && ACTIVE_MODEL_STATES.has(t.state))
    : undefined;

  let effectiveStatus = dep.status;
  if (bound) {
    effectiveStatus = bound.state === 'ready' ? 'matched' : (task?.state || bound.state || 'pending');
  } else if (dep.status !== 'manual_review') {
    effectiveStatus = candidates.length ? 'needs_confirmation' : 'unresolved';
  }

  return {
    dep,
    bound,
    task,
    candidates,
    effectiveStatus,
    sourceUrl,
  };
}

export type BatchResolvePick =
  | { ok: true; file: ResolvedModelFile; matchNote?: string }
  | { ok: false; reason: string };

export function pickUniqueResolvedFile(dep: Dependency, files: ResolvedModelFile[]): BatchResolvePick {
  if (!files.length) {
    return { ok: false, reason: '该链接未返回受支持的模型文件（仅支持 safetensors / gguf / onnx 及受控超分 .pt/.pth）' };
  }
  if (dep.expected_sha256) {
    const expected = dep.expected_sha256.toLowerCase();
    const exactSha = files.filter((f) => typeof f.expected_sha256 === 'string' && f.expected_sha256.toLowerCase() === expected);
    if (exactSha.length === 1) {
      return { ok: true, file: exactSha[0], matchNote: 'SHA-256 精确匹配' };
    }
    if (exactSha.length > 1) {
      return { ok: false, reason: '存在多个相同 SHA-256 的候选文件，请手动选择' };
    }
  }
  if (files.length === 1) {
    return { ok: true, file: files[0] };
  }
  const expectedName = modelBasename(dep.reference);
  const exactName = files.filter((f) => f.filename === expectedName);
  if (exactName.length === 1) {
    return { ok: true, file: exactName[0], matchNote: '按文件名从多文件来源中匹配' };
  }
  const caseInsensitive = files.filter((f) => f.filename.toLowerCase() === expectedName.toLowerCase());
  if (caseInsensitive.length === 1) {
    return { ok: true, file: caseInsensitive[0], matchNote: '按文件名从多文件来源中匹配' };
  }
  return {
    ok: false,
    reason: `来源返回了 ${files.length} 个文件且无法按名称唯一确定，请在列表中手动解析并选择具体文件`,
  };
}

// Drive resumable sessions are bearer capabilities. Keep them out of localStorage and logs.
export async function uploadModel(file: File, onProgress: (sent: number) => void) {
  const session = await libraryApi<{ upload_url: string; chunk_size: number }>('/upload-session', { filename: file.name, size: file.size });
  let offset = 0;
  let result: { id?: string } = {};
  while (offset < file.size) {
    const end = Math.min(offset + session.chunk_size, file.size);
    let completed = false;
    for (let attempt = 0; attempt < 3 && !completed; attempt++) {
      try {
        const response = await fetch(session.upload_url, {
          method: 'PUT',
          body: file.slice(offset, end),
          headers: { 'Content-Range': `bytes ${offset}-${end - 1}/${file.size}`, 'Content-Type': 'application/octet-stream' },
        });
        if (response.status === 308) {
          const range = response.headers.get('Range');
          offset = range ? Number(range.match(/-(\d+)$/)?.[1] ?? -1) + 1 : end;
          completed = true;
        } else if (response.ok) {
          result = await response.json();
          offset = end;
          completed = true;
        } else {
          throw new Error(`Drive 上传失败（HTTP ${response.status}）`);
        }
      } catch (error) {
        if (attempt === 2) throw error;
        const probe = await fetch(session.upload_url, { method: 'PUT', headers: { 'Content-Range': `bytes */${file.size}` } });
        if (probe.ok) {
          result = await probe.json();
          offset = file.size;
          completed = true;
        } else if (probe.status === 308) {
          const range = probe.headers.get('Range');
          offset = range ? Number(range.match(/-(\d+)$/)?.[1] ?? -1) + 1 : 0;
          completed = true;
        } else {
          throw new Error(`Drive 上传会话失效（HTTP ${probe.status}），请重新上传`);
        }
      }
    }
    onProgress(offset);
  }
  if (!result.id) throw new Error('Drive 未返回文件 ID，不能提交模型索引');
  return result.id;
}
