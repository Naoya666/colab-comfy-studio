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
  managed_file?: boolean;
  reference_count?: number;
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
  display_name?: string;
  version_name?: string;
  target?: string;
  sha256?: string;
  size?: number;
  architecture?: string;
  drive_file_id?: string;
  source_url?: string;
  download_url?: string;
  url?: string;
  source_identity?: Record<string, unknown>;
}

export interface Dependency {
  id: string;
  node_id: string;
  node_type: string;
  field: string;
  reference: string;
  display_name?: string;
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
  source_name?: string;
  source_host?: string;
  source_note?: string;
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

export const libraryApi = <T,>(
  path = '',
  body?: unknown,
  method = body === undefined ? 'GET' : 'POST',
  timeoutMs?: number,
) =>
  apiFetch<T>(
    `/api/gpu/model-library${path}`,
    { method, ...(body === undefined ? {} : { body: JSON.stringify(body) }) },
    timeoutMs,
  );

export const getDriveUsage = (refresh = false, signal?: AbortSignal) =>
  apiFetch<DriveUsageReport>(`/api/gpu/model-library/drive-usage${refresh ? '?refresh=true' : ''}`, { cache: 'no-store', signal }, 45000);

export const dependencyApi = <T,>(workflow: string, action = '', body?: unknown) =>
  apiFetch<T>(
    `/api/gpu/workflows/${encodeURIComponent(workflow)}/model-dependencies${action ? '/' + action : ''}`,
    body === undefined && !action ? {} : { method: 'POST', body: JSON.stringify(body || {}) },
  );

export const inspectWorkflowDependencies = (payload: Record<string, unknown>) =>
  libraryApi<WorkflowInspectResult>('/inspect-workflow', payload, 'POST', 45000);

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

export type ModelSortMode = 'smart' | 'size_desc' | 'name_asc' | 'status';

export interface TypeSegment {
  type: string;
  count: number;
  bytes: number;
  slot: number;
}

export function orderedTypeSegments(catalog: Catalog | null): { entries: TypeSegment[]; totalBytes: number } {
  const byType = catalog?.statistics?.by_type || {};
  const sorted = Object.entries(byType)
    .filter(([, stat]) => stat && (stat.bytes > 0 || stat.count > 0))
    .map(([type, stat]) => ({ type, count: stat.count, bytes: stat.bytes }))
    .sort((a, b) => b.bytes - a.bytes || b.count - a.count || a.type.localeCompare(b.type));
  const entries: TypeSegment[] = sorted.map((item, idx) => ({ ...item, slot: idx % 6 }));
  const totalBytes = entries.reduce((sum, item) => sum + item.bytes, 0);
  return { entries, totalBytes };
}

export function computeCategorySlots(catalog: Catalog | null): Record<string, number> {
  const { entries } = orderedTypeSegments(catalog);
  const map: Record<string, number> = {};
  for (const item of entries) {
    map[item.type] = item.slot;
  }
  return map;
}

export function displayModelTitles(
  model: Pick<LibraryModel, 'display_name' | 'filename' | 'relative_path'>,
): { primary: string; secondary: string | null } {
  const displayName = (model.display_name || '').trim();
  const filename = (model.filename || '').trim();
  const relPath = (model.relative_path || '').trim();

  if (displayName && filename && displayName.includes('/') && modelBasename(displayName) === filename) {
    const pathStr = relPath || displayName;
    return {
      primary: filename,
      secondary: pathStr !== filename ? pathStr : null,
    };
  }

  const primary = displayName || filename || relPath;
  const pathStr = relPath || filename;
  return {
    primary,
    secondary: pathStr && pathStr !== primary ? pathStr : null,
  };
}

function modelPriorityRank(model: LibraryModel, activeTaskIds: Set<string>): number {
  if (activeTaskIds.has(model.id) || ACTIVE_MODEL_STATES.has(model.state)) return 0;
  if (model.state === 'failed' || model.state === 'missing') return 1;
  if (model.state === 'unverified') return 2;
  return 3;
}

export function sortModels(items: LibraryModel[], tasks: ModelTask[], mode: ModelSortMode): LibraryModel[] {
  const activeTaskIds = new Set(tasks.filter((t) => ACTIVE_MODEL_STATES.has(t.state)).map((t) => t.model_id));
  return [...items].sort((a, b) => {
    if (mode === 'smart') {
      const rankDiff = modelPriorityRank(a, activeTaskIds) - modelPriorityRank(b, activeTaskIds);
      if (rankDiff !== 0) return rankDiff;
      const sizeDiff = (b.size || 0) - (a.size || 0);
      if (sizeDiff !== 0) return sizeDiff;
      return displayModelTitles(a).primary.localeCompare(displayModelTitles(b).primary);
    }
    if (mode === 'size_desc') {
      const sizeDiff = (b.size || 0) - (a.size || 0);
      if (sizeDiff !== 0) return sizeDiff;
      return displayModelTitles(a).primary.localeCompare(displayModelTitles(b).primary);
    }
    if (mode === 'status') {
      const rankDiff = modelPriorityRank(a, activeTaskIds) - modelPriorityRank(b, activeTaskIds);
      if (rankDiff !== 0) return rankDiff;
      return displayModelTitles(a).primary.localeCompare(displayModelTitles(b).primary);
    }
    return displayModelTitles(a).primary.localeCompare(displayModelTitles(b).primary);
  });
}

export interface DuplicateCandidate {
  filename?: string | null;
  expected_sha256?: string | null;
  expected_size?: number | null;
  reported_size?: number | null;
  model_type?: string | null;
  source_url?: string | null;
  original_source_url?: string | null;
  download_url?: string | null;
  drive_file_id?: string | null;
}

export interface DuplicateMatchResult {
  level: 'none' | 'exact' | 'name';
  exactMatches: LibraryModel[];
  nameMatches: LibraryModel[];
  reason: string;
}

function normalizeUrlValue(url?: string | null): string | null {
  if (!url || typeof url !== 'string') return null;
  const trimmed = url.trim();
  return trimmed ? trimmed : null;
}

export function findCatalogDuplicates(
  candidate: DuplicateCandidate,
  catalogItems: LibraryModel[],
): DuplicateMatchResult {
  const candSha = typeof candidate.expected_sha256 === 'string' && candidate.expected_sha256.trim()
    ? candidate.expected_sha256.trim().toLowerCase()
    : null;
  const candDriveId = typeof candidate.drive_file_id === 'string' && candidate.drive_file_id.trim()
    ? candidate.drive_file_id.trim()
    : null;
  const candBase = typeof candidate.filename === 'string' && candidate.filename.trim()
    ? modelBasename(candidate.filename.trim()).toLowerCase()
    : null;
  const candType = typeof candidate.model_type === 'string' && candidate.model_type.trim() && candidate.model_type !== 'unknown'
    ? candidate.model_type.trim()
    : null;
  const candSizeRaw = candidate.expected_size ?? candidate.reported_size ?? null;
  const candSize = typeof candSizeRaw === 'number' && Number.isFinite(candSizeRaw) && candSizeRaw > 0 ? candSizeRaw : null;
  const candUrls = new Set(
    [candidate.original_source_url, candidate.source_url, candidate.download_url]
      .map(normalizeUrlValue)
      .filter((v): v is string => Boolean(v)),
  );

  const exactMatches: LibraryModel[] = [];
  const nameMatches: LibraryModel[] = [];
  let exactReason = '';

  for (const m of catalogItems) {
    const mSha = m.sha256?.trim().toLowerCase();
    const mExpectedSha = m.expected_sha256?.trim().toLowerCase();
    const mBase = modelBasename(m.filename || '').toLowerCase();
    const mRelBase = m.relative_path ? modelBasename(m.relative_path).toLowerCase() : '';
    const mUrls = [m.external_source?.url, m.original_source_url, m.source_url, m.download_url]
      .map(normalizeUrlValue)
      .filter((v): v is string => Boolean(v));

    if (candSha && (mSha === candSha || mExpectedSha === candSha)) {
      exactMatches.push(m);
      if (!exactReason) exactReason = 'SHA-256 与库内已有模型一致';
      continue;
    }
    if (candDriveId && m.drive_file_id && m.drive_file_id === candDriveId) {
      exactMatches.push(m);
      if (!exactReason) exactReason = '该 Drive 文件 ID 已在模型库中';
      continue;
    }
    if (candUrls.size > 0 && mUrls.some((u) => candUrls.has(u))) {
      exactMatches.push(m);
      if (!exactReason) exactReason = '相同下载来源已入库';
      continue;
    }
    const nameHit = Boolean(candBase && (mBase === candBase || (mRelBase && mRelBase === candBase)));
    if (nameHit) {
      const mSize = typeof m.size === 'number' && m.size > 0 ? m.size : null;
      const typeMatch = !candType || m.model_type === candType;
      if (candSize && mSize && candSize === mSize && typeMatch) {
        exactMatches.push(m);
        if (!exactReason) exactReason = '同类同名且文件体积完全一致';
      } else {
        nameMatches.push(m);
      }
    }
  }

  if (exactMatches.length > 0) {
    return { level: 'exact', exactMatches, nameMatches, reason: exactReason };
  }
  if (nameMatches.length > 0) {
    return { level: 'name', exactMatches: [], nameMatches, reason: '库内已存在同名文件' };
  }
  return { level: 'none', exactMatches: [], nameMatches: [], reason: '' };
}

export interface PlannedBatchDeleteItem {
  model: LibraryModel;
  deletePhysical: boolean;
  fallbackReason?: string;
}

export interface BatchDeletionPlan {
  mode: 'remove_index' | 'delete_file';
  deletable: PlannedBatchDeleteItem[];
  activeBlocked: LibraryModel[];
  referencedModels: LibraryModel[];
  fallbackToIndexOnly: LibraryModel[];
  selectedBytes: number;
  reclaimableBytes: number;
}

export function planBatchDeletion(
  selectedModels: LibraryModel[],
  allModels: LibraryModel[],
  tasks: ModelTask[],
  mode: 'remove_index' | 'delete_file',
): BatchDeletionPlan {
  const activeIds = new Set(tasks.filter((t) => ACTIVE_MODEL_STATES.has(t.state)).map((t) => t.model_id));
  const activeBlocked: LibraryModel[] = [];
  const candidates: LibraryModel[] = [];

  for (const m of selectedModels) {
    if (activeIds.has(m.id) || ACTIVE_MODEL_STATES.has(m.state)) {
      activeBlocked.push(m);
    } else {
      candidates.push(m);
    }
  }

  const candidateIds = new Set(candidates.map((m) => m.id));
  const sharedOutsideDriveIds = new Set<string>();
  for (const m of allModels) {
    if (!candidateIds.has(m.id) && m.drive_file_id) {
      sharedOutsideDriveIds.add(m.drive_file_id);
    }
  }

  // Within the selected candidates, if multiple indices share one managed drive_file_id,
  // only the final index deletes the physical file after sibling indices are removed first.
  const lastOwnerForDriveId = new Map<string, string>();
  if (mode === 'delete_file') {
    for (const m of candidates) {
      if (m.drive_file_id && m.managed_file && !sharedOutsideDriveIds.has(m.drive_file_id)) {
        lastOwnerForDriveId.set(m.drive_file_id, m.id);
      }
    }
  }

  const fallbackToIndexOnly: LibraryModel[] = [];
  const planned: PlannedBatchDeleteItem[] = [];

  for (const m of candidates) {
    if (mode === 'remove_index' || !m.drive_file_id) {
      planned.push({ model: m, deletePhysical: false });
      continue;
    }
    if (!m.managed_file) {
      fallbackToIndexOnly.push(m);
      planned.push({
        model: m,
        deletePhysical: false,
        fallbackReason: '外部导入或历史缓存文件，仅移除索引',
      });
      continue;
    }
    if (sharedOutsideDriveIds.has(m.drive_file_id)) {
      fallbackToIndexOnly.push(m);
      planned.push({
        model: m,
        deletePhysical: false,
        fallbackReason: '物理文件被其他未选中索引共享，仅移除索引',
      });
      continue;
    }
    const isLastOwner = lastOwnerForDriveId.get(m.drive_file_id) === m.id;
    planned.push({ model: m, deletePhysical: isLastOwner });
  }

  // Delete index-only entries first, then physical-delete entries last so shared Drive files succeed cleanly.
  const deletable = [
    ...planned.filter((p) => !p.deletePhysical),
    ...planned.filter((p) => p.deletePhysical),
  ];

  const seenSelectedKeys = new Set<string>();
  let selectedBytes = 0;
  for (const m of selectedModels) {
    const key = m.drive_file_id || m.id;
    if (!seenSelectedKeys.has(key)) {
      seenSelectedKeys.add(key);
      selectedBytes += m.size || 0;
    }
  }

  const seenReclaimIds = new Set<string>();
  let reclaimableBytes = 0;
  for (const item of deletable) {
    if (item.deletePhysical && item.model.drive_file_id && !seenReclaimIds.has(item.model.drive_file_id)) {
      seenReclaimIds.add(item.model.drive_file_id);
      reclaimableBytes += item.model.size || 0;
    }
  }

  const referencedModels = candidates.filter((m) => (m.reference_count || 0) > 0);

  return {
    mode,
    deletable,
    activeBlocked,
    referencedModels,
    fallbackToIndexOnly,
    selectedBytes,
    reclaimableBytes,
  };
}

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
    if (
      host === 'civitai.com' || host.endsWith('.civitai.com')
      || host === 'civitai.red' || host.endsWith('.civitai.red')
    ) return 'Civitai';
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
    if (weight.source === 'civitai' && weight.source_identity && typeof weight.source_identity === 'object') {
      const modelId = weight.source_identity.model_id;
      const versionId = weight.source_identity.version_id;
      if (modelId !== undefined && /^\d+$/.test(String(modelId))) {
        const base = `https://civitai.com/models/${modelId}`;
        if (versionId !== undefined && /^\d+$/.test(String(versionId))) {
          return `${base}?modelVersionId=${versionId}`;
        }
        return base;
      }
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

export interface CatalogIndex {
  byId: Map<string, LibraryModel>;
  bySha: Map<string, LibraryModel[]>;
  byUrl: Map<string, LibraryModel[]>;
  byFilename: Map<string, LibraryModel[]>;
  activeTaskByModelId: Map<string, ModelTask>;
}

export function buildCatalogIndex(models: LibraryModel[], tasks: ModelTask[] = []): CatalogIndex {
  const byId = new Map<string, LibraryModel>();
  const bySha = new Map<string, LibraryModel[]>();
  const byUrl = new Map<string, LibraryModel[]>();
  const byFilename = new Map<string, LibraryModel[]>();
  const push = (map: Map<string, LibraryModel[]>, key: string, item: LibraryModel) => {
    const bucket = map.get(key);
    if (bucket) bucket.push(item);
    else map.set(key, [item]);
  };
  for (const m of models) {
    byId.set(m.id, m);
    const sha = m.sha256?.toLowerCase();
    const expSha = m.expected_sha256?.toLowerCase();
    if (sha) push(bySha, sha, m);
    if (expSha && expSha !== sha) push(bySha, expSha, m);
    const seenUrls = new Set<string>();
    for (const u of [m.external_source?.url, m.original_source_url, m.source_url, m.download_url]) {
      if (u && !seenUrls.has(u)) {
        seenUrls.add(u);
        push(byUrl, u, m);
      }
    }
    if (m.filename) push(byFilename, m.filename, m);
  }
  const activeTaskByModelId = new Map<string, ModelTask>();
  for (const t of tasks) {
    if (ACTIVE_MODEL_STATES.has(t.state) && !activeTaskByModelId.has(t.model_id)) {
      activeTaskByModelId.set(t.model_id, t);
    }
  }
  return { byId, bySha, byUrl, byFilename, activeTaskByModelId };
}

export function reconcileDependencyWithCatalog(
  dep: Dependency,
  models: LibraryModel[],
  tasks: ModelTask[] = [],
  knownModelId?: string | null,
  index?: CatalogIndex,
): ReconciledDependency {
  const sourceUrl = dependencySourceUrl(dep);
  const typeCompatible = (m: LibraryModel) => dep.model_type === 'unknown' || m.model_type === dep.model_type;
  const preferredId = knownModelId || dep.model_id || null;

  let bound = preferredId
    ? (index ? index.byId.get(preferredId) : models.find((m) => m.id === preferredId))
    : undefined;
  if (!bound && dep.expected_sha256) {
    const targetSha = dep.expected_sha256.toLowerCase();
    const pool = index ? (index.bySha.get(targetSha) || []) : models.filter(
      (m) => m.sha256?.toLowerCase() === targetSha || m.expected_sha256?.toLowerCase() === targetSha,
    );
    const matches = pool.filter(typeCompatible);
    bound = matches.find((m) => m.state === 'ready') || matches[0];
  }
  if (!bound && sourceUrl) {
    const pool = index ? (index.byUrl.get(sourceUrl) || []) : models.filter((m) => {
      const urls = [m.external_source?.url, m.original_source_url, m.source_url, m.download_url].filter(Boolean);
      return urls.includes(sourceUrl);
    });
    const matches = pool.filter(typeCompatible);
    bound = matches.find((m) => m.state === 'ready') || matches[0];
  }

  const baseName = modelBasename(dep.reference);
  const candidates = bound
    ? []
    : (index ? (index.byFilename.get(baseName) || []) : models.filter((m) => m.filename === baseName)).filter(typeCompatible);

  const task = bound
    ? (index ? index.activeTaskByModelId.get(bound.id) : tasks.find((t) => t.model_id === bound.id && ACTIVE_MODEL_STATES.has(t.state)))
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
