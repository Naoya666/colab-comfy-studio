import { WorkflowsPanel } from './pages/WorkflowManager';
import { CreationApiPage } from "./pages/CreationApiPage";
import { ModelRunOverrides, type ModelOverride } from './components/ModelDependencies';
import { NativeComfyPage } from './pages/NativeComfyPage';
import { ModelLibraryPage } from './pages/ModelLibraryPage';
import { SettingsPanel } from './pages/SettingsPage';
import { GalleryPanel } from './pages/GalleryPage';
import { TasksPanel, type JobClearRequest } from './pages/JobsPage';
import { WorkflowGpuSelection, computeLabel } from './components/WorkflowGpuSelection';
import { CreatePanel } from './pages/CreatePage';
import { CreateResults } from './components/CreateResults';
import { ComputePage } from './pages/ComputePage';
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  Sparkles, Zap, Image as ImageIcon, Cpu, Settings as SettingsIcon, Folder, ChevronDown, X, Plus, Check,
  Maximize2, Star, Download, Trash2, RotateCcw, Triangle, Boxes, Workflow, Layers, KeyRound, ArrowLeft,
  ChevronLeft, ChevronRight, Copy, ImagePlus, Pencil, Search,
} from 'lucide-react';
import {
  apiMessage, assetUrl, BATCH_DELETE_ASSET_LIMIT, batchDeleteAssets, cancelGpuJob, clearGpuJobHistory, createGpuJob, createGpuBatch,
  createProject, deleteAsset, deleteGpuPreset, deleteGpuWorkflow, deleteProject,
  apiFetch,
  getGpuComputeSpecs, getGpuGallery, getGpuJob, getGpuJobs, getGpuModels, listGpuInstances,
  getGpuPresets, getReferenceAssets, getGpuSettings, getGpuUsage, getGpuWorkflowSchema, getGpuWorkflows, getProjects, importGpuWorkflow,
  isLoggedIn, login, publishGpuWorkflow, renameProject, retryGpuJob, retryGpuAttempt, stopGpuCompute, updateAsset,
  updateGpuPreset, uploadAsset,
  validateGpuWorkflow, rollbackGpuWorkflow, deactivateGpuWorkflow, saveGpuPreset, ApiError, type ComputeSpec, type GalleryAsset, type GpuJob, type GpuPreset,
  type GpuManifest, type GpuModel, type GpuSettings, type GpuUsage, type GpuWorkflow, type Project, type Provider,
} from './lib/gpu-api';
import type { GpuAccount, GpuInstance } from './components/AccountDirectory';
import type { RuntimeChoice } from './pages/CreatePage';
import { referenceAssets, supportsReferences } from './components/ReferencePicker';
import { Segmented } from './components/ui';
import { ConfirmDialog, ToastStack, type ToastItem } from './components/Feedback';
import { resolveGpuCheckRequest } from './lib/colab-check';
import { jobStateLabel } from './lib/job-stage-rows';
import { jobDetailTimings, paramRows, stageSecondsTotal, taskStatusNote } from './lib/job-detail-copy';
import { resolveInstanceSubmission } from './lib/instance-submission';
import { projectOptionLabel } from './lib/project-label';
import { carriedDraft, mergeWorkflowDraft, readWorkflowDraft, writeWorkflowDraft } from './lib/workflow-draft';
import { formatInstanceTime } from './lib/instance-time';
import { JobHistoryDetails } from './components/JobHistoryDetails';
import { useDesktopLayout } from './lib/desktop-layout';
import { parameterKey, parameterList, readableState } from './lib/ui-helpers';
import { useSmartPoll } from './lib/smart-poll';
import { formatHashRoute, parseHashRoute, type AssetsSubTab, type MainTab } from './lib/route-state';
import styles from './gpu.module.css';

const mobileTabs: MainTab[] = ['创作', '任务', '作品', '资产', '算力', '原生 ComfyUI'];
const desktopTabs: MainTab[] = ['创作', '任务', '作品', '原生 ComfyUI', '资产', '算力', '设置'];
const tabIcons: Record<MainTab, typeof Sparkles> = {
  创作: Sparkles,
  任务: Zap,
  作品: ImageIcon,
  算力: Cpu,
  资产: Layers,
  '原生 ComfyUI': Workflow,
  设置: SettingsIcon,
};
const mobileLabels: Partial<Record<MainTab, string>> = { '原生 ComfyUI': '原生' };
const terminalStates = new Set(['completed', 'succeeded', 'failed', 'failed_final', 'cancelled', 'cancelled_local', 'blocked']);
const activeInstanceStates = new Set(['requested', 'allocating', 'preparing', 'ready', 'busy', 'retained_idle', 'releasing']);

function record(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function uniqueBy<T>(items: T[], keyOf: (item: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = keyOf(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function instanceIdOf(instance: Pick<GpuInstance, 'id' | 'instance_id'>): string {
  return instance.id || instance.instance_id || '';
}

function keyId() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `gpu-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function shown(value: unknown) {
  return value === null || value === undefined || value === '' ? '未知' : typeof value === 'object' ? JSON.stringify(value) : String(value);
}

function assetIdsFromParams(params: Record<string, unknown>, validIds: Set<string>) {
  return Object.values(params).filter((value): value is string => typeof value === 'string' && validIds.has(value));
}

function extractPromptText(params?: Record<string, unknown>): string {
  if (!params) return '';
  if (typeof params.prompt === 'string' && params.prompt.trim()) return params.prompt;
  for (const [k, v] of Object.entries(params)) {
    if (/prompt|caption/i.test(k) && !/negative/i.test(k) && typeof v === 'string' && v.trim()) return v;
  }
  return '';
}

function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      await login(password);
      onLoggedIn();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="loginCard" onSubmit={submit}>
        <div className="brand"><span className="brandMark"><Triangle size={16} strokeWidth={1.6} aria-hidden /></span><h1>Colab GPU Studio</h1></div>
        <p className="muted">输入访问密码进入工作台</p>
        <label>
          访问密码
          <input
            autoFocus
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            placeholder="请输入密码"
            required
          />
        </label>
        {error && <p className="status error" role="alert">{error}</p>}
        <button className="button big" disabled={busy} type="submit">
          {busy ? '登录中…' : '进入工作台'}
        </button>
      </form>
    </main>
  );
}

const PROJECT_KEY = 'colab-gpu-studio-project';
function savedProjectId(): string { try { return localStorage.getItem(PROJECT_KEY) || ''; } catch { return ''; } }
function saveProjectId(id: string) { try { if (id) localStorage.setItem(PROJECT_KEY, id); } catch { /* storage can be unavailable */ } }

function ProjectSheet({
  open,
  projects,
  projectId,
  onClose,
  onSelect,
  onRename,
  onDelete,
}: {
  open: boolean;
  projects: Project[];
  projectId: string;
  onClose: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => Promise<boolean>;
  onDelete: (project: Project) => Promise<boolean>;
}) {
  const [query, setQuery] = useState('');
  const [editingId, setEditingId] = useState('');
  const [editingTitle, setEditingTitle] = useState('');
  const [busyId, setBusyId] = useState('');
  const [confirmDelete, setConfirmDelete] = useState<Project | null>(null);

  if (!open) return null;

  const visible = projects.filter((p) => !query.trim() || `${p.title} ${p.id}`.toLowerCase().includes(query.trim().toLowerCase()));

  async function saveEdit(id: string) {
    const trimmed = editingTitle.trim();
    if (!trimmed || busyId) return;
    setBusyId(id);
    try {
      if (await onRename(id, trimmed)) {
        setEditingId('');
        setEditingTitle('');
      }
    } finally {
      setBusyId('');
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="切换与管理项目" onClick={(e) => e.stopPropagation()}>
        <div className={styles.cardHead}>
          <h2>切换与管理项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
        {projects.length > 5 && (
          <label className={styles.searchBox}>
            <Search size={16} strokeWidth={1.6} aria-hidden />
            <input aria-label="搜索项目" placeholder="搜索项目名称" value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>
        )}
        {projects.length === 0 && <p className="muted">还没有项目，点右上角的 ＋ 新建一个。</p>}
        <ul className="projectList">
          {visible.map((p) => {
            const isEditing = editingId === p.id;
            return (
              <li key={p.id} className={styles.projectManageRow}>
                {isEditing ? (
                  <form
                    className={styles.inlineForm}
                    style={{ width: '100%' }}
                    onSubmit={(e) => { e.preventDefault(); void saveEdit(p.id); }}
                  >
                    <input
                      autoFocus
                      maxLength={80}
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      aria-label="项目新名称"
                    />
                    <button className="button small" type="submit" disabled={busyId === p.id || !editingTitle.trim()}>保存</button>
                    <button className="button secondary small" type="button" onClick={() => setEditingId('')}>取消</button>
                  </form>
                ) : (
                  <>
                    <button
                      type="button"
                      className="projectRow"
                      aria-current={p.id === projectId ? 'true' : undefined}
                      onClick={() => { onSelect(p.id); onClose(); }}
                    >
                      <span>{projectOptionLabel(p, projects)}</span>
                      {p.id === projectId && <Check size={16} strokeWidth={1.8} aria-label="当前项目" />}
                    </button>
                    <button
                      type="button"
                      className={styles.iconGhost}
                      title="重命名项目"
                      aria-label={`重命名 ${p.title}`}
                      onClick={() => { setEditingId(p.id); setEditingTitle(p.title); }}
                    >
                      <Pencil size={15} strokeWidth={1.6} aria-hidden />
                    </button>
                    <button
                      type="button"
                      className={styles.iconGhost}
                      title="归档移除项目"
                      aria-label={`移除 ${p.title}`}
                      disabled={projects.length <= 1 || busyId === p.id}
                      onClick={() => setConfirmDelete(p)}
                    >
                      <Trash2 size={15} strokeWidth={1.6} aria-hidden />
                    </button>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      <ConfirmDialog
        open={Boolean(confirmDelete)}
        title="归档移除项目？"
        description={confirmDelete ? `将从项目列表移除「${confirmDelete.title}」。若该项目有正在运行的任务会阻止移除；历史账单记录仍会保留。` : ''}
        confirmLabel="确认移除"
        danger
        busy={Boolean(busyId)}
        onCancel={() => setConfirmDelete(null)}
        onConfirm={async () => {
          if (!confirmDelete) return;
          setBusyId(confirmDelete.id);
          try {
            if (await onDelete(confirmDelete)) setConfirmDelete(null);
          } finally {
            setBusyId('');
          }
        }}
      />
    </div>
  );
}

function NewProjectDialog({
  open,
  onClose,
  onCreate,
}: {
  open: boolean;
  onClose: () => void;
  onCreate: (title: string) => Promise<boolean>;
}) {
  const [title, setTitle] = useState('');
  const [creating, setCreating] = useState(false);
  if (!open) return null;

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    const name = title.trim();
    if (!name || creating) return;
    setCreating(true);
    try {
      if (await onCreate(name)) {
        setTitle('');
        onClose();
      }
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <form className={styles.dialog} role="dialog" aria-modal="true" aria-label="新建项目" onClick={(e) => e.stopPropagation()} onSubmit={handleCreate}>
        <div className={styles.cardHead}>
          <h2>新建项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
        <div className="field">
          <label htmlFor="project-new">项目名称</label>
          <input id="project-new" autoFocus maxLength={80} placeholder="例如：符玄和青雀" value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
        <div className={styles.inlineForm}>
          <button className="button secondary" type="button" onClick={onClose}>取消</button>
          <button className="button" disabled={creating || !title.trim()} type="submit">{creating ? '创建中…' : '创建'}</button>
        </div>
      </form>
    </div>
  );
}

export default function App() {
  const desktop = useDesktopLayout();
  const initialRoute = useMemo(
    () => parseHashRoute(typeof window !== 'undefined' ? window.location.hash : ''),
    [],
  );
  const lastRouteHash = useRef(formatHashRoute(initialRoute));
  const [loggedIn, setLoggedIn] = useState(isLoggedIn());
  const [tab, setTab] = useState<MainTab>(initialRoute.tab);
  const [assetsSubTab, setAssetsSubTab] = useState<AssetsSubTab>(initialRoute.assetsSubTab);
  const [creationApiOpen, setCreationApiOpen] = useState(initialRoute.creationApiOpen);
  const [adminRole, setAdminRole] = useState(false);
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const pushToast = useCallback((message: string, tone: ToastItem['tone'] = 'info') => {
    const id = keyId();
    setToasts((curr) => [...curr.slice(-3), { id, tone, message }]);
    window.setTimeout(() => {
      setToasts((curr) => curr.filter((item) => item.id !== id));
    }, tone === 'error' ? 5500 : 3500);
  }, []);

  useEffect(() => {
    let current = true;
    setAdminRole(false);
    if (loggedIn) {
      apiFetch<{ role: string }>('/api/me').then(user => {
        if (current) setAdminRole(user.role === 'admin');
      }).catch(() => {});
    }
    return () => { current = false; };
  }, [loggedIn]);

  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState('');
  const [showProjectSheet, setShowProjectSheet] = useState(false);
  const [showNewProject, setShowNewProject] = useState(false);
  const currentProjectId = useRef(projectId);
  currentProjectId.current = projectId;

  const [models, setModels] = useState<GpuModel[]>([]);
  const [workflows, setWorkflows] = useState<GpuWorkflow[]>([]);
  const [presets, setPresets] = useState<GpuPreset[]>([]);
  const [modelId, setModelId] = useState('');
  const [workflowId, setWorkflowId] = useState('');
  const [gpuChoices, setGpuChoices] = useState<Record<string, Record<string, unknown> | null>>(() => { try { return JSON.parse(localStorage.getItem('studio-workflow-gpus') || '{}'); } catch { return {}; } });
  const selectedCompute = gpuChoices[workflowId] || null;
  const [modelOverrides, setModelOverrides] = useState<Record<string, ModelOverride[]>>({});
  function chooseGpu(key: Record<string, unknown> | null) {
    const next = { ...gpuChoices, [workflowId]: key }; setGpuChoices(next); idempotencyKey.current = '';
    try { localStorage.setItem('studio-workflow-gpus', JSON.stringify(next)); } catch { /* browser storage unavailable */ }
  }
  const [manifest, setManifest] = useState<GpuManifest | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [gallery, setGallery] = useState<GalleryAsset[]>([]);
  const [galleryError, setGalleryError] = useState('');
  const [galleryRevision, setGalleryRevision] = useState(0);
  const [referenceImages, setReferenceImages] = useState<GalleryAsset[]>([]);
  const [jobs, setJobs] = useState<GpuJob[]>([]);
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const [runtime, setRuntime] = useState<Provider>('mock');
  const [maxRuntime, setMaxRuntime] = useState(30);
  const [batchCount, setBatchCount] = useState(1);
  const [batchMode, setBatchMode] = useState<'save' | 'smart' | 'batch' | 'retain'>('save');
  const [runtimeChoice, setRuntimeChoice] = useState<RuntimeChoice>({ accountId: '', instanceId: '', lifecycle: 'save', retainIdleMinutes: 15 });
  const [accounts, setAccounts] = useState<GpuAccount[]>([]);
  const [instances, setInstances] = useState<GpuInstance[]>([]);
  const [, setResourceError] = useState('');
  const resourceRefresh = useRef<(() => void) | null>(null);
  const submitting = useRef(false);
  const [settings, setSettings] = useState<GpuSettings | null>(null);
  const [usage, setUsage] = useState<GpuUsage | null>(null);
  const [specs, setSpecs] = useState<ComputeSpec[]>([]);
  const [specsNote, setSpecsNote] = useState('');
  const [error, setError] = useState('');
  const [jobError, setJobError] = useState('');
  const [loading, setLoading] = useState('');
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [confirm, setConfirm] = useState<'submit' | 'stop' | null>(null);
  const [skipConfirmSig, setSkipConfirmSig] = useState('');
  const [rememberSkipConfirm, setRememberSkipConfirm] = useState(false);
  const [detail, setDetail] = useState<GpuJob | null>(null);
  const [detailId, setDetailId] = useState(initialRoute.jobId);
  const [detailError, setDetailError] = useState('');
  const [detailRefresh, setDetailRefresh] = useState(0);
  const [viewerSelection, setViewer] = useState<GalleryAsset | null>(null);
  const [viewerList, setViewerList] = useState<GalleryAsset[]>([]);
  const [pendingRouteAssetId, setPendingRouteAssetId] = useState(initialRoute.assetId);
  const [confirmDeleteViewer, setConfirmDeleteViewer] = useState<GalleryAsset | null>(null);
  const [viewerJob, setViewerJob] = useState<GpuJob | null>(null);
  const [viewerJobError, setViewerJobError] = useState('');
  const [submission, setSubmission] = useState<{ projectId: string; ids: string[] } | null>(null);
  const [againJob, setAgainJob] = useState<GpuJob | null>(null);
  const [loginRefresh, setLoginRefresh] = useState(0);
  const [refreshingJobs, setRefreshingJobs] = useState(false);
  const idempotencyKey = useRef('');
  const retryKeys = useRef<Record<string, string>>({});
  const instanceRetentionKeys = useRef<Record<string, string>>({});
  const jobSequence = useRef(0);
  const resourceSequenceRef = useRef(0);
  const appliedResourceSequenceRef = useRef(0);
  const draftReady = useRef('');
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const workflowIdRef = useRef(workflowId);
  workflowIdRef.current = workflowId;
  const incomingDraft = useRef<Record<string, unknown> | null>(null);
  const draftLocation = useRef({ projectId: '', workflowId: '' });

  const activeProject = projects.find((p) => p.id === projectId);
  const activeWorkflow = workflows.find((item) => item.id === workflowId);
  const workflowParamsReady = Boolean(projectId && workflowId && draftReady.current === `${projectId}:${workflowId}`);
  const projectJobs = useMemo(() => jobs.filter(job => job.project_id === projectId), [jobs, projectId]);
  const projectGallery = useMemo(() => gallery.filter(asset => asset.project_id === projectId), [gallery, projectId]);
  const latestJob = projectJobs.find(job => !job.prep);
  const currentSubmission = submission?.projectId === projectId ? submission : null;
  const previewJobs = currentSubmission
    ? projectJobs.filter(job => currentSubmission.ids.includes(job.id))
    : latestJob?.batch_id
      ? projectJobs.filter(job => job.batch_id === latestJob.batch_id)
      : latestJob ? [latestJob] : [];
  const galleryOutputs = desktop && (tab === '创作' || tab === '作品')
    ? projectJobs.flatMap(job => job.output_asset_ids || []).join(',')
    : '';
  const activeInstance = instances.find(instance => instance.provider === 'colab' && !instance.runtime_deleted
    && instanceIdOf(instance) === runtimeChoice.instanceId)
    || instances.find(instance => instance.provider === 'colab' && !instance.runtime_deleted
      && ['ready', 'busy', 'retained_idle', 'allocating', 'preparing'].includes(instance.state || ''));
  const viewer = viewerSelection?.project_id === projectId
    ? projectGallery.find(asset => asset.id === viewerSelection.id) ?? viewerSelection
    : null;
  const routeAssetId = tab === '作品' ? pendingRouteAssetId || viewer?.id || '' : '';
  const viewerJobId = viewer?.source_job_id || viewer?.gpu_job_id;
  const inlineViewer = desktop && tab === '作品';
  const detailRevision = projectJobs.find(job => job.id === detailId)?.updated_at;

  const activeJobsCount = useMemo(() => {
    return projectJobs.filter((j) => !terminalStates.has(String(j.state || '').toLowerCase())).length;
  }, [projectJobs]);
  const hasActiveInstances = useMemo(
    () => instances.some((i) => !i.runtime_deleted && activeInstanceStates.has(String(i.state || '').toLowerCase())),
    [instances],
  );

  const navList = useMemo(() => {
    const base = viewerList.length ? viewerList.filter((a) => projectGallery.some((g) => g.id === a.id)) : projectGallery;
    return base.map((item) => projectGallery.find((g) => g.id === item.id) || item);
  }, [viewerList, projectGallery]);
  const viewerIndex = viewer ? navList.findIndex((a) => a.id === viewer.id) : -1;

  // Sync route state to URL hash & listen to browser back/forward
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const nextHash = formatHashRoute({
      tab,
      assetsSubTab,
      creationApiOpen,
      jobId: tab === '任务' ? detailId : '',
      assetId: routeAssetId,
    });
    if (lastRouteHash.current === nextHash) return;
    lastRouteHash.current = nextHash;
    if (window.location.hash !== nextHash) {
      window.history.pushState(null, '', nextHash);
    }
  }, [tab, assetsSubTab, creationApiOpen, detailId, routeAssetId]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onHashChange = () => {
      const parsed = parseHashRoute(window.location.hash);
      lastRouteHash.current = formatHashRoute(parsed);
      setTab(parsed.tab);
      setAssetsSubTab(parsed.assetsSubTab);
      setCreationApiOpen(parsed.creationApiOpen);
      setDetailId(parsed.jobId);
      setPendingRouteAssetId(parsed.assetId);
      setViewer(null);
      setViewerList([]);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    if (!pendingRouteAssetId || !projectGallery.length) return;
    const found = projectGallery.find((a) => a.id === pendingRouteAssetId);
    if (found) {
      setViewer(found);
      setViewerList(projectGallery);
      setPendingRouteAssetId('');
    }
  }, [pendingRouteAssetId, projectGallery]);

  useEffect(() => {
    const expire = () => setLoggedIn(false);
    window.addEventListener('gpu-auth-expired', expire);
    return () => window.removeEventListener('gpu-auth-expired', expire);
  }, []);

  useEffect(() => {
    if (!loggedIn) return;
    let alive = true;
    Promise.allSettled([
      getProjects(),
      getGpuModels(),
      getGpuWorkflows(),
      getGpuPresets(),
      getGpuSettings(),
      getGpuUsage(),
    ]).then((results) => {
      if (!alive) return;
      const [projR, modR, wfR, preR, setR, useR] = results;
      if (projR.status === 'fulfilled') {
        setProjects(projR.value);
        const list = projR.value;
        setProjectId((curr) => {
          if (curr && list.some((p) => p.id === curr)) return curr;
          const saved = savedProjectId();
          return (saved && list.some((p) => p.id === saved) ? saved : list[0]?.id) || '';
        });
      } else setError(apiMessage(projR.reason));
      if (modR.status === 'fulfilled') setModels(modR.value);
      if (wfR.status === 'fulfilled') {
        const list = wfR.value;
        setWorkflows(list);
        setWorkflowId((curr) => {
          if (curr && list.some((w) => w.id === curr)) return curr;
          const active = list.find((w) => w.active && w.state === 'published') || list.find((w) => w.state === 'published') || list[0];
          if (active) {
            setModelId(active.model_id);
            return active.id;
          }
          return '';
        });
      }
      if (preR.status === 'fulfilled') setPresets(preR.value);
      if (setR.status === 'fulfilled') setSettings(setR.value);
      if (useR.status === 'fulfilled') setUsage(useR.value);
    }).catch((cause) => alive && setError(apiMessage(cause)));
    return () => { alive = false; };
  }, [loggedIn, loginRefresh]);

  const refreshResources = useCallback(async () => {
    const seq = ++resourceSequenceRef.current;
    try {
      const [accountResult, instanceResult] = await Promise.all([
        apiFetch<unknown>('/api/gpu/accounts'),
        listGpuInstances(),
      ]);
      if (seq < appliedResourceSequenceRef.current) return;
      appliedResourceSequenceRef.current = seq;
      const nextAccounts = (Array.isArray(accountResult) ? accountResult : record(accountResult) && Array.isArray(accountResult.data) ? accountResult.data : []) as GpuAccount[];
      const nextInstances = instanceResult as unknown as GpuInstance[];
      setAccounts(uniqueBy(nextAccounts, (account) => account.account_id));
      setInstances(uniqueBy(nextInstances, (instance) => instanceIdOf(instance)));
      setResourceError('');
    } catch (cause) {
      if (seq === resourceSequenceRef.current) setResourceError(`账号/实例：${apiMessage(cause)}`);
    }
  }, []);

  useEffect(() => {
    resourceRefresh.current = () => void refreshResources();
    return () => { resourceRefresh.current = null; };
  }, [refreshResources]);

  useSmartPoll(refreshResources, {
    enabled: loggedIn,
    active: hasActiveInstances || activeJobsCount > 0 || tab === '算力',
    activeMs: 5000,
    idleMs: 25000,
  });

  useEffect(() => {
    if (!modelId) return;
    const allowed = workflows.filter((w) => w.model_id === modelId);
    if (workflowId && !allowed.some((w) => w.id === workflowId)) {
      if (projectId && draftReady.current === `${projectId}:${workflowId}`) writeWorkflowDraft(projectId, workflowId, paramsRef.current);
      draftReady.current = '';
      setWorkflowId('');
      setManifest(null);
      setParams({});
    }
  }, [modelId, workflows, workflowId, projectId]);

  useEffect(() => {
    if (!workflowId || !projectId) return;
    const previous = draftLocation.current;
    const sameProject = previous.projectId === projectId;
    if (previous.workflowId && previous.projectId && draftReady.current === `${previous.projectId}:${previous.workflowId}`) {
      writeWorkflowDraft(previous.projectId, previous.workflowId, paramsRef.current);
    }
    const carried = sameProject && previous.workflowId && previous.workflowId !== workflowId ? carriedDraft(paramsRef.current) : null;
    draftLocation.current = { projectId, workflowId };
    let alive = true;
    const requestWorkflow = workflowId;
    const requestProject = projectId;
    setLoading('schema');
    getGpuWorkflowSchema(workflowId).then((next) => {
      if (!alive) return;
      const forced = incomingDraft.current;
      incomingDraft.current = null;
      const saved = forced ?? readWorkflowDraft(requestProject, requestWorkflow);
      setManifest(next);
      draftReady.current = `${requestProject}:${requestWorkflow}`;
      setParams(mergeWorkflowDraft(next, saved, forced ? null : carried));
    }).catch((cause) => alive && setError(`参数 schema：${apiMessage(cause)}`)).finally(() => alive && setLoading(''));
    return () => { alive = false; };
  }, [workflowId, projectId]);

  useEffect(() => {
    if (draftReady.current !== `${projectId}:${workflowId}` || !workflowId || !projectId) return;
    writeWorkflowDraft(projectId, workflowId, params);
  }, [params, projectId, workflowId]);

  const refreshJobs = useCallback(async () => {
    const pid = currentProjectId.current;
    if (!pid) return;
    const seq = ++jobSequence.current;
    const current = () => currentProjectId.current === pid && seq === jobSequence.current;
    setRefreshingJobs(true);
    try {
      const next = await getGpuJobs(pid);
      if (!current()) return;
      setJobs(next.filter((j) => j.project_id === pid));
      setJobError('');
    } catch (cause) {
      if (current()) setJobError(apiMessage(cause));
    } finally {
      if (current()) setRefreshingJobs(false);
    }
  }, []);

  useSmartPoll(refreshJobs, {
    enabled: Boolean(projectId && loggedIn),
    active: activeJobsCount > 0,
    activeMs: 4000,
    idleMs: 25000,
  });

  useEffect(() => {
    if (projectId && loggedIn) void refreshJobs();
  }, [projectId, loggedIn, refreshJobs]);

  useEffect(() => {
    setReferenceImages([]);
    setSelectedAssets([]);
    if (!projectId || !loggedIn) return;
    let alive = true;
    getReferenceAssets(projectId).then((next) => {
      if (alive) setReferenceImages(referenceAssets(next));
    }).catch((cause) => {
      if (alive) setError(`素材：${apiMessage(cause)}`);
    });
    return () => { alive = false; };
  }, [projectId, loggedIn]);

  useEffect(() => {
    setSelectedAssets([]);
  }, [workflowId, projectId]);

  useEffect(() => {
    if (!projectId || !loggedIn) return;
    let alive = true;
    setGalleryError('');
    getGpuGallery(projectId).then(next => {
      if (alive) setGallery(next.map(asset => asset.project_id ? asset : { ...asset, project_id: projectId }));
    }).catch(cause => {
      if (alive) setGalleryError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [projectId, loggedIn, tab, galleryOutputs, galleryRevision]);

  useEffect(() => {
    if (tab !== '任务' || !detailId || !projectId || !loggedIn) return;
    let alive = true;
    getGpuJob(detailId).then(job => {
      if (!alive || job.project_id !== projectId) return;
      setDetail(job);
      setDetailError('');
    }).catch(cause => {
      if (alive) setDetailError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [tab, detailId, detailRevision, detailRefresh, projectId, loggedIn]);

  useEffect(() => {
    if (!inlineViewer || !viewerJobId || !projectId || !loggedIn) return;
    let alive = true;
    setViewerJobError('');
    getGpuJob(viewerJobId).then(job => {
      if (alive && job.project_id === projectId) setViewerJob(job);
    }).catch(cause => {
      if (alive) setViewerJobError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [inlineViewer, viewerJobId, projectId, loggedIn]);

  useEffect(() => {
    if (againJob && (againJob.resolved_params || againJob.params)) {
      const restored = { ...(againJob.resolved_params || againJob.params) };
      setModelId(String(againJob.model_id || ''));
      const workflow = workflows.find((item) => item.workflow_key === againJob.workflow_key && String(item.workflow_version) === String(againJob.workflow_version));
      if (workflow && workflow.id !== workflowIdRef.current) incomingDraft.current = restored;
      if (workflow) setWorkflowId(workflow.id);
      setParams(restored);
      setAgainJob(null);
      setTab('创作');
      pushToast('已载入历史参数到创作工作区', 'success');
    }
  }, [againJob, workflows, pushToast]);

  function chooseProject(id: string) {
    if (id !== projectId) {
      setDetailId('');
      setDetail(null);
      setViewer(null);
      setPendingRouteAssetId('');
    }
    setProjectId(id);
    saveProjectId(id);
  }

  async function createLocalProject(title: string): Promise<boolean> {
    try {
      const created = await createProject(title);
      setProjects((curr) => [created, ...curr]);
      chooseProject(created.id);
      pushToast(`已创建项目「${created.title}」`, 'success');
      return true;
    } catch (cause) {
      pushToast(apiMessage(cause), 'error');
      return false;
    }
  }

  async function renameLocalProject(id: string, title: string): Promise<boolean> {
    try {
      const updated = await renameProject(id, title);
      setProjects((curr) => curr.map((item) => item.id === id ? { ...item, ...updated } : item));
      pushToast(`项目已重命名为「${updated.title}」`, 'success');
      return true;
    } catch (cause) {
      pushToast(`重命名失败：${apiMessage(cause)}`, 'error');
      return false;
    }
  }

  async function removeLocalProject(project: Project): Promise<boolean> {
    try {
      await deleteProject(project.id);
      const remaining = projects.filter((item) => item.id !== project.id);
      setProjects(remaining);
      if (projectId === project.id) {
        chooseProject(remaining[0]?.id || '');
      }
      pushToast(`已移除项目「${project.title}」`, 'success');
      return true;
    } catch (cause) {
      pushToast(apiMessage(cause), 'error');
      return false;
    }
  }

  async function handleUpload(file: File): Promise<GalleryAsset | void> {
    const uploadProjectId = projectId;
    const uploadWorkflowId = workflowId;
    if (!uploadProjectId) return;
    setUploadProgress((curr) => ({ ...curr, [file.name]: 0 }));
    try {
      const asset = await uploadAsset(uploadProjectId, file, (progress) => {
        if (currentProjectId.current === uploadProjectId) setUploadProgress((curr) => ({ ...curr, [file.name]: progress }));
      });
      if (currentProjectId.current !== uploadProjectId || asset.project_id !== uploadProjectId) return;
      setReferenceImages((curr) => [asset, ...curr.filter((item) => item.id !== asset.id)]);
      pushToast(`已上传素材「${asset.filename || file.name}」`, 'success');
      if (workflowIdRef.current !== uploadWorkflowId) return;
      setSelectedAssets((curr) => [...new Set([...curr, asset.id])]);
      return asset;
    } catch (cause) {
      if (currentProjectId.current === uploadProjectId) setError(`上传 ${file.name}：${apiMessage(cause)}`);
    } finally {
      if (currentProjectId.current === uploadProjectId) setUploadProgress((curr) => {
        const next = { ...curr };
        delete next[file.name];
        return next;
      });
    }
  }

  function submitSignature() {
    return JSON.stringify([
      workflowId, runtime, batchCount, runtimeChoice.accountId, runtimeChoice.instanceId,
      runtimeChoice.lifecycle, selectedCompute, maxRuntime, runtimeChoice.retainIdleMinutes,
    ]);
  }

  function requestSubmit() {
    if (skipConfirmSig && skipConfirmSig === submitSignature()) {
      void submitJob();
      return;
    }
    setRememberSkipConfirm(false);
    setConfirm('submit');
  }

  async function submitJob() {
    if (submitting.current || !workflowParamsReady || !modelId || !activeWorkflow) return;
    submitting.current = true;
    if (!idempotencyKey.current) idempotencyKey.current = keyId();
    setLoading('submit');
    setError('');
    try {
      const decision = resolveInstanceSubmission(runtimeChoice, instances);
      if (!decision.ok) {
        setRuntimeChoice({ ...runtimeChoice, accountId: decision.accountId, instanceId: decision.instanceId });
        setError(decision.message);
        return;
      }
      const validAssetIds = new Set(referenceAssets(referenceImages).filter((a) => a.project_id === projectId).map((a) => a.id));
      const body = {
        project_id: projectId,
        model_id: modelId,
        workflow_key: activeWorkflow.workflow_key,
        workflow_version: activeWorkflow.workflow_version,
        params,
        ...(modelOverrides[workflowId]?.length ? { model_overrides: modelOverrides[workflowId] } : {}),
        asset_ids: supportsReferences(manifest)
          ? [...new Set([...selectedAssets.filter((id) => validAssetIds.has(id)), ...assetIdsFromParams(params, validAssetIds)])]
          : [],
        execution_policy: {
          provider: runtime,
          ...(runtime === 'colab' && selectedCompute ? { compute_spec: selectedCompute } : {}),
          max_runtime_minutes: maxRuntime,
          allocation: decision.allocation,
          account_id: decision.accountId,
          instance_id: decision.instanceId,
          lifecycle: runtimeChoice.lifecycle,
          retain_idle_minutes: runtimeChoice.retainIdleMinutes,
        },
      };
      let ids: string[];
      if (runtime === 'mock' || batchCount > 1) {
        const batch = await createGpuBatch(Array.from({ length: batchCount }, (_, index) => ({
          ...body,
          params: { ...body.params, ...(typeof body.params.seed === 'number' && body.params.seed >= 0 ? { seed: body.params.seed + index } : {}) },
        })),
          runtimeChoice.lifecycle, idempotencyKey.current);
        ids = batch.job_ids || [];
      } else {
        ids = [(await createGpuJob(body, idempotencyKey.current)).id];
      }
      if (rememberSkipConfirm) setSkipConfirmSig(submitSignature());
      setConfirm(null);
      if (currentProjectId.current === projectId) {
        setSubmission(ids.length ? { projectId, ids } : null);
        if (!desktop) setTab('任务');
      }
      idempotencyKey.current = '';
      pushToast(`已提交 ${ids.length} 个生成任务`, 'success');
      await refreshJobs();
    } catch (cause) {
      const code = cause instanceof ApiError ? cause.code : '';
      if (code === 'INSTANCE_UNAVAILABLE' || code === 'INSTANCE_NOT_FOUND') {
        const decision = resolveInstanceSubmission(runtimeChoice, instances);
        setRuntimeChoice({ ...runtimeChoice, accountId: decision.accountId || runtimeChoice.accountId, instanceId: null });
        setError('所选 GPU 机器已释放或不能复用。这次没有提交任务，也没有换账号。请再点一次确认，改为自动分配，或另选一台可用机器。');
      } else if (code === 'NETWORK_TIMEOUT') {
        setError(`提交任务：${apiMessage(cause)}。网络中断时请先到「任务」页确认是否已经提交，不要连续改参数重试。`);
      } else {
        setError(`提交任务：${apiMessage(cause)}。这次没有确认任务已创建。`);
      }
    } finally {
      submitting.current = false;
      setLoading('');
    }
  }

  async function action(name: string, operation: () => Promise<unknown>) {
    setLoading(name);
    setError('');
    try {
      await operation();
    } catch (cause) {
      const msg = apiMessage(cause);
      setError(msg);
      pushToast(msg, 'error');
    } finally {
      setLoading('');
    }
  }

  async function importWorkflow(raw: string) {
    try {
      const value = JSON.parse(raw) as unknown;
      const source = record(value) && record(value.workflow) ? value.workflow : value;
      if (!record(source) || (!record(source.image_extraction) && (!record(source.manifest) || !record(source.api_graph) || !record(source.object_info)))) {
        throw new Error('JSON 必须包含 manifest、api_graph、object_info');
      }
      await action('import', async () => {
        const workflow = await importGpuWorkflow(value as never);
        setWorkflows((curr) => [workflow, ...curr.filter((item) => item.id !== workflow.id)]);
        pushToast(
          workflow.pending_configuration
            ? '图片工作流已保存为待配置草稿，请展开草稿补充运行配置。'
            : '工作流已作为草稿导入，请校验后发布。',
          'success',
        );
      });
    } catch (cause) {
      setError(`导入失败：${apiMessage(cause)}`);
    }
  }

  async function clearJobs(request: JobClearRequest) {
    if (!projectId) return;
    await action('clear-jobs', async () => {
      const { cleared } = await clearGpuJobHistory({
        project_id: projectId,
        scope: request.scope,
        job_ids: request.jobIds,
      });
      if (request.jobIds?.includes(detailId) || !request.jobIds) {
        setDetailId('');
        setDetail(null);
      }
      await refreshJobs();
      pushToast(cleared ? `已清理 ${cleared} 条任务历史` : '没有可清理的已结束任务', 'success');
    });
  }

  async function reuseAsset(asset: GalleryAsset) {
    const sourceJobId = asset.source_job_id || asset.gpu_job_id;
    if (!sourceJobId) {
      setGalleryError('该素材没有来源任务记录。');
      return;
    }
    setGalleryError('');
    try {
      const job = await getGpuJob(sourceJobId);
      if (currentProjectId.current !== asset.project_id) return;
      setAgainJob(job);
      setViewer(current => current?.id === asset.id ? null : current);
    } catch (cause) {
      if (currentProjectId.current === asset.project_id) setGalleryError(`再次生成：${apiMessage(cause)}`);
    }
  }

  function useAssetAsReference(asset: GalleryAsset) {
    setReferenceImages((curr) => [asset, ...curr.filter((item) => item.id !== asset.id)]);
    setSelectedAssets((curr) => [...new Set([...curr, asset.id])]);
    if (manifest) {
      const plist = parameterList(manifest);
      const fileParams = plist
        .map((p, idx) => ({ p, key: parameterKey(p, idx) }))
        .filter(({ p }) => p.widget === 'file' || ['image', 'video', 'file'].includes(String(p.type)));
      if (fileParams.length > 0) {
        const slot = fileParams.find(({ key }) => !paramsRef.current[key]) || fileParams[0];
        setParams((curr) => ({ ...curr, [slot.key]: asset.id }));
      }
    }
    setViewer(null);
    setTab('创作');
    setCreationApiOpen(false);
    pushToast(`已将「${asset.filename || '作品'}」设为参考素材`, 'success');
  }

  async function favorite(asset: GalleryAsset) {
    setGalleryError('');
    try {
      const next = await updateAsset(asset.id, !asset.favorite);
      setGallery((curr) => curr.map((i) => i.id === next.id ? { ...i, ...next } : i));
      setViewer(current => current?.id === next.id ? { ...current, ...next } : current);
      pushToast(next.favorite ? '已收藏作品' : '已取消收藏', 'info');
    } catch (cause) {
      if (currentProjectId.current === asset.project_id) setGalleryError(apiMessage(cause));
    }
  }

  async function remove(asset: GalleryAsset) {
    setGalleryError('');
    try {
      await deleteAsset(asset.id);
      setGallery((curr) => curr.filter((i) => i.id !== asset.id));
      setViewer(current => current?.id === asset.id ? null : current);
      pushToast('已删除作品', 'success');
    } catch (cause) {
      if (currentProjectId.current === asset.project_id) setGalleryError(`删除失败：${apiMessage(cause)}`);
    }
  }

  async function batchRemove(ids: string[]) {
    const removedIds = new Set<string>();
    let error = '';
    setGalleryError('');
    try {
      for (let offset = 0; offset < ids.length; offset += BATCH_DELETE_ASSET_LIMIT) {
        const batch = ids.slice(offset, offset + BATCH_DELETE_ASSET_LIMIT);
        const result = await batchDeleteAssets(batch);
        result.ids.forEach((id) => removedIds.add(id));
      }
    } catch (cause) {
      // A request can fail after deleting some files. Reconcile before retrying those IDs.
      const remaining = await getGpuGallery(projectId).catch(() => null);
      if (remaining) {
        const existing = new Set(remaining.map((asset) => asset.id));
        ids.filter((id) => !existing.has(id)).forEach((id) => removedIds.add(id));
      }
      if (removedIds.size < ids.length) error = `批量删除失败：${apiMessage(cause)}`;
    }
    if (removedIds.size > 0) {
      setGallery((current) => current.filter((asset) => !removedIds.has(asset.id)));
      setViewer((current) => current && removedIds.has(current.id) ? null : current);
      if (currentProjectId.current === projectId) setGalleryRevision((revision) => revision + 1);
    }
    if (!error) pushToast(`已批量删除 ${removedIds.size} 件作品`, 'success');
    return { removedIds: [...removedIds], error };
  }

  async function savePreset(name: string) {
    if (!workflowId) return;
    await action('preset', async () => {
      const next = await saveGpuPreset({ name, workflow_version_id: workflowId, overrides: params });
      setPresets((curr) => [next, ...curr]);
      pushToast(`已保存预设「${next.name}」`, 'success');
    });
  }

  async function updatePreset(preset: GpuPreset) {
    await action('preset', async () => {
      const next = await updateGpuPreset(preset.id, { overrides: params });
      setPresets((curr) => curr.map((item) => item.id === next.id ? next : item));
      pushToast(`已更新预设「${next.name}」`, 'success');
    });
  }

  async function removePreset(preset: GpuPreset) {
    await action('preset', async () => {
      await deleteGpuPreset(preset.id);
      setPresets((curr) => curr.filter((item) => item.id !== preset.id));
      pushToast(`已删除预设「${preset.name}」`, 'success');
    });
  }

  function applyPreset(preset: GpuPreset) {
    if (preset.workflow_version_id !== workflowId) {
      setError('预设版本不兼容。');
      return;
    }
    setParams({ ...params, ...preset.overrides });
    pushToast(`已套用预设「${preset.name}」`, 'info');
  }

  async function checkGpu() {
    setSpecs([]);
    setSpecsNote('');
    await action('check', async () => {
      const [currentAccounts, currentSettings] = await Promise.all([
        apiFetch<GpuAccount[]>('/api/gpu/accounts'),
        getGpuSettings(),
      ]);
      setAccounts(uniqueBy(currentAccounts, (account) => account.account_id));
      setSettings(currentSettings);
      const checkRequest = resolveGpuCheckRequest(currentSettings, currentAccounts, runtimeChoice.accountId || null);
      if (checkRequest.provider === 'mock') {
        setSpecs(await getGpuComputeSpecs('mock'));
        setSpecsNote(checkRequest.reason === 'offline'
          ? '站点处于离线测试模式，下面是「模拟」GPU，不是真实 Colab 机器。'
          : checkRequest.reason === 'real_disabled'
            ? '真实 GPU 未启用，下面是「模拟」GPU，不是真实 Colab 机器。'
            : runtimeChoice.accountId
              ? '所选 Google 账号未连接或已停用，下面是「模拟」GPU。请先到「设置」完成授权。'
              : '账号池没有已连接且启用的 Google 账号，下面是「模拟」GPU。请先到「设置」完成授权。');
        return;
      }
      setSpecs(await getGpuComputeSpecs('colab', checkRequest.accountId));
      setSpecsNote(runtimeChoice.accountId
        ? '已查询所选 Google 账号的 Colab：有申请资格不等于现在一定有空闲 GPU。'
        : '已按账号池选择 Google 账号并查询 Colab：有申请资格不等于现在一定有空闲 GPU。');
    });
  }

  async function stopCompute() {
    await action('stop', async () => {
      await stopGpuCompute();
      setConfirm(null);
      pushToast('已发出取消任务请求；未自动释放没有任务的保留实例。', 'info');
    });
  }

  function changeRuntimeChoice(next: RuntimeChoice) {
    setRuntimeChoice(next);
    setBatchMode(next.lifecycle);
  }

  function openInstanceWorkbench(instance: GpuInstance) {
    const id = instanceIdOf(instance);
    if (!id) return;
    changeRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: runtimeChoice.lifecycle, retainIdleMinutes: runtimeChoice.retainIdleMinutes });
    setRuntime(instance.provider === 'mock' ? 'mock' : 'colab');
    setTab('算力');
  }

  function continueInstance(instance: GpuInstance) {
    const id = instanceIdOf(instance);
    if (!id) return;
    changeRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: 'save', retainIdleMinutes: runtimeChoice.retainIdleMinutes });
    setRuntime(instance.provider === 'mock' ? 'mock' : 'colab');
    setTab('创作');
  }

  async function retainJob(job: GpuJob) {
    const instanceId = String(job.instance_id || '');
    if (!instanceId) { setJobError('该任务尚未分配实例，暂不能保留。'); return; }
    await action('retain', async () => {
      const operation = `retain:${instanceId}`;
      const writeKey = instanceRetentionKeys.current[operation] ||= `retain:${instanceId}:${keyId()}`;
      const updatedInstance = await apiFetch<GpuInstance>(`/api/gpu/instances/${encodeURIComponent(instanceId)}/retention`, {
        method: 'PATCH', headers: { 'Idempotency-Key': writeKey },
        body: JSON.stringify({ retention_mode: 'retain', idle_minutes: runtimeChoice.retainIdleMinutes }),
      });
      delete instanceRetentionKeys.current[operation];
      setInstances((current) => {
        const index = current.findIndex((item) => item.instance_id === updatedInstance.instance_id);
        const resolvedIndex = index >= 0 ? index : current.findIndex((item) => instanceIdOf(item) === instanceIdOf(updatedInstance));
        if (resolvedIndex < 0) return uniqueBy([updatedInstance, ...current], (instance) => instanceIdOf(instance));
        if ((current[resolvedIndex].version ?? 0) > (updatedInstance.version ?? 0)) return current;
        const next = [...current];
        next[resolvedIndex] = { ...next[resolvedIndex], ...updatedInstance, id: instanceIdOf(updatedInstance), instance_id: instanceIdOf(updatedInstance) };
        return next;
      });
      setJobs((current) => current.map((item) => item.instance_id === instanceId && !item.runtime_deleted
        ? { ...item, lifecycle: 'retain', retained_until: updatedInstance.retained_until, instance_state: updatedInstance.state }
        : item));
      pushToast('已确认保留当前 GPU 机器', 'success');
      await refreshJobs();
    });
  }

  const stepViewer = useCallback((delta: number) => {
    if (!viewer || viewerIndex < 0 || navList.length < 2) return;
    const nextIndex = (viewerIndex + delta + navList.length) % navList.length;
    setViewer(navList[nextIndex]);
  }, [viewer, viewerIndex, navList]);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);
      if (e.key === 'Escape') {
        if (confirmDeleteViewer) setConfirmDeleteViewer(null);
        else if (confirm) setConfirm(null);
        else if (showNewProject) setShowNewProject(false);
        else if (showProjectSheet) setShowProjectSheet(false);
        else if (viewer) setViewer(null);
        else if (detailId && tab === '任务') setDetailId('');
      } else if (!typing && viewer && !confirm && !confirmDeleteViewer && !showNewProject && !showProjectSheet) {
        if (e.key === 'ArrowLeft') { e.preventDefault(); stepViewer(-1); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); stepViewer(1); }
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [viewer, detailId, tab, confirm, confirmDeleteViewer, showProjectSheet, showNewProject, stepViewer]);

  if (!loggedIn) {
    return <Login onLoggedIn={() => { setLoggedIn(true); setLoginRefresh((c) => c + 1); }} />;
  }

  const inspectedJob = detail?.id === detailId && detail.project_id === projectId ? detail : null;
  const jobDetailContent = <>
    <div className={styles.cardHead}>
      <h2>任务详情{detailId && ` · ${detailId.slice(0, 8)}`}</h2>
      {detailId && <button className={styles.iconGhost} type="button" aria-label="关闭任务详情" onClick={() => setDetailId('')}><X size={18} aria-hidden /></button>}
    </div>
    {detailId && detailError && <p className="status error" role="alert">{detailError}</p>}
    {inspectedJob ? (
      <>
        <dl className={styles.kv}>
          <dt>状态</dt><dd>{jobStateLabel(inspectedJob.state)}</dd>
          <dt>运行方式</dt><dd>{inspectedJob.provider === 'mock' ? '离线模拟' : inspectedJob.provider === 'colab' ? 'Colab GPU' : shown(inspectedJob.provider)}</dd>
          <dt>创建时间</dt><dd>{formatInstanceTime(inspectedJob.created_at)}</dd>
        </dl>
        <p className="muted small">{taskStatusNote(inspectedJob)}</p>
        {Array.isArray(inspectedJob.attempts) && <details className="foldable" open>
          <summary>执行记录</summary>
          <div className="foldContent">
            {inspectedJob.attempts.length ? inspectedJob.attempts.map((attempt: Record<string, unknown>) => <p key={String(attempt.id)}>
              第 {String(attempt.attempt)} 次 · {jobStateLabel(String(attempt.state))}
              {attempt.error_code ? ` · ${String(attempt.error_code)}` : ''}
            </p>) : <p className="muted">还没有开始执行</p>}
          </div>
        </details>}
        {(inspectedJob.timings || inspectedJob.timing_details || inspectedJob.logs || inspectedJob.events) ? <JobHistoryDetails job={inspectedJob} /> : null}
        {stageSecondsTotal(jobDetailTimings(inspectedJob)) ? <p className="muted small">各阶段相加约 {stageSecondsTotal(jobDetailTimings(inspectedJob))}。阶段按秒独立计时，可能重叠，不是工作流发布又额外等了这么久。</p> : null}
        {inspectedJob.resolved_params ? (
          <details className="foldable">
            <summary>本次使用的参数</summary>
            <div className="foldContent">
              <dl className={styles.kv}>
                {paramRows(inspectedJob.resolved_params).map((row) => <Fragment key={row.key}><dt>{row.label}</dt><dd>{row.value}</dd></Fragment>)}
              </dl>
            </div>
          </details>
        ) : null}
      </>
    ) : <p className="muted small">{detailId ? detailError ? '可重新选择任务以加载详情。' : '正在读取任务详情…' : '选择任务查看执行阶段、日志和生成参数。'}</p>}
  </>;

  const viewerPrompt = viewerJob && viewerJob.id === viewerJobId ? extractPromptText(viewerJob.resolved_params || viewerJob.params) : '';
  const viewerContent = viewer && <>
    <div className={`${styles.cardHead} ${styles.viewerHead}`}>
      <h2>{viewer.filename || viewer.id}</h2>
      <div className={styles.viewerHeadActions}>
        {navList.length > 1 && viewerIndex >= 0 && (
          <div className={styles.viewerNavGroup}>
            <button type="button" className="button secondary small" aria-label="上一张" title="上一张 (←)" onClick={() => stepViewer(-1)}>
              <ChevronLeft size={14} aria-hidden />
            </button>
            <span className="muted small">{viewerIndex + 1}/{navList.length}</span>
            <button type="button" className="button secondary small" aria-label="下一张" title="下一张 (→)" onClick={() => stepViewer(1)}>
              <ChevronRight size={14} aria-hidden />
            </button>
          </div>
        )}
        <button className="button secondary small" type="button" onClick={async (event) => {
          const panel = event.currentTarget.closest('section');
          try {
            if (document.fullscreenElement) await document.exitFullscreen();
            else if (panel?.requestFullscreen) await panel.requestFullscreen();
            else throw new Error('当前浏览器不支持全屏');
          } catch (cause) { setGalleryError(apiMessage(cause)); }
        }}><Maximize2 size={14} strokeWidth={1.6} aria-hidden />全屏查看</button>
        <button className="button secondary small" type="button" onClick={() => {
          if (document.fullscreenElement) void document.exitFullscreen().catch(() => {});
          setViewer(null);
        }}>关闭</button>
      </div>
    </div>
    {viewer.mime.startsWith('video/') ? (
      <video className={styles.viewerVideo} src={assetUrl(viewer)} controls autoPlay playsInline />
    ) : viewer.mime.startsWith('image/') ? (
      <img className={styles.viewerImage} src={assetUrl(viewer)} alt="" />
    ) : (
      <p className="muted">这是诊断文件，不能直接预览，请下载查看。</p>
    )}
    {inlineViewer && viewerJobId && (
      viewerJob?.id === viewerJobId ? (
        <details className="foldable" open>
          <summary>生成参数</summary>
          <div className="foldContent">
            {viewerPrompt && (
              <button
                type="button"
                className="button secondary small"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(viewerPrompt);
                    pushToast('已复制提示词', 'success');
                  } catch {
                    pushToast('无法访问剪贴板，请手动复制', 'error');
                  }
                }}
              >
                <Copy size={13} aria-hidden />复制提示词
              </button>
            )}
            <dl className={styles.kv}>
              {paramRows(viewerJob.resolved_params || viewerJob.params || {}).map(row => (
                <Fragment key={row.key}><dt>{row.label}</dt><dd>{row.value}</dd></Fragment>
              ))}
            </dl>
          </div>
        </details>
      ) : <p className="muted small">{viewerJobError || '正在读取生成参数…'}</p>
    )}
    <div className={styles.viewerActions}>
      <button className="button secondary small" type="button" onClick={() => void favorite(viewer)}>
        <Star size={14} strokeWidth={1.6} fill={viewer.favorite ? 'currentColor' : 'none'} aria-hidden />{viewer.favorite ? '取消收藏' : '收藏'}
      </button>
      <button className="button secondary small" type="button" disabled={!viewerJobId} onClick={() => void reuseAsset(viewer)}>
        <RotateCcw size={14} strokeWidth={1.6} aria-hidden />用同样参数再来一次
      </button>
      {(viewer.mime.startsWith('image/') || viewer.mime.startsWith('video/')) && (
        <button className="button secondary small" type="button" onClick={() => useAssetAsReference(viewer)}>
          <ImagePlus size={14} strokeWidth={1.6} aria-hidden />用作参考图
        </button>
      )}
      <a className="button secondary small" href={assetUrl(viewer)} download={viewer.filename || viewer.id} target="_blank" rel="noreferrer">
        <Download size={14} strokeWidth={1.6} aria-hidden />下载
      </a>
      <button className="button danger small" type="button" onClick={() => setConfirmDeleteViewer(viewer)}>
        <Trash2 size={14} strokeWidth={1.6} aria-hidden />删除
      </button>
    </div>
    {galleryError && <p className="status error" role="alert">{galleryError}</p>}
  </>;

  function inspectJob(id: string) {
    setDetailError('');
    if (id === detailId && tab === '任务') {
      setDetailRefresh(value => value + 1);
    }
    setDetailId(id);
  }

  function navigate(next: MainTab) {
    if (next === tab) return;
    setViewer(null);
    setPendingRouteAssetId('');
    setDetailId('');
    setTab(next);
    window.scrollTo(0, 0);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brandMark"><Triangle size={16} strokeWidth={1.6} aria-hidden /></span>
          <span className="brandText">
            <strong>Colab GPU Studio</strong>
            <small>{settings?.offline_mode ? '离线演示' : '个人创作工作台'}</small>
          </span>
        </div>
        <div className="topbarActions">
          <button className="projectChip" type="button" aria-label="切换项目" onClick={() => setShowProjectSheet(true)}>
            <Folder size={15} strokeWidth={1.6} aria-hidden />
            <span>{activeProject ? projectOptionLabel(activeProject, projects) : '选择项目'}</span>
            <ChevronDown size={14} strokeWidth={1.6} aria-hidden />
          </button>
          <button className="iconButton" type="button" aria-label="新建项目" title="新建项目" onClick={() => setShowNewProject(true)}>
            <Plus size={18} strokeWidth={1.6} aria-hidden />
          </button>
          <button className="iconButton mobileSettings" aria-label="打开设置" onClick={() => navigate('设置')}>
            <SettingsIcon size={18} strokeWidth={1.6} aria-hidden />
          </button>
          <button className="desktopGpu projectChip" type="button" onClick={() => navigate('算力')}
            title="查看算力" data-online={Boolean(activeInstance && ['ready', 'busy', 'retained_idle'].includes(activeInstance.state || ''))}>
            <Cpu size={15} strokeWidth={1.6} aria-hidden />
            <span>{settings?.offline_mode ? '离线演示' : activeInstance
              ? `${activeInstance.gpu_reported || 'Colab GPU'} · ${readableState(activeInstance.state)}`
              : '无活动 GPU'}</span>
          </button>
        </div>
      </header>

      <nav className="bottomNav" aria-label="工作区导航">
        {(desktop ? desktopTabs : mobileTabs).map((item) => {
          const Icon = tabIcons[item];
          return (
            <button key={item} type="button" data-tab={item} title={item} aria-label={item === '设置' ? '打开设置' : undefined}
              aria-current={tab === item ? 'page' : undefined} onClick={() => navigate(item)}>
              <Icon size={20} strokeWidth={1.6} aria-hidden />
              <span>{desktop ? item : (mobileLabels[item] || item)}</span>
              {item === '任务' && activeJobsCount > 0 && <span className="tabBadge">{activeJobsCount}</span>}
            </button>
          );
        })}
      </nav>

      {/* 项目切换与管理弹层 */}
      <ProjectSheet
        open={showProjectSheet}
        projects={projects}
        projectId={projectId}
        onClose={() => setShowProjectSheet(false)}
        onSelect={chooseProject}
        onRename={renameLocalProject}
        onDelete={removeLocalProject}
      />
      <NewProjectDialog open={showNewProject} onClose={() => setShowNewProject(false)} onCreate={createLocalProject} />

      {/* 主工作区 */}
      <main className={styles.page}>
        {tab === '创作' && (
          <section className={styles.createWorkspace} aria-label="创作工作区">
            <div className={styles.createHeader}>
              <h1 className={styles.desktopTitle}>创作</h1>
              {adminRole && <button type="button" className={creationApiOpen ? 'button secondary' : 'button'} aria-expanded={creationApiOpen}
                onClick={() => setCreationApiOpen(open => !open)}>
                {creationApiOpen ? <ArrowLeft size={16} aria-hidden /> : <KeyRound size={16} aria-hidden />}
                {creationApiOpen ? '返回创作' : 'API 设置'}
              </button>}
            </div>
            {adminRole && creationApiOpen ? <CreationApiPage accounts={accounts} /> :
              <div className={styles.createColumns}>
                <CreatePanel
                  desktop={desktop}
                  draftReady={workflowParamsReady}
                  gpuSelection={runtime === 'colab' ? <WorkflowGpuSelection manifest={manifest} accountId={runtimeChoice.accountId} value={selectedCompute} onChange={chooseGpu} /> : null}
                  projectId={projectId}
                  models={models}
                  workflows={workflows}
                  presets={presets}
                  settings={settings}
                  manifest={workflowParamsReady ? manifest : null}
                  params={workflowParamsReady ? params : {}}
                  setParams={setParams}
                  gallery={referenceImages}
                  selectedAssets={selectedAssets}
                  setSelectedAssets={setSelectedAssets}
                  uploadProgress={uploadProgress}
                  onUpload={handleUpload}
                  modelId={modelId}
                  setModelId={setModelId}
                  workflowId={workflowId}
                  setWorkflowId={setWorkflowId}
                  runtime={runtime}
                  setRuntime={setRuntime}
                  maxRuntime={maxRuntime}
                  setMaxRuntime={setMaxRuntime}
                  batchCount={batchCount}
                  setBatchCount={setBatchCount}
                  batchMode={batchMode}
                  setBatchMode={setBatchMode}
                  runtimeChoice={runtimeChoice}
                  accounts={accounts}
                  instances={instances}
                  onRuntimeChoiceChange={changeRuntimeChoice}
                  loading={loading}
                  error={error}
                  onSubmit={requestSubmit}
                  onApplyPreset={applyPreset}
                  onSavePreset={(name) => void savePreset(name)}
                  onUpdatePreset={(preset) => void updatePreset(preset)}
                  onDeletePreset={(preset) => void removePreset(preset)}
                >
                  {workflowId && <ModelRunOverrides workflowId={workflowId} value={modelOverrides[workflowId] || []} onChange={value => setModelOverrides(current => ({ ...current, [workflowId]: value }))} />}
                </CreatePanel>
                {desktop && projectId && (
                  <CreateResults
                    key={projectId}
                    jobs={previewJobs}
                    count={currentSubmission?.ids.length || previewJobs.length}
                    gallery={projectGallery}
                    error={galleryError}
                    onOpen={(asset) => { setViewerList(projectGallery); setViewer(asset); }}
                    onJobs={() => navigate('任务')}
                    onFavorite={(asset) => void favorite(asset)}
                    onDelete={(asset) => void remove(asset)}
                    onUseAsReference={useAssetAsReference}
                  />
                )}
              </div>}
          </section>
        )}

        {tab === '任务' && (
          <div className={styles.detailWorkspace}>
            <TasksPanel
              jobs={projectJobs}
              error={jobError}
              refreshing={refreshingJobs}
              clearingHistory={loading === 'clear-jobs'}
              onRefresh={() => void refreshJobs()}
              onCancel={(id) => void action('cancel', async () => { await cancelGpuJob(id); await refreshJobs(); })}
              onRetry={(id) => void action('retry', async () => {
                const job = projectJobs.find((item) => item.id === id);
                const key = retryKeys.current[id] || `attempt:${id}:${keyId()}`;
                retryKeys.current[id] = key;
                if (job?.provider === 'mock') await retryGpuAttempt(id, key);
                else await retryGpuJob(id, key);
                delete retryKeys.current[id];
                await refreshJobs();
              })}
              onDetail={(job) => inspectJob(job.id)}
              selectedId={detailId}
              onRetain={(job) => void retainJob(job)}
              onClearHistory={clearJobs}
              onInstance={(job) => { const instance = instances.find((item) => instanceIdOf(item) === String(job.instance_id || '')); if (instance) { const id = instanceIdOf(instance); setRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: runtimeChoice.lifecycle, retainIdleMinutes: runtimeChoice.retainIdleMinutes }); setTab('算力'); } }}
            />
            {desktop && <section className={`${styles.card} ${styles.inspector}`} aria-label="任务详情">{jobDetailContent}</section>}
          </div>
        )}

        {tab === '作品' && (
          <div className={styles.detailWorkspace} data-gallery>
            <GalleryPanel
              key={projectId}
              gallery={projectGallery}
              error={viewer ? '' : galleryError}
              inspectedId={desktop ? viewer?.id || '' : undefined}
              onFavorite={(asset) => void favorite(asset)}
              onBatchDelete={batchRemove}
              onOpen={(asset, ordered) => {
                setViewerList(ordered || projectGallery);
                setViewer(asset);
                setPendingRouteAssetId('');
              }}
              onReuse={(asset) => void reuseAsset(asset)}
              onGoCreate={() => navigate('创作')}
            />
            {desktop && <section className={`${styles.card} ${styles.inspector} ${styles.assetInspector}`} aria-label="作品详情">
              {viewerContent || <><h2>作品详情</h2><p className="muted small">选择作品查看原图、视频和生成参数，支持键盘 ← / → 切换。</p></>}
            </section>}
          </div>
        )}

        {tab === '算力' && (
          <ComputePage
            settings={settings}
            usage={usage}
            instances={instances}
            onContinue={continueInstance}
            onOpenWorkbench={openInstanceWorkbench}
            onRefreshInstances={() => resourceRefresh.current?.()}
            onStop={() => setConfirm('stop')}
            onCheck={() => void checkGpu()}
            loadingStop={loading === 'stop'}
            loadingCheck={loading === 'check'}
            specs={specs}
            specsNote={specsNote}
            error={error}
          />
        )}

        {tab === '资产' && (
          <section className={styles.stack} aria-label="工作流与模型资产">
            <div className={styles.assetsSubBar}>
              <Segmented<AssetsSubTab>
                ariaLabel="资产分类"
                value={assetsSubTab}
                onChange={setAssetsSubTab}
                options={[
                  { value: 'workflows', label: '工作流', icon: <Layers size={16} strokeWidth={1.6} aria-hidden /> },
                  { value: 'models', label: '模型库', icon: <Boxes size={16} strokeWidth={1.6} aria-hidden /> },
                ]}
              />
            </div>

            {assetsSubTab === 'workflows' ? (
              <>
                {error && <p className="status error" role="alert">{error}</p>}
                <WorkflowsPanel
                  workflows={workflows}
                  onImport={(bundle) => void importWorkflow(bundle)}
                  onValidate={(wf) => void action('validate', async () => {
                    await validateGpuWorkflow(wf.id);
                    setWorkflows(await getGpuWorkflows());
                    pushToast(`已校验「${wf.workflow_key}」v${wf.workflow_version}`, 'success');
                  })}
                  onPublish={(wf) => void action('publish', async () => {
                    await publishGpuWorkflow(wf.id);
                    setWorkflows(await getGpuWorkflows());
                    pushToast(`已发布「${wf.workflow_key}」v${wf.workflow_version}`, 'success');
                  })}
                  onRollback={(wf) => void action('rollback', async () => {
                    await rollbackGpuWorkflow(wf.workflow_key, wf.workflow_version);
                    setWorkflows(await getGpuWorkflows());
                    pushToast(`已将 v${wf.workflow_version} 设为当前版本`, 'success');
                  })}
                  onDelete={adminRole ? (wf) => void action('delete-workflow', async () => {
                    await deleteGpuWorkflow(wf.id);
                    setWorkflows(current => current.filter(item => item.id !== wf.id));
                    pushToast(`已删除「${wf.workflow_key}」v${wf.workflow_version}`, 'success');
                  }) : undefined}
                  deleting={loading === 'delete-workflow'}
                  onDeactivate={(wf) => void action('deactivate', async () => {
                    await deactivateGpuWorkflow(wf.id);
                    setWorkflows(await getGpuWorkflows());
                    pushToast(`已停用「${wf.workflow_key}」v${wf.workflow_version}`, 'info');
                  })}
                  projectId={projectId}
                  projectTitle={activeProject ? projectOptionLabel(activeProject, projects) : ''}
                  onOpenJob={(jobId, jobProjectId) => {
                    if (jobProjectId) chooseProject(jobProjectId);
                    setTab('任务');
                    inspectJob(jobId);
                  }}
                  onUseProject={(id) => { chooseProject(id); setTab('任务'); }}
                  onActivated={() => { void getGpuWorkflows().then(setWorkflows); void getGpuModels().then(setModels).catch(() => undefined); }}
                />
              </>
            ) : (
              <ModelLibraryPage onOpenSettings={() => navigate('设置')} />
            )}
          </section>
        )}

        {tab === '原生 ComfyUI' && <NativeComfyPage accounts={accounts} instances={instances} desktop={desktop} />}

        {tab === '设置' && (
          <SettingsPanel
            accounts={accounts}
            onAccountsChange={setAccounts}
            settings={settings}
            usage={usage}
            error={error}
            onOpenCompute={() => navigate('算力')}
            onLogout={() => {
              sessionStorage.removeItem('colab-gpu-studio-token');
              setLoggedIn(false);
            }}
            onBack={() => setTab('创作')}
          />
        )}
      </main>

      {/* 确认生成弹窗 */}
      {confirm === 'submit' && activeWorkflow && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setConfirm(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="确认提交任务"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>确认开始生成</h2>
            <dl className={styles.kv}>
              <dt>工作流</dt><dd>{activeWorkflow.workflow_key} · v{activeWorkflow.workflow_version}</dd>
              <dt>运行方式</dt><dd>{runtime === 'colab' ? 'Colab GPU（会扣 CU）' : '离线模拟（不扣 CU）'}</dd>
              {runtime === 'colab' && <><dt>GPU 规格</dt><dd>{computeLabel(selectedCompute || manifest?.compute_spec as Record<string, unknown>)}</dd></>}
              <dt>机器</dt><dd>{resolveInstanceSubmission(runtimeChoice, instances).machineLabel}</dd>
              <dt>数量</dt><dd>{batchCount}</dd>
              {runtime === 'colab' && <><dt>最长运行</dt><dd>{maxRuntime} 分钟</dd></>}
              {runtime === 'colab' && (runtimeChoice.lifecycle === 'smart' || runtimeChoice.lifecycle === 'retain') && (
                <><dt>空闲保留</dt><dd>{runtimeChoice.retainIdleMinutes} 分钟</dd></>
              )}
            </dl>
            <label className={styles.checkRow}>
              <input
                type="checkbox"
                checked={rememberSkipConfirm}
                onChange={(e) => setRememberSkipConfirm(e.target.checked)}
              />
              <span className="small muted">本次会话相同配置直接提交，不再弹窗确认</span>
            </label>
            {error && <p className="status error" role="alert">{error}</p>}
            <div className="actions">
              <button className="button secondary" type="button" onClick={() => setConfirm(null)}>取消</button>
              <button className="button" disabled={loading === 'submit'} type="button" onClick={() => void submitJob()}>
                {loading === 'submit' ? '提交中…' : '确认启动'}
              </button>
            </div>
          </section>
        </div>
      )}

      {/* 确认急停弹窗 */}
      {confirm === 'stop' && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setConfirm(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="确认停止"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>停止全部任务？</h2>
            <p className="muted">会取消所有正在跑的任务，并关闭对应的 Colab GPU 机器。已经花掉的 CU 不会退回。</p>
            <div className="actions">
              <button className="button secondary" type="button" onClick={() => setConfirm(null)}>取消</button>
              <button className="button danger" type="button" onClick={() => void stopCompute()}>确认停止</button>
            </div>
          </section>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(confirmDeleteViewer)}
        title="删除此作品？"
        description={confirmDeleteViewer ? `将永久删除「${confirmDeleteViewer.filename || confirmDeleteViewer.id}」，删除后无法恢复。` : ''}
        confirmLabel="确认删除"
        danger
        onCancel={() => setConfirmDeleteViewer(null)}
        onConfirm={() => {
          if (confirmDeleteViewer) void remove(confirmDeleteViewer);
          setConfirmDeleteViewer(null);
        }}
      />

      {detailId && tab === '任务' && !desktop && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setDetailId('')}>
          <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="任务详情" onClick={event => event.stopPropagation()}>
            {jobDetailContent}
          </section>
        </div>
      )}

      {viewer && !inlineViewer && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setViewer(null)}>
          <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="查看作品" onClick={event => event.stopPropagation()}>
            {viewerContent}
          </section>
        </div>
      )}

      <ToastStack items={toasts} onDismiss={(id) => setToasts((curr) => curr.filter((item) => item.id !== id))} />
    </div>
  );
}
