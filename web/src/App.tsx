import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ModelRunOverrides, type ModelOverride } from './components/ModelDependencies';
import { WorkflowGpuSelection, computeLabel } from './components/WorkflowGpuSelection';
import { CreatePanel } from './pages/CreatePage';
import { CreateResults } from './components/CreateResults';
import { LoginScreen } from './components/LoginScreen';
import { MoveProjectDialog, NewProjectDialog, ProjectSheet } from './components/ProjectDialogs';
import { GpuStatusPopover } from './components/GpuStatusPopover';
import type { GpuStatusActionTarget } from './lib/gpu-status-summary';
import type { JobClearRequest } from './pages/JobsPage';
import { useGalleryState, type GallerySource } from './hooks/useGalleryState';

const WorkflowsPanel = lazy(() => import('./pages/WorkflowManager').then((m) => ({ default: m.WorkflowsPanel })));
const CreationApiPage = lazy(() => import('./pages/CreationApiPage').then((m) => ({ default: m.CreationApiPage })));
const NativeComfyPage = lazy(() => import('./pages/NativeComfyPage').then((m) => ({ default: m.NativeComfyPage })));
const ModelLibraryPage = lazy(() => import('./pages/ModelLibraryPage').then((m) => ({ default: m.ModelLibraryPage })));
const SettingsPanel = lazy(() => import('./pages/SettingsPage').then((m) => ({ default: m.SettingsPanel })));
const GalleryPanel = lazy(() => import('./pages/GalleryPage').then((m) => ({ default: m.GalleryPanel })));
const TasksPanel = lazy(() => import('./pages/JobsPage').then((m) => ({ default: m.TasksPanel })));
const ComputePage = lazy(() => import('./pages/ComputePage').then((m) => ({ default: m.ComputePage })));
const JobDetailInspector = lazy(() => import('./components/JobDetailInspector').then((m) => ({ default: m.JobDetailInspector })));
const GalleryViewer = lazy(() => import('./components/GalleryViewer').then((m) => ({ default: m.GalleryViewer })));
const NativeGalleryPanel = lazy(() => import('./components/NativeGalleryPanel').then((m) => ({ default: m.NativeGalleryPanel })));
import {
  Sparkles, Zap, Image as ImageIcon, Cpu, Settings as SettingsIcon, Folder, ChevronDown, Plus,
  Triangle, Boxes, Workflow, Layers, KeyRound, ArrowLeft, HardDrive,
} from 'lucide-react';
import {
  apiMessage, cancelGpuJob, clearGpuJobHistory, createGpuJob, createGpuBatch,
  createProject, deleteGpuPreset, deleteGpuWorkflow, deleteProject,
  apiFetch, exportGpuWorkflow,
  getGpuComputeSpecs, getGpuFixedPrompts, getGpuJob, getGpuJobs, getGpuModels, listGpuInstances,
  getGpuPresets, getReferenceAssets, getGpuSettings, getGpuSpecOrder, getGpuUsage, getGpuWorkflowSchema, getGpuWorkflows, getProjects,
  importGpuWorkflow,
  isLoggedIn, previewGpuJobReuse, publishGpuWorkflow, renameProject, retryGpuJob, retryGpuAttempt, saveGpuFixedPrompts, setDefaultProject, stopGpuCompute,
  updateGpuPreset, uploadAsset,
  validateGpuWorkflow, rollbackGpuWorkflow, deactivateGpuWorkflow, saveGpuPreset, ApiError,
  type ComputeSpec, type GalleryAsset, type GpuJob, type GpuPreset,
  type GpuManifest, type GpuModel, type GpuSettings, type GpuSpecKey, type GpuUsage, type GpuWorkflow, type Project, type Provider, type ReusePreviewResponse,
  type WorkflowExportResponse,
} from './lib/gpu-api';
import type { WorkflowImportResult } from './components/WorkflowImportWorkspace';
import { formatExportSummaryMessage, parseWorkflowVersionConflict } from './lib/workflow-import';
import type { GpuAccount, GpuInstance } from './components/AccountDirectory';
import { isAccountSelectable } from './components/RuntimeSelection';
import type { RuntimeChoice } from './pages/CreatePage';
import { referenceAssets, supportsReferences } from './components/ReferencePicker';
import { Segmented } from './components/ui';
import { ConfirmDialog, ToastStack, type ToastItem } from './components/Feedback';
import { resolveGpuCheckRequest } from './lib/colab-check';
import { DEFAULT_GPU_SPEC_ORDER, isBelowWorkflowDefault } from './lib/gpu-spec-order';
import {
  buildConfirmActions, formatReuseDecisionTitle, isCreationPoolInstance,
  resolveInstanceSubmission, shouldBypassSkipConfirm, type ConfirmActionOption,
} from './lib/instance-submission';
import type { TransferFocusTarget } from './lib/domain-transfer';
import { isSystemJob, pickInitialProjectId, projectOptionLabel, projectTitleById } from './lib/project-label';
import { carriedDraft, mergeWorkflowDraft, readWorkflowDraft, writeWorkflowDraft } from './lib/workflow-draft';
import {
  applyFixedPromptsToParams,
  collectActiveLoraTriggerWords,
  createFixedPromptSyncer,
  defaultFixedPromptConfig,
  fixedPromptStorageKey,
  normalizeFixedPromptConfig,
  readCachedFixedPrompts,
  restoreFromHistoricalJob,
  type FixedPromptConfig,
  type FixedPromptSyncer,
  type PromptComposition,
} from './lib/fixed-prompts';
import {
  dependencyApi,
  libraryApi,
  readStoredWorkflowInspect,
  writeStoredWorkflowInspect,
  type Catalog,
  type DependencyRecord,
  type LibraryModel,
} from './lib/model-library';
import { useDesktopLayout } from './lib/desktop-layout';
import { useBodyScrollLock } from './lib/modal-scroll-lock';
import { parameterKey, parameterList } from './lib/ui-helpers';
import { useSmartPoll } from './lib/smart-poll';
import { formatHashRoute, parseHashRoute, type AssetsSubTab, type MainTab } from './lib/route-state';
import {
  assetIdsFromParams, instanceIdOf, keyId, loadCreateRuntimePrefs, record,
  saveCreateRuntimePrefs, savedProjectId, saveProjectId, uniqueBy,
} from './lib/create-runtime-prefs';
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
const activeInstanceStates = new Set(['requested', 'allocating', 'preparing', 'ready', 'busy', 'retained_idle', 'transferring', 'releasing']);

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
  const [settingsFocusSection, setSettingsFocusSection] = useState<'model-storage' | null>(null);
  const [focusTransferTarget, setFocusTransferTarget] = useState<TransferFocusTarget | null>(null);
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
      apiFetch<{ role: string }>('/api/me').then((user) => {
        if (current) setAdminRole(user.role === 'admin');
      }).catch(() => {});
    }
    return () => { current = false; };
  }, [loggedIn]);

  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState('');
  const [jobsProjectFilter, setJobsProjectFilter] = useState('');
  const [showSystemJobs, setShowSystemJobs] = useState(false);
  const [referenceScope, setReferenceScope] = useState<'project' | 'all'>('project');
  const [pinnedReferenceAssets, setPinnedReferenceAssets] = useState<GalleryAsset[]>([]);
  const [moveDialog, setMoveDialog] = useState<{
    ids: string[];
    currentProjectId?: string;
    onMoved?: (movedIds: string[]) => void;
  } | null>(null);
  const [showProjectSheet, setShowProjectSheet] = useState(false);
  const [showNewProject, setShowNewProject] = useState(false);
  const [showGpuStatus, setShowGpuStatus] = useState(false);
  const lastStatusRefreshRef = useRef(0);
  const currentProjectId = useRef(projectId);
  currentProjectId.current = projectId;

  const [models, setModels] = useState<GpuModel[]>([]);
  const [workflows, setWorkflows] = useState<GpuWorkflow[]>([]);
  const [presets, setPresets] = useState<GpuPreset[]>([]);
  const [modelId, setModelId] = useState('');
  const [workflowId, setWorkflowId] = useState('');
  const [gpuChoices, setGpuChoices] = useState<Record<string, Record<string, unknown> | null>>(() => {
    try { return JSON.parse(localStorage.getItem('studio-workflow-gpus') || '{}'); } catch { return {}; }
  });
  const selectedCompute = gpuChoices[workflowId] || null;
  const [modelOverrides, setModelOverrides] = useState<Record<string, ModelOverride[]>>({});
  function chooseGpu(key: Record<string, unknown> | null) {
    const next = { ...gpuChoices, [workflowId]: key }; setGpuChoices(next); idempotencyKey.current = '';
    try { localStorage.setItem('studio-workflow-gpus', JSON.stringify(next)); } catch { /* browser storage unavailable */ }
  }
  const [manifest, setManifest] = useState<GpuManifest | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [referenceImages, setReferenceImages] = useState<GalleryAsset[]>([]);
  const [jobs, setJobs] = useState<GpuJob[]>([]);
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const initialRuntimePrefs = useMemo(() => loadCreateRuntimePrefs(), []);
  const [runtime, setRuntime] = useState<Provider>(() => initialRuntimePrefs.runtime);
  const [maxRuntime, setMaxRuntime] = useState(() => initialRuntimePrefs.maxRuntime);
  const [batchCount, setBatchCount] = useState(1);
  const [batchMode, setBatchMode] = useState<'save' | 'smart' | 'batch' | 'retain'>(() => initialRuntimePrefs.lifecycle);
  const [runtimeChoice, setRuntimeChoice] = useState<RuntimeChoice>(() => ({
    accountId: initialRuntimePrefs.accountId,
    instanceId: '',
    lifecycle: initialRuntimePrefs.lifecycle,
    retainIdleMinutes: initialRuntimePrefs.retainIdleMinutes,
  }));
  const [accounts, setAccounts] = useState<GpuAccount[]>([]);
  const [accountsLoaded, setAccountsLoaded] = useState(false);
  const [instances, setInstances] = useState<GpuInstance[]>([]);
  const [, setResourceError] = useState('');
  const resourceRefresh = useRef<(() => void) | null>(null);
  const submitting = useRef(false);
  const [settings, setSettings] = useState<GpuSettings | null>(null);
  const [usage, setUsage] = useState<GpuUsage | null>(null);
  const usageSequence = useRef(0);
  const refreshUsage = useCallback(async (signal?: AbortSignal) => {
    const sequence = ++usageSequence.current;
    const value = await getGpuUsage(signal);
    if (!signal?.aborted && sequence === usageSequence.current) setUsage(value);
  }, []);
  const [specs, setSpecs] = useState<ComputeSpec[]>([]);
  const [specsNote, setSpecsNote] = useState('');
  const [error, setError] = useState('');
  const [jobError, setJobError] = useState('');
  const [loading, setLoading] = useState('');
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [confirm, setConfirm] = useState<'submit' | 'stop' | null>(null);
  const [reusePreview, setReusePreview] = useState<ReusePreviewResponse | null>(null);
  const [gpuOrder, setGpuOrder] = useState<GpuSpecKey[]>(DEFAULT_GPU_SPEC_ORDER);
  const lastSubmitActionSig = useRef('');
  const [skipConfirmSig, setSkipConfirmSig] = useState('');
  const [rememberSkipConfirm, setRememberSkipConfirm] = useState(false);
  const [detail, setDetail] = useState<GpuJob | null>(null);
  const [detailId, setDetailId] = useState(initialRoute.jobId);
  const [detailError, setDetailError] = useState('');
  const [detailRefresh, setDetailRefresh] = useState(0);
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
  const pendingReuseRef = useRef<{
    workflowId: string;
    resolvedParams: Record<string, unknown>;
    promptComposition?: PromptComposition | null;
  } | null>(null);
  const draftLocation = useRef({ projectId: '', workflowId: '' });
  const [fixedPrompts, setFixedPrompts] = useState<FixedPromptConfig>(() => defaultFixedPromptConfig());
  const [fixedPromptsSyncing, setFixedPromptsSyncing] = useState(false);
  const fixedPromptsRef = useRef(fixedPrompts);
  fixedPromptsRef.current = fixedPrompts;
  const fixedPromptSyncRef = useRef<FixedPromptSyncer | null>(null);
  if (!fixedPromptSyncRef.current) {
    fixedPromptSyncRef.current = createFixedPromptSyncer({
      save: saveGpuFixedPrompts,
      onSyncingChange: setFixedPromptsSyncing,
    });
  }
  const fixedPromptTimersRef = useRef<Map<string, number>>(new Map());
  const [workflowDependencyState, setWorkflowDependencyState] = useState<{ workflowId: string; record: DependencyRecord } | null>(null);
  const [libraryModelsCatalog, setLibraryModelsCatalog] = useState<LibraryModel[] | null>(null);

  const promptKeysOf = useCallback((targetManifest: GpuManifest | null | undefined) => {
    const plist = parameterList(targetManifest || null);
    const keys = plist.map((p, i) => parameterKey(p, i));
    const promptKey = keys.find((k) => k === 'prompt')
      || keys.find((k) => /prompt|caption/i.test(k) && !/negative/i.test(k))
      || null;
    const negativePromptKey = keys.find((k) => /(^|_)negative(_prompt)?$/i.test(k)) || null;
    return { promptKey, negativePromptKey };
  }, []);

  const flushFixedPrompts = useCallback(async (
    override?: { projectId: string; workflowId: string },
  ) => {
    const syncer = fixedPromptSyncRef.current;
    if (!syncer) return;
    if (override) {
      if (!override.projectId || !override.workflowId) return;
      const key = fixedPromptStorageKey(override.projectId, override.workflowId);
      const timer = fixedPromptTimersRef.current.get(key);
      if (timer !== undefined) {
        window.clearTimeout(timer);
        fixedPromptTimersRef.current.delete(key);
      }
      await syncer.flush(override.projectId, override.workflowId);
      return;
    }
    for (const timer of fixedPromptTimersRef.current.values()) {
      window.clearTimeout(timer);
    }
    fixedPromptTimersRef.current.clear();
    await syncer.flushAll();
  }, []);

  const updateFixedPrompts = useCallback((
    next: FixedPromptConfig,
    opts: { immediate?: boolean; projectId?: string; workflowId?: string } = {},
  ) => {
    const pid = opts.projectId ?? currentProjectId.current;
    const wid = opts.workflowId ?? workflowIdRef.current;
    const normalized = normalizeFixedPromptConfig(next);
    if (pid === currentProjectId.current && wid === workflowIdRef.current) {
      setFixedPrompts(normalized);
      fixedPromptsRef.current = normalized;
    }
    if (!pid || !wid) return;
    const syncer = fixedPromptSyncRef.current;
    if (!syncer) return;
    syncer.markDirty(pid, wid, normalized);
    const key = fixedPromptStorageKey(pid, wid);
    const existingTimer = fixedPromptTimersRef.current.get(key);
    if (existingTimer !== undefined) {
      window.clearTimeout(existingTimer);
      fixedPromptTimersRef.current.delete(key);
    }
    if (opts.immediate) {
      void flushFixedPrompts({ projectId: pid, workflowId: wid });
    } else {
      const timer = window.setTimeout(() => {
        fixedPromptTimersRef.current.delete(key);
        void flushFixedPrompts({ projectId: pid, workflowId: wid });
      }, 400);
      fixedPromptTimersRef.current.set(key, timer);
    }
  }, [flushFixedPrompts]);

  const activeLoraTriggerWords = useMemo(
    () => collectActiveLoraTriggerWords(
      workflowDependencyState?.workflowId === workflowId ? workflowDependencyState.record.dependencies : undefined,
      libraryModelsCatalog,
      modelOverrides[workflowId],
    ),
    [workflowDependencyState, libraryModelsCatalog, modelOverrides, workflowId],
  );

  const activeProject = projects.find((p) => p.id === projectId);
  const activeWorkflow = workflows.find((item) => item.id === workflowId);
  const workflowParamsReady = Boolean(projectId && workflowId && draftReady.current === `${projectId}:${workflowId}`);
  const createProjectJobs = useMemo(
    () => jobs.filter((job) => job.project_id === projectId),
    [jobs, projectId],
  );
  const latestJob = createProjectJobs.find((job) => !job.prep && !isSystemJob(job));
  const currentSubmission = submission?.projectId === projectId ? submission : null;
  const previewJobs = currentSubmission
    ? createProjectJobs.filter((job) => currentSubmission.ids.includes(job.id))
    : latestJob?.batch_id
      ? createProjectJobs.filter((job) => job.batch_id === latestJob.batch_id)
      : latestJob ? [latestJob] : [];
  const galleryOutputs = (tab === '创作' || tab === '作品')
    ? jobs.flatMap((job) => job.output_asset_ids || []).join(',')
    : '';
  const detailRevision = jobs.find((job) => job.id === detailId)?.updated_at;

  const activeJobsCount = useMemo(
    () => jobs.filter((j) => !isSystemJob(j) && !terminalStates.has(String(j.state || '').toLowerCase())).length,
    [jobs],
  );
  const hasAnyActiveJob = useMemo(
    () => jobs.some((j) => !terminalStates.has(String(j.state || '').toLowerCase())),
    [jobs],
  );
  const hasActiveInstances = useMemo(
    () => instances.some((i) => !i.runtime_deleted && activeInstanceStates.has(String(i.state || '').toLowerCase())),
    [instances],
  );
  const availableReferenceImages = useMemo(
    () => uniqueBy([...pinnedReferenceAssets, ...referenceImages], (asset) => asset.id),
    [pinnedReferenceAssets, referenceImages],
  );

  const bindReferenceAsset = useCallback((asset: GalleryAsset) => {
    setPinnedReferenceAssets((curr) => [asset, ...curr.filter((item) => item.id !== asset.id)]);
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
    setTab('创作');
    setCreationApiOpen(false);
    pushToast(`已将「${asset.filename || '作品'}」设为参考素材`, 'success');
  }, [manifest, pushToast]);

  const gs = useGalleryState({
    projectId,
    projects,
    loggedIn,
    tab,
    desktop,
    galleryOutputs,
    initialRouteAssetId: initialRoute.assetId,
    pushToast,
    onLoadedAgainJob: setAgainJob,
    onUseReference: bindReferenceAsset,
  });

  function useAssetAsReference(asset: GalleryAsset) {
    gs.closeViewers();
    bindReferenceAsset(asset);
  }

  // Sync route state to URL hash & listen to browser back/forward
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const nextHash = formatHashRoute({
      tab,
      assetsSubTab,
      creationApiOpen,
      jobId: tab === '任务' ? detailId : '',
      assetId: gs.routeAssetId,
    });
    if (lastRouteHash.current === nextHash) return;
    lastRouteHash.current = nextHash;
    if (window.location.hash !== nextHash) {
      window.history.pushState(null, '', nextHash);
    }
  }, [tab, assetsSubTab, creationApiOpen, detailId, gs.routeAssetId]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onHashChange = () => {
      const parsed = parseHashRoute(window.location.hash);
      lastRouteHash.current = formatHashRoute(parsed);
      setTab(parsed.tab);
      setAssetsSubTab(parsed.assetsSubTab);
      setCreationApiOpen(parsed.creationApiOpen);
      setDetailId(parsed.jobId);
      gs.setPendingRouteAssetId(parsed.assetId);
      gs.closeViewers();
      gs.setViewerList([]);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, [gs.closeViewers, gs.setPendingRouteAssetId, gs.setViewerList]);

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
    ]).then((results) => {
      if (!alive) return;
      const [projR, modR, wfR, preR, setR] = results;
      if (projR.status === 'fulfilled') {
        setProjects(projR.value);
        const list = projR.value;
        setProjectId((curr) => {
          if (curr && list.some((p) => p.id === curr)) return curr;
          return pickInitialProjectId(list, savedProjectId());
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
    }).catch((cause) => alive && setError(apiMessage(cause)));
    return () => { alive = false; };
  }, [loggedIn, loginRefresh]);

  const refreshResources = useCallback(async (signal?: AbortSignal) => {
    const seq = ++resourceSequenceRef.current;
    try {
      const [accountResult, instanceResult] = await Promise.all([
        apiFetch<unknown>('/api/gpu/accounts', { signal }),
        listGpuInstances({}, signal),
      ]);
      if (signal?.aborted || seq < appliedResourceSequenceRef.current) return;
      appliedResourceSequenceRef.current = seq;
      const nextAccounts = (Array.isArray(accountResult) ? accountResult : record(accountResult) && Array.isArray(accountResult.data) ? accountResult.data : []) as GpuAccount[];
      const nextInstances = instanceResult as unknown as GpuInstance[];
      setAccounts(uniqueBy(nextAccounts, (account) => account.account_id));
      setAccountsLoaded(true);
      setInstances(uniqueBy(nextInstances, (instance) => instanceIdOf(instance)));
      setResourceError('');
    } catch (cause) {
      if (!signal?.aborted && seq === resourceSequenceRef.current) setResourceError(`账号/实例：${apiMessage(cause)}`);
    }
  }, []);

  useEffect(() => {
    saveCreateRuntimePrefs({
      runtime,
      accountId: runtimeChoice.accountId || '',
      lifecycle: runtimeChoice.lifecycle,
      retainIdleMinutes: Number.isInteger(runtimeChoice.retainIdleMinutes) && runtimeChoice.retainIdleMinutes >= 1
        ? runtimeChoice.retainIdleMinutes : 15,
      maxRuntime: Number.isInteger(maxRuntime) && maxRuntime >= 1 ? maxRuntime : 30,
    });
  }, [runtime, runtimeChoice.accountId, runtimeChoice.lifecycle, runtimeChoice.retainIdleMinutes, maxRuntime]);

  useEffect(() => {
    if (!settings) return;
    const realAvailable = Boolean(settings.real_enabled) && settings.offline_mode !== true;
    if (!realAvailable && runtime === 'colab') setRuntime('mock');
  }, [settings, runtime]);

  useEffect(() => {
    if (!accountsLoaded || !runtimeChoice.accountId) return;
    const matched = accounts.find((account) => account.account_id === runtimeChoice.accountId);
    if (!matched || !isAccountSelectable(matched)) {
      setRuntimeChoice((current) => (current.accountId === runtimeChoice.accountId
        ? { ...current, accountId: null, instanceId: null }
        : current));
    }
  }, [accountsLoaded, accounts, runtimeChoice.accountId]);

  useEffect(() => {
    resourceRefresh.current = () => void refreshResources();
    return () => { resourceRefresh.current = null; };
  }, [refreshResources]);

  useSmartPoll(async (signal) => {
    try { await refreshUsage(signal); } catch { /* Retry on the next refresh. */ }
  }, {
    enabled: loggedIn,
    active: tab === '算力' || tab === '设置' || tab === '原生 ComfyUI' || hasActiveInstances,
    activeMs: 10000,
    idleMs: 25000,
  });

  useSmartPoll(refreshResources, {
    enabled: loggedIn,
    active: hasActiveInstances || hasAnyActiveJob || tab === '算力',
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
    const syncer = fixedPromptSyncRef.current;
    if (
      syncer
      && previous.projectId
      && previous.workflowId
      && (previous.projectId !== projectId || previous.workflowId !== workflowId)
      && syncer.getPending(previous.projectId, previous.workflowId)
    ) {
      void flushFixedPrompts({ projectId: previous.projectId, workflowId: previous.workflowId });
    }
    const carried = sameProject && previous.workflowId && previous.workflowId !== workflowId ? carriedDraft(paramsRef.current) : null;
    draftLocation.current = { projectId, workflowId };
    let alive = true;
    const requestWorkflow = workflowId;
    const requestProject = projectId;
    const pendingFixed = syncer?.getPending(requestProject, requestWorkflow)?.config ?? null;
    const cachedFixed = readCachedFixedPrompts(requestProject, requestWorkflow);
    const optimisticFixed = pendingFixed ?? cachedFixed ?? defaultFixedPromptConfig();
    setFixedPrompts(optimisticFixed);
    fixedPromptsRef.current = optimisticFixed;
    const loadVersion = syncer?.keyVersion(requestProject, requestWorkflow) ?? 0;
    setLoading('schema');
    Promise.all([
      getGpuWorkflowSchema(requestWorkflow),
      getGpuFixedPrompts(requestProject, requestWorkflow).catch(() => null),
    ]).then(([next, remoteFixed]) => {
      if (!alive) return;
      const pendingBeforeApply = syncer?.getPending(requestProject, requestWorkflow)?.config ?? null;
      const loadedFixed = pendingBeforeApply
        ?? (remoteFixed ? normalizeFixedPromptConfig(remoteFixed) : (cachedFixed ?? defaultFixedPromptConfig()));
      const pendingReuse = pendingReuseRef.current;
      if (pendingReuse && pendingReuse.workflowId === requestWorkflow) {
        pendingReuseRef.current = null;
        incomingDraft.current = null;
        const { promptKey, negativePromptKey } = promptKeysOf(next);
        const restored = restoreFromHistoricalJob(
          pendingReuse.resolvedParams,
          pendingReuse.promptComposition,
          loadedFixed,
          promptKey,
          negativePromptKey,
        );
        setManifest(next);
        draftReady.current = `${requestProject}:${requestWorkflow}`;
        setParams(mergeWorkflowDraft(next, restored.params, null));
        updateFixedPrompts(restored.fixedPrompts, {
          immediate: true,
          projectId: requestProject,
          workflowId: requestWorkflow,
        });
        pushToast(
          restored.pausedLegacy
            ? '已载入历史参数；为避免重复拼接，已暂停当前固定提示词'
            : '已载入历史参数到创作工作区',
          'success',
        );
        return;
      }
      const forced = incomingDraft.current;
      incomingDraft.current = null;
      const saved = forced ?? readWorkflowDraft(requestProject, requestWorkflow);
      setManifest(next);
      draftReady.current = `${requestProject}:${requestWorkflow}`;
      setParams(mergeWorkflowDraft(next, saved, forced ? null : carried));
      if (syncer && syncer.keyVersion(requestProject, requestWorkflow) === loadVersion) {
        const resolved = syncer.resolveLoaded(requestProject, requestWorkflow, remoteFixed);
        setFixedPrompts(resolved.config);
        fixedPromptsRef.current = resolved.config;
        if (resolved.hasPending && remoteFixed !== null) {
          void flushFixedPrompts({ projectId: requestProject, workflowId: requestWorkflow });
        }
      }
    }).catch((cause) => alive && setError(`参数 schema：${apiMessage(cause)}`)).finally(() => alive && setLoading(''));
    return () => { alive = false; };
  }, [workflowId, projectId, flushFixedPrompts, promptKeysOf, pushToast, updateFixedPrompts]);

  useEffect(() => {
    const flushOnLeave = () => {
      void flushFixedPrompts();
    };
    window.addEventListener('pagehide', flushOnLeave);
    return () => window.removeEventListener('pagehide', flushOnLeave);
  }, [flushFixedPrompts]);

  useEffect(() => {
    if (!loggedIn || !workflowId || tab !== '创作') return;
    let alive = true;
    const targetWorkflowId = workflowId;
    void Promise.all([
      dependencyApi<DependencyRecord>(targetWorkflowId).catch(() => null),
      libraryApi<Catalog>().catch(() => null),
    ]).then(([deps, catalog]) => {
      if (!alive || workflowIdRef.current !== targetWorkflowId) return;
      if (deps) setWorkflowDependencyState({ workflowId: targetWorkflowId, record: deps });
      if (catalog?.items) setLibraryModelsCatalog(catalog.items);
    });
    return () => { alive = false; };
  }, [loggedIn, workflowId, tab]);

  useEffect(() => {
    if (draftReady.current !== `${projectId}:${workflowId}` || !workflowId || !projectId) return;
    const pid = projectId;
    const wid = workflowId;
    const snapshot = params;
    const timer = window.setTimeout(() => {
      writeWorkflowDraft(pid, wid, snapshot);
    }, 250);
    const flush = () => writeWorkflowDraft(pid, wid, paramsRef.current);
    window.addEventListener('pagehide', flush);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('pagehide', flush);
    };
  }, [params, projectId, workflowId]);

  const refreshJobs = useCallback(async (signal?: AbortSignal) => {
    const seq = ++jobSequence.current;
    const current = () => !signal?.aborted && seq === jobSequence.current;
    setRefreshingJobs(true);
    try {
      const next = await getGpuJobs(undefined, signal);
      if (!current()) return;
      setJobs(next);
      setJobError('');
    } catch (cause) {
      if (current()) setJobError(apiMessage(cause));
    } finally {
      if (seq === jobSequence.current) setRefreshingJobs(false);
    }
  }, []);

  useSmartPoll(refreshJobs, {
    enabled: loggedIn,
    active: hasAnyActiveJob,
    activeMs: 4000,
    idleMs: 25000,
  });

  useEffect(() => {
    setPinnedReferenceAssets([]);
    setSelectedAssets([]);
  }, [projectId]);

  useEffect(() => {
    setReferenceImages([]);
    if (!loggedIn) return;
    if (referenceScope === 'project' && !projectId) return;
    let alive = true;
    getReferenceAssets(referenceScope === 'all' ? undefined : projectId).then((next) => {
      if (alive) setReferenceImages(referenceAssets(next));
    }).catch((cause) => {
      if (alive) setError(`素材：${apiMessage(cause)}`);
    });
    return () => { alive = false; };
  }, [projectId, referenceScope, loggedIn]);

  useEffect(() => {
    setSelectedAssets([]);
  }, [workflowId]);

  useEffect(() => {
    if (tab !== '任务' || !detailId || !loggedIn) return;
    let alive = true;
    getGpuJob(detailId).then((job) => {
      if (!alive || job.id !== detailId) return;
      setDetail(job);
      setDetailError('');
    }).catch((cause) => {
      if (alive) setDetailError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [tab, detailId, detailRevision, detailRefresh, loggedIn]);

  useEffect(() => {
    if (againJob && (againJob.resolved_params || againJob.params)) {
      const restored = { ...(againJob.resolved_params || againJob.params) };
      setModelId(String(againJob.model_id || ''));
      const workflow = workflows.find((item) => item.workflow_key === againJob.workflow_key && String(item.workflow_version) === String(againJob.workflow_version));
      if (workflow) {
        const sameLoaded = workflow.id === workflowIdRef.current
          && draftReady.current === `${currentProjectId.current}:${workflow.id}`;
        if (sameLoaded) {
          const { promptKey, negativePromptKey } = promptKeysOf(manifest);
          const result = restoreFromHistoricalJob(
            restored,
            againJob.prompt_composition,
            fixedPromptsRef.current,
            promptKey,
            negativePromptKey,
          );
          setParams(mergeWorkflowDraft(manifest, result.params, null));
          updateFixedPrompts(result.fixedPrompts, { immediate: true });
          pushToast(
            result.pausedLegacy
              ? '已载入历史参数；为避免重复拼接，已暂停当前固定提示词'
              : '已载入历史参数到创作工作区',
            'success',
          );
        } else {
          pendingReuseRef.current = {
            workflowId: workflow.id,
            resolvedParams: restored,
            promptComposition: againJob.prompt_composition,
          };
          setWorkflowId(workflow.id);
        }
      } else {
        setParams(restored);
        pushToast('已载入历史参数到创作工作区', 'success');
      }
      setAgainJob(null);
      setTab('创作');
    }
  }, [againJob, workflows, manifest, promptKeysOf, pushToast, updateFixedPrompts]);

  function chooseProject(id: string, opts: { syncView?: boolean } = {}) {
    setProjectId(id);
    saveProjectId(id);
    if (opts.syncView) {
      if (tab === '作品') {
        gs.setGalleryProjectFilter(id);
        gs.setViewer(null);
        gs.setPendingRouteAssetId('');
      } else if (tab === '任务') {
        setJobsProjectFilter(id);
        setDetailId('');
        setDetail(null);
      }
    }
  }

  async function createLocalProject(title: string): Promise<boolean> {
    try {
      const created = await createProject(title);
      setProjects((curr) => {
        const defaults = curr.filter((item) => item.is_default);
        const others = curr.filter((item) => !item.is_default);
        return [...defaults, created, ...others];
      });
      chooseProject(created.id, { syncView: true });
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
      setProjects((curr) => curr.map((item) => (item.id === id ? { ...item, ...updated } : item)));
      pushToast(`项目已重命名为「${updated.title}」`, 'success');
      return true;
    } catch (cause) {
      pushToast(`重命名失败：${apiMessage(cause)}`, 'error');
      return false;
    }
  }

  async function setDefaultLocalProject(id: string): Promise<boolean> {
    try {
      const updated = await setDefaultProject(id);
      setProjects((curr) => {
        const next = curr.map((item) => (item.id === id ? { ...item, ...updated, is_default: true } : { ...item, is_default: false }));
        const target = next.find((item) => item.id === id);
        return target ? [target, ...next.filter((item) => item.id !== id)] : next;
      });
      pushToast(`已将「${updated.title}」设为默认项目`, 'success');
      return true;
    } catch (cause) {
      pushToast(`设置默认失败：${apiMessage(cause)}`, 'error');
      return false;
    }
  }

  async function removeLocalProject(project: Project): Promise<boolean> {
    try {
      await deleteProject(project.id);
      const remaining = projects.filter((item) => item.id !== project.id);
      setProjects(remaining);
      if (projectId === project.id) chooseProject(pickInitialProjectId(remaining, ''));
      if (gs.galleryProjectFilter === project.id) gs.setGalleryProjectFilter('');
      if (jobsProjectFilter === project.id) setJobsProjectFilter('');
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
      if (currentProjectId.current === uploadProjectId) {
        setUploadProgress((curr) => {
          const next = { ...curr };
          delete next[file.name];
          return next;
        });
      }
    }
  }

  useEffect(() => {
    if (!loggedIn) return;
    const controller = new AbortController();
    let alive = true;
    void getGpuSpecOrder(controller.signal).then((res) => {
      if (alive && Array.isArray(res.ordered_specs) && res.ordered_specs.length) {
        setGpuOrder(res.ordered_specs);
      }
    }).catch(() => {});
    return () => {
      alive = false;
      controller.abort();
    };
  }, [loggedIn, loginRefresh]);

  const workflowDefaultSpec = (manifest?.compute_spec as Record<string, unknown> | undefined) || null;
  const specBelowDefaultError = runtime === 'colab' && selectedCompute && workflowDefaultSpec
    && isBelowWorkflowDefault(selectedCompute, workflowDefaultSpec, gpuOrder)
    ? `所选 GPU（${computeLabel(selectedCompute)}）低于工作流默认规格（${computeLabel(workflowDefaultSpec)}），不支持降配运行；请改回默认规格或选择更高规格 GPU。`
    : '';

  function submitSignature() {
    return JSON.stringify([
      workflowId, runtime, batchCount, runtimeChoice.accountId, runtimeChoice.instanceId,
      runtimeChoice.lifecycle, selectedCompute, maxRuntime, runtimeChoice.retainIdleMinutes,
    ]);
  }

  async function requestSubmit() {
    if (submitting.current || !workflowParamsReady || !modelId || !activeWorkflow) return;
    if (specBelowDefaultError) {
      setError(specBelowDefaultError);
      return;
    }
    setError('');
    if (runtime === 'colab') {
      setLoading('preview');
      try {
        const preview = await previewGpuJobReuse({
          workflow_key: activeWorkflow.workflow_key,
          workflow_version: activeWorkflow.workflow_version,
          project_id: projectId,
          provider: runtime,
          account_id: runtimeChoice.accountId || null,
          instance_id: runtimeChoice.instanceId || null,
          compute_spec: selectedCompute,
          lifecycle: runtimeChoice.lifecycle,
          retain_idle_minutes: runtimeChoice.retainIdleMinutes,
          ...(modelOverrides[workflowId]?.length ? { model_overrides: modelOverrides[workflowId] as unknown as Array<Record<string, unknown>> } : {}),
        });
        setReusePreview(preview);
        if (preview.decision === 'unsupported_spec') {
          setError(preview.reason || '所选 GPU 规格低于工作流默认最低规格，不支持降配运行。');
          return;
        }
        const forced = shouldBypassSkipConfirm(preview);
        if (!forced && skipConfirmSig && skipConfirmSig === submitSignature()) {
          void submitJob();
          return;
        }
        setRememberSkipConfirm(false);
        setConfirm('submit');
        return;
      } catch (cause) {
        setError(`复用预检失败：${apiMessage(cause)}`);
        return;
      } finally {
        setLoading('');
      }
    }
    setReusePreview(null);
    if (skipConfirmSig && skipConfirmSig === submitSignature()) {
      void submitJob();
      return;
    }
    setRememberSkipConfirm(false);
    setConfirm('submit');
  }

  async function submitJob(actionChoice?: ConfirmActionOption) {
    if (submitting.current || !workflowParamsReady || !modelId || !activeWorkflow) return;
    if (specBelowDefaultError) {
      setError(specBelowDefaultError);
      return;
    }
    submitting.current = true;
    const reuseAction = actionChoice?.reuseAction || 'auto';
    const actionSig = JSON.stringify([reuseAction, actionChoice?.instanceId || null, actionChoice?.replaceInstanceId || null]);
    if (lastSubmitActionSig.current !== actionSig) {
      idempotencyKey.current = '';
      lastSubmitActionSig.current = actionSig;
    }
    if (!idempotencyKey.current) idempotencyKey.current = keyId();
    setLoading('submit');
    setError('');
    try {
      let allocation: 'auto' | 'existing' = 'auto';
      let resolvedAccountId: string | null = runtimeChoice.accountId || null;
      let resolvedInstanceId: string | null = null;
      let replaceInstanceId: string | null = null;
      if (reuseAction === 'replace') {
        allocation = 'auto';
        replaceInstanceId = actionChoice?.replaceInstanceId || null;
      } else if (reuseAction === 'new') {
        allocation = 'auto';
      } else if (reuseAction === 'reuse' && actionChoice?.instanceId) {
        allocation = 'existing';
        resolvedInstanceId = actionChoice.instanceId;
        resolvedAccountId = runtimeChoice.accountId || reusePreview?.target_instance?.account_id || null;
      } else {
        const decision = resolveInstanceSubmission(runtimeChoice, instances);
        if (!decision.ok) {
          setRuntimeChoice({ ...runtimeChoice, accountId: decision.accountId, instanceId: decision.instanceId });
          setError(decision.message);
          return;
        }
        allocation = decision.allocation;
        resolvedAccountId = decision.accountId;
        resolvedInstanceId = decision.instanceId;
      }
      if (fixedPromptSyncRef.current?.getPending(projectId, workflowId)) {
        void flushFixedPrompts({ projectId, workflowId });
      }
      const { promptKey, negativePromptKey } = promptKeysOf(manifest);
      const composed = applyFixedPromptsToParams(params, fixedPromptsRef.current, promptKey, negativePromptKey);
      const validAssetIds = new Set(referenceAssets(availableReferenceImages).map((a) => a.id));
      const body = {
        project_id: projectId,
        model_id: modelId,
        workflow_key: activeWorkflow.workflow_key,
        workflow_version: activeWorkflow.workflow_version,
        params: composed.params,
        ...(composed.promptComposition ? { prompt_composition: composed.promptComposition } : {}),
        ...(modelOverrides[workflowId]?.length ? { model_overrides: modelOverrides[workflowId] } : {}),
        asset_ids: supportsReferences(manifest)
          ? [...new Set([...selectedAssets.filter((id) => validAssetIds.has(id)), ...assetIdsFromParams(composed.params, validAssetIds)])]
          : [],
        execution_policy: {
          provider: runtime,
          ...(runtime === 'colab' && selectedCompute ? { compute_spec: selectedCompute } : {}),
          max_runtime_minutes: maxRuntime,
          allocation,
          account_id: resolvedAccountId,
          instance_id: resolvedInstanceId,
          ...(reuseAction !== 'auto' ? { reuse_action: reuseAction } : {}),
          ...(replaceInstanceId ? { replace_instance_id: replaceInstanceId } : {}),
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
      if (rememberSkipConfirm && !shouldBypassSkipConfirm(reusePreview)) {
        setSkipConfirmSig(submitSignature());
      }
      setConfirm(null);
      setReusePreview(null);
      if (currentProjectId.current === projectId) {
        setSubmission(ids.length ? { projectId, ids } : null);
        if (!desktop) setTab('任务');
      }
      idempotencyKey.current = '';
      lastSubmitActionSig.current = '';
      pushToast(`已提交 ${ids.length} 个生成任务`, 'success');
      await Promise.all([refreshJobs(), refreshResources()]);
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

  async function importWorkflow(raw: string): Promise<WorkflowImportResult> {
    try {
      const value = JSON.parse(raw) as unknown;
      const source = record(value) && record(value.workflow) ? value.workflow : value;
      if (!record(source) || (!record(source.image_extraction) && (!record(source.manifest) || !record(source.api_graph) || !record(source.object_info)))) {
        throw new Error('JSON 必须包含 manifest、api_graph、object_info');
      }
      setLoading('import');
      setError('');
      try {
        const workflow = await importGpuWorkflow(value as never);
        setWorkflows((curr) => [workflow, ...curr.filter((item) => item.id !== workflow.id)]);
        if (workflow.model_auto_registered) {
          void getGpuModels().then(setModels).catch(() => undefined);
        }
        const weightSummary = workflow.import_weight_summary;
        let toastMsg = workflow.pending_configuration
          ? '图片工作流已保存为待配置草稿，请到「资产 › 工作流」展开草稿补充运行配置。'
          : '工作流已作为草稿导入，请校验后发布。';
        if (!workflow.pending_configuration && weightSummary?.requires_rebind) {
          const parts: string[] = [];
          if (weightSummary.civitai > 0) parts.push(`${weightSummary.civitai} 个 Civitai 来源`);
          if (weightSummary.manual > 0) parts.push(`${weightSummary.manual} 个待重绑模型`);
          toastMsg = `工作流已作为草稿导入（含 ${parts.join('、')}），请展开草稿中的「模型依赖」准备或重新绑定模型后生成新草稿再启用。`;
        } else if (!workflow.pending_configuration && workflow.model_auto_registered) {
          toastMsg = '工作流已作为草稿导入，并已自动登记待审查的模型记录，请校验并完成启用。';
        }
        pushToast(toastMsg, 'success');
        return { ok: true };
      } catch (cause) {
        const conflict = parseWorkflowVersionConflict(cause);
        if (conflict) {
          return { ok: false, conflict };
        }
        const msg = apiMessage(cause);
        setError(msg);
        pushToast(msg, 'error');
        return { ok: false, error: msg };
      } finally {
        setLoading('');
      }
    } catch (cause) {
      const msg = apiMessage(cause);
      setError(`导入失败：${msg}`);
      return { ok: false, error: msg };
    }
  }

  async function exportWorkflow(wf: GpuWorkflow): Promise<WorkflowExportResponse | null> {
    setError('');
    try {
      const exported = await exportGpuWorkflow(wf.id);
      const blobUrl = URL.createObjectURL(
        new Blob([JSON.stringify(exported.bundle, null, 2)], { type: 'application/json' }),
      );
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = exported.filename || `${wf.workflow_key}-v${wf.workflow_version}.json`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
      pushToast(formatExportSummaryMessage(exported.summary), 'success');
      return exported;
    } catch (cause) {
      const msg = apiMessage(cause);
      setError(msg);
      pushToast(`导出失败：${msg}`, 'error');
      return null;
    }
  }

  async function clearJobs(request: JobClearRequest) {
    await action('clear-jobs', async () => {
      const { cleared } = await clearGpuJobHistory({
        project_id: jobsProjectFilter || undefined,
        scope: request.scope,
        job_ids: request.jobIds,
        include_system: request.includeSystem,
      });
      if (request.jobIds?.includes(detailId) || !request.jobIds) {
        setDetailId('');
        setDetail(null);
      }
      await refreshJobs();
      pushToast(cleared ? `已清理 ${cleared} 条任务历史` : '没有可清理的已结束任务', 'success');
    });
  }

  function buildPresetOverrides(includeMainPrompt: boolean): Record<string, unknown> {
    const nextOverrides: Record<string, unknown> = { ...params };
    if (!includeMainPrompt) {
      const { promptKey } = promptKeysOf(manifest);
      if (promptKey) delete nextOverrides[promptKey];
    }
    return nextOverrides;
  }

  async function savePreset(name: string, includeMainPrompt: boolean) {
    if (!workflowId) return;
    await action('preset', async () => {
      const next = await saveGpuPreset({
        name,
        workflow_version_id: workflowId,
        overrides: buildPresetOverrides(includeMainPrompt),
        fixed_prompts: normalizeFixedPromptConfig(fixedPromptsRef.current),
      });
      setPresets((curr) => [next, ...curr]);
      pushToast(`已保存预设「${next.name}」`, 'success');
    });
  }

  async function updatePreset(preset: GpuPreset, includeMainPrompt: boolean) {
    await action('preset', async () => {
      const next = await updateGpuPreset(preset.id, {
        overrides: buildPresetOverrides(includeMainPrompt),
        fixed_prompts: normalizeFixedPromptConfig(fixedPromptsRef.current),
      });
      setPresets((curr) => curr.map((item) => (item.id === next.id ? next : item)));
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
    setParams((curr) => ({ ...curr, ...preset.overrides }));
    if (preset.fixed_prompts) {
      updateFixedPrompts(normalizeFixedPromptConfig(preset.fixed_prompts), { immediate: true });
    }
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
    if (!id || !isCreationPoolInstance(instance)) return;
    const mode = instance.retention_mode === 'smart' || instance.retention_mode === 'retain'
      ? instance.retention_mode
      : runtimeChoice.lifecycle;
    const retainMinutes = typeof instance.retain_idle_minutes === 'number' && instance.retain_idle_minutes >= 1
      ? instance.retain_idle_minutes
      : runtimeChoice.retainIdleMinutes;
    changeRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: mode, retainIdleMinutes: retainMinutes });
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
      setJobs((current) => current.map((item) => (item.instance_id === instanceId && !item.runtime_deleted
        ? { ...item, lifecycle: 'retain', retained_until: updatedInstance.retained_until, instance_state: updatedInstance.state }
        : item)));
      pushToast('已确认保留当前 GPU 机器', 'success');
      await refreshJobs();
    });
  }

  const keyStateRef = useRef({
    gs, detailId, tab, moveDialog, confirm, showProjectSheet, showNewProject, showGpuStatus,
  });
  keyStateRef.current = {
    gs, detailId, tab, moveDialog, confirm, showProjectSheet, showNewProject, showGpuStatus,
  };

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const state = keyStateRef.current;
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);
      if (e.key === 'Escape') {
        if (document.fullscreenElement) return;
        if (document.querySelector('[data-lightbox="true"]')) return;
        if (state.gs.confirmDeleteViewer) state.gs.setConfirmDeleteViewer(null);
        else if (state.gs.confirmDeleteNative) state.gs.setConfirmDeleteNative(null);
        else if (state.moveDialog) setMoveDialog(null);
        else if (state.confirm) setConfirm(null);
        else if (state.showNewProject) setShowNewProject(false);
        else if (state.showProjectSheet) setShowProjectSheet(false);
        else if (state.showGpuStatus) setShowGpuStatus(false);
        else if (state.gs.activeViewerItem) state.gs.closeViewers();
        else if (state.detailId && state.tab === '任务') setDetailId('');
      } else if (!typing && state.gs.activeViewerItem && !state.moveDialog && !state.confirm && !state.gs.confirmDeleteViewer && !state.gs.confirmDeleteNative && !state.showNewProject && !state.showProjectSheet && !state.showGpuStatus) {
        if (e.key === 'ArrowLeft') { e.preventDefault(); state.gs.stepActiveViewer(-1); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); state.gs.stepActiveViewer(1); }
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useBodyScrollLock(Boolean(
    loggedIn && (
      confirm
      || (detailId && tab === '任务' && !desktop)
      || (gs.activeViewerItem && !gs.inlineViewer)
    ),
  ));

  if (!loggedIn) {
    return <LoginScreen onLoggedIn={() => { setLoggedIn(true); setLoginRefresh((c) => c + 1); }} />;
  }

  const inspectedJob = detail?.id === detailId ? detail : null;

  function inspectJob(id: string) {
    setDetailError('');
    if (id === detailId && tab === '任务') setDetailRefresh((value) => value + 1);
    setDetailId(id);
  }

  function navigate(next: MainTab) {
    setShowGpuStatus(false);
    if (next === tab) return;
    gs.closeViewers();
    gs.setPendingRouteAssetId('');
    setDetailId('');
    setTab(next);
    window.scrollTo(0, 0);
  }

  function toggleGpuStatus() {
    setShowGpuStatus((curr) => {
      const next = !curr;
      if (next) {
        const now = Date.now();
        if (now - lastStatusRefreshRef.current >= 3000) {
          lastStatusRefreshRef.current = now;
          void refreshResources();
          void refreshJobs();
          void refreshUsage().catch(() => {});
        }
      }
      return next;
    });
  }

  function handleGpuStatusAction(action: GpuStatusActionTarget) {
    if (action.kind === 'job') {
      openJobFromViewer(action.jobId);
      return;
    }
    if (action.kind === 'compute') navigate('算力');
    else if (action.kind === 'native') navigate('原生 ComfyUI');
    else if (action.kind === 'jobs') navigate('任务');
    else if (action.kind === 'gallery') navigate('作品');
  }

  function jumpToTransfer(target: TransferFocusTarget) {
    setFocusTransferTarget(target);
    navigate('算力');
  }

  function openJobFromViewer(jobId: string) {
    gs.closeViewers();
    gs.setPendingRouteAssetId('');
    const target = jobs.find((j) => j.id === jobId);
    if (jobsProjectFilter && target && target.project_id !== jobsProjectFilter) {
      setJobsProjectFilter('');
    }
    if (target && isSystemJob(target)) {
      setShowSystemJobs(true);
    }
    setTab('任务');
    inspectJob(jobId);
    window.scrollTo(0, 0);
  }

  function openOutputAsset(assetId: string) {
    gs.setGallerySource('studio');
    const existing = gs.allGallery.find((asset) => asset.id === assetId);
    if (existing) {
      if (gs.galleryProjectFilter && existing.project_id !== gs.galleryProjectFilter) {
        gs.setGalleryProjectFilter('');
      }
      gs.setViewerList(gs.galleryProjectFilter && existing.project_id === gs.galleryProjectFilter ? gs.projectGallery : gs.allGallery);
      gs.setViewer(existing);
    } else {
      gs.setPendingRouteAssetId(assetId);
    }
    setDetailId('');
    setTab('作品');
    window.scrollTo(0, 0);
  }

  const studioViewerAsset = gs.activeViewerItem?.kind === 'studio' ? gs.activeViewerItem.asset : null;
  const viewerNode = gs.activeViewerItem && (
    <Suspense fallback={<p className="muted small" role="status">正在加载预览…</p>}>
      <GalleryViewer
        item={gs.activeViewerItem}
        workflows={workflows}
        projectName={studioViewerAsset ? projectTitleById(studioViewerAsset.project_id, projects) : undefined}
        navCount={gs.activeNavCount}
        navIndex={gs.activeNavIndex}
        inlineViewer={gs.inlineViewer}
        viewerJob={gs.viewerJob}
        viewerJobError={gs.viewerJobError}
        error={gs.galleryError}
        importingRef={gs.importingNativeRef}
        onStep={gs.stepActiveViewer}
        onClose={gs.closeViewers}
        onError={gs.setGalleryError}
        onToast={pushToast}
        onOpenJob={openJobFromViewer}
        onFavoriteStudio={(asset) => void gs.favorite(asset)}
        onReuseStudio={(asset) => void gs.reuseAsset(asset)}
        onUseStudioAsReference={useAssetAsReference}
        onMoveStudio={studioViewerAsset && projects.some((p) => p.id !== studioViewerAsset.project_id)
          ? (asset) => setMoveDialog({ ids: [asset.id], currentProjectId: asset.project_id })
          : undefined}
        onDeleteStudio={gs.setConfirmDeleteViewer}
        onFavoriteNative={(asset) => void gs.favoriteNative(asset)}
        onImportNativeReference={gs.importNativeRef}
        onImportWorkflowBundle={async (bundle) => { await importWorkflow(bundle); }}
        onDeleteNative={gs.setConfirmDeleteNative}
      />
    </Suspense>
  );

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
          <button
            className="projectChip"
            type="button"
            aria-label="切换项目"
            title={activeProject ? `当前创作保存项目：${projectOptionLabel(activeProject, projects)}（点击切换与管理）` : '选择与管理项目'}
            onClick={() => { setShowGpuStatus(false); setShowProjectSheet(true); }}
          >
            <Folder size={15} strokeWidth={1.6} aria-hidden />
            <span>{activeProject ? projectOptionLabel(activeProject, projects) : '选择项目'}</span>
            <ChevronDown size={14} strokeWidth={1.6} aria-hidden />
          </button>
          <button className="iconButton" type="button" aria-label="新建项目" title="新建项目" onClick={() => { setShowGpuStatus(false); setShowNewProject(true); }}>
            <Plus size={18} strokeWidth={1.6} aria-hidden />
          </button>
          <GpuStatusPopover
            open={showGpuStatus}
            desktop={desktop}
            settings={settings}
            instances={instances}
            jobs={jobs}
            usage={usage}
            workflows={workflows}
            onToggle={toggleGpuStatus}
            onClose={() => setShowGpuStatus(false)}
            onAction={handleGpuStatusAction}
          />
          <button
            className="iconButton mobileSettings"
            type="button"
            aria-label="打开设置"
            aria-current={tab === '设置' ? 'page' : undefined}
            onClick={() => navigate('设置')}
          >
            <SettingsIcon size={18} strokeWidth={1.6} aria-hidden />
          </button>
        </div>
      </header>

      <nav className="bottomNav" aria-label="工作区导航">
        {(desktop ? desktopTabs : mobileTabs).map((item) => {
          const Icon = tabIcons[item];
          return (
            <button
              key={item}
              type="button"
              data-tab={item}
              title={item}
              aria-label={item === '设置' ? '打开设置' : undefined}
              aria-current={tab === item ? 'page' : undefined}
              onClick={() => navigate(item)}
            >
              <Icon size={20} strokeWidth={1.6} aria-hidden />
              <span>{desktop ? item : (mobileLabels[item] || item)}</span>
              {item === '任务' && activeJobsCount > 0 && <span className="tabBadge">{activeJobsCount}</span>}
            </button>
          );
        })}
      </nav>

      <ProjectSheet
        open={showProjectSheet}
        projects={projects}
        projectId={projectId}
        onClose={() => setShowProjectSheet(false)}
        onSelect={(id) => chooseProject(id, { syncView: true })}
        onRename={renameLocalProject}
        onSetDefault={setDefaultLocalProject}
        onDelete={removeLocalProject}
      />
      <NewProjectDialog open={showNewProject} onClose={() => setShowNewProject(false)} onCreate={createLocalProject} />
      <MoveProjectDialog
        open={Boolean(moveDialog)}
        assetIds={moveDialog?.ids || []}
        currentProjectId={moveDialog?.currentProjectId}
        projects={projects}
        onClose={() => setMoveDialog(null)}
        onConfirm={async (ids, targetProjectId) => {
          const res = await gs.moveAssets(ids, targetProjectId);
          if (res.movedIds.length > 0) {
            const movedSet = new Set(res.movedIds);
            setReferenceImages((curr) => curr.map((a) => (movedSet.has(a.id) ? { ...a, project_id: targetProjectId } : a)));
            setPinnedReferenceAssets((curr) => curr.map((a) => (movedSet.has(a.id) ? { ...a, project_id: targetProjectId } : a)));
            moveDialog?.onMoved?.(res.movedIds);
          }
          return res;
        }}
      />

      <main className={styles.page}>
        <Suspense fallback={<p className="muted small" role="status">正在加载页面…</p>}>
        {tab === '创作' && (
          <section className={styles.createWorkspace} aria-label="创作工作区">
            <div className={styles.createHeader}>
              <h1 className={styles.desktopTitle}>创作</h1>
              {adminRole && (
                <button
                  type="button"
                  className={creationApiOpen ? 'button secondary' : 'button'}
                  aria-expanded={creationApiOpen}
                  onClick={() => setCreationApiOpen((open) => !open)}
                >
                  {creationApiOpen ? <ArrowLeft size={16} aria-hidden /> : <KeyRound size={16} aria-hidden />}
                  {creationApiOpen ? '返回创作' : 'API 设置'}
                </button>
              )}
            </div>
            {adminRole && creationApiOpen ? (
              <CreationApiPage
                accounts={accounts}
                projects={projects}
                onProjectsRefresh={() => void getProjects().then(setProjects).catch(() => {})}
              />
            ) : (
              <div className={styles.createColumns}>
                <CreatePanel
                  desktop={desktop}
                  draftReady={workflowParamsReady}
                  usage={usage}
                  computeSpec={selectedCompute}
                  specBelowDefaultError={specBelowDefaultError}
                  onEditGuard={() => {
                    navigate('算力');
                    window.requestAnimationFrame(() => {
                      window.requestAnimationFrame(() => document.getElementById('cu-guard')?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
                    });
                  }}
                  gpuSelection={runtime === 'colab' ? <WorkflowGpuSelection manifest={manifest} accountId={runtimeChoice.accountId} value={selectedCompute} onChange={chooseGpu} gpuOrder={gpuOrder} /> : null}
                  projectId={projectId}
                  projectTitle={activeProject ? projectOptionLabel(activeProject, projects) : ''}
                  projects={projects}
                  referenceScope={referenceScope}
                  onReferenceScopeChange={setReferenceScope}
                  models={models}
                  workflows={workflows}
                  presets={presets}
                  settings={settings}
                  manifest={workflowParamsReady ? manifest : null}
                  params={workflowParamsReady ? params : {}}
                  setParams={setParams}
                  fixedPrompts={fixedPrompts}
                  onFixedPromptsChange={(next) => updateFixedPrompts(next)}
                  fixedPromptsSyncing={fixedPromptsSyncing}
                  triggerWords={activeLoraTriggerWords}
                  gallery={availableReferenceImages}
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
                  gpuOrder={gpuOrder}
                  onFocusTransfer={jumpToTransfer}
                  loading={loading}
                  error={error}
                  onSubmit={requestSubmit}
                  onApplyPreset={applyPreset}
                  onSavePreset={(name, includeMainPrompt) => void savePreset(name, includeMainPrompt)}
                  onUpdatePreset={(preset, includeMainPrompt) => void updatePreset(preset, includeMainPrompt)}
                  onDeletePreset={(preset) => void removePreset(preset)}
                >
                  {workflowId && (
                    <ModelRunOverrides
                      workflowId={workflowId}
                      value={modelOverrides[workflowId] || []}
                      onChange={(value) => setModelOverrides((current) => ({ ...current, [workflowId]: value }))}
                      dependencyRecord={workflowDependencyState?.workflowId === workflowId ? workflowDependencyState.record : null}
                      libraryModels={libraryModelsCatalog}
                    />
                  )}
                </CreatePanel>
                {desktop && projectId && (
                  <CreateResults
                    key={projectId}
                    jobs={previewJobs}
                    count={currentSubmission?.ids.length || previewJobs.length}
                    gallery={gs.createProjectGallery}
                    error={gs.galleryError}
                    onOpen={(asset) => { gs.setGallerySource('studio'); gs.setViewerList(gs.createProjectGallery); gs.setViewer(asset); }}
                    onJobs={() => navigate('任务')}
                    onFavorite={(asset) => void gs.favorite(asset)}
                    onDelete={(asset) => void gs.remove(asset)}
                    onUseAsReference={useAssetAsReference}
                  />
                )}
              </div>
            )}
          </section>
        )}

        {tab === '任务' && (
          <div className={styles.detailWorkspace}>
            <TasksPanel
              jobs={jobs}
              projects={projects}
              projectFilter={jobsProjectFilter}
              onProjectFilterChange={setJobsProjectFilter}
              showSystemJobs={showSystemJobs}
              onShowSystemJobsChange={setShowSystemJobs}
              workflows={workflows}
              error={jobError}
              refreshing={refreshingJobs}
              clearingHistory={loading === 'clear-jobs'}
              onRefresh={() => void refreshJobs()}
              onCancel={(id) => void action('cancel', async () => { await cancelGpuJob(id); await refreshJobs(); })}
              onRetry={(id) => void action('retry', async () => {
                const job = jobs.find((item) => item.id === id);
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
              onInstance={(job) => {
                const instance = instances.find((item) => instanceIdOf(item) === String(job.instance_id || ''));
                if (instance) {
                  const id = instanceIdOf(instance);
                  setRuntimeChoice({ accountId: instance.account_id || null, instanceId: id, lifecycle: runtimeChoice.lifecycle, retainIdleMinutes: runtimeChoice.retainIdleMinutes });
                  setTab('算力');
                }
              }}
            />
            {desktop && (
              <section className={`${styles.card} ${styles.inspector}`} aria-label="任务详情">
                <JobDetailInspector
                  detailId={detailId}
                  detailError={detailError}
                  inspectedJob={inspectedJob}
                  workflows={workflows}
                  onOpenOutput={openOutputAsset}
                  onClose={() => setDetailId('')}
                />
              </section>
            )}
          </div>
        )}

        {tab === '作品' && (
          <div className={styles.stack}>
            <div className={styles.gallerySourceBar}>
              <Segmented<GallerySource>
                ariaLabel="作品来源"
                value={gs.gallerySource}
                onChange={(next) => {
                  gs.setGallerySource(next);
                  gs.setGalleryError('');
                  gs.closeViewers();
                }}
                options={[
                  { value: 'studio', label: 'Studio 作品', icon: <ImageIcon size={15} strokeWidth={1.6} aria-hidden /> },
                  { value: 'native', label: '原生作品（Drive）', icon: <HardDrive size={15} strokeWidth={1.6} aria-hidden /> },
                ]}
              />
            </div>

            <div className={styles.detailWorkspace} data-gallery>
              {gs.gallerySource === 'studio' ? (
                <GalleryPanel
                  key={gs.galleryProjectFilter || '__all__'}
                  gallery={gs.projectGallery}
                  allGallery={gs.allGallery}
                  projects={projects}
                  projectFilter={gs.galleryProjectFilter}
                  onProjectFilterChange={gs.setGalleryProjectFilter}
                  workflows={workflows}
                  error={gs.viewer ? '' : gs.galleryError}
                  refreshing={gs.refreshingGallery}
                  onRefresh={gs.refreshGallery}
                  inspectedId={desktop ? gs.viewer?.id || '' : undefined}
                  onFavorite={(asset) => void gs.favorite(asset)}
                  onBatchDelete={gs.batchRemove}
                  onBatchDownload={gs.batchDownloadStudio}
                  onRequestMove={(ids, opts) => setMoveDialog({ ids, currentProjectId: opts?.currentProjectId, onMoved: opts?.onMoved })}
                  onOpen={(asset, ordered) => {
                    gs.setViewerList(ordered || gs.projectGallery);
                    gs.setViewer(asset);
                    gs.setPendingRouteAssetId('');
                  }}
                  onReuse={(asset) => void gs.reuseAsset(asset)}
                  onGoCreate={() => navigate('创作')}
                />
              ) : (
                <NativeGalleryPanel
                  inspectedId={desktop ? gs.nativeViewer?.id || '' : undefined}
                  refreshToken={gs.nativeRefreshToken}
                  favoriteOverrides={gs.nativeFavOverrides}
                  deletedIds={gs.nativeDeletedIds}
                  onOpen={(asset, ordered) => {
                    gs.setNativeViewerList(ordered);
                    gs.setNativeViewer(asset);
                  }}
                  onFavorite={(asset) => void gs.favoriteNative(asset)}
                  onBatchDelete={gs.batchRemoveNative}
                  onBatchDownload={gs.batchDownloadNative}
                />
              )}

              {desktop && (
                <section className={`${styles.card} ${styles.inspector} ${styles.assetInspector}`} aria-label="作品详情">
                  {viewerNode || (
                    <>
                      <h2>作品详情</h2>
                      <p className="muted small">
                        {gs.gallerySource === 'native'
                          ? '选择原生作品查看原图、视频，支持收藏、导入参考素材与提取 PNG 工作流（键盘 ← / → 切换）。'
                          : '选择作品查看原图、视频和生成参数，支持键盘 ← / → 切换。'}
                      </p>
                    </>
                  )}
                </section>
              )}
            </div>
          </div>
        )}

        {tab === '算力' && (
          <ComputePage
            settings={settings}
            usage={usage}
            canEditGuard={adminRole}
            onRefreshUsage={refreshUsage}
            instances={instances}
            onContinue={continueInstance}
            onOpenWorkbench={openInstanceWorkbench}
            onOpenNative={() => navigate('原生 ComfyUI')}
            onRefreshInstances={() => resourceRefresh.current?.()}
            focusTransferTarget={focusTransferTarget}
            onFocusTransferHandled={() => setFocusTransferTarget(null)}
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
                  onImport={(bundle) => importWorkflow(bundle)}
                  onExport={(wf) => exportWorkflow(wf)}
                  onOpenModelLibrary={() => {
                    writeStoredWorkflowInspect({ ...readStoredWorkflowInspect(), open: true });
                    setAssetsSubTab('models');
                  }}
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
                    setWorkflows((current) => current.filter((item) => item.id !== wf.id));
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
                    if (jobsProjectFilter && jobProjectId && jobProjectId !== jobsProjectFilter) {
                      setJobsProjectFilter('');
                    }
                    setTab('任务');
                    inspectJob(jobId);
                  }}
                  onUseProject={(id) => { setJobsProjectFilter(id); setTab('任务'); }}
                  onActivated={() => { void getGpuWorkflows().then(setWorkflows); void getGpuModels().then(setModels).catch(() => undefined); }}
                  onFocusTransfer={jumpToTransfer}
                />
              </>
            ) : (
              <ModelLibraryPage onOpenSettings={() => { setSettingsFocusSection('model-storage'); navigate('设置'); }} />
            )}
          </section>
        )}

        {tab === '原生 ComfyUI' && (
          <NativeComfyPage
            usage={usage}
            accounts={accounts}
            accountsLoaded={accountsLoaded}
            instances={instances}
            desktop={desktop}
            onOpenSettings={() => navigate('设置')}
            onFocusTransfer={jumpToTransfer}
          />
        )}

        {tab === '设置' && (
          <SettingsPanel
            accounts={accounts}
            accountsLoaded={accountsLoaded}
            projects={projects}
            onLocalStorageMutated={gs.refreshGallery}
            onAccountsChange={setAccounts}
            onGpuOrderChange={setGpuOrder}
            settings={settings}
            usage={usage}
            canEditGuard={adminRole}
            desktop={desktop}
            focusSection={settingsFocusSection}
            onFocusHandled={() => setSettingsFocusSection(null)}
            error={error}
            onLogout={() => {
              sessionStorage.removeItem('colab-gpu-studio-token');
              setLoggedIn(false);
            }}
            onBack={() => setTab('创作')}
          />
        )}
        </Suspense>
      </main>

      {confirm === 'submit' && activeWorkflow && (() => {
        const forcedConfirm = shouldBypassSkipConfirm(reusePreview);
        const confirmActions = buildConfirmActions(reusePreview);
        const targetInst = reusePreview?.target_instance;
        return (
          <div className={styles.dialogBackdrop} role="presentation" onClick={() => { setConfirm(null); setReusePreview(null); }}>
            <section
              className={styles.dialog}
              role="dialog"
              aria-modal="true"
              aria-label="确认提交任务"
              onClick={(e) => e.stopPropagation()}
            >
              <h2>{formatReuseDecisionTitle(reusePreview)}</h2>
              <dl className={styles.kv}>
                <dt>工作流</dt><dd>{activeWorkflow.workflow_key} · v{activeWorkflow.workflow_version}</dd>
                <dt>运行方式</dt><dd>{runtime === 'colab' ? 'Colab GPU（会扣 CU）' : '离线模拟（不扣 CU）'}</dd>
                {runtime === 'colab' && <><dt>目标 GPU</dt><dd>{computeLabel(selectedCompute || manifest?.compute_spec as Record<string, unknown>)}</dd></>}
                <dt>机器</dt><dd>{targetInst ? `${targetInst.label || targetInst.instance_id.slice(0, 8)} · ${targetInst.gpu_reported || targetInst.spec_id || 'GPU'} · ${targetInst.is_busy ? '运行中' : '空闲可复用'}` : resolveInstanceSubmission(runtimeChoice, instances).machineLabel}</dd>
                <dt>数量</dt><dd>{batchCount}</dd>
                {runtime === 'colab' && <><dt>最长运行</dt><dd>{maxRuntime} 分钟</dd></>}
                <dt>结束策略</dt><dd>{runtimeChoice.lifecycle === 'smart' ? `智能保留 ${runtimeChoice.retainIdleMinutes} 分钟` : runtimeChoice.lifecycle === 'retain' ? `保留机器 ${runtimeChoice.retainIdleMinutes} 分钟` : runtimeChoice.lifecycle === 'batch' ? '批次复用' : '节省（结束后释放）'}</dd>
              </dl>
              {targetInst && (
                <div className="status" role="status">
                  <strong>
                    创作池目标实例：{targetInst.label || `#${targetInst.instance_id.slice(0, 8)}`}（{targetInst.gpu_reported || targetInst.spec_id || 'GPU'}）
                    {targetInst.warm_hit ? ' · 命中常驻热复用' : ''}
                  </strong>
                  <p className="muted small">
                    当前状态：{targetInst.is_busy ? '正在运行任务' : '空闲保留中'}
                    {targetInst.loaded_workflow_key ? ` · 已加载 ${targetInst.loaded_workflow_key}${targetInst.loaded_workflow_version != null ? ` v${targetInst.loaded_workflow_version}` : ''}` : ''}
                    {targetInst.retention_mode ? ` · 原策略 ${targetInst.retention_mode}${targetInst.retain_idle_minutes ? `（${targetInst.retain_idle_minutes}分）` : ''}` : ''}
                    {targetInst.remaining_seconds != null ? ` · 剩余约 ${Math.max(1, Math.round(targetInst.remaining_seconds / 60))} 分钟` : ''}
                  </p>
                </div>
              )}
              {reusePreview?.reason && (
                <p className="status" role="alert">
                  <strong>实例复用提示：</strong>{reusePreview.reason}
                </p>
              )}
              {reusePreview?.shortens_lifetime && reusePreview.shorten_reasons.length > 0 && (
                <div className="status error" role="alert">
                  <strong>注意：复用该实例将缩短其原有存活时长</strong>
                  {reusePreview.shorten_reasons.map((item) => <p key={item} className="small">{item}</p>)}
                </div>
              )}
              {!forcedConfirm && (
                <label className={styles.checkRow}>
                  <input
                    type="checkbox"
                    checked={rememberSkipConfirm}
                    onChange={(e) => setRememberSkipConfirm(e.target.checked)}
                  />
                  <span className="small muted">本次会话相同配置直接提交，不再弹窗确认</span>
                </label>
              )}
              {forcedConfirm && (
                <p className="muted small">本次涉及规格差异、实例忙碌或缩短存活时长，为避免误操作已强制开启确认。</p>
              )}
              {error && <p className="status error" role="alert">{error}</p>}
              <div className="actions">
                <button className="button secondary" type="button" onClick={() => { setConfirm(null); setReusePreview(null); }}>取消</button>
                {confirmActions.map((opt) => (
                  <button
                    key={opt.key}
                    className={`button ${opt.variant === 'danger' ? 'danger' : opt.variant === 'secondary' ? 'secondary' : ''}`}
                    disabled={loading === 'submit'}
                    type="button"
                    onClick={() => void submitJob(opt)}
                  >
                    {loading === 'submit' ? '提交中…' : opt.label}
                  </button>
                ))}
              </div>
            </section>
          </div>
        );
      })()}

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
        open={Boolean(gs.confirmDeleteViewer)}
        title="删除此作品？"
        description={gs.confirmDeleteViewer ? `将永久删除「${gs.confirmDeleteViewer.filename || gs.confirmDeleteViewer.id}」，删除后无法恢复。` : ''}
        confirmLabel="确认删除"
        danger
        onCancel={() => gs.setConfirmDeleteViewer(null)}
        onConfirm={() => {
          if (gs.confirmDeleteViewer) void gs.remove(gs.confirmDeleteViewer);
          gs.setConfirmDeleteViewer(null);
        }}
      />

      <ConfirmDialog
        open={Boolean(gs.confirmDeleteNative)}
        title="移入 Google Drive 回收站？"
        description={gs.confirmDeleteNative ? `将把原生作品「${gs.confirmDeleteNative.filename}」移入 Google Drive 回收站。` : ''}
        confirmLabel="移至回收站"
        danger
        onCancel={() => gs.setConfirmDeleteNative(null)}
        onConfirm={() => {
          if (gs.confirmDeleteNative) void gs.removeNative(gs.confirmDeleteNative);
          gs.setConfirmDeleteNative(null);
        }}
      />

      {detailId && tab === '任务' && !desktop && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setDetailId('')}>
          <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="任务详情" onClick={(event) => event.stopPropagation()}>
            <Suspense fallback={<p className="muted small" role="status">正在加载详情…</p>}>
              <JobDetailInspector
                detailId={detailId}
                detailError={detailError}
                inspectedJob={inspectedJob}
                workflows={workflows}
                onOpenOutput={openOutputAsset}
                onClose={() => setDetailId('')}
              />
            </Suspense>
          </section>
        </div>
      )}

      {gs.activeViewerItem && !gs.inlineViewer && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={gs.closeViewers}>
          <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="查看作品" onClick={(event) => event.stopPropagation()}>
            {viewerNode}
          </section>
        </div>
      )}

      <ToastStack items={toasts} onDismiss={(id) => setToasts((curr) => curr.filter((item) => item.id !== id))} />
    </div>
  );
}
