import { WorkflowsPanel } from './pages/WorkflowManager';
import { CreationApiPage } from './pages/CreationApiPage';
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
import { LoginScreen } from './components/LoginScreen';
import { NewProjectDialog, ProjectSheet } from './components/ProjectDialogs';
import { JobDetailInspector } from './components/JobDetailInspector';
import { GalleryViewer } from './components/GalleryViewer';
import { NativeGalleryPanel } from './components/NativeGalleryPanel';
import { useGalleryState, type GallerySource } from './hooks/useGalleryState';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Sparkles, Zap, Image as ImageIcon, Cpu, Settings as SettingsIcon, Folder, ChevronDown, Plus,
  Triangle, Boxes, Workflow, Layers, KeyRound, ArrowLeft, HardDrive,
} from 'lucide-react';
import {
  apiMessage, cancelGpuJob, clearGpuJobHistory, createGpuJob, createGpuBatch,
  createProject, deleteGpuPreset, deleteGpuWorkflow, deleteProject,
  apiFetch,
  getGpuComputeSpecs, getGpuJob, getGpuJobs, getGpuModels, listGpuInstances,
  getGpuPresets, getReferenceAssets, getGpuSettings, getGpuSpecOrder, getGpuUsage, getGpuWorkflowSchema, getGpuWorkflows, getProjects,
  importGpuWorkflow,
  isLoggedIn, previewGpuJobReuse, publishGpuWorkflow, renameProject, retryGpuJob, retryGpuAttempt, stopGpuCompute,
  updateGpuPreset, uploadAsset,
  validateGpuWorkflow, rollbackGpuWorkflow, deactivateGpuWorkflow, saveGpuPreset, ApiError,
  type ComputeSpec, type GalleryAsset, type GpuJob, type GpuPreset,
  type GpuManifest, type GpuModel, type GpuSettings, type GpuSpecKey, type GpuUsage, type GpuWorkflow, type Project, type Provider, type ReusePreviewResponse,
} from './lib/gpu-api';
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
import { projectOptionLabel } from './lib/project-label';
import { carriedDraft, mergeWorkflowDraft, readWorkflowDraft, writeWorkflowDraft } from './lib/workflow-draft';
import { useDesktopLayout } from './lib/desktop-layout';
import { parameterKey, parameterList, readableState } from './lib/ui-helpers';
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
const activeInstanceStates = new Set(['requested', 'allocating', 'preparing', 'ready', 'busy', 'retained_idle', 'releasing']);

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
  const [showProjectSheet, setShowProjectSheet] = useState(false);
  const [showNewProject, setShowNewProject] = useState(false);
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
    const value = await getGpuUsage();
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
  const draftLocation = useRef({ projectId: '', workflowId: '' });

  const activeProject = projects.find((p) => p.id === projectId);
  const activeWorkflow = workflows.find((item) => item.id === workflowId);
  const workflowParamsReady = Boolean(projectId && workflowId && draftReady.current === `${projectId}:${workflowId}`);
  const projectJobs = useMemo(() => jobs.filter((job) => job.project_id === projectId), [jobs, projectId]);
  const latestJob = projectJobs.find((job) => !job.prep);
  const currentSubmission = submission?.projectId === projectId ? submission : null;
  const previewJobs = currentSubmission
    ? projectJobs.filter((job) => currentSubmission.ids.includes(job.id))
    : latestJob?.batch_id
      ? projectJobs.filter((job) => job.batch_id === latestJob.batch_id)
      : latestJob ? [latestJob] : [];
  const galleryOutputs = desktop && (tab === '创作' || tab === '作品')
    ? projectJobs.flatMap((job) => job.output_asset_ids || []).join(',')
    : '';
  const activeInstance = instances.find((instance) => instance.provider === 'colab' && !instance.runtime_deleted
    && instanceIdOf(instance) === runtimeChoice.instanceId)
    || instances.find((instance) => instance.provider === 'colab' && !instance.runtime_deleted
      && ['ready', 'busy', 'retained_idle', 'allocating', 'preparing'].includes(instance.state || ''));
  const detailRevision = projectJobs.find((job) => job.id === detailId)?.updated_at;

  const activeJobsCount = useMemo(
    () => projectJobs.filter((j) => !terminalStates.has(String(j.state || '').toLowerCase())).length,
    [projectJobs],
  );
  const hasActiveInstances = useMemo(
    () => instances.some((i) => !i.runtime_deleted && activeInstanceStates.has(String(i.state || '').toLowerCase())),
    [instances],
  );

  const bindReferenceAsset = useCallback((asset: GalleryAsset) => {
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
      setAccountsLoaded(true);
      setInstances(uniqueBy(nextInstances, (instance) => instanceIdOf(instance)));
      setResourceError('');
    } catch (cause) {
      if (seq === resourceSequenceRef.current) setResourceError(`账号/实例：${apiMessage(cause)}`);
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
    if (tab !== '任务' || !detailId || !projectId || !loggedIn) return;
    let alive = true;
    getGpuJob(detailId).then((job) => {
      if (!alive || job.project_id !== projectId) return;
      setDetail(job);
      setDetailError('');
    }).catch((cause) => {
      if (alive) setDetailError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [tab, detailId, detailRevision, detailRefresh, projectId, loggedIn]);

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
      gs.setViewer(null);
      gs.setPendingRouteAssetId('');
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
      setProjects((curr) => curr.map((item) => (item.id === id ? { ...item, ...updated } : item)));
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
      if (projectId === project.id) chooseProject(remaining[0]?.id || '');
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
    let alive = true;
    void getGpuSpecOrder().then((res) => {
      if (alive && Array.isArray(res.ordered_specs) && res.ordered_specs.length) {
        setGpuOrder(res.ordered_specs);
      }
    }).catch(() => {});
    return () => { alive = false; };
  }, [loggedIn, loginRefresh, tab]);

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
            ? '图片工作流已保存为待配置草稿，请到「资产 › 工作流」展开草稿补充运行配置。'
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

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);
      if (e.key === 'Escape') {
        if (gs.confirmDeleteViewer) gs.setConfirmDeleteViewer(null);
        else if (gs.confirmDeleteNative) gs.setConfirmDeleteNative(null);
        else if (confirm) setConfirm(null);
        else if (showNewProject) setShowNewProject(false);
        else if (showProjectSheet) setShowProjectSheet(false);
        else if (gs.activeViewerItem) gs.closeViewers();
        else if (detailId && tab === '任务') setDetailId('');
      } else if (!typing && gs.activeViewerItem && !confirm && !gs.confirmDeleteViewer && !gs.confirmDeleteNative && !showNewProject && !showProjectSheet) {
        if (e.key === 'ArrowLeft') { e.preventDefault(); gs.stepActiveViewer(-1); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); gs.stepActiveViewer(1); }
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [gs, detailId, tab, confirm, showProjectSheet, showNewProject]);

  if (!loggedIn) {
    return <LoginScreen onLoggedIn={() => { setLoggedIn(true); setLoginRefresh((c) => c + 1); }} />;
  }

  const inspectedJob = detail?.id === detailId && detail.project_id === projectId ? detail : null;

  function inspectJob(id: string) {
    setDetailError('');
    if (id === detailId && tab === '任务') setDetailRefresh((value) => value + 1);
    setDetailId(id);
  }

  function navigate(next: MainTab) {
    if (next === tab) return;
    gs.closeViewers();
    gs.setPendingRouteAssetId('');
    setDetailId('');
    setTab(next);
    window.scrollTo(0, 0);
  }

  function openJobFromViewer(jobId: string) {
    gs.closeViewers();
    gs.setPendingRouteAssetId('');
    setTab('任务');
    inspectJob(jobId);
    window.scrollTo(0, 0);
  }

  function openOutputAsset(assetId: string) {
    gs.setGallerySource('studio');
    const existing = gs.projectGallery.find((asset) => asset.id === assetId);
    if (existing) {
      gs.setViewerList(gs.projectGallery);
      gs.setViewer(existing);
    } else {
      gs.setPendingRouteAssetId(assetId);
    }
    setDetailId('');
    setTab('作品');
    window.scrollTo(0, 0);
  }

  const viewerNode = gs.activeViewerItem && (
    <GalleryViewer
      item={gs.activeViewerItem}
      workflows={workflows}
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
      onDeleteStudio={gs.setConfirmDeleteViewer}
      onFavoriteNative={(asset) => void gs.favoriteNative(asset)}
      onImportNativeReference={gs.importNativeRef}
      onImportWorkflowBundle={importWorkflow}
      onDeleteNative={gs.setConfirmDeleteNative}
    />
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
          <button
            className="desktopGpu projectChip"
            type="button"
            onClick={() => navigate('算力')}
            title="查看算力"
            data-online={Boolean(activeInstance && ['ready', 'busy', 'retained_idle'].includes(activeInstance.state || ''))}
          >
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
        onSelect={chooseProject}
        onRename={renameLocalProject}
        onDelete={removeLocalProject}
      />
      <NewProjectDialog open={showNewProject} onClose={() => setShowNewProject(false)} onCreate={createLocalProject} />

      <main className={styles.page}>
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
            {adminRole && creationApiOpen ? <CreationApiPage accounts={accounts} /> : (
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
                  {workflowId && (
                    <ModelRunOverrides
                      workflowId={workflowId}
                      value={modelOverrides[workflowId] || []}
                      onChange={(value) => setModelOverrides((current) => ({ ...current, [workflowId]: value }))}
                    />
                  )}
                </CreatePanel>
                {desktop && projectId && (
                  <CreateResults
                    key={projectId}
                    jobs={previewJobs}
                    count={currentSubmission?.ids.length || previewJobs.length}
                    gallery={gs.projectGallery}
                    error={gs.galleryError}
                    onOpen={(asset) => { gs.setGallerySource('studio'); gs.setViewerList(gs.projectGallery); gs.setViewer(asset); }}
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
              jobs={projectJobs}
              workflows={workflows}
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
                  key={projectId}
                  gallery={gs.projectGallery}
                  workflows={workflows}
                  error={gs.viewer ? '' : gs.galleryError}
                  refreshing={gs.refreshingGallery}
                  onRefresh={gs.refreshGallery}
                  inspectedId={desktop ? gs.viewer?.id || '' : undefined}
                  onFavorite={(asset) => void gs.favorite(asset)}
                  onBatchDelete={gs.batchRemove}
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
                    if (jobProjectId) chooseProject(jobProjectId);
                    setTab('任务');
                    inspectJob(jobId);
                  }}
                  onUseProject={(id) => { chooseProject(id); setTab('任务'); }}
                  onActivated={() => { void getGpuWorkflows().then(setWorkflows); void getGpuModels().then(setModels).catch(() => undefined); }}
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
          />
        )}

        {tab === '设置' && (
          <SettingsPanel
            accounts={accounts}
            onAccountsChange={setAccounts}
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
            <JobDetailInspector
              detailId={detailId}
              detailError={detailError}
              inspectedJob={inspectedJob}
              workflows={workflows}
              onOpenOutput={openOutputAsset}
              onClose={() => setDetailId('')}
            />
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
