import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  FileSearch,
  HardDrive,
  MoreHorizontal,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Trash2,
  X,
} from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import {
  ACTIVE_MODEL_STATES,
  bytesLabel,
  computeCategorySlots,
  displayModelTitles,
  formatModelCreatedFull,
  formatModelCreatedShort,
  libraryApi,
  modelStatus,
  planBatchDeletion,
  pruneSessionAddedModels,
  readStoredModelSortMode,
  readStoredWorkflowInspect,
  recordSessionAddedModels,
  sortModels,
  writeStoredModelSortMode,
  writeStoredWorkflowInspect,
  type Catalog,
  type LibraryModel,
  type ModelSortMode,
  type ModelTask,
  type SessionAddedModel,
} from '../lib/model-library';
import { useSmartPoll } from '../lib/smart-poll';
import { ConfirmDialog } from '../components/Feedback';
import { ModelAddDialog } from '../components/ModelAddDialog';
import { ModelDetailDrawer } from '../components/ModelDetailDrawer';
import { ModelLibrarySettingsDialog, type ModelStorageAccountOption } from '../components/ModelLibrarySettings';
import { ModelStorageBreakdown } from '../components/ModelStorageBreakdown';
import { WorkflowDependencyInspector } from '../components/WorkflowDependencyInspector';
import styles from '../gpu.module.css';

const DEFAULT_TYPES: Record<string, string> = {
  checkpoints: '',
  loras: '',
  vae: '',
  diffusion_models: '',
  text_encoders: '',
  clip_vision: '',
  controlnet: '',
  upscale_models: '',
  embeddings: '',
};

const COMPACT_STATUS_LABEL: Record<string, string> = {
  pending: '待准备',
  resolving: '解析中',
  downloading: '下载中',
  verifying: '校验中',
  uploading: '上传中',
  ready: '已就绪',
  failed: '失败',
  missing: '文件缺失',
  unverified: '待校验',
  cancelled: '已取消',
};

export { ModelLibrarySettingsDialog, ModelSourceSettings, ModelStorageSettings } from '../components/ModelLibrarySettings';

export function ModelLibraryPage({ accounts }: { accounts?: ModelStorageAccountOption[] } = {}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [tasks, setTasks] = useState<ModelTask[]>([]);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('');
  const [architectureFilter, setArchitectureFilter] = useState('');
  const [stateFilter, setStateFilter] = useState('');
  const [unreferencedOnly, setUnreferencedOnly] = useState(false);
  const [sortMode, setSortMode] = useState<ModelSortMode>(() => readStoredModelSortMode());
  const [detailId, setDetailId] = useState<string | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [openMenuUp, setOpenMenuUp] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [sessionAdded, setSessionAdded] = useState<SessionAddedModel[]>([]);
  const [batchDeleteMode, setBatchDeleteMode] = useState<'remove_index' | 'delete_file' | null>(null);
  const [singleDeleteTarget, setSingleDeleteTarget] = useState<{ model: LibraryModel; deleteFile: boolean } | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');
  const [inspector, setInspector] = useState(() => readStoredWorkflowInspect());
  const selectAllRef = useRef<HTMLInputElement>(null);
  const seenCatalogIdsRef = useRef<Set<string>>(new Set());

  const updateSortMode = useCallback((next: ModelSortMode | ((curr: ModelSortMode) => ModelSortMode)) => {
    setSortMode((curr) => {
      const resolved = typeof next === 'function' ? next(curr) : next;
      writeStoredModelSortMode(resolved);
      return resolved;
    });
  }, []);

  const recordAddedModels = useCallback((added: LibraryModel[]) => {
    if (!added.length) return;
    setSessionAdded((curr) => recordSessionAddedModels(curr, added));
  }, []);

  const updateInspector = useCallback((patch: Partial<typeof inspector>) => {
    setInspector((curr) => {
      const next = { ...curr, ...patch };
      writeStoredWorkflowInspect(next);
      return next;
    });
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [data, jobs] = await Promise.all([libraryApi<Catalog>(), libraryApi<ModelTask[]>('/tasks')]);
      setCatalog(data);
      setTasks(jobs);
    } catch (e) {
      setError(apiMessage(e));
    }
  }, []);

  const hasActiveTasks = tasks.some((t) => ACTIVE_MODEL_STATES.has(t.state));
  useSmartPoll(refresh, { enabled: true, active: hasActiveTasks, activeMs: 3000, idleMs: 30000 });

  // Prune selection and session queue when catalog changes (e.g. models deleted).
  useEffect(() => {
    if (!catalog) return;
    const valid = new Set(catalog.items.map((m) => m.id));
    for (const id of valid) seenCatalogIdsRef.current.add(id);
    setSelectedIds((prev) => {
      let changed = false;
      const next = new Set<string>();
      for (const id of prev) {
        if (valid.has(id)) next.add(id);
        else changed = true;
      }
      return changed ? next : prev;
    });
    setSessionAdded((prev) => pruneSessionAddedModels(prev, valid, seenCatalogIdsRef.current));
  }, [catalog]);

  // Close row overflow menu on outside tap/click or Escape.
  useEffect(() => {
    if (!openMenuId) return;
    const onPointerDown = (e: Event) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest('[data-row-menu="true"]')) return;
      setOpenMenuId(null);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpenMenuId(null);
    };
    window.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [openMenuId]);

  async function runAction(fn: () => Promise<unknown>, successMessage?: string) {
    setBusy(true);
    setError('');
    try {
      await fn();
      await refresh();
      if (successMessage) setProgress(successMessage);
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const allItems = catalog?.items || [];
  const availableTypes = useMemo(() => Object.keys(catalog?.types || DEFAULT_TYPES), [catalog]);
  const categorySlots = useMemo(() => computeCategorySlots(catalog), [catalog]);

  const architectures = useMemo(() => {
    const uniq = [...new Set(allItems.map((m) => m.architecture).filter(Boolean))];
    return uniq.sort((a, b) => {
      if (a === 'unknown') return 1;
      if (b === 'unknown') return -1;
      return a.localeCompare(b);
    });
  }, [allItems]);

  const modelById = useMemo(
    () => new Map(allItems.map((m) => [m.id, m])),
    [allItems],
  );

  const taskByModelId = useMemo(() => {
    const map = new Map<string, ModelTask>();
    for (const t of tasks) {
      const existing = map.get(t.model_id);
      if (!existing || ACTIVE_MODEL_STATES.has(t.state)) {
        map.set(t.model_id, t);
      }
    }
    return map;
  }, [tasks]);

  const sessionAddedView = useMemo(() => {
    let ready = 0;
    let active = 0;
    let failed = 0;
    let other = 0;
    const items = sessionAdded.map((entry) => {
      const model = modelById.get(entry.modelId);
      const task = taskByModelId.get(entry.modelId);
      const hasActiveTask = Boolean(task && ACTIVE_MODEL_STATES.has(task.state));
      const isActive = hasActiveTask || Boolean(model && ACTIVE_MODEL_STATES.has(model.state));
      const effectiveState = hasActiveTask && task ? task.state : (model?.state || 'pending');
      const title = model ? displayModelTitles(model).primary : entry.fallbackTitle;
      const pct = hasActiveTask && task?.total && task.total > 0
        ? Math.min(100, Math.round((task.bytes / task.total) * 100))
        : null;
      const baseLabel = COMPACT_STATUS_LABEL[effectiveState] || modelStatus[effectiveState] || effectiveState;
      const showPct = pct !== null && (effectiveState === 'downloading' || effectiveState === 'uploading' || effectiveState === 'verifying');
      const statusLabel = showPct ? `${baseLabel} ${pct}%` : baseLabel;
      const tone = isActive
        ? 'active'
        : effectiveState === 'ready'
          ? 'ready'
          : (effectiveState === 'failed' || effectiveState === 'missing')
            ? 'danger'
            : 'warn';
      if (isActive) active += 1;
      else if (effectiveState === 'ready') ready += 1;
      else if (effectiveState === 'failed' || effectiveState === 'missing') failed += 1;
      else other += 1;
      return {
        modelId: entry.modelId,
        model,
        title,
        statusLabel,
        tone,
      };
    });
    const summaryParts: string[] = [];
    if (ready > 0) summaryParts.push(`${ready} 已就绪`);
    if (active > 0) summaryParts.push(`${active} 进行中`);
    if (failed > 0) summaryParts.push(`${failed} 失败`);
    if (other > 0) summaryParts.push(`${other} 待处理`);
    return { items, summaryParts };
  }, [sessionAdded, modelById, taskByModelId]);

  const visibleItems = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = allItems.filter((m) => {
      if (q) {
        const hay = `${m.display_name} ${m.filename} ${m.relative_path || ''} ${(m.tags || []).join(' ')}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      if (category && m.model_type !== category) return false;
      if (stateFilter && m.state !== stateFilter) return false;
      if (architectureFilter && m.architecture !== architectureFilter) return false;
      if (unreferencedOnly && (m.reference_count || 0) > 0) return false;
      return true;
    });
    return sortModels(filtered, tasks, sortMode);
  }, [allItems, query, category, stateFilter, architectureFilter, unreferencedOnly, sortMode, tasks]);

  const maxVisibleSize = useMemo(
    () => visibleItems.reduce((max, item) => Math.max(max, item.size || 0), 0),
    [visibleItems],
  );

  const selectedModels = useMemo(
    () => allItems.filter((m) => selectedIds.has(m.id)),
    [allItems, selectedIds],
  );

  const selectedBytes = useMemo(() => {
    const seen = new Set<string>();
    let total = 0;
    for (const m of selectedModels) {
      const key = m.drive_file_id || m.id;
      if (!seen.has(key)) {
        seen.add(key);
        total += m.size || 0;
      }
    }
    return total;
  }, [selectedModels]);

  const visibleSelectedCount = useMemo(
    () => visibleItems.filter((m) => selectedIds.has(m.id)).length,
    [visibleItems, selectedIds],
  );
  const allVisibleSelected = visibleItems.length > 0 && visibleSelectedCount === visibleItems.length;
  const someVisibleSelected = visibleSelectedCount > 0 && !allVisibleSelected;

  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = someVisibleSelected;
    }
  }, [someVisibleSelected]);

  const batchPlan = useMemo(
    () => (batchDeleteMode ? planBatchDeletion(selectedModels, allItems, tasks, batchDeleteMode) : null),
    [batchDeleteMode, selectedModels, allItems, tasks],
  );

  const activeDetailIndex = useMemo(
    () => (detailId ? visibleItems.findIndex((m) => m.id === detailId) : -1),
    [detailId, visibleItems],
  );
  const inspectedModel = useMemo(
    () => (detailId ? allItems.find((m) => m.id === detailId) || null : null),
    [detailId, allItems],
  );

  const hasFilters = Boolean(query.trim() || category || stateFilter || architectureFilter || unreferencedOnly);
  const inspectCount = inspector.result?.dependencies.length || 0;

  function toggleRowSelection(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleSelectAllVisible() {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (allVisibleSelected) {
        for (const m of visibleItems) next.delete(m.id);
      } else {
        for (const m of visibleItems) next.add(m.id);
      }
      return next;
    });
  }

  function locateModel(model: LibraryModel) {
    setQuery('');
    setCategory('');
    setStateFilter('');
    setArchitectureFilter('');
    setUnreferencedOnly(false);
    setDetailId(model.id);
  }

  async function executeBatchMaintenance(kind: 'check' | 'verify') {
    const eligible = selectedModels.filter((m) => {
      if (!m.drive_file_id) return false;
      const t = taskByModelId.get(m.id);
      return !(t && ACTIVE_MODEL_STATES.has(t.state));
    });
    if (!eligible.length) {
      setError('所选模型暂无可执行 Drive 检查或校验的文件（需先准备完成且不处于运行中）。');
      return;
    }
    setBusy(true);
    setError('');
    let okCount = 0;
    const failures: string[] = [];
    for (let i = 0; i < eligible.length; i++) {
      const m = eligible[i];
      setProgress(`${kind === 'check' ? '正在检查 Drive' : '正在加入校验队列'}（${i + 1} / ${eligible.length}）…`);
      try {
        if (kind === 'check') {
          await libraryApi(`/${m.id}/check`, {});
        } else {
          await libraryApi(`/${m.id}/prepare`, { action: 'verify' });
        }
        okCount += 1;
      } catch (e) {
        failures.push(`${displayModelTitles(m).primary}：${apiMessage(e)}`);
      }
    }
    await refresh();
    setBusy(false);
    if (failures.length) {
      setError(`部分失败（成功 ${okCount} / ${eligible.length}）：${failures.slice(0, 2).join('；')}`);
    } else {
      setProgress(kind === 'check' ? `已完成 ${okCount} 个模型的 Drive 检查` : `已将 ${okCount} 个模型加入重新校验队列`);
      setSelectedIds(new Set());
    }
  }

  async function executeBatchDelete() {
    if (!batchPlan || !batchPlan.deletable.length) {
      setBatchDeleteMode(null);
      return;
    }
    setBusy(true);
    setError('');
    const plan = batchPlan;
    setBatchDeleteMode(null);
    let deletedCount = 0;
    let deletedPhysicalCount = 0;
    const failures: string[] = [];

    for (let i = 0; i < plan.deletable.length; i++) {
      const entry = plan.deletable[i];
      setProgress(`正在清理 ${i + 1} / ${plan.deletable.length}：${displayModelTitles(entry.model).primary}…`);
      try {
        const qs = entry.deletePhysical ? '?confirm_references=true&delete_file=true' : '?confirm_references=true';
        await libraryApi(`/${entry.model.id}${qs}`, undefined, 'DELETE');
        deletedCount += 1;
        if (entry.deletePhysical) deletedPhysicalCount += 1;
        if (detailId === entry.model.id) setDetailId(null);
      } catch (e) {
        failures.push(`${displayModelTitles(entry.model).primary}：${apiMessage(e)}`);
      }
    }

    await refresh();
    setSelectedIds(new Set());
    setBusy(false);

    if (failures.length) {
      setError(`已清理 ${deletedCount} 项，${failures.length} 项失败：${failures.slice(0, 2).join('；')}`);
    } else {
      const extra = plan.mode === 'delete_file' && deletedPhysicalCount > 0
        ? `（含 ${deletedPhysicalCount} 个 Drive 物理文件${plan.reclaimableBytes > 0 ? `，预计释放 ${bytesLabel(plan.reclaimableBytes)}` : ''}）`
        : '';
      const skipped = plan.activeBlocked.length ? `；跳过 ${plan.activeBlocked.length} 个进行中模型` : '';
      setProgress(`已清理 ${deletedCount} 个模型${extra}${skipped}`);
    }
  }

  async function executeSingleDelete() {
    if (!singleDeleteTarget) return;
    const { model, deleteFile } = singleDeleteTarget;
    setSingleDeleteTarget(null);
    const title = displayModelTitles(model).primary;
    await runAction(async () => {
      const qs = deleteFile ? '?confirm_references=true&delete_file=true' : '?confirm_references=true';
      await libraryApi(`/${model.id}${qs}`, undefined, 'DELETE');
      if (detailId === model.id) setDetailId(null);
    }, deleteFile ? `已删除「${title}」及 Drive 物理文件` : `已移除「${title}」索引`);
  }

  const batchDriveReadyCount = selectedModels.filter((m) => {
    const t = taskByModelId.get(m.id);
    return Boolean(m.drive_file_id) && !(t && ACTIVE_MODEL_STATES.has(t.state));
  }).length;
  const renderNowMs = Date.now();

  return (
    <div className={styles.stack}>
      <div className={styles.pageTitle}>
        <h2>统一模型库</h2>
        <div className={styles.titleActions}>
          <button
            className={`button ${inspector.open ? '' : 'secondary'} small`}
            type="button"
            aria-expanded={inspector.open}
            onClick={() => updateInspector({ open: !inspector.open })}
          >
            <FileSearch size={14} strokeWidth={1.8} aria-hidden />
            检测工作流模型
            {inspectCount > 0 && !inspector.open && <span className={styles.depBtnCount}>{inspectCount}</span>}
          </button>
          <button
            className="button small"
            type="button"
            onClick={() => setAddOpen(true)}
          >
            <Plus size={14} strokeWidth={2} aria-hidden />
            添加模型
          </button>
          <button
            className="button ghost small"
            type="button"
            onClick={() => setSettingsOpen(true)}
          >
            <Settings size={14} strokeWidth={1.8} aria-hidden />
            存储与下载源设置
          </button>
          <button className="button secondary small" type="button" disabled={busy} onClick={() => void runAction(refresh)}>
            刷新
          </button>
        </div>
      </div>

      <p className="muted small">
        {catalog?.statistics.count || 0} 个模型 · Drive 已索引 {bytesLabel(catalog?.statistics.bytes)}。下载与校验由独立模型进程执行，无需启动 GPU。
      </p>

      <WorkflowDependencyInspector
        open={inspector.open}
        fileName={inspector.fileName}
        result={inspector.result}
        importedMap={inspector.importedMap}
        models={allItems}
        tasks={tasks}
        availableTypes={availableTypes}
        onStateChange={updateInspector}
        onClose={() => updateInspector({ open: false })}
        onRefreshLibrary={refresh}
        onLocateModel={locateModel}
        onModelsAdded={recordAddedModels}
      />

      <ModelStorageBreakdown
        catalog={catalog}
        selectedCategory={category}
        onSelectCategory={setCategory}
      />

      {error && (
        <div className={styles.modelNoticeRow} data-tone="error" role="alert">
          <span>{error}</span>
          <button type="button" className={styles.iconGhost} aria-label="关闭错误提示" onClick={() => setError('')}>
            <X size={14} aria-hidden />
          </button>
        </div>
      )}
      {progress && (
        <div className={styles.modelNoticeRow} role="status">
          <span>{progress}</span>
          <button type="button" className={styles.iconGhost} aria-label="关闭状态提示" onClick={() => setProgress('')}>
            <X size={14} aria-hidden />
          </button>
        </div>
      )}
      {sessionAddedView.items.length > 0 && (
        <section className={styles.modelSessionQueue} aria-label="本次已入库模型记录" role="status">
          <div className={styles.modelSessionQueueHead}>
            <div className={styles.modelSessionQueueSummary}>
              <strong>本次已加入队列（{sessionAddedView.items.length} 个）</strong>
              {sessionAddedView.summaryParts.length > 0 && (
                <span className="muted small">{sessionAddedView.summaryParts.join(' · ')}</span>
              )}
            </div>
            <button
              type="button"
              className={styles.modelSessionClearBtn}
              aria-label="清空本次入库记录"
              onClick={() => setSessionAdded([])}
            >
              清空
            </button>
          </div>
          <div className={styles.modelSessionChipList}>
            {sessionAddedView.items.map((item) => (
              <button
                key={item.modelId}
                type="button"
                className={styles.modelSessionChip}
                data-tone={item.tone}
                disabled={!item.model}
                title={`${item.title} · ${item.statusLabel}${item.model ? '（点击查看详情）' : ''}`}
                onClick={() => {
                  if (!item.model) return;
                  if (visibleItems.some((m) => m.id === item.model!.id)) {
                    setDetailId(item.model.id);
                  } else {
                    locateModel(item.model);
                  }
                }}
              >
                <i className={styles.modelSessionDot} aria-hidden />
                <span className={styles.modelSessionChipName}>{item.title}</span>
                <span className={styles.modelSessionChipState}>{item.statusLabel}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* Unified Compact Asset Table Card */}
      <section className={styles.modelInventoryCard} aria-label="模型清单">
        {selectedIds.size > 0 ? (
          <div className={styles.modelBatchBar} role="region" aria-label="批量操作栏">
            <div className={styles.modelBatchInfo}>
              <label className={styles.modelCheckWrap}>
                <input
                  ref={selectAllRef}
                  type="checkbox"
                  aria-label="全选当前筛选结果"
                  checked={allVisibleSelected}
                  onChange={toggleSelectAllVisible}
                />
              </label>
              <strong>已选 {selectedIds.size} 项</strong>
              <span className={styles.modelBatchBytes}>{bytesLabel(selectedBytes)}</span>
              <button
                type="button"
                className="button ghost small"
                disabled={busy}
                onClick={() => setSelectedIds(new Set())}
              >
                取消选择
              </button>
            </div>
            <div className={styles.modelBatchActions}>
              <button
                type="button"
                className="button secondary small"
                disabled={busy || batchDriveReadyCount === 0}
                onClick={() => void executeBatchMaintenance('check')}
              >
                <HardDrive size={13} aria-hidden />
                检查 Drive
              </button>
              <button
                type="button"
                className="button secondary small"
                disabled={busy || batchDriveReadyCount === 0}
                onClick={() => void executeBatchMaintenance('verify')}
              >
                <ShieldCheck size={13} aria-hidden />
                重新校验
              </button>
              <button
                type="button"
                className="button danger small"
                disabled={busy}
                onClick={() => setBatchDeleteMode('remove_index')}
              >
                移除索引
              </button>
              <button
                type="button"
                className="button danger small"
                disabled={busy}
                onClick={() => setBatchDeleteMode('delete_file')}
              >
                <Trash2 size={13} aria-hidden />
                删除 Drive 文件
              </button>
            </div>
          </div>
        ) : (
          <div className={styles.modelToolbar}>
            <div className={styles.modelToolbarLeft}>
              <label className={styles.modelSearchBox}>
                <Search size={15} strokeWidth={1.7} aria-hidden />
                <input
                  aria-label="搜索名称 / 文件名"
                  placeholder="搜索名称、文件名或路径…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                {query && (
                  <button
                    type="button"
                    className={styles.modelSearchClear}
                    aria-label="清空搜索"
                    onClick={() => setQuery('')}
                  >
                    <X size={13} aria-hidden />
                  </button>
                )}
              </label>
              {category && (
                <button
                  type="button"
                  className={styles.modelActiveCategoryChip}
                  title="点击清除类别筛选"
                  onClick={() => setCategory('')}
                >
                  <i className={styles.legendDot} data-slot={categorySlots[category] ?? 5} aria-hidden />
                  <span>{category}</span>
                  <X size={12} aria-hidden />
                </button>
              )}
            </div>

            <div className={styles.modelToolbarRight}>
              <button
                type="button"
                className={styles.depFilterChip}
                data-active={unreferencedOnly}
                aria-pressed={unreferencedOnly}
                title="仅显示未被任何工作流引用的模型，方便清理空间"
                onClick={() => setUnreferencedOnly((v) => !v)}
              >
                仅未引用
              </button>
              <select
                aria-label="按状态筛选"
                className={styles.miniSelect}
                value={stateFilter}
                onChange={(e) => setStateFilter(e.target.value)}
              >
                <option value="">全部状态</option>
                {['pending', 'downloading', 'verifying', 'uploading', 'ready', 'failed', 'missing', 'unverified'].map((s) => (
                  <option key={s} value={s}>{modelStatus[s]}</option>
                ))}
              </select>
              <select
                aria-label="按架构筛选"
                className={styles.miniSelect}
                value={architectureFilter}
                onChange={(e) => setArchitectureFilter(e.target.value)}
              >
                <option value="">全部架构</option>
                {architectures.map((arch) => (
                  <option key={arch} value={arch}>{arch}</option>
                ))}
              </select>
              <select
                aria-label="排序方式"
                className={styles.miniSelect}
                value={sortMode}
                onChange={(e) => updateSortMode(e.target.value as ModelSortMode)}
              >
                <option value="smart">默认排序（体积 ↓）</option>
                <option value="created_desc">按入库时间（最新优先）</option>
                <option value="size_desc">按体积降序</option>
                <option value="name_asc">按名称 A-Z</option>
                <option value="status">按状态优先</option>
              </select>
              <span className={styles.modelCountHint}>
                {visibleItems.length === allItems.length
                  ? `${allItems.length} 项`
                  : `${visibleItems.length} / ${allItems.length}`}
              </span>
            </div>
          </div>
        )}

        {visibleItems.length === 0 ? (
          <div className={styles.modelEmptyState}>
            <strong>{allItems.length ? '暂无符合条件的模型' : '模型库为空'}</strong>
            <span className="muted small">
              {allItems.length
                ? '可以放宽筛选条件，或清除顶部的类别选中状态。'
                : '点击右上角「添加模型」解析链接、上传本地权重或导入已有 Drive 文件。'}
            </span>
            <div className="actions">
              {hasFilters ? (
                <button
                  type="button"
                  className="button secondary small"
                  onClick={() => {
                    setQuery('');
                    setCategory('');
                    setStateFilter('');
                    setArchitectureFilter('');
                    setUnreferencedOnly(false);
                  }}
                >
                  重置全部筛选
                </button>
              ) : (
                <button type="button" className="button small" onClick={() => setAddOpen(true)}>
                  <Plus size={14} aria-hidden />
                  添加模型
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className={styles.modelTable} role="table" aria-label="模型资产列表">
            <div className={styles.modelTableHead} role="row">
              <div className={styles.modelColCheck} role="columnheader">
                <label className={styles.modelCheckWrap}>
                  <input
                    ref={selectedIds.size === 0 ? selectAllRef : undefined}
                    type="checkbox"
                    aria-label="全选当前列表"
                    checked={allVisibleSelected}
                    onChange={toggleSelectAllVisible}
                  />
                </label>
              </div>
              <div className={styles.modelColName} role="columnheader">模型</div>
              <div className={styles.modelColType} role="columnheader">分类 / 架构</div>
              <div className={styles.modelColRef} role="columnheader">引用</div>
              <div className={styles.modelColStatus} role="columnheader">状态</div>
              <div className={styles.modelColSize} role="columnheader">
                <button
                  type="button"
                  className={styles.modelSortColBtn}
                  onClick={() => updateSortMode((curr) => (curr === 'size_desc' ? 'smart' : 'size_desc'))}
                >
                  体积{sortMode === 'size_desc' || sortMode === 'smart' ? ' ↓' : ''}
                </button>
              </div>
              <div className={styles.modelColActions} role="columnheader">
                <span className="muted small">操作</span>
              </div>
            </div>

            <div className={styles.modelTableBody} role="rowgroup">
              {visibleItems.map((m, idx) => {
                const task = taskByModelId.get(m.id);
                const isActive = Boolean(task && ACTIVE_MODEL_STATES.has(task.state));
                const titles = displayModelTitles(m);
                const slot = categorySlots[m.model_type] ?? 5;
                const isSelected = selectedIds.has(m.id);
                const isInspected = detailId === m.id;
                const menuOpen = openMenuId === m.id;
                const flipUp = openMenuUp || idx >= Math.max(2, visibleItems.length - 2);
                const refCount = m.reference_count || 0;
                const sizeRatio = m.size && maxVisibleSize > 0 ? Math.max(3, Math.round((m.size / maxVisibleSize) * 100)) : 0;
                const effectiveState = isActive && task ? task.state : m.state;
                const statusTone = isActive
                  ? 'active'
                  : m.state === 'ready'
                    ? 'ready'
                    : m.state === 'failed' || m.state === 'missing'
                      ? 'danger'
                      : 'warn';
                const canDeleteFile = Boolean(m.drive_file_id && m.managed_file && !isActive);
                const createdShort = formatModelCreatedShort(m.created_at, renderNowMs);
                const createdFull = formatModelCreatedFull(m.created_at);

                return (
                  <div
                    key={m.id}
                    role="row"
                    tabIndex={0}
                    className={styles.modelRow}
                    data-selected={isSelected}
                    data-inspected={isInspected}
                    data-selecting={selectedIds.size > 0}
                    data-menu-open={menuOpen}
                    onClick={() => setDetailId(m.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'BUTTON') return;
                        e.preventDefault();
                        setDetailId(m.id);
                      }
                    }}
                  >
                    <div
                      className={styles.modelColCheck}
                      role="cell"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <label className={styles.modelCheckWrap}>
                        <input
                          type="checkbox"
                          aria-label={`选择 ${titles.primary}`}
                          checked={isSelected}
                          onChange={() => toggleRowSelection(m.id)}
                        />
                      </label>
                    </div>

                    <div className={styles.modelColName} role="cell">
                      <div className={styles.modelTitleLine}>
                        <i className={styles.legendDot} data-slot={slot} aria-hidden />
                        <span className={styles.modelPrimaryName} title={titles.primary}>{titles.primary}</span>
                        {m.external_source?.status === 'verified' && (
                          <span className={styles.modelVerifiedIcon} title="外部来源已验证">
                            <CheckCircle2 size={13} strokeWidth={2} aria-hidden />
                          </span>
                        )}
                      </div>
                      {(titles.secondary || createdShort || m.external_source?.status === 'unverified') && (
                        <div className={styles.modelSubLine}>
                          {titles.secondary && (
                            <span className={styles.modelSubPath} title={titles.secondary}>{titles.secondary}</span>
                          )}
                          {createdShort && (
                            <span className={styles.modelSubTime} title={createdFull ? `入库时间：${createdFull}` : undefined}>
                              {createdShort} 入库
                            </span>
                          )}
                          {m.external_source?.status === 'unverified' && (
                            <span className={styles.modelSubWarn}>来源待验证</span>
                          )}
                        </div>
                      )}
                      {isActive && task && (
                        <div className={styles.modelRowProgress}>
                          <div className={styles.progress} data-indeterminate={!task.total}>
                            <span style={{ width: task.total ? `${Math.min(100, (task.bytes / task.total) * 100)}%` : undefined }} />
                          </div>
                          <small className="muted">
                            {modelStatus[task.state] || task.state} · {task.bytes ? bytesLabel(task.bytes) : '0 B'} / {bytesLabel(task.total)}
                            {task.speed ? ` · ${(task.speed / 1024 ** 2).toFixed(1)} MB/s` : ''}
                            {task.attempt && task.attempt > 1 ? ` · 第 ${task.attempt} 次尝试` : ''}
                          </small>
                        </div>
                      )}
                      {m.error && !isActive && (
                        <p className={styles.modelRowError} title={m.error}>{m.error}</p>
                      )}
                    </div>

                    <div className={styles.modelColType} role="cell" onClick={(e) => e.stopPropagation()}>
                      <button
                        type="button"
                        className={styles.modelTypeChip}
                        data-active={category === m.model_type}
                        title={`按 ${m.model_type} 筛选`}
                        onClick={() => setCategory(category === m.model_type ? '' : m.model_type)}
                      >
                        {m.model_type}
                      </button>
                      {m.architecture && m.architecture !== 'unknown' && (
                        <span className={styles.modelArchTag}>{m.architecture}</span>
                      )}
                      {refCount > 0 && (
                        <span className={`${styles.modelRefBadge} ${styles.modelMobileOnly}`}>
                          {refCount} 引用
                        </span>
                      )}
                      <span className={styles.modelMobileOnly}>
                        {statusTone === 'ready' ? (
                          <span className={styles.modelStatusReady}>
                            <i aria-hidden />
                            {COMPACT_STATUS_LABEL[effectiveState] || '已就绪'}
                          </span>
                        ) : (
                          <span className={styles.depStatusBadge} data-tone={statusTone}>
                            {COMPACT_STATUS_LABEL[effectiveState] || modelStatus[effectiveState] || effectiveState}
                          </span>
                        )}
                      </span>
                    </div>

                    <div className={styles.modelColRef} role="cell">
                      {refCount > 0 ? (
                        <span className={styles.modelRefBadge} title={`被 ${refCount} 个工作流引用`}>
                          {refCount} 引用
                        </span>
                      ) : (
                        <span className={styles.modelMutedDash} title="暂无工作流引用">—</span>
                      )}
                    </div>

                    <div className={styles.modelColStatus} role="cell">
                      {statusTone === 'ready' ? (
                        <span className={styles.modelStatusReady}>
                          <i aria-hidden />
                          {COMPACT_STATUS_LABEL[effectiveState] || '已就绪'}
                        </span>
                      ) : (
                        <span className={styles.depStatusBadge} data-tone={statusTone}>
                          {COMPACT_STATUS_LABEL[effectiveState] || modelStatus[effectiveState] || effectiveState}
                        </span>
                      )}
                    </div>

                    <div className={styles.modelColSize} role="cell">
                      <span className={styles.modelSizeText}>{bytesLabel(m.size)}</span>
                      {sizeRatio > 0 && (
                        <span className={styles.modelSizeTrack} aria-hidden="true">
                          <i style={{ width: `${sizeRatio}%` }} data-slot={slot} />
                        </span>
                      )}
                    </div>

                    <div
                      className={styles.modelColActions}
                      role="cell"
                      data-row-menu="true"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {isActive && task ? (
                        <button
                          type="button"
                          className="button secondary small"
                          disabled={busy}
                          onClick={() => void runAction(() => libraryApi(`/tasks/${task.id}/cancel`, {}), '已取消模型准备')}
                        >
                          取消
                        </button>
                      ) : (m.state === 'failed' || m.state === 'missing') ? (
                        <button
                          type="button"
                          className="button secondary small"
                          disabled={busy}
                          onClick={() => void runAction(() => libraryApi(`/${m.id}/prepare`, { action: 'prepare' }), '已重新加入准备队列')}
                        >
                          重试
                        </button>
                      ) : (m.state === 'pending' || m.state === 'unverified') ? (
                        <button
                          type="button"
                          className="button secondary small"
                          disabled={busy}
                          onClick={() => void runAction(
                            () => libraryApi(`/${m.id}/prepare`, { action: m.state === 'unverified' && m.drive_file_id ? 'verify' : 'prepare' }),
                            m.state === 'unverified' ? '已加入校验队列' : '已加入准备队列',
                          )}
                        >
                          {m.state === 'unverified' ? '校验' : '准备'}
                        </button>
                      ) : null}

                      <div className={styles.rowMenuWrap}>
                        <button
                          type="button"
                          className={styles.iconGhost}
                          aria-label={`${titles.primary} 更多操作`}
                          aria-expanded={menuOpen}
                          aria-haspopup="menu"
                          onClick={(e) => {
                            if (openMenuId === m.id) {
                              setOpenMenuId(null);
                              return;
                            }
                            const rect = e.currentTarget.getBoundingClientRect();
                            setOpenMenuUp(rect.bottom + 240 > window.innerHeight - 72);
                            setOpenMenuId(m.id);
                          }}
                        >
                          <MoreHorizontal size={16} aria-hidden />
                        </button>

                        {menuOpen && (
                          <div className={styles.rowMenuPopover} data-up={flipUp} role="menu">
                            <button
                              type="button"
                              role="menuitem"
                              onClick={() => { setOpenMenuId(null); setDetailId(m.id); }}
                            >
                              详情与引用
                            </button>
                            <button
                              type="button"
                              role="menuitem"
                              disabled={busy || isActive}
                              onClick={() => {
                                setOpenMenuId(null);
                                void runAction(() => libraryApi(`/${m.id}/prepare`, { action: 'prepare' }), '已加入准备队列');
                              }}
                            >
                              <RefreshCw size={13} aria-hidden />
                              准备 / 重试
                            </button>
                            <button
                              type="button"
                              role="menuitem"
                              disabled={busy || !m.drive_file_id || isActive}
                              onClick={() => {
                                setOpenMenuId(null);
                                void runAction(() => libraryApi(`/${m.id}/prepare`, { action: 'verify' }), '已加入重新校验队列');
                              }}
                            >
                              <ShieldCheck size={13} aria-hidden />
                              重新校验
                            </button>
                            <button
                              type="button"
                              role="menuitem"
                              disabled={busy || !m.drive_file_id || isActive}
                              onClick={() => {
                                setOpenMenuId(null);
                                void runAction(() => libraryApi(`/${m.id}/check`, {}), 'Drive 文件状态已检查');
                              }}
                            >
                              <HardDrive size={13} aria-hidden />
                              检查 Drive 文件
                            </button>
                            <hr />
                            <button
                              type="button"
                              role="menuitem"
                              data-danger="true"
                              disabled={busy || isActive}
                              onClick={() => {
                                setOpenMenuId(null);
                                setSingleDeleteTarget({ model: m, deleteFile: false });
                              }}
                            >
                              移除索引
                            </button>
                            <button
                              type="button"
                              role="menuitem"
                              data-danger="true"
                              disabled={busy || !canDeleteFile}
                              title={!m.managed_file ? '外部导入/历史缓存文件仅支持移除索引' : undefined}
                              onClick={() => {
                                setOpenMenuId(null);
                                setSingleDeleteTarget({ model: m, deleteFile: true });
                              }}
                            >
                              <Trash2 size={13} aria-hidden />
                              删除 Drive 文件
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      <ModelAddDialog
        open={addOpen}
        availableTypes={availableTypes}
        defaultType={category || undefined}
        models={allItems}
        onClose={() => setAddOpen(false)}
        onAdded={async (model) => {
          recordAddedModels([model]);
          setProgress('');
          await refresh();
        }}
        onLocateModel={locateModel}
      />

      <ModelLibrarySettingsDialog
        open={settingsOpen}
        accounts={accounts}
        onClose={() => setSettingsOpen(false)}
      />

      <ModelDetailDrawer
        model={inspectedModel}
        task={inspectedModel ? taskByModelId.get(inspectedModel.id) : undefined}
        categorySlot={inspectedModel ? (categorySlots[inspectedModel.model_type] ?? 5) : 5}
        hasPrev={activeDetailIndex > 0}
        hasNext={activeDetailIndex >= 0 && activeDetailIndex < visibleItems.length - 1}
        onPrev={() => {
          if (activeDetailIndex > 0) setDetailId(visibleItems[activeDetailIndex - 1].id);
        }}
        onNext={() => {
          if (activeDetailIndex >= 0 && activeDetailIndex < visibleItems.length - 1) {
            setDetailId(visibleItems[activeDetailIndex + 1].id);
          }
        }}
        onClose={() => setDetailId(null)}
        onRefresh={refresh}
        onProgress={setProgress}
      />

      {/* Single Row Delete Confirmation */}
      <ConfirmDialog
        open={Boolean(singleDeleteTarget)}
        title={singleDeleteTarget?.deleteFile ? '删除 Drive 物理文件并移除索引？' : '移除模型索引？'}
        description={singleDeleteTarget
          ? (singleDeleteTarget.deleteFile
            ? `将把「${displayModelTitles(singleDeleteTarget.model).primary}」（${bytesLabel(singleDeleteTarget.model.size)}）移至 Drive 回收站并移除索引，当前被 ${singleDeleteTarget.model.reference_count || 0} 个工作流引用。`
            : `将从模型库移除「${displayModelTitles(singleDeleteTarget.model).primary}」的索引（保留 Drive 物理文件），当前被 ${singleDeleteTarget.model.reference_count || 0} 个工作流引用。`)
          : ''}
        confirmLabel={singleDeleteTarget?.deleteFile ? '确认删除' : '确认移除'}
        danger
        busy={busy}
        onCancel={() => setSingleDeleteTarget(null)}
        onConfirm={() => void executeSingleDelete()}
      />

      {/* Smart Batch Cleanup Confirmation */}
      <ConfirmDialog
        open={Boolean(batchPlan)}
        title={batchPlan?.mode === 'delete_file'
          ? `批量删除 ${batchPlan.deletable.length} 个模型？`
          : `批量移除 ${batchPlan?.deletable.length || 0} 个模型索引？`}
        description={batchPlan
          ? (() => {
            if (batchPlan.deletable.length === 0) {
              return '当前所选模型均处于下载/校验中，请先取消任务后再清理。';
            }
            if (batchPlan.mode === 'remove_index') {
              return `将从模型库移除 ${batchPlan.deletable.length} 条索引，保留所有 Drive 物理文件。`;
            }
            const physicalCount = batchPlan.deletable.filter((d) => d.deletePhysical).length;
            if (physicalCount === 0) {
              return `本次所选 ${batchPlan.deletable.length} 个模型均无应用托管 Drive 文件或被其他索引共享，将仅移除索引（不删除 Drive 原文件）。`;
            }
            return `将清理 ${batchPlan.deletable.length} 个模型（其中 ${physicalCount} 个将移入 Drive 回收站${batchPlan.reclaimableBytes > 0 ? `，预计释放 ${bytesLabel(batchPlan.reclaimableBytes)}` : ''}）。`;
          })()
          : ''}
        confirmLabel={batchPlan?.deletable.length
          ? (batchPlan.mode === 'delete_file' ? `确认删除（${batchPlan.deletable.length}）` : `确认移除（${batchPlan.deletable.length}）`)
          : '知道了'}
        danger={Boolean(batchPlan?.deletable.length)}
        busy={busy}
        onCancel={() => setBatchDeleteMode(null)}
        onConfirm={() => void executeBatchDelete()}
      >
        {batchPlan && (
          <div className={styles.stack} style={{ gap: '8px' }}>
            {batchPlan.referencedModels.length > 0 && (
              <div className={styles.depNoticeStrip}>
                <span className={styles.batchWarnLine}>
                  <AlertTriangle size={14} aria-hidden />
                  其中 <strong>{batchPlan.referencedModels.length} 个模型</strong>正被工作流引用，清理后对应工作流依赖将失效。
                </span>
              </div>
            )}
            {batchPlan.fallbackToIndexOnly.length > 0 && (
              <p className="muted small">
                其中 {batchPlan.fallbackToIndexOnly.length} 个为外部导入/历史缓存或被其他未选中索引共享的文件，将自动仅移除索引、保留 Drive 原文件。
              </p>
            )}
            {batchPlan.activeBlocked.length > 0 && (
              <p className="muted small">
                其中 {batchPlan.activeBlocked.length} 个模型正在下载或校验中，本次将自动跳过。
              </p>
            )}
            {batchPlan.deletable.length > 0 && (
              <div className={styles.batchPreviewList}>
                {batchPlan.deletable.map(({ model: m, deletePhysical, fallbackReason }) => (
                  <div key={m.id} className={styles.batchPreviewItem}>
                    <div className={styles.batchPreviewName}>
                      <span>{displayModelTitles(m).primary}</span>
                      <small className="muted">
                        {m.model_type} · {bytesLabel(m.size)}
                        {(m.reference_count || 0) > 0 ? ` · 被 ${m.reference_count} 个工作流引用` : ''}
                      </small>
                    </div>
                    <span className={styles.depTag} title={fallbackReason}>
                      {deletePhysical ? '删文件' : '仅移索引'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </ConfirmDialog>
    </div>
  );
}
