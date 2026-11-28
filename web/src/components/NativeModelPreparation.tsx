import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock,
  Cloud,
  Copy,
  Download,
  Globe,
  HardDrive,
  History,
  Loader2,
  PackageCheck,
  RefreshCw,
  SlidersHorizontal,
  Square,
} from 'lucide-react';
import { ApiError, apiFetch, apiMessage } from '../lib/gpu-api';
import {
  countExplicitDeclarations,
  formatDuration,
  formatKiB,
  formatMiB,
  formatModelCategoryBadge,
  sortPreparedModels,
  splitModelName,
  summarizePreparation,
  type DriveWorkflowItem,
  type Preparation,
  type PreparedModel,
} from '../lib/native-model-prep';
import { Segmented } from './ui';
import styles from './NativeModelPreparation.module.css';

export type { DriveWorkflowItem } from '../lib/native-model-prep';

type InputMode = 'drive' | 'manual';

const PREP_PATH = '/api/gpu/native/model-preparation';
const WORKFLOWS_PATH = '/api/gpu/native/workflows';
const SAVED_WORKFLOW_PREF_KEY = 'studio-native-prep-workflow';

const phases: Record<PreparedModel['state'], string> = {
  pending: '等待中',
  downloading: '下载中',
  ready: '已就绪',
  failed: '准备失败',
};

const sources: Record<NonNullable<PreparedModel['source']>, string> = {
  external: '外部来源',
  drive: 'Drive',
  local: '本地命中',
};

function workflowKey(item: Pick<DriveWorkflowItem, 'source' | 'path'>): string {
  return `${item.source}:${item.path}`;
}

function parseWorkflowKey(key: string): { source: 'comfy' | 'warmup'; path: string } | null {
  const idx = key.indexOf(':');
  if (idx <= 0) return null;
  const source = key.slice(0, idx);
  const path = key.slice(idx + 1);
  if ((source !== 'comfy' && source !== 'warmup') || !path) return null;
  return { source, path };
}

function loadSavedWorkflowKey(): string {
  try { return localStorage.getItem(SAVED_WORKFLOW_PREF_KEY) || ''; } catch { return ''; }
}

function persistWorkflowKey(key: string) {
  try {
    if (key) localStorage.setItem(SAVED_WORKFLOW_PREF_KEY, key);
    else localStorage.removeItem(SAVED_WORKFLOW_PREF_KEY);
  } catch { /* browser storage unavailable */ }
}

function formatWorkflowOption(item: DriveWorkflowItem): string {
  const prefix = item.source === 'warmup' ? '[API] ' : '';
  return `${prefix}${item.path} (${formatKiB(item.size)})`;
}

function StateIcon({ state }: { state: PreparedModel['state'] }) {
  if (state === 'ready') {
    return <CheckCircle2 size={15} strokeWidth={2} className={styles.stateIcon} data-state={state} aria-hidden />;
  }
  if (state === 'downloading') {
    return <Loader2 size={15} strokeWidth={2} className={styles.stateIcon} data-state={state} aria-hidden />;
  }
  if (state === 'failed') {
    return <AlertCircle size={15} strokeWidth={2} className={styles.stateIcon} data-state={state} aria-hidden />;
  }
  return <Clock size={15} strokeWidth={1.9} className={styles.stateIcon} data-state={state} aria-hidden />;
}

function SourceIcon({ source }: { source: NonNullable<PreparedModel['source']> }) {
  if (source === 'local') return <HardDrive size={11} strokeWidth={2} aria-hidden />;
  if (source === 'drive') return <Cloud size={11} strokeWidth={2} aria-hidden />;
  return <Globe size={11} strokeWidth={2} aria-hidden />;
}

export function NativeModelPreparation() {
  const [mode, setMode] = useState<InputMode>('drive');
  const [driveWorkflows, setDriveWorkflows] = useState<DriveWorkflowItem[]>([]);
  const [workflowsLoaded, setWorkflowsLoaded] = useState(false);
  const [loadingWorkflows, setLoadingWorkflows] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string>(() => loadSavedWorkflowKey());
  const [workflow, setWorkflow] = useState('');
  const [saveName, setSaveName] = useState('');
  const [savingWorkflow, setSavingWorkflow] = useState(false);
  const [confirmOverwrite, setConfirmOverwrite] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deletingWorkflow, setDeletingWorkflow] = useState(false);
  const [declarations, setDeclarations] = useState('');
  const [id, setId] = useState('');
  const [result, setResult] = useState<Preparation>();
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const [copiedPath, setCopiedPath] = useState('');
  const initialFetched = useRef(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
  }, []);

  const loadWorkflows = useCallback(async (refresh = false, preferredKey?: string) => {
    setLoadingWorkflows(true);
    if (refresh) {
      setError('');
      setNotice('');
    }
    try {
      const suffix = refresh ? '?refresh=true' : '';
      const data = await apiFetch<{ workflows: DriveWorkflowItem[] }>(WORKFLOWS_PATH + suffix, undefined, 25_000);
      const list = Array.isArray(data.workflows) ? data.workflows : [];
      setDriveWorkflows(list);
      setWorkflowsLoaded(true);
      setSelectedKey(current => {
        const target = preferredKey || current || loadSavedWorkflowKey();
        if (target && list.some(item => workflowKey(item) === target)) {
          persistWorkflowKey(target);
          return target;
        }
        const first = list[0] ? workflowKey(list[0]) : '';
        if (first) persistWorkflowKey(first);
        return first;
      });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setLoadingWorkflows(false);
    }
  }, []);

  useEffect(() => {
    if (!id) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const status = await apiFetch<Preparation>(`${PREP_PATH}/${id}`);
        if (!active) return;
        setResult(status);
        setError('');
        if (status.state === 'preparing') timer = setTimeout(() => void poll(), 1200);
      } catch (cause) {
        if (active) setError(apiMessage(cause));
      }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [id]);

  const selectedRef = parseWorkflowKey(selectedKey);
  const selectedItem = driveWorkflows.find(item => workflowKey(item) === selectedKey);
  const canSubmit = mode === 'drive' ? Boolean(selectedRef && selectedItem) : Boolean(workflow.trim());

  const explicitCount = useMemo(() => countExplicitDeclarations(declarations), [declarations]);
  const summary = useMemo(() => (result ? summarizePreparation(result) : null), [result]);
  const orderedModels = useMemo(
    () => (result?.models ? sortPreparedModels(result.models) : []),
    [result],
  );

  function copyModelPath(fullPath: string) {
    void navigator.clipboard?.writeText(fullPath).then(() => {
      setCopiedPath(fullPath);
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
      copyTimerRef.current = setTimeout(() => {
        setCopiedPath(curr => (curr === fullPath ? '' : curr));
      }, 1500);
    }).catch(() => {});
  }

  async function prepare() {
    setSending(true); setError(''); setNotice(''); setId(''); setResult(undefined);
    try {
      const dependencies = declarations.trim() ? JSON.parse(declarations) : [];
      const body = mode === 'drive'
        ? { workflow_ref: selectedRef, dependencies }
        : { workflow: JSON.parse(workflow), dependencies };
      const job = await apiFetch<{ id: string }>(PREP_PATH, { method: 'POST', body: JSON.stringify(body) }, 25_000);
      setResult({ state: 'preparing', models: [] });
      setId(job.id);
    } catch (cause) {
      setError(cause instanceof SyntaxError ? '请检查工作流或显式依赖的 JSON 格式。' : apiMessage(cause));
    } finally {
      setSending(false);
    }
  }

  async function saveToDrive(overwrite: boolean) {
    if (!workflow.trim() || !saveName.trim() || savingWorkflow) return;
    setSavingWorkflow(true);
    setError('');
    setNotice('');
    try {
      const parsed = JSON.parse(workflow);
      const data = await apiFetch<{ workflow: DriveWorkflowItem }>(WORKFLOWS_PATH, {
        method: 'POST',
        body: JSON.stringify({ name: saveName.trim(), workflow: parsed, overwrite }),
      }, 25_000);
      const savedKey = workflowKey(data.workflow);
      setConfirmOverwrite(false);
      persistWorkflowKey(savedKey);
      await loadWorkflows(false, savedKey);
      setMode('drive');
      setNotice(`已保存到 Drive：${data.workflow.source === 'warmup' ? '[API] ' : ''}${data.workflow.path}`);
    } catch (cause) {
      if (cause instanceof SyntaxError) {
        setError('请先填写有效的 ComfyUI 工作流 JSON。');
      } else if (cause instanceof ApiError && (cause.code === 'NATIVE_WORKFLOW_EXISTS' || cause.status === 409)) {
        setConfirmOverwrite(true);
      } else {
        setError(apiMessage(cause));
      }
    } finally {
      setSavingWorkflow(false);
    }
  }

  async function removeWarmupWorkflow() {
    if (!selectedItem || selectedItem.source !== 'warmup' || deletingWorkflow) return;
    setDeletingWorkflow(true);
    setError('');
    setNotice('');
    try {
      const query = new URLSearchParams({ source: selectedItem.source, path: selectedItem.path });
      await apiFetch(`${WORKFLOWS_PATH}?${query.toString()}`, { method: 'DELETE' }, 20_000);
      setConfirmDelete(false);
      persistWorkflowKey('');
      await loadWorkflows(false);
      setNotice(`已从预热列表移除：${selectedItem.path}`);
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setDeletingWorkflow(false);
    }
  }

  async function latest() {
    setError('');
    setNotice('');
    try {
      const status = await apiFetch<Preparation & { id: string }>(`${PREP_PATH}/latest`);
      setResult(status);
      setId(status.id);
    } catch (cause) {
      setError(apiMessage(cause));
    }
  }

  const isPreparing = result?.state === 'preparing';

  return <details className="foldable" onToggle={event => {
    if ((event.currentTarget as HTMLDetailsElement).open && !initialFetched.current) {
      initialFetched.current = true;
      void loadWorkflows(false);
    }
  }}>
    <summary>
      <span className={styles.summaryTrigger}>
        <PackageCheck size={16} strokeWidth={1.8} aria-hidden />
        <span className={styles.summaryTitle}>检查并下载此工作流所需模型</span>
        {summary && summary.total > 0 && (
          <span className={styles.summaryPill} data-state={result?.state}>
            {result?.state === 'ready'
              ? `${summary.ready} / ${summary.total} 就绪`
              : result?.state === 'failed'
                ? `${summary.failed} 失败`
                : `${summary.ready} / ${summary.total} 准备中`}
          </span>
        )}
      </span>
    </summary>
    <div className="foldContent">
      <p className={styles.intro}>首次用酒馆等客户端调用前，可以先在这里把模型下载到当前实例。冷下载仍受客户端自身的请求超时限制。</p>

      <Segmented<InputMode>
        ariaLabel="工作流来源"
        value={mode}
        onChange={next => {
          setMode(next);
          setError('');
          setConfirmOverwrite(false);
          setConfirmDelete(false);
          if (next === 'drive' && !workflowsLoaded && !loadingWorkflows) {
            void loadWorkflows(false);
          }
        }}
        options={[
          { value: 'drive', label: '从 Drive 选择' },
          { value: 'manual', label: '手动粘贴 / 上传' },
        ]}
      />

      {mode === 'drive' ? (
        <div className="field">
          <div className="actions nativeApiHeading">
            <label htmlFor="native-drive-workflow-select">Drive 工作流</label>
            <button
              type="button"
              className="button ghost small"
              disabled={loadingWorkflows || sending}
              onClick={() => void loadWorkflows(true)}
            >
              <RefreshCw size={14} strokeWidth={1.8} className={loadingWorkflows ? styles.spinIcon : undefined} aria-hidden />
              {loadingWorkflows ? '刷新中…' : '刷新列表'}
            </button>
          </div>
          {driveWorkflows.length > 0 ? (
            <>
              <select
                id="native-drive-workflow-select"
                value={selectedKey}
                onChange={event => {
                  const next = event.target.value;
                  setSelectedKey(next);
                  setConfirmDelete(false);
                  persistWorkflowKey(next);
                }}
              >
                {driveWorkflows.map(item => {
                  const key = workflowKey(item);
                  return <option key={key} value={key}>{formatWorkflowOption(item)}</option>;
                })}
              </select>
              <div className="actions nativeApiHeading">
                <span className="muted small">
                  {selectedItem?.source === 'warmup'
                    ? '自定义保存的 API 预热工作流（独立存放，不影响原生 ComfyUI 侧边栏）。'
                    : '读取原生 ComfyUI 保存在 Drive 的工作流，自动识别所需模型。'}
                </span>
                {selectedItem?.source === 'warmup' && (
                  confirmDelete ? (
                    <span className="actions">
                      <button
                        type="button"
                        className="button danger small"
                        disabled={deletingWorkflow}
                        onClick={() => void removeWarmupWorkflow()}
                      >
                        {deletingWorkflow ? '移除中…' : '确认移除'}
                      </button>
                      <button
                        type="button"
                        className="button ghost small"
                        disabled={deletingWorkflow}
                        onClick={() => setConfirmDelete(false)}
                      >
                        取消
                      </button>
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="button ghost small"
                      onClick={() => setConfirmDelete(true)}
                    >
                      从列表移除
                    </button>
                  )
                )}
              </div>
            </>
          ) : (
            <p className="status">
              {loadingWorkflows
                ? '正在读取 Drive 工作流列表…'
                : 'Drive 中暂无已保存的工作流。可在原生 ComfyUI 中保存工作流后点击「刷新列表」，或切换到「手动粘贴 / 上传」保存到 Drive。'}
            </p>
          )}
        </div>
      ) : (
        <>
          <label className="field">ComfyUI 工作流 JSON
            <input type="file" accept=".json,application/json" onChange={async e => {
              const file = e.target.files?.[0];
              if (file) {
                try {
                  setWorkflow(await file.text());
                  setSaveName(curr => curr.trim() || file.name);
                  setConfirmOverwrite(false);
                } catch (cause) {
                  setError(apiMessage(cause));
                }
              }
            }} />
            <textarea
              rows={5}
              value={workflow}
              onChange={e => { setWorkflow(e.target.value); setConfirmOverwrite(false); }}
              placeholder="粘贴 API 格式或原生工作流 JSON"
            />
          </label>
          <div className="field">
            <label htmlFor="native-save-workflow-name">保存到 Drive 下拉列表（可选）</label>
            <div className="actions">
              <input
                id="native-save-workflow-name"
                style={{ flex: '1 1 200px' }}
                value={saveName}
                maxLength={200}
                onChange={e => { setSaveName(e.target.value); setConfirmOverwrite(false); }}
                placeholder="例如：anima-v1.json"
              />
              <button
                type="button"
                className="button secondary"
                disabled={savingWorkflow || !workflow.trim() || !saveName.trim()}
                onClick={() => void saveToDrive(false)}
              >
                {savingWorkflow ? '保存中…' : '保存到 Drive'}
              </button>
            </div>
            {confirmOverwrite && (
              <div className="actions status" role="alert">
                <span>同名工作流已存在，是否覆盖保存？</span>
                <button
                  type="button"
                  className="button danger small"
                  disabled={savingWorkflow}
                  onClick={() => void saveToDrive(true)}
                >
                  确认覆盖
                </button>
                <button
                  type="button"
                  className="button ghost small"
                  disabled={savingWorkflow}
                  onClick={() => setConfirmOverwrite(false)}
                >
                  取消
                </button>
              </div>
            )}
            <span className="muted small">原生工作流自动存入 ComfyUI 工作流目录；纯 API 工作流独立存入预热目录并标记 [API]。</span>
          </div>
        </>
      )}

      <details className={styles.explicitDetails}>
        <summary className={styles.explicitSummary}>
          <ChevronRight size={14} strokeWidth={2} className={styles.explicitChevron} aria-hidden />
          <SlidersHorizontal size={13} strokeWidth={1.8} aria-hidden />
          <span>显式模型依赖（可选）</span>
          {declarations.trim() && (
            <span
              className={styles.explicitBadge}
              data-invalid={explicitCount === null ? 'true' : undefined}
            >
              {explicitCount === null ? 'JSON 格式有误' : `已填 ${explicitCount} 项`}
            </span>
          )}
        </summary>
        <div className={styles.explicitBody}>
          <label className="field">自定义模型依赖声明
            <textarea
              rows={3}
              value={declarations}
              onChange={e => setDeclarations(e.target.value)}
              placeholder={'[{"category":"loras","name":"styles/example.safetensors"}]'}
            />
            <span className="muted small">自定义节点或动态模型名可填类别及相对路径。详细来源可在模型库绑定，也可在这里填写 url、sha256 和 size（字节）。</span>
          </label>
        </div>
      </details>

      <div className={styles.actionBar}>
        <button
          type="button"
          className="button"
          disabled={sending || !canSubmit || isPreparing}
          onClick={() => void prepare()}
        >
          {sending || isPreparing
            ? <Loader2 size={15} strokeWidth={2} className={styles.spinIcon} aria-hidden />
            : <Download size={15} strokeWidth={2} aria-hidden />}
          {isPreparing ? '检查下载中…' : '检查并下载'}
        </button>
        <button
          type="button"
          className="button secondary"
          disabled={sending}
          onClick={() => void latest()}
        >
          <History size={15} strokeWidth={1.8} aria-hidden />
          查看最近一次准备
        </button>
        {isPreparing && (
          <button
            type="button"
            className={`button ghost small ${styles.stopButton}`}
            onClick={() => { setId(''); setResult(undefined); setError(''); }}
          >
            <Square size={12} strokeWidth={2} aria-hidden />
            停止查看
          </button>
        )}
      </div>

      {notice && <p className="status" role="status">{notice}</p>}
      {error && <p className="status error" role="alert">{error}</p>}

      {result && summary && (
        <div className={styles.resultPanel}>
          <div
            className={styles.summaryBanner}
            data-state={result.state}
            data-empty={orderedModels.length === 0 ? 'true' : undefined}
            role="status"
          >
            <div className={styles.bannerLead}>
              {result.state === 'ready' && (
                <CheckCircle2 size={16} strokeWidth={2} className={styles.stateIcon} data-state="ready" aria-hidden />
              )}
              {result.state === 'preparing' && (
                <Loader2 size={16} strokeWidth={2} className={styles.stateIcon} data-state="downloading" aria-hidden />
              )}
              {result.state === 'failed' && (
                <AlertCircle size={16} strokeWidth={2} className={styles.stateIcon} data-state="failed" aria-hidden />
              )}
              <span className={styles.bannerLeadText}>{summary.headline}</span>
            </div>

            <div className={styles.bannerMeta}>
              {summary.sources.local > 0 && (
                <span className={styles.bannerChip}>
                  <HardDrive size={11} strokeWidth={2} aria-hidden />
                  本地命中 {summary.sources.local}
                </span>
              )}
              {summary.sources.drive > 0 && (
                <span className={styles.bannerChip}>
                  <Cloud size={11} strokeWidth={2} aria-hidden />
                  Drive {summary.sources.drive}
                </span>
              )}
              {summary.sources.external > 0 && (
                <span className={styles.bannerChip}>
                  <Globe size={11} strokeWidth={2} aria-hidden />
                  外部 {summary.sources.external}
                </span>
              )}
              {summary.totalBytes > 0 && isPreparing && (
                <span>{formatMiB(summary.bytes)} / {formatMiB(summary.totalBytes)}</span>
              )}
              {summary.elapsedSeconds !== null && (
                <span>{formatDuration(summary.elapsedSeconds)}</span>
              )}
            </div>

            {isPreparing && (
              <div
                className={styles.bannerProgress}
                data-indeterminate={summary.percent === null ? 'true' : undefined}
                role="progressbar"
                aria-label="模型准备总进度"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={summary.percent ?? undefined}
              >
                <span
                  className={styles.bannerProgressFill}
                  style={summary.percent !== null ? { width: `${summary.percent}%` } : undefined}
                />
              </div>
            )}
          </div>

          {orderedModels.length > 0 && (
            <div className={styles.modelList} role="list" aria-label="模型准备清单">
              {orderedModels.map(model => {
                const fullPath = `${model.category}/${model.name}`;
                const badge = formatModelCategoryBadge(model.category);
                const { dir, file } = splitModelName(model.name);
                const pct = model.state === 'downloading' && model.total
                  ? Math.min(100, Math.max(0, Math.round(((model.bytes || 0) / model.total) * 100)))
                  : null;
                const copied = copiedPath === fullPath;

                return (
                  <div
                    key={fullPath}
                    className={styles.modelRow}
                    data-state={model.state}
                    role="listitem"
                  >
                    <div className={styles.modelMain}>
                      <StateIcon state={model.state} />
                      <span
                        className={styles.catBadge}
                        data-tone={badge.tone}
                        title={badge.raw}
                      >
                        {badge.short}
                      </span>
                      <span className={styles.modelName} title={fullPath}>
                        {dir && <span className={styles.modelDir}>{dir}</span>}
                        {file}
                      </span>
                      <div className={styles.modelMeta}>
                        {model.source && (
                          <span className={styles.sourceTag} title={model.host || undefined}>
                            <SourceIcon source={model.source} />
                            {sources[model.source]}
                            {model.host ? ` · ${model.host}` : ''}
                          </span>
                        )}
                        {model.state !== 'ready' && (
                          <span className={styles.stateTag} data-state={model.state}>
                            {pct !== null ? `${pct}%` : phases[model.state]}
                          </span>
                        )}
                        {model.attempt && model.attempt > 1 ? (
                          <span>第 {model.attempt} 次</span>
                        ) : null}
                        {model.seconds !== undefined && (
                          <span className={styles.durationText}>{formatDuration(model.seconds)}</span>
                        )}
                        <button
                          type="button"
                          className={styles.copyBtn}
                          data-copied={copied ? 'true' : undefined}
                          title={copied ? '已复制路径' : `复制路径 ${fullPath}`}
                          aria-label={copied ? `已复制 ${fullPath}` : `复制模型路径 ${fullPath}`}
                          onClick={() => copyModelPath(fullPath)}
                        >
                          {copied
                            ? <Check size={13} strokeWidth={2.2} aria-hidden />
                            : <Copy size={13} strokeWidth={1.8} aria-hidden />}
                        </button>
                      </div>
                    </div>

                    {((model.state === 'downloading' && model.total)
                      || model.fallback
                      || model.integrity === 'transport'
                      || model.error) && (
                      <div className={styles.modelSub}>
                        {model.state === 'downloading' && model.total ? (
                          <div className={styles.progressRow}>
                            <div
                              className={styles.progressTrack}
                              role="progressbar"
                              aria-label={`${model.name} 下载进度`}
                              aria-valuemin={0}
                              aria-valuemax={model.total}
                              aria-valuenow={model.bytes || 0}
                            >
                              <span
                                className={styles.progressFill}
                                style={{ width: `${pct ?? 0}%` }}
                              />
                            </div>
                            <span className={styles.progressStats}>
                              {formatMiB(model.bytes || 0)} / {formatMiB(model.total)}
                            </span>
                          </div>
                        ) : null}
                        {model.integrity === 'transport' && (
                          <span className={styles.subNote}>下载长度已校验，来源 SHA-256 未提供</span>
                        )}
                        {model.fallback && (
                          <span className={styles.fallbackNote}>外部来源失败，已尝试 Drive · {model.fallback}</span>
                        )}
                        {model.error && (
                          <p className={styles.errorNote}>{model.error}</p>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  </details>;
}
