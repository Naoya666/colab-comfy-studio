import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Archive,
  ArrowLeft,
  HardDrive,
  History,
  Loader2,
  Package,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Trash2,
  Workflow,
} from 'lucide-react';
import { apiFetch, apiMessage, cancelRecoveryPlan } from '../lib/gpu-api';
import {
  computeNewlyOrphanedPlugins,
  defaultLinkedPluginSelection,
  formatWorkspaceBytes,
  type WorkspaceHistoryRow,
  type WorkspaceOverview,
  type WorkspacePluginCategory,
  type WorkspacePluginRow,
  type WorkspaceWorkflowItem,
} from '../lib/native-workspace';
import styles from '../gpu.module.css';

type TabKey = 'workflows' | 'plugins' | 'caches';
type PluginFilter = 'all' | 'in_use' | 'cleanable' | 'unused_ui' | 'disabled' | 'manager_disabled';
type PluginSort = 'size' | 'import' | 'name';

const BASE = '/api/gpu/native/workspace';

function formatEpoch(ts?: number | null): string {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString('zh-CN', {
    hour12: false,
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function categoryMeta(cat: WorkspacePluginCategory): { label: string; tone: 'ok' | 'warn' | 'muted' | 'danger' } {
  switch (cat) {
    case 'in_use':
      return { label: '活跃引用中', tone: 'ok' };
    case 'archived_only':
      return { label: '仅归档引用', tone: 'muted' };
    case 'unused_ui':
      return { label: '无节点引用 · 含 UI', tone: 'warn' };
    case 'unused':
      return { label: '无工作流引用', tone: 'muted' };
    case 'manager_disabled':
      return { label: 'Manager 禁用（仍占下载）', tone: 'warn' };
    case 'disabled':
      return { label: '已离线停用（免下载）', tone: 'muted' };
  }
}

export function NativeWorkspaceManager({
  onBack,
  onMutated,
}: {
  onBack: () => void;
  onMutated?: () => void;
}) {
  const [tab, setTab] = useState<TabKey>('workflows');
  const [data, setData] = useState<WorkspaceOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [keepHistory, setKeepHistory] = useState<number>(5);

  // Workflows tab state
  const [selectedWfs, setSelectedWfs] = useState<Record<string, boolean>>({});
  const [wfModalAction, setWfModalAction] = useState<'archive' | 'trash' | null>(null);
  const [linkedPluginAction, setLinkedPluginAction] = useState<'disable' | 'remove'>('disable');
  const [linkedPluginSelection, setLinkedPluginSelection] = useState<Record<string, boolean>>({});
  const wfDialogRef = useRef<HTMLDialogElement>(null);

  // Plugins tab state
  const [pluginFilter, setPluginFilter] = useState<PluginFilter>('all');
  const [pluginSort, setPluginSort] = useState<PluginSort>('size');
  const [selectedPlugins, setSelectedPlugins] = useState<Record<string, boolean>>({});
  const [highlightedPlugin, setHighlightedPlugin] = useState<string | null>(null);
  const [pluginConfirm, setPluginConfirm] = useState<{
    action: 'disable' | 'remove' | 'enable' | 'purge_disabled';
    paths: string[];
    inUseRows: WorkspacePluginRow[];
  } | null>(null);
  const pluginDialogRef = useRef<HTMLDialogElement>(null);
  const [rollbackTarget, setRollbackTarget] = useState<WorkspaceHistoryRow | null>(null);
  const rollbackDialogRef = useRef<HTMLDialogElement>(null);

  // Caches tab state
  const [confirmClearAllCache, setConfirmClearAllCache] = useState(false);
  const [confirmCleanOrphans, setConfirmCleanOrphans] = useState(false);
  const [confirmCancelRecovery, setConfirmCancelRecovery] = useState(false);

  const load = useCallback(async (opts?: { refresh?: boolean; keep?: number }) => {
    const k = opts?.keep ?? keepHistory;
    const refresh = opts?.refresh ? 'true' : 'false';
    setLoading(true);
    setError('');
    try {
      const overview = await apiFetch<WorkspaceOverview>(`${BASE}/overview?refresh=${refresh}&keep_history=${k}`);
      setData(overview);
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setLoading(false);
    }
  }, [keepHistory]);

  useEffect(() => {
    void load();
  }, [load]);

  const writable = Boolean(data?.offline_writable && data?.drive_connected);

  const selectedActiveWorkflowItems = useMemo(() => {
    if (!data) return [] as WorkspaceWorkflowItem[];
    return data.workflows.active.filter(wf => selectedWfs[`${wf.source}:${wf.path}`]);
  }, [data, selectedWfs]);

  const newlyOrphaned = useMemo(() => {
    if (!data || !selectedActiveWorkflowItems.length) return [] as WorkspacePluginRow[];
    return computeNewlyOrphanedPlugins(
      selectedActiveWorkflowItems.map(w => ({ source: w.source, path: w.path })),
      data.plugins.active,
    );
  }, [data, selectedActiveWorkflowItems]);

  function openWorkflowActionModal(action: 'archive' | 'trash') {
    if (!selectedActiveWorkflowItems.length) return;
    setLinkedPluginSelection(defaultLinkedPluginSelection(newlyOrphaned));
    setLinkedPluginAction('disable');
    setWfModalAction(action);
    wfDialogRef.current?.showModal();
  }

  async function executeWorkflowModal() {
    if (!wfModalAction || !selectedActiveWorkflowItems.length || busy) return;
    const alsoPaths = newlyOrphaned.filter(p => linkedPluginSelection[p.path]).map(p => p.path);
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const res = await apiFetch<{
        processed: unknown[];
        plugins_result?: { changed_paths?: string[]; generation?: string } | null;
      }>(`${BASE}/workflows/action`, {
        method: 'POST',
        body: JSON.stringify({
          action: wfModalAction,
          items: selectedActiveWorkflowItems.map(w => ({ source: w.source, path: w.path })),
          ...(alsoPaths.length ? { also_plugins: { action: linkedPluginAction, paths: alsoPaths } } : {}),
        }),
      });
      wfDialogRef.current?.close();
      setWfModalAction(null);
      setSelectedWfs({});
      const wfVerb = wfModalAction === 'archive' ? '停用归档' : '移入 Google Drive 回收站';
      const plMsg = res.plugins_result?.changed_paths?.length
        ? `，并同步${linkedPluginAction === 'disable' ? '离线停用' : '卸载'}了 ${res.plugins_result.changed_paths.length} 个独占插件（将于下次启动生效）`
        : '';
      setNotice(`已${wfVerb} ${res.processed.length} 个工作流${plMsg}。`);
      onMutated?.();
      await load({ refresh: true });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function runSingleWorkflowAction(action: 'archive' | 'restore' | 'trash', wf: WorkspaceWorkflowItem) {
    if (busy || !writable) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      await apiFetch(`${BASE}/workflows/action`, {
        method: 'POST',
        body: JSON.stringify({ action, items: [{ source: wf.source, path: wf.path }] }),
      });
      const label = action === 'restore' ? '已恢复到原生工作流列表' : action === 'archive' ? '已停用归档' : '已移入回收站';
      setNotice(`${wf.path}：${label}。`);
      onMutated?.();
      await load({ refresh: true });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function requestPluginAction(action: 'disable' | 'remove' | 'enable' | 'purge_disabled', paths: string[]) {
    if (!data || !paths.length || !writable) return;
    const activeByPath = new Map(data.plugins.active.map(r => [r.path, r]));
    const inUseRows =
      action === 'disable' || action === 'remove'
        ? paths.map(p => activeByPath.get(p)).filter((r): r is WorkspacePluginRow => Boolean(r && r.used_by_active.length))
        : [];
    if (inUseRows.length > 0 || action === 'remove' || action === 'purge_disabled') {
      setPluginConfirm({ action, paths, inUseRows });
      pluginDialogRef.current?.showModal();
      return;
    }
    void executePluginAction(action, paths, false);
  }

  async function executePluginAction(
    action: 'disable' | 'remove' | 'enable' | 'purge_disabled',
    paths: string[],
    forceInUse: boolean,
  ) {
    if (busy) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const res = await apiFetch<{ changed_paths: string[]; generation: string }>(`${BASE}/plugins/action`, {
        method: 'POST',
        body: JSON.stringify({ action, paths, force_in_use: forceInUse }),
      });
      pluginDialogRef.current?.close();
      setPluginConfirm(null);
      setSelectedPlugins({});
      const verb: Record<string, string> = {
        disable: '已离线停用',
        enable: '已恢复启用',
        remove: '已卸载',
        purge_disabled: '已移除停用记录',
      };
      setNotice(`${verb[action]} ${res.changed_paths.length} 个插件（目标清单 ${res.generation.slice(0, 8)}，将于下次启动生效）。`);
      onMutated?.();
      await load({ refresh: true });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function executeRollback() {
    if (!rollbackTarget || busy) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const res = await apiFetch<{ generation: string; plugin_count: number }>(`${BASE}/plugins/rollback`, {
        method: 'POST',
        body: JSON.stringify({ generation: rollbackTarget.generation }),
      });
      rollbackDialogRef.current?.close();
      setRollbackTarget(null);
      setNotice(`已回滚至快照 ${res.generation.slice(0, 8)}（共 ${res.plugin_count} 个插件），将于下次启动生效。`);
      onMutated?.();
      await load({ refresh: true });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function executeDeleteEnvCache(opts: { keys?: string[]; all?: boolean }) {
    if (busy || !writable) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const res = await apiFetch<{ deleted_files: number; freed_bytes: number }>(`${BASE}/env-cache`, {
        method: 'DELETE',
        body: JSON.stringify({ ...(opts.keys ? { keys: opts.keys } : {}), ...(opts.all ? { all: true } : {}) }),
      });
      setConfirmClearAllCache(false);
      setNotice(
        opts.all
          ? `已清空全部构建缓存（删除 ${res.deleted_files} 个文件，释放 ${formatWorkspaceBytes(res.freed_bytes)}），下次启动将进入全新冷构建。`
          : `已删除指定缓存（释放 ${formatWorkspaceBytes(res.freed_bytes)}）。`,
      );
      onMutated?.();
      await load({ refresh: true });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function executeCleanupOrphans() {
    if (busy || !writable) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const res = await apiFetch<{ deleted_count: number; freed_bytes: number; keep_history: number }>(
        `${BASE}/orphan-objects/cleanup`,
        {
          method: 'POST',
          body: JSON.stringify({ keep_history: keepHistory, dry_run: false }),
        },
      );
      setConfirmCleanOrphans(false);
      setNotice(
        `已清理 ${res.deleted_count} 个孤立插件归档，释放 ${formatWorkspaceBytes(res.freed_bytes)}（已保留最近 ${res.keep_history} 版历史快照）。`,
      );
      onMutated?.();
      await load({ refresh: true });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  const filteredActivePlugins = useMemo(() => {
    if (!data) return [] as WorkspacePluginRow[];
    let rows = [...data.plugins.active];
    if (pluginFilter === 'in_use') rows = rows.filter(r => r.status_category === 'in_use');
    else if (pluginFilter === 'cleanable')
      rows = rows.filter(r => r.status_category === 'archived_only' || r.status_category === 'unused');
    else if (pluginFilter === 'unused_ui') rows = rows.filter(r => r.status_category === 'unused_ui');
    else if (pluginFilter === 'manager_disabled') rows = rows.filter(r => r.status_category === 'manager_disabled');
    else if (pluginFilter === 'disabled') rows = [];

    rows.sort((a, b) => {
      if (pluginSort === 'size') return b.archive_size_bytes - a.archive_size_bytes || a.path.localeCompare(b.path);
      if (pluginSort === 'import')
        return (b.import_seconds ?? -1) - (a.import_seconds ?? -1) || b.archive_size_bytes - a.archive_size_bytes;
      return a.path.localeCompare(b.path);
    });
    return rows;
  }, [data, pluginFilter, pluginSort]);

  const selectedPluginPaths = useMemo(
    () => filteredActivePlugins.filter(r => selectedPlugins[r.path]).map(r => r.path),
    [filteredActivePlugins, selectedPlugins],
  );

  const selectedLinkedBytes = useMemo(
    () => newlyOrphaned.filter(p => linkedPluginSelection[p.path]).reduce((acc, p) => acc + p.archive_size_bytes, 0),
    [newlyOrphaned, linkedPluginSelection],
  );
  async function executeCancelRecoveryUnlock() {
    if (busy || !data?.recovery_plan_id) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const res = await cancelRecoveryPlan(data.recovery_plan_id, `workspace-unlock:${data.recovery_plan_id}`);
      setConfirmCancelRecovery(false);
      setNotice(
        res.state === 'cancelling'
          ? '已提交取消自动恢复请求，正在等待在途租约释放确认后解锁写入。'
          : '已取消原生自动恢复计划，离线管理写入权限已解锁。',
      );
      onMutated?.();
      await load({ refresh: true });
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  const selectedLinkedImportSec = useMemo(
    () =>
      newlyOrphaned
        .filter(p => linkedPluginSelection[p.path])
        .reduce((acc, p) => acc + (typeof p.import_seconds === 'number' ? p.import_seconds : 0), 0),
    [newlyOrphaned, linkedPluginSelection],
  );

  return (
    <section className={styles.nativeWorkspace} aria-label="原生 ComfyUI 离线管理">
      <div className="actions nativeApiHeading">
        <div className={styles.wsHeadingGroup}>
          <button type="button" className="button secondary" onClick={onBack}>
            <ArrowLeft size={16} aria-hidden /> 返回原生 ComfyUI
          </button>
          <h1>工作流、插件与构建缓存管理</h1>
        </div>
        <div className="actions">
          {data?.drive_account_alias && (
            <span className={styles.sizeBadge} title="当前操作的 Google Drive 主账号">
              Drive：{data.drive_account_alias}
            </span>
          )}
          <button type="button" className="button secondary" disabled={loading || busy} onClick={() => void load({ refresh: true })}>
            <RefreshCw size={15} className={loading ? styles.spin : undefined} aria-hidden /> 刷新
          </button>
        </div>
      </div>

      <p className="muted small">
        免开 GPU、零 CU 消耗直接管理 Google Drive 中的原生工作流、自定义节点与构建缓存。所有修改均在原生实例离线时安全写入，并于下次启动时生效。
      </p>

      {data && !data.offline_writable && (
        <div className={styles.wsBannerWarn} role="status">
          <ShieldAlert size={18} aria-hidden />
          <div style={{ flex: 1 }}>
            <strong>当前为只读模式</strong>
            <p className="small">{data.lock_reason || '原生 ComfyUI 实例正在运行或启动中，请等待实例完全停止后再执行修改。'}</p>
            {data.recovery_plan_id && (
              <div className="actions" style={{ marginTop: 8 }}>
                {!confirmCancelRecovery ? (
                  <button
                    type="button"
                    className="button secondary"
                    disabled={busy}
                    onClick={() => setConfirmCancelRecovery(true)}
                  >
                    取消自动恢复以解锁
                  </button>
                ) : (
                  <>
                    <span className="small">确认取消原生自动恢复计划并解锁离线写入？</span>
                    <button
                      type="button"
                      className="button danger"
                      disabled={busy}
                      onClick={() => void executeCancelRecoveryUnlock()}
                    >
                      {busy ? '正在取消…' : '确认取消并解锁'}
                    </button>
                    <button
                      type="button"
                      className="button secondary"
                      disabled={busy}
                      onClick={() => setConfirmCancelRecovery(false)}
                    >
                      暂不取消
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {data?.plugin_state.has_pending_offline_changes && (
        <div className={styles.wsBannerInfo} role="status">
          <div>
            <strong>已有离线插件变更待生效</strong>
            <p className="small">
              目标插件清单 <code>{data.plugin_state.pending_generation?.slice(0, 8)}</code>
              {data.plugin_state.head_generation ? `（上次已验证：${data.plugin_state.head_generation.slice(0, 8)}）` : ''}
              ，将在下次启动原生 ComfyUI 时自动恢复并验证。
            </p>
          </div>
        </div>
      )}

      {error && <p className="status error" role="alert">{error}</p>}
      {notice && <p className="status" role="status">{notice}</p>}

      {loading && !data ? (
        <div className={styles.wsEmpty}>
          <Loader2 size={20} className={styles.spin} aria-hidden /> 正在从 Google Drive 读取工作流、插件清单与缓存信息…
        </div>
      ) : data && !data.drive_connected ? (
        <div className={styles.wsCard}>
          <p>{data.message || '尚未连接 Google Drive 账号，无法读取原生 ComfyUI 工作区。'}</p>
        </div>
      ) : data ? (
        <>
          <div className={styles.wsSummaryGrid}>
            <div className={styles.wsStatCard}>
              <span className="muted small">活跃原生工作流</span>
              <strong>{data.workflows.active.length} 个</strong>
              <span className="muted small">
                已归档 {data.workflows.archived.length} · 预热 {data.workflows.warmup.length}
              </span>
            </div>
            <div className={styles.wsStatCard}>
              <span className="muted small">启用中插件归档</span>
              <strong>{formatWorkspaceBytes(data.plugins.total_active_archive_bytes)}</strong>
              <span className="muted small">
                启用 {data.plugins.active.length} 项 · 离线停用 {data.plugins.disabled.length} 项
              </span>
            </div>
            <div className={styles.wsStatCard}>
              <span className="muted small">上次启动插件导入耗时</span>
              <strong>
                {data.plugin_state.telemetry_available
                  ? `${data.plugins.total_active_import_seconds.toFixed(1)} s`
                  : '待采集'}
              </strong>
              <span className="muted small">
                {data.plugin_state.telemetry_available
                  ? `${data.plugin_state.telemetry_gpu || 'GPU'} · ${formatEpoch(data.plugin_state.telemetry_collected_at)}`
                  : '下次启动原生 ComfyUI 后自动采集'}
              </span>
            </div>
            <div className={styles.wsStatCard}>
              <span className="muted small">构建缓存 / 可回收归档</span>
              <strong>{formatWorkspaceBytes(data.env_caches.total_bytes)}</strong>
              <span className="muted small">
                缓存 {data.env_caches.items.length} 份 · 孤立归档 {formatWorkspaceBytes(data.orphan_objects.orphan_bytes)}
              </span>
            </div>
          </div>

          <div className={styles.chipRow} role="tablist" aria-label="管理分类">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'workflows'}
              className={styles.chip}
              data-active={tab === 'workflows' ? 'true' : undefined}
              onClick={() => setTab('workflows')}
            >
              <Workflow size={15} aria-hidden /> 工作流与联动清理（{data.workflows.active.length}）
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'plugins'}
              className={styles.chip}
              data-active={tab === 'plugins' ? 'true' : undefined}
              onClick={() => setTab('plugins')}
            >
              <Package size={15} aria-hidden /> 自定义节点 / 插件（{data.plugins.active.length + data.plugins.disabled.length}）
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'caches'}
              className={styles.chip}
              data-active={tab === 'caches' ? 'true' : undefined}
              onClick={() => setTab('caches')}
            >
              <HardDrive size={15} aria-hidden /> 构建缓存与空间（{data.env_caches.items.length}）
            </button>
          </div>

          {tab === 'workflows' && (
            <div className={styles.wsSectionStack}>
              <div className={styles.wsCard}>
                <div className={styles.wsCardHead}>
                  <div>
                    <h2>活跃原生工作流（{data.workflows.active.length}）</h2>
                    <p className="muted small">
                      勾选不再使用的工作流进行停用归档或移入回收站，系统会自动计算可随之安全释放的独占插件。
                    </p>
                  </div>
                  <div className="actions">
                    {data.workflows.active.length > 0 && (
                      <button
                        type="button"
                        className="button secondary"
                        onClick={() => {
                          if (selectedActiveWorkflowItems.length === data.workflows.active.length) {
                            setSelectedWfs({});
                          } else {
                            const all: Record<string, boolean> = {};
                            for (const w of data.workflows.active) all[`${w.source}:${w.path}`] = true;
                            setSelectedWfs(all);
                          }
                        }}
                      >
                        {selectedActiveWorkflowItems.length === data.workflows.active.length ? '取消全选' : '全选'}
                      </button>
                    )}
                    <button
                      type="button"
                      className="button secondary"
                      disabled={!writable || busy || !selectedActiveWorkflowItems.length}
                      onClick={() => openWorkflowActionModal('archive')}
                    >
                      <Archive size={15} aria-hidden /> 停用归档（{selectedActiveWorkflowItems.length}）
                    </button>
                    <button
                      type="button"
                      className="button danger"
                      disabled={!writable || busy || !selectedActiveWorkflowItems.length}
                      onClick={() => openWorkflowActionModal('trash')}
                    >
                      <Trash2 size={15} aria-hidden /> 移入回收站（{selectedActiveWorkflowItems.length}）
                    </button>
                  </div>
                </div>

                {data.workflows.active.length === 0 ? (
                  <p className="muted">Drive 中暂无原生工作流文件。</p>
                ) : (
                  <div className={styles.wsList}>
                    {data.workflows.active.map(wf => {
                      const key = `${wf.source}:${wf.path}`;
                      const checked = Boolean(selectedWfs[key]);
                      return (
                        <div key={key} className={styles.wsRow} data-selected={checked ? 'true' : undefined}>
                          <label className={styles.wsCheck}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={e => setSelectedWfs(cur => ({ ...cur, [key]: e.target.checked }))}
                            />
                            <div className={styles.wsRowMain}>
                              <div className={styles.wsRowTitle}>
                                <strong>{wf.path}</strong>
                                <span className="muted small">
                                  {formatWorkspaceBytes(wf.size_bytes)} · {wf.node_count} 节点 · {formatEpoch(wf.modified_at)}
                                </span>
                              </div>
                              <div className={styles.wsTagList}>
                                {wf.plugins.length === 0 ? (
                                  <span className={styles.wsBadgeMuted}>纯内置节点</span>
                                ) : (
                                  wf.plugins.map(pl => (
                                    <button
                                      key={pl}
                                      type="button"
                                      className={styles.wsPluginChip}
                                      title={`引用节点：${(wf.plugin_nodes[pl] || []).join(', ')}`}
                                      onClick={ev => {
                                        ev.preventDefault();
                                        setHighlightedPlugin(pl);
                                        setPluginFilter('all');
                                        setTab('plugins');
                                      }}
                                    >
                                      {pl}
                                    </button>
                                  ))
                                )}
                                {wf.unresolved_nodes.length > 0 && (
                                  <span
                                    className={styles.wsBadgeWarn}
                                    title={`未安装或未识别节点：${wf.unresolved_nodes.map(u => u.type).join(', ')}`}
                                  >
                                    未安装节点 ×{wf.unresolved_nodes.length}
                                  </span>
                                )}
                                {wf.parse_error && <span className={styles.wsBadgeDanger}>{wf.parse_error}</span>}
                              </div>
                            </div>
                          </label>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {data.workflows.archived.length > 0 && (
                <div className={styles.wsCard}>
                  <div className={styles.wsCardHead}>
                    <div>
                      <h2>已停用归档工作流（{data.workflows.archived.length}）</h2>
                      <p className="muted small">
                        位于 <code>native-comfy/workflows-archived/</code>，不在 ComfyUI 列表显示且不保护插件，可随时一键恢复。
                      </p>
                    </div>
                  </div>
                  <div className={styles.wsList}>
                    {data.workflows.archived.map(wf => (
                      <div key={`archived:${wf.path}`} className={styles.wsRow}>
                        <div className={styles.wsRowMain}>
                          <div className={styles.wsRowTitle}>
                            <strong>{wf.path}</strong>
                            <span className="muted small">
                              {formatWorkspaceBytes(wf.size_bytes)} · {wf.node_count} 节点
                            </span>
                          </div>
                          <div className={styles.wsTagList}>
                            {wf.plugins.length === 0 ? (
                              <span className={styles.wsBadgeMuted}>纯内置节点</span>
                            ) : (
                              wf.plugins.map(pl => (
                                <span key={pl} className={styles.wsBadgeMuted}>{pl}</span>
                              ))
                            )}
                          </div>
                        </div>
                        <div className="actions">
                          <button
                            type="button"
                            className="button secondary"
                            disabled={!writable || busy}
                            onClick={() => void runSingleWorkflowAction('restore', wf)}
                          >
                            <RotateCcw size={14} aria-hidden /> 恢复启用
                          </button>
                          <button
                            type="button"
                            className="button danger"
                            disabled={!writable || busy}
                            onClick={() => void runSingleWorkflowAction('trash', wf)}
                          >
                            移入回收站
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {data.workflows.warmup.length > 0 && (
                <div className={styles.wsCard}>
                  <div className={styles.wsCardHead}>
                    <div>
                      <h2>API 预热工作流（{data.workflows.warmup.length}）</h2>
                      <p className="muted small">位于 <code>native-comfy/warmup/</code>，计入活跃插件依赖保护。</p>
                    </div>
                  </div>
                  <div className={styles.wsList}>
                    {data.workflows.warmup.map(wf => (
                      <div key={`warmup:${wf.path}`} className={styles.wsRow}>
                        <div className={styles.wsRowMain}>
                          <div className={styles.wsRowTitle}>
                            <strong>{wf.path}</strong>
                            <span className="muted small">{formatWorkspaceBytes(wf.size_bytes)} · {wf.node_count} 节点</span>
                          </div>
                          <div className={styles.wsTagList}>
                            {wf.plugins.length === 0
                              ? <span className={styles.wsBadgeMuted}>纯内置节点</span>
                              : wf.plugins.map(pl => <span key={pl} className={styles.wsBadgeOk}>{pl}</span>)}
                          </div>
                        </div>
                        <button
                          type="button"
                          className="button danger"
                          disabled={!writable || busy}
                          onClick={() => void runSingleWorkflowAction('trash', wf)}
                        >
                          移入回收站
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <details className={styles.wsCard}>
                <summary className={styles.wsSummaryToggle}>
                  Studio 创作页已发布工作流（{data.workflows.studio_published.length} 个 · 只读依赖保护参考）
                </summary>
                <div className={styles.wsList} style={{ marginTop: 10 }}>
                  {data.workflows.studio_published.map(wf => (
                    <div key={`studio:${wf.id}`} className={styles.wsRow}>
                      <div className={styles.wsRowMain}>
                        <div className={styles.wsRowTitle}>
                          <strong>{wf.path}</strong>
                          <span className="muted small">{wf.node_count} 节点</span>
                        </div>
                        <div className={styles.wsTagList}>
                          {wf.plugins.length === 0
                            ? <span className={styles.wsBadgeMuted}>纯内置节点</span>
                            : wf.plugins.map(pl => <span key={pl} className={styles.wsBadgeOk}>{pl}</span>)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}

          {tab === 'plugins' && (
            <div className={styles.wsSectionStack}>
              <div className={styles.wsCard}>
                <div className={styles.wsCardHead}>
                  <div className={styles.chipRow}>
                    {(
                      [
                        ['all', `全部启用（${data.plugins.active.length}）`],
                        ['in_use', `活跃引用（${data.plugins.active.filter(r => r.status_category === 'in_use').length}）`],
                        [
                          'cleanable',
                          `可安全清理（${data.plugins.active.filter(r => r.status_category === 'archived_only' || r.status_category === 'unused').length}）`,
                        ],
                        ['unused_ui', `无节点·含 UI（${data.plugins.active.filter(r => r.status_category === 'unused_ui').length}）`],
                        ['manager_disabled', `Manager 禁用（${data.plugins.active.filter(r => r.status_category === 'manager_disabled').length}）`],
                        ['disabled', `已离线停用（${data.plugins.disabled.length}）`],
                      ] as [PluginFilter, string][]
                    ).map(([k, label]) => (
                      <button
                        key={k}
                        type="button"
                        className={styles.chip}
                        data-active={pluginFilter === k ? 'true' : undefined}
                        onClick={() => {
                          setPluginFilter(k);
                          setSelectedPlugins({});
                        }}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <div className="actions">
                    <label className="small muted">
                      排序{' '}
                      <select value={pluginSort} onChange={e => setPluginSort(e.target.value as PluginSort)}>
                        <option value="size">按归档体积（大→小）</option>
                        <option value="import">按启动导入耗时（慢→快）</option>
                        <option value="name">按名称（A→Z）</option>
                      </select>
                    </label>
                    {selectedPluginPaths.length > 0 && (
                      <>
                        <button
                          type="button"
                          className="button secondary"
                          disabled={!writable || busy}
                          onClick={() => requestPluginAction('disable', selectedPluginPaths)}
                        >
                          批量离线停用（{selectedPluginPaths.length}）
                        </button>
                        <button
                          type="button"
                          className="button danger"
                          disabled={!writable || busy}
                          onClick={() => requestPluginAction('remove', selectedPluginPaths)}
                        >
                          批量卸载（{selectedPluginPaths.length}）
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {pluginFilter !== 'disabled' && (
                  filteredActivePlugins.length === 0 ? (
                    <p className="muted">当前筛选下没有插件。</p>
                  ) : (
                    <div className={styles.wsList}>
                      {filteredActivePlugins.map(plugin => {
                        const badge = categoryMeta(plugin.status_category);
                        const checked = Boolean(selectedPlugins[plugin.path]);
                        const highlighted = highlightedPlugin === plugin.path;
                        return (
                          <div
                            key={plugin.path}
                            className={styles.wsRow}
                            data-selected={checked || highlighted ? 'true' : undefined}
                          >
                            <label className={styles.wsCheck}>
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={e => setSelectedPlugins(cur => ({ ...cur, [plugin.path]: e.target.checked }))}
                              />
                              <div className={styles.wsRowMain}>
                                <div className={styles.wsRowTitle}>
                                  <strong>{plugin.path}</strong>
                                  {plugin.version && <span className="muted small">v{plugin.version}</span>}
                                  <span
                                    className={
                                      badge.tone === 'ok'
                                        ? styles.wsBadgeOk
                                        : badge.tone === 'warn'
                                          ? styles.wsBadgeWarn
                                          : badge.tone === 'danger'
                                            ? styles.wsBadgeDanger
                                            : styles.wsBadgeMuted
                                    }
                                  >
                                    {badge.label}
                                  </span>
                                </div>
                                <div className={styles.wsMetaLine}>
                                  <span>归档：<strong>{formatWorkspaceBytes(plugin.archive_size_bytes)}</strong></span>
                                  <span>
                                    启动导入：
                                    <strong>
                                      {typeof plugin.import_seconds === 'number'
                                        ? `${plugin.import_seconds.toFixed(2)} s`
                                        : '待下次开机采集'}
                                    </strong>
                                  </span>
                                  {typeof plugin.node_count === 'number' && <span>{plugin.node_count} 节点</span>}
                                  {typeof plugin.js_count === 'number' && plugin.js_count > 0 && <span>{plugin.js_count} 前端 JS</span>}
                                  {plugin.has_install_py && <span className={styles.wsBadgeWarn}>含 install.py</span>}
                                  {plugin.has_requirements && <span className={styles.wsBadgeMuted}>requirements.txt</span>}
                                </div>
                                <div className={styles.wsTagList}>
                                  {plugin.used_by_active.map(ref => (
                                    <span key={`${ref.source}:${ref.path}`} className={styles.wsBadgeOk}>
                                      {ref.source === 'studio_db' ? `Studio: ${ref.path}` : ref.source === 'warmup' ? `预热: ${ref.path}` : ref.path}
                                    </span>
                                  ))}
                                  {plugin.used_by_archived.map(ref => (
                                    <span key={`arch:${ref.path}`} className={styles.wsBadgeMuted}>
                                      已归档: {ref.path}
                                    </span>
                                  ))}
                                  {plugin.status_category === 'unused_ui' && (
                                    <span className="muted small">可能为全局界面辅助工具（如标签补全/翻译/工作流商店），请按日常习惯决定。</span>
                                  )}
                                </div>
                              </div>
                            </label>
                            <div className="actions">
                              <button
                                type="button"
                                className="button secondary"
                                disabled={!writable || busy}
                                onClick={() => requestPluginAction('disable', [plugin.path])}
                              >
                                {plugin.status_category === 'manager_disabled' ? '转为离线停用（免下载）' : '离线停用'}
                              </button>
                              <button
                                type="button"
                                className="button danger"
                                disabled={!writable || busy}
                                onClick={() => requestPluginAction('remove', [plugin.path])}
                              >
                                卸载
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )
                )}

                {(pluginFilter === 'all' || pluginFilter === 'disabled') && data.plugins.disabled.length > 0 && (
                  <div style={{ marginTop: pluginFilter === 'disabled' ? 0 : 18 }}>
                    <h3>已离线停用插件（{data.plugins.disabled.length}）</h3>
                    <p className="muted small">
                      已从下次启动清单移出，冷启动时完全跳过下载、解压、依赖安装与导入；归档保留在 Drive，可随时一键恢复。
                    </p>
                    <div className={styles.wsList}>
                      {data.plugins.disabled.map(plugin => (
                        <div key={`dis:${plugin.path}`} className={styles.wsRow}>
                          <div className={styles.wsRowMain}>
                            <div className={styles.wsRowTitle}>
                              <strong>{plugin.path}</strong>
                              {plugin.version && <span className="muted small">v{plugin.version}</span>}
                              <span className={styles.wsBadgeMuted}>已离线停用</span>
                              {!plugin.archive_present && <span className={styles.wsBadgeDanger}>归档已丢失</span>}
                            </div>
                            <div className={styles.wsMetaLine}>
                              <span>归档：{formatWorkspaceBytes(plugin.archive_size_bytes)}</span>
                              <span>停用时间：{formatEpoch(plugin.disabled_at)}</span>
                            </div>
                          </div>
                          <div className="actions">
                            <button
                              type="button"
                              className="button secondary"
                              disabled={!writable || busy || !plugin.archive_present}
                              onClick={() => requestPluginAction('enable', [plugin.path])}
                            >
                              <RotateCcw size={14} aria-hidden /> 恢复启用
                            </button>
                            <button
                              type="button"
                              className="button danger"
                              disabled={!writable || busy}
                              onClick={() => requestPluginAction('purge_disabled', [plugin.path])}
                            >
                              移除记录
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <details className={styles.wsCard}>
                <summary className={styles.wsSummaryToggle}>
                  <History size={15} aria-hidden /> 历史插件快照（最近 {data.plugins.history.length} 版 · 一键回滚）
                </summary>
                <div className={styles.wsList} style={{ marginTop: 10 }}>
                  {data.plugins.history.length === 0 ? (
                    <p className="muted">暂无历史插件快照。</p>
                  ) : (
                    data.plugins.history.map(h => (
                      <div key={h.generation} className={styles.wsRow}>
                        <div className={styles.wsRowMain}>
                          <div className={styles.wsRowTitle}>
                            <code>{h.generation.slice(0, 12)}</code>
                            <span className="muted small">
                              {formatEpoch(h.backed_up_at)} · {h.plugin_count} 个插件 · {formatWorkspaceBytes(h.total_archive_bytes)}
                            </span>
                            {h.is_head && <span className={styles.wsBadgeOk}>当前已验证 head</span>}
                            {h.is_pending && <span className={styles.wsBadgeWarn}>待生效 pending</span>}
                            {!h.restorable && (
                              <span className={styles.wsBadgeDanger} title={`缺失归档：${h.missing_archives.join(', ')}`}>
                                归档已清理 · 不可回滚
                              </span>
                            )}
                          </div>
                        </div>
                        <button
                          type="button"
                          className="button secondary"
                          disabled={!writable || busy || !h.restorable || h.is_pending}
                          onClick={() => {
                            setRollbackTarget(h);
                            rollbackDialogRef.current?.showModal();
                          }}
                        >
                          回滚到此版本
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </details>
            </div>
          )}

          {tab === 'caches' && (
            <div className={styles.wsSectionStack}>
              <div className={styles.wsCard}>
                <div className={styles.wsCardHead}>
                  <div>
                    <h2>Python 环境构建缓存（<code>native-comfy/env-cache/</code>）</h2>
                    <p className="muted small">
                      共 {data.env_caches.items.length} 份缓存，合计 {formatWorkspaceBytes(data.env_caches.total_bytes)}。删除缓存后，下次冷启动将自动执行干净的冷构建并生成新缓存包。
                    </p>
                  </div>
                  <div className="actions">
                    {!confirmClearAllCache ? (
                      <button
                        type="button"
                        className="button danger"
                        disabled={!writable || busy || data.env_caches.items.length === 0}
                        onClick={() => setConfirmClearAllCache(true)}
                      >
                        <Trash2 size={15} aria-hidden /> 清空全部缓存（下次强制冷构建）
                      </button>
                    ) : (
                      <>
                        <button type="button" className="button secondary" onClick={() => setConfirmClearAllCache(false)}>
                          取消
                        </button>
                        <button
                          type="button"
                          className="button danger"
                          disabled={!writable || busy}
                          onClick={() => void executeDeleteEnvCache({ all: true })}
                        >
                          确认清空全部缓存
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {data.env_caches.items.length === 0 ? (
                  <p className="muted">当前 Drive 中没有环境构建缓存（下次冷启动后将自动在后台生成）。</p>
                ) : (
                  <div className={styles.wsList}>
                    {data.env_caches.items.map(cache => (
                      <div key={cache.key} className={styles.wsRow}>
                        <div className={styles.wsRowMain}>
                          <div className={styles.wsRowTitle}>
                            <code className={styles.wsCacheKey}>{cache.key}</code>
                            {cache.schema != null && (
                              <span
                                className={
                                  cache.flags.includes('legacy_schema') ? styles.wsBadgeWarn : styles.wsBadgeOk
                                }
                              >
                                v{cache.schema}
                                {cache.flags.includes('legacy_schema') ? ' · 旧版' : ''}
                              </span>
                            )}
                            {cache.flags.includes('oversized') && (
                              <span className={styles.wsBadgeDanger}>异常偏大（&gt;500 MB）</span>
                            )}
                            {cache.flags.includes('incomplete') && (
                              <span className={styles.wsBadgeWarn}>半成品残缺</span>
                            )}
                          </div>
                          <div className={styles.wsMetaLine}>
                            <span>体积：<strong>{formatWorkspaceBytes(cache.size_bytes)}</strong></span>
                            <span>创建：{formatEpoch(cache.created_at)}</span>
                            {cache.python && <span>Python {cache.python}</span>}
                            {cache.comfy_commit && <span>ComfyUI {cache.comfy_commit.slice(0, 8)}</span>}
                            {typeof cache.added_count === 'number' && <span>{cache.added_count} 包条目</span>}
                          </div>
                        </div>
                        <button
                          type="button"
                          className="button danger"
                          disabled={!writable || busy}
                          onClick={() => void executeDeleteEnvCache({ keys: [cache.key] })}
                        >
                          删除
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className={styles.wsCard}>
                <div className={styles.wsCardHead}>
                  <div>
                    <h2>孤立插件归档清理（<code>native-comfy/plugins/objects/</code>）</h2>
                    <p className="muted small">
                      卸载或更新插件后，历史归档对象会保留以支持回滚。自动保护当前 <code>head</code>、<code>pending</code>、已停用列表以及最近 N 版历史快照引用的对象，其余孤立归档可安全清理释放空间。
                    </p>
                  </div>
                  <div className="actions">
                    <label className="small muted">
                      保留历史快照{' '}
                      <select
                        value={keepHistory}
                        disabled={loading || busy}
                        onChange={e => {
                          const next = Number(e.target.value);
                          setKeepHistory(next);
                          void load({ refresh: true, keep: next });
                        }}
                      >
                        <option value={3}>最近 3 版</option>
                        <option value={5}>最近 5 版（默认）</option>
                        <option value={10}>最近 10 版</option>
                      </select>
                    </label>
                    {!confirmCleanOrphans ? (
                      <button
                        type="button"
                        className="button danger"
                        disabled={!writable || busy || data.orphan_objects.orphan_count === 0}
                        onClick={() => setConfirmCleanOrphans(true)}
                      >
                        清理孤立归档（可释放 {formatWorkspaceBytes(data.orphan_objects.orphan_bytes)}）
                      </button>
                    ) : (
                      <>
                        <button type="button" className="button secondary" onClick={() => setConfirmCleanOrphans(false)}>
                          取消
                        </button>
                        <button
                          type="button"
                          className="button danger"
                          disabled={!writable || busy}
                          onClick={() => void executeCleanupOrphans()}
                        >
                          确认清理（{data.orphan_objects.orphan_count} 项）
                        </button>
                      </>
                    )}
                  </div>
                </div>
                <div className={styles.wsMetaLine}>
                  <span>全部归档对象：{data.orphan_objects.total_objects_count} 个（{formatWorkspaceBytes(data.orphan_objects.total_objects_bytes)}）</span>
                  <span>受保护对象：{data.orphan_objects.protected_count} 个（{formatWorkspaceBytes(data.orphan_objects.protected_bytes)}）</span>
                  <span>可回收孤立对象：<strong>{data.orphan_objects.orphan_count} 个（{formatWorkspaceBytes(data.orphan_objects.orphan_bytes)}）</strong></span>
                </div>
                {confirmCleanOrphans && (
                  <p className="small" style={{ marginTop: 8 }}>
                    提示：清理后受保护的当前清单、已停用插件及最近 {keepHistory} 版历史快照不受影响，但更早的历史快照将无法再回滚。
                  </p>
                )}
              </div>
            </div>
          )}
        </>
      ) : null}

      {/* Linked Workflow + Orphaned Plugins Modal */}
      <dialog ref={wfDialogRef} className="nativeRestartDialog" aria-labelledby="ws-wf-dialog-title">
        <h2 id="ws-wf-dialog-title">
          {wfModalAction === 'archive' ? '停用归档所选工作流' : '将所选工作流移入回收站'}
        </h2>
        <p className="small">
          {wfModalAction === 'archive'
            ? `即将把 ${selectedActiveWorkflowItems.length} 个工作流移至 workflows-archived/（不再在 ComfyUI 显示，可随时一键恢复）：`
            : `即将把 ${selectedActiveWorkflowItems.length} 个工作流移入 Google Drive 回收站（30 天内可在云盘恢复）：`}
        </p>
        <ul className={styles.wsDialogList}>
          {selectedActiveWorkflowItems.map(w => <li key={w.path}>{w.path}</li>)}
        </ul>
        <p className="muted small">如曾在 ComfyUI 浏览器顶部标签栏打开过旧工作流，下次打开后手动关闭旧标签页即可。</p>

        {newlyOrphaned.length > 0 && (
          <div className={styles.wsLinkedBox}>
            <strong>可联动清理的独占插件（{newlyOrphaned.length} 个）</strong>
            <p className="muted small">
              以下插件仅被所选工作流引用，移除后不再被任何活跃工作流依赖。已选预计减重{' '}
              <strong>{formatWorkspaceBytes(selectedLinkedBytes)}</strong>
              {selectedLinkedImportSec > 0 ? `，导入省约 ${selectedLinkedImportSec.toFixed(1)} s` : ''}：
            </p>
            <label className="small">
              联动方式{' '}
              <select value={linkedPluginAction} onChange={e => setLinkedPluginAction(e.target.value as 'disable' | 'remove')}>
                <option value="disable">离线停用（免下载 · 可随时恢复，推荐）</option>
                <option value="remove">直接卸载（移出清单）</option>
              </select>
            </label>
            <div className={styles.wsList} style={{ marginTop: 8 }}>
              {newlyOrphaned.map(p => {
                const hasUi = p.js_count == null || p.js_count > 0;
                return (
                  <label key={p.path} className={styles.wsCheck}>
                    <input
                      type="checkbox"
                      checked={Boolean(linkedPluginSelection[p.path])}
                      onChange={e => setLinkedPluginSelection(cur => ({ ...cur, [p.path]: e.target.checked }))}
                    />
                    <span>
                      <strong>{p.path}</strong>（{formatWorkspaceBytes(p.archive_size_bytes)}
                      {typeof p.import_seconds === 'number' ? ` · 导入 ${p.import_seconds.toFixed(1)}s` : ''}）
                      {hasUi && <span className={styles.wsBadgeWarn}>含前端 UI / 未采集，默认不勾选</span>}
                    </span>
                  </label>
                );
              })}
            </div>
            <p className="muted small" style={{ marginTop: 6 }}>
              请确认 SillyTavern 等外部客户端本地工作流未依赖上述插件。
            </p>
          </div>
        )}

        <div className="actions" style={{ marginTop: 14 }}>
          <button
            type="button"
            className="button secondary"
            onClick={() => {
              wfDialogRef.current?.close();
              setWfModalAction(null);
            }}
          >
            取消
          </button>
          <button
            type="button"
            className={wfModalAction === 'trash' ? 'button danger' : 'button'}
            disabled={busy}
            onClick={() => void executeWorkflowModal()}
          >
            {busy ? '正在写入 Drive…' : '确认执行'}
          </button>
        </div>
      </dialog>

      {/* Plugin Confirmation Modal (including in_use force warning) */}
      <dialog ref={pluginDialogRef} className="nativeRestartDialog" aria-labelledby="ws-pl-dialog-title">
        <h2 id="ws-pl-dialog-title">
          {pluginConfirm?.action === 'disable'
            ? '确认离线停用插件'
            : pluginConfirm?.action === 'remove'
              ? '确认卸载插件'
              : '确认移除停用记录'}
        </h2>
        {pluginConfirm && (
          <>
            <p className="small">目标插件：{pluginConfirm.paths.join('、')}</p>
            {pluginConfirm.inUseRows.length > 0 && (
              <div className={styles.wsBannerWarn}>
                <ShieldAlert size={18} aria-hidden />
                <div>
                  <strong>警告：以下插件仍被活跃工作流引用！</strong>
                  <ul className={styles.wsDialogList}>
                    {pluginConfirm.inUseRows.map(r => (
                      <li key={r.path}>
                        <strong>{r.path}</strong> ← {r.used_by_active.map(u => u.path).join('、')}
                      </li>
                    ))}
                  </ul>
                  <p className="small">继续操作可能导致上述工作流下次打开时缺失节点。</p>
                </div>
              </div>
            )}
            <p className="muted small">
              离线修改会生成新的历史快照并更新 <code>pending.json</code>，下次启动原生 ComfyUI 时生效；随时可在历史快照中回滚。
            </p>
            <div className="actions" style={{ marginTop: 14 }}>
              <button
                type="button"
                className="button secondary"
                onClick={() => {
                  pluginDialogRef.current?.close();
                  setPluginConfirm(null);
                }}
              >
                取消
              </button>
              <button
                type="button"
                className="button danger"
                disabled={busy}
                onClick={() =>
                  void executePluginAction(
                    pluginConfirm.action,
                    pluginConfirm.paths,
                    pluginConfirm.inUseRows.length > 0,
                  )
                }
              >
                {busy ? '正在提交…' : pluginConfirm.inUseRows.length > 0 ? '确认强制执行' : '确认执行'}
              </button>
            </div>
          </>
        )}
      </dialog>

      {/* Rollback Confirmation Modal */}
      <dialog ref={rollbackDialogRef} className="nativeRestartDialog" aria-labelledby="ws-rb-dialog-title">
        <h2 id="ws-rb-dialog-title">回滚历史插件快照</h2>
        {rollbackTarget && (
          <>
            <p className="small">
              将把待生效清单恢复为快照 <code>{rollbackTarget.generation.slice(0, 12)}</code>（{rollbackTarget.plugin_count}{' '}
              个插件，备份于 {formatEpoch(rollbackTarget.backed_up_at)}），下次启动原生 ComfyUI 时生效。
            </p>
            <div className="actions" style={{ marginTop: 14 }}>
              <button
                type="button"
                className="button secondary"
                onClick={() => {
                  rollbackDialogRef.current?.close();
                  setRollbackTarget(null);
                }}
              >
                取消
              </button>
              <button type="button" className="button" disabled={busy} onClick={() => void executeRollback()}>
                {busy ? '正在回滚…' : '确认回滚'}
              </button>
            </div>
          </>
        )}
      </dialog>
    </section>
  );
}
