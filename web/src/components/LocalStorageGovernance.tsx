import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Database,
  Download,
  Eraser,
  Folder,
  HardDrive,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Trash2,
} from 'lucide-react';
import {
  apiMessage,
  createLocalSqliteSnapshot,
  executeLocalStorageCleanup,
  getLocalStorageOverview,
  previewLocalStorageCleanup,
  triggerFileDownload,
  vacuumLocalSqlite,
  type LocalStorageCleanupAction,
  type LocalStorageCleanupPreview,
  type LocalStorageOverview,
  type Project,
} from '../lib/gpu-api';
import { projectOptionLabel } from '../lib/project-label';
import { formatDateTime } from '../lib/ui-helpers';
import { ConfirmDialog } from './Feedback';
import styles from '../gpu.module.css';

export function formatStorageBytes(value?: number | null): string {
  if (value == null || !Number.isFinite(value) || value < 0) return '—';
  if (value === 0) return '0 B';
  if (value >= 1024 ** 4) return `${(value / 1024 ** 4).toFixed(2)} TB`;
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${Math.round(value)} B`;
}

const ACTION_TITLES: Record<LocalStorageCleanupAction, string> = {
  thumb_cache: '清空本地缩略图缓存',
  temp_and_orphans: '清理临时与孤立残留',
  diagnostics: '清理诊断与任务元数据',
  unfavorited_outputs: '按时间清理未收藏作品',
};

const AGE_OPTIONS = [
  { value: 7, label: '7 天前' },
  { value: 14, label: '14 天前' },
  { value: 30, label: '30 天前（推荐）' },
  { value: 60, label: '60 天前' },
  { value: 90, label: '90 天前' },
  { value: 0, label: '全部未收藏' },
] as const;

export function LocalStorageGovernance({
  canEdit,
  projects = [],
  onMutated,
}: {
  canEdit: boolean;
  projects?: Project[];
  onMutated?: () => void;
}) {
  const [overview, setOverview] = useState<LocalStorageOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const [olderThanDays, setOlderThanDays] = useState<number>(30);
  const [cleanupProjectId, setCleanupProjectId] = useState<string>('');
  const [previewBusy, setPreviewBusy] = useState<LocalStorageCleanupAction | null>(null);
  const [pendingPreview, setPendingPreview] = useState<LocalStorageCleanupPreview | null>(null);
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [cleanupError, setCleanupError] = useState('');

  const [confirmVacuum, setConfirmVacuum] = useState(false);
  const [vacuumBusy, setVacuumBusy] = useState(false);
  const [vacuumError, setVacuumError] = useState('');
  const [snapshotBusy, setSnapshotBusy] = useState(false);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError('');
    try {
      const next = await getLocalStorageOverview(signal);
      if (!signal?.aborted) setOverview(next);
    } catch (cause) {
      if (!signal?.aborted) setError(apiMessage(cause));
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const categoryMap = useMemo(() => {
    const map = new Map<string, LocalStorageOverview['categories'][number]>();
    for (const cat of overview?.categories || []) map.set(cat.key, cat);
    return map;
  }, [overview]);

  const nonZeroCategories = useMemo(
    () => (overview?.categories || []).filter((c) => c.bytes > 0),
    [overview],
  );

  async function openCleanupPreview(action: LocalStorageCleanupAction) {
    if (!canEdit || previewBusy) return;
    setPreviewBusy(action);
    setError('');
    setNotice('');
    setCleanupError('');
    try {
      const preview = await previewLocalStorageCleanup({
        action,
        ...(action === 'unfavorited_outputs'
          ? {
              older_than_days: olderThanDays,
              project_id: cleanupProjectId || null,
            }
          : {}),
      });
      if (preview.item_count === 0) {
        const extra = preview.protected_skipped_count > 0
          ? `（另有 ${preview.protected_skipped_count} 件正被活跃任务引用已保护）`
          : '';
        setNotice(`${ACTION_TITLES[action]}：当前没有可清理项目${extra}。`);
        return;
      }
      setPendingPreview(preview);
    } catch (cause) {
      setError(`预演清理失败：${apiMessage(cause)}`);
    } finally {
      setPreviewBusy(null);
    }
  }

  async function confirmCleanupExecution() {
    if (!pendingPreview || cleanupBusy) return;
    setCleanupBusy(true);
    setCleanupError('');
    try {
      const result = await executeLocalStorageCleanup({
        action: pendingPreview.action,
        older_than_days: pendingPreview.older_than_days ?? undefined,
        project_id: pendingPreview.project_id ?? undefined,
        confirmed: true,
      });
      setOverview(result.overview);
      setPendingPreview(null);
      const skippedHint = result.skipped_reasons.length > 0 ? ` ${result.skipped_reasons.join(' ')}` : '';
      setNotice(
        `${ACTION_TITLES[result.action]}完成：已清理 ${result.deleted_count} 项，释放 ${formatStorageBytes(result.freed_bytes)}。${skippedHint}`,
      );
      if (result.action === 'unfavorited_outputs' || result.action === 'diagnostics') {
        onMutated?.();
      }
    } catch (cause) {
      setCleanupError(apiMessage(cause));
    } finally {
      setCleanupBusy(false);
    }
  }

  async function executeVacuum() {
    if (!canEdit || vacuumBusy) return;
    setVacuumBusy(true);
    setVacuumError('');
    setNotice('');
    try {
      const res = await vacuumLocalSqlite();
      setOverview(res.overview);
      setConfirmVacuum(false);
      setNotice(
        `SQLite 空间回收完成：${formatStorageBytes(res.before_bytes)} → ${formatStorageBytes(res.after_bytes)}（释放 ${formatStorageBytes(res.freed_bytes)}）。`,
      );
    } catch (cause) {
      setVacuumError(apiMessage(cause));
    } finally {
      setVacuumBusy(false);
    }
  }

  async function handleDownloadSnapshot() {
    if (!canEdit || snapshotBusy) return;
    setSnapshotBusy(true);
    setError('');
    setNotice('');
    try {
      const ticket = await createLocalSqliteSnapshot();
      triggerFileDownload(ticket.download_url, ticket.filename);
      setNotice(`已开始下载数据库快照「${ticket.filename}」。${ticket.notice}`);
    } catch (cause) {
      setError(`生成数据库快照失败：${apiMessage(cause)}`);
    } finally {
      setSnapshotBusy(false);
    }
  }

  const diskPct = overview?.disk.used_percent ?? 0;
  const thumbCat = categoryMap.get('thumb_cache');
  const tempCat = categoryMap.get('temp_and_orphans');
  const diagCat = categoryMap.get('diagnostics');
  const unfavCat = categoryMap.get('outputs_unfavorited');
  const dbInfo = overview?.database;

  const previewDescription = useMemo(() => {
    if (!pendingPreview) return '';
    const lines = [
      `预计清理 ${pendingPreview.item_count} 项，释放约 ${formatStorageBytes(pendingPreview.reclaimable_bytes)}。`,
    ];
    if (pendingPreview.action === 'unfavorited_outputs') {
      const ageText = pendingPreview.older_than_days === 0
        ? '全部时间范围的未收藏作品'
        : `${pendingPreview.older_than_days} 天前的未收藏作品`;
      const proj = pendingPreview.project_id
        ? projects.find((p) => p.id === pendingPreview.project_id)
        : null;
      lines.unshift(`清理范围：${proj ? `项目「${projectOptionLabel(proj, projects)}」中` : '全部项目中'}${ageText}。`);
    }
    if (pendingPreview.samples.length > 0) {
      const sampleText = pendingPreview.samples
        .slice(0, 5)
        .map((s) => `${s.name}（${formatStorageBytes(s.bytes)}）`)
        .join('、');
      lines.push(`示例文件：${sampleText}${pendingPreview.item_count > 5 ? ' 等' : ''}`);
    }
    for (const w of pendingPreview.warnings) lines.push(w);
    return lines.join('\n');
  }, [pendingPreview, projects]);

  return (
    <section className={styles.card} aria-label="本地存储治理">
      <div className={styles.cardHead}>
        <h2><HardDrive size={17} strokeWidth={1.6} aria-hidden />本地存储治理</h2>
        <div className={styles.cardAside}>
          {overview?.scanned_at && (
            <span className="muted small">扫描于 {formatDateTime(overview.scanned_at * 1000)}</span>
          )}
          <button
            type="button"
            className="button secondary small"
            disabled={loading}
            onClick={() => void load()}
          >
            <RefreshCw size={13} className={loading ? styles.spin : undefined} aria-hidden />
            {loading ? '扫描中…' : '刷新扫描'}
          </button>
        </div>
      </div>

      <p className="muted small">
        管理本机 / VPS 数据目录（作品、参考素材、缩略图缓存、临时残留与 SQLite 数据库）。所有清理均先统计预演并自动保护已收藏作品与活跃任务。
      </p>

      {error && <p className="status error" role="alert">{error}</p>}
      {notice && <p className="status" role="status">{notice}</p>}

      {loading && !overview ? (
        <p className="muted small" role="status">正在扫描本地数据目录…</p>
      ) : overview ? (
        <>
          {/* 1. Host partition disk bar */}
          {overview.disk.total_bytes > 0 && (
            <div className={styles.storageBlock}>
              <div className={styles.storageRowHead}>
                <span className="small">宿主机数据分区</span>
                <span className="small">
                  已用 <strong>{formatStorageBytes(overview.disk.used_bytes)}</strong> / {formatStorageBytes(overview.disk.total_bytes)}
                  {' '}({diskPct.toFixed(1)}%) · 剩余 <strong>{formatStorageBytes(overview.disk.free_bytes)}</strong>
                </span>
              </div>
              <div className={styles.meterTrack} aria-hidden="true">
                <span
                  className={styles.meterFill}
                  data-tone={diskPct >= 95 ? 'danger' : diskPct >= 85 ? 'warn' : 'normal'}
                  style={{ width: `${Math.min(100, diskPct)}%` }}
                />
              </div>
            </div>
          )}

          {/* 2. Studio data_dir category breakdown */}
          <div className={styles.storageBlock}>
            <div className={styles.storageRowHead}>
              <span className="small">Studio 本地数据目录合计</span>
              <span className="small">
                <strong>{formatStorageBytes(overview.studio_total_bytes)}</strong>
                {overview.safe_reclaimable_bytes > 0 && (
                  <> · 可安全回收 <strong>{formatStorageBytes(overview.safe_reclaimable_bytes)}</strong></>
                )}
              </span>
            </div>

            {overview.studio_total_bytes > 0 && nonZeroCategories.length > 0 && (
              <div className={styles.stackedBar} role="img" aria-label="本地存储按类别空间分布条">
                {nonZeroCategories.map((cat, idx) => {
                  const pct = (cat.bytes / overview.studio_total_bytes) * 100;
                  return (
                    <span
                      key={cat.key}
                      className={styles.stackedSeg}
                      data-slot={idx % 6}
                      style={{ width: `${Math.max(pct, 1.5)}%` }}
                      title={`${cat.label}：${formatStorageBytes(cat.bytes)}（${cat.count} 项，${pct.toFixed(1)}%）`}
                    />
                  );
                })}
              </div>
            )}

            <ul className={styles.driveBreakdownList}>
              {overview.categories.map((cat, idx) => {
                const pct = overview.studio_total_bytes > 0
                  ? (cat.bytes / overview.studio_total_bytes) * 100
                  : 0;
                return (
                  <li key={cat.key} className={styles.driveBreakdownRow}>
                    <div className={styles.driveBreakdownName}>
                      <i className={styles.legendDot} data-slot={idx % 6} aria-hidden />
                      <div>
                        <strong>{cat.label}</strong>
                        <small className="muted">{cat.description}</small>
                      </div>
                    </div>
                    <div className={styles.driveBreakdownMetric}>
                      <strong>{formatStorageBytes(cat.bytes)}</strong>
                      <small className="muted">{cat.count} 项 · {pct.toFixed(1)}%</small>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* 3. Per-project disk usage ranking */}
          {overview.projects.length > 0 && (
            <details className="foldable">
              <summary>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <Folder size={14} aria-hidden />
                  按项目占用排行（{overview.projects.length} 个项目）
                </span>
              </summary>
              <div className="foldContent">
                <ul className={styles.driveBreakdownList}>
                  {overview.projects.map((proj) => (
                    <li key={proj.project_id || '__unassigned__'} className={styles.driveBreakdownRow}>
                      <div className={styles.driveBreakdownName}>
                        <div>
                          <strong>{proj.title}</strong>
                          <small className="muted">
                            收藏 {proj.favorite_count}（{formatStorageBytes(proj.favorite_bytes)}）
                            · 未收藏 {proj.unfavorited_count}（{formatStorageBytes(proj.unfavorited_bytes)}）
                            {proj.reference_count > 0
                              ? ` · 参考素材 ${proj.reference_count}（${formatStorageBytes(proj.reference_bytes)}）`
                              : ''}
                          </small>
                        </div>
                      </div>
                      <div className={styles.driveBreakdownMetric}>
                        <strong>{formatStorageBytes(proj.total_bytes)}</strong>
                        <small className="muted">{proj.total_count} 件</small>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </details>
          )}

          {/* 4. Safe cleanup actions */}
          <div className={styles.inlinePanel}>
            <div className={styles.storageRowHead}>
              <strong className="small" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <Eraser size={14} aria-hidden />安全空间清理
              </strong>
              {overview.protected_asset_count > 0 && (
                <span className="pill" title="正被活跃任务引用的资产会自动跳过">
                  <ShieldCheck size={12} aria-hidden />已保护 {overview.protected_asset_count} 项活跃引用
                </span>
              )}
            </div>

            {!canEdit ? (
              <p className="muted small">仅管理员可执行本地存储清理与数据库维护。</p>
            ) : (
              <>
                <div className={styles.localStorageQuickActions}>
                  <button
                    type="button"
                    className="button secondary small"
                    disabled={Boolean(previewBusy) || !thumbCat?.count}
                    onClick={() => void openCleanupPreview('thumb_cache')}
                  >
                    <Sparkles size={13} aria-hidden />
                    {previewBusy === 'thumb_cache'
                      ? '统计中…'
                      : `清空缩略图缓存${thumbCat?.bytes ? ` (${formatStorageBytes(thumbCat.bytes)})` : ''}`}
                  </button>
                  <button
                    type="button"
                    className="button secondary small"
                    disabled={Boolean(previewBusy) || !tempCat?.count}
                    onClick={() => void openCleanupPreview('temp_and_orphans')}
                  >
                    <Eraser size={13} aria-hidden />
                    {previewBusy === 'temp_and_orphans'
                      ? '统计中…'
                      : `清理临时与孤立残留${tempCat?.bytes ? ` (${formatStorageBytes(tempCat.bytes)})` : ''}`}
                  </button>
                  <button
                    type="button"
                    className="button secondary small"
                    disabled={Boolean(previewBusy) || !diagCat?.count}
                    onClick={() => void openCleanupPreview('diagnostics')}
                  >
                    <Trash2 size={13} aria-hidden />
                    {previewBusy === 'diagnostics'
                      ? '统计中…'
                      : `清理诊断与元数据${diagCat?.count ? ` (${diagCat.count} 项)` : ''}`}
                  </button>
                </div>

                <div className={styles.localStorageAgeRow}>
                  <div className={styles.localStorageAgeControls}>
                    <label className="small muted" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <span>清理未收藏作品</span>
                      <select
                        aria-label="未收藏作品保留天数"
                        className={styles.miniSelect}
                        value={olderThanDays}
                        onChange={(e) => setOlderThanDays(Number(e.target.value))}
                      >
                        {AGE_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="small muted" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <Folder size={13} aria-hidden />
                      <select
                        aria-label="按项目限定清理范围"
                        className={styles.miniSelect}
                        value={cleanupProjectId}
                        onChange={(e) => setCleanupProjectId(e.target.value)}
                      >
                        <option value="">全部项目</option>
                        {projects.map((p) => (
                          <option key={p.id} value={p.id}>{projectOptionLabel(p, projects)}</option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <button
                    type="button"
                    className="button danger small"
                    disabled={Boolean(previewBusy) || !unfavCat?.count}
                    onClick={() => void openCleanupPreview('unfavorited_outputs')}
                  >
                    <Trash2 size={13} aria-hidden />
                    {previewBusy === 'unfavorited_outputs' ? '统计中…' : '预演清理范围'}
                  </button>
                </div>
                <p className="muted small">
                  仅清理未收藏的 Studio 生成作品，已收藏作品、上传/导入参考素材与运行中任务引用的文件均受保护。
                </p>
              </>
            )}
          </div>

          {/* 5. SQLite maintenance & snapshot */}
          {dbInfo && (
            <div className={styles.inlinePanel}>
              <div className={styles.storageRowHead}>
                <strong className="small" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <Database size={14} aria-hidden />
                  数据库维护与快照（{dbInfo.dialect.toUpperCase()}）
                </strong>
                {dbInfo.supported && (
                  <span className="muted small">
                    总计 {formatStorageBytes(dbInfo.total_bytes)}
                    {dbInfo.wal_bytes > 0 ? `（含 WAL ${formatStorageBytes(dbInfo.wal_bytes)}）` : ''}
                    {dbInfo.reclaimable_bytes > 0 ? ` · 可回收约 ${formatStorageBytes(dbInfo.reclaimable_bytes)}` : ''}
                  </span>
                )}
              </div>

              {!dbInfo.supported ? (
                <p className="muted small">{dbInfo.message || '当前数据库不支持在线整理与网页快照下载。'}</p>
              ) : !canEdit ? (
                <p className="muted small">仅管理员可执行 SQLite 空间回收与快照下载。</p>
              ) : (
                <>
                  <div className={styles.localStorageQuickActions}>
                    <button
                      type="button"
                      className="button secondary small"
                      disabled={vacuumBusy}
                      onClick={() => { setVacuumError(''); setConfirmVacuum(true); }}
                    >
                      <Database size={13} aria-hidden />
                      {vacuumBusy ? '回收中…' : '整理并回收空间 (VACUUM)'}
                    </button>
                    <button
                      type="button"
                      className="button secondary small"
                      disabled={snapshotBusy}
                      onClick={() => void handleDownloadSnapshot()}
                    >
                      <Download size={13} aria-hidden />
                      {snapshotBusy ? '生成快照中…' : '下载一致性数据库快照 (.db)'}
                    </button>
                  </div>
                  <p className="muted small">
                    网页快照使用 SQLite 在线一致性备份并经完整性校验，仅含数据库记录（凭据已加密），不含 `.env`、`ENCRYPTION_KEY` 及本地媒体文件；迁移前请另行妥善备份 `.env`。
                  </p>
                </>
              )}
            </div>
          )}
        </>
      ) : null}

      <ConfirmDialog
        open={Boolean(pendingPreview)}
        title={pendingPreview ? `确认${ACTION_TITLES[pendingPreview.action]}？` : ''}
        description={previewDescription}
        confirmLabel={pendingPreview?.action === 'unfavorited_outputs' ? '确认永久删除' : '确认清理'}
        danger={pendingPreview?.action === 'unfavorited_outputs'}
        busy={cleanupBusy}
        error={cleanupError}
        onCancel={() => { if (!cleanupBusy) { setPendingPreview(null); setCleanupError(''); } }}
        onConfirm={() => void confirmCleanupExecution()}
      />

      <ConfirmDialog
        open={confirmVacuum}
        title="执行 SQLite 空间回收（VACUUM）？"
        description={
          `当前数据库占用 ${formatStorageBytes(dbInfo?.total_bytes)}（其中空闲页约 ${formatStorageBytes(dbInfo?.reclaimable_bytes)}）。`
          + '执行期间会短暂加锁整理数据库文件并截断 WAL 日志，不影响已有业务数据。'
        }
        confirmLabel="开始整理"
        busy={vacuumBusy}
        error={vacuumError}
        onCancel={() => { if (!vacuumBusy) { setConfirmVacuum(false); setVacuumError(''); } }}
        onConfirm={() => void executeVacuum()}
      />
    </section>
  );
}
