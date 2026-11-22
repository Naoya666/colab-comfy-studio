import { useMemo, useState } from 'react';
import { HardDrive, RefreshCw } from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import {
  bytesLabel,
  getDriveUsage,
  type Catalog,
  type DriveUsageReport,
} from '../lib/model-library';
import { formatDateTime } from '../lib/ui-helpers';
import styles from '../gpu.module.css';

function formatBytes(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '—';
  if (value === 0) return '0 B';
  if (value >= 1024 ** 4) return `${(value / 1024 ** 4).toFixed(2)} TB`;
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

export function ModelStorageBreakdown({
  catalog,
  selectedCategory,
  onSelectCategory,
}: {
  catalog: Catalog | null;
  selectedCategory: string;
  onSelectCategory: (category: string) => void;
}) {
  const [driveReport, setDriveReport] = useState<DriveUsageReport | null>(null);
  const [loadingDrive, setLoadingDrive] = useState(false);
  const [driveError, setDriveError] = useState('');

  const typeSegments = useMemo(() => {
    const byType = catalog?.statistics?.by_type || {};
    const entries = Object.entries(byType)
      .filter(([, stat]) => stat && (stat.bytes > 0 || stat.count > 0))
      .map(([type, stat]) => ({ type, count: stat.count, bytes: stat.bytes }))
      .sort((a, b) => b.bytes - a.bytes || b.count - a.count);
    const totalBytes = entries.reduce((sum, item) => sum + item.bytes, 0);
    return { entries, totalBytes };
  }, [catalog]);

  const driveCategories = useMemo(
    () => (driveReport?.categories || []).filter((item) => item.bytes > 0 || item.count > 0),
    [driveReport],
  );

  async function scanDrive(refresh: boolean) {
    if (loadingDrive) return;
    setLoadingDrive(true);
    setDriveError('');
    try {
      const report = await getDriveUsage(refresh);
      setDriveReport(report);
    } catch (cause) {
      setDriveError(apiMessage(cause));
    } finally {
      setLoadingDrive(false);
    }
  }

  const quotaLimit = driveReport?.quota?.limit ?? null;
  const quotaUsage = driveReport?.quota?.usage ?? null;
  const quotaTrash = driveReport?.quota?.usage_in_drive_trash ?? null;
  const quotaPct = quotaLimit && quotaUsage != null ? Math.min(100, (quotaUsage / quotaLimit) * 100) : null;

  return (
    <section className={styles.card} aria-label="模型与 Drive 空间拆解">
      <div className={styles.cardHead}>
        <h2><HardDrive size={16} strokeWidth={1.6} aria-hidden />空间占用拆解</h2>
        <div className={styles.cardAside}>
          <button
            type="button"
            className="button secondary small"
            disabled={loadingDrive}
            onClick={() => void scanDrive(Boolean(driveReport))}
          >
            <RefreshCw size={13} className={loadingDrive ? styles.spin : undefined} aria-hidden />
            {loadingDrive ? '扫描中…' : driveReport ? '刷新 Drive 拆解' : '扫描 Drive 全局占用'}
          </button>
        </div>
      </div>

      {/* Layer 1: Instant local catalog breakdown by model category */}
      <div className={styles.storageBlock}>
        <div className={styles.storageRowHead}>
          <span className="small">模型库按类别分布（点选可联动筛选）</span>
          <strong className="small">
            {catalog?.statistics.count || 0} 个 · {bytesLabel(catalog?.statistics.bytes)}
          </strong>
        </div>

        {typeSegments.totalBytes > 0 ? (
          <>
            <div className={styles.stackedBar} role="img" aria-label="模型库按类型空间分布条">
              {typeSegments.entries.map((seg, idx) => {
                const pct = (seg.bytes / typeSegments.totalBytes) * 100;
                if (pct <= 0) return null;
                return (
                  <button
                    key={seg.type}
                    type="button"
                    className={styles.stackedSeg}
                    data-slot={idx % 6}
                    data-active={selectedCategory === seg.type}
                    style={{ width: `${Math.max(pct, 1.5)}%` }}
                    title={`${seg.type}：${formatBytes(seg.bytes)}（${seg.count} 个，${pct.toFixed(1)}%）`}
                    aria-label={`筛选 ${seg.type}，占用 ${formatBytes(seg.bytes)}`}
                    onClick={() => onSelectCategory(selectedCategory === seg.type ? '' : seg.type)}
                  />
                );
              })}
            </div>
            <div className={styles.legendGrid}>
              {typeSegments.entries.map((seg, idx) => {
                const pct = typeSegments.totalBytes > 0 ? (seg.bytes / typeSegments.totalBytes) * 100 : 0;
                const active = selectedCategory === seg.type;
                return (
                  <button
                    key={seg.type}
                    type="button"
                    className={styles.legendChip}
                    data-active={active}
                    onClick={() => onSelectCategory(active ? '' : seg.type)}
                  >
                    <i className={styles.legendDot} data-slot={idx % 6} aria-hidden />
                    <span>{seg.type}</span>
                    <small>{formatBytes(seg.bytes)} · {pct.toFixed(0)}%</small>
                  </button>
                );
              })}
            </div>
          </>
        ) : (
          <p className="muted small">模型库暂无已统计大小的权重文件。</p>
        )}
      </div>

      {/* Layer 2: On-demand read-only Drive global breakdown */}
      {driveError && <p className="status error" role="alert">{driveError}</p>}

      {driveReport && (
        <div className={styles.inlinePanel}>
          <div className={styles.storageRowHead}>
            <strong className="small">
              Google Drive 全局拆解（只读）
              {driveReport.account_alias ? ` · ${driveReport.account_alias}` : ''}
            </strong>
            {driveReport.cached_at && (
              <span className="muted small">快照于 {formatDateTime(driveReport.cached_at * 1000)}</span>
            )}
          </div>

          {!driveReport.connected ? (
            <p className="muted small">{driveReport.message || '未能读取 Drive 空间信息。'}</p>
          ) : (
            <>
              {quotaLimit != null && quotaUsage != null && (
                <div className={styles.storageBlock}>
                  <div className={styles.storageRowHead}>
                    <span className="small">账号 Drive 总配额</span>
                    <span className="small">
                      <strong>{formatBytes(quotaUsage)}</strong> / {formatBytes(quotaLimit)}
                      {quotaPct !== null ? ` (${quotaPct.toFixed(1)}%)` : ''}
                      {quotaTrash ? ` · 回收站 ${formatBytes(quotaTrash)}` : ''}
                    </span>
                  </div>
                  <div className={styles.meterTrack} aria-hidden="true">
                    <span
                      className={styles.meterFill}
                      data-tone={quotaPct != null && quotaPct >= 95 ? 'danger' : quotaPct != null && quotaPct >= 85 ? 'warn' : 'normal'}
                      style={{ width: `${quotaPct ?? 0}%` }}
                    />
                  </div>
                </div>
              )}

              <div className={styles.storageBlock}>
                <div className={styles.storageRowHead}>
                  <span className="small">AI-Studio 目录合计</span>
                  <strong className="small">
                    {formatBytes(driveReport.studio_total_bytes)} · {driveReport.studio_file_count} 个文件
                  </strong>
                </div>

                {driveReport.studio_total_bytes > 0 && (
                  <div className={styles.stackedBar} role="img" aria-label="AI-Studio 目录空间分布条">
                    {driveCategories.map((cat, idx) => {
                      const pct = (cat.bytes / driveReport.studio_total_bytes) * 100;
                      if (pct <= 0) return null;
                      return (
                        <span
                          key={cat.key}
                          className={styles.stackedSeg}
                          data-slot={idx % 6}
                          style={{ width: `${Math.max(pct, 1.5)}%` }}
                          title={`${cat.label}：${formatBytes(cat.bytes)}（${cat.count} 个，${pct.toFixed(1)}%）`}
                        />
                      );
                    })}
                  </div>
                )}

                <ul className={styles.driveBreakdownList}>
                  {driveReport.categories.map((cat, idx) => {
                    const pct = driveReport.studio_total_bytes > 0
                      ? (cat.bytes / driveReport.studio_total_bytes) * 100
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
                          <strong>{formatBytes(cat.bytes)}</strong>
                          <small className="muted">{cat.count} 文件 · {pct.toFixed(1)}%</small>
                        </div>
                      </li>
                    );
                  })}
                </ul>
                {driveReport.truncated && (
                  <p className="muted small">文件数超过单次扫描上限，以上为部分统计结果。</p>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
