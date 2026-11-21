import { useMemo, useState } from 'react';
import {
  Zap, Search, RefreshCw, Image as ImageIcon, Film, Loader, ChevronRight, ChevronDown,
  Square, RotateCcw, Trash2, Layers, Eraser,
} from 'lucide-react';
import { type GpuJob } from '../lib/gpu-api';
import styles from '../gpu.module.css';
import { formatDateTime, jobSize, progressPercent, readableState, terminalStates } from '../lib/ui-helpers';
import { jobStageLabel, jobStateLabel } from '../lib/job-stage-rows';
import { canRetainJobInstance, gpuLabel } from '../lib/gpu-label';
import { groupJobsByBatch, partitionJobEntries, type DateBucketKey, type JobBatchEntry, type JobDisplayEntry } from '../lib/history-grouping';
import { ConfirmDialog } from '../components/Feedback';

// 与后端 scheduler.ACTIVE_STATES 对应：前四个算排队，后面算运行。
const WAITING = ['queued', 'validating', 'waiting_for_capacity', 'allocating'];
const RUNNING = ['preparing', 'loading', 'running', 'uploading', 'cleaning_up', 'cleanup_pending'];
const DONE = ['completed', 'succeeded'];
const FAILED = ['failed', 'failed_final', 'cancelled', 'cancelled_local', 'blocked'];

type Filter = 'all' | 'running' | 'completed' | 'failed' | string;
export type JobClearRequest = { scope?: 'all_terminal' | 'failed_cancelled' | 'mock'; jobIds?: string[] };

export function TasksPanel({
  jobs, error, refreshing, clearingHistory = false, onRefresh, onCancel, onRetry, onDetail, onRetain, onInstance, onClearHistory, selectedId,
}: {
  jobs: GpuJob[];
  error: string;
  refreshing: boolean;
  clearingHistory?: boolean;
  onRefresh: () => void;
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onDetail: (job: GpuJob) => void;
  selectedId?: string;
  onRetain?: (job: GpuJob) => void;
  onInstance?: (job: GpuJob) => void;
  onClearHistory?: (request: JobClearRequest) => Promise<void> | void;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(30);
  const [stackBatches, setStackBatches] = useState(true);
  const [expandedBatches, setExpandedBatches] = useState<Set<string>>(() => new Set());
  const [collapsedBuckets, setCollapsedBuckets] = useState<Record<DateBucketKey, boolean>>({
    today: false,
    yesterday: false,
    older: true,
  });
  const [cleanupMenuOpen, setCleanupMenuOpen] = useState(false);
  const [pendingClear, setPendingClear] = useState<{ title: string; description: string; request: JobClearRequest } | null>(null);

  const stateOf = (job: GpuJob) => String(job.state || 'queued').toLowerCase();
  const isActive = (job: GpuJob) => !terminalStates.has(stateOf(job));

  const uniqueJobs = useMemo(() => {
    const seen = new Set<string>();
    return jobs.filter((job, index) => {
      const key = String(job.id || job.job_id || `job-${index}`);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [jobs]);

  const counts = useMemo(() => ({
    running: uniqueJobs.filter(isActive).length,
    completed: uniqueJobs.filter((j) => DONE.includes(stateOf(j))).length,
    failed: uniqueJobs.filter((j) => FAILED.includes(stateOf(j))).length,
    terminal: uniqueJobs.filter((j) => terminalStates.has(stateOf(j))).length,
    mockTerminal: uniqueJobs.filter((j) => j.provider === 'mock' && terminalStates.has(stateOf(j))).length,
  }), [uniqueJobs]);

  const filtered = useMemo(() => uniqueJobs.filter((job) => {
    const s = stateOf(job);
    if (filter === 'running' && !isActive(job)) return false;
    if (filter === 'completed' && !DONE.includes(s)) return false;
    if (filter === 'failed' && !FAILED.includes(s)) return false;
    if (!['all', 'running', 'completed', 'failed'].includes(filter) && job.state !== filter) return false;
    return `${job.workflow_key || ''} ${job.id} ${job.batch_id || ''}`.toLowerCase().includes(query.toLowerCase());
  }), [uniqueJobs, filter, query]);

  const entries = useMemo<JobDisplayEntry[]>(() => {
    if (!stackBatches) {
      return filtered.map((job, index) => ({
        kind: 'single' as const,
        key: String(job.id || job.job_id || `job-${index}`),
        job,
        latestCreatedAt: job.created_at || '',
        allTerminal: terminalStates.has(stateOf(job)),
      }));
    }
    return groupJobsByBatch(filtered);
  }, [filtered, stackBatches]);

  const { active: activeEntries, buckets } = partitionJobEntries(entries, Date.now(), limit);
  const hasMultiBatches = useMemo(() => {
    const seen = new Map<string, number>();
    for (const job of filtered) {
      if (!job.batch_id) continue;
      const next = (seen.get(job.batch_id) || 0) + 1;
      if (next >= 2) return true;
      seen.set(job.batch_id, next);
    }
    return false;
  }, [filtered]);

  const singleBucketOnly = activeEntries.length === 0 && buckets.length === 1;

  function toggleBatch(batchId: string) {
    setExpandedBatches((curr) => {
      const next = new Set(curr);
      if (next.has(batchId)) next.delete(batchId);
      else next.add(batchId);
      return next;
    });
  }

  function toggleBucket(key: DateBucketKey) {
    setCollapsedBuckets((curr) => ({ ...curr, [key]: !curr[key] }));
  }

  async function confirmClear() {
    if (!pendingClear || !onClearHistory) return;
    const req = pendingClear.request;
    setPendingClear(null);
    setCleanupMenuOpen(false);
    await onClearHistory(req);
  }

  const chip = (value: Filter, label: string) => (
    <button type="button" className={styles.chip} data-active={filter === value} onClick={() => setFilter(value)}>{label}</button>
  );

  function renderJobCard(job: GpuJob, index: number, nested = false) {
    const jobKey = String(job.id || job.job_id || `job-${index}`);
    const s = stateOf(job);
    const pct = progressPercent(job.progress);
    const done = DONE.includes(s);
    const running = RUNNING.includes(s);
    const failed = FAILED.includes(s);
    const waiting = WAITING.includes(s);
    const video = /i2v|video|ref2v|t2v/i.test(job.workflow_key || '');
    const thumb = done && job.output_asset_ids?.[0] && !video ? `/api/assets/${job.output_asset_ids[0]}/thumbnail` : '';
    const size = jobSize(job);
    const err = job.error_message || job.error;
    const instanceId = String(job.instance_id || '');
    const accountSnapshot = String(job.account_alias_snapshot || job.account_id || '');
    const machineInfo = [
      accountSnapshot ? `账号 ${accountSnapshot}` : '',
      String(job.instance_label || instanceId.slice(0, 8)),
      gpuLabel(job),
      instanceId ? job.instance_deleted_reason === 'provider_runtime_missing' ? '已回收' : readableState(job.instance_state || '未知')
        : terminalStates.has(s) ? '未分配机器' : '等待分配机器',
    ].filter(Boolean).join(' · ');
    const liveInstance = canRetainJobInstance(job);
    const retained = liveInstance && Boolean(job.retained_until || job.instance_state === 'retained_idle');
    const isTerminal = terminalStates.has(s);

    return (
      <article className={`${styles.jobCard} ${nested ? styles.nestedJobCard : ''}`} key={jobKey} data-inspected={selectedId === job.id}>
        <div className={styles.jobTop}>
          <div className={styles.jobThumb}>
            {thumb ? <img src={thumb} alt="" loading="lazy" /> : video ? <Film size={22} strokeWidth={1.4} aria-hidden /> : <ImageIcon size={22} strokeWidth={1.4} aria-hidden />}
          </div>
          <div className={styles.jobInfo}>
            <div className={styles.jobTitleRow}>
              <strong title={job.workflow_key || job.id}>{job.workflow_key || `任务 ${job.id.slice(0, 8)}`}</strong>
              <span className={`pill ${done ? 'solid' : failed ? 'danger' : ''}`}>{jobStateLabel(job.state)}</span>
            </div>
            <p className="muted small">
              {[video ? '视频' : '图片', size, job.provider === 'mock' ? '模拟' : job.provider === 'colab' ? 'Colab' : ''].filter(Boolean).join(' · ')}
            </p>
            <p className="muted small">创建于 {formatDateTime(job.created_at)}</p>
            {instanceId ? <button type="button" className={styles.linkButton} onClick={() => onInstance?.(job)}>{machineInfo}</button>
              : <p className="muted small">{machineInfo}</p>}
          </div>
        </div>

        {(running || waiting) && (
          <div className={styles.progressBlock}>
            <div className={styles.progressText}>
              <span>{running ? (job.stage ? jobStageLabel(job.stage) : pct !== null ? `生成中 ${pct}%` : '生成中…') : '排队等待中'}</span>
            </div>
            <div className={styles.progress} data-indeterminate={pct === null}><span style={pct !== null ? { width: `${pct}%` } : undefined} /></div>
          </div>
        )}

        {err && <p className="status error" role="alert">{err}</p>}

        <div className={styles.cardActions}>
          <button className="button small" style={{ flex: 1 }} type="button" aria-pressed={selectedId === job.id} onClick={() => onDetail(job)}>查看详情</button>
          {liveInstance && !retained && onRetain && <button className="button secondary small" type="button" onClick={() => onRetain(job)}>保留这台机器</button>}
          {retained && <span className="pill">已保留{job.retained_until ? `至 ${formatDateTime(String(job.retained_until))}` : ''}</span>}
          {!isTerminal ? (
            <button className="button secondary small" style={{ flex: 1 }} type="button" onClick={() => onCancel(job.id)}>
              <Square size={13} strokeWidth={1.8} aria-hidden />取消
            </button>
          ) : (
            <>
              <button className="button secondary small" style={{ flex: 1 }} type="button" onClick={() => onRetry(job.id)}>
                <RotateCcw size={13} strokeWidth={1.8} aria-hidden />{failed ? '重试' : '再来一次'}
              </button>
              {onClearHistory && (
                <button
                  className={styles.iconGhost}
                  type="button"
                  disabled={clearingHistory}
                  title="隐藏此任务历史"
                  aria-label="隐藏此任务历史"
                  onClick={() => void onClearHistory({ jobIds: [job.id] })}
                >
                  <Trash2 size={15} strokeWidth={1.6} aria-hidden />
                </button>
              )}
            </>
          )}
        </div>
      </article>
    );
  }

  function renderBatchCard(entry: JobBatchEntry) {
    const isOpen = expandedBatches.has(entry.batchId) || entry.jobs.some((j) => j.id === selectedId);
    const total = entry.jobs.length;
    const video = /i2v|video|ref2v|t2v/i.test(entry.workflowKey);
    const thumbs = entry.jobs
      .filter((j) => DONE.includes(stateOf(j)) && j.output_asset_ids?.[0] && !video)
      .slice(0, 4)
      .map((j) => `/api/assets/${j.output_asset_ids![0]}/thumbnail`);
    const summaryParts = [
      entry.activeCount > 0 ? `${entry.activeCount} 进行中` : '',
      entry.doneCount > 0 ? `${entry.doneCount} 完成` : '',
      entry.failedCount > 0 ? `${entry.failedCount} 失败/取消` : '',
    ].filter(Boolean).join(' · ');

    const activeJobs = entry.jobs.filter((j) => !terminalStates.has(stateOf(j)));
    const failedJobs = entry.jobs.filter((j) => FAILED.includes(stateOf(j)));
    const terminalJobIds = entry.jobs.filter((j) => terminalStates.has(stateOf(j))).map((j) => j.id);

    return (
      <article className={styles.batchCard} key={entry.key} data-open={isOpen}>
        <div className={styles.batchHead}>
          <button
            type="button"
            className={styles.batchToggle}
            aria-expanded={isOpen}
            onClick={() => toggleBatch(entry.batchId)}
          >
            <span className={styles.batchIcon}><Layers size={18} strokeWidth={1.6} aria-hidden /></span>
            <span className={styles.batchMeta}>
              <span className={styles.jobTitleRow}>
                <strong>{entry.workflowKey}</strong>
                <span className={`pill ${entry.activeCount > 0 ? '' : entry.failedCount === total ? 'danger' : 'solid'}`}>
                  批次 ×{total}
                </span>
              </span>
              <small className="muted">
                {summaryParts} · {entry.provider === 'mock' ? '模拟' : entry.provider === 'colab' ? 'Colab' : '任务'} · {formatDateTime(entry.latestCreatedAt)}
              </small>
            </span>
            <ChevronDown size={16} strokeWidth={1.6} aria-hidden className={styles.chev} />
          </button>

          {thumbs.length > 0 && !isOpen && (
            <div className={styles.batchThumbs} aria-label="本批次缩略图预览">
              {thumbs.map((src, idx) => (
                <img key={idx} src={src} alt="" loading="lazy" />
              ))}
              {entry.doneCount > thumbs.length && <span className={styles.batchMore}>+{entry.doneCount - thumbs.length}</span>}
            </div>
          )}

          <div className={styles.cardActions}>
            <button type="button" className="button secondary small" onClick={() => toggleBatch(entry.batchId)}>
              {isOpen ? '收起子任务' : `展开 ${total} 个子任务`}
            </button>
            {activeJobs.length > 0 && (
              <button
                type="button"
                className="button secondary small"
                onClick={() => { activeJobs.forEach((j) => onCancel(j.id)); }}
              >
                <Square size={13} strokeWidth={1.8} aria-hidden />取消本批 ({activeJobs.length})
              </button>
            )}
            {failedJobs.length > 0 && (
              <button
                type="button"
                className="button secondary small"
                onClick={() => { failedJobs.forEach((j) => onRetry(j.id)); }}
              >
                <RotateCcw size={13} strokeWidth={1.8} aria-hidden />重试失败 ({failedJobs.length})
              </button>
            )}
            {entry.allTerminal && onClearHistory && (
              <button
                type="button"
                className="button ghost small"
                disabled={clearingHistory}
                onClick={() => setPendingClear({
                  title: '清理本批次记录？',
                  description: `将从任务列表隐藏这 ${terminalJobIds.length} 条已结束记录；已生成的作品和 CU 账单不受影响。`,
                  request: { jobIds: terminalJobIds },
                })}
              >
                <Trash2 size={13} strokeWidth={1.6} aria-hidden />清理此批
              </button>
            )}
          </div>
        </div>

        {isOpen && (
          <div className={styles.batchChildren}>
            {entry.jobs.map((job, idx) => renderJobCard(job, idx, true))}
          </div>
        )}
      </article>
    );
  }

  function renderEntry(entry: JobDisplayEntry, index: number) {
    return entry.kind === 'batch' ? renderBatchCard(entry) : renderJobCard(entry.job, index);
  }

  return (
    <div className={styles.stack}>
      <div className={styles.pageTitle}>
        <h2><Zap size={20} strokeWidth={1.6} aria-hidden />任务队列</h2>
        <div className={styles.titleActions}>
          {onClearHistory && counts.terminal > 0 && (
            <button
              className="button secondary small"
              type="button"
              disabled={clearingHistory}
              aria-expanded={cleanupMenuOpen}
              onClick={() => setCleanupMenuOpen((open) => !open)}
            >
              <Eraser size={14} strokeWidth={1.7} aria-hidden />清理历史
            </button>
          )}
          <button className="button secondary small" type="button" onClick={onRefresh}>
            <RefreshCw size={14} strokeWidth={1.8} aria-hidden className={refreshing ? styles.spin : undefined} />{refreshing ? '刷新中' : '刷新'}
          </button>
        </div>
      </div>

      {cleanupMenuOpen && onClearHistory && counts.terminal > 0 && (
        <div className={styles.cleanupBar} role="region" aria-label="历史清理选项">
          <span className="muted small">仅隐藏任务列表记录，作品库和 CU 账单会保留：</span>
          <div className="actions">
            {counts.failed > 0 && (
              <button
                type="button"
                className="button secondary small"
                disabled={clearingHistory}
                onClick={() => setPendingClear({
                  title: '清理失败和已取消任务？',
                  description: `将隐藏当前项目中 ${counts.failed} 条失败/已取消的任务记录。`,
                  request: { scope: 'failed_cancelled' },
                })}
              >
                清理失败/取消 ({counts.failed})
              </button>
            )}
            {counts.mockTerminal > 0 && (
              <button
                type="button"
                className="button secondary small"
                disabled={clearingHistory}
                onClick={() => setPendingClear({
                  title: '清理离线模拟任务？',
                  description: `将隐藏当前项目中 ${counts.mockTerminal} 条已结束的模拟任务记录。`,
                  request: { scope: 'mock' },
                })}
              >
                清理模拟任务 ({counts.mockTerminal})
              </button>
            )}
            <button
              type="button"
              className="button danger small"
              disabled={clearingHistory}
              onClick={() => setPendingClear({
                title: '清理全部已结束任务？',
                description: `将隐藏当前项目中 ${counts.terminal} 条已结束任务（进行中的任务不受影响），已生成的作品和账单会保留。`,
                request: { scope: 'all_terminal' },
              })}
            >
              清理全部已结束 ({counts.terminal})
            </button>
          </div>
        </div>
      )}

      {error && <p className="status error" role="alert">{error}</p>}

      <label className={styles.searchBox}>
        <Search size={17} strokeWidth={1.6} aria-hidden />
        <input aria-label="搜索任务" placeholder="搜索任务（名称、编号、批次）" value={query} onChange={(e) => { setQuery(e.target.value); setLimit(30); }} />
      </label>

      <div className={styles.filterBar}>
        <div className={styles.chipRow}>
          {chip('all', `全部 ${uniqueJobs.length}`)}
          {chip('running', `进行中 ${counts.running}`)}
          {chip('completed', `已完成 ${counts.completed}`)}
          {counts.failed > 0 && chip('failed', `失败/取消 ${counts.failed}`)}
        </div>
        <div className={styles.filterRight}>
          {hasMultiBatches && (
            <button
              type="button"
              className={styles.chip}
              data-active={stackBatches}
              onClick={() => setStackBatches((v) => !v)}
              title="把同批次提交的任务折叠为一组"
            >
              <Layers size={13} strokeWidth={1.7} aria-hidden />{stackBatches ? '批次已堆叠' : '按批次堆叠'}
            </button>
          )}
          <select aria-label="按状态筛选" className={styles.miniSelect} value={['all', 'running', 'completed', 'failed'].includes(filter) ? '' : filter} onChange={(e) => setFilter(e.target.value || 'all')}>
            <option value="">更多状态</option>
            {[...new Set(uniqueJobs.map((job) => job.state || 'queued'))].map((state) => <option value={state} key={state}>{jobStateLabel(state)}</option>)}
          </select>
        </div>
      </div>

      {counts.running > 0 && filter !== 'running' && (
        <button type="button" className={styles.banner} onClick={() => setFilter('running')}>
          <Loader size={16} strokeWidth={1.6} aria-hidden className={styles.spin} />
          <span><strong>{counts.running} 个任务</strong>正在排队或运行</span>
          <ChevronRight size={16} strokeWidth={1.6} aria-hidden />
        </button>
      )}

      {entries.length ? (
        <div className={styles.list}>
          {activeEntries.length > 0 && (
            <div className={styles.list}>
              {activeEntries.map((entry, idx) => renderEntry(entry, idx))}
            </div>
          )}

          {buckets.map((bucket) => {
            const forceExpanded = singleBucketOnly && bucket.entries.length <= 5;
            const collapsed = !forceExpanded && collapsedBuckets[bucket.key];
            const showHeader = buckets.length > 1 || activeEntries.length > 0 || bucket.entries.length > 3;
            return (
              <section key={bucket.key} className={styles.bucketSection}>
                {showHeader && (
                  <button
                    type="button"
                    className={styles.bucketHead}
                    aria-expanded={!collapsed}
                    onClick={() => toggleBucket(bucket.key)}
                  >
                    <span>{bucket.label}</span>
                    <span className="pill">{bucket.entries.length}</span>
                    <ChevronDown size={15} strokeWidth={1.6} aria-hidden className={collapsed ? styles.chevCollapsed : styles.chev} />
                  </button>
                )}
                {!collapsed && (
                  <div className={styles.list}>
                    {bucket.entries.map((entry, idx) => renderEntry(entry, idx))}
                  </div>
                )}
              </section>
            );
          })}
        </div>
      ) : (
        <div className="empty">
          <strong>还没有任务</strong>
          <span>在「创作」里点「立即生成」，任务会出现在这里。</span>
        </div>
      )}

      {entries.length - activeEntries.length > limit && (
        <button className="button secondary block" type="button" onClick={() => setLimit(limit + 30)}>加载更多任务</button>
      )}

      <ConfirmDialog
        open={Boolean(pendingClear)}
        title={pendingClear?.title || ''}
        description={pendingClear?.description}
        confirmLabel="确认清理"
        danger
        busy={clearingHistory}
        onCancel={() => setPendingClear(null)}
        onConfirm={() => void confirmClear()}
      />
    </div>
  );
}
