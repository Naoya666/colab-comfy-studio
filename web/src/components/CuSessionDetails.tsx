import { useEffect, useMemo, useRef, useState } from 'react';
import { Calendar, ChevronDown, ChevronRight, RotateCcw } from 'lucide-react';
import {
  apiMessage,
  getCuDaily,
  getCuSessions,
  type CuDailyEntry,
  type CuDailyPage,
  type CuSession,
  type CuSessionKind,
  type GpuUsage,
} from '../lib/gpu-api';
import { cuNumber, formatDateTime } from '../lib/ui-helpers';
import { CuDailyChart } from './CuDailyChart';
import { Card, Segmented } from './ui';
import styles from '../gpu.module.css';

const labels: Record<CuSessionKind, string> = {
  single_job: '单次任务', smart_reuse: '智能复用', multi_domain: '跨域复用', native_comfy: '原生 ComfyUI', prep: '缓存准备', smoke: '冒烟测试',
};
const PAGE_SIZE = 10;

type RangePresetKey = 'today' | '7d' | '30d' | 'month' | 'all';

const PRESETS: { key: RangePresetKey; label: string }[] = [
  { key: 'today', label: '今天' },
  { key: '7d', label: '近 7 天' },
  { key: '30d', label: '近 30 天' },
  { key: 'month', label: '本月' },
  { key: 'all', label: '全部' },
];

const WEEKDAYS = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];

function localTodayFallback(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function shiftIsoDate(isoDate: string, deltaDays: number): string {
  const [y, m, d] = isoDate.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d + deltaDays));
  const yyyy = dt.getUTCFullYear();
  const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(dt.getUTCDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function presetDates(today: string, preset: RangePresetKey): { start: string; end: string } {
  if (preset === 'all') return { start: '', end: '' };
  if (preset === 'today') return { start: today, end: today };
  if (preset === '7d') return { start: shiftIsoDate(today, -6), end: today };
  if (preset === '30d') return { start: shiftIsoDate(today, -29), end: today };
  return { start: `${today.slice(0, 7)}-01`, end: today };
}

function matchedPreset(today: string, startDate: string, endDate: string): RangePresetKey | null {
  for (const preset of PRESETS) {
    const range = presetDates(today, preset.key);
    if (startDate === range.start && endDate === range.end) return preset.key;
  }
  return null;
}

function formatTriggerLabel(today: string, startDate: string, endDate: string): string {
  if (!startDate && !endDate) return '全部日期';
  const preset = matchedPreset(today, startDate, endDate);
  if (preset === 'today') return `今天 · ${today.slice(5)}`;
  if (preset === '7d') return `近 7 天 · ${startDate.slice(5)} ~ ${endDate.slice(5)}`;
  if (preset === '30d') return `近 30 天 · ${startDate.slice(5)} ~ ${endDate.slice(5)}`;
  if (preset === 'month') return `本月 · ${startDate.slice(5)} ~ ${endDate.slice(5)}`;
  if (startDate && endDate) return startDate === endDate ? startDate : `${startDate} ~ ${endDate}`;
  return startDate ? `${startDate} 起` : `截至 ${endDate}`;
}

function formatSessionTimeRange(startedAtSec: number, endedAtSec?: number | null): string {
  const startMs = startedAtSec * 1000;
  const startStr = formatDateTime(startMs);
  if (endedAtSec == null) return `${startStr} 起`;
  const endMs = endedAtSec * 1000;
  const startDate = new Date(startMs);
  const endDate = new Date(endMs);
  const sameDay = !Number.isNaN(startDate.getTime())
    && !Number.isNaN(endDate.getTime())
    && startDate.toDateString() === endDate.toDateString();
  if (sameDay) {
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${startStr} – ${pad(endDate.getHours())}:${pad(endDate.getMinutes())}`;
  }
  return `${startStr} – ${formatDateTime(endMs)}`;
}

function dayRelativeBadge(isoDate: string, today: string): string {
  if (isoDate === today) return '今天';
  if (isoDate === shiftIsoDate(today, -1)) return '昨天';
  const parts = isoDate.split('-').map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) return '';
  const dt = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
  return WEEKDAYS[dt.getUTCDay()] || '';
}

function CuDateRangePicker({
  today,
  startDate,
  endDate,
  onApply,
}: {
  today: string;
  startDate: string;
  endDate: string;
  onApply: (start: string, end: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [draftStart, setDraftStart] = useState(startDate);
  const [draftEnd, setDraftEnd] = useState(endDate);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setDraftStart(startDate);
      setDraftEnd(endDate);
    }
  }, [open, startDate, endDate]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const draftInvalid = Boolean(draftStart && draftEnd && draftStart > draftEnd);
  const activePreset = matchedPreset(today, draftStart, draftEnd);
  const default7d = presetDates(today, '7d');
  const isDefault = startDate === default7d.start && endDate === default7d.end;

  return (
    <div className={styles.cuDatePickerWrap} ref={wrapRef}>
      <button
        type="button"
        className={styles.cuDateTrigger}
        data-open={open}
        data-filtered={!isDefault}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label="筛选时间范围"
        onClick={() => setOpen(v => !v)}
      >
        <Calendar size={14} strokeWidth={1.7} aria-hidden />
        <span>{formatTriggerLabel(today, startDate, endDate)}</span>
        <ChevronDown size={14} strokeWidth={1.7} className={styles.cuTriggerChevron} aria-hidden />
      </button>

      {open && (
        <div className={styles.cuDatePopover} role="dialog" aria-label="选择统计日期范围">
          <div className={styles.cuDateInputsRow}>
            <label className={styles.cuDateField}>
              <span>起始日期</span>
              <input
                type="date"
                value={draftStart}
                max={draftEnd || undefined}
                onChange={(event) => setDraftStart(event.target.value)}
              />
            </label>
            <span className={styles.cuDateTil} aria-hidden>~</span>
            <label className={styles.cuDateField}>
              <span>终止日期</span>
              <input
                type="date"
                value={draftEnd}
                min={draftStart || undefined}
                onChange={(event) => setDraftEnd(event.target.value)}
              />
            </label>
          </div>

          {draftInvalid && <p className={styles.cuDateError} role="alert">起始日期不能晚于终止日期</p>}

          <div className={styles.cuPresetRow} role="group" aria-label="快捷日期范围">
            {PRESETS.map((preset) => {
              const range = presetDates(today, preset.key);
              const active = activePreset === preset.key;
              return (
                <button
                  key={preset.key}
                  type="button"
                  className={styles.cuPresetChip}
                  data-active={active}
                  aria-pressed={active}
                  onClick={() => {
                    onApply(range.start, range.end);
                    setOpen(false);
                  }}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>

          <div className={styles.cuDatePopoverFoot}>
            <button
              type="button"
              className="button ghost small"
              onClick={() => {
                onApply(default7d.start, default7d.end);
                setOpen(false);
              }}
            >
              恢复近 7 天
            </button>
            <button
              type="button"
              className="button small"
              disabled={draftInvalid}
              onClick={() => {
                onApply(draftStart, draftEnd);
                setOpen(false);
              }}
            >
              确认
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function CuPager({
  offset,
  total,
  unit,
  loading,
  onChange,
}: {
  offset: number;
  total: number;
  unit: string;
  loading?: boolean;
  onChange: (nextOffset: number) => void;
}) {
  if (total <= 0) return null;
  const page = Math.floor(offset / PAGE_SIZE);
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if (pages <= 1) return null;
  return (
    <div className={styles.cuPager}>
      <button
        type="button"
        className="button secondary small"
        disabled={loading || page <= 0}
        onClick={() => onChange(Math.max(0, (page - 1) * PAGE_SIZE))}
      >
        上一页
      </button>
      <span className="muted small">第 {page + 1} / {pages} 页 · 共 {total} {unit}</span>
      <button
        type="button"
        className="button secondary small"
        disabled={loading || page >= pages - 1}
        onClick={() => onChange((page + 1) * PAGE_SIZE)}
      >
        下一页
      </button>
    </div>
  );
}

function SessionLine({
  session,
  accountAlias,
  active = false,
}: {
  session: CuSession;
  accountAlias?: string;
  active?: boolean;
}) {
  const expandable = session.session_kind === 'smart_reuse' || session.session_kind === 'multi_domain';
  const rowContent = (
    <div className={styles.cuSessionRow} data-expandable={expandable}>
      <div className={styles.cuSessionMain}>
        <div className={styles.cuSessionTop}>
          {active && <span className={styles.liveDot} data-busy="true" aria-hidden />}
          <strong>{labels[session.session_kind]}</strong>
          <span className={styles.cuGpuBadge}>{session.gpu_label}</span>
          {active && <span className={styles.cuMetaTag} data-tone="live">运行中</span>}
          {session.state === 'reconciling' && <span className={styles.cuMetaTag} data-tone="warn">对账中</span>}
          {session.draining && <span className={styles.cuMetaTag} data-tone="warn">排空中</span>}
          {expandable && session.jobs.length > 0 && (
            <span className={styles.cuMetaTag}>{session.jobs.length} 个任务</span>
          )}
        </div>
        <div className={styles.cuSessionSub}>
          <span>{formatSessionTimeRange(session.started_at, session.ended_at)}</span>
          {accountAlias && <span>· {accountAlias}</span>}
        </div>
      </div>

      <div className={styles.cuSessionDur}>
        {session.elapsed_minutes.toFixed(1)} 分钟
      </div>

      <div className={styles.cuSessionCost}>
        <strong className={session.actual_cu > 0 ? styles.cuCostValue : styles.cuCostZero}>
          {cuNumber(session.actual_cu)}<em> CU</em>
        </strong>
        {session.estimated_cu != null && (
          <small className="muted">估算 {cuNumber(session.estimated_cu)}</small>
        )}
      </div>

      <span className={styles.cuSessionCaret} aria-hidden>
        {expandable && <ChevronDown size={15} strokeWidth={1.7} className={styles.cuRowChevron} />}
      </span>
    </div>
  );

  if (!expandable) {
    return <div className={styles.cuListRow}>{rowContent}</div>;
  }

  return (
    <details className={styles.cuListFold}>
      <summary className={styles.cuListRow}>{rowContent}</summary>
      <div className={styles.cuSubJobs}>
        {session.jobs.map(job => (
          <div key={job.job_id} className={styles.cuSubJobRow}>
            <span className={styles.cuSubJobName}>{job.workflow_key || job.job_id.slice(0, 8)}</span>
            <span className={styles.cuSubJobDur}>{(job.execution_seconds / 60).toFixed(1)} 分钟</span>
            <strong className={styles.cuSubJobCost}>{cuNumber(job.actual_cu)} CU</strong>
          </div>
        ))}
        <div className={`${styles.cuSubJobRow} ${styles.cuSubJobIdle}`}>
          <span className={styles.cuSubJobName}>启动与空闲待机</span>
          <span className={styles.cuSubJobDur}>—</span>
          <span className={styles.cuSubJobCost}>{cuNumber(session.idle_cu)} CU</span>
        </div>
      </div>
    </details>
  );
}

type CuPageQuery = { kind: CuSessionKind | 'all'; startDate: string; endDate: string; offset: number };

function useCuPage<T extends { total: number }>(
  load: (query: CuPageQuery & { limit: number }, signal: AbortSignal) => Promise<T>,
  { kind, startDate, endDate, offset }: CuPageQuery,
  enabled: boolean,
  refreshKey: string,
  setOffset: (offset: number) => void,
) {
  const key = `${kind}:${startDate}:${endDate}:${offset}`;
  const [result, setResult] = useState<{ key: string; page?: T; loading: boolean; error?: string }>();

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setResult(previous => ({ key, page: previous?.key === key ? previous.page : undefined, loading: true }));
    void load({ kind, startDate, endDate, offset, limit: PAGE_SIZE }, controller.signal).then(
      (page) => {
        if (controller.signal.aborted) return;
        if (offset > 0 && offset >= page.total) {
          setOffset(Math.floor(Math.max(0, page.total - 1) / PAGE_SIZE) * PAGE_SIZE);
          return;
        }
        setResult({ key, page, loading: false });
      },
      (cause) => {
        if (controller.signal.aborted) return;
        setResult(previous => ({ key, page: previous?.key === key ? previous.page : undefined,
          loading: false, error: apiMessage(cause) }));
      },
    );
    return () => controller.abort();
  }, [enabled, load, key, kind, startDate, endDate, offset, refreshKey, setOffset]);

  const current = enabled && result?.key === key ? result : undefined;
  return { page: current?.page, loading: enabled && (current?.loading ?? true), error: current?.error };
}

export function CuSessionDetails({ usage }: { usage: GpuUsage | null }) {
  const [tab, setTab] = useState<'sessions' | 'daily'>('sessions');
  const today = usage?.local_date || localTodayFallback();
  const initial7d = useMemo(() => presetDates(localTodayFallback(), '7d'), []);
  const default7d = useMemo(() => presetDates(today, '7d'), [today]);
  const tzNote = usage?.timezone ? `按 ${usage.timezone} 自然日` : '';

  const accountAliasMap = useMemo(() => {
    const colabAccounts = (usage?.accounts || []).filter(a => a.colab_connected !== false);
    if (colabAccounts.length <= 1) return new Map<string, string>();
    return new Map(colabAccounts.map(a => [a.account_id, a.alias || a.account_id]));
  }, [usage?.accounts]);

  // Stable scalar signature: only trigger background refresh when CU/session totals actually change.
  const usageRefreshKey = `${usage?.recent_sessions_count ?? ''}:${usage?.recent_sessions_actual_cu ?? ''}:${usage?.today_actual_cu ?? ''}:${usage?.history_actual_cu ?? ''}:${usage?.daily_history?.length ?? ''}`;

  // Tab 1: sessions filter + pagination state (defaults to trailing 7 days)
  const [kind, setKind] = useState<CuSessionKind | 'all'>('all');
  const [startDate, setStartDate] = useState(initial7d.start);
  const [endDate, setEndDate] = useState(initial7d.end);
  const [sessionRangeTouched, setSessionRangeTouched] = useState(false);
  const [offset, setOffset] = useState(0);

  // Sync default 7-day range once server studio timezone local_date loads, unless user already chose a range.
  useEffect(() => {
    if (!sessionRangeTouched && usage?.local_date) {
      const next = presetDates(usage.local_date, '7d');
      setStartDate(next.start);
      setEndDate(next.end);
    }
  }, [sessionRangeTouched, usage?.local_date]);

  const sessionRangeInvalid = Boolean(startDate && endDate && startDate > endDate);
  const useRecent = !startDate && !endDate && kind === 'all' && offset === 0;
  const sessionModified = kind !== 'all' || startDate !== default7d.start || endDate !== default7d.end;
  const { page: sessionPage, loading: sessionLoading, error: sessionRequestError } = useCuPage(
    getCuSessions, { kind, startDate, endDate, offset },
    tab === 'sessions' && !useRecent && !sessionRangeInvalid, usageRefreshKey, setOffset,
  );
  const sessionError = sessionRangeInvalid ? '起始日期不能晚于终止日期' : sessionRequestError;

  const sessionRows: CuSession[] | undefined = useRecent
    ? usage?.recent_sessions?.slice(0, PAGE_SIZE)
    : sessionPage?.sessions;
  const sessionTotal = useRecent ? usage?.recent_sessions_count : sessionPage?.total;
  const sessionTotalActual = useRecent
    ? usage?.recent_sessions_actual_cu
    : sessionPage?.total_actual_cu;

  // Tab 2: daily summary filter + pagination state (defaults to trailing 7 days)
  const [dailyStartDate, setDailyStartDate] = useState(initial7d.start);
  const [dailyEndDate, setDailyEndDate] = useState(initial7d.end);
  const [dailyRangeTouched, setDailyRangeTouched] = useState(false);
  const [dailyOffset, setDailyOffset] = useState(0);

  useEffect(() => {
    if (!dailyRangeTouched && usage?.local_date) {
      const next = presetDates(usage.local_date, '7d');
      setDailyStartDate(next.start);
      setDailyEndDate(next.end);
    }
  }, [dailyRangeTouched, usage?.local_date]);

  const dailyRangeInvalid = Boolean(dailyStartDate && dailyEndDate && dailyStartDate > dailyEndDate);
  const useDefaultDaily = !dailyStartDate && !dailyEndDate && dailyOffset === 0;
  const hasDailyDateFilter = Boolean(dailyStartDate) || Boolean(dailyEndDate);
  const dailyModified = dailyStartDate !== default7d.start || dailyEndDate !== default7d.end;

  // Single daily fetch (up to 90 days) drives both the SVG trend chart and the first 90 items of the list,
  // avoiding firing two identical /api/gpu/cu/daily requests for the same date range.
  const [chartState, setChartState] = useState<{
    key: string;
    page?: CuDailyPage;
    loading: boolean;
    error?: string;
  }>({
    key: '',
    loading: false,
  });
  const chartQueryKey = `${dailyStartDate}:${dailyEndDate}:${usageRefreshKey}`;
  const chartNeedsFetch = tab === 'daily' && !dailyRangeInvalid && hasDailyDateFilter;
  useEffect(() => {
    if (!chartNeedsFetch) return;
    const controller = new AbortController();
    setChartState((prev) => ({
      key: chartQueryKey,
      page: prev.key === chartQueryKey ? prev.page : undefined,
      loading: true,
    }));
    void getCuDaily({ startDate: dailyStartDate, endDate: dailyEndDate, limit: 90, offset: 0 }, controller.signal).then(
      (page) => {
        if (!controller.signal.aborted) {
          setChartState({ key: chartQueryKey, page, loading: false });
        }
      },
      (cause) => {
        if (!controller.signal.aborted) {
          setChartState((prev) => ({ ...prev, loading: false, error: apiMessage(cause) }));
        }
      },
    );
    return () => controller.abort();
  }, [chartNeedsFetch, chartQueryKey, dailyStartDate, dailyEndDate]);

  const chartSliceCoversPage = chartNeedsFetch && dailyOffset + PAGE_SIZE <= 90;
  const { page: pagedDaily, loading: pagedDailyLoading, error: pagedDailyError } = useCuPage(
    getCuDaily, { kind: 'all', startDate: dailyStartDate, endDate: dailyEndDate, offset: dailyOffset },
    tab === 'daily' && !useDefaultDaily && !dailyRangeInvalid && !chartSliceCoversPage, usageRefreshKey, setDailyOffset,
  );
  const chartPageForKey = chartState.key === chartQueryKey ? chartState.page : undefined;
  const dailyPage: CuDailyPage | undefined = chartSliceCoversPage
    ? (chartPageForKey
      ? { ...chartPageForKey, days: chartPageForKey.days.slice(dailyOffset, dailyOffset + PAGE_SIZE), limit: PAGE_SIZE, offset: dailyOffset }
      : undefined)
    : pagedDaily;
  const dailyLoading = chartSliceCoversPage ? (chartNeedsFetch && chartState.loading) : pagedDailyLoading;
  const dailyRequestError = chartSliceCoversPage ? (chartState.key === chartQueryKey ? chartState.error : undefined) : pagedDailyError;
  const dailyError = dailyRangeInvalid ? '起始日期不能晚于终止日期' : dailyRequestError;

  const chartDays: CuDailyEntry[] = !hasDailyDateFilter
    ? (usage?.daily_history || [])
    : (chartPageForKey?.days || dailyPage?.days || []);
  const chartLoading = chartNeedsFetch && chartState.loading;

  const defaultDailySummary = useMemo(() => {
    const history = usage?.daily_history;
    if (!history) return null;
    const totalActual = usage?.history_actual_cu ?? history.reduce((acc, day) => acc + day.actual_cu, 0);
    const totalSessions = history.reduce((acc, day) => acc + day.sessions, 0);
    return {
      days: history.slice(0, PAGE_SIZE),
      total: history.length,
      total_actual_cu: totalActual,
      total_sessions: totalSessions,
    };
  }, [usage?.daily_history, usage?.history_actual_cu]);

  const dailyRows: CuDailyEntry[] | undefined = useDefaultDaily
    ? defaultDailySummary?.days
    : dailyPage?.days;
  const dailyTotal = useDefaultDaily ? defaultDailySummary?.total : dailyPage?.total;
  const dailyTotalActual = useDefaultDaily ? defaultDailySummary?.total_actual_cu : dailyPage?.total_actual_cu;
  const dailyTotalSessions = useDefaultDaily ? defaultDailySummary?.total_sessions : dailyPage?.total_sessions;

  const maxDailyCu = useMemo(() => {
    const source = chartDays.length ? chartDays : (dailyRows || []);
    return source.reduce((peak, entry) => Math.max(peak, entry.actual_cu || 0), 0);
  }, [chartDays, dailyRows]);

  function drillToDate(selectedDate: string) {
    setStartDate(selectedDate);
    setEndDate(selectedDate);
    setSessionRangeTouched(true);
    setKind('all');
    setOffset(0);
    setTab('sessions');
  }

  const activeSessions = usage?.active_sessions || [];

  return (
    <Card
      title="CU 消耗明细"
      aside={tzNote ? <span className="muted small">{tzNote}</span> : undefined}
    >
      <Segmented
        ariaLabel="扣费明细视图"
        value={tab}
        onChange={setTab}
        options={[
          { value: 'sessions', label: '运行会话与任务' },
          { value: 'daily', label: '按自然日汇总' },
        ]}
      />

      {tab === 'daily' ? (
        <div className={styles.stack}>
          <div className={styles.cuToolbar}>
            <div className={styles.cuToolbarFilters}>
              <CuDateRangePicker
                today={today}
                startDate={dailyStartDate}
                endDate={dailyEndDate}
                onApply={(nextStart, nextEnd) => {
                  setDailyStartDate(nextStart);
                  setDailyEndDate(nextEnd);
                  setDailyRangeTouched(true);
                  setDailyOffset(0);
                }}
              />
              {dailyModified && (
                <button
                  type="button"
                  className={styles.cuResetBtn}
                  onClick={() => {
                    setDailyStartDate(default7d.start);
                    setDailyEndDate(default7d.end);
                    setDailyRangeTouched(false);
                    setDailyOffset(0);
                  }}
                >
                  <RotateCcw size={13} aria-hidden />重置
                </button>
              )}
            </div>

            {!dailyRangeInvalid && dailyTotal != null && (
              <div className={styles.cuToolbarStats} aria-label="统计结果">
                <span className={styles.cuStatPill} data-accent="true">
                  <small>实扣</small>
                  <strong>{cuNumber(dailyTotalActual)} CU</strong>
                </span>
                <span className={styles.cuStatPill}>
                  <small>天数</small>
                  <strong>{dailyTotal} 天</strong>
                </span>
                <span className={styles.cuStatPill}>
                  <small>开机</small>
                  <strong>{dailyTotalSessions ?? 0} 场</strong>
                </span>
                {dailyLoading && <span className="muted small">更新中…</span>}
              </div>
            )}
          </div>

          {dailyError && <p role="alert" className="status error">{dailyError}</p>}

          {!dailyRangeInvalid && (
            <CuDailyChart
              days={chartDays}
              today={today}
              startDate={dailyStartDate}
              endDate={dailyEndDate}
              dailyLimitCu={usage?.guard?.daily_limit_cu}
              loading={chartLoading}
              onSelectDate={drillToDate}
            />
          )}

          {!dailyRangeInvalid && dailyRows && dailyRows.length > 0 && (
            <div className={styles.cuList}>
              {dailyRows.map(day => {
                const barPct = maxDailyCu > 0 && day.actual_cu > 0
                  ? Math.max(3, Math.min(100, (day.actual_cu / maxDailyCu) * 100))
                  : 0;
                const badge = dayRelativeBadge(day.date, today);
                const clickable = day.sessions > 0;
                const inner = (
                  <div className={styles.cuDailyRow}>
                    <div className={styles.cuDailyDateCol}>
                      <div className={styles.cuDailyDateTop}>
                        <strong className={styles.cuDateMono}>{day.date}</strong>
                        {badge && <span className={styles.cuWeekdayTag} data-today={badge === '今天'}>{badge}</span>}
                      </div>
                      {day.legacy_actual_cu != null && (
                        <small className="muted">含迁移前 {cuNumber(day.legacy_actual_cu)} CU</small>
                      )}
                    </div>

                    <div className={styles.cuDailyBarCol}>
                      <div className={styles.cuDailyBarTrack} aria-hidden>
                        <span className={styles.cuDailyBarFill} style={{ width: `${barPct}%` }} />
                      </div>
                      <span className={day.sessions > 0 ? styles.cuDailySessions : styles.cuDailySessionsZero}>
                        {day.sessions} 场
                      </span>
                    </div>

                    <div className={styles.cuDailyCostCol}>
                      <strong className={day.actual_cu > 0 ? styles.cuCostValue : styles.cuCostZero}>
                        {cuNumber(day.actual_cu)}<em> CU</em>
                      </strong>
                    </div>

                    <span className={styles.cuDailyArrow} aria-hidden>
                      {clickable && <ChevronRight size={15} strokeWidth={1.7} />}
                    </span>
                  </div>
                );
                return clickable ? (
                  <button
                    key={day.date}
                    type="button"
                    className={`${styles.cuListRow} ${styles.cuListRowButton}`}
                    title={`查看 ${day.date} 的运行会话`}
                    onClick={() => drillToDate(day.date)}
                  >
                    {inner}
                  </button>
                ) : (
                  <div key={day.date} className={styles.cuListRow}>{inner}</div>
                );
              })}
            </div>
          )}

          {!dailyRangeInvalid && dailyRows?.length === 0 && !dailyLoading && (
            <p className="muted small">所选时间范围内暂无扣费记录。</p>
          )}
          {!dailyRangeInvalid && !dailyRows && !dailyError && (
            <p className="muted small">正在加载汇总…</p>
          )}

          {!dailyRangeInvalid && dailyTotal != null && (
            <CuPager
              offset={dailyOffset}
              total={dailyTotal}
              unit="天"
              loading={dailyLoading}
              onChange={setDailyOffset}
            />
          )}
        </div>
      ) : (
        <div className={styles.stack}>
          {activeSessions.length > 0 && (
            <div className={styles.cuActiveSection}>
              <div className={styles.cuSubLabel}>运行中（{activeSessions.length}）</div>
              <div className={styles.cuList} data-active-list="true">
                {activeSessions.map(session => (
                  <SessionLine
                    key={session.instance_id}
                    session={session}
                    accountAlias={accountAliasMap.get(session.account_id)}
                    active
                  />
                ))}
              </div>
            </div>
          )}

          <div className={styles.cuToolbar}>
            <div className={styles.cuToolbarFilters}>
              <CuDateRangePicker
                today={today}
                startDate={startDate}
                endDate={endDate}
                onApply={(nextStart, nextEnd) => {
                  setStartDate(nextStart);
                  setEndDate(nextEnd);
                  setSessionRangeTouched(true);
                  setOffset(0);
                }}
              />
              <select
                aria-label="会话类型"
                className={styles.cuKindSelect}
                value={kind}
                onChange={event => {
                  setKind(event.target.value as CuSessionKind | 'all');
                  setOffset(0);
                }}
              >
                <option value="all">全部类型</option>
                {Object.entries(labels).map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
              {sessionModified && (
                <button
                  type="button"
                  className={styles.cuResetBtn}
                  onClick={() => {
                    setKind('all');
                    setStartDate(default7d.start);
                    setEndDate(default7d.end);
                    setSessionRangeTouched(false);
                    setOffset(0);
                  }}
                >
                  <RotateCcw size={13} aria-hidden />重置
                </button>
              )}
            </div>

            {!sessionRangeInvalid && sessionTotal != null && (
              <div className={styles.cuToolbarStats} aria-label="统计结果">
                {sessionTotalActual != null && (
                  <span className={styles.cuStatPill} data-accent="true">
                    <small>实扣</small>
                    <strong>{cuNumber(sessionTotalActual)} CU</strong>
                  </span>
                )}
                <span className={styles.cuStatPill}>
                  <small>会话</small>
                  <strong>{sessionTotal} 场</strong>
                </span>
                {sessionLoading && <span className="muted small">更新中…</span>}
              </div>
            )}
          </div>

          {sessionError && <p role="alert" className="status error">{sessionError}</p>}

          {!sessionRangeInvalid && sessionRows && sessionRows.length > 0 && (
            <div className={styles.cuList}>
              {sessionRows.map(session => (
                <SessionLine
                  key={session.instance_id}
                  session={session}
                  accountAlias={accountAliasMap.get(session.account_id)}
                />
              ))}
            </div>
          )}

          {!sessionRangeInvalid && sessionRows?.length === 0 && !sessionLoading && (
            <p className="muted small">所选条件下暂无运行会话。</p>
          )}
          {!sessionRangeInvalid && !sessionRows && !sessionError && (
            <p className="muted small">正在加载会话…</p>
          )}

          {!sessionRangeInvalid && sessionTotal != null && (
            <CuPager
              offset={offset}
              total={sessionTotal}
              unit="条"
              loading={sessionLoading}
              onChange={setOffset}
            />
          )}
        </div>
      )}
    </Card>
  );
}
