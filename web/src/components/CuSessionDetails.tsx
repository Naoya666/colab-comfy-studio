import { useEffect, useMemo, useState } from 'react';
import {
  apiMessage,
  getCuDaily,
  getCuSessions,
  type CuDailyEntry,
  type CuSession,
  type CuSessionKind,
  type GpuUsage,
} from '../lib/gpu-api';
import { cuNumber, formatDateTime } from '../lib/ui-helpers';
import { CuDailyChart } from './CuDailyChart';
import { Card, Segmented } from './ui';
import styles from '../gpu.module.css';

const labels: Record<CuSessionKind, string> = {
  single_job: '单次任务', smart_reuse: '智能复用', native_comfy: '原生 ComfyUI', prep: '缓存准备', smoke: '冒烟测试',
};
const PAGE_SIZE = 10;

type RangePresetKey = 'all' | 'today' | '7d' | '30d' | 'month';

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

function formatDateRangeLabel(startDate: string, endDate: string): string {
  if (!startDate && !endDate) return '全部日期';
  if (startDate && endDate) return startDate === endDate ? startDate : `${startDate} 至 ${endDate}`;
  return startDate ? `${startDate} 起` : `截至 ${endDate}`;
}

function DatePresets({
  today,
  startDate,
  endDate,
  onSelect,
}: {
  today: string;
  startDate: string;
  endDate: string;
  onSelect: (start: string, end: string) => void;
}) {
  const presets: { key: RangePresetKey; label: string }[] = [
    { key: 'all', label: '全部' },
    { key: 'today', label: '今天' },
    { key: '7d', label: '近 7 天' },
    { key: '30d', label: '近 30 天' },
    { key: 'month', label: '本月' },
  ];
  return (
    <div className={styles.chipRow} role="group" aria-label="快捷日期范围">
      {presets.map((preset) => {
        const range = presetDates(today, preset.key);
        const active = startDate === range.start && endDate === range.end;
        return (
          <button
            key={preset.key}
            type="button"
            className={styles.chip}
            data-active={active}
            aria-pressed={active}
            onClick={() => onSelect(range.start, range.end)}
          >
            {preset.label}
          </button>
        );
      })}
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

function SessionLine({ session }: { session: CuSession }) {
  const body = <>
    <strong>{labels[session.session_kind]} · {session.gpu_label}</strong>
    <span> · {session.elapsed_minutes.toFixed(1)} 分钟 · 实扣 {cuNumber(session.actual_cu)} CU</span>
    {session.state === 'reconciling' && <span> · 对账中</span>}
    {session.draining && <span> · 排空中</span>}
    <span className={styles.cuSessionTiming}>
      {formatDateTime(session.started_at * 1000)}
      {session.ended_at != null && <> 至 {formatDateTime(session.ended_at * 1000)}</>}
      {session.estimated_cu != null && <> · 估算 {cuNumber(session.estimated_cu)} CU</>}
    </span>
  </>;
  if (session.session_kind !== 'smart_reuse') return <div className={styles.inlinePanel}>{body}</div>;
  return <details className="foldable">
    <summary>{body}</summary>
    <div className="foldContent">
      {session.jobs.map(job => <p key={job.job_id} className="small">
        {job.workflow_key || job.job_id.slice(0, 8)} · {(job.execution_seconds / 60).toFixed(1)} 分钟 · {cuNumber(job.actual_cu)} CU
      </p>)}
      <p className="small">启动与空闲待机 · {cuNumber(session.idle_cu)} CU</p>
    </div>
  </details>;
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
  const tzNote = usage?.timezone ? `按 ${usage.timezone} 自然日` : '';

  // Stable scalar signature: only trigger background refresh when CU/session totals actually change.
  const usageRefreshKey = `${usage?.recent_sessions_count ?? ''}:${usage?.recent_sessions_actual_cu ?? ''}:${usage?.today_actual_cu ?? ''}:${usage?.history_actual_cu ?? ''}:${usage?.daily_history?.length ?? ''}`;

  // Tab 1: sessions filter + pagination state
  const [kind, setKind] = useState<CuSessionKind | 'all'>('all');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [offset, setOffset] = useState(0);
  const sessionRangeInvalid = Boolean(startDate && endDate && startDate > endDate);
  const useRecent = !startDate && !endDate && kind === 'all' && offset === 0;
  const hasSessionFilter = kind !== 'all' || Boolean(startDate) || Boolean(endDate);
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

  // Tab 2: daily summary filter + pagination state
  const [dailyStartDate, setDailyStartDate] = useState('');
  const [dailyEndDate, setDailyEndDate] = useState('');
  const [dailyOffset, setDailyOffset] = useState(0);
  const dailyRangeInvalid = Boolean(dailyStartDate && dailyEndDate && dailyStartDate > dailyEndDate);
  const useDefaultDaily = !dailyStartDate && !dailyEndDate && dailyOffset === 0;
  const hasDailyFilter = Boolean(dailyStartDate) || Boolean(dailyEndDate);
  const { page: dailyPage, loading: dailyLoading, error: dailyRequestError } = useCuPage(
    getCuDaily, { kind: 'all', startDate: dailyStartDate, endDate: dailyEndDate, offset: dailyOffset },
    tab === 'daily' && !useDefaultDaily && !dailyRangeInvalid, usageRefreshKey, setDailyOffset,
  );
  const dailyError = dailyRangeInvalid ? '起始日期不能晚于终止日期' : dailyRequestError;

  // Separate unpaginated daily fetch for the SVG trend chart when filtered.
  const [chartState, setChartState] = useState<{ key: string; days: CuDailyEntry[]; loading: boolean }>({
    key: '',
    days: [],
    loading: false,
  });
  const chartQueryKey = `${dailyStartDate}:${dailyEndDate}:${usageRefreshKey}`;
  const chartNeedsFetch = tab === 'daily' && !dailyRangeInvalid && hasDailyFilter;
  useEffect(() => {
    if (!chartNeedsFetch) return;
    const controller = new AbortController();
    setChartState((prev) => ({ key: chartQueryKey, days: prev.key === chartQueryKey ? prev.days : [], loading: true }));
    void getCuDaily({ startDate: dailyStartDate, endDate: dailyEndDate, limit: 90, offset: 0 }, controller.signal).then(
      (page) => {
        if (!controller.signal.aborted) {
          setChartState({ key: chartQueryKey, days: page.days, loading: false });
        }
      },
      () => {
        if (!controller.signal.aborted) {
          setChartState((prev) => ({ ...prev, loading: false }));
        }
      },
    );
    return () => controller.abort();
  }, [chartNeedsFetch, chartQueryKey, dailyStartDate, dailyEndDate]);

  const chartDays: CuDailyEntry[] = !hasDailyFilter
    ? (usage?.daily_history || [])
    : (chartState.key === chartQueryKey && chartState.days.length ? chartState.days : (dailyPage?.days || []));
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

  return <Card title="CU 消耗明细">
    <Segmented ariaLabel="扣费明细视图" value={tab} onChange={setTab}
      options={[{ value: 'sessions', label: '运行会话与任务' }, { value: 'daily', label: '按自然日汇总' }]} />

    {tab === 'daily' ? <div className={styles.stack}>
      <div className={styles.cuSectionHead}>
        <div>
          <h3>按自然日统计</h3>
          {tzNote && <span className="muted small">{tzNote}</span>}
        </div>
        {hasDailyFilter && <button
          type="button"
          className="button ghost small"
          onClick={() => { setDailyStartDate(''); setDailyEndDate(''); setDailyOffset(0); }}
        >
          清空筛选
        </button>}
      </div>

      <DatePresets
        today={today}
        startDate={dailyStartDate}
        endDate={dailyEndDate}
        onSelect={(nextStart, nextEnd) => {
          setDailyStartDate(nextStart);
          setDailyEndDate(nextEnd);
          setDailyOffset(0);
        }}
      />

      <div className={styles.cuFilters}>
        <label className="field">起始日期
          <input
            type="date"
            value={dailyStartDate}
            max={dailyEndDate || undefined}
            onChange={(event) => { setDailyStartDate(event.target.value); setDailyOffset(0); }}
          />
        </label>
        <label className="field">终止日期
          <input
            type="date"
            value={dailyEndDate}
            min={dailyStartDate || undefined}
            onChange={(event) => { setDailyEndDate(event.target.value); setDailyOffset(0); }}
          />
        </label>
      </div>

      {dailyError && <p role="alert" className="status error">{dailyError}</p>}

      {!dailyRangeInvalid && dailyTotal != null && (
        <div className={styles.cuRangeSummary}>
          <span className="muted">{formatDateRangeLabel(dailyStartDate, dailyEndDate)}</span>
          <span className={styles.cuRangeMetrics}>
            <span>实扣合计 <strong>{cuNumber(dailyTotalActual)} CU</strong></span>
            <span>· 共 <strong>{dailyTotal}</strong> 天</span>
            <span>· 开机 <strong>{dailyTotalSessions ?? 0}</strong> 场</span>
            {dailyLoading && <span className="muted">更新中…</span>}
          </span>
        </div>
      )}

      {!dailyRangeInvalid && (
        <CuDailyChart
          days={chartDays}
          today={today}
          startDate={dailyStartDate}
          endDate={dailyEndDate}
          dailyLimitCu={usage?.guard?.daily_limit_cu}
          loading={chartLoading}
          onSelectDate={(selectedDate) => {
            setStartDate(selectedDate);
            setEndDate(selectedDate);
            setKind('all');
            setOffset(0);
            setTab('sessions');
          }}
        />
      )}

      {!dailyRangeInvalid && dailyRows?.map(day => <div className={styles.inlinePanel} key={day.date}>
        <strong>{day.date}</strong> · 扣费 {cuNumber(day.actual_cu)} CU · 开机 {day.sessions} 场
        {day.legacy_actual_cu != null && <span> · 其中迁移前实扣 {cuNumber(day.legacy_actual_cu)} CU</span>}
      </div>)}
      {!dailyRangeInvalid && dailyRows?.length === 0 && !dailyLoading && <p className="muted small">暂无符合条件的扣费记录。</p>}
      {!dailyRangeInvalid && !dailyRows && !dailyError && <p className="muted small">正在加载汇总…</p>}

      {!dailyRangeInvalid && dailyTotal != null && (
        <CuPager
          offset={dailyOffset}
          total={dailyTotal}
          unit="天"
          loading={dailyLoading}
          onChange={setDailyOffset}
        />
      )}
    </div> : <>
      <h3>运行中</h3>
      {(usage?.active_sessions || []).map(session => <SessionLine key={session.instance_id} session={session} />)}
      {usage && !usage.active_sessions?.length && <p className="muted small">暂无运行中会话。</p>}

      <div className={styles.cuSectionHead}>
        <div>
          <h3>已结束与对账中</h3>
          {tzNote && <span className="muted small">{tzNote}</span>}
        </div>
        {hasSessionFilter && <button
          type="button"
          className="button ghost small"
          onClick={() => { setKind('all'); setStartDate(''); setEndDate(''); setOffset(0); }}
        >
          清空筛选
        </button>}
      </div>

      <DatePresets
        today={today}
        startDate={startDate}
        endDate={endDate}
        onSelect={(nextStart, nextEnd) => {
          setStartDate(nextStart);
          setEndDate(nextEnd);
          setOffset(0);
        }}
      />

      <div className={styles.cuFilters}>
        <label className="field">会话类型
          <select value={kind} onChange={event => { setKind(event.target.value as CuSessionKind | 'all'); setOffset(0); }}>
            <option value="all">全部</option>
            {Object.entries(labels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="field">起始日期
          <input
            type="date"
            value={startDate}
            max={endDate || undefined}
            onChange={event => { setStartDate(event.target.value); setOffset(0); }}
          />
        </label>
        <label className="field">终止日期
          <input
            type="date"
            value={endDate}
            min={startDate || undefined}
            onChange={event => { setEndDate(event.target.value); setOffset(0); }}
          />
        </label>
      </div>

      {sessionError && <p role="alert" className="status error">{sessionError}</p>}

      {!sessionRangeInvalid && sessionTotal != null && (
        <div className={styles.cuRangeSummary}>
          <span className="muted">
            {formatDateRangeLabel(startDate, endDate)}
            {kind !== 'all' && ` · ${labels[kind]}`}
          </span>
          <span className={styles.cuRangeMetrics}>
            {sessionTotalActual != null && <span>实扣合计 <strong>{cuNumber(sessionTotalActual)} CU</strong></span>}
            <span>{sessionTotalActual != null ? '· ' : ''}共 <strong>{sessionTotal}</strong> 场</span>
            {sessionLoading && <span className="muted">更新中…</span>}
          </span>
        </div>
      )}

      {!sessionRangeInvalid && sessionRows?.map(session => <SessionLine key={session.instance_id} session={session} />)}
      {!sessionRangeInvalid && sessionRows?.length === 0 && !sessionLoading && <p className="muted small">暂无符合条件的会话。</p>}
      {!sessionRangeInvalid && !sessionRows && !sessionError && <p className="muted small">正在加载会话…</p>}

      {!sessionRangeInvalid && sessionTotal != null && (
        <CuPager
          offset={offset}
          total={sessionTotal}
          unit="条"
          loading={sessionLoading}
          onChange={setOffset}
        />
      )}
    </>}
  </Card>;
}
