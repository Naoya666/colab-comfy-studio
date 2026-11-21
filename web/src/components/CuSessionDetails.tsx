import { useEffect, useState } from 'react';
import { apiMessage, getCuSessions, type CuSession, type CuSessionKind, type GpuUsage } from '../lib/gpu-api';
import { cuNumber, formatDateTime } from '../lib/ui-helpers';
import { Card, Segmented } from './ui';
import styles from '../gpu.module.css';

const labels: Record<CuSessionKind, string> = {
  single_job: '单次任务', smart_reuse: '智能复用', native_comfy: '原生 ComfyUI', prep: '缓存准备', smoke: '冒烟测试',
};
const pageSize = 30;

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

export function CuSessionDetails({ usage }: { usage: GpuUsage | null }) {
  const [tab, setTab] = useState<'sessions' | 'daily'>('sessions');
  const [kind, setKind] = useState<CuSessionKind | 'all'>('all');
  const [date, setDate] = useState('');
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<Awaited<ReturnType<typeof getCuSessions>>>();
  const [error, setError] = useState('');
  const useRecent = !date && kind === 'all' && offset === 0;
  const recent = usage?.recent_sessions;
  const count = usage?.recent_sessions_count;
  useEffect(() => {
    if (tab !== 'sessions' || useRecent) return;
    let current = true;
    setPage(undefined);
    setError('');
    void getCuSessions({ kind, date, offset, limit: pageSize }).then(value => {
      if (current) setPage(value);
    }, cause => { if (current) setError(apiMessage(cause)); });
    return () => { current = false; };
  }, [tab, kind, date, offset, useRecent, recent]);
  const rows = useRecent ? recent : page?.sessions;
  const total = useRecent ? count : page?.total;

  return <Card title="CU 消耗明细">
    <Segmented ariaLabel="扣费明细视图" value={tab} onChange={setTab}
      options={[{ value: 'sessions', label: '运行会话与任务' }, { value: 'daily', label: '按自然日汇总' }]} />
    {tab === 'daily' ? <div className={styles.stack}>
      {(usage?.daily_history || []).map(day => <div className={styles.inlinePanel} key={day.date}>
        <strong>{day.date}</strong> · 扣费 {cuNumber(day.actual_cu)} CU · 开机 {day.sessions} 场
        {day.legacy_actual_cu != null && <span> · 其中迁移前实扣 {cuNumber(day.legacy_actual_cu)} CU</span>}
      </div>)}
      {usage && !usage.daily_history?.length && <p className="muted small">暂无扣费记录。</p>}
    </div> : <>
      <h3>运行中</h3>
      {(usage?.active_sessions || []).map(session => <SessionLine key={session.instance_id} session={session} />)}
      {usage && !usage.active_sessions?.length && <p className="muted small">暂无运行中会话。</p>}
      <h3>已结束与对账中</h3>
      <div className={styles.grid2}>
        <label className="field">会话类型
          <select value={kind} onChange={event => { setKind(event.target.value as CuSessionKind | 'all'); setOffset(0); }}>
            <option value="all">全部</option>
            {Object.entries(labels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="field">开机日期 {usage?.timezone && `（${usage.timezone}）`}
          <input type="date" value={date} onChange={event => { setDate(event.target.value); setOffset(0); }} />
        </label>
      </div>
      {rows?.map(session => <SessionLine key={session.instance_id} session={session} />)}
      {tab === 'sessions' && !useRecent && error && <p role="alert" className="status error">{error}</p>}
      {rows?.length === 0 && <p className="muted small">暂无符合条件的会话。</p>}
      {!rows && !error && <p className="muted small">正在加载会话…</p>}
      {(offset > 0 || (total ?? 0) > pageSize) && <div className="actions">
        <button type="button" className="button secondary small" disabled={offset === 0} onClick={() => setOffset(offset - pageSize)}>上一页</button>
        <button type="button" className="button secondary small" disabled={offset + pageSize >= (total ?? 0)} onClick={() => setOffset(offset + pageSize)}>下一页</button>
      </div>}
    </>}
  </Card>;
}
