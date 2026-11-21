import { useState } from 'react';
import { timingRows, type TimingDetail } from '../lib/job-history';
import { jobStageLabel, jobStateLabel } from '../lib/job-stage-rows';

const LOG_PREVIEW_COUNT = 6;

export interface JobHistoryDetailRecord {
  state?: string;
  stage?: string | null;
  timings?: Record<string, number>;
  timing_details?: Record<string, TimingDetail>;
  logs?: Array<{ id?: string; state?: string; created_at?: string; message?: string }>;
  events?: Array<{ id?: string; state?: string; created_at?: string; message?: string }>;
}

export function JobHistoryDetails({ job }: { job: JobHistoryDetailRecord }) {
  const [showAllLogs, setShowAllLogs] = useState(false);
  const rows = timingRows(job);
  const logs = job.logs || job.events || [];
  const hiddenCount = Math.max(0, logs.length - LOG_PREVIEW_COUNT);
  const visibleLogs = showAllLogs || hiddenCount === 0 ? logs : logs.slice(-LOG_PREVIEW_COUNT);

  return (
    <>
      <p className="muted small">
        状态：{jobStateLabel(job.state)}
        {job.stage ? ` · 当前阶段：${jobStageLabel(job.stage)}` : ''}
      </p>
      {rows.length ? (
        <details className="foldable" open>
          <summary>各阶段耗时</summary>
          <div className="foldContent">
            <dl className="kv">
              {rows.map((row) => (
                <div key={row.key} style={{ display: 'contents' }}>
                  <dt>{row.label}</dt>
                  <dd>{row.value} <span className="muted small">· {row.source}</span></dd>
                </div>
              ))}
            </dl>
            <p className="muted small">单位是秒，按服务端墙钟记录。各阶段独立计时，可能重叠，相加不是工作流发布又等待的时间。</p>
          </div>
        </details>
      ) : null}
      {logs.length ? (
        <details className="foldable" open={logs.length <= LOG_PREVIEW_COUNT}>
          <summary>持久任务日志（{logs.length} 条）</summary>
          <div className="foldContent">
            {hiddenCount > 0 && (
              <button
                type="button"
                className="button ghost small"
                onClick={() => setShowAllLogs((v) => !v)}
              >
                {showAllLogs ? `收起早期日志，仅看最新 ${LOG_PREVIEW_COUNT} 条` : `展开更早的 ${hiddenCount} 条日志`}
              </button>
            )}
            {visibleLogs.map((log, index) => (
              <p key={log.id || index} className="muted small">
                {log.created_at ? `${log.created_at} · ` : ''}{log.message || (log.state ? jobStateLabel(log.state) : '状态事件')}
              </p>
            ))}
          </div>
        </details>
      ) : null}
    </>
  );
}
