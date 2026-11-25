import { Fragment } from 'react';
import { ArrowUpRight, X } from 'lucide-react';
import type { GpuJob, GpuWorkflow } from '../lib/gpu-api';
import { formatInstanceTime } from '../lib/instance-time';
import { jobDetailTimings, paramRows, promptCompositionRows, stageSecondsTotal, taskStatusNote } from '../lib/job-detail-copy';
import { jobErrorMessage, jobErrorRows } from '../lib/job-error-copy';
import { formatAllocationRetryHeadline, formatRetrySpec } from '../lib/allocation-retry-copy';
import { jobStateLabel } from '../lib/job-stage-rows';
import { gpuLabel } from '../lib/gpu-label';
import { shown } from '../lib/create-runtime-prefs';
import { workflowTitle } from '../lib/workflow-copy';
import { JobHistoryDetails } from './JobHistoryDetails';
import styles from '../gpu.module.css';

export function JobDetailInspector({
  detailId,
  detailError,
  inspectedJob,
  workflows = [],
  onOpenOutput,
  onClose,
}: {
  detailId: string;
  detailError: string;
  inspectedJob: GpuJob | null;
  workflows?: GpuWorkflow[];
  onOpenOutput?: (assetId: string) => void;
  onClose: () => void;
}) {
  const wfKey = inspectedJob?.workflow_key || '';
  const wfObj = wfKey
    ? (workflows.find((w) => w.workflow_key === wfKey && w.active) || workflows.find((w) => w.workflow_key === wfKey))
    : undefined;
  const wfName = wfKey ? workflowTitle(wfObj || { workflow_key: wfKey }) : '';
  const outputIds = (inspectedJob?.output_asset_ids || []).filter(Boolean);
  const isVideo = /i2v|video|ref2v|t2v/i.test(wfKey);
  const failureMessage = jobErrorMessage(inspectedJob);
  const failureRows = jobErrorRows(inspectedJob);

  return (
    <>
      <div className={styles.cardHead}>
        <h2>任务详情{detailId && ` · ${detailId.slice(0, 8)}`}</h2>
        {detailId && (
          <button className={styles.iconGhost} type="button" aria-label="关闭任务详情" onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        )}
      </div>
      {detailId && detailError && <p className="status error" role="alert">{detailError}</p>}
      {inspectedJob ? (
        <>
          <dl className={styles.kv}>
            {wfName && (
              <>
                <dt>工作流</dt>
                <dd title={wfKey}>
                  {wfName}
                  {inspectedJob.workflow_version != null && ` · v${inspectedJob.workflow_version}`}
                </dd>
              </>
            )}
            <dt>状态</dt><dd>{jobStateLabel(inspectedJob.state)}</dd>
            <dt>运行方式</dt>
            <dd>{inspectedJob.provider === 'mock' ? '离线模拟' : inspectedJob.provider === 'colab' ? 'Colab GPU' : shown(inspectedJob.provider)}</dd>
            <dt>GPU</dt><dd>{gpuLabel(inspectedJob)}</dd>
            <dt>创建时间</dt><dd>{formatInstanceTime(inspectedJob.created_at)}</dd>
          </dl>
          {inspectedJob.allocation_retry && (
            <details className="foldable" open={Boolean(inspectedJob.allocation_retry.active || ['exhausted', 'aborted'].includes(inspectedJob.allocation_retry.phase || ''))}>
              <summary>
                自动排队抢卡（已尝试 {inspectedJob.allocation_retry.attempt ?? 1} 次 · 重试 {inspectedJob.allocation_retry.retries_used ?? 0}/{inspectedJob.allocation_retry.max_retries ?? 20}）
              </summary>
              <div className="foldContent">
                <p className={inspectedJob.allocation_retry.active ? styles.retryInlineNote : 'muted small'}>
                  {formatAllocationRetryHeadline(inspectedJob.allocation_retry)}
                </p>
                {inspectedJob.allocation_retry.candidate_specs && inspectedJob.allocation_retry.candidate_specs.length > 0 && (
                  <p className="muted small">
                    候选显卡顺序：{inspectedJob.allocation_retry.candidate_specs.map((s) => formatRetrySpec(s)).join(' → ')}
                  </p>
                )}
                {inspectedJob.allocation_retry.excluded_accounts && Object.keys(inspectedJob.allocation_retry.excluded_accounts).length > 0 && (
                  <p className="muted small">
                    已移出账号：{Object.values(inspectedJob.allocation_retry.excluded_accounts).map((item) => `${item.alias || '账号'}（${item.code || '异常'}）`).join('、')}
                  </p>
                )}
                {Array.isArray(inspectedJob.allocation_retry.history) && inspectedJob.allocation_retry.history.length > 0 && (
                  <ul className={styles.retryHistoryList}>
                    {inspectedJob.allocation_retry.history.map((entry, idx) => {
                      const timeText = entry.at ? new Date(entry.at * 1000).toLocaleTimeString([], { hour12: false }) : '';
                      return (
                        <li key={`${entry.attempt}-${idx}`} className={styles.retryHistoryItem}>
                          <div className={styles.retryHistoryHead}>
                            <span className={styles.retryHistoryTitle}>
                              第 {entry.attempt} 次 · {entry.account_alias || entry.account_id || '账号'} · {formatRetrySpec(entry.spec)}
                            </span>
                            {timeText && <span className={styles.activityLogTime}>{timeText}</span>}
                          </div>
                          <p className={styles.retryHistoryDetail}>
                            {entry.error_message || entry.error_code || '分配失败'}
                          </p>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            </details>
          )}
          {(failureMessage || failureRows.length > 0) && (
            <details className="foldable" open>
              <summary>失败诊断</summary>
              <div className="foldContent">
                {failureMessage && <p className="status error" role="alert">{failureMessage}</p>}
                {failureRows.length > 0 && (
                  <dl className={styles.kv}>
                    {failureRows.map((row) => (
                      <Fragment key={row.key}>
                        <dt>{row.label}</dt>
                        <dd>{row.mono ? <code>{row.value}</code> : row.value}</dd>
                      </Fragment>
                    ))}
                  </dl>
                )}
              </div>
            </details>
          )}
          {outputIds.length > 0 && onOpenOutput && (
            <div className={styles.outputStrip} aria-label="任务产出作品">
              {outputIds.slice(0, 4).map((assetId, idx) => (
                <button
                  key={assetId}
                  type="button"
                  className={styles.outputThumbBtn}
                  onClick={() => onOpenOutput(assetId)}
                >
                  {!isVideo && <img src={`/api/assets/${assetId}/thumbnail`} alt="" loading="lazy" />}
                  <span>查看作品{outputIds.length > 1 ? ` #${idx + 1}` : ''}</span>
                  <ArrowUpRight size={13} aria-hidden />
                </button>
              ))}
            </div>
          )}
          <p className="muted small">{taskStatusNote(inspectedJob)}</p>
          {Array.isArray(inspectedJob.attempts) && (
            <details className="foldable" open>
              <summary>执行记录</summary>
              <div className="foldContent">
                {inspectedJob.attempts.length ? inspectedJob.attempts.map((attempt: Record<string, unknown>) => (
                  <p key={String(attempt.id)}>
                    第 {String(attempt.attempt)} 次 · {jobStateLabel(String(attempt.state))}
                    {attempt.error_code ? ` · ${String(attempt.error_code)}` : ''}
                  </p>
                )) : <p className="muted">还没有开始执行</p>}
              </div>
            </details>
          )}
          {(inspectedJob.timings || inspectedJob.timing_details || inspectedJob.logs || inspectedJob.events) ? (
            <JobHistoryDetails job={inspectedJob} showErrorDetails={false} />
          ) : null}
          {stageSecondsTotal(jobDetailTimings(inspectedJob)) ? (
            <p className="muted small">
              各阶段相加约 {stageSecondsTotal(jobDetailTimings(inspectedJob))}。阶段按秒独立计时，可能重叠，不是工作流发布又额外等了这么久。
            </p>
          ) : null}
          {inspectedJob.resolved_params ? (
            <details className="foldable">
              <summary>本次使用的参数</summary>
              <div className="foldContent">
                <dl className={styles.kv}>
                  {promptCompositionRows(inspectedJob.prompt_composition).map((row) => (
                    <Fragment key={row.key}><dt>{row.label}</dt><dd>{row.value}</dd></Fragment>
                  ))}
                  {paramRows(inspectedJob.resolved_params).map((row) => (
                    <Fragment key={row.key}><dt>{row.label}</dt><dd>{row.value}</dd></Fragment>
                  ))}
                </dl>
              </div>
            </details>
          ) : null}
        </>
      ) : (
        <p className="muted small">
          {detailId ? (detailError ? '可重新选择任务以加载详情。' : '正在读取任务详情…') : '选择任务查看执行阶段、日志和生成参数。'}
        </p>
      )}
    </>
  );
}
