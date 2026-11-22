import { Fragment } from 'react';
import { ArrowUpRight, X } from 'lucide-react';
import type { GpuJob, GpuWorkflow } from '../lib/gpu-api';
import { formatInstanceTime } from '../lib/instance-time';
import { jobDetailTimings, paramRows, stageSecondsTotal, taskStatusNote } from '../lib/job-detail-copy';
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
            <JobHistoryDetails job={inspectedJob} />
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
