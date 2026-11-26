import type { GpuInstance } from '../components/AccountDirectory';
import { formatAllocationRetryHeadline, formatRetrySpec } from './allocation-retry-copy';
import { domainBadgeLabel, instanceIdOfCard } from './domain-transfer';
import type { GpuJob, GpuSettings, GpuUsage, GpuWorkflow } from './gpu-api';
import { jobStageLabel, jobStateLabel } from './job-stage-rows';
import { isSystemJob } from './project-label';
import { cuNumber, formatDateTime, progressPercent, readableState, terminalStates } from './ui-helpers';
import { workflowTitle } from './workflow-copy';

const ACTIVE_INSTANCE_STATES = new Set([
  'requested',
  'allocating',
  'preparing',
  'ready',
  'busy',
  'retained_idle',
  'transferring',
  'releasing',
]);

const DONE_JOB_STATES = new Set(['completed', 'succeeded']);
const FAILED_JOB_STATES = new Set(['failed', 'failed_final', 'cancelled', 'cancelled_local', 'blocked']);

export type GpuStatusTone = 'idle' | 'live' | 'retrying' | 'retained' | 'warning' | 'danger' | 'offline';

export type GpuStatusPhase =
  | 'offline'
  | 'guard_tripped'
  | 'retrying'
  | 'running'
  | 'transferring'
  | 'native_starting'
  | 'retained'
  | 'native_online'
  | 'online_idle'
  | 'idle';

export type GpuStatusActionTarget =
  | { kind: 'compute'; label: string }
  | { kind: 'job'; jobId: string; label: string }
  | { kind: 'jobs'; label: string }
  | { kind: 'native'; label: string }
  | { kind: 'gallery'; label: string };

export interface GpuStatusRow {
  key: string;
  label: string;
  value: string;
  tone?: 'default' | 'accent' | 'warning' | 'danger' | 'muted';
  title?: string;
}

export interface GpuOnlineInstanceItem {
  id: string;
  gpu: string;
  pool: string;
  stateText: string;
  countdown?: string;
  account?: string;
}

export interface GpuStatusSummary {
  phase: GpuStatusPhase;
  tone: GpuStatusTone;
  pulsing: boolean;
  pillLabel: string;
  pillTitle: string;
  mobileBadge?: string;
  panelTitle: string;
  bannerText: string;
  bannerSub?: string;
  progress: {
    percent: number | null;
    indeterminate: boolean;
    active: boolean;
  };
  rows: GpuStatusRow[];
  onlineInstances: GpuOnlineInstanceItem[];
  primaryAction: GpuStatusActionTarget;
  secondaryAction: GpuStatusActionTarget;
  footerNote: string;
  needsCountdownTick: boolean;
}

export interface GpuStatusInput {
  settings: GpuSettings | null;
  instances: GpuInstance[];
  jobs: GpuJob[];
  usage: GpuUsage | null;
  workflows: GpuWorkflow[];
  nowSec?: number;
}

function toEpochSeconds(value: string | number | null | undefined): number | null {
  if (value == null || value === '') return null;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null;
    return value > 1e12 ? value / 1000 : value;
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric) && String(value).trim() !== '') {
    return numeric > 1e12 ? numeric / 1000 : numeric;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed / 1000 : null;
}

export function shortGpuName(raw?: string | null): string {
  const text = String(raw || '').trim();
  if (!text) return 'GPU';
  const upper = text.toUpperCase();
  if (upper === 'CPU' || upper === 'NONE') return 'CPU';
  const match = upper.match(/\b(H100|A100|L4|T4|V100|G4|P100|K80|V5E1|V6E1)\b/);
  if (match) return match[1];
  const cleaned = text.replace(/^(NVIDIA|Tesla|Google)\s+/i, '').trim();
  if (cleaned.length <= 12) return cleaned || 'GPU';
  return cleaned.split(/\s+/)[0] || 'GPU';
}

export function formatCountdownLong(deadline: string | number | null | undefined, nowSec: number): string | null {
  const epoch = toEpochSeconds(deadline);
  if (epoch == null) return null;
  const rem = Math.max(0, Math.ceil(epoch - nowSec));
  if (rem <= 0) return '即将到期';
  const mins = Math.floor(rem / 60);
  const secs = rem % 60;
  if (mins >= 60) {
    const hours = Math.floor(mins / 60);
    const remMins = mins % 60;
    return remMins > 0 ? `${hours} 小时 ${remMins} 分` : `${hours} 小时`;
  }
  return mins > 0 ? `${mins} 分 ${secs} 秒` : `${secs} 秒`;
}

export function formatCountdownCompact(deadline: string | number | null | undefined, nowSec: number): string | null {
  const epoch = toEpochSeconds(deadline);
  if (epoch == null) return null;
  const rem = Math.max(0, Math.ceil(epoch - nowSec));
  if (rem <= 0) return '<1m';
  if (rem < 60) return `${rem}s`;
  const mins = Math.ceil(rem / 60);
  return `${mins}m`;
}

function isNativeJob(job: GpuJob): boolean {
  const wf = String(job.workflow_key || '').toLowerCase();
  const pid = String(job.project_id || '').toLowerCase();
  return wf === 'system/native-comfy' || wf.includes('native') || pid.startsWith('native-comfy-');
}

function isNativeInstance(instance: GpuInstance): boolean {
  return String(instance.instance_pool || '').toLowerCase() === 'native' || instance.native_comfy === true;
}

function isTransferringInstance(instance: GpuInstance): boolean {
  const state = String(instance.state || '').toLowerCase();
  if (state === 'transferring') return true;
  return Boolean(instance.transfer_state && !instance.transfer_state.completed_at && !instance.transfer_state.error_code);
}

function jobTimestamp(job: GpuJob): number {
  return toEpochSeconds(job.created_at) ?? toEpochSeconds(job.updated_at) ?? 0;
}

function jobGpuFullName(job: GpuJob, instancesById: Map<string, GpuInstance>): string {
  if (job.gpu_reported) return job.gpu_reported;
  const inst = job.instance_id ? instancesById.get(job.instance_id) : undefined;
  if (inst?.gpu_reported) return inst.gpu_reported;
  const spec = job.runtime_spec_requested;
  if (spec?.accelerator && spec.accelerator !== 'NONE') return spec.accelerator;
  if (inst?.runtime_spec_requested?.accelerator && inst.runtime_spec_requested.accelerator !== 'NONE') {
    return inst.runtime_spec_requested.accelerator;
  }
  return '';
}

function instanceGpuFullName(instance: GpuInstance): string {
  if (instance.gpu_reported) return instance.gpu_reported;
  const req = instance.runtime_spec_requested;
  if (req?.accelerator && req.accelerator !== 'NONE') return req.accelerator;
  return 'Colab GPU';
}

function resolveWorkflowLabel(job: GpuJob | undefined, workflows: GpuWorkflow[]): string {
  if (!job) return '';
  if (isNativeJob(job)) return '原生 ComfyUI';
  if (job.workflow_key === 'system/cuda-smoke') return 'GPU 可行性冒烟检查';
  if (job.prep) return '环境 / 模型预热';
  const key = job.workflow_key || '';
  if (!key) return '';
  const match = workflows.find((w) => w.workflow_key === key && String(w.workflow_version) === String(job.workflow_version))
    || workflows.find((w) => w.workflow_key === key && w.active)
    || workflows.find((w) => w.workflow_key === key);
  const title = workflowTitle(match || { workflow_key: key });
  return job.workflow_version != null && !title.includes(`v${job.workflow_version}`)
    ? `${title} v${job.workflow_version}`
    : title;
}

interface BatchSnapshot {
  isBatch: boolean;
  total: number;
  active: number;
  done: number;
  failed: number;
  jobs: GpuJob[];
  leadJob: GpuJob;
}

function buildBatchSnapshot(leadJob: GpuJob, allJobs: GpuJob[]): BatchSnapshot {
  const grouped = leadJob.batch_id
    ? allJobs.filter((item) => item.batch_id === leadJob.batch_id && !isSystemJob(item) && !item.prep)
    : [leadJob];
  const list = grouped.length >= 2 ? grouped : [leadJob];
  let active = 0;
  let done = 0;
  let failed = 0;
  for (const item of list) {
    const state = String(item.state || 'queued').toLowerCase();
    if (!terminalStates.has(state)) active += 1;
    else if (DONE_JOB_STATES.has(state)) done += 1;
    else if (FAILED_JOB_STATES.has(state)) failed += 1;
  }
  return {
    isBatch: list.length >= 2,
    total: list.length,
    active,
    done,
    failed,
    jobs: list,
    leadJob,
  };
}

function formatColabGpuRow(colabInstances: GpuInstance[], usage: GpuUsage | null, nowSec: number, offline: boolean): GpuStatusRow {
  if (offline) {
    return { key: 'gpu', label: '运行环境', value: '离线演示（0 消耗）', tone: 'muted' };
  }
  if (colabInstances.length === 0) {
    const activeRuntimes = Number(usage?.active_runtimes ?? 0);
    if (activeRuntimes > 0) {
      return { key: 'gpu', label: 'Colab 显卡', value: `${activeRuntimes} 台运行中`, tone: 'accent' };
    }
    return { key: 'gpu', label: 'Colab 显卡', value: '已释放（0 消耗）', tone: 'muted' };
  }
  if (colabInstances.length === 1) {
    const inst = colabInstances[0];
    const state = String(inst.state || '').toLowerCase();
    const gpuShort = shortGpuName(instanceGpuFullName(inst));
    const pool = domainBadgeLabel(inst);
    if (state === 'retained_idle') {
      const compact = formatCountdownCompact(inst.idle_deadline_at || inst.retained_until, nowSec);
      return {
        key: 'gpu',
        label: 'Colab 显卡',
        value: `1 台保留中（${gpuShort}${compact ? ` · 剩 ${compact}` : ''}）`,
        tone: 'warning',
        title: `${instanceGpuFullName(inst)} · ${pool}`,
      };
    }
    if (state === 'releasing') {
      return {
        key: 'gpu',
        label: 'Colab 显卡',
        value: `正在释放（${gpuShort}）`,
        tone: 'muted',
      };
    }
    if (isTransferringInstance(inst)) {
      return {
        key: 'gpu',
        label: 'Colab 显卡',
        value: `1 台域转移中（${gpuShort}）`,
        tone: 'accent',
      };
    }
    return {
      key: 'gpu',
      label: 'Colab 显卡',
      value: `1 台运行中（${gpuShort} · ${pool}）`,
      tone: 'accent',
      title: `${instanceGpuFullName(inst)} · ${pool}`,
    };
  }
  const retainedCount = colabInstances.filter((i) => String(i.state || '').toLowerCase() === 'retained_idle').length;
  const activeCount = colabInstances.length - retainedCount;
  const detail = retainedCount > 0 && activeCount > 0
    ? `${colabInstances.length} 台在线（${activeCount} 运行 / ${retainedCount} 保留）`
    : retainedCount === colabInstances.length
      ? `${retainedCount} 台保留中`
      : `${colabInstances.length} 台运行中`;
  return {
    key: 'gpu',
    label: 'Colab 显卡',
    value: detail,
    tone: retainedCount === colabInstances.length ? 'warning' : 'accent',
  };
}

function formatRecentBatchRow(snapshot: BatchSnapshot | null, isActive: boolean): GpuStatusRow | null {
  if (!snapshot) return null;
  const label = isActive
    ? (snapshot.isBatch ? '当前批次' : '当前任务')
    : (snapshot.isBatch ? '最近一批' : '最近任务');
  if (snapshot.isBatch) {
    if (snapshot.active > 0) {
      return {
        key: 'batch',
        label,
        value: `进行中 ${snapshot.done}/${snapshot.total}${snapshot.failed > 0 ? `（失败 ${snapshot.failed}）` : ''}`,
        tone: 'accent',
      };
    }
    if (snapshot.failed === 0) {
      return {
        key: 'batch',
        label,
        value: `已完成 ${snapshot.done}/${snapshot.total}`,
        tone: 'default',
      };
    }
    if (snapshot.done === 0) {
      return {
        key: 'batch',
        label,
        value: `失败 ${snapshot.failed}/${snapshot.total}`,
        tone: 'danger',
      };
    }
    return {
      key: 'batch',
      label,
      value: `完成 ${snapshot.done}/${snapshot.total} · 失败 ${snapshot.failed}`,
      tone: 'warning',
    };
  }
  const job = snapshot.leadJob;
  const state = String(job.state || '').toLowerCase();
  const stageText = job.stage ? jobStageLabel(job.stage) : jobStateLabel(job.state);
  if (!terminalStates.has(state)) {
    const pct = progressPercent(job.progress);
    return {
      key: 'batch',
      label,
      value: pct != null && pct > 0 ? `${stageText} · ${pct}%` : stageText,
      tone: 'accent',
    };
  }
  const timeText = job.created_at ? formatDateTime(job.created_at) : '';
  return {
    key: 'batch',
    label,
    value: `${jobStateLabel(job.state)}${timeText && timeText !== '未知时间' ? ` · ${timeText}` : ''}`,
    tone: FAILED_JOB_STATES.has(state) ? 'danger' : 'default',
  };
}

function formatCuRow(usage: GpuUsage | null, offline: boolean): GpuStatusRow | null {
  if (offline || !usage) return null;
  const today = usage.today_actual_cu != null ? `${cuNumber(usage.today_actual_cu)} CU` : null;
  const balance = usage.ccu?.total_balance != null ? `余 ${cuNumber(usage.ccu.total_balance)} CU` : null;
  const guardStatus = usage.guard?.status || 'ok';
  const guardTag = guardStatus === 'hard_balance_floor'
    ? ' · 余额熔断'
    : guardStatus === 'soft_daily_limit'
      ? ' · 达日限'
      : '';
  const parts = [today, balance].filter(Boolean);
  if (parts.length === 0 && !guardTag) return null;
  return {
    key: 'cu',
    label: '今日算力',
    value: `${parts.join(' · ') || '已开启守卫'}${guardTag}`,
    tone: guardStatus === 'hard_balance_floor'
      ? 'danger'
      : guardStatus === 'soft_daily_limit'
        ? 'warning'
        : 'default',
  };
}

function buildOnlineInstanceItems(colabInstances: GpuInstance[], nowSec: number): GpuOnlineInstanceItem[] {
  if (colabInstances.length < 2) return [];
  return colabInstances.map((inst) => {
    const state = String(inst.state || '').toLowerCase();
    const countdown = state === 'retained_idle'
      ? (formatCountdownCompact(inst.idle_deadline_at || inst.retained_until, nowSec) || undefined)
      : undefined;
    return {
      id: instanceIdOfCard(inst) || `${inst.gpu_reported || 'gpu'}-${inst.instance_pool || 'pool'}`,
      gpu: shortGpuName(instanceGpuFullName(inst)),
      pool: domainBadgeLabel(inst),
      stateText: readableState(inst.state),
      countdown,
      account: inst.account_alias_snapshot || undefined,
    };
  });
}

function retentionProgressPercent(inst: GpuInstance, nowSec: number): number | null {
  const deadline = toEpochSeconds(inst.idle_deadline_at || inst.retained_until);
  const retainMins = inst.retain_idle_minutes;
  if (deadline == null || !retainMins || retainMins <= 0) return null;
  const totalSec = retainMins * 60;
  const rem = Math.max(0, deadline - nowSec);
  return Math.max(0, Math.min(100, Math.round((rem / totalSec) * 100)));
}

export function buildGpuStatusSummary(input: GpuStatusInput): GpuStatusSummary {
  const { settings, instances, jobs, usage, workflows } = input;
  const nowSec = input.nowSec ?? Math.floor(Date.now() / 1000);
  const offline = settings?.offline_mode === true;
  const footerNote = '状态自动同步，切换页面不影响后台任务运行。';

  const instancesById = new Map<string, GpuInstance>();
  for (const inst of instances) {
    const id = instanceIdOfCard(inst);
    if (id) instancesById.set(id, inst);
  }

  const colabInstances = instances.filter(
    (inst) => inst.provider === 'colab'
      && !inst.runtime_deleted
      && ACTIVE_INSTANCE_STATES.has(String(inst.state || '').toLowerCase()),
  );
  const onlineInstances = buildOnlineInstanceItems(colabInstances, nowSec);

  const sortedJobs = [...jobs].sort((a, b) => jobTimestamp(b) - jobTimestamp(a));
  const activeJobs = sortedJobs.filter((job) => !terminalStates.has(String(job.state || '').toLowerCase()));
  const activeCreationJobs = activeJobs.filter((job) => !isSystemJob(job) && !job.prep);
  const activeSystemJobs = activeJobs.filter((job) => isSystemJob(job) || Boolean(job.prep));
  const latestCreationJob = sortedJobs.find((job) => !isSystemJob(job) && !job.prep);
  const recentCreationBatch = latestCreationJob ? buildBatchSnapshot(latestCreationJob, sortedJobs) : null;

  const gpuRow = formatColabGpuRow(colabInstances, usage, nowSec, offline);
  const cuRow = formatCuRow(usage, offline);

  const buildRows = (
    batchSnapshot: BatchSnapshot | null,
    batchActive: boolean,
    workflowJob: GpuJob | undefined,
    extraRow?: GpuStatusRow | null,
  ): GpuStatusRow[] => {
    const list: GpuStatusRow[] = [gpuRow];
    if (extraRow) list.push(extraRow);
    const batchRow = formatRecentBatchRow(batchSnapshot, batchActive);
    if (batchRow) list.push(batchRow);
    const wfName = resolveWorkflowLabel(workflowJob, workflows);
    if (wfName) {
      list.push({
        key: 'workflow',
        label: batchActive ? '当前工作流' : '最近工作流',
        value: wfName,
        title: wfName,
      });
    }
    if (cuRow) list.push(cuRow);
    return list;
  };

  // 1. Active allocation retry (creation preferred over native/system)
  const retryingCreation = activeCreationJobs.find((job) => job.allocation_retry?.active);
  const retryingAny = retryingCreation || activeJobs.find((job) => job.allocation_retry?.active);
  if (!offline && retryingAny && retryingAny.allocation_retry) {
    const retry = retryingAny.allocation_retry;
    const isNative = isNativeJob(retryingAny);
    const used = retry.retries_used ?? 0;
    const max = retry.max_retries ?? 20;
    const targetSpecRaw = retry.phase === 'waiting' ? (retry.next_spec || retry.current_spec) : (retry.current_spec || retry.next_spec);
    const shortSpec = shortGpuName(formatRetrySpec(targetSpecRaw));
    const pillLabel = retry.phase === 'reconciling'
      ? '核验旧租约中…'
      : used > 0
        ? `排队抢 ${shortSpec} (${used}/${max})`
        : `申请 ${shortSpec} 中…`;
    const bannerText = formatAllocationRetryHeadline(retry, nowSec) || `正在自动排队抢 ${shortSpec} 显卡…`;

    let progressPercentValue: number | null = null;
    let indeterminate = true;
    if (retry.phase === 'waiting' && retry.next_retry_at && retry.interval_seconds && retry.interval_seconds > 0) {
      const rem = Math.max(0, retry.next_retry_at - nowSec);
      const elapsedRatio = 1 - Math.min(1, rem / retry.interval_seconds);
      progressPercentValue = Math.round(elapsedRatio * 100);
      indeterminate = false;
    }

    const activeBatch = !isNative ? buildBatchSnapshot(retryingAny, sortedJobs) : recentCreationBatch;
    const rows = isNative
      ? buildRows(recentCreationBatch, false, latestCreationJob, {
          key: 'target',
          label: '目标环境',
          value: '原生 ComfyUI',
          tone: 'accent',
        })
      : buildRows(activeBatch, true, retryingAny);

    return {
      phase: 'retrying',
      tone: 'retrying',
      pulsing: true,
      pillLabel,
      pillTitle: bannerText,
      mobileBadge: used > 0 ? `${used}` : undefined,
      panelTitle: '算力与任务实时状态',
      bannerText,
      bannerSub: retry.last_error_message ? `上次反馈：${retry.last_error_message}` : undefined,
      progress: {
        percent: progressPercentValue,
        indeterminate,
        active: true,
      },
      rows,
      onlineInstances,
      primaryAction: isNative
        ? { kind: 'native', label: '查看原生状态' }
        : { kind: 'job', jobId: retryingAny.id, label: '查看抢卡日志' },
      secondaryAction: { kind: 'compute', label: '算力中心' },
      footerNote,
      needsCountdownTick: retry.phase === 'waiting' && Boolean(retry.next_retry_at),
    };
  }

  // 2. Active creation job(s) (Colab or mock)
  if (activeCreationJobs.length > 0) {
    const runningStates = new Set(['running', 'uploading', 'loading', 'preparing', 'allocating']);
    const leadJob = activeCreationJobs.find((job) => runningStates.has(String(job.state || '').toLowerCase()))
      || activeCreationJobs[0];
    const batch = buildBatchSnapshot(leadJob, sortedJobs);
    const isMock = leadJob.provider === 'mock' || offline;
    const gpuFull = isMock ? '本地模拟' : jobGpuFullName(leadJob, instancesById);
    const gpuShort = isMock ? '模拟' : (gpuFull ? shortGpuName(gpuFull) : '');
    const stageText = leadJob.stage ? jobStageLabel(leadJob.stage) : jobStateLabel(leadJob.state);
    const pct = progressPercent(leadJob.progress);
    const finishedCount = batch.done + batch.failed;
    const currentIndex = Math.min(batch.total, finishedCount + 1);

    const pillLabel = isMock
      ? (batch.isBatch ? `模拟中 ${batch.done}/${batch.total}` : (pct != null && pct > 0 ? `模拟中 ${pct}%` : '模拟生成中'))
      : batch.isBatch
        ? `生成中 ${batch.done}/${batch.total}`
        : pct != null && pct > 0
          ? `${stageText} ${pct}%`
          : gpuShort
            ? `${gpuShort} · ${stageText}`
            : stageText;

    const bannerText = isMock
      ? `正在离线模拟【${stageText}】${batch.isBatch ? `（第 ${currentIndex}/${batch.total} 个）` : ''}${pct != null && pct > 0 ? ` · ${pct}%` : ''}`
      : gpuFull
        ? `已分配 ${gpuFull}，正在【${stageText}】${batch.isBatch ? `（第 ${currentIndex}/${batch.total} 个）` : ''}${pct != null && pct > 0 ? ` · ${pct}%` : ''}`
        : `正在【${stageText}】${batch.isBatch ? `（第 ${currentIndex}/${batch.total} 个）` : ''}${pct != null && pct > 0 ? ` · ${pct}%` : ''}`;

    let barPercent: number | null = pct;
    let barIndeterminate = pct == null || pct <= 0;
    if (batch.isBatch) {
      const combined = ((finishedCount + (typeof leadJob.progress === 'number' ? Math.max(0, Math.min(1, leadJob.progress)) : 0)) / batch.total) * 100;
      if (combined > 0) {
        barPercent = Math.min(100, Math.round(combined));
        barIndeterminate = false;
      } else {
        barPercent = null;
        barIndeterminate = true;
      }
    }

    const hasRetainedElsewhere = colabInstances.some((i) => String(i.state || '').toLowerCase() === 'retained_idle');

    return {
      phase: 'running',
      tone: isMock ? 'offline' : 'live',
      pulsing: true,
      pillLabel,
      pillTitle: bannerText,
      mobileBadge: batch.isBatch ? `${batch.done}/${batch.total}` : (activeCreationJobs.length > 1 ? String(activeCreationJobs.length) : undefined),
      panelTitle: '算力与任务实时状态',
      bannerText,
      bannerSub: leadJob.account_alias_snapshot ? `执行账号：${leadJob.account_alias_snapshot}` : undefined,
      progress: {
        percent: barPercent,
        indeterminate: barIndeterminate,
        active: true,
      },
      rows: buildRows(batch, true, leadJob),
      onlineInstances,
      primaryAction: { kind: 'job', jobId: leadJob.id, label: '查看任务日志' },
      secondaryAction: { kind: 'compute', label: '算力中心' },
      footerNote,
      needsCountdownTick: hasRetainedElsewhere,
    };
  }

  // 3. Domain transferring instance
  const transferringInstance = !offline ? colabInstances.find(isTransferringInstance) : undefined;
  if (transferringInstance) {
    const gpuFull = instanceGpuFullName(transferringInstance);
    const gpuShort = shortGpuName(gpuFull);
    const toPool = transferringInstance.transfer_state?.to_pool;
    const toLabel = toPool === 'native' ? '原生 ComfyUI' : toPool === 'creation' ? '创作池' : '目标工作域';
    const bannerText = `${gpuFull} 正在执行域转移（切往${toLabel}），期间保持实例在线。`;
    return {
      phase: 'transferring',
      tone: 'live',
      pulsing: true,
      pillLabel: `域转移 · ${gpuShort}`,
      pillTitle: bannerText,
      panelTitle: '算力与任务实时状态',
      bannerText,
      progress: { percent: null, indeterminate: true, active: true },
      rows: buildRows(recentCreationBatch, false, latestCreationJob),
      onlineInstances,
      primaryAction: toPool === 'native'
        ? { kind: 'native', label: '打开原生 ComfyUI' }
        : { kind: 'compute', label: '查看转移进度' },
      secondaryAction: toPool === 'native'
        ? { kind: 'compute', label: '算力中心' }
        : { kind: 'jobs', label: '查看任务队列' },
      footerNote,
      needsCountdownTick: false,
    };
  }

  // 4. Native ComfyUI starting or other active system/prep job
  const activeNativeJob = !offline ? activeSystemJobs.find(isNativeJob) : undefined;
  const startingNativeInstance = !offline
    ? colabInstances.find((i) => isNativeInstance(i) && ['requested', 'allocating', 'preparing'].includes(String(i.state || '').toLowerCase()))
    : undefined;
  if (!offline && (activeNativeJob || startingNativeInstance)) {
    const stageText = activeNativeJob?.stage
      ? jobStageLabel(activeNativeJob.stage)
      : activeNativeJob
        ? jobStateLabel(activeNativeJob.state)
        : readableState(startingNativeInstance?.state);
    const gpuFull = activeNativeJob
      ? jobGpuFullName(activeNativeJob, instancesById)
      : (startingNativeInstance ? instanceGpuFullName(startingNativeInstance) : '');
    const pct = progressPercent(activeNativeJob?.progress);
    const bannerText = `原生 ComfyUI 正在启动：${stageText}${gpuFull ? `（${gpuFull}）` : ''}${pct != null && pct > 0 ? ` · ${pct}%` : ''}`;
    return {
      phase: 'native_starting',
      tone: 'live',
      pulsing: true,
      pillLabel: gpuFull ? `原生启动 · ${shortGpuName(gpuFull)}` : '原生启动中…',
      pillTitle: bannerText,
      panelTitle: '算力与任务实时状态',
      bannerText,
      progress: {
        percent: pct != null && pct > 0 ? pct : null,
        indeterminate: pct == null || pct <= 0,
        active: true,
      },
      rows: buildRows(recentCreationBatch, false, latestCreationJob, {
        key: 'native_stage',
        label: '原生状态',
        value: stageText,
        tone: 'accent',
      }),
      onlineInstances,
      primaryAction: { kind: 'native', label: '打开原生 ComfyUI' },
      secondaryAction: activeNativeJob
        ? { kind: 'job', jobId: activeNativeJob.id, label: '查看启动日志' }
        : { kind: 'compute', label: '算力中心' },
      footerNote,
      needsCountdownTick: false,
    };
  }

  const otherSystemJob = !offline ? activeSystemJobs[0] : undefined;
  if (otherSystemJob) {
    const stageText = otherSystemJob.stage ? jobStageLabel(otherSystemJob.stage) : jobStateLabel(otherSystemJob.state);
    const label = resolveWorkflowLabel(otherSystemJob, workflows) || '系统任务';
    const pct = progressPercent(otherSystemJob.progress);
    const bannerText = `正在执行${label}：${stageText}${pct != null && pct > 0 ? ` · ${pct}%` : ''}`;
    return {
      phase: 'running',
      tone: 'live',
      pulsing: true,
      pillLabel: `${stageText}…`,
      pillTitle: bannerText,
      panelTitle: '算力与任务实时状态',
      bannerText,
      progress: {
        percent: pct != null && pct > 0 ? pct : null,
        indeterminate: pct == null || pct <= 0,
        active: true,
      },
      rows: buildRows(recentCreationBatch, false, otherSystemJob),
      onlineInstances,
      primaryAction: { kind: 'job', jobId: otherSystemJob.id, label: '查看任务日志' },
      secondaryAction: { kind: 'compute', label: '算力中心' },
      footerNote,
      needsCountdownTick: false,
    };
  }

  // 5. Offline demo mode (idle)
  if (offline) {
    return {
      phase: 'offline',
      tone: 'offline',
      pulsing: false,
      pillLabel: '离线演示',
      pillTitle: '当前为离线演示模式，不连接 Colab 也不消耗 CU',
      panelTitle: '算力与任务实时状态',
      bannerText: '当前处于离线演示模式，所有任务在本地模拟运行，不消耗 Colab CU。',
      progress: { percent: null, indeterminate: false, active: false },
      rows: buildRows(recentCreationBatch, false, latestCreationJob),
      onlineInstances: [],
      primaryAction: { kind: 'jobs', label: '查看任务队列' },
      secondaryAction: { kind: 'compute', label: '算力中心' },
      footerNote,
      needsCountdownTick: false,
    };
  }

  // 6. Online Colab instance(s) without active studio jobs
  if (colabInstances.length > 0) {
    const nativeOnline = colabInstances.find((i) => isNativeInstance(i) && ['ready', 'busy', 'retained_idle'].includes(String(i.state || '').toLowerCase()));
    const retainedInst = colabInstances.find((i) => String(i.state || '').toLowerCase() === 'retained_idle');
    const releasingInst = colabInstances.find((i) => String(i.state || '').toLowerCase() === 'releasing');

    if (nativeOnline) {
      const gpuFull = instanceGpuFullName(nativeOnline);
      const gpuShort = shortGpuName(gpuFull);
      const countdownLong = formatCountdownLong(nativeOnline.idle_deadline_at || nativeOnline.retained_until, nowSec);
      const pillLabel = colabInstances.length > 1
        ? `${colabInstances.length} 台 GPU 在线`
        : `原生在线 · ${gpuShort}`;
      const bannerText = `原生 ComfyUI（${gpuFull}）运行中，期间持续消耗 Colab CU${countdownLong ? `（空闲到期剩 ${countdownLong}）` : ''}。`;
      return {
        phase: 'native_online',
        tone: 'live',
        pulsing: false,
        pillLabel,
        pillTitle: bannerText,
        mobileBadge: colabInstances.length > 1 ? String(colabInstances.length) : undefined,
        panelTitle: '算力与任务实时状态',
        bannerText,
        progress: {
          percent: retentionProgressPercent(nativeOnline, nowSec),
          indeterminate: false,
          active: true,
        },
        rows: buildRows(recentCreationBatch, false, latestCreationJob),
        onlineInstances,
        primaryAction: { kind: 'native', label: '打开原生 ComfyUI' },
        secondaryAction: { kind: 'compute', label: '算力中心' },
        footerNote,
        needsCountdownTick: Boolean(nativeOnline.idle_deadline_at || nativeOnline.retained_until),
      };
    }

    if (retainedInst) {
      const gpuFull = instanceGpuFullName(retainedInst);
      const gpuShort = shortGpuName(gpuFull);
      const countdownLong = formatCountdownLong(retainedInst.idle_deadline_at || retainedInst.retained_until, nowSec);
      const countdownShort = formatCountdownCompact(retainedInst.idle_deadline_at || retainedInst.retained_until, nowSec);
      const pillLabel = colabInstances.length > 1
        ? `${colabInstances.length} 台 GPU 保留中`
        : `${gpuShort} 保留中${countdownShort ? ` · ${countdownShort}` : ''}`;
      const bannerText = `${gpuFull} 正在热保留待机${countdownLong ? `（约 ${countdownLong} 后自动释放）` : ''}，期间仍会消耗 Colab CU，可直接复用出图。`;
      return {
        phase: 'retained',
        tone: 'retained',
        pulsing: false,
        pillLabel,
        pillTitle: bannerText,
        mobileBadge: colabInstances.length > 1 ? String(colabInstances.length) : undefined,
        panelTitle: '算力与任务实时状态',
        bannerText,
        progress: {
          percent: retentionProgressPercent(retainedInst, nowSec),
          indeterminate: false,
          active: true,
        },
        rows: buildRows(recentCreationBatch, false, latestCreationJob),
        onlineInstances,
        primaryAction: { kind: 'compute', label: '管理算力与释放' },
        secondaryAction: latestCreationJob
          ? { kind: 'jobs', label: '查看任务队列' }
          : { kind: 'gallery', label: '查看作品' },
        footerNote,
        needsCountdownTick: Boolean(retainedInst.idle_deadline_at || retainedInst.retained_until),
      };
    }

    if (releasingInst && colabInstances.length === 1) {
      const gpuFull = instanceGpuFullName(releasingInst);
      const gpuShort = shortGpuName(gpuFull);
      const bannerText = `正在释放 ${gpuFull} 运行时，释放完成后将停止消耗 Colab CU。`;
      return {
        phase: 'online_idle',
        tone: 'live',
        pulsing: true,
        pillLabel: `释放 ${gpuShort} 中…`,
        pillTitle: bannerText,
        panelTitle: '算力与任务实时状态',
        bannerText,
        progress: { percent: null, indeterminate: true, active: true },
        rows: buildRows(recentCreationBatch, false, latestCreationJob),
        onlineInstances,
        primaryAction: { kind: 'compute', label: '查看算力状态' },
        secondaryAction: { kind: 'jobs', label: '查看任务队列' },
        footerNote,
        needsCountdownTick: false,
      };
    }

    const firstInst = colabInstances[0];
    const gpuFull = instanceGpuFullName(firstInst);
    const gpuShort = shortGpuName(gpuFull);
    const stateText = readableState(firstInst.state);
    const pillLabel = colabInstances.length > 1
      ? `${colabInstances.length} 台 GPU 在线`
      : `${gpuShort} · ${stateText}`;
    const bannerText = `${gpuFull} 当前处于【${stateText}】状态，期间会按 Colab 规则计费。`;
    return {
      phase: 'online_idle',
      tone: 'live',
      pulsing: ['requested', 'allocating', 'preparing', 'busy'].includes(String(firstInst.state || '').toLowerCase()),
      pillLabel,
      pillTitle: bannerText,
      panelTitle: '算力与任务实时状态',
      bannerText,
      progress: {
        percent: null,
        indeterminate: ['requested', 'allocating', 'preparing', 'busy'].includes(String(firstInst.state || '').toLowerCase()),
        active: true,
      },
      rows: buildRows(recentCreationBatch, false, latestCreationJob),
      onlineInstances,
      primaryAction: { kind: 'compute', label: '算力中心' },
      secondaryAction: { kind: 'jobs', label: '查看任务队列' },
      footerNote,
      needsCountdownTick: false,
    };
  }

  // 7. CU Guard tripped while idle
  const guardStatus = usage?.guard?.status;
  if (guardStatus === 'hard_balance_floor' || guardStatus === 'soft_daily_limit') {
    const isHard = guardStatus === 'hard_balance_floor';
    const bannerText = (isHard ? usage?.guard?.message : (usage?.guard?.daily_message || usage?.guard?.message))
      || (isHard ? '账号 CU 余额已低于下限熔断阈值，已阻止新分配。' : '今日 CU 消耗已达到上限，请前往算力页检查或调整预算。');
    return {
      phase: 'guard_tripped',
      tone: isHard ? 'danger' : 'warning',
      pulsing: false,
      pillLabel: isHard ? 'CU 余额熔断' : '今日 CU 达上限',
      pillTitle: bannerText,
      panelTitle: '算力与任务实时状态',
      bannerText,
      progress: { percent: 100, indeterminate: false, active: true },
      rows: buildRows(recentCreationBatch, false, latestCreationJob),
      onlineInstances: [],
      primaryAction: { kind: 'compute', label: '前往算力检查' },
      secondaryAction: { kind: 'jobs', label: '查看任务队列' },
      footerNote,
      needsCountdownTick: false,
    };
  }

  // 8. Fully idle & released (0 CU burn)
  const latestHasOutput = Boolean(latestCreationJob?.output_asset_ids?.length);
  return {
    phase: 'idle',
    tone: 'idle',
    pulsing: false,
    pillLabel: '无活动 GPU',
    pillTitle: '当前无活动 Colab GPU（0 消耗），点击查看实时状态',
    panelTitle: '算力与任务实时状态',
    bannerText: '当前无在线 Colab GPU 实例，所有算力已释放（0 消耗）。',
    progress: { percent: null, indeterminate: false, active: false },
    rows: buildRows(recentCreationBatch, false, latestCreationJob),
    onlineInstances: [],
    primaryAction: latestHasOutput
      ? { kind: 'gallery', label: '查看最近作品' }
      : { kind: 'jobs', label: '查看任务队列' },
    secondaryAction: { kind: 'compute', label: '算力中心' },
    footerNote,
    needsCountdownTick: false,
  };
}
