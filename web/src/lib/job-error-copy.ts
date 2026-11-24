import type { GpuErrorDetails } from './gpu-api';
import { jobStageLabel } from './job-stage-rows';

const PHASE_LABELS: Record<string, string> = {
  auth: '账号鉴权（auth）',
  assign_get: '分配握手（assign_get）',
  assign_post: '申请算力（assign_post）',
  v1beta_create: '创建运行时（v1beta_create）',
  create: '创建运行时（create）',
  reconcile: '远端协调（reconcile）',
  fencing: '分配状态围栏（fencing）',
  prepare: '准备运行时（prepare）',
  load: '加载执行器（load）',
  run: '运行任务（run）',
  upload: '保存产物（upload）',
  cleanup: '释放算力（cleanup）',
};

const CATEGORY_LABELS: Record<string, string> = {
  upstream: '上游服务异常',
  timeout: '请求超时',
  network: '网络连接异常',
  capacity: '上游算力不足',
  quota: '配额或限流受限',
  auth: '账号授权异常',
  protocol: '上游响应异常',
  internal: '内部调度异常',
};

const TIMEOUT_LABELS: Record<string, string> = {
  read: '等待响应超时',
  connect: '建立连接失败/超时',
  write: '发送请求超时',
  pool: '连接池等待超时',
  proxy: '代理连接失败',
  transport: '网络传输中断',
};

const RECONCILIATION_LABELS: Record<string, string> = {
  confirmed_unassigned: '已确认远端未创建实例，租约已释放',
  late_runtime_deleted: '发现远端延迟创建实例，已自动释放',
  pre_post_unallocated: '未发送开机请求，已直接释放',
  explicit_provider_refusal: '上游明确拒绝，已直接释放',
  cleaned: '已确认远端释放',
  confirmed_deleted: '已确认远端释放',
};

export interface JobErrorLike {
  state?: string | null;
  stage?: string | null;
  error?: string | { code?: string | null; message?: string | null } | null;
  error_code?: string | null;
  error_message?: string | null;
  error_details?: GpuErrorDetails | null;
}

export interface JobErrorRow {
  key: string;
  label: string;
  value: string;
  mono?: boolean;
}

export function errorPhaseLabel(phase: string | null | undefined): string {
  const key = String(phase || '').trim();
  if (!key) return '';
  return PHASE_LABELS[key] || jobStageLabel(key);
}

export function errorCategoryLabel(category: string | null | undefined): string {
  const key = String(category || '').trim();
  if (!key) return '';
  return CATEGORY_LABELS[key] || key;
}

export function errorTimeoutLabel(timeoutKind: string | null | undefined): string {
  const key = String(timeoutKind || '').trim();
  if (!key) return '';
  return TIMEOUT_LABELS[key] || key;
}

export function errorReconciliationLabel(
  result: string | null | undefined,
  reconciling?: boolean | null,
): string {
  if (reconciling === true) {
    return '等待远端释放确认（约 60 秒窗口）';
  }
  const key = String(result || '').trim();
  if (!key) return '';
  return RECONCILIATION_LABELS[key] || key;
}

export function formatErrorElapsed(seconds: number | null | undefined): string {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return '';
  return `${seconds.toFixed(1)} 秒`;
}

export function jobErrorMessage(job: JobErrorLike | null | undefined): string {
  if (!job) return '';
  if (typeof job.error_message === 'string' && job.error_message.trim()) return job.error_message.trim();
  if (typeof job.error === 'string' && job.error.trim()) return job.error.trim();
  if (job.error && typeof job.error === 'object' && typeof job.error.message === 'string' && job.error.message.trim()) {
    return job.error.message.trim();
  }
  return '';
}

export function jobErrorCode(job: JobErrorLike | null | undefined): string {
  if (!job) return '';
  if (typeof job.error_code === 'string' && job.error_code.trim()) return job.error_code.trim();
  if (typeof job.error_details?.error_code === 'string' && job.error_details.error_code.trim()) {
    return job.error_details.error_code.trim();
  }
  if (job.error && typeof job.error === 'object' && typeof job.error.code === 'string' && job.error.code.trim()) {
    return job.error.code.trim();
  }
  return '';
}

export function jobErrorRows(job: JobErrorLike | null | undefined): JobErrorRow[] {
  if (!job) return [];
  const rows: JobErrorRow[] = [];
  const details = job.error_details || null;
  const code = jobErrorCode(job);
  if (code) {
    rows.push({ key: 'error_code', label: '错误码', value: code, mono: true });
  }
  const phase = details?.phase || (code || jobErrorMessage(job) ? job.stage : null);
  const phaseText = errorPhaseLabel(phase);
  if (phaseText) {
    rows.push({ key: 'phase', label: '失败阶段', value: phaseText });
  }
  if (typeof details?.http_status === 'number' && Number.isFinite(details.http_status)) {
    rows.push({ key: 'http_status', label: 'HTTP 状态码', value: `HTTP ${details.http_status}`, mono: true });
  }
  const elapsedText = formatErrorElapsed(details?.elapsed_seconds);
  if (elapsedText) {
    rows.push({ key: 'elapsed_seconds', label: '请求耗时', value: elapsedText, mono: true });
  }
  const categoryText = errorCategoryLabel(details?.category);
  if (categoryText) {
    rows.push({ key: 'category', label: '异常归类', value: categoryText });
  }
  const timeoutText = errorTimeoutLabel(details?.timeout_kind);
  if (timeoutText) {
    rows.push({ key: 'timeout_kind', label: '网络/超时类型', value: timeoutText });
  }
  if (typeof details?.outcome === 'string' && details.outcome.trim()) {
    rows.push({ key: 'outcome', label: '上游拒绝码', value: details.outcome.trim(), mono: true });
  }
  const reconText = errorReconciliationLabel(details?.reconciliation_result, details?.reconciling);
  if (reconText) {
    rows.push({ key: 'reconciliation', label: '远端协调', value: reconText });
  }
  if (typeof details?.retry_attempt === 'number' && details.retry_attempt > 1) {
    rows.push({ key: 'retry_attempt', label: '尝试次数', value: `第 ${details.retry_attempt} 次` });
  }
  return rows;
}
