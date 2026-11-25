import type { GpuInstance } from '../components/AccountDirectory';
import type { GpuSpecKey } from './gpu-spec-order';

const HIGH_MEM_ONLY = new Set(['L4', 'V5E1', 'V6E1']);
const FALLBACK_SPEC_ORDER: readonly GpuSpecKey[] = [
  { variant: 'VARIANT_GPU', accelerator: 'T4', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'T4', shape: 'SHAPE_HIGHMEM' },
  { variant: 'VARIANT_GPU', accelerator: 'L4', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'A100', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'A100', shape: 'SHAPE_HIGHMEM' },
  { variant: 'VARIANT_GPU', accelerator: 'G4', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'G4', shape: 'SHAPE_HIGHMEM' },
  { variant: 'VARIANT_GPU', accelerator: 'H100', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'H100', shape: 'SHAPE_HIGHMEM' },
];

function normalizeSpec(raw: Record<string, unknown> | GpuSpecKey | null | undefined): GpuSpecKey | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  const source = (r.key && typeof r.key === 'object' ? r.key : r) as Record<string, unknown>;
  const accelerator = String(source.accelerator || '').trim().toUpperCase();
  if (!accelerator) return null;
  const rawVariant = String(source.variant || 'VARIANT_GPU').trim().toUpperCase();
  const variant = rawVariant.startsWith('VARIANT_') ? rawVariant : `VARIANT_${rawVariant}`;
  const rawShape = String(source.shape || 'SHAPE_STANDARD').trim().toUpperCase();
  let shape = rawShape.includes('HIGHMEM') ? 'SHAPE_HIGHMEM' : 'SHAPE_STANDARD';
  if (HIGH_MEM_ONLY.has(accelerator)) shape = 'SHAPE_STANDARD';
  return { variant, accelerator, shape };
}

function compareSpecs(
  actual: Record<string, unknown> | GpuSpecKey | null | undefined,
  required: Record<string, unknown> | GpuSpecKey | null | undefined,
  order: readonly GpuSpecKey[] = FALLBACK_SPEC_ORDER,
): 'lower' | 'equal' | 'higher' | 'unknown' {
  const a = normalizeSpec(actual);
  const b = normalizeSpec(required);
  if (!a || !b) return 'unknown';
  if (a.variant === b.variant && a.accelerator === b.accelerator && a.shape === b.shape) return 'equal';
  if (a.variant !== 'VARIANT_GPU' || b.variant !== 'VARIANT_GPU') return 'unknown';
  const ranks = new Map<string, number>();
  for (const item of order) {
    const n = normalizeSpec(item);
    if (n) {
      const id = `${n.accelerator}:${n.shape}`;
      if (!ranks.has(id)) ranks.set(id, ranks.size);
    }
  }
  const ar = ranks.get(`${a.accelerator}:${a.shape}`);
  const br = ranks.get(`${b.accelerator}:${b.shape}`);
  if (ar === undefined || br === undefined) return 'unknown';
  return ar < br ? 'lower' : ar > br ? 'higher' : 'equal';
}

export type TransferPool = 'creation' | 'native' | 'ephemeral';

export interface TransferFocusTarget {
  instanceId: string;
  targetPool: TransferPool;
}

export interface TransferStateLike {
  transfer_id?: string;
  from_pool?: string;
  to_pool?: string;
  phase?: string;
  transfer_job_id?: string;
  native_bootstrap_job_id?: string;
  started_at?: number | null;
  completed_at?: number | null;
  error_code?: string | null;
  error_message?: string | null;
}

export function instanceIdOfCard(instance: Pick<GpuInstance, 'id' | 'instance_id'>): string {
  return instance.id || instance.instance_id || '';
}

export function instanceSpecObject(instance: Pick<GpuInstance, 'runtime_spec_reported' | 'runtime_spec_requested'>): Record<string, unknown> | null {
  const raw = instance.runtime_spec_reported || instance.runtime_spec_requested;
  if (!raw || typeof raw !== 'object') return null;
  return raw as Record<string, unknown>;
}

export function isCpuInstance(instance: Pick<GpuInstance, 'runtime_spec_reported' | 'runtime_spec_requested' | 'gpu_reported'>): boolean {
  const spec = instanceSpecObject(instance);
  if (spec) {
    const inner = (spec.key && typeof spec.key === 'object' ? spec.key : spec) as Record<string, unknown>;
    const variant = String(inner.variant || '').toUpperCase();
    const accel = String(inner.accelerator || '').toUpperCase();
    if (variant === 'CPU' || variant === 'VARIANT_CPU' || accel === 'CPU' || accel === 'NONE') {
      return true;
    }
  }
  const reported = String(instance.gpu_reported || '').toUpperCase();
  return reported === 'CPU';
}

export function isGpuInstance(instance: Pick<GpuInstance, 'runtime_spec_reported' | 'runtime_spec_requested' | 'gpu_reported' | 'provider'>): boolean {
  if (isCpuInstance(instance)) return false;
  const spec = instanceSpecObject(instance);
  if (spec) {
    const inner = (spec.key && typeof spec.key === 'object' ? spec.key : spec) as Record<string, unknown>;
    const variant = String(inner.variant || '').toUpperCase();
    if (variant) return variant === 'GPU' || variant === 'VARIANT_GPU';
  }
  return instance.provider === 'mock' || Boolean(instance.gpu_reported);
}

export function domainBadgeLabel(instance: Pick<GpuInstance, 'instance_pool' | 'runtime_spec_reported' | 'runtime_spec_requested' | 'gpu_reported'>): string {
  const pool = String(instance.instance_pool || 'creation').toLowerCase();
  if (pool === 'native') return '原生 ComfyUI';
  if (pool === 'ephemeral') {
    return isCpuInstance(instance) ? '一次性（CPU）' : '冒烟域';
  }
  return '创作池';
}

export function domainBadgeTone(instance: Pick<GpuInstance, 'instance_pool' | 'runtime_spec_reported' | 'runtime_spec_requested' | 'gpu_reported'>): 'creation' | 'native' | 'ephemeral' | 'cpu' {
  const pool = String(instance.instance_pool || 'creation').toLowerCase();
  if (pool === 'native') return 'native';
  if (pool === 'ephemeral') return isCpuInstance(instance) ? 'cpu' : 'ephemeral';
  return 'creation';
}

export function formatVmElapsed(vmAllocatedAt?: number | null, nowSec: number = Math.floor(Date.now() / 1000)): string {
  if (typeof vmAllocatedAt !== 'number' || !Number.isFinite(vmAllocatedAt) || vmAllocatedAt <= 0) {
    return '—';
  }
  const totalSec = Math.max(0, Math.floor(nowSec - vmAllocatedAt));
  const totalMin = Math.floor(totalSec / 60);
  if (totalMin < 1) return '< 1 分钟';
  if (totalMin < 60) return `${totalMin} 分钟`;
  const hours = Math.floor(totalMin / 60);
  const mins = totalMin % 60;
  return mins > 0 ? `${hours} 小时 ${mins} 分钟` : `${hours} 小时`;
}

export function canShowTransferButton(instance: GpuInstance): boolean {
  if (!instance || instance.runtime_deleted) return false;
  const state = String(instance.state || '').toLowerCase();
  if (state !== 'ready' && state !== 'retained_idle') return false;
  return isGpuInstance(instance);
}

export function transferDisabledReason(instance: GpuInstance): string | null {
  if (instance.venv_isolated === false) {
    return '旧版运行时环境，不支持域转移';
  }
  if (instance.manual_stop_requested) {
    return '实例正在停止，暂不可转移';
  }
  if (instance.current_job_id) {
    return '请等待当前任务完成后再转移';
  }
  return null;
}

export interface TransferStepItem {
  key: string;
  label: string;
  status: 'done' | 'active' | 'pending' | 'failed';
}

export function buildTransferSteps(ts?: TransferStateLike | null): TransferStepItem[] {
  const phase = String(ts?.phase || 'cleaning');
  const toNative = ts?.to_pool === 'native';
  const steps: TransferStepItem[] = [
    { key: 'clean', label: '清理旧域环境', status: 'pending' },
    { key: 'verify', label: '校验环境纯净度', status: 'pending' },
  ];
  if (toNative) {
    steps.push({ key: 'native_boot', label: '拉起原生 ComfyUI', status: 'pending' });
  }
  if (phase === 'cleaning') {
    steps[0].status = 'active';
  } else if (phase === 'verifying') {
    steps[0].status = 'done';
    steps[1].status = 'active';
  } else if (phase === 'native_bootstrapping') {
    steps[0].status = 'done';
    steps[1].status = 'done';
    if (steps[2]) steps[2].status = 'active';
  } else if (phase === 'succeeded') {
    for (const s of steps) s.status = 'done';
  } else if (phase === 'native_bootstrap_failed') {
    steps[0].status = 'done';
    steps[1].status = 'done';
    if (steps[2]) steps[2].status = 'failed';
  } else {
    steps[0].status = 'failed';
  }
  return steps;
}

function isIdleTransferableCandidate(instance: GpuInstance, nowSec: number): boolean {
  if (!canShowTransferButton(instance)) return false;
  if (instance.venv_isolated !== true) return false;
  if (instance.manual_stop_requested || instance.current_job_id) return false;
  if (instance.provider && instance.provider !== 'colab') return false;
  const idleDeadline = typeof instance.idle_deadline_at === 'number'
    ? instance.idle_deadline_at
    : (typeof instance.idle_deadline_at === 'string' && instance.idle_deadline_at ? Date.parse(instance.idle_deadline_at) / 1000 : null);
  const pool = String(instance.instance_pool || 'creation').toLowerCase();
  const isNativeReady = pool === 'native' && !instance.native_bootstrap_failed;
  if (!isNativeReady && idleDeadline != null && Number.isFinite(idleDeadline) && nowSec >= idleDeadline) {
    return false;
  }
  return true;
}

export function findTransferableInstancesForNative(
  instances: GpuInstance[],
  nowSec: number = Math.floor(Date.now() / 1000),
): GpuInstance[] {
  const hasNativeLive = instances.some((inst) => {
    if (inst.runtime_deleted) return false;
    const state = String(inst.state || '').toLowerCase();
    if (state === 'deleted') return false;
    const pool = String(inst.instance_pool || 'creation').toLowerCase();
    if (pool === 'native') return true;
    const ts = inst.transfer_state as TransferStateLike | null | undefined;
    if (state === 'transferring' && ts?.to_pool === 'native') return true;
    return false;
  });
  if (hasNativeLive) return [];
  return instances.filter((inst) => {
    const pool = String(inst.instance_pool || 'creation').toLowerCase();
    return pool !== 'native' && isIdleTransferableCandidate(inst, nowSec);
  });
}

export function findTransferableInstancesForCreation(
  instances: GpuInstance[],
  requiredSpec: Record<string, unknown> | GpuSpecKey | null | undefined,
  gpuOrder: readonly GpuSpecKey[] = FALLBACK_SPEC_ORDER,
  nowSec: number = Math.floor(Date.now() / 1000),
): GpuInstance[] {
  const hasActiveCreation = instances.some((inst) => {
    if (inst.runtime_deleted) return false;
    const state = String(inst.state || '').toLowerCase();
    const pool = String(inst.instance_pool || 'creation').toLowerCase();
    if (pool === 'creation' && ['requested', 'allocating', 'ready', 'busy', 'retained_idle'].includes(state)) {
      return true;
    }
    const ts = inst.transfer_state as TransferStateLike | null | undefined;
    if (state === 'transferring' && ts?.to_pool === 'creation') return true;
    return false;
  });
  if (hasActiveCreation) return [];
  const reqNorm = normalizeSpec(requiredSpec);
  return instances.filter((inst) => {
    const pool = String(inst.instance_pool || 'creation').toLowerCase();
    if (pool === 'creation') return false;
    if (!isIdleTransferableCandidate(inst, nowSec)) return false;
    if (!reqNorm) return true;
    const actual = instanceSpecObject(inst);
    if (!actual) return false;
    const cmp = compareSpecs(actual, reqNorm, gpuOrder);
    return cmp === 'equal' || cmp === 'higher';
  });
}
