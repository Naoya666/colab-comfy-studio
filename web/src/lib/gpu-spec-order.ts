export interface GpuSpecKey {
  variant: string;
  accelerator: string;
  shape: string;
}

export type GpuSpecComparison = 'lower' | 'equal' | 'higher' | 'unknown';

export const HIGH_MEM_ONLY_ACCELERATORS = new Set(['L4', 'V5E1', 'V6E1']);

export const DEFAULT_GPU_SPEC_ORDER: GpuSpecKey[] = [
  { variant: 'VARIANT_GPU', accelerator: 'T4', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'T4', shape: 'SHAPE_HIGHMEM' },
  { variant: 'VARIANT_GPU', accelerator: 'L4', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'A100', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'A100', shape: 'SHAPE_HIGHMEM' },
  { variant: 'VARIANT_GPU', accelerator: 'G4', shape: 'SHAPE_STANDARD' },
  { variant: 'VARIANT_GPU', accelerator: 'H100', shape: 'SHAPE_STANDARD' },
];

export function normalizeGpuSpecKey(raw: Record<string, unknown> | null | undefined): GpuSpecKey | null {
  if (!raw || typeof raw !== 'object') return null;
  const source = (raw.key && typeof raw.key === 'object' ? raw.key : raw) as Record<string, unknown>;
  const accelerator = String(source.accelerator || '').trim().toUpperCase();
  if (!accelerator) return null;
  const rawVariant = String(source.variant || 'VARIANT_GPU').trim().toUpperCase();
  const variant = rawVariant.startsWith('VARIANT_') ? rawVariant : `VARIANT_${rawVariant}`;
  const rawShape = String(source.shape || 'SHAPE_STANDARD').trim().toUpperCase();
  let shape = rawShape.includes('HIGHMEM') ? 'SHAPE_HIGHMEM' : 'SHAPE_STANDARD';
  if (HIGH_MEM_ONLY_ACCELERATORS.has(accelerator)) {
    shape = 'SHAPE_STANDARD';
  }
  return { variant, accelerator, shape };
}

export function gpuSpecId(raw: Record<string, unknown> | GpuSpecKey | null | undefined): string | null {
  const key = normalizeGpuSpecKey(raw as Record<string, unknown> | null | undefined);
  if (!key) return null;
  return `${key.accelerator}:${key.shape}`;
}

export function compareGpuSpecs(
  actual: Record<string, unknown> | GpuSpecKey | null | undefined,
  required: Record<string, unknown> | GpuSpecKey | null | undefined,
  order: readonly GpuSpecKey[] = DEFAULT_GPU_SPEC_ORDER,
): GpuSpecComparison {
  const actualKey = normalizeGpuSpecKey(actual as Record<string, unknown> | null | undefined);
  const requiredKey = normalizeGpuSpecKey(required as Record<string, unknown> | null | undefined);
  if (!actualKey || !requiredKey) return 'unknown';
  if (
    actualKey.variant === requiredKey.variant
    && actualKey.accelerator === requiredKey.accelerator
    && actualKey.shape === requiredKey.shape
  ) {
    return 'equal';
  }
  if (actualKey.variant !== 'VARIANT_GPU' || requiredKey.variant !== 'VARIANT_GPU') {
    return 'unknown';
  }
  const ranks = new Map<string, number>();
  for (const item of order) {
    const id = gpuSpecId(item);
    if (id && !ranks.has(id)) ranks.set(id, ranks.size);
  }
  const actualId = `${actualKey.accelerator}:${actualKey.shape}`;
  const requiredId = `${requiredKey.accelerator}:${requiredKey.shape}`;
  const actualRank = ranks.get(actualId);
  const requiredRank = ranks.get(requiredId);
  if (actualRank === undefined || requiredRank === undefined) return 'unknown';
  if (actualRank < requiredRank) return 'lower';
  if (actualRank > requiredRank) return 'higher';
  return 'equal';
}

export function isBelowWorkflowDefault(
  candidate: Record<string, unknown> | GpuSpecKey | null | undefined,
  workflowDefault: Record<string, unknown> | GpuSpecKey | null | undefined,
  order: readonly GpuSpecKey[] = DEFAULT_GPU_SPEC_ORDER,
): boolean {
  if (!candidate || !workflowDefault) return false;
  return compareGpuSpecs(candidate, workflowDefault, order) === 'lower';
}

export function formatGpuSpecName(raw: Record<string, unknown> | GpuSpecKey | null | undefined): string {
  const key = normalizeGpuSpecKey(raw as Record<string, unknown> | null | undefined);
  if (!key) return '未知 GPU';
  if (HIGH_MEM_ONLY_ACCELERATORS.has(key.accelerator)) {
    return `${key.accelerator} · 标准配置`;
  }
  return `${key.accelerator} · ${key.shape === 'SHAPE_HIGHMEM' ? '高系统内存' : '标准系统内存'}`;
}
