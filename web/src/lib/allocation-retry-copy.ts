import type { AllocationRetryState, GpuSpecKey } from './gpu-api';
import { formatGpuSpecName } from './gpu-spec-order';

export function formatRetrySpec(spec?: GpuSpecKey | Record<string, unknown> | null): string {
  if (!spec || typeof spec !== 'object') return 'GPU';
  const inner = (spec as Record<string, unknown>).key;
  const target = (inner && typeof inner === 'object' ? inner : spec) as GpuSpecKey;
  return formatGpuSpecName(target);
}

export function formatAllocationRetryHeadline(
  retry: AllocationRetryState | null | undefined,
  nowSec = Math.floor(Date.now() / 1000),
): string {
  if (!retry) return '';
  const used = retry.retries_used ?? 0;
  const max = retry.max_retries ?? 20;
  const attempt = retry.attempt ?? 1;
  const phase = retry.phase || '';
  if (phase === 'reconciling') {
    return `自动排队抢卡 · 等待旧租约核验释放（防双开 · 已尝试 ${attempt} 次）`;
  }
  if (phase === 'waiting') {
    const countdown = retry.next_retry_at ? Math.max(0, Math.ceil(retry.next_retry_at - nowSec)) : 0;
    const targetSpec = formatRetrySpec(retry.next_spec);
    const targetAcct = retry.next_account_alias ? `${retry.next_account_alias} · ` : '';
    return `自动排队抢卡中（第 ${used}/${max} 次重试）· ${countdown > 0 ? `${countdown} 秒后尝试` : '即将尝试'} ${targetAcct}${targetSpec}`;
  }
  if (phase === 'allocating') {
    const targetSpec = formatRetrySpec(retry.current_spec);
    const targetAcct = retry.current_account_alias ? `${retry.current_account_alias} · ` : '';
    return used > 0
      ? `自动排队抢卡中（第 ${used}/${max} 次重试）· 正在申请 ${targetAcct}${targetSpec}`
      : `正在申请 ${targetAcct}${targetSpec}（已启用自动排队抢卡）`;
  }
  if (phase === 'exhausted') {
    return retry.stop_reason || `自动排队抢卡已停止（已达最大重试次数 ${used}/${max}）`;
  }
  if (phase === 'aborted') {
    return retry.stop_reason || '自动排队抢卡已中止';
  }
  if (phase === 'cancelled') {
    return retry.stop_reason || '已取消自动排队抢卡';
  }
  if (phase === 'succeeded' && used > 0) {
    return `自动排队抢卡成功（经过 ${used} 次重试，已分配 ${formatRetrySpec(retry.current_spec)}）`;
  }
  return '';
}

export function countDistinctDriveAccounts(
  accounts: Array<{ account_id: string; drive_account_id?: string | null }>,
): number {
  const drives = new Set<string>();
  for (const account of accounts) {
    const resolved = (account.drive_account_id || account.account_id || '').trim();
    if (resolved) drives.add(resolved);
  }
  return drives.size;
}
