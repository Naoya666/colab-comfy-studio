import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeftRight, ChevronDown, RefreshCw, Square, Timer, Trash2, Wrench } from 'lucide-react';
import {
  apiMessage,
  cancelRecoveryPlan,
  clearGpuInstanceHistory,
  getGpuInstance,
  getInstanceTransferPreview,
  getRecoveryStatus,
  setInstanceRetention,
  stopInstance,
  transferGpuInstance,
  type RecoveryPlan,
  type TransferInstanceRequest,
  type TransferPreviewResponse,
} from '../lib/gpu-api';
import type { GpuInstance } from './AccountDirectory';
import { InstanceWorkbench } from './InstanceWorkbench';
import { formatInstanceTime as formatTime } from '../lib/instance-time';
import { gpuLabel } from '../lib/gpu-label';
import { isCreationPoolInstance } from '../lib/instance-submission';
import {
  buildTransferSteps,
  canShowTransferButton,
  domainBadgeLabel,
  domainBadgeTone,
  formatVmElapsed,
  transferDisabledReason,
  type TransferFocusTarget,
  type TransferPool,
} from '../lib/domain-transfer';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import { ConfirmDialog } from './Feedback';
import styles from './v020.module.css';

type Filter = 'all' | 'active' | 'retained' | 'history';
type Action = 'retain' | 'stop';
const COLLAPSED_LIMIT = 5;
type Detail = GpuInstance & {
  jobs?: Array<Record<string, unknown>>;
  events?: Array<Record<string, unknown>>;
};

export interface InstanceCenterProps {
  instances?: GpuInstance[];
  onContinue?: (instance: GpuInstance) => void;
  onOpenWorkbench?: (instance: GpuInstance) => void;
  onOpenNative?: () => void;
  onRefresh?: () => void;
  focusTransferTarget?: TransferFocusTarget | null;
  onFocusTransferHandled?: () => void;
}

let keySequence = 0;
function newIdempotencyKey(prefix: string): string {
  keySequence += 1;
  return `${prefix}-${Date.now().toString(36)}-${keySequence.toString(36)}`;
}

function instanceId(instance: Pick<GpuInstance, 'id' | 'instance_id'>): string {
  return instance.id || instance.instance_id || '';
}

function normalizedInstance(instance: GpuInstance): GpuInstance {
  const id = instanceId(instance);
  return { ...instance, id, instance_id: instance.instance_id || id };
}

function shortId(value?: string | null): string {
  const raw = String(value || '').trim();
  if (!raw) return '—';
  return raw.length > 14 ? `#${raw.slice(0, 8)}` : raw;
}

function label(instance: GpuInstance): string {
  const id = instanceId(instance);
  return instance.instance_label || instance.label || (id ? `实例 ${shortId(id)}` : '未命名实例');
}

function stateLabel(value?: string): string {
  return ({
    requested: '请求中', allocating: '分配中', ready: '就绪', busy: '运行中',
    retained_idle: '保留空闲', transferring: '域转移中', releasing: '释放中', deleted: '已删除',
    unknown: '未知', cleanup_pending: '待清理', quarantined: '保护隔离',
  } as Record<string, string>)[String(value || '')] || value || '状态未知';
}

function poolLabel(pool?: string): string {
  return ({
    native: '原生 ComfyUI',
    creation: '创作池',
    ephemeral: '冒烟域',
  } as Record<string, string>)[String(pool || 'creation')] || String(pool);
}

function recoveryStateLabel(state?: string): string {
  return ({
    scheduled: '等待重试分配',
    dispatching: '拉起继任实例中',
    cancelling: '等待远端释放确认',
    restoring_env: '后台恢复本地模型',
    recovered: '已恢复',
    exhausted: '已耗尽重试',
    aborted_fatal: '已中止',
    aborted_vps_outage: 'VPS 失联保护放弃',
    cancelled: '已取消',
  } as Record<string, string>)[String(state || '')] || String(state || '恢复中');
}

function recoverySpecLabel(spec?: Record<string, unknown> | null): string {
  if (!spec) return '原规格';
  const acc = typeof spec.accelerator === 'string' ? spec.accelerator : '';
  const shape = typeof spec.shape === 'string' && spec.shape !== 'standard' ? ` ${spec.shape}` : '';
  return acc ? `${acc}${shape}` : '原规格';
}

function isHistory(instance: GpuInstance): boolean {
  return ['deleted', 'unknown', 'cleanup_pending', 'quarantined'].includes(String(instance.state || '').toLowerCase());
}

function isActive(instance: GpuInstance): boolean {
  return ['requested', 'allocating', 'ready', 'busy', 'retained_idle', 'transferring', 'releasing'].includes(String(instance.state || '').toLowerCase());
}

function isRetained(instance: GpuInstance): boolean {
  return instance.state === 'retained_idle' && !instance.runtime_deleted;
}

function remainingCountdownText(deadline: string | number | null | undefined, nowSec: number): string | null {
  if (deadline == null || deadline === '') return null;
  const epoch = typeof deadline === 'number' ? deadline : (Number.isFinite(Number(deadline)) ? Number(deadline) : Date.parse(deadline) / 1000);
  if (!Number.isFinite(epoch)) return null;
  const rem = Math.max(0, Math.ceil(epoch - nowSec));
  if (rem <= 0) return '即将到期';
  const mins = Math.floor(rem / 60);
  const secs = rem % 60;
  return mins > 0 ? `剩余 ${mins} 分 ${secs} 秒` : `剩余 ${secs} 秒`;
}

function errorText(cause: unknown): string {
  return apiMessage(cause);
}

const DOMAIN_TONE_CLASS: Record<ReturnType<typeof domainBadgeTone>, string> = {
  creation: styles.domainCreation,
  native: styles.domainNative,
  ephemeral: styles.domainEphemeral,
  cpu: styles.domainCpu,
};

export function InstanceCenter({
  instances = [],
  onContinue,
  onOpenWorkbench,
  onOpenNative,
  onRefresh,
  focusTransferTarget,
  onFocusTransferHandled,
}: InstanceCenterProps) {
  const [filter, setFilter] = useState<Filter>('all');
  const [expanded, setExpanded] = useState(false);
  const [clearingHistory, setClearingHistory] = useState(false);
  const [confirmClearHistory, setConfirmClearHistory] = useState(false);
  const [selectedId, setSelectedId] = useState('');
  const [detail, setDetail] = useState<Detail | null>(null);
  const [workbench, setWorkbench] = useState<GpuInstance | null>(null);
  const currentWorkbench = workbench ? instances.find((item) => instanceId(item) === instanceId(workbench)) || workbench : null;
  const [confirm, setConfirm] = useState<{ action: Action; instance: GpuInstance } | null>(null);
  const [retainMinutes, setRetainMinutes] = useState(15);

  // Domain transfer modal state
  const [transferModal, setTransferModal] = useState<{ instance: GpuInstance; preferredTarget?: TransferPool } | null>(null);
  const [transferPreview, setTransferPreview] = useState<TransferPreviewResponse | null>(null);
  const [transferPreviewLoading, setTransferPreviewLoading] = useState(false);
  const [transferTarget, setTransferTarget] = useState<TransferPool | ''>('');
  const [transferCreationIdle, setTransferCreationIdle] = useState(15);
  const [transferNativeMode, setTransferNativeMode] = useState<'local' | 'drive_mount'>('local');
  const [transferNativeCacheGb, setTransferNativeCacheGb] = useState(20);
  const [transferNativeCacheAge, setTransferNativeCacheAge] = useState(60);
  const [transferNativeIdle, setTransferNativeIdle] = useState(120);
  const [transferNativeMaxRuntime, setTransferNativeMaxRuntime] = useState(720);
  const [transferSubmitting, setTransferSubmitting] = useState(false);
  const [transferError, setTransferError] = useState('');

  useBodyScrollLock(Boolean(confirm || transferModal));
  const [busy, setBusy] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [activePlans, setActivePlans] = useState<RecoveryPlan[]>([]);
  const [clockSec, setClockSec] = useState(() => Math.floor(Date.now() / 1000));
  const keyRef = useRef<Record<string, string>>({});

  const refreshRecovery = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await getRecoveryStatus(signal);
      if (!signal?.aborted) setActivePlans(res.active_plans || []);
    } catch {
      // Recovery status is supplemental; do not disrupt the instance list on transient read errors.
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshRecovery(controller.signal);
    const interval = window.setInterval(() => void refreshRecovery(), activePlans.length ? 4000 : 15000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [refreshRecovery, activePlans.length, instances.length]);

  const rows = useMemo(() => {
    const seen = new Set<string>();
    return instances.map(normalizedInstance).filter((item) => {
      const id = instanceId(item);
      if (!id || seen.has(id)) return false;
      seen.add(id);
      return true;
    }).sort((a, b) => (Date.parse(b.created_at || '') || 0) - (Date.parse(a.created_at || '') || 0));
  }, [instances]);

  const needsLiveClock = useMemo(
    () => activePlans.some((plan) => Boolean(plan.next_retry_at))
      || rows.some((row) => isActive(row) && (Boolean(row.vm_allocated_at) || Boolean(row.native_bootstrap_failed))),
    [activePlans, rows],
  );

  useEffect(() => {
    if (!needsLiveClock) return;
    setClockSec(Math.floor(Date.now() / 1000));
    const timer = window.setInterval(() => setClockSec(Math.floor(Date.now() / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [needsLiveClock]);

  const openTransferModal = useCallback(async (instance: GpuInstance, preferredTarget?: TransferPool) => {
    const id = instanceId(instance);
    if (!id) return;
    setTransferModal({ instance, preferredTarget });
    setTransferPreview(null);
    setTransferError('');
    setTransferTarget('');
    setTransferPreviewLoading(true);
    try {
      const preview = await getInstanceTransferPreview(id);
      setTransferPreview(preview);
      const targets = preview.targets || {};
      const order: TransferPool[] = ['creation', 'native', 'ephemeral'];
      const preferred = preferredTarget && targets[preferredTarget]?.allowed ? preferredTarget : undefined;
      const fallback = order.find((pool) => targets[pool]?.allowed);
      const chosen = preferred || fallback || (preferredTarget && targets[preferredTarget] ? preferredTarget : order.find((pool) => Boolean(targets[pool])) || '');
      setTransferTarget(chosen);
      const cDef = targets.creation?.defaults as { retain_idle_minutes?: number } | undefined;
      if (cDef?.retain_idle_minutes) setTransferCreationIdle(Number(cDef.retain_idle_minutes) || 15);
      const nDef = targets.native?.defaults as {
        model_storage_mode?: 'local' | 'drive_mount';
        cache_gb?: number;
        cache_max_age_minutes?: number;
        retain_idle_minutes?: number;
        max_runtime_minutes?: number;
      } | undefined;
      if (nDef) {
        if (nDef.model_storage_mode === 'local' || nDef.model_storage_mode === 'drive_mount') {
          setTransferNativeMode(nDef.model_storage_mode);
        }
        if (nDef.cache_gb) setTransferNativeCacheGb(Number(nDef.cache_gb) || 20);
        if (nDef.cache_max_age_minutes) setTransferNativeCacheAge(Number(nDef.cache_max_age_minutes) || 60);
        if (nDef.retain_idle_minutes) setTransferNativeIdle(Number(nDef.retain_idle_minutes) || 120);
        if (nDef.max_runtime_minutes) setTransferNativeMaxRuntime(Number(nDef.max_runtime_minutes) || 720);
      }
    } catch (cause) {
      setTransferError(errorText(cause));
    } finally {
      setTransferPreviewLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!focusTransferTarget?.instanceId) return;
    const match = rows.find((item) => instanceId(item) === focusTransferTarget.instanceId);
    if (!match) return;
    setFilter('all');
    setExpanded(true);
    void openTransferModal(match, focusTransferTarget.targetPool);
    onFocusTransferHandled?.();
    window.requestAnimationFrame(() => {
      document.getElementById(`instance-card-${focusTransferTarget.instanceId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }, [focusTransferTarget, rows, openTransferModal, onFocusTransferHandled]);

  async function submitTransfer() {
    if (!transferModal || !transferTarget || transferSubmitting) return;
    const id = instanceId(transferModal.instance);
    if (!id) return;
    const targetInfo = transferPreview?.targets?.[transferTarget];
    if (targetInfo && !targetInfo.allowed) {
      setTransferError(targetInfo.reason || '该目标域当前不可转入');
      return;
    }
    const opKey = `transfer:${id}:${transferTarget}`;
    const idempotencyKey = (keyRef.current[opKey] ||= newIdempotencyKey('instance-transfer'));
    const version = transferPreview?.version ?? (typeof transferModal.instance.version === 'number' ? transferModal.instance.version : undefined);
    const payload: TransferInstanceRequest = {
      target_pool: transferTarget,
      confirmed: true,
      ...(version !== undefined ? { expected_version: version } : {}),
      ...(transferTarget === 'creation'
        ? { creation_options: { retain_idle_minutes: Math.max(1, Math.min(30, Math.round(transferCreationIdle || 15))) } }
        : {}),
      ...(transferTarget === 'native'
        ? {
            native_options: {
              model_storage_mode: transferNativeMode,
              cache_gb: Math.max(1, Math.round(transferNativeCacheGb || 20)),
              cache_max_age_minutes: Math.max(1, Math.round(transferNativeCacheAge || 60)),
              retain_idle_minutes: Math.max(1, Math.round(transferNativeIdle || 120)),
              max_runtime_minutes: Math.max(1, Math.round(transferNativeMaxRuntime || 720)),
            },
          }
        : {}),
    };
    setTransferSubmitting(true);
    setTransferError('');
    try {
      const res = await transferGpuInstance(id, payload, idempotencyKey);
      delete keyRef.current[opKey];
      setTransferModal(null);
      setTransferPreview(null);
      updateAfterWrite(
        res.instance as unknown as GpuInstance,
        `已发起域转移（→ ${poolLabel(transferTarget)}），正在远端清理并校验环境纯净度…`,
      );
    } catch (cause) {
      setTransferError(errorText(cause));
    } finally {
      setTransferSubmitting(false);
    }
  }

  async function cancelPlan(plan: RecoveryPlan) {
    if (busy) return;
    setBusy(`cancel-plan:${plan.plan_id}`);
    setError('');
    setStatus('正在取消自动恢复计划…');
    try {
      const res = await cancelRecoveryPlan(plan.plan_id, `instance-center-cancel:${plan.plan_id}`);
      setStatus(res.state === 'cancelling' ? '已提交取消，等待在途租约确认释放。' : '已取消自动恢复计划。');
      await refreshRecovery();
      onRefresh?.();
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setBusy(null);
    }
  }

  useEffect(() => {
    if (selectedId && !rows.some((item) => instanceId(item) === selectedId)) {
      setSelectedId(''); setDetail(null);
    }
  }, [rows, selectedId]);

  const activeCount = rows.filter((instance) => isActive(instance) && !isHistory(instance)).length;
  const retainedCount = rows.filter(isRetained).length;
  const clearableCount = rows.filter((instance) => instance.state === 'deleted').length;
  const pendingCleanupCount = rows.filter((instance) => isHistory(instance) && instance.state !== 'deleted').length;
  const hasClearableHistory = clearableCount > 0;
  const filtered = rows.filter((instance) => {
    if (filter === 'active') return isActive(instance) && !isHistory(instance);
    if (filter === 'retained') return isRetained(instance);
    if (filter === 'history') return isHistory(instance);
    return true;
  });
  const visibleRows = expanded ? filtered : filtered.slice(0, COLLAPSED_LIMIT);

  async function clearHistory() {
    if (clearingHistory || !hasClearableHistory) return;
    setClearingHistory(true);
    setError('');
    setStatus('');
    try {
      const { cleared } = await clearGpuInstanceHistory();
      setConfirmClearHistory(false);
      onRefresh?.();
      setStatus(cleared ? `已清除 ${cleared} 条实例历史。` : '没有可清除的实例历史。');
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setClearingHistory(false);
    }
  }

  function keyFor(action: Action, id: string) {
    const operation = `${action}:${id}`;
    return keyRef.current[operation] ||= newIdempotencyKey(`instance-${action}`);
  }

  function clearKey(action: Action, id: string) { delete keyRef.current[`${action}:${id}`]; }

  async function openDetail(instance: GpuInstance) {
    const id = instanceId(instance);
    if (!id) return;
    setSelectedId(id); setDetail(null); setError(''); setStatus('正在读取实例详情…');
    try {
      const result = normalizedInstance(await getGpuInstance(id) as unknown as GpuInstance) as Detail;
      setDetail(result);
      setStatus('实例详情已刷新。');
    } catch (cause) {
      setError(errorText(cause)); setStatus('详情读取未确认，请重试。');
    }
  }

  function updateAfterWrite(result: GpuInstance, message: string) {
    const next = normalizedInstance(result);
    const id = instanceId(next);
    setDetail((current) => current && instanceId(current) === id ? { ...current, ...next } : current);
    setStatus(message);
    setError('');
    onRefresh?.();
  }

  async function submitAction() {
    if (!confirm || busy) return;
    const { action, instance } = confirm;
    const id = instanceId(instance);
    if (!id) return;
    setBusy(`${action}:${id}`); setError(''); setStatus('正在等待服务端确认…');
    try {
      if (action === 'retain') {
        const version = typeof instance.version === 'number' && Number.isFinite(instance.version) ? instance.version : 0;
        const result = await setInstanceRetention(id, {
          retention_mode: 'retain',
          idle_minutes: Math.max(0, Math.min(1440, retainMinutes)),
          expected_version: version,
        }, keyFor(action, id)) as unknown as GpuInstance;
        clearKey(action, id);
        updateAfterWrite(result, '实例已确认进入保留策略。');
      } else {
        const result = await stopInstance(id, keyFor(action, id)) as unknown as GpuInstance;
        clearKey(action, id);
        updateAfterWrite(result, '已确认停止请求；实例仍需等待远端释放完成。');
      }
      setConfirm(null);
    } catch (cause) {
      setError(errorText(cause)); setStatus('操作未确认；再次重试会复用同一幂等键。');
    } finally { setBusy(null); }
  }

  function chooseWorkbench(instance: GpuInstance) {
    onOpenWorkbench?.(instance);
    setWorkbench(normalizedInstance(instance));
  }

  if (workbench) {
    return <div className={styles.center}>
      <InstanceWorkbench instance={currentWorkbench || workbench} onBack={() => setWorkbench(null)} onQueued={() => onRefresh?.()} />
    </div>;
  }

  return <section className={styles.center} aria-label="实例中心">
    <div className={styles.sectionHeader}>
      <div><p className={styles.eyebrow}>资源管理</p><h2>实例中心</h2><p className={styles.muted}>查看真实实例所属域与状态；支持显式转移域（保留原 GPU 重建洁净环境）。</p></div>
      <button type="button" className={styles.iconButton} aria-label="刷新实例" onClick={() => { onRefresh?.(); void refreshRecovery(); }}><RefreshCw size={18} aria-hidden /></button>
    </div>
    {status && <p className={styles.status} role="status">{status}</p>}
    {error && <p className={styles.error} role="alert">{error}</p>}
    {activePlans.length > 0 && (
      <div className={styles.recoveryPlanStack} aria-label="自动恢复中的计划">
        {activePlans.map((plan) => {
          const countdown = plan.next_retry_at ? Math.max(0, Math.ceil(plan.next_retry_at - clockSec)) : 0;
          const displayRound = plan.state === 'restoring_env'
            ? Math.max(1, plan.rounds_used)
            : Math.min(plan.rounds_used + 1, plan.max_rounds);
          return (
            <article className={styles.recoveryPlanCard} key={plan.plan_id}>
              <div className={styles.cardTop}>
                <div className={styles.minWidth}>
                  <h3>
                    自动恢复 · {poolLabel(plan.pool)}（第 {displayRound}/{plan.max_rounds} 轮）
                  </h3>
                  <p className={styles.muted}>
                    {plan.account_alias_snapshot || (plan.account_id ? `账号 ${shortId(plan.account_id)}` : '原账号')} · {recoverySpecLabel(plan.requested_spec)}
                    {plan.source_instance_id ? ` · 接替 ${shortId(plan.source_instance_id)}` : ''}
                    {plan.state === 'scheduled' && countdown > 0 ? ` · ${countdown}s 后重试` : ''}
                  </p>
                </div>
                <span className={`${styles.badge} ${styles.recoveryBadge}`}>{recoveryStateLabel(plan.state)}</span>
              </div>
              {plan.model_restore && plan.state === 'restoring_env' && (
                <p className={styles.muted}>
                  后台恢复本地模型：{plan.model_restore.completed} / {plan.model_restore.total}
                  {plan.model_restore.skipped > 0 ? `（跳过 ${plan.model_restore.skipped}）` : ''}
                </p>
              )}
              {plan.last_error_message && <p className={styles.muted}>{plan.last_error_message}</p>}
              {plan.state !== 'cancelling' && (
                <div className={styles.cardActions}>
                  <button
                    type="button"
                    className={styles.buttonDanger}
                    disabled={busy === `cancel-plan:${plan.plan_id}`}
                    onClick={() => void cancelPlan(plan)}
                  >
                    {busy === `cancel-plan:${plan.plan_id}` ? '正在取消…' : '取消自动恢复'}
                  </button>
                </div>
              )}
            </article>
          );
        })}
      </div>
    )}
    <div className={styles.summaryRow} aria-label="实例统计">
      <div className={styles.summaryCell}><span className={styles.muted}>全部</span><b>{rows.length}</b></div>
      <div className={styles.summaryCell}><span className={styles.muted}>活动</span><b>{activeCount}</b></div>
      <div className={styles.summaryCell}><span className={styles.muted}>保留</span><b>{retainedCount}</b></div>
    </div>
    <div className={styles.toolbar}>
      <div className={styles.filterGroup} role="group" aria-label="实例筛选">
        {([['all', '全部'], ['active', '活动'], ['retained', '保留'], ['history', '历史']] as const).map(([value, text]) => <button key={value} type="button" className={styles.filterButton} aria-pressed={filter === value} onClick={() => { setFilter(value); setExpanded(false); }}>{text}</button>)}
      </div>
      <button type="button" className={styles.buttonSecondary} disabled={!hasClearableHistory || clearingHistory} onClick={() => setConfirmClearHistory(true)}>
        <Trash2 size={15} aria-hidden />{clearingHistory ? '正在清除…' : clearableCount ? `清除历史 (${clearableCount})` : '清除历史'}
      </button>
    </div>
    {filter === 'history' && !hasClearableHistory && pendingCleanupCount > 0 && (
      <p className={styles.muted}>有 {pendingCleanupCount} 条实例处于待清理或异常状态，确认远端释放后才可清除。</p>
    )}
    {!filtered.length ? <div className={styles.empty}><strong>{rows.length ? '当前筛选没有实例' : '还没有实例'}</strong><span>{rows.length ? '切换筛选查看其他状态。' : '提交任务后，服务端分配的实例会显示在这里。'}</span></div> : <div className={styles.instanceMeta} id="instance-center-list">
      {visibleRows.map((instance) => {
        const id = instanceId(instance);
        const isSelected = selectedId === id;
        const isTransferring = instance.state === 'transferring';
        const isQuarantined = instance.state === 'quarantined';
        const canMutate = !isHistory(instance) && instance.state !== 'releasing';
        const canStop = canMutate || isQuarantined;
        const canRetain = canMutate && !isTransferring && instance.instance_pool !== 'ephemeral' && !instance.native_bootstrap_failed;
        const canContinue = canMutate && !isTransferring && isCreationPoolInstance(instance);
        const showTransfer = canShowTransferButton(instance);
        const transferBlock = transferDisabledReason(instance);
        const state = instance.remote_delete_source === 'provider_runtime_missing' ? '已回收' : stateLabel(instance.state);
        const domainText = domainBadgeLabel(instance);
        const domainToneClass = DOMAIN_TONE_CLASS[domainBadgeTone(instance)];
        const runtime = instance.creation_runtime as { workflow_key?: string; workflow_version?: string | number; process?: string } | null | undefined;
        const transferSteps = isTransferring ? buildTransferSteps(instance.transfer_state) : null;
        const nativeFailedStandby = Boolean(instance.native_bootstrap_failed && instance.state === 'retained_idle');
        const standbyRemain = nativeFailedStandby ? remainingCountdownText(instance.idle_deadline_at, clockSec) : null;
        return <article className={styles.instanceCard} id={`instance-card-${id}`} key={id}>
          <div className={styles.cardTop}>
            <div className={styles.minWidth}>
              <h3 title={id}>{label(instance)}</h3>
              <p className={styles.muted}>
                {instance.account_alias_snapshot || (instance.account_id ? `账号 ${shortId(instance.account_id)}` : '账号未确认')} · {instance.provider || '运行时未确认'}
                {instance.domain_generation ? ` · 已转移 ${instance.domain_generation} 次` : ''}
                {instance.parent_instance_id ? ` · 接替 ${shortId(instance.parent_instance_id)}` : ''}
                {instance.replaced_by_instance_id ? ` · 已由 ${shortId(instance.replaced_by_instance_id)} 接替` : ''}
              </p>
            </div>
            <div className={styles.badgeGroup}>
              <span className={`${styles.badge} ${styles.domainBadge} ${domainToneClass}`} data-domain-badge={instance.instance_pool || 'creation'}>
                {domainText}
              </span>
              {instance.recovery_source === 'auto_recovery' && (
                <span className={`${styles.badge} ${styles.recoveryBadge}`} title={`恢复组 ${instance.recovery_group_id || ''}`}>
                  自动恢复 #{instance.recovery_generation ?? 1}
                </span>
              )}
              <span className={`${styles.badge} ${instance.state === 'ready' || instance.state === 'busy' || instance.state === 'retained_idle' ? styles.good : isQuarantined ? styles.bad : ''}`}>{state}</span>
            </div>
          </div>

          {transferSteps && (
            <div className={styles.transferProgress} role="status" aria-label="域转移进度">
              <span>→ {poolLabel(instance.transfer_state?.to_pool)}：</span>
              {transferSteps.map((step, idx) => (
                <span key={step.key} className={`${styles.transferStep} ${step.status === 'active' ? styles.transferStepActive : step.status === 'done' ? styles.transferStepDone : step.status === 'failed' ? styles.transferStepFailed : ''}`}>
                  {idx > 0 ? '→ ' : ''}{step.label}
                </span>
              ))}
            </div>
          )}

          {nativeFailedStandby && (
            <div className={styles.standbyAlert} role="status" aria-label="原生启动失败待命提示">
              <strong>原生 ComfyUI 启动失败，GPU 正在干净待命{standbyRemain ? `（${standbyRemain}）` : '（10 分钟）'}</strong>
              <span>{instance.native_bootstrap_error || '已清理失败残留，未销毁物理 GPU。可直接前往原生页修改参数原地重试，或直接转移至其他域。'}</span>
              <div className={styles.cardActions}>
                {onOpenNative && <button type="button" className={styles.buttonSecondary} onClick={onOpenNative}>前往原生页重试</button>}
                <button type="button" className={styles.buttonSecondary} onClick={() => void openTransferModal(instance, 'creation')}>转回创作池</button>
              </div>
            </div>
          )}

          {isQuarantined && (
            <div className={styles.quarantineAlert} role="alert" aria-label="数据保护隔离告警">
              <strong>保护性隔离告警：{instance.quarantine_reason === 'aborted_unsynced' ? '检测到原生 ComfyUI 存在未上传的 Drive 缓存' : '域转移状态未确认'}</strong>
              <span>为避免未同步数据丢失，系统已中止清理且未自动销毁此 GPU（仍在产生 CU 计费）。请确认数据安全后，点击下方「停止」按钮手动释放实例。</span>
            </div>
          )}

          <div className={styles.detailGrid}>
            <div className={styles.metaCell}><span>所属域</span><b>{domainText}</b></div>
            <div className={styles.metaCell}><span>GPU</span><b>{gpuLabel(instance)}</b></div>
            <div className={styles.metaCell}><span>物理卡已运行</span><b>{formatVmElapsed(instance.vm_allocated_at, clockSec)}</b></div>
            <div className={styles.metaCell}><span>实例 ID</span><b title={id}>{shortId(id)}</b></div>
            <div className={styles.metaCell}><span>当前任务</span><b title={instance.current_job_id || ''}>{instance.current_job_id ? shortId(instance.current_job_id) : '空闲'}</b></div>
            <div className={styles.metaCell}><span>保留截止</span><b>{formatTime(instance.retained_until || instance.idle_deadline_at)}</b></div>
            {runtime && (runtime.workflow_key || runtime.process) && (
              <div className={styles.metaCell}><span>常驻工作流</span><b>{runtime.workflow_key ? `${runtime.workflow_key}${runtime.workflow_version != null ? ` v${runtime.workflow_version}` : ''}` : `进程 ${runtime.process}`}</b></div>
            )}
          </div>
          {showTransfer && instance.venv_isolated === false && (
            <p className={styles.muted}>旧版运行时环境，不支持域转移（新启动的 GPU 实例将自动支持）。</p>
          )}
          <div className={styles.cardActions}>
            <button type="button" className={styles.buttonSecondary} onClick={() => void openDetail(instance)}><ChevronDown size={15} aria-hidden />{isSelected ? '刷新详情' : '查看详情'}</button>
            {onContinue && canContinue && <button type="button" className={styles.button} onClick={() => onContinue(instance)}>续用</button>}
            {showTransfer && (
              <button
                type="button"
                className={styles.buttonSecondary}
                disabled={Boolean(transferBlock) || Boolean(busy)}
                title={transferBlock || '保留当前物理 GPU 并洁净重建目标域环境'}
                onClick={() => void openTransferModal(instance)}
              >
                <ArrowLeftRight size={15} aria-hidden />转移域
              </button>
            )}
            {onOpenWorkbench && <button type="button" className={styles.buttonSecondary} onClick={() => chooseWorkbench(instance)}><Wrench size={15} aria-hidden />{canMutate ? '工作台' : '工作台历史'}</button>}
            {canMutate && <button type="button" className={styles.buttonSecondary} disabled={!canRetain || busy === `retain:${id}`} title={isTransferring ? '域转移期间不可修改保留策略' : undefined} onClick={() => setConfirm({ action: 'retain', instance })}><Timer size={15} aria-hidden />保留</button>}
            {canStop && <button type="button" className={styles.buttonDanger} disabled={busy === `stop:${id}`} title={isTransferring ? '将安全中止转移并释放实例' : undefined} onClick={() => setConfirm({ action: 'stop', instance })}><Square size={14} aria-hidden />{isQuarantined ? '强制停止' : '停止'}</button>}
          </div>
          {isSelected && detail && <div className={styles.detailPanel} aria-label={`${label(instance)}详情`}>
            <div className={styles.detailHeader}><strong>实例详情</strong><span className={styles.muted}>版本 {String(detail.version ?? '—')} · 创建于 {formatTime(detail.created_at)}</span></div>
            <div className={styles.jobList}>{(detail.jobs || []).map((job, index) => <div className={styles.jobItem} key={String(job.id || index)}><b>{String(job.title || job.workflow_key || '任务')} · {shortId(String(job.id || ''))}</b><span>{stateLabel(String(job.state || '状态未知'))} · {formatTime(typeof job.created_at === 'string' ? job.created_at : null)}</span></div>)}</div>
            {!detail.jobs?.length && <p className={styles.muted}>暂无关联任务。</p>}
            <div className={styles.eventList}>{(detail.events || []).slice(-5).map((event, index) => <div className={styles.eventItem} key={String(event.id || index)}><b>{String(event.message || event.event_type || event.state || '事件')}</b><span>{formatTime(typeof event.created_at === 'string' ? event.created_at : null)}</span></div>)}</div>
            {!detail.events?.length && <p className={styles.muted}>暂无实例事件。</p>}
          </div>}
        </article>;
      })}
    </div>}
    {filtered.length > COLLAPSED_LIMIT && <button type="button" className={styles.buttonSecondary} aria-controls="instance-center-list"
      aria-expanded={expanded} onClick={() => setExpanded((current) => !current)}>
      {expanded ? '收起，仅显示最新 5 条' : `展开其余 ${filtered.length - COLLAPSED_LIMIT} 条`}
    </button>}

    {transferModal && (
      <div className={styles.sheetBackdrop} role="presentation" onClick={() => { if (!transferSubmitting) setTransferModal(null); }}>
        <div className={`${styles.sheet} ${styles.cliWizardSheet}`} role="dialog" aria-modal="true" aria-labelledby="instance-transfer-title" onClick={(e) => e.stopPropagation()}>
          <h3 id="instance-transfer-title">转移 GPU 实例所属域</h3>
          <p className={styles.muted}>
            当前实例：{label(transferModal.instance)} · 当前域：{domainBadgeLabel(transferModal.instance)} · GPU：{gpuLabel(transferModal.instance)}
          </p>
          {transferPreviewLoading && <p className={styles.muted} role="status">正在预检可转移目标域…</p>}
          {transferPreview && !transferPreview.transferable && transferPreview.blocked_reason && (
            <p className={styles.error} role="alert">{transferPreview.blocked_reason}</p>
          )}
          {transferPreview && (
            <>
              <div className={styles.transferTargetList} role="radiogroup" aria-label="选择目标域">
                {(['creation', 'native', 'ephemeral'] as TransferPool[])
                  .filter((pool) => Boolean(transferPreview.targets?.[pool]))
                  .map((pool) => {
                    const info = transferPreview.targets[pool]!;
                    const selected = transferTarget === pool;
                    return (
                      <button
                        key={pool}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        disabled={!info.allowed}
                        data-selected={selected ? 'true' : 'false'}
                        data-disabled={info.allowed ? 'false' : 'true'}
                        className={styles.transferTargetOption}
                        onClick={() => { if (info.allowed) { setTransferTarget(pool); setTransferError(''); } }}
                      >
                        <strong>转移到 {poolLabel(pool)}</strong>
                        <span className={styles.muted}>
                          {info.allowed
                            ? (pool === 'creation' ? '洁净重建创作池环境，并保留 1~30 分钟空闲待命' : pool === 'native' ? '洁净清空旧域后在原卡拉起原生 ComfyUI + Tunnel' : '洁净清空旧域后保留 10 分钟，供工作流真实冒烟认领')
                            : (info.reason || '当前不可转入')}
                        </span>
                      </button>
                    );
                  })}
              </div>

              {transferTarget === 'creation' && transferPreview.targets.creation?.allowed && (
                <label className={styles.field}>
                  <span>创作池空闲保留时长（1~30 分钟）</span>
                  <input
                    type="number"
                    min={1}
                    max={30}
                    step={1}
                    value={transferCreationIdle}
                    onChange={(e) => setTransferCreationIdle(Number(e.target.value) || 15)}
                  />
                </label>
              )}

              {transferTarget === 'native' && transferPreview.targets.native?.allowed && (
                <div className={styles.stack}>
                  <label className={styles.field}>
                    <span>原生模型存储模式</span>
                    <select value={transferNativeMode} onChange={(e) => setTransferNativeMode(e.target.value as 'local' | 'drive_mount')}>
                      <option value="local">本地模式（local，按需高速下载）</option>
                      <option value="drive_mount">Google Drive 挂载模式（drive_mount）</option>
                    </select>
                  </label>
                  <div className={styles.transferFormGrid}>
                    <label className={styles.field}>
                      <span>空闲回收（分钟）</span>
                      <input type="number" min={1} step={1} value={transferNativeIdle} onChange={(e) => setTransferNativeIdle(Number(e.target.value) || 120)} />
                    </label>
                    <label className={styles.field}>
                      <span>会话上限（分钟）</span>
                      <input type="number" min={1} step={1} value={transferNativeMaxRuntime} onChange={(e) => setTransferNativeMaxRuntime(Number(e.target.value) || 720)} />
                    </label>
                  </div>
                  {transferNativeMode === 'drive_mount' && (
                    <div className={styles.transferFormGrid}>
                      <label className={styles.field}>
                        <span>VFS 缓存上限（GB）</span>
                        <input type="number" min={1} step={1} value={transferNativeCacheGb} onChange={(e) => setTransferNativeCacheGb(Number(e.target.value) || 20)} />
                      </label>
                      <label className={styles.field}>
                        <span>缓存保留（分钟）</span>
                        <input type="number" min={1} step={1} value={transferNativeCacheAge} onChange={(e) => setTransferNativeCacheAge(Number(e.target.value) || 60)} />
                      </label>
                    </div>
                  )}
                </div>
              )}

              {transferTarget === 'ephemeral' && transferPreview.targets.ephemeral?.allowed && (
                <p className={styles.muted}>转入冒烟域后将保留 10 分钟待命；请前往「资产 → 工作流」打开目标草稿启用向导直接认领该待命卡。</p>
              )}

              <ol className={styles.transferWarnings} aria-label="域转移重要提醒">
                <li>转移会彻底清空旧域全部进程、虚拟环境（venv）及本地模型权重目录，确保目标域 100% 洁净重建。</li>
                <li>底层 Google 账号（{transferPreview.account_alias || transferPreview.account_id || '原账号'}）与 GPU 规格保持不变。</li>
                <li>Colab 上游物理运行时累计总时长不会重置（当前物理卡已运行 {formatVmElapsed(transferPreview.vm_allocated_at, clockSec)}）。</li>
              </ol>
            </>
          )}
          {transferError && <p className={styles.error} role="alert">{transferError}</p>}
          <div className={styles.sheetActions}>
            <button type="button" className={styles.buttonSecondary} disabled={transferSubmitting} onClick={() => setTransferModal(null)}>取消</button>
            <button
              type="button"
              className={styles.button}
              disabled={transferPreviewLoading || transferSubmitting || !transferPreview?.transferable || !transferTarget || !transferPreview.targets?.[transferTarget]?.allowed}
              onClick={() => void submitTransfer()}
            >
              {transferSubmitting ? '提交转移中…' : '确认转移域'}
            </button>
          </div>
        </div>
      </div>
    )}

    {confirm && <div className={styles.sheetBackdrop} role="presentation"><div className={styles.sheet} role="dialog" aria-modal="true" aria-labelledby="instance-confirm-title"><h3 id="instance-confirm-title">{confirm.action === 'stop' ? (confirm.instance.state === 'quarantined' ? '确认强制停止隔离实例？' : '确认停止实例？') : '确认保留实例？'}</h3><p className={styles.muted}>{confirm.action === 'stop' ? (confirm.instance.state === 'transferring' ? '实例正在执行域转移，确认停止将安全中止转移并释放实例。' : confirm.instance.state === 'quarantined' ? '该实例处于数据保护隔离态，请确认已处理未同步缓存；强制停止将直接销毁远端 Colab GPU。' : '服务端会将实例标记为释放中；远端删除完成前不会显示为已删除。') : '实例会保持可续用状态，并按服务端策略在空闲后释放。'}</p>{confirm.action === 'retain' && <label className={styles.field}>空闲保留分钟数<input type="number" min={0} max={30} value={retainMinutes} onChange={(event) => setRetainMinutes(Number(event.target.value) || 0)} /></label>}{error && <p className={styles.error} role="alert">{error}</p>}<div className={styles.sheetActions}><button type="button" className={styles.buttonSecondary} onClick={() => { if (!busy) setConfirm(null); }}>取消</button><button type="button" className={confirm.action === 'stop' ? styles.buttonDanger : styles.button} disabled={Boolean(busy)} onClick={() => void submitAction()}>{busy ? '提交中…' : confirm.action === 'stop' ? '确认停止' : '确认保留'}</button></div></div></div>}
    <ConfirmDialog
      open={confirmClearHistory}
      title="清除已释放实例历史？"
      description={`将从列表隐藏 ${clearableCount} 条已释放的实例记录，关联任务和 CU 用量仍可查询。`}
      confirmLabel="确认清除"
      danger
      busy={clearingHistory}
      onCancel={() => setConfirmClearHistory(false)}
      onConfirm={() => void clearHistory()}
    />
  </section>;
}
