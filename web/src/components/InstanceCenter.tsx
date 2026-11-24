import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, RefreshCw, Square, Timer, Trash2, Wrench } from 'lucide-react';
import {
  apiMessage,
  cancelRecoveryPlan,
  clearGpuInstanceHistory,
  getGpuInstance,
  getRecoveryStatus,
  setInstanceRetention,
  stopInstance,
  type RecoveryPlan,
} from '../lib/gpu-api';
import type { GpuInstance } from './AccountDirectory';
import { InstanceWorkbench } from './InstanceWorkbench';
import { formatInstanceTime as formatTime } from '../lib/instance-time';
import { gpuLabel } from '../lib/gpu-label';
import { isCreationPoolInstance } from '../lib/instance-submission';
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
  onRefresh?: () => void;
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
    retained_idle: '保留空闲', releasing: '释放中', deleted: '已删除',
    unknown: '未知', cleanup_pending: '待清理', quarantined: '已隔离',
  } as Record<string, string>)[String(value || '')] || value || '状态未知';
}

function poolLabel(pool?: string): string {
  return ({
    native: '原生 ComfyUI',
    creation: '创作池',
    ephemeral: '一次性',
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
  return ['requested', 'allocating', 'ready', 'busy', 'retained_idle', 'releasing'].includes(String(instance.state || '').toLowerCase());
}

function isRetained(instance: GpuInstance): boolean {
  return instance.state === 'retained_idle' && !instance.runtime_deleted;
}

function errorText(cause: unknown): string {
  return apiMessage(cause);
}

export function InstanceCenter({ instances = [], onContinue, onOpenWorkbench, onRefresh }: InstanceCenterProps) {
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
  useBodyScrollLock(Boolean(confirm));
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

  useEffect(() => {
    if (!activePlans.some(plan => Boolean(plan.next_retry_at))) return;
    setClockSec(Math.floor(Date.now() / 1000));
    const timer = window.setInterval(() => setClockSec(Math.floor(Date.now() / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [activePlans]);

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

  const rows = useMemo(() => {
    const seen = new Set<string>();
    return instances.map(normalizedInstance).filter((item) => {
      const id = instanceId(item);
      if (!id || seen.has(id)) return false;
      seen.add(id);
      return true;
    }).sort((a, b) => (Date.parse(b.created_at || '') || 0) - (Date.parse(a.created_at || '') || 0));
  }, [instances]);

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
    // The parent is the authoritative list owner. The returned row is applied
    // only after the server confirms the write; refresh then reconciles it.
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
      // Keep the same key and confirmation open: retrying is safe and is not
      // presented as success until the backend returns a result.
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
      <div><p className={styles.eyebrow}>资源管理</p><h2>实例中心</h2><p className={styles.muted}>查看真实实例状态；停止和保留均以服务端返回为准。</p></div>
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
        const canMutate = !isHistory(instance) && instance.state !== 'releasing';
        const canContinue = canMutate && isCreationPoolInstance(instance);
        const state = instance.remote_delete_source === 'provider_runtime_missing' ? '已回收' : stateLabel(instance.state);
        const runtime = instance.creation_runtime as { workflow_key?: string; workflow_version?: string | number; process?: string } | null | undefined;
        return <article className={styles.instanceCard} key={id}>
          <div className={styles.cardTop}>
            <div className={styles.minWidth}>
              <h3 title={id}>{label(instance)}</h3>
              <p className={styles.muted}>
                {poolLabel(instance.instance_pool)} · {instance.account_alias_snapshot || (instance.account_id ? `账号 ${shortId(instance.account_id)}` : '账号未确认')} · {instance.provider || '运行时未确认'}
                {instance.parent_instance_id ? ` · 接替 ${shortId(instance.parent_instance_id)}` : ''}
                {instance.replaced_by_instance_id ? ` · 已由 ${shortId(instance.replaced_by_instance_id)} 接替` : ''}
              </p>
            </div>
            <div className={styles.badgeGroup}>
              {instance.recovery_source === 'auto_recovery' && (
                <span className={`${styles.badge} ${styles.recoveryBadge}`} title={`恢复组 ${instance.recovery_group_id || ''}`}>
                  自动恢复 #{instance.recovery_generation ?? 1}
                </span>
              )}
              <span className={`${styles.badge} ${instance.state === 'ready' || instance.state === 'busy' || instance.state === 'retained_idle' ? styles.good : ''}`}>{state}</span>
            </div>
          </div>
          <div className={styles.detailGrid}>
            <div className={styles.metaCell}><span>GPU</span><b>{gpuLabel(instance)}</b></div>
            <div className={styles.metaCell}><span>实例 ID</span><b title={id}>{shortId(id)}</b></div>
            <div className={styles.metaCell}><span>当前任务</span><b title={instance.current_job_id || ''}>{instance.current_job_id ? shortId(instance.current_job_id) : '空闲'}</b></div>
            <div className={styles.metaCell}><span>保留截止</span><b>{formatTime(instance.retained_until || instance.idle_deadline_at)}</b></div>
            {runtime && (runtime.workflow_key || runtime.process) && (
              <div className={styles.metaCell}><span>常驻工作流</span><b>{runtime.workflow_key ? `${runtime.workflow_key}${runtime.workflow_version != null ? ` v${runtime.workflow_version}` : ''}` : `进程 ${runtime.process}`}</b></div>
            )}
          </div>
          <div className={styles.cardActions}>
            <button type="button" className={styles.buttonSecondary} onClick={() => void openDetail(instance)}><ChevronDown size={15} aria-hidden />{isSelected ? '刷新详情' : '查看详情'}</button>
            {onContinue && canContinue && <button type="button" className={styles.button} onClick={() => onContinue(instance)}>续用</button>}
            {onOpenWorkbench && <button type="button" className={styles.buttonSecondary} onClick={() => chooseWorkbench(instance)}><Wrench size={15} aria-hidden />{canMutate ? '工作台' : '工作台历史'}</button>}
            {canMutate && <button type="button" className={styles.buttonSecondary} disabled={busy === `retain:${id}`} onClick={() => setConfirm({ action: 'retain', instance })}><Timer size={15} aria-hidden />保留</button>}
            {canMutate && <button type="button" className={styles.buttonDanger} disabled={busy === `stop:${id}`} onClick={() => setConfirm({ action: 'stop', instance })}><Square size={14} aria-hidden />停止</button>}
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
    {confirm && <div className={styles.sheetBackdrop} role="presentation"><div className={styles.sheet} role="dialog" aria-modal="true" aria-labelledby="instance-confirm-title"><h3 id="instance-confirm-title">{confirm.action === 'stop' ? '确认停止实例？' : '确认保留实例？'}</h3><p className={styles.muted}>{confirm.action === 'stop' ? '服务端会将实例标记为释放中；远端删除完成前不会显示为已删除。' : '实例会保持可续用状态，并按服务端策略在空闲后释放。'}</p>{confirm.action === 'retain' && <label className={styles.field}>空闲保留分钟数<input type="number" min={0} max={30} value={retainMinutes} onChange={(event) => setRetainMinutes(Number(event.target.value) || 0)} /></label>}{error && <p className={styles.error} role="alert">{error}</p>}<div className={styles.sheetActions}><button type="button" className={styles.buttonSecondary} onClick={() => { if (!busy) setConfirm(null); }}>取消</button><button type="button" className={confirm.action === 'stop' ? styles.buttonDanger : styles.button} disabled={Boolean(busy)} onClick={() => void submitAction()}>{busy ? '提交中…' : confirm.action === 'stop' ? '确认停止' : '确认保留'}</button></div></div></div>}
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
