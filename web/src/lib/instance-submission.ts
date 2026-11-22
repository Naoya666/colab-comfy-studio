export type InstanceChoice = {
  accountId?: string | null;
  instanceId?: string | null;
};

export type InstanceSnapshot = {
  id?: string;
  instance_id?: string;
  instance_pool?: string | null;
  state?: string | null;
  account_id?: string | null;
  current_job_id?: string | null;
  runtime_deleted?: boolean | null;
  instance_label?: string | null;
  label?: string | null;
  gpu_reported?: string | null;
  retention_mode?: string | null;
  retain_idle_minutes?: number | null;
  idle_deadline_at?: string | number | null;
  hard_deadline_at?: string | number | null;
  creation_runtime?: Record<string, unknown> | null;
};

export type SubmissionDecision = {
  ok: boolean;
  allocation: 'auto' | 'existing';
  accountId: string | null;
  instanceId: string | null;
  message: string;
  machineLabel: string;
};

export type ReusePreviewLike = {
  decision:
    | 'unsupported_spec'
    | 'reuse_direct'
    | 'confirm_shorten'
    | 'confirm_busy_same_spec'
    | 'confirm_higher_spec'
    | 'confirm_lower_spec'
    | 'create_new';
  reason?: string | null;
  target_instance?: {
    instance_id: string;
    label?: string | null;
    account_id?: string | null;
    state?: string;
    is_busy?: boolean;
    spec_id?: string | null;
    gpu_reported?: string | null;
    retention_mode?: string;
    retain_idle_minutes?: number | null;
    idle_deadline_at?: number | null;
    warm_hit?: boolean;
  } | null;
  shortens_lifetime?: boolean;
  shorten_reasons?: string[];
  allowed_actions?: string[];
  force_confirm?: boolean;
};

export type ConfirmActionOption = {
  key: 'reuse' | 'replace' | 'new' | 'auto';
  label: string;
  reuseAction: 'auto' | 'reuse' | 'replace' | 'new';
  instanceId: string | null;
  replaceInstanceId: string | null;
  variant: 'primary' | 'secondary' | 'danger';
};

export type LifetimeShorteningResult = {
  shortens: boolean;
  reasons: string[];
  currentExpiry: number | null;
  newExpiry: number | null;
};

const REUSABLE = new Set(['ready', 'retained_idle']);
const RETAIN_MODES = new Set(['smart', 'retain']);
const FORCED_CONFIRM_DECISIONS = new Set([
  'unsupported_spec',
  'confirm_shorten',
  'confirm_busy_same_spec',
  'confirm_higher_spec',
  'confirm_lower_spec',
]);

export function instanceIdentity(instance: InstanceSnapshot): string {
  return instance.id || instance.instance_id || '';
}

export function isCreationPoolInstance(instance: InstanceSnapshot | null | undefined): boolean {
  if (!instance) return false;
  if (!instance.instance_pool) return true;
  return String(instance.instance_pool).toLowerCase() === 'creation';
}

export function instanceReusable(instance: InstanceSnapshot | null | undefined): boolean {
  if (!instance || instance.runtime_deleted) return false;
  if (!isCreationPoolInstance(instance)) return false;
  const state = String(instance.state || '').toLowerCase();
  if (!REUSABLE.has(state)) return false;
  if (instance.current_job_id) return false;
  return true;
}

export function instanceMachineLabel(instance: InstanceSnapshot | null | undefined): string {
  if (!instance) return '未选择';
  const name = instance.instance_label || instance.label || instanceIdentity(instance).slice(0, 8) || '机器';
  const gpu = instance.gpu_reported ? ` · ${instance.gpu_reported}` : '';
  const state = instance.state ? ` · ${instance.state}` : '';
  return `${name}${gpu}${state}`;
}

function toEpochSeconds(value: string | number | null | undefined): number | null {
  if (value == null || value === '') return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const parsed = Number(value);
  if (Number.isFinite(parsed)) return parsed;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms / 1000 : null;
}

/**
 * Dual-criteria lifetime shortening evaluation (Spec §5.2):
 * - Criterion A (Policy config): switching retained instance to save/batch, or lowering retain_idle_minutes.
 * - Criterion B (Actual expiry): new estimated expiry < current instance expiry.
 */
export function evaluateLifetimeShortening(
  instance: {
    retention_mode?: string | null;
    retain_idle_minutes?: number | null;
    idle_deadline_at?: string | number | null;
    state?: string | null;
    is_busy?: boolean;
  } | null | undefined,
  newLifecycle: 'save' | 'batch' | 'smart' | 'retain',
  newRetainIdleMinutes: number,
  nowSec: number = Math.floor(Date.now() / 1000),
): LifetimeShorteningResult {
  if (!instance) {
    return { shortens: false, reasons: [], currentExpiry: null, newExpiry: null };
  }
  const reasons: string[] = [];
  const oldMode = String(instance.retention_mode || 'save').toLowerCase();
  const oldIsRetained = RETAIN_MODES.has(oldMode);
  const newIsRetained = RETAIN_MODES.has(newLifecycle);
  const oldMinutes = typeof instance.retain_idle_minutes === 'number' && instance.retain_idle_minutes > 0
    ? instance.retain_idle_minutes
    : null;
  const newMinutes = Number.isFinite(newRetainIdleMinutes) && newRetainIdleMinutes > 0
    ? newRetainIdleMinutes
    : 0;
  const isBusy = Boolean(instance.is_busy) || ['busy', 'allocating', 'requested'].includes(String(instance.state || '').toLowerCase());

  // Criterion A: policy config shortening
  if (oldIsRetained && !newIsRetained) {
    reasons.push(`原实例采用保留策略（${oldMode}），本次「${newLifecycle}」模式将在任务结束后释放实例`);
  } else if (oldIsRetained && newIsRetained && oldMinutes != null && newMinutes < oldMinutes) {
    reasons.push(`空闲保留时长将由 ${oldMinutes} 分钟缩短为 ${newMinutes} 分钟`);
  }

  // Criterion B: actual expiry shortening
  const idleDeadline = toEpochSeconds(instance.idle_deadline_at);
  const currentExpiry = isBusy
    ? (oldIsRetained && oldMinutes != null ? nowSec + oldMinutes * 60 : idleDeadline)
    : idleDeadline;
  const newExpiry = newIsRetained ? nowSec + newMinutes * 60 : nowSec;
  if (currentExpiry != null && currentExpiry > nowSec && newExpiry < currentExpiry) {
    const oldRemainMin = Math.max(1, Math.round((currentExpiry - nowSec) / 60));
    if (!newIsRetained) {
      if (!reasons.length) {
        reasons.push(`原实例预计还可保留约 ${oldRemainMin} 分钟，本次任务结束后将立即释放`);
      }
    } else {
      reasons.push(`实际预计到期时间将缩短（原剩余约 ${oldRemainMin} 分钟，本次新策略约 ${newMinutes} 分钟）`);
    }
  }

  return {
    shortens: reasons.length > 0,
    reasons,
    currentExpiry,
    newExpiry,
  };
}

/**
 * Forced confirmation must ALWAYS bypass `skipConfirmSig` whenever:
 * - Spec conflict / busy conflict / unsupported spec
 * - Lifetime shortening (Criterion A or B)
 * - Backend `force_confirm === true`
 */
export function shouldBypassSkipConfirm(preview: ReusePreviewLike | null | undefined): boolean {
  if (!preview) return false;
  if (preview.force_confirm) return true;
  if (preview.shortens_lifetime) return true;
  return FORCED_CONFIRM_DECISIONS.has(preview.decision);
}

/**
 * Build deterministic modal action buttons for each reuse-preview decision (Spec §4.2 & §5.2):
 * - confirm_higher_spec: 3 buttons (reuse / replace / new)
 * - confirm_lower_spec: 2 buttons (replace / new)
 * - confirm_busy_same_spec: 2 buttons (reuse / new)
 * - confirm_shorten / reuse_direct / create_new: 1 confirm button
 * - unsupported_spec: 0 buttons (blocked)
 */
export function buildConfirmActions(preview: ReusePreviewLike | null | undefined): ConfirmActionOption[] {
  if (!preview) {
    return [{
      key: 'auto',
      label: '确认启动',
      reuseAction: 'auto',
      instanceId: null,
      replaceInstanceId: null,
      variant: 'primary',
    }];
  }
  const targetId = preview.target_instance?.instance_id || (preview.target_instance as { id?: string } | null | undefined)?.id || null;
  const targetGpu = preview.target_instance?.gpu_reported || preview.target_instance?.spec_id || '已有实例';
  switch (preview.decision) {
    case 'unsupported_spec':
      return [];
    case 'confirm_higher_spec':
      return [
        {
          key: 'reuse',
          label: `复用已有 ${targetGpu}`,
          reuseAction: 'reuse',
          instanceId: targetId,
          replaceInstanceId: null,
          variant: 'primary',
        },
        {
          key: 'replace',
          label: '停止旧机并新开目标规格',
          reuseAction: 'replace',
          instanceId: null,
          replaceInstanceId: targetId,
          variant: 'danger',
        },
        {
          key: 'new',
          label: '保留旧机，额外新开（双卡）',
          reuseAction: 'new',
          instanceId: null,
          replaceInstanceId: null,
          variant: 'secondary',
        },
      ];
    case 'confirm_lower_spec':
      return [
        {
          key: 'replace',
          label: '温和停止旧机并新开目标规格',
          reuseAction: 'replace',
          instanceId: null,
          replaceInstanceId: targetId,
          variant: 'danger',
        },
        {
          key: 'new',
          label: '保留旧机，额外新开（双卡）',
          reuseAction: 'new',
          instanceId: null,
          replaceInstanceId: null,
          variant: 'secondary',
        },
      ];
    case 'confirm_busy_same_spec':
      return [
        {
          key: 'reuse',
          label: '排队复用当前忙碌机器',
          reuseAction: 'reuse',
          instanceId: targetId,
          replaceInstanceId: null,
          variant: 'primary',
        },
        {
          key: 'new',
          label: '保留旧机，额外新开（双卡）',
          reuseAction: 'new',
          instanceId: null,
          replaceInstanceId: null,
          variant: 'secondary',
        },
      ];
    case 'confirm_shorten':
      return [{
        key: 'reuse',
        label: '确认复用并缩短保留策略',
        reuseAction: targetId ? 'reuse' : 'auto',
        instanceId: targetId,
        replaceInstanceId: null,
        variant: 'primary',
      }];
    case 'reuse_direct':
      return [{
        key: 'reuse',
        label: '确认复用并启动',
        reuseAction: 'auto',
        instanceId: null,
        replaceInstanceId: null,
        variant: 'primary',
      }];
    case 'create_new':
    default:
      return [{
        key: 'auto',
        label: '确认启动',
        reuseAction: 'auto',
        instanceId: null,
        replaceInstanceId: null,
        variant: 'primary',
      }];
  }
}

export function formatReuseDecisionTitle(preview: ReusePreviewLike | null | undefined): string {
  switch (preview?.decision) {
    case 'unsupported_spec':
      return '所选 GPU 规格不受支持';
    case 'confirm_higher_spec':
      return '已有更高规格创作实例';
    case 'confirm_lower_spec':
      return '当前仅有低规格创作实例';
    case 'confirm_busy_same_spec':
      return '同规格创作实例正在运行';
    case 'confirm_shorten':
      return '确认缩短实例保留时长';
    case 'reuse_direct':
      return '确认复用已有创作实例';
    default:
      return '确认开始生成';
  }
}

/**
 * A stale or released machine must not be submitted, and must not be silently
 * replaced by another account. The caller keeps the current account and asks
 * the user to confirm automatic allocation on a later click.
 */
export function resolveInstanceSubmission(choice: InstanceChoice, instances: InstanceSnapshot[]): SubmissionDecision {
  const accountId = choice.accountId || null;
  const instanceId = choice.instanceId || null;
  if (!instanceId) {
    return {
      ok: true,
      allocation: 'auto',
      accountId,
      instanceId: null,
      message: '',
      machineLabel: '自动分配',
    };
  }
  const selected = instances.find((instance) => instanceIdentity(instance) === instanceId) || null;
  if (instanceReusable(selected) && selected) {
    const owner = selected.account_id || null;
    if (accountId && owner && accountId !== owner) {
      return {
        ok: false,
        allocation: 'auto',
        accountId,
        instanceId: null,
        machineLabel: instanceMachineLabel(selected),
        message: '所选机器不属于当前账号。这次没有提交任务，也没有改用其他账号。请改选这台机器对应的账号，或清空机器后重新确认。',
      };
    }
    return {
      ok: true,
      allocation: 'existing',
      accountId: accountId || owner,
      instanceId,
      message: '',
      machineLabel: instanceMachineLabel(selected),
    };
  }
  const keptAccount = accountId || selected?.account_id || null;
  return {
    ok: false,
    allocation: 'auto',
    accountId: keptAccount,
    instanceId: null,
    machineLabel: selected ? instanceMachineLabel(selected) : '已失效的机器',
    message: keptAccount
      ? '所选 GPU 机器已释放或不能复用。这次没有提交任务，也没有换账号。请再点一次确认，按当前账号自动分配机器，或另选一台可用机器。'
      : '所选 GPU 机器已释放或不能复用。这次没有提交任务。请先选定账号，或再点一次确认后按已授权账号自动分配。',
  };
}
