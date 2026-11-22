import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { apiFetch, apiMessage, type GpuUsage } from '../lib/gpu-api';
import { WorkflowGpuSelection } from '../components/WorkflowGpuSelection';
import { NativeModelSettings, type ModelStorage } from '../components/NativeModelSettings';
import { NativeModelPreparation } from '../components/NativeModelPreparation';
import { NativeApiSettings } from '../components/NativeApiSettings';
import { NativeSetupWizard } from '../components/NativeSetupWizard';
import type { GpuAccount, GpuInstance } from '../components/AccountDirectory';
import { formatInstanceTime } from '../lib/instance-time';
import { submitNativeHandoff } from '../lib/native-handoff';
import { jobStateLabel } from '../lib/job-stage-rows';
import { cuNumber, terminalStates } from '../lib/ui-helpers';
import { useSmartPoll } from '../lib/smart-poll';
import styles from '../gpu.module.css';

const EVENTS_COLLAPSED_LIMIT = 5;
const NATIVE_PREFS_KEY = 'studio-native-comfy-prefs';

type Timeouts = { retain_idle_minutes: number; max_runtime_minutes: number };
type NativePrefs = {
  account?: string;
  spec?: Record<string, unknown> | null;
  storage?: ModelStorage;
  idleMinutes?: string;
  runtimeMinutes?: string;
};

function isValidStorage(value: unknown): value is ModelStorage {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const item = value as Record<string, unknown>;
  return (item.model_storage_mode === 'local' || item.model_storage_mode === 'drive_mount')
    && typeof item.cache_gb === 'number' && Number.isInteger(item.cache_gb) && item.cache_gb >= 1 && item.cache_gb <= 2147483647
    && typeof item.cache_max_age_minutes === 'number' && Number.isInteger(item.cache_max_age_minutes)
    && item.cache_max_age_minutes >= 1 && item.cache_max_age_minutes <= 153722867;
}

function isPositiveIntText(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '' && Number.isInteger(Number(value)) && Number(value) >= 1;
}

function loadNativePrefs(): NativePrefs {
  try {
    const raw = localStorage.getItem(NATIVE_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
    const spec = parsed.spec && typeof parsed.spec === 'object' && !Array.isArray(parsed.spec)
      ? parsed.spec as Record<string, unknown>
      : null;
    return {
      account: typeof parsed.account === 'string' ? parsed.account : undefined,
      spec,
      storage: isValidStorage(parsed.storage) ? parsed.storage : undefined,
      idleMinutes: isPositiveIntText(parsed.idleMinutes) ? parsed.idleMinutes : undefined,
      runtimeMinutes: isPositiveIntText(parsed.runtimeMinutes) ? parsed.runtimeMinutes : undefined,
    };
  } catch {
    return {};
  }
}

function saveNativePrefs(prefs: NativePrefs) {
  try { localStorage.setItem(NATIVE_PREFS_KEY, JSON.stringify(prefs)); } catch { /* browser storage unavailable */ }
}
type RestartPhase = 'checking' | 'syncing' | 'rebooting' | 'recovering' | 'succeeded' | 'failed';
type NativeRestart = { restart_id: string; instance_id: string; phase: RestartPhase; failed_phase?: RestartPhase; error?: string };
type NativeStatus = { configured: boolean; tunnel_configured?: boolean; web_ui_configured?: boolean; url?: string; ready: boolean; can_restart: boolean; restart?: NativeRestart; state: string; instance_id?: string; job_id?: string | null; error?: string; timeouts: { defaults: Timeouts }; model_storage_defaults: ModelStorage; model_storage?: ModelStorage; model_storage_known?: boolean; disk?: { disk_free_bytes?: number; vfs_cache_bytes?: number }; events: { state?: string; message?: string }[] };
const restartMessages: Record<RestartPhase, string> = {
  checking: '检查生成、安装和 Drive 同步状态',
  syncing: '同步模型索引',
  rebooting: '重启 ComfyUI',
  recovering: '等待服务恢复',
  succeeded: 'ComfyUI 已重启，模型列表已刷新',
  failed: 'ComfyUI 重启失败',
};

export function NativeComfyPage({ usage, accounts, accountsLoaded, instances, desktop, onOpenSettings }: {
  usage: GpuUsage | null; accounts: GpuAccount[]; accountsLoaded: boolean; instances: GpuInstance[]; desktop: boolean; onOpenSettings?: () => void;
}) {
  const initialPrefs = useMemo(() => loadNativePrefs(), []);
  const eligibleAccounts = useMemo(
    () => accounts.filter(a => a.enabled !== false && a.allow_allocation !== false),
    [accounts],
  );
  const [account, setAccount] = useState(() => {
    if (initialPrefs.account) {
      if (!accountsLoaded || eligibleAccounts.some(a => a.account_id === initialPrefs.account)) return initialPrefs.account;
    }
    return eligibleAccounts[0]?.account_id || '';
  });
  const [spec, setSpec] = useState<Record<string, unknown> | null>(() => initialPrefs.spec ?? null);
  const [status, setStatus] = useState<NativeStatus | null>(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [pendingStart, setPendingStart] = useState(false);
  const [entering, setEntering] = useState(false);
  const [stopConfirm, setStopConfirm] = useState(false);
  const [submittingRestart, setSubmittingRestart] = useState(false);
  const [pendingRestart, setPendingRestart] = useState<{ id: string; previous?: string }>();
  const [autoOpen, setAutoOpen] = useState(false);
  const [showAllEvents, setShowAllEvents] = useState(false);
  const [storage, setStorage] = useState<ModelStorage | undefined>(() => initialPrefs.storage);
  const [idleMinutes, setIdleMinutes] = useState(() => initialPrefs.idleMinutes ?? '');
  const [runtimeMinutes, setRuntimeMinutes] = useState(() => initialPrefs.runtimeMinutes ?? '');
  const [wizardRequest, setWizardRequest] = useState<{ seq: number; step?: 1 | 2 | 3 | 4 | 5 } | null>(null);
  const timeoutsInitialized = useRef(Boolean(initialPrefs.storage && initialPrefs.idleMinutes && initialPrefs.runtimeMinutes));
  const accountEligible = eligibleAccounts.some(a => a.account_id === account);

  useEffect(() => {
    if (!accountsLoaded || accountEligible) return;
    const fallback = eligibleAccounts[0]?.account_id || '';
    if (fallback !== account) setAccount(fallback);
  }, [accountsLoaded, accountEligible, eligibleAccounts, account]);

  useEffect(() => {
    const next: NativePrefs = {};
    if (account) next.account = account;
    if (spec) next.spec = spec;
    if (isValidStorage(storage)) next.storage = storage;
    if (isPositiveIntText(idleMinutes)) next.idleMinutes = idleMinutes;
    if (isPositiveIntText(runtimeMinutes)) next.runtimeMinutes = runtimeMinutes;
    saveNativePrefs(next);
  }, [account, spec, storage, idleMinutes, runtimeMinutes]);
  const key = useRef(crypto.randomUUID());
  const opening = useRef(false);
  const restartDialog = useRef<HTMLDialogElement>(null);
  const sendingRestart = useRef(false);
  const sendingStart = useRef(false);
  const scheduledSession = usage?.active_sessions?.find(session => session.instance_id === status?.instance_id);
  const selectedBalance = usage?.accounts?.find(row => row.account_id === account);
  const currentGuard = usage?.guard;
  const blocked = currentGuard?.soft_tripped || selectedBalance?.below_floor;

  async function enter(newTab: boolean, signal?: AbortSignal) {
    if (opening.current || signal?.aborted) return;
    const popup = newTab ? window.open('', '_blank') : null;
    if (newTab && !popup) {
      setError('无法打开新标签页，请允许此站点打开弹出窗口后重试。');
      return;
    }
    opening.current = true;
    setEntering(true);
    setError('');
    try {
      if (popup) {
        popup.opener = null;
        popup.document.title = '原生 ComfyUI';
        popup.document.body.textContent = '正在连接原生 ComfyUI…';
      }
      const data = await apiFetch<{ url: string; ticket: string }>('/api/gpu/native/ticket', { method: 'POST', signal });
      if (signal?.aborted) return;
      if (popup?.closed) throw new Error('新标签页已关闭，请重新打开。');
      submitNativeHandoff(popup?.document ?? document, data);
    } catch (cause) {
      popup?.close();
      if (!signal?.aborted) setError(apiMessage(cause));
    } finally {
      opening.current = false;
      setEntering(false);
    }
  }
  const pollStatus = useCallback(async (signal: AbortSignal) => {
    try {
      const data = await apiFetch<NativeStatus>('/api/gpu/native/status', { signal });
      if (signal.aborted) return;
      setStatus(data);
      setPendingStart(false);
      setPendingRestart(pending => {
        const observed = data.restart?.restart_id;
        return pending && data.instance_id && observed !== pending.id && observed === pending.previous ? pending : undefined;
      });
      if (!timeoutsInitialized.current) {
        setStorage(current => current ?? data.model_storage_defaults);
        setIdleMinutes(current => current || String(data.timeouts.defaults.retain_idle_minutes));
        setRuntimeMinutes(current => current || String(data.timeouts.defaults.max_runtime_minutes));
        timeoutsInitialized.current = true;
      }
      if (data.ready && autoOpen) {
        setAutoOpen(false);
        if (!desktop && data.web_ui_configured !== false) await enter(false, signal);
      }
    } catch (cause) { if (!signal.aborted) setError(apiMessage(cause)); }
  }, [autoOpen, desktop]);

  const nativeActive = Boolean(
    pendingStart || pendingRestart || (status && status.state !== 'stopped' && !terminalStates.has(status.state)) || status?.ready,
  );
  useSmartPoll(pollStatus, { enabled: true, active: nativeActive, activeMs: 3000, idleMs: 20000 });
  const startupPending = Boolean(pendingStart || (status?.job_id && !status.ready && status.state !== 'stopped' && !terminalStates.has(status.state)));
  const validTimeouts = [idleMinutes, runtimeMinutes].every(value => Number.isInteger(Number(value)) && Number(value) >= 1);
  const validStorage = storage && Number.isInteger(storage.cache_gb) && storage.cache_gb >= 1
    && storage.cache_gb <= 2147483647 && Number.isInteger(storage.cache_max_age_minutes)
    && storage.cache_max_age_minutes >= 1 && storage.cache_max_age_minutes <= 153722867;
  const effective = status?.model_storage;
  const instance = status?.instance_id && instances.find(item => (item.id || item.instance_id) === status.instance_id);
  const restart = status?.restart;
  const restartInProgress = submittingRestart || Boolean(pendingRestart)
    || restart?.phase === 'checking' || restart?.phase === 'syncing'
    || restart?.phase === 'rebooting' || restart?.phase === 'recovering';
  const restartError = restart?.phase === 'failed'
    ? `${restartMessages[restart.failed_phase || 'failed']}：${restart.error || 'ComfyUI 重启失败'}`
    : '';
  async function restartNative() {
    const instance = status?.instance_id;
    if (!instance || !status?.can_restart || sendingRestart.current) return;
    sendingRestart.current = true;
    restartDialog.current?.close();
    setSubmittingRestart(true);
    setError('');
    try {
      const result = await apiFetch<{ restart_id: string }>('/api/gpu/native/restart', {
        method: 'POST',
        headers: { 'Idempotency-Key': `native-restart:${instance}:${restart?.restart_id || 'first'}` },
        body: JSON.stringify({ instance_id: instance }),
      }, 40_000);
      setPendingRestart({ id: result.restart_id, previous: restart?.restart_id });
    } catch (cause) { setError(apiMessage(cause)); }
    finally { sendingRestart.current = false; setSubmittingRestart(false); }
  }
  async function start() {
    if (sendingStart.current || startupPending || !accountsLoaded || !accountEligible) return;
    if (!validTimeouts) { setError('请填写正整数分钟数。'); return; }
    sendingStart.current = true;
    setStarting(true); setError('');
    try {
      await apiFetch('/api/gpu/native/start', { method: 'POST', headers: { 'Idempotency-Key': key.current }, body: JSON.stringify({ account_id: account, spec: { key: spec }, retain_idle_minutes: Number(idleMinutes), max_runtime_minutes: Number(runtimeMinutes), ...storage }) });
      key.current = crypto.randomUUID();
      setPendingStart(true);
      setAutoOpen(!desktop);
    } catch (cause) { setError(apiMessage(cause)); }
    finally { sendingStart.current = false; setStarting(false); }
  }
  const openSetupWizard = useCallback((step?: 1 | 2 | 3 | 4 | 5) => {
    setWizardRequest(current => ({ seq: (current?.seq || 0) + 1, step }));
  }, []);
  const refreshStatusNow = useCallback(() => {
    const controller = new AbortController();
    void pollStatus(controller.signal);
  }, [pollStatus]);
  const webUiReady = status?.web_ui_configured !== false;
  return <section className={styles.nativeWorkspace}>
    <div className="actions nativeApiHeading">
      <h1>原生 ComfyUI</h1>
      <div className="actions">
        <NativeSetupWizard
          configured={status?.configured}
          webUiConfigured={status?.web_ui_configured}
          openRequest={wizardRequest}
          onSaved={refreshStatusNow}
          onOpenSettings={onOpenSettings}
        />
        <NativeApiSettings accounts={accounts} />
      </div>
    </div>
    <p>打开官方节点编辑器。可以直接导入工作流 JSON 或拖入带工作流的原图，使用原生保存、扩展和 Manager。</p>
    {status && (!status.configured || status.tunnel_configured === false) && (
      <div className={styles.wizardBanner} role="region" aria-label="原生 ComfyUI 初始配置引导">
        <div>
          <strong>首次使用？按向导完成 Cloudflare 隧道与凭据配置</strong>
          <p className="muted small">
            无需手动编辑服务器 .env 或设置 ./secrets 文件权限。通过向导填入 Cloudflare Named Tunnel、Access Service Token 与版本即可加密同步至后端并执行免开机预检；仅给酒馆调用 API 时还可省去第二个公网域名。
          </p>
        </div>
        <button type="button" className="button" onClick={() => openSetupWizard(1)}>打开配置向导</button>
      </div>
    )}
    {scheduledSession && <div className={styles.inlinePanel} role="status">
      已运行 {scheduledSession.elapsed_minutes.toFixed(1)} 分钟 · 实扣 {cuNumber(scheduledSession.actual_cu)} CU
      {scheduledSession.draining && <p className={styles.cuWarning}>正在排空会话，队列与 Drive 同步空闲后释放。</p>}
    </div>}
    <div className={styles.nativeColumns}>
      <div className={styles.nativeControls}>
        {desktop && <h2>{status?.ready || restartInProgress ? '当前会话' : '会话配置'}</h2>}
        {desktop && instance && <dl className={styles.kv}>
          <dt>GPU</dt><dd>{instance.gpu_reported || '型号待确认'}</dd>
          {instance.hard_deadline_at && <><dt>会话到期</dt><dd>{formatInstanceTime(instance.hard_deadline_at)}</dd></>}
          {instance.idle_deadline_at && <><dt>空闲回收</dt><dd>{formatInstanceTime(instance.idle_deadline_at)}</dd></>}
        </dl>}
        {effective && <div className="status">
          <p>{status?.ready ? '当前模型存储' : '本次启动配置'} · {effective.model_storage_mode === 'local' ? '本地模式' : 'Drive 挂载模式'}</p>
          {effective.model_storage_mode === 'drive_mount' && status?.model_storage_known === false &&
            <p>旧会话未记录实际 VFS 参数。重新启动后可查看。</p>}
          {effective.model_storage_mode === 'drive_mount' && status?.model_storage_known !== false &&
            <p>VFS 缓存 {effective.cache_gb} GB · 未访问后保留 {effective.cache_max_age_minutes} 分钟
              {status?.disk?.vfs_cache_bytes !== undefined && ` · 已占用 ${(status.disk.vfs_cache_bytes / 1024 ** 3).toFixed(2)} GiB`}</p>}
          {status?.disk?.disk_free_bytes !== undefined && <p>本地磁盘剩余 {(status.disk.disk_free_bytes / 1024 ** 3).toFixed(2)} GiB</p>}
          <p className="muted small">修改模型存储模式和缓存参数需停止当前实例后重新启动。</p>
        </div>}
        {status?.ready && effective?.model_storage_mode === 'local' && <NativeModelPreparation />}
        {(status?.ready || restartInProgress || (status?.instance_id && restart)) ? <>
          <div className="actions">
            {webUiReady ? (
              <button className="button" disabled={!status?.ready || restartInProgress || entering}
                title={desktop ? '在新标签页打开' : undefined} onClick={() => void enter(desktop)}>
                {entering ? '正在连接…' : '进入原生 ComfyUI'}{desktop && <ArrowUpRight size={16} aria-hidden />}
              </button>
            ) : (
              <button className="button secondary" type="button" onClick={() => openSetupWizard(4)}>
                补充网页域名以打开官方 UI
              </button>
            )}
            <button className="button secondary" disabled={!status?.can_restart || restartInProgress}
              onClick={() => restartDialog.current?.showModal()}>{restartInProgress ? '正在重启…' : '重启原生 ComfyUI'}</button>
          </div>
          {!webUiReady && (
            <p className="muted small">当前为「仅 API 模式」（未配置独立原生网页域名），可直接点右上角「API 设置」复制路径短码或独立端口地址供酒馆使用。</p>
          )}
        </> : <>
          <label className="field">Google 账号<select value={account} onChange={e => { setAccount(e.target.value); key.current = crypto.randomUUID(); }}>
            <option value="">选择账号</option>
            {eligibleAccounts.map(a => <option key={a.account_id} value={a.account_id}>{a.alias || a.account_id}</option>)}
          </select></label>
          <WorkflowGpuSelection manifest={null} accountId={account} value={spec} onChange={value => { setSpec(value); key.current = crypto.randomUUID(); }}
            rateHint={rate => <p className="muted small">{rate == null || rate <= 0 ? '估算费率未知' : <>
              {String(spec?.accelerator || 'GPU')} 约 {cuNumber(rate)} CU/小时
              {selectedBalance?.balance != null && !selectedBalance.error && ` · 当前余额约可运行 ${(selectedBalance.balance / rate).toFixed(1)} 小时`}
            </>}</p>} />
          {blocked && <p className="status error" role="alert">{selectedBalance?.below_floor ? currentGuard?.message : currentGuard?.daily_message}</p>}
          {storage && <NativeModelSettings value={storage} onChange={value => { setStorage(value); key.current = crypto.randomUUID(); }} />}
          <div className={styles.nativeFieldPair}>
            <label className="field">空闲回收时间（分钟）
              <input type="number" min={1} step={1} required value={idleMinutes} onChange={e => { setIdleMinutes(e.target.value); key.current = crypto.randomUUID(); }} />
              <span className="muted small">填写正整数分钟数，无最大值限制；生成、安装和同步等活动结束后开始计时。</span>
            </label>
            <label className="field">会话上限时间（分钟）
              <input type="number" min={1} step={1} required value={runtimeMinutes} onChange={e => { setRuntimeMinutes(e.target.value); key.current = crypto.randomUUID(); }} />
              <span className="muted small">填写正整数分钟数，无最大值限制，从提交启动时计时；复用 GPU 时仍受原实例截止时间限制。</span>
            </label>
          </div>
          <p className="muted small">参数在本次启动时生效。到期后停止接收新任务，等待生成和同步完成后释放 GPU。调整已有原生会话的参数需先停止旧实例。</p>
          <button className="button" disabled={!!blocked || starting || startupPending || !accountsLoaded || !accountEligible || !spec || !status?.configured || !validTimeouts || !validStorage} onClick={() => void start()}>
            {starting ? '正在提交…' : startupPending ? '正在启动…' : desktop || !webUiReady ? '启动 / 复用 GPU' : '启动 / 复用 GPU 并进入'}
          </button>
          {status && !status.configured && <p className="status">尚未完成 Cloudflare 隧道与凭据配置，请点击上方「打开配置向导」。</p>}
        </>}
        <dialog className="nativeRestartDialog" ref={restartDialog} aria-labelledby="native-restart-title" aria-describedby="native-restart-description">
          <h2 id="native-restart-title">重启原生 ComfyUI</h2>
          <p id="native-restart-description">{effective?.model_storage_mode === 'local'
            ? '将重启当前 GPU 实例上的 ComfyUI，本地模型文件继续保留。页面会短暂断开，请先保存工作流。'
            : '将重启当前 GPU 实例上的 ComfyUI，并同步已就绪的模型。页面会短暂断开，请先保存工作流。'}</p>
          <div className="actions">
            <button type="button" className="button secondary" autoFocus onClick={() => restartDialog.current?.close()}>取消</button>
            <button type="button" className="button" onClick={() => void restartNative()}>重启</button>
          </div>
        </dialog>
      </div>
      <section className={styles.nativeActivity} aria-label="会话活动">
        {desktop && <h2>会话活动</h2>}
        <p role="status">{restartInProgress
          ? (restart && !['failed', 'succeeded'].includes(restart.phase) ? restartMessages[restart.phase] : restartMessages.checking)
          : status?.ready && restart?.phase === 'succeeded' ? restartMessages.succeeded
          : status?.ready ? '服务就绪' : pendingStart ? '启动请求已提交' : !status ? '正在读取会话状态…'
          : status.state === 'stopped' ? '当前没有原生会话' : `当前状态：${jobStateLabel(status.state)}`}</p>
        {(error || restartError || status?.error) &&
          <p className="status error" role="alert">{error || restartError || status?.error}</p>}
        {Boolean(status?.events.length) && (() => {
          const events = status?.events || [];
          const hidden = Math.max(0, events.length - EVENTS_COLLAPSED_LIMIT);
          const shownEvents = showAllEvents || hidden === 0 ? events : events.slice(0, EVENTS_COLLAPSED_LIMIT);
          return <>
            <ol aria-label="启动进度">{shownEvents.map((event, i) => <li key={i}>{event.message}</li>)}</ol>
            {hidden > 0 && (
              <button type="button" className="button ghost small" onClick={() => setShowAllEvents(v => !v)}>
                {showAllEvents ? `收起早期事件，仅显示最新 ${EVENTS_COLLAPSED_LIMIT} 条` : `展开更早的 ${hidden} 条启动事件`}
              </button>
            )}
          </>;
        })()}
        {status?.instance_id && <div className="actions">
          <button className="button secondary" onClick={() => { setAutoOpen(false); setStopConfirm(true); }}>停止当前原生实例</button>
          {stopConfirm && <>
            <p>确认停止？正在生成或同步的任务完成后才会释放。切换 GPU 前请等待旧实例停止。</p>
            <button className="button danger" onClick={async () => {
              try {
                await apiFetch(`/api/gpu/instances/${encodeURIComponent(status.instance_id!)}/stop`, { method: 'POST', headers: { 'Idempotency-Key': `native-stop:${status.instance_id}` }, body: JSON.stringify({ confirmed: true }) });
                setStopConfirm(false);
              } catch (cause) { setError(apiMessage(cause)); }
            }}>确认停止</button>
          </>}
        </div>}
        <p className="muted small">与 Studio 共用 GPU。工作流、素材、结果和节点沿用 Google Drive 持久化。空闲回收按实际任务活动计时。</p>
      </section>
    </div>
  </section>;
}
