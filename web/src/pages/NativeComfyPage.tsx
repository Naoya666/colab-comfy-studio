import { useEffect, useRef, useState } from 'react';
import { apiFetch, apiMessage } from '../lib/gpu-api';
import { WorkflowGpuSelection } from '../components/WorkflowGpuSelection';
import type { GpuAccount } from '../components/AccountDirectory';
import { submitNativeHandoff } from '../lib/native-handoff';

type Timeouts = { retain_idle_minutes: number; max_runtime_minutes: number };
type NativeStatus = { configured: boolean; ready: boolean; state: string; url: string; instance_id?: string; error?: string; timeouts: { defaults: Timeouts }; events: { state?: string; message?: string }[] };

export function NativeComfyPage({ accounts }: { accounts: GpuAccount[] }) {
  const [account, setAccount] = useState(accounts.find(a => a.enabled !== false && a.allow_allocation !== false)?.account_id || '');
  const [spec, setSpec] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState<NativeStatus | null>(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [stopConfirm, setStopConfirm] = useState(false);
  const [autoOpen, setAutoOpen] = useState(false);
  const [idleMinutes, setIdleMinutes] = useState('');
  const [runtimeMinutes, setRuntimeMinutes] = useState('');
  const timeoutsInitialized = useRef(false);
  const key = useRef(crypto.randomUUID());
  const opening = useRef(false);
  async function enter() {
    if (opening.current) return;
    opening.current = true;
    try {
      const data = await apiFetch<{ url: string; ticket: string }>('/api/gpu/native/ticket', { method: 'POST' });
      submitNativeHandoff(document, data);
    } catch (cause) { setError(apiMessage(cause)); opening.current = false; }
  }
  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const data = await apiFetch<NativeStatus>('/api/gpu/native/status');
        if (!live) return;
        setStatus(data);
        if (!timeoutsInitialized.current) {
          setIdleMinutes(String(data.timeouts.defaults.retain_idle_minutes));
          setRuntimeMinutes(String(data.timeouts.defaults.max_runtime_minutes));
          timeoutsInitialized.current = true;
        }
        if (data.ready && autoOpen) { setAutoOpen(false); await enter(); }
      } catch (cause) { if (live) setError(apiMessage(cause)); }
      if (live) timer = setTimeout(() => void poll(), 3000);
    }
    void poll();
    return () => { live = false; clearTimeout(timer); };
  }, [autoOpen]);
  const validTimeouts = [idleMinutes, runtimeMinutes].every(value => Number.isInteger(Number(value)) && Number(value) >= 1);
  async function start() {
    if (!validTimeouts) { setError('请填写正整数分钟数。'); return; }
    setStarting(true); setError('');
    try {
      await apiFetch('/api/gpu/native/start', { method: 'POST', headers: { 'Idempotency-Key': key.current }, body: JSON.stringify({ account_id: account, spec: { key: spec }, retain_idle_minutes: Number(idleMinutes), max_runtime_minutes: Number(runtimeMinutes) }) });
      key.current = crypto.randomUUID();
      setAutoOpen(true);
    } catch (cause) { setError(apiMessage(cause)); }
    finally { setStarting(false); }
  }
  return <section className="card">
    <h1>原生 ComfyUI</h1>
    <p>打开官方节点编辑器。可以直接导入工作流 JSON 或拖入带工作流的原图，使用原生保存、扩展和 Manager。</p>
    <p className="muted">与 Studio 共用 GPU。工作流、设置、素材、结果、节点源码和模型保存至 Google Drive。页面打开时仍会按任务活动判断空闲。</p>
    {status?.ready ? <button className="button" onClick={() => void enter()}>进入原生 ComfyUI</button> : <>
      <label className="field">Google 账号<select value={account} onChange={e => { setAccount(e.target.value); key.current = crypto.randomUUID(); }}>
        <option value="">选择账号</option>
        {accounts.filter(a => a.enabled !== false && a.allow_allocation !== false).map(a => <option key={a.account_id} value={a.account_id}>{a.alias || a.account_id}</option>)}
      </select></label>
      <WorkflowGpuSelection manifest={null} accountId={account} value={spec} onChange={value => { setSpec(value); key.current = crypto.randomUUID(); }} />
      <label className="field">空闲回收时间（分钟）
        <input type="number" min={1} step={1} required value={idleMinutes} onChange={e => { setIdleMinutes(e.target.value); key.current = crypto.randomUUID(); }} />
        <span className="muted small">填写正整数分钟数，无最大值限制；生成、安装和同步等活动结束后开始计时。</span>
      </label>
      <label className="field">会话上限时间（分钟）
        <input type="number" min={1} step={1} required value={runtimeMinutes} onChange={e => { setRuntimeMinutes(e.target.value); key.current = crypto.randomUUID(); }} />
        <span className="muted small">填写正整数分钟数，无最大值限制，从提交启动时计时；复用 GPU 时仍受原实例截止时间限制。</span>
      </label>
      <p className="muted small">参数在本次启动时生效。到期后停止接收新任务，等待生成和同步完成后释放 GPU。调整已有原生会话的参数需先停止旧实例。</p>
      <button className="button" disabled={starting || !account || !spec || !status?.configured || !validTimeouts} onClick={() => void start()}>{starting ? '正在提交…' : '启动 / 复用 GPU 并进入'}</button>
      {status && !status.configured && <p className="status">请按部署文档配置原生 ComfyUI 域名和隧道。</p>}
    </>}
    {status && <p role="status">{status.ready ? '服务就绪' : `当前状态：${status.state}`}</p>}
    {(error || status?.error) && <p className="status error" role="alert">{error || status?.error}</p>}
    <ol aria-label="启动进度">{status?.events.map((event, i) => <li key={i}>{event.message}</li>)}</ol>
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
  </section>;
}
