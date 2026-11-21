import { useRef, useState, type FormEvent } from 'react';
import { apiFetch, apiMessage } from '../lib/gpu-api';
import type { GpuAccount } from './AccountDirectory';
import { NativeModelSettings, type ModelStorage } from './NativeModelSettings';
import { WorkflowGpuSelection } from './WorkflowGpuSelection';

type Policy = ModelStorage & {
  auto_start: boolean;
  account_id: string | null;
  gpu: Record<string, unknown> | null;
  retain_idle_minutes: number;
  max_runtime_minutes: number;
};
type ApiKey = {
  id: string;
  name: string;
  mode: 'trusted' | 'approved';
  auto_start: boolean;
  revoked_at: string | null;
  last_used_at: string | null;
  submitted: number;
};
type ApiSettings = {
  url: string;
  policy: Policy;
  keys: ApiKey[];
  recent_starts: { job_id: string; key_id: string; state: string; created_at: string }[];
};

const base = '/api/gpu/native';
const relay = 'http://127.0.0.1:8189';

export function NativeApiSettings({ accounts }: { accounts: GpuAccount[] }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [data, setData] = useState<ApiSettings>();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [token, setToken] = useState('');
  const [name, setName] = useState('');
  const [mode, setMode] = useState<ApiKey['mode']>('approved');
  const [workflow, setWorkflow] = useState('');
  const [canStart, setCanStart] = useState(false);
  const [confirmRevoke, setConfirmRevoke] = useState<string>();
  const [showRevokedKeys, setShowRevokedKeys] = useState(false);
  const [showAllStarts, setShowAllStarts] = useState(false);

  async function action(run: () => Promise<void>) {
    setPending(true);
    setError('');
    setMessage('');
    try { await run(); }
    catch (cause) { setError(apiMessage(cause)); }
    finally { setPending(false); }
  }
  function open() {
    setData(undefined);
    setToken('');
    setConfirmRevoke(undefined);
    dialog.current?.showModal();
    void action(async () => setData(await apiFetch<ApiSettings>(base + '/api-settings')));
  }
  function changePolicy(patch: Partial<Policy>) {
    setData(current => current && { ...current, policy: { ...current.policy, ...patch } });
  }
  function save(event: FormEvent) {
    event.preventDefault();
    if (!data) return;
    void action(async () => {
      const policy = await apiFetch<Policy>(base + '/api-settings', {
        method: 'PUT',
        body: JSON.stringify(data.policy),
      });
      setData(current => current && { ...current, policy });
      setMessage('自动启动设置已保存');
    });
  }
  function create(event: FormEvent) {
    event.preventDefault();
    void action(async () => {
      let graph: unknown;
      if (mode === 'approved') {
        try { graph = JSON.parse(workflow); }
        catch { throw new Error('请输入有效的 ComfyUI API 格式工作流 JSON'); }
      }
      const { token: secret, ...key } = await apiFetch<ApiKey & { token: string }>(base + '/api-keys', {
        method: 'POST',
        body: JSON.stringify({ name, mode, auto_start: canStart, workflow: graph }),
      });
      setData(current => current && { ...current, keys: [key, ...current.keys] });
      setName('');
      setWorkflow('');
      setCanStart(false);
      setToken(secret);
    });
  }
  function changeKey(key: ApiKey, revoke = false) {
    void action(async () => {
      const updated = await apiFetch<ApiKey>(base + '/api-keys/' + encodeURIComponent(key.id), revoke ? {
        method: 'DELETE',
      } : { method: 'PATCH', body: JSON.stringify({ auto_start: !key.auto_start }) });
      setData(current => current && { ...current, keys: current.keys.map(item => item.id === key.id ? updated : item) });
      setConfirmRevoke(undefined);
      setToken('');
      setMessage(revoke ? '接入 Key 已撤销' : '开机权限已更新');
    });
  }

  async function copy(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setMessage('已复制');
    } catch {
      setError('无法访问剪贴板，请选中文本后复制。');
    }
  }

  const policy = data?.policy;
  const configuredAccounts = accounts.filter(account => account.enabled !== false && account.allow_allocation !== false);

  return <>
    <button className="button secondary" type="button" disabled={pending} onClick={open}>API 设置</button>
    <dialog className="nativeRestartDialog nativeApiDialog" ref={dialog} aria-labelledby="native-api-title"
      onClose={() => setToken('')} onCancel={event => { if (pending) event.preventDefault(); }}>
      <div className="actions nativeApiHeading">
        <h2 id="native-api-title">API 设置</h2>
        <button className="button ghost" type="button" disabled={pending} onClick={() => dialog.current?.close()}>关闭</button>
      </div>
      {error && <p className="status error" role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      {!data && pending && <p role="status">正在读取设置…</p>}
      {data && policy && <>
        <section aria-label="接入地址">
          <label className="field">公网 ComfyUI 地址<input readOnly value={data.url} placeholder="尚未配置原生 ComfyUI 域名" /></label>
          <label className="field">同机酒馆中继地址示例<input readOnly value={relay} /></label>
          <div className="actions">
            {data.url && <button type="button" className="button secondary small" onClick={() => void copy(data.url)}>复制公网地址</button>}
            <button type="button" className="button secondary small" onClick={() => void copy(relay)}>复制中继地址</button>
            <button type="button" className="button secondary small" disabled={pending} onClick={() => void action(async () => {
              const status = await apiFetch<{ configured: boolean; ready: boolean }>(base + '/status');
              setMessage(!status.configured ? '尚未配置原生 ComfyUI 域名' : status.ready ? 'Studio 已连接原生 ComfyUI' : '原生 ComfyUI 尚未就绪');
            })}>检查实例连接</button>
          </div>
        </section>
        <form onSubmit={save}>
          <h3>自动启动</h3>
          <label className="field">实例未启动时
            <select value={String(policy.auto_start)} onChange={event => changePolicy({ auto_start: event.target.value === 'true' })}>
              <option value="false">直接报错</option>
              <option value="true">收到授权的生成请求时自动启动</option>
            </select>
          </label>
          {policy.auto_start && <>
            <label className="field">默认 Google 账号
              <select value={policy.account_id || ''} required onChange={event => changePolicy({ account_id: event.target.value || null, gpu: null })}>
                <option value="">选择账号</option>
                {configuredAccounts.map(account => <option key={account.account_id} value={account.account_id}>{account.alias || account.account_id}</option>)}
              </select>
            </label>
            {policy.account_id && <WorkflowGpuSelection manifest={null} accountId={policy.account_id} value={policy.gpu}
              label="API 默认 GPU" onChange={gpu => changePolicy({ gpu })} />}
          </>}
          <NativeModelSettings value={policy} onChange={changePolicy} />
          <label className="field">空闲回收时间（分钟）
            <input type="number" min={1} max={2147483647} required value={policy.retain_idle_minutes || ''}
              onChange={event => changePolicy({ retain_idle_minutes: Number(event.target.value) })} />
          </label>
          <label className="field">最长运行时间（分钟）
            <input type="number" min={1} max={2147483647} required value={policy.max_runtime_minutes || ''}
              onChange={event => changePolicy({ max_runtime_minutes: Number(event.target.value) })} />
          </label>
          <p className="muted small">这些设置用于 API 新启动的会话。生成和同步完成后开始计算空闲时间。到期后停止接收新任务。</p>
          <button className="button" disabled={pending || (policy.auto_start && (!policy.account_id || !policy.gpu))}>保存启动设置</button>
        </form>
        <form onSubmit={create}>
          <h3>接入 Key</h3>
          <label className="field">客户端名称<input value={name} required maxLength={80} onChange={event => setName(event.target.value)} placeholder="例如，本机酒馆" /></label>
          <label className="field">工作流权限
            <select value={mode} onChange={event => setMode(event.target.value === 'trusted' ? 'trusted' : 'approved')}>
              <option value="approved">限定已批准的工作流</option>
              <option value="trusted">信任客户端提交的工作流</option>
            </select>
          </label>
          {mode === 'approved' && <label className="field">已批准的工作流模板
            <textarea required value={workflow} onChange={event => setWorkflow(event.target.value)} spellCheck={false} />
            <span className="muted small">使用 API 格式 JSON。仅在批准修改的输入位置放完整占位符，例如 %prompt%、%negative_prompt%、%nl%、%seed%、%width% 或 %height%。</span>
          </label>}
          {mode === 'trusted' && <p className="muted small">此权限适合你管理的酒馆或脚本。客户端可以提交任意原生工作流。</p>}
          <label className="actions"><input type="checkbox" checked={canStart} onChange={event => setCanStart(event.target.checked)} />允许此 Key 触发自动启动</label>
          <button className="button" disabled={pending || !name.trim()}>创建 Key</button>
        </form>
        {token && <div className="status">
          <label className="field">接入 Key（仅本次显示）<input readOnly value={token} onFocus={event => event.target.select()} /></label>
          <button type="button" className="button secondary small" onClick={() => void copy(token)}>复制 Key</button>
        </div>}
        {data.keys.length > 0 && (() => {
          const activeKeys = data.keys.filter(key => !key.revoked_at);
          const revokedKeys = data.keys.filter(key => Boolean(key.revoked_at));
          const visibleKeys = showRevokedKeys ? data.keys : activeKeys;
          return <section>
            <div className="actions nativeApiHeading">
              <h3>已创建的接入 Key（{activeKeys.length} 个有效）</h3>
              {revokedKeys.length > 0 && (
                <button type="button" className="button ghost small" onClick={() => setShowRevokedKeys(v => !v)}>
                  {showRevokedKeys ? '隐藏已撤销' : `显示已撤销 (${revokedKeys.length})`}
                </button>
              )}
            </div>
            {visibleKeys.length > 0 ? <ul className="nativeApiKeys">
              {visibleKeys.map(key => <li key={key.id}>
                <strong>{key.name}</strong> · {key.mode === 'trusted' ? '信任工作流' : '已批准工作流'}
                <p className="muted small">{key.revoked_at ? '已撤销' : key.auto_start ? '允许开机' : '仅使用已就绪实例'} · 已提交 {key.submitted} 次
                  {key.last_used_at && <> · 最近使用 {new Date(key.last_used_at).toLocaleString()}</>}</p>
                {!key.revoked_at && <div className="actions">
                  <button type="button" className="button secondary small" disabled={pending} onClick={() => changeKey(key)}>
                    {key.auto_start ? '移除开机权限' : '允许开机'}
                  </button>
                  {confirmRevoke === key.id ? <>
                    <button type="button" className="button danger small" disabled={pending} onClick={() => changeKey(key, true)}>确认撤销</button>
                    <button type="button" className="button ghost small" disabled={pending} onClick={() => setConfirmRevoke(undefined)}>取消</button>
                  </> : <button type="button" className="button ghost small" disabled={pending} onClick={() => setConfirmRevoke(key.id)}>撤销 Key</button>}
                </div>}
              </li>)}
            </ul> : <p className="muted small">当前没有有效的接入 Key。</p>}
          </section>;
        })()}
        {data.recent_starts.length > 0 && (() => {
          const limit = 3;
          const visibleStarts = showAllStarts ? data.recent_starts : data.recent_starts.slice(0, limit);
          return <section>
            <h3>最近的 API 启动记录（{data.recent_starts.length}）</h3>
            <ul>{visibleStarts.map(start => <li key={start.job_id}>
              {data.keys.find(key => key.id === start.key_id)?.name || '管理员'} · {new Date(start.created_at).toLocaleString()} · {start.state}
            </li>)}</ul>
            {data.recent_starts.length > limit && (
              <button type="button" className="button ghost small" onClick={() => setShowAllStarts(v => !v)}>
                {showAllStarts ? `收起，仅看最新 ${limit} 条` : `展开其余 ${data.recent_starts.length - limit} 条记录`}
              </button>
            )}
          </section>;
        })()}
      </>}
    </dialog>
  </>;
}
