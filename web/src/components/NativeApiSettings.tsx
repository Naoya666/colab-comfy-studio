import { useRef, useState, type FormEvent } from 'react';
import { apiFetch, apiMessage } from '../lib/gpu-api';
import { countDistinctDriveAccounts } from '../lib/allocation-retry-copy';
import type { GpuAccount } from './AccountDirectory';
import { NativeModelSettings, type ModelStorage } from './NativeModelSettings';
import { WorkflowGpuSelection } from './WorkflowGpuSelection';

type AuthMode = 'strict' | 'url_path' | 'local_open';

type AccessUrls = {
  recommended: string;
  subpath_url: string;
  subpath_code_url: string;
  dedicated_url: string;
  dedicated_code_url: string;
  port_url: string;
  port_code_url: string;
  public_site_warning: boolean;
  allow_lan: boolean;
};

type ListenerStatus = {
  active: boolean;
  host: string | null;
  port: number | null;
  error: string | null;
};

type Policy = ModelStorage & {
  auto_start: boolean;
  account_id: string | null;
  gpu: Record<string, unknown> | null;
  retain_idle_minutes: number;
  max_runtime_minutes: number;
  auth_mode: AuthMode;
  allow_lan: boolean;
  listen_port: number | null;
  weak_workflow_mode: 'trusted' | 'approved';
  weak_workflow: Record<string, unknown> | null;
  allowed_origins: string[];
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
  path_code?: string | null;
  urls?: AccessUrls;
  listener?: ListenerStatus;
  policy: Policy;
  keys: ApiKey[];
  recent_starts: { job_id: string; key_id: string; state: string; created_at: string }[];
};

const base = '/api/gpu/native';

function parseOriginsInput(text: string): string[] {
  return text.split(/[\n,]+/).map(item => item.trim()).filter(Boolean);
}

export function NativeApiSettings({ accounts }: { accounts: GpuAccount[] }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [data, setData] = useState<ApiSettings>();
  const [savedAuthMode, setSavedAuthMode] = useState<AuthMode>('strict');
  const [savedPort, setSavedPort] = useState<number | null>(null);
  const [weakTemplateText, setWeakTemplateText] = useState('');
  const [originsText, setOriginsText] = useState('');
  const [portInput, setPortInput] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [token, setToken] = useState('');
  const [name, setName] = useState('');
  const [mode, setMode] = useState<ApiKey['mode']>('approved');
  const [workflow, setWorkflow] = useState('');
  const [canStart, setCanStart] = useState(false);
  const [confirmRevoke, setConfirmRevoke] = useState<string>();
  const [confirmResetCode, setConfirmResetCode] = useState(false);
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

  function applySettings(loaded: ApiSettings) {
    setData(loaded);
    setSavedAuthMode(loaded.policy.auth_mode || 'strict');
    setSavedPort(loaded.policy.listen_port ?? null);
    setWeakTemplateText(loaded.policy.weak_workflow ? JSON.stringify(loaded.policy.weak_workflow, null, 2) : '');
    setOriginsText((loaded.policy.allowed_origins || []).join('\n'));
    setPortInput(loaded.policy.listen_port != null ? String(loaded.policy.listen_port) : '');
  }

  function open() {
    setData(undefined);
    setToken('');
    setConfirmRevoke(undefined);
    setConfirmResetCode(false);
    dialog.current?.showModal();
    void action(async () => applySettings(await apiFetch<ApiSettings>(base + '/api-settings')));
  }

  function changePolicy(patch: Partial<Policy>) {
    setData(current => current && { ...current, policy: { ...current.policy, ...patch } });
  }

  function save(event: FormEvent) {
    event.preventDefault();
    if (!data) return;
    void action(async () => {
      let weakGraph: Record<string, unknown> | null = null;
      if (data.policy.weak_workflow_mode === 'approved') {
        try {
          const parsed = JSON.parse(weakTemplateText);
          if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error();
          weakGraph = parsed as Record<string, unknown>;
        } catch {
          throw new Error('弱鉴权限定模式下请输入有效的 ComfyUI API 格式工作流 JSON');
        }
      }
      const payload: Policy = { ...data.policy, weak_workflow: weakGraph };
      await apiFetch<Policy>(base + '/api-settings', {
        method: 'PUT',
        body: JSON.stringify(payload),
      });
      const refreshed = await apiFetch<ApiSettings>(base + '/api-settings');
      applySettings(refreshed);
      setMessage('API 接入与启动设置已保存');
    });
  }

  function resetCode() {
    void action(async () => {
      const rotated = await apiFetch<{ path_code: string; urls: AccessUrls }>(base + '/api-settings/reset-code', {
        method: 'POST',
        body: '{}',
      });
      setData(current => current && { ...current, path_code: rotated.path_code, urls: rotated.urls });
      setConfirmResetCode(false);
      setMessage('路径短码已重置，请复制更新后的接入地址');
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
  const urls = data?.urls;
  const isStrictMode = (policy?.auth_mode || 'strict') === 'strict';
  const activeKeys = (data?.keys || []).filter(key => !key.revoked_at);
  const revokedKeys = (data?.keys || []).filter(key => Boolean(key.revoked_at));
  const accessUnsaved = Boolean(policy && (policy.auth_mode !== savedAuthMode || (policy.listen_port ?? null) !== savedPort));
  const configuredAccounts = accounts.filter(account => account.enabled !== false && account.allow_allocation !== false);

  return <>
    <button className="button secondary" type="button" disabled={pending} onClick={open}>API 设置</button>
    <dialog className="nativeRestartDialog nativeApiDialog" ref={dialog} aria-labelledby="native-api-title"
      onClose={() => setToken('')} onCancel={event => { if (pending) event.preventDefault(); }}>
      <div className="actions nativeApiHeading">
        <h2 id="native-api-title">原生 ComfyUI API 设置</h2>
        <button className="button ghost" type="button" disabled={pending} onClick={() => dialog.current?.close()}>关闭</button>
      </div>
      {error && <p className="status error" role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      {!data && pending && <p role="status">正在读取设置…</p>}
      {data && policy && <>
        <form onSubmit={save}>
          <section aria-label="接入地址与鉴权模式">
            <div className="actions nativeApiHeading">
              <h3>接入方式与地址</h3>
              {data.listener?.active && data.listener.port && (
                <span className="pill solid">独立端口 :{data.listener.port} 已监听</span>
              )}
            </div>

            <label className="field">API 鉴权模式
              <select value={policy.auth_mode} onChange={event => changePolicy({ auth_mode: event.target.value as AuthMode })}>
                <option value="strict">强鉴权（仅专用 API Key，需请求头携带 Bearer）</option>
                <option value="url_path">弱鉴权（URL 路径短码，免中继直连推荐）</option>
                <option value="local_open">免鉴权（本机 / 受信内网免密直连）</option>
              </select>
            </label>

            {policy.auth_mode !== 'strict' && (
              <p className="muted small">
                弱鉴权 / 免鉴权模式已开放完整生图能力（含 /ws 实时进度、/upload/image 上传图片、/queue 与 /interrupt），并自动封禁 ComfyUI-Manager 节点管理接口。
              </p>
            )}

            {policy.auth_mode === 'local_open' && <>
              {urls?.public_site_warning && (
                <p className="status" role="status">
                  提示：当前站点配置了公网域名。免鉴权仅放行本机{policy.allow_lan ? '及局域网/容器私网' : ''}源 IP；若通过公网域名直连，请改用「URL 路径短码」模式。
                </p>
              )}
              <label className="actions">
                <input
                  type="checkbox"
                  checked={policy.allow_lan}
                  onChange={event => changePolicy({ allow_lan: event.target.checked })}
                /> 允许局域网 / 容器私网免密访问
              </label>
              <label className="field">允许浏览器跨域直连的前端 Origin（可选，每行一个）
                <textarea
                  rows={2}
                  value={originsText}
                  placeholder="本机 localhost / 127.0.0.1 已默认放行"
                  onChange={event => {
                    setOriginsText(event.target.value);
                    changePolicy({ allowed_origins: parseOriginsInput(event.target.value) });
                  }}
                />
              </label>
            </>}

            {policy.auth_mode !== 'strict' && <>
              <label className="field">弱鉴权工作流限制
                <select
                  value={policy.weak_workflow_mode}
                  onChange={event => changePolicy({ weak_workflow_mode: event.target.value === 'approved' ? 'approved' : 'trusted' })}
                >
                  <option value="trusted">信任客户端提交的任意工作流（默认）</option>
                  <option value="approved">限定已批准的工作流模板</option>
                </select>
              </label>
              {policy.weak_workflow_mode === 'approved' && (
                <label className="field">弱鉴权批准的工作流模板（API JSON）
                  <textarea
                    required
                    value={weakTemplateText}
                    onChange={event => setWeakTemplateText(event.target.value)}
                    spellCheck={false}
                  />
                  <span className="muted small">支持占位符 %prompt%、%negative_prompt%、%nl%、%seed%、%width%、%height%。</span>
                </label>
              )}
            </>}

            <label className="field">独立监听端口（可选，1024–65535，留空关闭）
              <input
                type="number"
                min={1024}
                max={65535}
                step={1}
                placeholder="例如 8188（供不支持填子路径的插件直连）"
                value={portInput}
                onChange={event => {
                  const raw = event.target.value.trim();
                  setPortInput(event.target.value);
                  changePolicy({ listen_port: raw === '' ? null : Number(raw) });
                }}
              />
            </label>
            {data.listener?.error && <p className="status error">{data.listener.error}</p>}

            {accessUnsaved && (
              <p className="status" role="status">鉴权模式或独立端口尚未保存，点击下方「保存接入与启动设置」后生成对应接入地址。</p>
            )}

            {urls && !accessUnsaved && <>
              {savedAuthMode === 'url_path' ? <>
                {urls.dedicated_code_url && (
                  <label className="field">独立域名短码直连地址（推荐）
                    <input readOnly value={urls.dedicated_code_url} />
                  </label>
                )}
                {urls.subpath_code_url && (
                  <label className="field">同端口子路径短码直连地址（无需独立域名）
                    <input readOnly value={urls.subpath_code_url} />
                  </label>
                )}
                {urls.port_code_url && (
                  <label className="field">独立端口短码地址
                    <input readOnly value={urls.port_code_url} />
                  </label>
                )}
                <div className="actions">
                  {urls.dedicated_code_url && (
                    <button type="button" className="button secondary small" onClick={() => void copy(urls.dedicated_code_url)}>复制域名短码地址</button>
                  )}
                  {urls.subpath_code_url && (
                    <button type="button" className="button secondary small" onClick={() => void copy(urls.subpath_code_url)}>复制子路径短码地址</button>
                  )}
                  {urls.port_code_url && (
                    <button type="button" className="button secondary small" onClick={() => void copy(urls.port_code_url)}>复制端口地址</button>
                  )}
                  {confirmResetCode ? <>
                    <button type="button" className="button danger small" disabled={pending} onClick={resetCode}>确认重置短码</button>
                    <button type="button" className="button ghost small" disabled={pending} onClick={() => setConfirmResetCode(false)}>取消</button>
                  </> : (
                    <button type="button" className="button ghost small" disabled={pending} onClick={() => setConfirmResetCode(true)}>重置路径短码</button>
                  )}
                </div>
              </> : <>
                {data.url && (
                  <label className="field">公网 / 独立域名地址
                    <input readOnly value={data.url} />
                  </label>
                )}
                <label className="field">同端口子路径地址{savedAuthMode === 'local_open' ? '（本机免密直连）' : '（需携带 Bearer Key）'}
                  <input readOnly value={urls.subpath_url} />
                </label>
                {urls.port_url && (
                  <label className="field">独立监听端口地址{savedAuthMode === 'local_open' ? '（免密直连）' : ''}
                    <input readOnly value={urls.port_url} />
                  </label>
                )}
                <div className="actions">
                  {data.url && <button type="button" className="button secondary small" onClick={() => void copy(data.url)}>复制独立域名地址</button>}
                  <button type="button" className="button secondary small" onClick={() => void copy(urls.subpath_url)}>复制子路径地址</button>
                  {urls.port_url && <button type="button" className="button secondary small" onClick={() => void copy(urls.port_url)}>复制独立端口地址</button>}
                </div>
              </>}
            </>}

            <div className="actions">
              <button type="button" className="button secondary small" disabled={pending} onClick={() => void action(async () => {
                const status = await apiFetch<{ configured: boolean; ready: boolean }>(base + '/status');
                setMessage(!status.configured ? '尚未配置原生 ComfyUI 隧道' : status.ready ? 'Studio 已连接原生 ComfyUI' : '原生 ComfyUI 尚未就绪');
              })}>检查实例连接</button>
            </div>
          </section>

          <h3>自动启动</h3>
          <label className="field">实例未启动时
            <select value={String(policy.auto_start)} onChange={event => changePolicy({ auto_start: event.target.value === 'true' })}>
              <option value="false">直接报错</option>
              <option value="true">收到授权的生成请求时自动启动</option>
            </select>
          </label>
          {policy.auto_start && <>
            <label className="field">默认 Google 账号
              <select value={policy.account_id || ''} onChange={event => changePolicy({ account_id: event.target.value || null })}>
                <option value="">自动选择账号（多账号轮换）</option>
                {configuredAccounts.map(account => <option key={account.account_id} value={account.account_id}>{account.alias || account.account_id}</option>)}
              </select>
            </label>
            {!policy.account_id && countDistinctDriveAccounts(configuredAccounts) > 1 && (
              <p className="status" role="status">
                提示：自动选择账号会在所有可用账号间轮换；建议在「设置 → Google 账号」将小号存储盘绑定到主号，以确保 API 自动开机挂载一致的工作区与插件。
              </p>
            )}
            <WorkflowGpuSelection manifest={null} accountId={policy.account_id} value={policy.gpu}
              label="API 默认 GPU" onChange={gpu => changePolicy({ gpu })} />
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
          <button className="button" disabled={pending || (policy.auto_start && (!policy.gpu || (!policy.account_id && configuredAccounts.length === 0)))}>保存接入与启动设置</button>
        </form>

        {isStrictMode ? (
          <>
            <form onSubmit={create}>
              <h3>专用接入 Key（强鉴权与多客户端隔离）</h3>
              <label className="field">客户端名称<input value={name} required maxLength={80} onChange={event => setName(event.target.value)} placeholder="例如，外部脚本" /></label>
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
              {mode === 'trusted' && <p className="muted small">此权限适合你管理的客户端或脚本。客户端可以提交任意原生工作流。</p>}
              <label className="actions"><input type="checkbox" checked={canStart} onChange={event => setCanStart(event.target.checked)} />允许此 Key 触发自动启动</label>
              <button className="button" disabled={pending || !name.trim()}>创建 Key</button>
            </form>
            {token && <div className="status">
              <label className="field">接入 Key（仅本次显示）<input readOnly value={token} onFocus={event => event.target.select()} /></label>
              <button type="button" className="button secondary small" onClick={() => void copy(token)}>复制 Key</button>
            </div>}
            {data.keys.length > 0 && (() => {
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
          </>
        ) : activeKeys.length > 0 ? (
          <details className="foldable">
            <summary>已有 {activeKeys.length} 个历史专用 Key 仍在后台生效（展开查看或撤销）</summary>
            <div className="foldContent">
              <p className="muted small">
                当前处于{policy.auth_mode === 'local_open' ? '免鉴权' : '弱鉴权'}模式，常规客户端无需配置 Key；如需新建专用 Key，请切换回「强鉴权」模式。
              </p>
              <ul className="nativeApiKeys">
                {activeKeys.map(key => <li key={key.id}>
                  <strong>{key.name}</strong> · {key.mode === 'trusted' ? '信任工作流' : '已批准工作流'}
                  <p className="muted small">{key.auto_start ? '允许开机' : '仅使用已就绪实例'} · 已提交 {key.submitted} 次
                    {key.last_used_at && <> · 最近使用 {new Date(key.last_used_at).toLocaleString()}</>}</p>
                  <div className="actions">
                    <button type="button" className="button secondary small" disabled={pending} onClick={() => changeKey(key)}>
                      {key.auto_start ? '移除开机权限' : '允许开机'}
                    </button>
                    {confirmRevoke === key.id ? <>
                      <button type="button" className="button danger small" disabled={pending} onClick={() => changeKey(key, true)}>确认撤销</button>
                      <button type="button" className="button ghost small" disabled={pending} onClick={() => setConfirmRevoke(undefined)}>取消</button>
                    </> : <button type="button" className="button ghost small" disabled={pending} onClick={() => setConfirmRevoke(key.id)}>撤销 Key</button>}
                  </div>
                </li>)}
              </ul>
            </div>
          </details>
        ) : null}
        {data.recent_starts.length > 0 && (() => {
          const limit = 3;
          const visibleStarts = showAllStarts ? data.recent_starts : data.recent_starts.slice(0, limit);
          return <section>
            <h3>最近的 API 启动记录（{data.recent_starts.length}）</h3>
            <ul>{visibleStarts.map(start => <li key={start.job_id}>
              {data.keys.find(key => key.id === start.key_id)?.name || (start.key_id === 'weak-api' ? '弱鉴权通道' : '管理员')} · {new Date(start.created_at).toLocaleString()} · {start.state}
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
