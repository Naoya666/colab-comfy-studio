import { useCallback, useEffect, useState } from 'react';
import { Trash2 } from 'lucide-react';
import type { GpuAccount } from '../components/AccountDirectory';
import { RuntimeSelection, type RuntimeLifecycle } from '../components/RuntimeSelection';
import { WorkflowGpuSelection } from '../components/WorkflowGpuSelection';
import { ConfirmDialog } from '../components/Feedback';
import { Card } from '../components/ui';
import { apiFetch, clearCreationWarmups, type Project } from '../lib/gpu-api';
import { projectOptionLabel } from '../lib/project-label';
import { useSmartPoll } from '../lib/smart-poll';
import styles from '../gpu.module.css';

type AuthMode = 'strict' | 'url_path' | 'local_open';
type WorkflowScope = 'active' | 'selected';

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

type Config = {
  auto_start: boolean;
  account_id: string | null;
  project_id?: string | null;
  gpu: Record<string, unknown>;
  lifecycle: RuntimeLifecycle;
  retain_idle_minutes: number;
  max_runtime_minutes: number;
  auth_mode: AuthMode;
  allow_lan: boolean;
  listen_port: number | null;
  workflow_scope: WorkflowScope;
  workflow_ids: string[];
  allowed_origins: string[];
};

type Workflow = { id: string; workflow_key: string; workflow_version: string | number; active?: boolean };
type Key = { id: string; name: string; workflow_ids: string[]; revoked_at: string | null; token?: string };
type Warmup = {
  id: string;
  workflow_key: string;
  workflow_version: string | number;
  state: string;
  stage?: string;
  ready: boolean;
  expires_at?: number;
  error_message?: string;
  compute_warnings?: string[];
  gpu_reported?: string;
  creation_runtime?: { instance_id: string; process: string; devices?: { name: string }[] };
};
type PoolInstance = {
  id: string;
  instance_label?: string | null;
  account_id?: string | null;
  account_alias_snapshot?: string | null;
  state: string;
  retention_mode?: string | null;
  retain_idle_minutes?: number | null;
  expires_at?: number | null;
  gpu_reported?: string | null;
  runtime_spec_requested?: { key?: { accelerator?: string; shape?: string } } | null;
  runtime_spec_reported?: { key?: { accelerator?: string; shape?: string } } | null;
  creation_runtime?: {
    instance_id?: string;
    process?: string;
    workflow_key?: string;
    workflow_version?: string | number;
    cache_reason?: string;
  } | null;
  current_job_id?: string | null;
};
type Status = {
  url: string;
  path_code?: string | null;
  urls?: AccessUrls;
  listener?: ListenerStatus;
  config: Config;
  default_project_id?: string;
  keys: Key[];
  workflows: Workflow[];
  warmups: Warmup[];
  pool_instances?: PoolInstance[];
};
type Match = {
  status: 'matched' | 'unmatched' | 'ambiguous';
  matches: { workflow: Workflow; params: Record<string, unknown> }[];
  differences: { workflow: Workflow; reason: string }[];
};

const prefix = '/api/gpu/creation-api';
const WEAK_CHANNEL_ID = '__weak_default__';
const WARMUP_COLLAPSED_LIMIT = 3;
const write = (method: string, body?: unknown): RequestInit => ({ method, body: body === undefined ? undefined : JSON.stringify(body) });
const title = (workflow: Pick<Workflow, 'workflow_key' | 'workflow_version'>) => `${workflow.workflow_key} · ${workflow.workflow_version}`;
const finished = new Set(['completed', 'failed', 'cancelled']);
const stages: Record<string, string> = {
  restore_env: '准备依赖', restore_weights: '下载模型', restore_loras: '下载模型',
  comfy_start: '启动 ComfyUI', warm_reuse: '复用 ComfyUI', creation_warming: '加载模型',
};

function parseOriginsInput(text: string): string[] {
  return text.split(/[\n,]+/).map(item => item.trim()).filter(Boolean);
}

export function CreationApiPage({ accounts, projects = [], onProjectsRefresh }: {
  accounts: GpuAccount[];
  projects?: Project[];
  onProjectsRefresh?: () => void;
}) {
  const [status, setStatus] = useState<Status>();
  const [draft, setDraft] = useState<Config>();
  const [originsText, setOriginsText] = useState('');
  const [portInput, setPortInput] = useState('');
  const [keys, setKeys] = useState<Key[]>([]);
  const [keyId, setKeyId] = useState('');
  const [checkTarget, setCheckTarget] = useState('');
  const [keyName, setKeyName] = useState('');
  const [allowed, setAllowed] = useState<string[]>([]);
  const [token, setToken] = useState('');
  const [workflowId, setWorkflowId] = useState('');
  const [graphText, setGraphText] = useState('');
  const [match, setMatch] = useState<Match>();
  const [warmups, setWarmups] = useState<Warmup[]>([]);
  const [poolInstances, setPoolInstances] = useState<PoolInstance[]>([]);
  const [showAllWarmups, setShowAllWarmups] = useState(false);
  const [confirmRevokeKey, setConfirmRevokeKey] = useState(false);
  const [confirmResetCode, setConfirmResetCode] = useState(false);
  const [confirmClearWarmups, setConfirmClearWarmups] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const applyLoadedStatus = useCallback((data: Status) => {
    setStatus(data);
    setDraft(data.config);
    setOriginsText((data.config.allowed_origins || []).join('\n'));
    setPortInput(data.config.listen_port != null ? String(data.config.listen_port) : '');
    setKeys(data.keys);
    setWarmups(data.warmups);
    setPoolInstances(data.pool_instances || []);
    setWorkflowId(current => current || data.workflows[0]?.id || '');
  }, []);

  useEffect(() => {
    let live = true;
    apiFetch<Status>(prefix).then(data => {
      if (!live) return;
      applyLoadedStatus(data);
    }).catch(e => { if (live) setError(String(e.message || e)); });
    return () => { live = false; };
  }, [applyLoadedStatus]);

  const hasPollableWarmups = warmups.some(job => !finished.has(job.state) || job.ready) || poolInstances.length > 0;
  const pollWarmups = useCallback(async () => {
    const pending = warmups.filter(job => !finished.has(job.state) || job.ready);
    try {
      const [rows, refreshed] = await Promise.all([
        pending.length ? Promise.all(pending.map(job => apiFetch<Warmup>(`${prefix}/warmups/${job.id}`))) : Promise.resolve([]),
        apiFetch<Status>(prefix),
      ]);
      if (rows.length) {
        setWarmups(current => current.map(job => rows.find(row => row.id === job.id) || job));
      }
      setPoolInstances(refreshed.pool_instances || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [warmups]);

  useSmartPoll(pollWarmups, {
    enabled: hasPollableWarmups,
    active: warmups.some(job => !finished.has(job.state)),
    activeMs: 4000,
    idleMs: 15000,
    immediate: false,
  });

  async function run(task: () => Promise<void>) {
    setBusy(true);
    setError('');
    setNotice('');
    try { await task(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  function selectKey(id: string) {
    const key = keys.find(item => item.id === id);
    setKeyId(id);
    setCheckTarget(id);
    setKeyName(key?.name || '');
    setAllowed((key?.workflow_ids || []).filter(wid => status?.workflows.some(workflow => workflow.id === wid)));
    setMatch(undefined);
    setToken('');
  }

  async function copyUrl(value: string, label = '已复制接入地址') {
    await navigator.clipboard.writeText(value);
    setNotice(label);
  }

  const dirty = status && draft && JSON.stringify(status.config) !== JSON.stringify(draft);
  const accessDirty = status && draft && (
    status.config.auth_mode !== draft.auth_mode
    || status.config.listen_port !== draft.listen_port
    || status.config.allow_lan !== draft.allow_lan
  );
  const liveKeys = keys.filter(key => !key.revoked_at);
  const clearableWarmupsCount = warmups.filter(job => finished.has(job.state) && !job.ready).length;
  const visibleWarmups = showAllWarmups ? warmups : warmups.slice(0, WARMUP_COLLAPSED_LIMIT);
  const urls = status?.urls;
  const activeMode = status?.config.auth_mode || 'strict';

  async function saveAllConfig() {
    if (!draft) return;
    await apiFetch<Config>(`${prefix}/config`, write('PUT', draft));
    const refreshed = await apiFetch<Status>(prefix);
    applyLoadedStatus(refreshed);
    setNotice('接入与运行配置已保存');
  }

  async function start(id: string) {
    const job = await apiFetch<Warmup>(`${prefix}/warmups`, write('POST', { workflow_id: id }));
    setWarmups(rows => [job, ...rows.filter(row => row.id !== job.id)].slice(0, 20));
    onProjectsRefresh?.();
  }

  return <div className={styles.stack}>
    <h2>创作 API 设置</h2>
    {error && <p className="status error" role="alert">{error}</p>}
    {notice && <p className="status" role="status">{notice}</p>}
    {!status || !draft ? (!error && <p role="status">正在读取 API 设置…</p>) : <>
      <Card
        title="接入方式与地址"
        aside={status.listener?.active && status.listener.port ? (
          <span className="pill solid">独立端口 :{status.listener.port} 已监听</span>
        ) : undefined}
      >
        <div className="field">
          <label htmlFor="creation-auth-mode">API 鉴权模式（三档并存，切换弱鉴权后专用 Key 仍可独立使用）</label>
          <select
            id="creation-auth-mode"
            value={draft.auth_mode}
            onChange={e => setDraft({ ...draft, auth_mode: e.target.value as AuthMode })}
          >
            <option value="strict">强鉴权（仅专用 API Key，需请求头携带 Bearer）</option>
            <option value="url_path">弱鉴权（URL 路径短码，免中继直连推荐）</option>
            <option value="local_open">免鉴权（本机 / 受信内网免密直连）</option>
          </select>
        </div>

        {draft.auth_mode === 'url_path' && (
          <p className="muted small">
            将短密钥直接嵌入接入 URL 路径中。酒馆（如柏宝绘插件）等不支持填写 API Key 请求头的客户端，只需粘贴下方带短码的地址即可直连出图，无需额外部署 Nginx 中继。
          </p>
        )}
        {draft.auth_mode === 'local_open' && (
          <>
            <p className="muted small">
              无需任何 Key 或路径短码，适合端口未暴露公网的本机或内网环境。
            </p>
            {urls?.public_site_warning && (
              <p className="status" role="status">
                提示：检测到当前站点配置了公网域名。免鉴权模式仅放行本机{draft.allow_lan ? '及局域网/容器私网' : ''}源 IP；若需经公网域名访问，建议改用「URL 路径短码」模式。
              </p>
            )}
            <label>
              <input
                type="checkbox"
                checked={draft.allow_lan}
                onChange={e => setDraft({ ...draft, allow_lan: e.target.checked })}
              /> 允许局域网 / 容器私网免密访问（如 192.168.x.x、172.16-31.x.x、10.x.x.x）
            </label>
            <div className="field">
              <label htmlFor="creation-allowed-origins">允许浏览器跨域直连的前端 Origin（可选，每行一个，如 https://st.example.com）</label>
              <textarea
                id="creation-allowed-origins"
                rows={2}
                value={originsText}
                placeholder="本机 localhost / 127.0.0.1 已默认放行"
                onChange={e => {
                  setOriginsText(e.target.value);
                  setDraft({ ...draft, allowed_origins: parseOriginsInput(e.target.value) });
                }}
              />
            </div>
          </>
        )}

        {draft.auth_mode !== 'strict' && (
          <div className={styles.stack}>
            <div className="field">
              <label htmlFor="creation-wf-scope">弱鉴权 / 免鉴权开放的工作流范围</label>
              <select
                id="creation-wf-scope"
                value={draft.workflow_scope}
                onChange={e => setDraft({ ...draft, workflow_scope: e.target.value as WorkflowScope })}
              >
                <option value="active">自动开放所有当前激活版本（推荐，发布新版本无需重勾）</option>
                <option value="selected">仅开放手动勾选的发布版本</option>
              </select>
            </div>
            {draft.workflow_scope === 'selected' && (
              <fieldset>
                <legend>勾选向弱鉴权通道开放的发布版本</legend>
                {status.workflows.map(workflow => (
                  <div key={workflow.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={draft.workflow_ids.includes(workflow.id)}
                        onChange={e => setDraft({
                          ...draft,
                          workflow_ids: e.target.checked
                            ? [...draft.workflow_ids, workflow.id]
                            : draft.workflow_ids.filter(id => id !== workflow.id),
                        })}
                      /> {title(workflow)}{workflow.active ? '（当前激活）' : ''}
                    </label>
                  </div>
                ))}
                {!status.workflows.length && <p className="muted small">暂无已通过真机冒烟、可执行的发布版本。</p>}
              </fieldset>
            )}
          </div>
        )}

        <div className="field">
          <label htmlFor="creation-listen-port">独立监听端口（可选，1024–65535，留空则仅使用同端口子路径）</label>
          <input
            id="creation-listen-port"
            type="number"
            min={1024}
            max={65535}
            step={1}
            placeholder="例如 8189（留空关闭独立端口）"
            value={portInput}
            onChange={e => {
              const raw = e.target.value.trim();
              setPortInput(e.target.value);
              setDraft({ ...draft, listen_port: raw === '' ? null : Number(raw) });
            }}
          />
          <span className="muted small">
            默认直接用下方子路径即可免额外端口接入；遇到不允许填写路径前缀的插件时可启用独立端口（Docker 部署需将该端口映射至宿主机或同一容器网络）。
          </span>
        </div>
        {status.listener?.error && <p className="status error">{status.listener.error}</p>}

        {accessDirty && (
          <p className="status" role="status">鉴权模式或端口已修改，请点击下方「保存配置」使新接入地址生效。</p>
        )}

        {urls && !accessDirty && (
          <div className={styles.stack}>
            {activeMode === 'url_path' ? (
              <>
                {urls.dedicated_code_url && (
                  <div className="field">
                    <label htmlFor="creation-dedicated-code-url">独立域名短码直连地址（推荐）</label>
                    <input id="creation-dedicated-code-url" aria-label="创作 API 接入地址" readOnly value={urls.dedicated_code_url} />
                    <div className="actions">
                      <button type="button" className="button secondary small" onClick={() => void run(() => copyUrl(urls.dedicated_code_url))}>复制地址</button>
                    </div>
                  </div>
                )}
                {urls.subpath_code_url && (
                  <div className="field">
                    <label htmlFor="creation-subpath-code-url">同端口子路径短码直连地址（无需独立域名）</label>
                    <input id="creation-subpath-code-url" aria-label={urls.dedicated_code_url ? '同端口子路径短码地址' : '创作 API 接入地址'} readOnly value={urls.subpath_code_url} />
                    <div className="actions">
                      <button type="button" className="button secondary small" onClick={() => void run(() => copyUrl(urls.subpath_code_url))}>复制子路径地址</button>
                      <button type="button" className="button ghost small" disabled={busy} onClick={() => setConfirmResetCode(true)}>重置路径短码</button>
                    </div>
                  </div>
                )}
                {urls.port_code_url && (
                  <div className="field">
                    <label htmlFor="creation-port-code-url">独立端口短码地址</label>
                    <input id="creation-port-code-url" readOnly value={urls.port_code_url} />
                    <button type="button" className="button secondary small" onClick={() => void run(() => copyUrl(urls.port_code_url))}>复制端口地址</button>
                  </div>
                )}
              </>
            ) : (
              <>
                {urls.dedicated_url && (
                  <div className="field">
                    <label htmlFor="creation-dedicated-url">独立域名接入地址</label>
                    <input id="creation-dedicated-url" aria-label="创作 API 接入地址" readOnly value={urls.dedicated_url} />
                    <button type="button" className="button ghost small" onClick={() => void run(() => copyUrl(urls.dedicated_url))}>复制地址</button>
                  </div>
                )}
                <div className="field">
                  <label htmlFor="creation-subpath-url">同端口子路径地址{activeMode === 'local_open' ? '（本机免密直连）' : '（需携带 Bearer Key）'}</label>
                  <input id="creation-subpath-url" aria-label={urls.dedicated_url ? '同端口子路径地址' : '创作 API 接入地址'} readOnly value={urls.subpath_url} />
                  <button type="button" className="button ghost small" onClick={() => void run(() => copyUrl(urls.subpath_url))}>复制子路径地址</button>
                </div>
                {urls.port_url && (
                  <div className="field">
                    <label htmlFor="creation-port-url">独立监听端口地址{activeMode === 'local_open' ? '（免密直连）' : ''}</label>
                    <input id="creation-port-url" readOnly value={urls.port_url} />
                    <button type="button" className="button ghost small" onClick={() => void run(() => copyUrl(urls.port_url))}>复制端口地址</button>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        <button type="button" className="button" disabled={busy || !dirty} onClick={() => void run(saveAllConfig)}>
          保存配置
        </button>
      </Card>

      <Card
        title="创作池活跃实例（前端与 API 互通复用）"
        aside={
          <button
            type="button"
            className="button ghost small"
            disabled={busy}
            onClick={() => void run(async () => {
              const refreshed = await apiFetch<Status>(prefix);
              setPoolInstances(refreshed.pool_instances || []);
            })}
          >
            刷新实例
          </button>
        }
      >
        <p className="muted small">
          此处统一展示前端创作页与 API 预热启动的活跃创作池实例。创作 API 会优先复用已预热或规格足够的实例（原生 ComfyUI 实例隔离不参与复用）。
        </p>
        {!poolInstances.length ? (
          <p className="muted small">当前创作池没有活跃实例。可在前端创作页使用「智能/保留」启动，或点击下方「启动工作流」预热。</p>
        ) : (
          <div className={styles.stack}>
            {poolInstances.map((inst) => {
              const specAcc = inst.gpu_reported
                || inst.runtime_spec_reported?.key?.accelerator
                || inst.runtime_spec_requested?.key?.accelerator
                || 'GPU';
              const stateText = inst.state === 'retained_idle'
                ? '空闲可复用'
                : inst.state === 'busy'
                  ? '运行中（可排队复用）'
                  : inst.state === 'allocating' || inst.state === 'requested'
                    ? '分配中'
                    : inst.state;
              const wfText = inst.creation_runtime?.workflow_key
                ? `${inst.creation_runtime.workflow_key}${inst.creation_runtime.workflow_version != null ? ` v${inst.creation_runtime.workflow_version}` : ''}`
                : '尚未加载常驻工作流';
              return (
                <div key={inst.id} className="status">
                  <strong>
                    {inst.instance_label || `实例 #${inst.id.slice(0, 8)}`} · {specAcc} · {stateText}
                  </strong>
                  <p className="muted small">
                    账号：{inst.account_alias_snapshot || inst.account_id || '默认'} · 常驻工作流：{wfText}
                    {inst.creation_runtime?.process ? ` · 进程 ${inst.creation_runtime.process}` : ''}
                  </p>
                  <p className="muted small">
                    保留策略：{inst.retention_mode || 'save'}
                    {inst.retain_idle_minutes ? `（${inst.retain_idle_minutes} 分钟）` : ''}
                    {inst.expires_at ? ` · 可用至 ${new Date(inst.expires_at * 1000).toLocaleString()}` : ''}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      <Card title="默认运行配置">
        <div className="field">
          <label htmlFor="creation-target-project">任务与作品保存项目</label>
          <select
            id="creation-target-project"
            value={draft.project_id || ''}
            onChange={e => setDraft({ ...draft, project_id: e.target.value || null })}
          >
            <option value="">默认独立项目（创作 API）</option>
            {projects
              .filter(p => p.id !== status.default_project_id)
              .map(p => (
                <option key={p.id} value={p.id}>
                  {projectOptionLabel(p, projects)}{p.is_default ? '（默认项目）' : ''}
                </option>
              ))}
          </select>
          <span className="muted small">
            经创作 API 提交的正式出图与下方「启动工作流」预热任务都会归入所选项目；保持默认时自动写入独立的「创作 API」项目。
          </span>
        </div>
        <RuntimeSelection value={{ accountId: draft.account_id, instanceId: null, lifecycle: draft.lifecycle,
          retainIdleMinutes: draft.retain_idle_minutes }} accounts={accounts} instances={[]} provider="colab" showInstances={false}
          onChange={value => setDraft({ ...draft, account_id: value.accountId, lifecycle: value.lifecycle,
            retain_idle_minutes: value.retainIdleMinutes })} />
        <WorkflowGpuSelection manifest={null} accountId={draft.account_id} value={draft.gpu}
          onChange={gpu => { if (gpu) setDraft({ ...draft, gpu }); }} label="默认 GPU（仅新开机生效）" requireSelection />
        <p className="muted small">
          「默认 GPU」仅用于需要新开机器（手动预热或自动启动且创作池为空）时生效。复用已有创作池实例时按工作流默认规格与已预热实例优先匹配，且绝不会在已有低规格实例时自动多开双卡。
        </p>
        <div className="field">
          <label htmlFor="creation-timeout">单个任务最长运行时间（分钟）</label>
          <input id="creation-timeout" type="number" min={1} step={1}
            value={draft.max_runtime_minutes} onChange={e => setDraft({ ...draft, max_runtime_minutes: Number(e.target.value) })} />
        </div>
        <label><input type="checkbox" checked={draft.auto_start}
          onChange={e => setDraft({ ...draft, auto_start: e.target.checked })} /> 允许 API 按需自动启动</label>
        <p className="muted small">关闭时，API 需要可复用的实例。请先手动启动工作流。</p>
        {dirty && <p className="status" role="status">有未保存的修改。手动启动前请保存，API 当前使用上一次保存的配置。</p>}
        <button type="button" className="button" disabled={busy || !dirty} onClick={() => void run(saveAllConfig)}>
          保存运行配置
        </button>
      </Card>

      <Card title="专用 Key（强鉴权与多客户端隔离）">
        <div className="field">
          <label htmlFor="creation-key">选择 Key</label>
          <select id="creation-key" value={keyId} onChange={e => selectKey(e.target.value)}>
            <option value="">新建 Key</option>
            {liveKeys.map(key => <option value={key.id} key={key.id}>{key.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="creation-key-name">名称</label>
          <input id="creation-key-name" value={keyName} maxLength={80} onChange={e => setKeyName(e.target.value)} />
        </div>
        <fieldset>
          <legend>授权发布版本</legend>
          {status.workflows.map(workflow => <div key={workflow.id}><label>
            <input type="checkbox" checked={allowed.includes(workflow.id)} onChange={e => setAllowed(e.target.checked
              ? [...allowed, workflow.id] : allowed.filter(id => id !== workflow.id))} /> {title(workflow)}{workflow.active ? '（当前激活）' : ''}
          </label></div>)}
          {!status.workflows.length && <p>暂无已通过真机冒烟、可执行的发布版本。</p>}
        </fieldset>
        <div className="actions">
          <button type="button" className="button" disabled={busy || !keyName.trim()} onClick={() => void run(async () => {
            const row = await apiFetch<Key>(keyId ? `${prefix}/keys/${keyId}` : `${prefix}/keys`,
              write(keyId ? 'PUT' : 'POST', { name: keyName, workflow_ids: allowed }));
            const { token: plaintext, ...saved } = row;
            setKeys(current => [...current.filter(key => key.id !== saved.id), saved]);
            setKeyId(saved.id);
            setCheckTarget(saved.id);
            setKeyName(saved.name);
            setToken(plaintext || '');
            setMatch(undefined);
            setNotice('Key 权限已保存');
          })}>{keyId ? '保存授权' : '创建 Key'}</button>
          {keyId && <button type="button" className="button danger" disabled={busy} onClick={() => setConfirmRevokeKey(true)}>撤销 Key</button>}
        </div>
        {token && <div className="field">
          <label htmlFor="creation-key-token">Key 仅显示一次</label>
          <input id="creation-key-token" readOnly value={token} />
          <button className="button ghost" onClick={() => void run(async () => {
            await navigator.clipboard.writeText(token);
            setNotice('已复制 Key');
          })}>复制 Key</button>
        </div>}
      </Card>

      <Card title="检查工作流匹配">
        <div className="field">
          <label htmlFor="creation-check-target">校验目标通道</label>
          <select id="creation-check-target" value={checkTarget} onChange={e => { setCheckTarget(e.target.value); setMatch(undefined); }}>
            <option value="">全部可执行发布版本</option>
            <option value={WEAK_CHANNEL_ID}>弱鉴权 / 免鉴权默认通道</option>
            {liveKeys.map(key => <option value={key.id} key={key.id}>专用 Key：{key.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="creation-graph">酒馆使用的完整 API JSON</label>
          <textarea id="creation-graph" rows={6} value={graphText}
            onChange={e => { setGraphText(e.target.value); setMatch(undefined); }} />
          <input aria-label="上传 API JSON" type="file" accept=".json,application/json" onChange={e => {
            const file = e.target.files?.[0];
            if (!file) return;
            if (file.size > 1024 * 1024) { setError('工作流 JSON 超过 1 MiB'); return; }
            void run(async () => { setGraphText(await file.text()); setMatch(undefined); });
            e.target.value = '';
          }} />
        </div>
        <button type="button" className="button" disabled={busy || !graphText.trim()} onClick={() => void run(async () => {
          setMatch(await apiFetch<Match>(`${prefix}/check`, write('POST', { workflow: JSON.parse(graphText), key_id: checkTarget || undefined })));
        })}>检查匹配</button>
        {match && <div role="status">
          <p>{match.status === 'matched' ? '唯一匹配' : match.status === 'ambiguous' ? '匹配到多个版本，请调整开放版本范围' : '没有匹配的版本'}</p>
          {match.matches.map(({ workflow, params }) => <div key={workflow.id}>
            <strong>{title(workflow)}</strong>
            <pre>{JSON.stringify(params, null, 2)}</pre>
            {match.status === 'matched' && <button className="button ghost" disabled={busy || Boolean(dirty)}
              onClick={() => void run(() => start(workflow.id))}>启动此工作流</button>}
          </div>)}
          {match.differences.map(({ workflow, reason }) => <p key={workflow.id}>{title(workflow)} · {reason}</p>)}
        </div>}
      </Card>

      <Card
        title="启动工作流"
        aside={clearableWarmupsCount > 0 ? (
          <button type="button" className="button secondary small" disabled={busy} onClick={() => setConfirmClearWarmups(true)}>
            <Trash2 size={14} aria-hidden />清理历史 ({clearableWarmupsCount})
          </button>
        ) : undefined}
      >
        <div className="field">
          <label htmlFor="creation-warm-workflow">发布版本</label>
          <select id="creation-warm-workflow" value={workflowId} onChange={e => setWorkflowId(e.target.value)}>
            <option value="">选择版本</option>
            {status.workflows.map(workflow => <option key={workflow.id} value={workflow.id}>{title(workflow)}</option>)}
          </select>
        </div>
        <button type="button" className="button" disabled={busy || Boolean(dirty) || !workflowId}
          onClick={() => void run(() => start(workflowId))}>启动工作流</button>
        {visibleWarmups.map(job => <div key={job.id} className="status">
          <strong>{title(job)} · {job.ready ? '就绪' : job.state === 'completed' ? '预热完成，当前未报告可用实例'
            : job.state === 'failed' ? '启动失败' : job.state === 'cancelled' ? '已取消' : stages[job.stage || ''] || '准备实例'}</strong>
          {job.expires_at && <p>可用至 {new Date(job.expires_at * 1000).toLocaleString()}</p>}
          {job.error_message && <p>{job.error_message}</p>}
          {job.compute_warnings?.map(warning => <p key={warning}>{warning}</p>)}
          {job.creation_runtime && <p>{job.creation_runtime.devices?.map(device => device.name).join('、') || job.gpu_reported} · 实例 {job.creation_runtime.instance_id} · 进程 {job.creation_runtime.process}</p>}
        </div>)}
        {warmups.length > WARMUP_COLLAPSED_LIMIT && (
          <button type="button" className="button secondary small" onClick={() => setShowAllWarmups(v => !v)}>
            {showAllWarmups ? `收起，仅显示最新 ${WARMUP_COLLAPSED_LIMIT} 条` : `展开其余 ${warmups.length - WARMUP_COLLAPSED_LIMIT} 条预热记录`}
          </button>
        )}
      </Card>
    </>}

    <ConfirmDialog
      open={confirmResetCode}
      title="重置 URL 路径短码？"
      description="重置后旧的短码 URL 将立即失效，请将新生成的地址更新到酒馆或其他客户端。"
      confirmLabel="确认重置"
      danger
      busy={busy}
      onCancel={() => setConfirmResetCode(false)}
      onConfirm={() => {
        setConfirmResetCode(false);
        void run(async () => {
          const rotated = await apiFetch<{ path_code: string; urls: AccessUrls }>(`${prefix}/reset-code`, write('POST', {}));
          setStatus(current => current && { ...current, path_code: rotated.path_code, urls: rotated.urls });
          setNotice('路径短码已重置，请复制新地址');
        });
      }}
    />

    <ConfirmDialog
      open={confirmRevokeKey}
      title="撤销此 Key？"
      description="撤销后此 Key 将不能提交、查询或取图，且无法恢复。"
      confirmLabel="确认撤销"
      danger
      busy={busy}
      onCancel={() => setConfirmRevokeKey(false)}
      onConfirm={() => {
        setConfirmRevokeKey(false);
        void run(async () => {
          await apiFetch(`${prefix}/keys/${keyId}`, write('DELETE'));
          setKeys(current => current.filter(key => key.id !== keyId));
          selectKey('');
          setNotice('Key 已撤销');
        });
      }}
    />

    <ConfirmDialog
      open={confirmClearWarmups}
      title="清理历史预热记录？"
      description={`将隐藏 ${clearableWarmupsCount} 条已结束且不再就绪的预热记录，正在运行或当前就绪的预热实例不受影响。`}
      confirmLabel="确认清理"
      danger
      busy={busy}
      onCancel={() => setConfirmClearWarmups(false)}
      onConfirm={() => {
        setConfirmClearWarmups(false);
        void run(async () => {
          const { cleared } = await clearCreationWarmups();
          setWarmups(current => current.filter(job => !finished.has(job.state) || job.ready));
          setNotice(cleared ? `已清理 ${cleared} 条历史预热记录` : '没有可清理的历史预热记录');
        });
      }}
    />
  </div>;
}
