import { useEffect, useState } from 'react';
import type { GpuAccount } from '../components/AccountDirectory';
import { RuntimeSelection, type RuntimeLifecycle } from '../components/RuntimeSelection';
import { WorkflowGpuSelection } from '../components/WorkflowGpuSelection';
import { Card } from '../components/ui';
import { apiFetch } from '../lib/gpu-api';

type Config = {
  auto_start: boolean;
  account_id: string | null;
  gpu: Record<string, unknown>;
  lifecycle: RuntimeLifecycle;
  retain_idle_minutes: number;
  max_runtime_minutes: number;
};
type Workflow = { id: string; workflow_key: string; workflow_version: string | number };
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
type Status = { url: string; config: Config; keys: Key[]; workflows: Workflow[]; warmups: Warmup[] };
type Match = {
  status: 'matched' | 'unmatched' | 'ambiguous';
  matches: { workflow: Workflow; params: Record<string, unknown> }[];
  differences: { workflow: Workflow; reason: string }[];
};

const prefix = '/api/gpu/creation-api';
const write = (method: string, body?: unknown): RequestInit => ({ method, body: body === undefined ? undefined : JSON.stringify(body) });
const title = (workflow: Pick<Workflow, 'workflow_key' | 'workflow_version'>) => `${workflow.workflow_key} · ${workflow.workflow_version}`;
const finished = new Set(['completed', 'failed', 'cancelled']);
const stages: Record<string, string> = {
  restore_env: '准备依赖', restore_weights: '下载模型', restore_loras: '下载模型',
  comfy_start: '启动 ComfyUI', warm_reuse: '复用 ComfyUI', creation_warming: '加载模型',
};

export function CreationApiPage({ accounts }: { accounts: GpuAccount[] }) {
  const [status, setStatus] = useState<Status>();
  const [draft, setDraft] = useState<Config>();
  const [keys, setKeys] = useState<Key[]>([]);
  const [keyId, setKeyId] = useState('');
  const [keyName, setKeyName] = useState('');
  const [allowed, setAllowed] = useState<string[]>([]);
  const [token, setToken] = useState('');
  const [workflowId, setWorkflowId] = useState('');
  const [graphText, setGraphText] = useState('');
  const [match, setMatch] = useState<Match>();
  const [warmups, setWarmups] = useState<Warmup[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  useEffect(() => {
    let live = true;
    apiFetch<Status>(prefix).then(data => {
      if (!live) return;
      setStatus(data);
      setDraft(data.config);
      setKeys(data.keys);
      setWarmups(data.warmups);
      setWorkflowId(data.workflows[0]?.id || '');
    }).catch(e => { if (live) setError(String(e.message || e)); });
    return () => { live = false; };
  }, []);

  useEffect(() => {
    if (!warmups.length) return;
    let live = true;
    const timer = window.setInterval(() => {
      const pending = warmups.filter(job => !finished.has(job.state) || job.ready || job.state === 'completed');
      if (!pending.length) return;
      Promise.all(pending.map(job => apiFetch<Warmup>(`${prefix}/warmups/${job.id}`))).then(rows => {
        if (live) setWarmups(current => current.map(job => rows.find(row => row.id === job.id) || job));
      }).catch(e => { if (live) setError(String(e.message || e)); });
    }, 4000);
    return () => { live = false; window.clearInterval(timer); };
  }, [warmups]);

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
    setKeyName(key?.name || '');
    setAllowed((key?.workflow_ids || []).filter(id => status?.workflows.some(workflow => workflow.id === id)));
    setMatch(undefined);
    setToken('');
  }

  const dirty = status && draft && JSON.stringify(status.config) !== JSON.stringify(draft);
  const liveKeys = keys.filter(key => !key.revoked_at);

  async function start(id: string) {
    const job = await apiFetch<Warmup>(`${prefix}/warmups`, write('POST', { workflow_id: id }));
    setWarmups(rows => [job, ...rows.filter(row => row.id !== job.id)].slice(0, 20));
  }

  return <div className="stack">
    <h2>创作 API 设置</h2>
    {error && <p className="status error" role="alert">{error}</p>}
    {notice && <p className="status" role="status">{notice}</p>}
    {!status || !draft ? (!error && <p role="status">正在读取 API 设置…</p>) : <>
      <Card title="接入地址">
        {status.url ? <div className="field">
          <input aria-label="创作 API 接入地址" readOnly value={status.url} />
          <button className="button ghost" onClick={() => void run(async () => {
            await navigator.clipboard.writeText(status.url);
            setNotice('已复制接入地址');
          })}>复制地址</button>
        </div> : <p>服务端尚未配置创作 API 域名。</p>}
      </Card>
      <Card title="默认运行配置">
        <RuntimeSelection value={{ accountId: draft.account_id, instanceId: null, lifecycle: draft.lifecycle,
          retainIdleMinutes: draft.retain_idle_minutes }} accounts={accounts} instances={[]} provider="colab" showInstances={false}
          onChange={value => setDraft({ ...draft, account_id: value.accountId, lifecycle: value.lifecycle,
            retain_idle_minutes: value.retainIdleMinutes })} />
        <WorkflowGpuSelection manifest={null} accountId={draft.account_id} value={draft.gpu}
          onChange={gpu => { if (gpu) setDraft({ ...draft, gpu }); }} label="默认 GPU" requireSelection />
        <div className="field">
          <label htmlFor="creation-timeout">单个任务最长运行时间（分钟）</label>
          <input id="creation-timeout" type="number" min={1} step={1}
            value={draft.max_runtime_minutes} onChange={e => setDraft({ ...draft, max_runtime_minutes: Number(e.target.value) })} />
        </div>
        <label><input type="checkbox" checked={draft.auto_start}
          onChange={e => setDraft({ ...draft, auto_start: e.target.checked })} /> 允许 API 按需自动启动</label>
        <p className="muted small">关闭时，API 需要可复用的实例。请先手动启动工作流。</p>
        {dirty && <p className="status" role="status">有未保存的修改。手动启动前请保存，API 当前使用上一次保存的配置。</p>}
        <button type="button" className="button" disabled={busy || !dirty} onClick={() => void run(async () => {
          const saved = await apiFetch<Config>(`${prefix}/config`, write('PUT', draft));
          setStatus({ ...status, config: saved });
          setDraft(saved);
          setNotice('运行配置已保存');
        })}>保存运行配置</button>
      </Card>
      <Card title="专用 Key">
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
              ? [...allowed, workflow.id] : allowed.filter(id => id !== workflow.id))} /> {title(workflow)}
          </label></div>)}
          {!status.workflows.length && <p>暂无已通过真机冒烟、可执行的发布版本。</p>}
        </fieldset>
        <button type="button" className="button" disabled={busy || !keyName.trim()} onClick={() => void run(async () => {
          const row = await apiFetch<Key>(keyId ? `${prefix}/keys/${keyId}` : `${prefix}/keys`,
            write(keyId ? 'PUT' : 'POST', { name: keyName, workflow_ids: allowed }));
          const { token: plaintext, ...saved } = row;
          setKeys(current => [...current.filter(key => key.id !== saved.id), saved]);
          setKeyId(saved.id);
          setKeyName(saved.name);
          setToken(plaintext || '');
          setMatch(undefined);
          setNotice('Key 权限已保存');
        })}>{keyId ? '保存授权' : '创建 Key'}</button>
        {keyId && <button type="button" className="button ghost" disabled={busy} onClick={() => {
          if (!window.confirm('撤销后此 Key 将不能提交、查询或取图。撤销此 Key？')) return;
          void run(async () => {
            await apiFetch(`${prefix}/keys/${keyId}`, write('DELETE'));
            setKeys(current => current.filter(key => key.id !== keyId));
            selectKey('');
            setNotice('Key 已撤销');
          });
        }}>撤销 Key</button>}
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
          setMatch(await apiFetch<Match>(`${prefix}/check`, write('POST', { workflow: JSON.parse(graphText), key_id: keyId || undefined })));
        })}>检查匹配</button>
        {match && <div role="status">
          <p>{match.status === 'matched' ? '唯一匹配' : match.status === 'ambiguous' ? '匹配到多个版本，请调整 Key 授权' : '没有匹配的版本'}</p>
          {match.matches.map(({ workflow, params }) => <div key={workflow.id}>
            <strong>{title(workflow)}</strong>
            <pre>{JSON.stringify(params, null, 2)}</pre>
            {match.status === 'matched' && <button className="button ghost" disabled={busy || Boolean(dirty)}
              onClick={() => void run(() => start(workflow.id))}>启动此工作流</button>}
          </div>)}
          {match.differences.map(({ workflow, reason }) => <p key={workflow.id}>{title(workflow)} · {reason}</p>)}
        </div>}
      </Card>
      <Card title="启动工作流">
        <div className="field">
          <label htmlFor="creation-warm-workflow">发布版本</label>
          <select id="creation-warm-workflow" value={workflowId} onChange={e => setWorkflowId(e.target.value)}>
            <option value="">选择版本</option>
            {status.workflows.map(workflow => <option key={workflow.id} value={workflow.id}>{title(workflow)}</option>)}
          </select>
        </div>
        <button type="button" className="button" disabled={busy || Boolean(dirty) || !workflowId}
          onClick={() => void run(() => start(workflowId))}>启动工作流</button>
        {warmups.map(job => <div key={job.id} className="status">
          <strong>{title(job)} · {job.ready ? '就绪' : job.state === 'completed' ? '预热完成，当前未报告可用实例'
            : job.state === 'failed' ? '启动失败' : job.state === 'cancelled' ? '已取消' : stages[job.stage || ''] || '准备实例'}</strong>
          {job.expires_at && <p>可用至 {new Date(job.expires_at * 1000).toLocaleString()}</p>}
          {job.error_message && <p>{job.error_message}</p>}
          {job.compute_warnings?.map(warning => <p key={warning}>{warning}</p>)}
          {job.creation_runtime && <p>{job.creation_runtime.devices?.map(device => device.name).join('、') || job.gpu_reported} · 实例 {job.creation_runtime.instance_id} · 进程 {job.creation_runtime.process}</p>}
        </div>)}
      </Card>
    </>}
  </div>;
}
