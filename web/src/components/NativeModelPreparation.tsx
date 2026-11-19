import { useEffect, useState } from 'react';
import { apiFetch, apiMessage } from '../lib/gpu-api';

type PreparedModel = {
  category: string;
  name: string;
  state: 'pending' | 'downloading' | 'ready' | 'failed';
  source?: 'external' | 'drive' | 'local';
  host?: string;
  error?: string;
  fallback?: string;
  bytes?: number;
  total?: number;
  seconds?: number;
  attempt?: number;
  integrity?: string;
};

type Preparation = { state: 'preparing' | 'ready' | 'failed'; models: PreparedModel[]; message?: string };
const path = '/api/gpu/native/model-preparation';
const phases = { pending: '等待', downloading: '下载中', ready: '已就绪', failed: '失败' };
const sources = { external: '外部来源', drive: 'Drive', local: '本地命中' };
const mb = (bytes: number) => `${(bytes / 1024 ** 2).toFixed(1)} MiB`;

export function NativeModelPreparation() {
  const [workflow, setWorkflow] = useState('');
  const [declarations, setDeclarations] = useState('');
  const [id, setId] = useState('');
  const [result, setResult] = useState<Preparation>();
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);

  useEffect(() => {
    if (!id) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const status = await apiFetch<Preparation>(`${path}/${id}`);
        if (!active) return;
        setResult(status);
        setError('');
        if (status.state === 'preparing') timer = setTimeout(() => void poll(), 1200);
      } catch (cause) {
        if (active) setError(apiMessage(cause));
      }
    }
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [id]);

  async function prepare() {
    setSending(true); setError(''); setId(''); setResult(undefined);
    try {
      const body = { workflow: JSON.parse(workflow), dependencies: declarations.trim() ? JSON.parse(declarations) : [] };
      const job = await apiFetch<{ id: string }>(path, { method: 'POST', body: JSON.stringify(body) }, 25_000);
      setResult({ state: 'preparing', models: [] });
      setId(job.id);
    } catch (cause) {
      setError(cause instanceof SyntaxError ? '请检查工作流或显式依赖的 JSON 格式。' : apiMessage(cause));
    } finally { setSending(false); }
  }

  async function latest() {
    setError('');
    try {
      const status = await apiFetch<Preparation & { id: string }>(path + '/latest');
      setResult(status);
      setId(status.id);
    } catch (cause) { setError(apiMessage(cause)); }
  }

  return <details className="foldable">
    <summary>检查并下载此工作流所需模型</summary>
    <div className="foldContent">
      <p>首次用酒馆等客户端调用前，可以先在这里把模型下载到当前实例。冷下载仍受客户端自身的请求超时限制。</p>
      <label className="field">ComfyUI 工作流 JSON
        <input type="file" accept=".json,application/json" onChange={async e => {
          const file = e.target.files?.[0];
          if (file) {
            try { setWorkflow(await file.text()); }
            catch (cause) { setError(apiMessage(cause)); }
          }
        }} />
        <textarea rows={5} value={workflow} onChange={e => setWorkflow(e.target.value)}
          placeholder="粘贴 API 格式或原生工作流 JSON" />
      </label>
      <label className="field">显式模型依赖（可选）
        <textarea rows={3} value={declarations} onChange={e => setDeclarations(e.target.value)}
          placeholder={'[{"category":"loras","name":"styles/example.safetensors"}]'} />
        <span className="muted small">自定义节点或动态模型名可填类别及相对路径。详细来源可在模型库绑定，也可在这里填写 url、sha256 和 size（字节）。</span>
      </label>
      <div className="actions">
        <button className="button secondary" disabled={sending || !workflow.trim() || result?.state === 'preparing'}
          onClick={() => void prepare()}>检查并下载</button>
        <button className="button secondary" disabled={sending} onClick={() => void latest()}>查看最近一次准备</button>
        {result?.state === 'preparing' && <button className="button secondary" onClick={() => {
          setId(''); setResult(undefined); setError('');
        }}>停止查看</button>}
      </div>
      {error && <p className="status error" role="alert">{error}</p>}
      {result && <p role="status">{result.message || ({
        preparing: '正在检查或下载模型…',
        ready: result.models.length ? '已识别的模型均已就绪。' : '此工作流未识别出模型文件，可补充显式依赖。',
        failed: '部分模型准备失败，请核对模型库和下方原因。',
      })[result.state]}</p>}
      {result?.models.map(model => <div key={`${model.category}/${model.name}`}>
        <strong>{model.category}/{model.name}</strong>
        <p>{phases[model.state]}{model.source && ` · ${sources[model.source]}`}
          {model.host && ` · ${model.host}`}{model.seconds !== undefined && ` · ${model.seconds} 秒`}
          {model.attempt && model.attempt > 1 ? ` · 第 ${model.attempt} 次尝试` : ''}
          {model.integrity === 'transport' && ' · 下载长度已校验，来源 SHA-256 未提供'}</p>
        {model.state === 'downloading' && model.total && <>
          <progress aria-label={`${model.name} 下载进度`} value={model.bytes || 0} max={model.total} />
          <p>{mb(model.bytes || 0)} / {mb(model.total)}</p>
        </>}
        {model.fallback && <p>外部来源失败，已尝试 Drive · {model.fallback}</p>}
        {model.error && <p className="status error">{model.error}</p>}
      </div>)}
    </div>
  </details>;
}
