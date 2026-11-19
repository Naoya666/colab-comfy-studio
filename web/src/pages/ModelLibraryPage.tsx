import { useCallback, useEffect, useId, useState } from 'react';
import { apiFetch, apiMessage } from '../lib/gpu-api';
import { bytesLabel, libraryApi, modelStatus, uploadModel, type Catalog, type LibraryModel, type ModelTask } from '../lib/model-library';
import styles from '../gpu.module.css';

export function ModelSourceSettings() {
  const [status, setStatus] = useState<Record<string, boolean>>({});
  const [values, setValues] = useState<Record<string, string>>({});
  const [message, setMessage] = useState('');
  useEffect(() => { void libraryApi<Record<string, boolean>>('/source-settings').then(setStatus).catch(e => setMessage(apiMessage(e))); }, []);
  return <details className="foldable"><summary>模型下载源凭据</summary><div className="foldContent">
    {['huggingface', 'civitai'].map(source => <div key={source} className={styles.stack}>
      <label>{source === 'huggingface' ? 'Hugging Face Token' : 'Civitai API Key'} · {status[source] ? '已配置（已脱敏）' : '未配置'}
        <input type="password" autoComplete="new-password" value={values[source] || ''} onChange={e => setValues({...values, [source]:e.target.value})} /></label>
      <div className="actions"><button className="button secondary small" onClick={async () => {try {setStatus(await libraryApi('/source-settings/' + source, {token:values[source]}, 'PUT')); setValues({...values, [source]:''}); setMessage('凭据已加密保存');} catch(e) {setMessage(apiMessage(e));}}} disabled={!values[source]}>保存凭据</button>
      <button className="button secondary small" onClick={async () => {try {setStatus(await libraryApi('/source-settings/' + source, {token:''}, 'PUT'));} catch(e) {setMessage(apiMessage(e));}}}>清除</button></div>
    </div>)}
    {message && <p role="status">{message}</p>}
  </div></details>;
}

export function ModelStorageSettings() {
  const [accounts, setAccounts] = useState<{id:string; account_id?:string; alias?:string}[]>([]);
  const [account, setAccount] = useState('');
  const [message, setMessage] = useState('');
  useEffect(() => {void Promise.all([apiFetch<{id:string;account_id?:string;alias?:string}[]>('/api/gpu/accounts'), libraryApi<{account_id?:string}>('/storage-settings')]).then(([a,s]) => {setAccounts(a);setAccount(s.account_id || '');}).catch(e => setMessage(apiMessage(e)));}, []);
  return <details className="foldable"><summary>模型库 Drive 存储账号</summary><div className="foldContent"><label>长期存储账号<select value={account} onChange={e => setAccount(e.target.value)}><option value="">历史默认 Drive 账号</option>{accounts.map(a => <option key={a.id} value={a.account_id || a.id}>{a.alias || a.account_id || a.id}</option>)}</select></label><button className="button secondary small" onClick={async () => {try {await libraryApi('/storage-settings', {account_id:account || null}, 'PUT'); setMessage('存储账号已保存');} catch(e) {setMessage(apiMessage(e));}}}>保存存储账号</button><p className="muted small">所有 GPU 账号共享此模型库。已有模型后禁止直接切换存储账号，防止索引指向另一块 Drive。</p>{message && <p role="status">{message}</p>}</div></details>;
}

export function ModelLibraryPage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [tasks, setTasks] = useState<ModelTask[]>([]);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('');
  const [architectureFilter, setArchitectureFilter] = useState('');
  const [state, setState] = useState('');
  const [url, setUrl] = useState('');
  const [name, setName] = useState('');
  const [relativePath, setRelativePath] = useState('');
  const [type, setType] = useState('checkpoints');
  const [architecture, setArchitecture] = useState('unknown');
  const [expected, setExpected] = useState('');
  const [driveId, setDriveId] = useState('');
  const [choices, setChoices] = useState<Record<string, unknown>[]>([]);
  const [detailId, setDetailId] = useState<string | null>(null);
  const detailPanelId = useId();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');
  const refresh = useCallback(async () => {
    const [data, jobs] = await Promise.all([libraryApi<Catalog>(), libraryApi<ModelTask[]>('/tasks')]);
    setCatalog(data); setTasks(jobs);
  }, []);
  useEffect(() => {void refresh().catch(e => setError(apiMessage(e))); const id = window.setInterval(() => {void refresh().catch(() => {});}, 3000); return () => clearInterval(id);}, [refresh]);
  async function action(fn: () => Promise<unknown>) {setBusy(true); setError(''); try {await fn(); await refresh();} catch(e) {setError(apiMessage(e));} finally {setBusy(false);}}
  async function add(data: Record<string, unknown>) {
    const model = await libraryApi<LibraryModel>('', {...data, relative_path: relativePath.trim() || undefined, model_type: data.model_type === 'unknown' || !data.model_type ? type : data.model_type, architecture: data.architecture === 'unknown' || !data.architecture ? architecture : data.architecture});
    await libraryApi(`/${model.id}/prepare`, {});
    setChoices([]); setProgress('模型已入库并加入准备队列');
  }
  const items = (catalog?.items || []).filter(m => (!query || `${m.display_name} ${m.filename}`.toLowerCase().includes(query.toLowerCase())) && (!category || m.model_type === category) && (!state || m.state === state) && (!architectureFilter || m.architecture === architectureFilter));
  return <div className={styles.stack}>
    <div className={styles.subHead}><h2>统一模型库</h2><button className="button secondary small" onClick={() => void action(refresh)}>刷新</button></div>
    <p className="muted">{catalog?.statistics.count || 0} 个模型 · Drive 已索引 {bytesLabel(catalog?.statistics.bytes)}。下载与校验由独立模型进程执行，无需启动 GPU。</p>
    <ModelStorageSettings />
    <ModelSourceSettings />
    <details className="foldable"><summary>添加模型 / 上传 / 导入 Drive</summary><div className="foldContent">
      <div className={styles.grid2}>
        <label>模型类型<select value={type} onChange={e => setType(e.target.value)}>{Object.keys(catalog?.types || {checkpoints:''}).map(t => <option key={t}>{t}</option>)}</select></label>
        <label>模型架构<input value={architecture} onChange={e => setArchitecture(e.target.value)} placeholder="unknown" /></label>
        <label>文件名（通用链接 / Drive）<input value={name} onChange={e => setName(e.target.value)} placeholder="model.safetensors" /></label>
        <label>预期 SHA-256（可选）<input value={expected} onChange={e => setExpected(e.target.value)} /></label>
      </div>
      <label>工作流引用的相对路径（可选）<input value={relativePath} onChange={e => setRelativePath(e.target.value)} placeholder="子目录/model.safetensors" /></label>
      <label>模型页面或 HTTPS 下载链接<input value={url} onChange={e => setUrl(e.target.value)} /></label>
      <button disabled={busy || !url} className="button secondary" onClick={() => void action(async () => {const resolved = await libraryApi<{files:Record<string, unknown>[]}>('/resolve', {url, filename:name || undefined, model_type:type, architecture, expected_sha256:expected || undefined}); setChoices(resolved.files); if (!resolved.files.length) setProgress('该来源未返回受支持的模型文件');})}>解析来源并选择文件</button>
      {choices.map((c, i) => <div key={i} className={styles.card}><strong>{String(c.filename)}</strong><p>{String(c.version_name || '')} · {String(c.architecture)} · {bytesLabel(Number(c.expected_size || c.reported_size))}</p><button disabled={busy} className="button small" onClick={() => void action(() => add(c))}>选择此文件并准备</button></div>)}
      <label>已有 Drive 文件 ID<input value={driveId} onChange={e => setDriveId(e.target.value)} /></label>
      <button disabled={busy || !driveId || !name} className="button secondary" onClick={() => void action(() => add({source:'drive', drive_file_id:driveId, filename:name, model_type:type, architecture, expected_sha256:expected || undefined}))}>导入已有文件并校验</button>
      <label>本地模型（直接分块上传到 Drive）<input disabled={busy} type="file" accept=".safetensors,.gguf,.onnx" onChange={e => {const f = e.target.files?.[0]; if(f) void action(async () => {const id = await uploadModel(f, n => setProgress(`上传 ${Math.round(n / f.size * 100)}%`)); await add({source:'upload', drive_file_id:id, filename:f.name, model_type:type, architecture, expected_sha256:expected || undefined});});}} /></label>
      <p className="muted small">Drive OAuth 的 drive.file 权限仅可读取应用可见文件。上传失败会在当前页面重试分块；刷新页面后需重新上传。</p>
    </div></details>
    <div className={`${styles.grid2} ${styles.modelFilters}`}>
      <label>搜索名称 / 文件名<input value={query} onChange={e => setQuery(e.target.value)} /></label>
      <label>类型<select value={category} onChange={e => setCategory(e.target.value)}><option value="">全部类型</option>{Object.keys(catalog?.types || {}).map(t => <option key={t}>{t}</option>)}</select></label>
      <label>状态<select value={state} onChange={e => setState(e.target.value)}><option value="">全部状态</option>{['pending','downloading','verifying','uploading','ready','failed','missing','unverified'].map(s => <option key={s} value={s}>{modelStatus[s]}</option>)}</select></label>
      <label>架构<select value={architectureFilter} onChange={e => setArchitectureFilter(e.target.value)}><option value="">全部架构</option>{[...new Set(catalog?.items.map(m => m.architecture))].map(s => <option key={s}>{s}</option>)}</select></label>
    </div>
    {error && <p className="status error" role="alert">{error}</p>}{progress && <p role="status">{progress}</p>}
    {!items.length && <p className="muted">暂无符合条件的模型。添加来源链接、上传文件或从工作流迁移旧缓存。</p>}
    {!!items.length && <div className={`${styles.stack} ${styles.workspaceGrid}`}>
    {items.map(m => { const task = tasks.find(t => t.model_id === m.id); return <article className={`${styles.card} ${detailId === m.id ? styles.workspaceWide : ''}`} key={m.id}>
      <h3>{m.display_name}</h3><p>{m.relative_path || m.filename} · {m.model_type} · {m.architecture}</p><p>{modelStatus[m.state] || m.state} · {bytesLabel(m.size)} · {m.external_source ? (m.external_source.status === 'verified' ? '外部来源已验证' : '外部来源待验证') : '无外部来源'}</p>
      {task && ['pending','downloading','verifying','uploading'].includes(task.state) && <div>
        <progress aria-label="模型准备进度" max={task.total || 1} value={task.total ? task.bytes : undefined} />
        <p>{modelStatus[task.state]} · {task.bytes ? bytesLabel(task.bytes) : '0 B'} / {bytesLabel(task.total)} · {task.speed ? `${(task.speed / 1024**2).toFixed(1)} MB/s` : '等待进度'}</p>
        {!!task.attempt && task.attempt > 1 && <p className="muted small">第 {task.attempt} 次尝试
          {task.resume_phase && task.resume_offset !== undefined
            && ` · ${modelStatus[task.resume_phase]}从 ${task.resume_offset.toLocaleString()} 字节继续`}
        </p>}
        {task.restart_reason && <p className="muted small">{task.restart_reason}</p>}
        <button className="button secondary small" onClick={() => void action(() => libraryApi(`/tasks/${task.id}/cancel`, {}))}>取消准备</button>
      </div>}
      {m.error && <p className="status error">{m.error}</p>}
      <div className="actions"><button className="button secondary small" aria-expanded={detailId === m.id}
        aria-controls={detailId === m.id ? detailPanelId : undefined}
        onClick={() => setDetailId(detailId === m.id ? null : m.id)}>详情与引用</button>
        <button disabled={busy} className="button secondary small" onClick={() => void action(() => libraryApi(`/${m.id}/prepare`, {action:'prepare'}))}>准备 / 重试</button>
        <button disabled={busy || !m.drive_file_id} className="button secondary small" onClick={() => void action(() => libraryApi(`/${m.id}/prepare`, {action:'verify'}))}>重新校验</button>
        <button disabled={busy || !m.drive_file_id} className="button secondary small" onClick={() => void action(() => libraryApi(`/${m.id}/check`, {}))}>检查 Drive 文件</button>
      </div>
      {detailId === m.id && <ModelDetail modelId={m.id} panelId={detailPanelId} refresh={refresh}
        onClose={() => setDetailId(current => current === m.id ? null : current)} onProgress={setProgress} />}
    </article>;})}
    </div>}
  </div>;
}

type LibraryModelDetail = LibraryModel & {references?: {workflow_id: string}[]};

function ModelDetail({modelId, panelId, refresh, onClose, onProgress}: {
  modelId: string;
  panelId: string;
  refresh: () => Promise<void>;
  onClose: () => void;
  onProgress: (message: string) => void;
}) {
  const [detail, setDetail] = useState<LibraryModelDetail | null>(null);
  const [externalSource, setExternalSource] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    void libraryApi<LibraryModelDetail>('/' + modelId).then(model => {
      if (!active) return;
      setDetail(model);
      setExternalSource(model.external_source?.url || '');
    }).catch(e => { if (active) setError(apiMessage(e)); });
    return () => { active = false; };
  }, [modelId]);

  async function action(fn: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return <section id={panelId} className={styles.inlinePanel} aria-label="模型详情">
    {error && <p className="status error" role="alert">{error}</p>}
    {!detail && !error && <p role="status">正在加载模型详情…</p>}
    {detail && <><h3>{detail.display_name}</h3><p>SHA-256：{detail.sha256 || '未计算'}</p><p>来源：{detail.source} · 版本：{detail.version_name || '未知'}</p><p>预期 SHA-256：{detail.expected_sha256 || '未提供；计算 Hash 不代表来源可信'}</p><p>触发词：{detail.trigger_words?.join(', ') || '无'}</p><p>引用工作流：{detail.references?.length || 0}</p>{detail.references?.map(r => <p key={r.workflow_id}>{r.workflow_id}</p>)}
      <label>外部来源链接<input type="url" value={externalSource} onChange={e => setExternalSource(e.target.value)} /></label>
      <p className="muted small">可填写无凭据的 HTTPS 来源链接，凭据在「模型下载源凭据」中配置。留空并保存可移除绑定。</p>
      <button className="button secondary small" disabled={busy} onClick={() => void action(async () => {
        const updated = await libraryApi<LibraryModel>('/' + detail.id, { external_source_url: externalSource.trim() || null }, 'PATCH');
        setDetail({ ...detail, external_source: updated.external_source }); setExternalSource(updated.external_source?.url || '');
      })}>保存外部来源</button>
      <label>显示名称<input value={detail.display_name} onChange={e => setDetail({...detail, display_name:e.target.value})} /></label>
      <label>架构<input value={detail.architecture} onChange={e => setDetail({...detail, architecture:e.target.value})} /></label>
      <label>标签（逗号分隔）<input value={detail.tags?.join(', ') || ''} onChange={e => setDetail({...detail, tags:e.target.value.split(',').map(s => s.trim())})} /></label>
      <div className="actions"><button className="button small" disabled={busy} onClick={() => void action(() => libraryApi('/' + detail.id, {display_name:detail.display_name, architecture:detail.architecture, tags:detail.tags}, 'PATCH'))}>保存信息</button>
      <button className="button danger small" disabled={busy} onClick={() => {if(window.confirm(`移除模型索引？引用工作流 ${detail.references?.length || 0} 个，相关依赖将失效。`)) void action(async () => {await libraryApi(`/${detail.id}?confirm_references=true`, undefined, 'DELETE'); onClose();});}}>移除索引</button>
      <button className="button danger small" disabled={busy} onClick={() => {if(window.confirm(`将 Drive 物理文件移至回收站并移除索引？引用工作流 ${detail.references?.length || 0} 个。`)) void action(async () => {await libraryApi(`/${detail.id}?confirm_references=true&delete_file=true`, undefined, 'DELETE'); onClose();});}}>删除 Drive 文件</button>
      <button className="button secondary small" disabled={busy || detail.state !== 'ready'} onClick={() => {
        const key=window.prompt('登记到工作流审核流程：请输入唯一模型标识（字母、数字、短横线）'); if(!key) return;
        const license=window.prompt('请输入模型的实际许可证名称；登记后仍需人工接受许可和真实运行验证'); if(!license) return;
        const vram=window.prompt('请输入该模型实际需要的显存 GB（用于人工审核，不自动推断）'); if(!vram || !Number.isFinite(Number(vram)) || Number(vram) <= 0) return;
        void action(async () => {await libraryApi(`/${detail.id}/workflow-model`, {model_key:key, license, resource_profile:{vram_gb:Number(vram)}}); onProgress('已登记为未启用模型，请在工作流配置中关联该标识并完成审核与真实 GPU 验证');});
      }}>登记为工作流模型</button>
      </div>
    </>}
    <div className="actions"><button className="button secondary small" disabled={busy} onClick={onClose}>关闭详情</button></div>
  </section>;
}
