import { useCallback, useEffect, useState } from 'react';
import { apiMessage } from '../lib/gpu-api';
import { bytesLabel, dependencyApi, libraryApi, modelStatus, type Catalog, type Dependency, type DependencyRecord, type LibraryModel } from '../lib/model-library';
import styles from '../gpu.module.css';

function DependencyRow({dep, models, onBind, onSource, onSearch}: {dep:Dependency; models:LibraryModel[]; onBind:(id:string, strengths:Record<string,number>, disabled:boolean) => void; onSource:(url:string) => void; onSearch:() => void}) {
  const [chosen, setChosen] = useState(dep.model_id || '');
  const [strengths, setStrengths] = useState(dep.strengths || {});
  const [disabled, setDisabled] = useState(dep.disabled || false);
  const [url, setUrl] = useState('');
  useEffect(() => {setChosen(dep.model_id || ''); setStrengths(dep.strengths || {}); setDisabled(dep.disabled || false);}, [dep]);
  const bound = models.find(m => m.id === dep.model_id);
  return <article className={styles.card}><strong>{dep.reference}</strong><p>{dep.model_type} · {dep.architecture} · {dep.required ? '必需' : '可选'} · {modelStatus[dep.status] || dep.status}</p>
    <p className="muted small">节点 {dep.node_id} · {dep.node_type} · {dep.field}</p>
    {bound && <p>{bound.display_name} · {modelStatus[bound.state]} · {bytesLabel(bound.size)}{bound.error ? ` · ${bound.error}` : ''}</p>}
    <label>绑定 / 替换模型<select value={chosen} onChange={e => setChosen(e.target.value)}><option value="">选择模型库文件</option>{models.filter(m => m.model_type === dep.model_type).map(m => <option key={m.id} value={m.id}>{m.display_name} · {m.architecture} · {modelStatus[m.state]}</option>)}</select></label>
    {Object.entries(strengths).map(([key, value]) => <label key={key}>{key}<input type="number" min={-3} max={3} step={0.05} value={value} onChange={e => setStrengths({...strengths, [key]:Number(e.target.value)})} /></label>)}
    {!!Object.keys(strengths).length && <label><input type="checkbox" checked={disabled} onChange={e => setDisabled(e.target.checked)} />关闭此 LoRA</label>}
    {bound?.trigger_words?.length ? <p>触发词：{bound.trigger_words.join(', ')}</p> : null}
    <div className="actions"><button className="button secondary small" disabled={!chosen} onClick={() => onBind(chosen, strengths, disabled)}>保存绑定与权重</button><button className="button secondary small" onClick={onSearch}>搜索来源候选</button>
    {bound && <button className="button secondary small" onClick={() => void libraryApi(`/${bound.id}/prepare`, {}).catch(e => window.alert(apiMessage(e)))}>准备此模型</button>}</div>
    <label>补充模型链接<input value={url} onChange={e => setUrl(e.target.value)} placeholder="Civitai / Hugging Face / HTTPS" /></label><button className="button secondary small" disabled={!url} onClick={() => onSource(url)}>解析来源候选</button>
  </article>;
}

export function ModelDependencies({workflowId, onDraft}: {workflowId:string; onDraft?:() => void}) {
  const [record, setRecord] = useState<DependencyRecord | null>(null);
  const [models, setModels] = useState<LibraryModel[]>([]);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<{dep:Dependency; file:Record<string,unknown>}[]>([]);
  const [candidates, setCandidates] = useState<{name:string; url:string}[]>([]);
  const refresh = useCallback(async () => {const [r,c] = await Promise.all([dependencyApi<DependencyRecord>(workflowId), libraryApi<Catalog>()]); setRecord(r); setModels(c.items);}, [workflowId]);
  useEffect(() => {void refresh().catch(e => setError(apiMessage(e))); const id=setInterval(() => {void refresh().catch(() => {});}, 4000); return () => clearInterval(id);}, [refresh]);
  async function action(fn:() => Promise<unknown>) {setError(''); try {await fn(); await refresh();} catch(e) {setError(apiMessage(e));}}
  return <details className="foldable"><summary>模型依赖</summary><div className="foldContent">
    <p>{record?.models_ready ? '模型依赖已准备完成' : '模型依赖待准备'} · 工作流：{record?.workflow_state || '加载中'} · {record?.runtime_validated ? '已通过真实运行验证' : '运行环境 / 节点及真实运行仍需验证'}</p>
    <div className="actions">{[['scan','重新扫描'], ['match','重新匹配'], ['prepare','准备全部缺失模型'], ['adopt-cache','索引并校验旧缓存']].map(([key, text]) => <button key={key} className="button secondary small" onClick={() => void action(async () => {const result = await dependencyApi<{unresolved?:string[]}>(workflowId, key); setMessage(result.unresolved?.length ? `${result.unresolved.length} 项仍需补充来源或确认候选` : '操作已提交');})}>{text}</button>)}</div>
    <p className="muted small">模型准备、来源可信度和工作流运行验证分别管理。替换已审核模型后，可生成绑定草稿并按现有流程审核、验证和发布。</p>
    <button className="button secondary small" onClick={() => void action(async () => {await dependencyApi(workflowId, 'draft'); setMessage('已创建绑定草稿，请在工作流列表配置、审核并验证'); onDraft?.();})}>将当前绑定生成新草稿</button>
    {error && <p role="alert" className="status error">{error}</p>}{message && <p role="status">{message}</p>}
    {!record?.dependencies.length && <p className="muted">未发现可可靠识别的模型引用。可为自定义节点配置识别规则。</p>}
    {record?.dependencies.map(dep => <DependencyRow key={dep.id} dep={dep} models={models}
      onBind={(id, strengths, disabled) => {const chosen = models.find(m => m.id === id); const unknown = dep.architecture === 'unknown' || chosen?.architecture === 'unknown'; if (unknown && !window.confirm('依赖或模型架构未知，请确认该模型兼容当前工作流。模型身份变化仍需重新审核。')) return; void action(() => dependencyApi(workflowId, 'bind', {dependency_id:dep.id, model_id:id, strengths, disabled, confirm_unknown:unknown}));}}
      onSource={url => void action(async () => {const r=await libraryApi<{files:Record<string,unknown>[]}>('/resolve', {url, model_type:dep.model_type, filename:dep.reference}); setFiles(r.files.map(file => ({dep, file})));})}
      onSearch={() => void action(async () => {const r=await libraryApi<{candidates:{name:string;url:string}[]}>('/search?q=' + encodeURIComponent(dep.reference)); setCandidates(r.candidates);})} />)}
    {candidates.map(c => <p key={c.url}><a href={c.url} target="_blank" rel="noreferrer">{c.name}</a> · 候选来源，打开确认后填入模型链接</p>)}
    {files.map(({dep,file},i) => <div className={styles.card} key={i}><p>{String(file.filename)} · {String(file.architecture)} · {String(file.version_name || '')}</p><button className="button small" onClick={() => {if(!window.confirm('确认使用此具体文件？请核对模型版本与架构。')) return; void action(async () => {const m=await libraryApi<LibraryModel>('', {...file, model_type:dep.model_type}); await dependencyApi(workflowId,'bind',{dependency_id:dep.id, model_id:m.id, confirm_unknown:true}); await libraryApi(`/${m.id}/prepare`,{}); setFiles([]);});}}>确认此文件、绑定并准备</button></div>)}
  </div></details>;
}

export interface ModelOverride { dependency_id:string; model_id:string; strengths?:Record<string,number>; disabled?:boolean; confirm_unknown:boolean; }
export function ModelRunOverrides({workflowId, value, onChange}: {workflowId:string; value:ModelOverride[]; onChange:(v:ModelOverride[]) => void}) {
  const [record, setRecord] = useState<DependencyRecord | null>(null);
  const [models, setModels] = useState<LibraryModel[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {setError(''); setRecord(null); void Promise.all([dependencyApi<DependencyRecord>(workflowId), libraryApi<Catalog>()]).then(([r,c]) => {setRecord(r);setModels(c.items);}).catch(e => setError(apiMessage(e)));}, [workflowId]);
  return <details className="foldable"><summary>本次底模 / LoRA 覆盖（{value.length} 项）</summary><div className="foldContent"><p className="muted small">仅影响本次任务。新模型需先在同一工作流系列通过审核和真实运行验收，之后可直接切换；未知架构需确认。</p>
    {error && <p className="status error">{error}</p>}
    {record?.dependencies.filter(d => ['checkpoints','loras'].includes(d.model_type)).map(dep => {
      const override=value.find(o => o.dependency_id === dep.id);
      const chosen=override?.model_id || dep.model_id || '';
      function update(model_id:string, strengths = override?.strengths || dep.strengths, disabled = override?.disabled || false) {
        const model=models.find(m => m.id === model_id); if(!model) return;
        const unknown=dep.architecture === 'unknown' || model.architecture === 'unknown';
        if(unknown && !window.confirm('模型或依赖架构未知，请确认兼容当前工作流。')) return;
        onChange([...value.filter(v => v.dependency_id !== dep.id), {dependency_id:dep.id,model_id,strengths,disabled,confirm_unknown:unknown}]);
      }
      return <div key={dep.id}><label>{dep.model_type} · 节点 {dep.node_id}<select value={chosen} onChange={e => update(e.target.value)}><option value="">使用工作流原模型</option>{models.filter(m => m.model_type === dep.model_type && m.state === 'ready').map(m => <option key={m.id} value={m.id}>{m.display_name} · {m.architecture}</option>)}</select></label>
        {Object.entries(override?.strengths || dep.strengths || {}).map(([field,strength]) => <label key={field}>{field}<input type="number" min={-3} max={3} step={0.05} value={strength} onChange={e => update(chosen,{...(override?.strengths || dep.strengths),[field]:Number(e.target.value)})}/></label>)}
        {!!Object.keys(dep.strengths || {}).length && <label><input type="checkbox" checked={!!override?.disabled} onChange={e => update(chosen,undefined,e.target.checked)} />关闭此 LoRA</label>}
      </div>;
    })}
    {record && !record.dependencies.some(d => d.model_type === 'loras') && <p className="muted small">此工作流没有可识别的 LoRA 节点，不能直接新增 LoRA。</p>}
    <button className="button secondary small" onClick={() => onChange([])}>清除本次覆盖</button>
  </div></details>;
}
