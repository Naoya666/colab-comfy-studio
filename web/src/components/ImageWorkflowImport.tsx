import { useEffect, useRef, useState } from 'react';
import { extractPng, MAX_IMAGE, type ExtractedWorkflow } from '../lib/png-workflow';

export function ImageWorkflowImport({ onImport }: { onImport: (bundle: string) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const revision = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [result, setResult] = useState<ExtractedWorkflow | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<'workflow' | 'prompt'>('workflow');
  useEffect(() => { if (!file) { setUrl(''); return; } const next = URL.createObjectURL(file); setUrl(next); return () => URL.revokeObjectURL(next); }, [file]);
  async function read(file?: File) {
    const token = ++revision.current;
    setFile(null); setResult(null); setError('');
    if (!file) { setBusy(false); return; }
    setBusy(true);
    try {
      if (file.size > MAX_IMAGE) throw new Error('图片不能超过 32 MiB');
      const data = await extractPng(new Uint8Array(await file.arrayBuffer()));
      if (token !== revision.current) return;
      setFile(file); setResult(data); setSelected(data.workflow ? 'workflow' : 'prompt');
    } catch (e) { if (token === revision.current) setError(e instanceof Error ? e.message : '图片解析失败'); }
    finally { if (token === revision.current) setBusy(false); }
  }
  const json = result?.[selected] ? JSON.stringify(result[selected], null, 2) : '';
  function download(key: 'workflow' | 'prompt') {
    const url = URL.createObjectURL(new Blob([JSON.stringify(result?.[key], null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `${key}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <details className="foldable"><summary>从图片导入 ComfyUI 工作流</summary><div className="foldContent" style={{ overflowWrap: 'anywhere' }}
    onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); void read(e.dataTransfer.files[0]); }}>
    <p className="muted small">选择或拖入 PNG 原图（最多 32 MiB）。在浏览器本地解析，不上传原图，不申请 GPU。</p>
    <input ref={input} type="file" hidden accept="image/png,.png" onChange={e => { void read(e.target.files?.[0]); e.target.value = ''; }} />
    <button className="button secondary" type="button" onClick={() => input.current?.click()}>{busy ? '解析中…' : '选择 PNG 图片 / 相册'}</button>
    {error && <p role="alert" className="status error">{error}</p>}
    {result && <>
      <img src={url} alt="导入图片缩略图" style={{ maxWidth: '100%', maxHeight: 180, objectFit: 'contain' }} />
      <p>{file?.name}</p>
      <p>workflow：{result.workflow ? '已检测' : '未检测'} · API prompt：{result.prompt ? '已检测' : '未检测'} · 节点：{result.node_count}</p>
      <p className="small">模型 / LoRA / VAE / ControlNet 文件线索：{result.model_files.join('、') || '未识别'}</p>
      <p className="small">节点类型（含待审核自定义节点，不能仅凭类型确认来源）：{result.node_types.join('、')}</p>
      {result.warnings.map(w => <p key={w} className="status">{w}</p>)}
      <p className="muted small">草稿仍需补充模型来源与 SHA-256、ComfyUI 版本、节点依赖、object_info、参数映射、GPU 和许可。不会自动安装节点或标记为已验证。</p>
      <div className="actions">{(['workflow', 'prompt'] as const).filter(k => result[k]).map(k => <button type="button" className="button secondary small" key={k} onClick={() => download(k)}>下载 {k}.json</button>)}</div>
      <label>JSON 预览 <select value={selected} onChange={e => setSelected(e.target.value as typeof selected)}>{(['workflow', 'prompt'] as const).filter(k => result[k]).map(k => <option key={k}>{k}</option>)}</select></label>
      <textarea aria-label="提取 JSON 预览" readOnly rows={10} value={json} />
      <div className="actions">
        <button className="button secondary small" type="button" onClick={() => { void navigator.clipboard.writeText(json).catch(() => setError('复制失败，请从 JSON 预览中手动复制')); }}>复制 JSON</button>
        <button className="button small" type="button" onClick={() => onImport(JSON.stringify({ workflow_key: `image-${crypto.randomUUID()}`, image_extraction: { workflow: result.workflow, prompt: result.prompt } }))}>导入为待配置草稿</button>
        <button className="button ghost small" type="button" onClick={() => { void read(); }}>取消 / 重新上传</button>
      </div>
    </>}
  </div></details>;
}
