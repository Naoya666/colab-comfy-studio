import { useState } from 'react';
import { apiFetch, type GpuWorkflow } from '../lib/gpu-api';
export function ImageDraftConfiguration({ workflow, onDone }: { workflow: GpuWorkflow; onDone: () => void }) {
  const [text, setText] = useState(() => JSON.stringify({ model_id: workflow.model_id, manifest: workflow.manifest, api_graph: workflow.api_graph, object_info: workflow.object_info }, null, 2));
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const raw = workflow.image_extraction as Record<string, unknown> | undefined;
  const hints = workflow.image_import_summary as { model_files?: string[]; node_types?: string[] } | undefined;
  return <details className="foldable"><summary>待配置图片草稿 / 补充运行配置</summary><div className="foldContent">
    <p className="muted small">缺失信息：{Array.isArray(workflow.pending_configuration) ? workflow.pending_configuration.join('；') : '已配置，仍需按原流程审核和冒烟'}。原始 workflow / prompt 保留在此草稿中。</p>
    <p className="small">文件线索：{hints?.model_files?.join('、') || '未识别'}；节点类型（待审核）：{hints?.node_types?.join('、') || '未识别'}</p>
    <details><summary>原始提取 JSON（只读）</summary><textarea readOnly rows={8} aria-label="原始提取 JSON" value={JSON.stringify(raw, null, 2)} />
      {(['workflow', 'prompt'] as const).filter(k => raw?.[k]).map(k => <button key={k} className="button secondary small" type="button" onClick={() => {
        const url = URL.createObjectURL(new Blob([JSON.stringify(raw?.[k], null, 2)], { type: 'application/json' }));
        const a = document.createElement('a'); a.href = url; a.download = `${k}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      }}>下载原始 {k}.json</button>)}
    </details>
    <p className="muted small">补充 model_id、manifest（模型来源、校验值、ComfyUI 版本、参数映射、GPU 等）和 object_info。仅有可视化 JSON 时须另行提供 api_graph。</p>
    <textarea aria-label="草稿运行配置 JSON" rows={12} value={text} onChange={e => setText(e.target.value)} />
    <button className="button small" type="button" disabled={busy} onClick={async () => {
      setBusy(true); setError('');
      try { await apiFetch(`/api/gpu/workflows/${encodeURIComponent(workflow.id)}/configuration`, { method: 'PUT', body: JSON.stringify(JSON.parse(text)) }); onDone(); }
      catch (e) { setError(e instanceof Error ? e.message : '保存失败'); }
      finally { setBusy(false); }
    }}>保存配置（仍为草稿）</button>
    {error && <p className="status error" role="alert">{error}</p>}
  </div></details>;
}
