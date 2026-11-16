import { useEffect, useState } from 'react';
import { apiFetch, getGpuComputeSpecs, getGpuSettings, type ComputeSpec, type GpuManifest } from '../lib/gpu-api';
import { resolveGpuCheckRequest } from '../lib/colab-check';
import type { GpuAccount } from './AccountDirectory';

export function computeLabel(key: Record<string, unknown> | null | undefined): string {
  if (!key) return '工作流默认规格';
  return `${key.accelerator || '默认 GPU'} · ${String(key.shape || '').includes('HIGHMEM') ? '高系统内存' : '标准系统内存'}`;
}
export function WorkflowGpuSelection({ manifest, accountId, value, onChange }: {
  manifest: GpuManifest | null; accountId?: string | null; value: Record<string, unknown> | null;
  onChange: (key: Record<string, unknown> | null) => void;
}) {
  const [specs, setSpecs] = useState<ComputeSpec[]>([]);
  const [error, setError] = useState('');
  const [accountName, setAccountName] = useState('');
  const [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let live = true; setSpecs([]); setError(''); setAccountName(''); setLoading(true);
    async function load() {
      try {
        const [accounts, settings] = await Promise.all([
          apiFetch<GpuAccount[]>('/api/gpu/accounts'), getGpuSettings(),
        ]);
        if (!live) return;
        const request = resolveGpuCheckRequest(settings, accounts, accountId || null);
        if (request.provider !== 'colab') {
          throw new Error(request.reason === 'offline' ? '站点处于离线测试模式，不能查询真实 Colab GPU。'
            : request.reason === 'real_disabled' ? '真实 GPU 开关尚未启用，请到设置中检查。'
              : accountId ? '所选账号未连接 Colab 或已停用，请到设置中检查授权。'
                : '账号池中没有已连接 Colab 且启用的账号，请到设置中完成授权。');
        }
        setAccountName(accounts.find(a => a.account_id === request.accountId)?.alias || request.accountId);
        const specs = await getGpuComputeSpecs('colab', request.accountId);
        if (live) setSpecs(specs.filter(s => s.eligible === true && typeof s.key === 'object' && ['GPU', 'VARIANT_GPU'].includes(String(s.key?.variant))));
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : '规格查询失败');
      } finally {
        if (live) setLoading(false);
      }
    }
    void load();
    return () => { live = false; };
  }, [accountId, refresh]);
  const token = (k: Record<string, unknown>) => JSON.stringify([k.accelerator, String(k.shape).replace('SHAPE_', ''), String(k.variant).replace('VARIANT_', '')]);
  const current = value ? token(value) : '';
  const present = specs.some(s => token(s.key as Record<string, unknown>) === current);
  return <div className="field">
    <label htmlFor="workflow-gpu">本次 GPU</label>
    <select id="workflow-gpu" value={current} onChange={e => onChange((specs.find(s => token(s.key as Record<string, unknown>) === e.target.value)?.key as Record<string, unknown>) || null)}>
      <option value="">默认 / 推荐 GPU · {computeLabel(manifest?.compute_spec as Record<string, unknown>)}</option>
      {value && !present && <option value={current}>{computeLabel(value)}（上次选择，提交时重新校验）</option>}
      {specs.map(s => <option key={token(s.key as Record<string, unknown>)} value={token(s.key as Record<string, unknown>)}>{computeLabel(s.key as Record<string, unknown>)} · {s.vram_gb ? `${s.vram_gb} GB 显存` : '显存以实际设备为准'} · {typeof s.consumptionRateHourly === 'number' ? `${s.consumptionRateHourly} CU/h` : 'CU 速率未提供'}</option>)}
    </select>
    <p className="muted small">将申请：{computeLabel(value || manifest?.compute_spec as Record<string, unknown>)}。默认规格是作者推荐；其他规格可能未经验证，按实际显存检查最低要求。不会自动切换 GPU。</p>
    {value && <p className="status">所选规格可能尚未验证；保留工作流原有精度、模型和生成参数。运行失败也可能消耗 CU。</p>}
    {error && <p className="status error" role="alert">{error}</p>}
    {accountName && <p className="muted small">查询账号：{accountName}。有申请资格不代表当前有空闲 GPU。</p>}
    <button type="button" disabled={loading} className="button ghost small" onClick={() => setRefresh(n => n + 1)}>{loading ? '正在查询账号支持的 GPU…' : '刷新当前账号支持的 GPU'}</button>
  </div>;
}
