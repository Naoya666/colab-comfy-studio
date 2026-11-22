import { useEffect, useId, useState, type ReactNode } from 'react';
import {
  apiFetch, getGpuComputeSpecs, getGpuSettings, getGpuSpecOrder,
  type ComputeSpec, type GpuManifest, type GpuSpecKey,
} from '../lib/gpu-api';
import { resolveGpuCheckRequest } from '../lib/colab-check';
import { DEFAULT_GPU_SPEC_ORDER, isBelowWorkflowDefault } from '../lib/gpu-spec-order';
import type { GpuAccount } from './AccountDirectory';

export function computeLabel(key: Record<string, unknown> | null | undefined): string {
  if (!key) return '工作流默认规格';
  return `${key.accelerator || '默认 GPU'} · ${String(key.shape || '').includes('HIGHMEM') ? '高系统内存' : '标准系统内存'}`;
}

export function WorkflowGpuSelection({
  manifest, accountId, value, onChange, label = '本次 GPU', requireSelection = false, rateHint, gpuOrder,
}: {
  manifest: GpuManifest | null;
  accountId?: string | null;
  value: Record<string, unknown> | null;
  onChange: (key: Record<string, unknown> | null) => void;
  label?: string;
  requireSelection?: boolean;
  rateHint?: (rate?: number | null) => ReactNode;
  gpuOrder?: GpuSpecKey[];
}) {
  const selectId = useId();
  const [specs, setSpecs] = useState<ComputeSpec[]>([]);
  const [loadedOrder, setLoadedOrder] = useState<GpuSpecKey[]>(DEFAULT_GPU_SPEC_ORDER);
  const [error, setError] = useState('');
  const [accountName, setAccountName] = useState('');
  const [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (gpuOrder) return;
    let live = true;
    void getGpuSpecOrder().then((res) => {
      if (live && Array.isArray(res.ordered_specs) && res.ordered_specs.length) {
        setLoadedOrder(res.ordered_specs);
      }
    }).catch(() => {});
    return () => { live = false; };
  }, [gpuOrder]);

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

  const effectiveOrder = gpuOrder && gpuOrder.length ? gpuOrder : loadedOrder;
  const workflowDefault = (manifest?.compute_spec as Record<string, unknown> | undefined) || null;
  const token = (k: Record<string, unknown>) => JSON.stringify([k.accelerator, String(k.shape).replace('SHAPE_', ''), String(k.variant).replace('VARIANT_', '')]);
  const current = value ? token(value) : '';
  const selected = specs.find(s => token(s.key as Record<string, unknown>) === current);
  const selectedTooLow = Boolean(value && workflowDefault && isBelowWorkflowDefault(value, workflowDefault, effectiveOrder));

  return <div className="field">
    <label htmlFor={selectId}>{label}</label>
    <select id={selectId} value={current} onChange={e => onChange((specs.find(s => token(s.key as Record<string, unknown>) === e.target.value)?.key as Record<string, unknown>) || null)}>
      {!requireSelection && <option value="">默认 / 推荐 GPU · {computeLabel(workflowDefault)}</option>}
      {value && !selected && <option value={current}>{computeLabel(value)}{selectedTooLow ? '（低于工作流默认规格，不支持）' : '（上次选择，提交时重新校验）'}</option>}
      {specs.map(s => {
        const key = s.key as Record<string, unknown>;
        const belowDefault = Boolean(workflowDefault && isBelowWorkflowDefault(key, workflowDefault, effectiveOrder));
        return (
          <option key={token(key)} value={token(key)} disabled={belowDefault}>
            {computeLabel(key)} · {s.vram_gb ? `${s.vram_gb} GB 显存` : '显存以实际设备为准'} · {s.reference_rate_hourly != null ? `${s.reference_rate_hourly} CU/h` : '估算费率未知'}{belowDefault ? '（低于工作流默认规格，不支持）' : ''}
          </option>
        );
      })}
    </select>
    <p className="muted small">将申请：{computeLabel(value || workflowDefault)}。默认规格为工作流最低推荐规格；只能选择同规格或更高规格 GPU，不支持降配运行。</p>
    {selectedTooLow && (
      <p className="status error" role="alert">
        所选 GPU（{computeLabel(value)}）低于工作流默认规格（{computeLabel(workflowDefault)}），不支持降配运行。请改回默认规格或选择更高规格 GPU。
      </p>
    )}
    {value && rateHint?.(selected?.reference_rate_hourly)}
    {value && manifest && !selectedTooLow && <p className="status">所选规格可能尚未验证；保留工作流原有精度、模型和生成参数。运行失败也可能消耗 CU。</p>}
    {error && <p className="status error" role="alert">{error}</p>}
    {accountName && <p className="muted small">查询账号：{accountName}。有申请资格不代表当前有空闲 GPU。</p>}
    <button type="button" disabled={loading} className="button ghost small" onClick={() => setRefresh(n => n + 1)}>{loading ? '正在查询账号支持的 GPU…' : '刷新当前账号支持的 GPU'}</button>
  </div>;
}
