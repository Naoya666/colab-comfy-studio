import { useCallback, useEffect, useState } from 'react';
import { ArrowDown, ArrowUp, Cpu, Plus, RefreshCw, RotateCcw, Save, Trash2 } from 'lucide-react';
import {
  apiMessage, discoverGpuSpecs, getGpuSpecOrder, resetGpuSpecOrder, saveGpuSpecOrder,
  type DiscoveredGpuSpec, type GpuSpecKey, type GpuSpecOrderSettings,
} from '../lib/gpu-api';
import { formatGpuSpecName, gpuSpecId } from '../lib/gpu-spec-order';
import { Card } from './ui';
import styles from '../gpu.module.css';

export function GpuOrderSettings({ canEdit }: { canEdit: boolean }) {
  const [data, setData] = useState<GpuSpecOrderSettings | null>(null);
  const [draft, setDraft] = useState<GpuSpecKey[]>([]);
  const [discovered, setDiscovered] = useState<DiscoveredGpuSpec[]>([]);
  const [busy, setBusy] = useState('');
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const apply = useCallback((next: GpuSpecOrderSettings) => {
    setData(next);
    setDraft(next.ordered_specs || []);
    setDiscovered(next.discovered_specs || []);
  }, []);

  const load = useCallback(async () => {
    setError('');
    try {
      apply(await getGpuSpecOrder());
    } catch (cause) {
      setError(apiMessage(cause));
    }
  }, [apply]);

  useEffect(() => { void load(); }, [load]);

  const draftIds = new Set(draft.map((item) => gpuSpecId(item)).filter(Boolean));
  const pendingDiscovered = discovered.filter((item) => {
    const id = gpuSpecId(item.spec);
    return id && !draftIds.has(id);
  });
  const defaultIds = new Set((data?.defaults || []).map((item) => gpuSpecId(item)).filter(Boolean));
  const dirty = Boolean(data) && JSON.stringify(draft) !== JSON.stringify(data?.ordered_specs || []);

  function move(index: number, delta: -1 | 1) {
    const target = index + delta;
    if (target < 0 || target >= draft.length) return;
    const next = [...draft];
    [next[index], next[target]] = [next[target], next[index]];
    setDraft(next);
    setStatus('');
  }

  function removeAt(index: number) {
    if (draft.length <= 1) return;
    const removed = draft[index];
    setDraft(draft.filter((_, idx) => idx !== index));
    const removedId = gpuSpecId(removed);
    if (removedId && !defaultIds.has(removedId) && !discovered.some((item) => gpuSpecId(item.spec) === removedId)) {
      setDiscovered((curr) => [...curr, { spec: removed, discovered_at: Math.floor(Date.now() / 1000) }]);
    }
    setStatus('');
  }

  function addDiscovered(spec: GpuSpecKey) {
    const id = gpuSpecId(spec);
    if (!id || draftIds.has(id)) return;
    setDraft((curr) => [...curr, spec]);
    setStatus('已加入列表末尾，请调整到合适位置后点击「保存顺序」。');
  }

  async function save() {
    if (!canEdit || busy || !draft.length) return;
    setBusy('save');
    setError('');
    setStatus('');
    try {
      apply(await saveGpuSpecOrder(draft));
      setStatus('GPU 规格顺序已保存，并立即对所有复用与底线校验生效。');
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy('');
    }
  }

  async function reset() {
    if (!canEdit || busy) return;
    setBusy('reset');
    setError('');
    setStatus('');
    try {
      apply(await resetGpuSpecOrder());
      setStatus('已恢复默认 GPU 规格顺序。');
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy('');
    }
  }

  async function discover() {
    if (!canEdit || busy) return;
    setBusy('discover');
    setError('');
    setStatus('');
    try {
      const result = await discoverGpuSpecs();
      apply(result);
      const added = result.newly_discovered?.length || 0;
      const errCount = result.errors?.length || 0;
      setStatus(
        added > 0
          ? `已探测 ${result.checked_accounts} 个账号，新发现 ${added} 种未纳入排序的 GPU 规格。`
          : `已探测 ${result.checked_accounts} 个账号，暂未发现新的未排序 GPU${errCount ? `（${errCount} 个账号查询失败）` : ''}。`,
      );
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy('');
    }
  }

  return (
    <Card
      icon={<Cpu size={17} strokeWidth={1.6} aria-hidden />}
      title="GPU 规格顺序表（低 → 高）"
      aside={pendingDiscovered.length > 0 ? <span className="pill solid">新发现 {pendingDiscovered.length}</span> : undefined}
    >
      <p className="muted small">
        从低到高排列。调度复用时只允许同规格或更高规格替代；任何低于工作流默认规格的选择都会被拦截。未加入排序的新 GPU 仅允许精确同规格复用。
      </p>
      {status && <p className="status" role="status">{status}</p>}
      {error && <p className="status error" role="alert">{error}</p>}

      {!data ? (
        !error && <p className="muted small">正在读取 GPU 规格顺序…</p>
      ) : (
        <div className={styles.stack}>
          <ol className={styles.gpuOrderList}>
            {draft.map((spec, index) => {
              const id = gpuSpecId(spec) || `${spec.accelerator}:${spec.shape}`;
              const isCustom = !defaultIds.has(id);
              return (
                <li key={id} className={styles.gpuOrderRow}>
                  <span className={styles.gpuOrderInfo}>
                    <strong className={styles.gpuOrderRank}>#{index + 1}</strong>
                    <span>{formatGpuSpecName(spec)}</span>
                    <small className="muted">{id}</small>
                    {isCustom && <span className="pill">自定义</span>}
                  </span>
                  {canEdit && (
                    <span className={styles.gpuOrderControls}>
                      <button
                        type="button"
                        className="button ghost small"
                        aria-label={`上移 ${id}`}
                        disabled={Boolean(busy) || index === 0}
                        onClick={() => move(index, -1)}
                      >
                        <ArrowUp size={14} aria-hidden />
                      </button>
                      <button
                        type="button"
                        className="button ghost small"
                        aria-label={`下移 ${id}`}
                        disabled={Boolean(busy) || index === draft.length - 1}
                        onClick={() => move(index, 1)}
                      >
                        <ArrowDown size={14} aria-hidden />
                      </button>
                      {isCustom && (
                        <button
                          type="button"
                          className="button ghost small"
                          aria-label={`移出 ${id}`}
                          disabled={Boolean(busy) || draft.length <= 1}
                          onClick={() => removeAt(index)}
                        >
                          <Trash2 size={14} aria-hidden />
                        </button>
                      )}
                    </span>
                  )}
                </li>
              );
            })}
          </ol>

          {pendingDiscovered.length > 0 && (
            <div className={`status ${styles.stack}`} role="region" aria-label="新探测到的 GPU">
              <strong>新探测到的上游 GPU（尚未纳入排序，仅支持精确同规格复用）：</strong>
              <div className={styles.stack}>
                {pendingDiscovered.map((item) => {
                  const id = gpuSpecId(item.spec) || `${item.spec.accelerator}:${item.spec.shape}`;
                  return (
                    <div key={id} className={styles.gpuDiscoveredRow}>
                      <span>{formatGpuSpecName(item.spec)} <small className="muted">({id})</small></span>
                      {canEdit && (
                        <button
                          type="button"
                          className="button secondary small"
                          disabled={Boolean(busy)}
                          onClick={() => addDiscovered(item.spec)}
                        >
                          <Plus size={14} aria-hidden />加入排序
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {canEdit && (
            <div className="actions">
              <button
                type="button"
                className="button secondary small"
                disabled={Boolean(busy)}
                onClick={() => void discover()}
              >
                <RefreshCw size={14} aria-hidden />{busy === 'discover' ? '探测中…' : '探测上游 GPU'}
              </button>
              <button
                type="button"
                className="button small"
                disabled={Boolean(busy) || !dirty}
                onClick={() => void save()}
              >
                <Save size={14} aria-hidden />{busy === 'save' ? '保存中…' : '保存顺序'}
              </button>
              <button
                type="button"
                className="button ghost small"
                disabled={Boolean(busy)}
                onClick={() => void reset()}
              >
                <RotateCcw size={14} aria-hidden />{busy === 'reset' ? '恢复中…' : '恢复默认顺序'}
              </button>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
