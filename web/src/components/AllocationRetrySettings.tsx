import { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCw, Save } from 'lucide-react';
import {
  apiMessage,
  getAllocationRetrySettings,
  getGpuSpecOrder,
  saveAllocationRetrySettings,
  type AllocationRetrySettings as AllocationRetryConfig,
  type GpuSpecKey,
} from '../lib/gpu-api';
import { DEFAULT_GPU_SPEC_ORDER, formatGpuSpecName, gpuSpecId } from '../lib/gpu-spec-order';
import { Card } from './ui';
import styles from '../gpu.module.css';

const MIN_INTERVAL = 10;
const MAX_INTERVAL = 120;
const DEFAULT_INTERVAL = 15;
const MIN_RETRIES = 1;
const MAX_RETRIES = 60;

export function AllocationRetrySettings({
  canEdit,
  orderedSpecs,
}: {
  canEdit: boolean;
  orderedSpecs?: GpuSpecKey[];
}) {
  const [saved, setSaved] = useState<AllocationRetryConfig | null>(null);
  const [fallbackOrder, setFallbackOrder] = useState<GpuSpecKey[]>(DEFAULT_GPU_SPEC_ORDER);
  const [enabled, setEnabled] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [intervalText, setIntervalText] = useState(String(DEFAULT_INTERVAL));
  const [maxRetriesText, setMaxRetriesText] = useState('20');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const apply = useCallback((cfg: AllocationRetryConfig) => {
    setSaved(cfg);
    setEnabled(Boolean(cfg.enabled));
    setSelectedIds((cfg.gpu_pool || []).map((item) => gpuSpecId(item)).filter((id): id is string => Boolean(id)));
    setIntervalText(String(cfg.retry_interval_seconds ?? DEFAULT_INTERVAL));
    setMaxRetriesText(String(cfg.max_retries ?? 20));
  }, []);

  useEffect(() => {
    let alive = true;
    setError('');
    Promise.all([
      getAllocationRetrySettings(),
      orderedSpecs && orderedSpecs.length ? Promise.resolve(null) : getGpuSpecOrder().catch(() => null),
    ]).then(([cfg, orderRes]) => {
      if (!alive) return;
      apply(cfg);
      if (orderRes?.ordered_specs?.length) {
        setFallbackOrder(orderRes.ordered_specs);
      }
    }).catch((cause) => {
      if (alive) setError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [apply, orderedSpecs]);

  const activeOrder = useMemo(
    () => (orderedSpecs && orderedSpecs.length ? orderedSpecs : fallbackOrder),
    [orderedSpecs, fallbackOrder],
  );

  // Merge GPU order list with any previously saved custom spec so checkboxes never silently drop items.
  const availableSpecs = useMemo(() => {
    const result: GpuSpecKey[] = [];
    const seen = new Set<string>();
    for (const spec of activeOrder) {
      const id = gpuSpecId(spec);
      if (!id || seen.has(id)) continue;
      seen.add(id);
      result.push(spec);
    }
    for (const spec of saved?.gpu_pool || []) {
      const id = gpuSpecId(spec);
      if (!id || seen.has(id)) continue;
      seen.add(id);
      result.push(spec);
    }
    return result;
  }, [activeOrder, saved?.gpu_pool]);

  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const draftPool = useMemo(
    () => availableSpecs.filter((spec) => {
      const id = gpuSpecId(spec);
      return Boolean(id && selectedSet.has(id));
    }),
    [availableSpecs, selectedSet],
  );

  const parsedInterval = Number(intervalText);
  const parsedMaxRetries = Number(maxRetriesText);
  const validInterval = Number.isInteger(parsedInterval) && parsedInterval >= MIN_INTERVAL && parsedInterval <= MAX_INTERVAL;
  const validMaxRetries = Number.isInteger(parsedMaxRetries) && parsedMaxRetries >= MIN_RETRIES && parsedMaxRetries <= MAX_RETRIES;

  const dirty = useMemo(() => {
    if (!saved) return false;
    const savedIds = (saved.gpu_pool || []).map((item) => gpuSpecId(item)).filter(Boolean);
    const currentIds = draftPool.map((item) => gpuSpecId(item)).filter(Boolean);
    return (
      enabled !== Boolean(saved.enabled)
      || parsedInterval !== saved.retry_interval_seconds
      || parsedMaxRetries !== saved.max_retries
      || JSON.stringify(currentIds) !== JSON.stringify(savedIds)
    );
  }, [saved, enabled, parsedInterval, parsedMaxRetries, draftPool]);

  function toggleSpec(spec: GpuSpecKey) {
    const id = gpuSpecId(spec);
    if (!id) return;
    setSelectedIds((curr) => (curr.includes(id) ? curr.filter((x) => x !== id) : [...curr, id]));
    setStatus('');
  }

  async function save() {
    if (!canEdit || busy || !validInterval || !validMaxRetries) return;
    setBusy(true);
    setError('');
    setStatus('');
    try {
      const next = await saveAllocationRetrySettings({
        enabled,
        gpu_pool: draftPool,
        retry_interval_seconds: parsedInterval,
        max_retries: parsedMaxRetries,
      });
      apply(next);
      setStatus(next.enabled ? '自动排队抢卡策略已保存并生效。' : '自动排队抢卡已关闭并保存。');
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      icon={<RefreshCw size={17} strokeWidth={1.6} aria-hidden />}
      title="自动排队抢卡"
      aside={saved ? <span className={`pill ${saved.enabled ? 'solid' : ''}`}>{saved.enabled ? '已启用' : '已关闭'}</span> : undefined}
    >
      <p className="muted small">
        遇到 Colab 503、分配超时或无可用卡时，自动等待旧租约安全释放后继续排队重试（同时作用于创作模式、原生 ComfyUI 及 API 自动开机）；遇到 429 限流立即停止以保护账号。
      </p>
      {status && <p className="status" role="status">{status}</p>}
      {error && <p className="status error" role="alert">{error}</p>}

      {!saved ? (
        !error && <p className="muted small">正在读取自动排队抢卡策略…</p>
      ) : (
        <div className={styles.stack}>
          <label className={styles.retryToggleRow}>
            <input
              type="checkbox"
              checked={enabled}
              disabled={!canEdit || busy}
              onChange={(e) => {
                setEnabled(e.target.checked);
                setStatus('');
              }}
            />
            <span>
              <strong>启用自动排队抢卡</strong>（遇 503 / 无卡自动重试，可随时取消任务）
            </span>
          </label>

          <fieldset className={styles.retryPoolFieldset} disabled={!canEdit || busy}>
            <div className={styles.retryPoolHead}>
              <legend>参与重试的显卡池（复选，未勾选时仅死磕本次所选 GPU）</legend>
              {canEdit && (
                <div className={styles.retryPoolQuickActions}>
                  <button
                    type="button"
                    className="button ghost small"
                    disabled={busy || selectedIds.length === availableSpecs.length}
                    onClick={() => {
                      setSelectedIds(availableSpecs.map((s) => gpuSpecId(s)).filter((id): id is string => Boolean(id)));
                      setStatus('');
                    }}
                  >
                    全选
                  </button>
                  <button
                    type="button"
                    className="button ghost small"
                    disabled={busy || selectedIds.length === 0}
                    onClick={() => {
                      setSelectedIds([]);
                      setStatus('');
                    }}
                  >
                    清空
                  </button>
                </div>
              )}
            </div>
            <div className={styles.retryPoolGrid}>
              {availableSpecs.map((spec) => {
                const id = gpuSpecId(spec) || `${spec.accelerator}:${spec.shape}`;
                const checked = selectedSet.has(id);
                return (
                  <label key={id} className={styles.retryPoolChip} data-checked={checked ? 'true' : undefined}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleSpec(spec)}
                    />
                    <span>{formatGpuSpecName(spec)}</span>
                  </label>
                );
              })}
            </div>
            <p className="muted small">
              重试时始终优先尝试本次所选 GPU，失败后再按「GPU 规格顺序表」从低到高轮询上方勾选的显卡。创作模式会自动过滤低于工作流默认最低规格的显卡；选择「自动选择账号」时会先轮换多账号、再切换显卡。
            </p>
          </fieldset>

          <div className={styles.retryFieldsRow}>
            <label className="field">
              最小重试间隔（{MIN_INTERVAL} ~ {MAX_INTERVAL} 秒，含 0~5s 随机抖动）
              <input
                type="number"
                min={MIN_INTERVAL}
                max={MAX_INTERVAL}
                step={1}
                disabled={!canEdit || busy}
                value={intervalText}
                onChange={(e) => {
                  setIntervalText(e.target.value);
                  setStatus('');
                }}
              />
              {!validInterval && (
                <span className="status error">请输入 {MIN_INTERVAL} ~ {MAX_INTERVAL} 之间的整数秒</span>
              )}
            </label>
            <label className="field">
              最大重试次数（{MIN_RETRIES} ~ {MAX_RETRIES} 次）
              <input
                type="number"
                min={MIN_RETRIES}
                max={MAX_RETRIES}
                step={1}
                disabled={!canEdit || busy}
                value={maxRetriesText}
                onChange={(e) => {
                  setMaxRetriesText(e.target.value);
                  setStatus('');
                }}
              />
              {!validMaxRetries && (
                <span className="status error">请输入 {MIN_RETRIES} ~ {MAX_RETRIES} 之间的整数</span>
              )}
            </label>
          </div>
          <p className="muted small">
            重试间隔从上次发起申请起算，已消耗的请求与 60s 旧租约核验时间会自动抵扣，不重复叠加；遇到明确无卡秒拒时仍会补足冷却以防触发 429。
          </p>

          {canEdit && (
            <div className="actions">
              <button
                type="button"
                className="button small"
                disabled={busy || !dirty || !validInterval || !validMaxRetries}
                onClick={() => void save()}
              >
                <Save size={14} aria-hidden />
                {busy ? '保存中…' : '保存策略'}
              </button>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
