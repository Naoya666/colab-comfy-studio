import { useCallback, useMemo, useState } from 'react';
import { Activity, HardDrive, Layers, MemoryStick, Monitor, RefreshCw } from 'lucide-react';
import {
  getNativeTelemetry,
  type GpuSettings,
  type GpuUsage,
  type NativeTelemetryResponse,
} from '../lib/gpu-api';
import type { GpuInstance } from './AccountDirectory';
import { useSmartPoll } from '../lib/smart-poll';
import { formatDateTime } from '../lib/ui-helpers';
import { Card } from './ui';
import styles from '../gpu.module.css';

const ACTIVE_STATES = new Set(['requested', 'allocating', 'preparing', 'ready', 'busy', 'retained_idle', 'releasing']);

const BUSY_LABELS: Record<string, string> = {
  queue: '任务执行中',
  model_preparation: '准备本地模型中',
  manager: 'Manager 队列处理中',
  plugins: '自定义节点同步中',
  drive: 'Drive 文件同步中',
  unknown: '状态探测中',
};

function gib(bytes?: number | null): string {
  if (bytes == null || !Number.isFinite(bytes)) return '—';
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

function ratioTone(pct: number): 'normal' | 'warn' | 'danger' {
  if (pct >= 95) return 'danger';
  if (pct >= 85) return 'warn';
  return 'normal';
}

function MetricMeter({
  icon,
  label,
  primary,
  percent,
  sub,
  tone = 'normal',
  indeterminate = false,
}: {
  icon: React.ReactNode;
  label: string;
  primary: string;
  percent?: number | null;
  sub?: string;
  tone?: 'normal' | 'warn' | 'danger';
  indeterminate?: boolean;
}) {
  const clamped = percent != null && Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : null;
  return (
    <div className={styles.telemetryTile}>
      <div className={styles.telemetryTileHead}>
        <span className={styles.telemetryLabel}>{icon}{label}</span>
        {clamped !== null && <span className={styles.telemetryPct} data-tone={tone}>{Math.round(clamped)}%</span>}
      </div>
      <strong className={styles.telemetryValue}>{primary}</strong>
      {(clamped !== null || indeterminate) && (
        <div className={styles.meterTrack} data-indeterminate={indeterminate || undefined} aria-hidden="true">
          <span
            className={styles.meterFill}
            data-tone={tone}
            style={indeterminate ? undefined : { width: `${clamped}%` }}
          />
        </div>
      )}
      {sub && <small className="muted">{sub}</small>}
    </div>
  );
}

export function RuntimeTelemetryCard({
  settings,
  usage,
  instances,
}: {
  settings: GpuSettings | null;
  usage: GpuUsage | null;
  instances: GpuInstance[];
}) {
  const realEnabled = Boolean(settings?.real_enabled) && settings?.offline_mode !== true;
  const activeCount = Number(usage?.active_runtimes ?? 0);

  const activeColabInstances = useMemo(
    () => instances.filter((i) => i.provider === 'colab' && !i.runtime_deleted && ACTIVE_STATES.has(String(i.state || '').toLowerCase())),
    [instances],
  );
  const activeNativeInstances = useMemo(
    () => activeColabInstances.filter((i) => i.native_comfy === true || i.instance_pool === 'native'),
    [activeColabInstances],
  );
  const activeNativeSession = useMemo(
    () => usage?.active_sessions?.find((s) => s.session_kind === 'native_comfy'),
    [usage],
  );
  const shouldPollTelemetry = realEnabled && (activeNativeInstances.length > 0 || Boolean(activeNativeSession));

  const [snapshot, setSnapshot] = useState<NativeTelemetryResponse | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const pollTelemetry = useCallback(async (signal?: AbortSignal, force = false) => {
    try {
      const data = await getNativeTelemetry(force, signal);
      if (!signal?.aborted) setSnapshot(data);
    } catch {
      // Keep last valid sample on transient error; next poll will reconcile.
    }
  }, []);

  useSmartPoll(
    useCallback((signal) => pollTelemetry(signal, false), [pollTelemetry]),
    {
      enabled: shouldPollTelemetry,
      active: shouldPollTelemetry,
      activeMs: 5000,
      idleMs: 15000,
    },
  );

  async function manualRefresh() {
    if (refreshing) return;
    setRefreshing(true);
    try {
      await pollTelemetry(undefined, true);
    } finally {
      setRefreshing(false);
    }
  }

  const telemetry = shouldPollTelemetry && snapshot?.ready ? snapshot.telemetry : null;
  const vramPct = telemetry?.vram_total_bytes && telemetry.vram_used_bytes != null
    ? (telemetry.vram_used_bytes / telemetry.vram_total_bytes) * 100
    : null;
  const ramPct = telemetry?.ram_total_bytes && telemetry.ram_used_bytes != null
    ? (telemetry.ram_used_bytes / telemetry.ram_total_bytes) * 100
    : null;

  const vfsLimitBytes = telemetry?.model_storage_mode === 'drive_mount' && telemetry.cache_gb
    ? telemetry.cache_gb * 1024 ** 3
    : null;
  const vfsPct = vfsLimitBytes && telemetry?.vfs_cache_bytes != null
    ? (telemetry.vfs_cache_bytes / vfsLimitBytes) * 100
    : null;

  const queueRunning = telemetry?.queue_running ?? 0;
  const queuePending = telemetry?.queue_pending ?? 0;
  const queueTotal = queueRunning + queuePending;
  const busyText = telemetry?.busy_reason
    ? (BUSY_LABELS[telemetry.busy_reason] || telemetry.busy_reason)
    : (telemetry?.busy ? '处理中' : '空闲待命');

  const otherActiveSession = !telemetry
    ? (usage?.active_sessions?.[0] || null)
    : null;

  return (
    <Card
      className={styles.workspaceWide}
      icon={<Monitor size={17} strokeWidth={1.6} aria-hidden />}
      title="当前环境"
      aside={<span className={`pill ${realEnabled ? 'solid' : ''}`}>{realEnabled ? '真实 GPU' : '模拟模式'}</span>}
    >
      <div className={styles.envBanner}>
        <div className={styles.envIntro}>
          <p className="muted small">
            {realEnabled ? '任务会在 Google Colab 的 GPU 上真实运行。' : '现在是本地模拟，不会连接 Colab，也不会扣 CU。'}
          </p>
        </div>
        <div className={styles.triple}>
          <div><small className="muted">真实 GPU</small><strong>{realEnabled ? '已启用' : '未启用'}</strong></div>
          <div><small className="muted">在线机器</small><strong>{activeCount}</strong></div>
          <div><small className="muted">文件存放</small><strong>{settings?.storage === 'drive' ? 'Google 云盘' : '本站服务器'}</strong></div>
        </div>
      </div>

      {telemetry ? (
        <div className={styles.telemetryPanel} aria-label="原生 ComfyUI 实例实时监控">
          <div className={styles.telemetryBar}>
            <div className={styles.telemetryIdentity}>
              <span className={styles.liveDot} data-busy={Boolean(telemetry.busy)} aria-hidden />
              <strong>{telemetry.gpu_name || activeNativeSession?.gpu_label || '原生 ComfyUI GPU'}</strong>
              <span className="pill">原生 ComfyUI</span>
              <span className="pill" data-busy={Boolean(telemetry.busy)}>{busyText}</span>
            </div>
            <div className={styles.telemetryMeta}>
              {telemetry.sampled_at && (
                <span className="muted small">更新于 {formatDateTime(telemetry.sampled_at * 1000)}</span>
              )}
              <button
                type="button"
                className={styles.iconGhost}
                aria-label="刷新实时监控"
                title="刷新实时监控"
                disabled={refreshing}
                onClick={() => void manualRefresh()}
              >
                <RefreshCw size={14} className={refreshing ? styles.spin : undefined} aria-hidden />
              </button>
            </div>
          </div>

          <div className={styles.telemetryGrid}>
            <MetricMeter
              icon={<Activity size={14} strokeWidth={1.7} aria-hidden />}
              label="显存 VRAM"
              primary={
                telemetry.vram_used_bytes != null && telemetry.vram_total_bytes != null
                  ? `${gib(telemetry.vram_used_bytes)} / ${gib(telemetry.vram_total_bytes)}`
                  : '暂不可读'
              }
              percent={vramPct}
              tone={vramPct != null ? ratioTone(vramPct) : 'normal'}
              sub={
                telemetry.torch_vram_used_bytes != null
                  ? `Torch 已分配 ${gib(telemetry.torch_vram_used_bytes)} · 剩余 ${gib(telemetry.vram_free_bytes)}`
                  : telemetry.vram_free_bytes != null ? `可用 ${gib(telemetry.vram_free_bytes)}` : undefined
              }
            />

            <MetricMeter
              icon={<MemoryStick size={14} strokeWidth={1.7} aria-hidden />}
              label="系统内存 RAM"
              primary={
                telemetry.ram_used_bytes != null && telemetry.ram_total_bytes != null
                  ? `${gib(telemetry.ram_used_bytes)} / ${gib(telemetry.ram_total_bytes)}`
                  : '暂不可读'
              }
              percent={ramPct}
              tone={ramPct != null ? ratioTone(ramPct) : 'normal'}
              sub={telemetry.ram_free_bytes != null ? `剩余 ${gib(telemetry.ram_free_bytes)}` : undefined}
            />

            <MetricMeter
              icon={<Layers size={14} strokeWidth={1.7} aria-hidden />}
              label="执行队列"
              primary={queueTotal > 0 ? `运行 ${queueRunning} · 排队 ${queuePending}` : '队列空闲'}
              percent={queueTotal > 0 ? null : 0}
              indeterminate={queueRunning > 0 || Boolean(telemetry.busy)}
              sub={
                activeNativeSession
                  ? `${busyText} · 已运行 ${activeNativeSession.elapsed_minutes.toFixed(1)} 分钟`
                  : busyText
              }
            />

            <MetricMeter
              icon={<HardDrive size={14} strokeWidth={1.7} aria-hidden />}
              label={telemetry.model_storage_mode === 'drive_mount' ? '本地磁盘与 VFS' : '本地磁盘'}
              primary={telemetry.disk_free_bytes != null ? `剩余 ${gib(telemetry.disk_free_bytes)}` : '暂不可读'}
              percent={vfsPct}
              tone={vfsPct != null ? ratioTone(vfsPct) : 'normal'}
              sub={
                telemetry.model_storage_mode === 'drive_mount' && telemetry.vfs_cache_bytes != null
                  ? `VFS 缓存 ${gib(telemetry.vfs_cache_bytes)} / ${telemetry.cache_gb || 20} GiB`
                  : telemetry.model_storage_mode === 'local'
                    ? '本地模型模式（按需下载至 SSD）'
                    : undefined
              }
            />
          </div>
        </div>
      ) : shouldPollTelemetry && snapshot?.active ? (
        <div className={styles.telemetryStandby} role="status">
          <span className={styles.liveDot} data-busy="true" aria-hidden />
          <span>
            {snapshot.restarting
              ? '原生 ComfyUI 正在重启，恢复后将自动显示显存、内存与队列监控。'
              : '原生 ComfyUI 实例正在初始化，就绪后自动接入实时显存与队列监控。'}
          </span>
        </div>
      ) : otherActiveSession ? (
        <div className={styles.telemetryStandby} role="status">
          <span className={styles.liveDot} data-busy="true" aria-hidden />
          <span>
            当前运行 {otherActiveSession.gpu_label}（已运行 {otherActiveSession.elapsed_minutes.toFixed(1)} 分钟）· 实时显存与队列仪表盘在原生 ComfyUI 实例在线时显示。
          </span>
        </div>
      ) : (
        <div className={styles.telemetryStandby}>
          <span className={styles.idleDot} aria-hidden />
          <span className="muted small">
            {realEnabled
              ? '当前无运行中的原生 ComfyUI 实例。启动原生实例后，此处将实时展示显存、系统内存、队列与磁盘状态。'
              : '模拟模式下不采集远端 GPU 遥测。开启真实 GPU 并启动原生 ComfyUI 后可在此查看实时仪表盘。'}
          </span>
        </div>
      )}
    </Card>
  );
}
