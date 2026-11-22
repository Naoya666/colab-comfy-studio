import { useState } from 'react';
import { Cpu, Monitor, HardDrive, Info, Square, Search } from 'lucide-react';
import { apiMessage, queryCcuBalance, type ComputeSpec, type GpuSettings, type GpuUsage } from '../lib/gpu-api';
import { CuGuardSettings } from '../components/CuGuardSettings';
import { CuSessionDetails } from '../components/CuSessionDetails';
import { ComputeSpecs } from '../components/ComputeSpecs';
import { InstanceCenter } from '../components/InstanceCenter';
import type { GpuInstance } from '../components/AccountDirectory';
import { Card } from '../components/ui';
import styles from '../gpu.module.css';
import { cuNumber, formatDateTime, shown } from '../lib/ui-helpers';

function CcuBalanceQuery({ usage, refresh }: { usage: GpuUsage | null; refresh: () => Promise<void> }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const accounts = usage?.accounts || [];
  const partial = accounts.some(snapshot => snapshot.error || snapshot.balance == null);

  async function query() {
    setLoading(true);
    setError('');
    try {
      await queryCcuBalance();
      await refresh();
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setLoading(false);
    }
  }

  return <div className={styles.inlinePanel}>
    <div className="actions">
      <button type="button" className="button secondary small" disabled={loading} onClick={() => void query()}>
        <Search size={15} strokeWidth={1.8} aria-hidden />{loading ? '查询中…' : '查询剩余 CU'}
      </button>
    </div>
    {error && <p className="status error" role="alert">{error}</p>}
    {usage && <div className={styles.stack} role="status">
      {!accounts.length && <p className="muted small">暂无启用的 Colab 账号。可在设置中添加并授权。</p>}
      {usage.ccu?.total_balance != null && <p>
        {partial ? '已查到的余额合计' : 'Colab 剩余 CU 合计'} <strong>{cuNumber(usage.ccu?.total_balance)} CU</strong>
      </p>}
      {accounts.map(snapshot => <p key={snapshot.account_id} className="small">
        <strong>{snapshot.alias || snapshot.account_id}</strong> · {snapshot.balance == null ? '暂时无法读取余额' : `${cuNumber(snapshot.balance)} CU`}
        {snapshot.error && <span className="muted"> · 最近查询失败</span>}
        {snapshot.sampled_at != null && <span className="muted"> · {formatDateTime(snapshot.sampled_at * 1000)}</span>}
      </p>)}
    </div>}
  </div>;
}

export function ComputePage({
  settings, usage, canEditGuard, onRefreshUsage, instances = [], onStop, onCheck, onContinue, onOpenWorkbench, onRefreshInstances, loadingStop, loadingCheck, specs = [], specsNote = '', error = '',
}: {
  settings: GpuSettings | null;
  usage: GpuUsage | null;
  canEditGuard: boolean;
  onRefreshUsage: () => Promise<void>;
  instances?: GpuInstance[];
  onStop?: () => void;
  onCheck?: () => void;
  onContinue?: (instance: GpuInstance) => void;
  onOpenWorkbench?: (instance: GpuInstance) => void;
  onRefreshInstances?: () => void;
  loadingStop?: boolean;
  loadingCheck?: boolean;
  specs?: ComputeSpec[];
  specsNote?: string;
  error?: string;
}) {
  const realEnabled = Boolean(settings?.real_enabled) && settings?.offline_mode !== true;
  const activeCount = Number(usage?.active_runtimes ?? 0);
  const actual = usage?.today_actual_cu;
  const limit = usage?.guard?.daily_limit_cu;
  const unresolved = (usage?.ccu?.anomalies || []).filter((item) => !item.resolved_at);

  return (
    <div className={`${styles.stack} ${styles.workspaceGrid}`}>
      <div className={`${styles.pageHead} ${styles.workspaceWide}`}>
        <h2><Cpu size={20} strokeWidth={1.6} aria-hidden />算力</h2>
        <p className="muted small">任务在后台运行，关闭页面后仍可回来查看进度。</p>
      </div>

      <Card
        className={styles.workspaceWide}
        icon={<Monitor size={17} strokeWidth={1.6} aria-hidden />}
        title="当前环境"
        aside={<span className={`pill ${realEnabled ? 'solid' : ''}`}>{realEnabled ? '真实 GPU' : '模拟模式'}</span>}
      >
        <div className={styles.envBanner}>
          <p className="muted small">{realEnabled ? '任务会在 Google Colab 的 GPU 上真实运行。' : '现在是本地模拟，不会连接 Colab，也不会扣 CU。'}</p>
          <div className={styles.triple}>
            <div><small className="muted">真实 GPU</small><strong>{realEnabled ? '已启用' : '未启用'}</strong></div>
            <div><small className="muted">在线机器</small><strong>{activeCount}</strong></div>
            <div><small className="muted">文件存放</small><strong>{settings?.storage === 'drive' ? 'Google 云盘' : '本站服务器'}</strong></div>
          </div>
        </div>
      </Card>

      <Card icon={<HardDrive size={17} strokeWidth={1.6} aria-hidden />} title="CU 用量">
        <div className={styles.grid2}>
          <div className={styles.statBox}><small className="muted">今日扣费 · {usage?.local_date || '—'}</small>
            <strong>{cuNumber(actual)}<em> CU</em></strong>
            {limit != null && actual != null && <>
              <small>{cuNumber(actual)} / {cuNumber(limit)} CU</small>
              <progress aria-label="今日 CU 上限使用进度" max={limit || 1} value={limit === 0 ? 1 : Math.min(actual, limit)} />
            </>}
          </div>
          <div className={styles.statBox}><small className="muted">历史累计扣费</small><strong>{cuNumber(usage?.history_actual_cu)}<em> CU</em></strong></div>
          <div className={styles.statBox}><small className="muted">当前剩余 CU</small><strong>{cuNumber(usage?.ccu?.total_balance)}<em> CU</em></strong>
            {usage?.ccu?.sampled_at && <small className="muted">{formatDateTime(usage.ccu.sampled_at * 1000)}</small>}
          </div>
        </div>
        <CcuBalanceQuery usage={usage} refresh={onRefreshUsage} />
        {unresolved.length > 0 && <p className={styles.cuWarning}>
          检测到空转机器 · {unresolved.map(item => `${item.alias || item.account_id}（${shown(item.assignments)} 台）`).join('、')}
        </p>}
      </Card>

      <CuGuardSettings guard={usage?.guard} canEdit={canEditGuard} onSaved={onRefreshUsage} />
      <div className={styles.workspaceWide}><CuSessionDetails usage={usage} /></div>

      <div className={styles.workspaceWide}>
        <InstanceCenter instances={instances} onContinue={onContinue} onOpenWorkbench={onOpenWorkbench} onRefresh={onRefreshInstances} />
      </div>

      <div className={styles.note}><Info size={16} strokeWidth={1.6} aria-hidden /><span>{realEnabled ? '每日扣费来自 Google 账户余额流水，按工作台时区切日。' : '模拟模式的数据仅供演示，不代表真实账单。'}</span></div>

      <div className={styles.twoButtons}>
        <button className="button" style={{ flex: 1 }} disabled={loadingCheck} type="button" onClick={onCheck}>
          <Search size={16} strokeWidth={1.8} aria-hidden />{loadingCheck ? '检查中…' : '检查可用 GPU'}
        </button>
        <button className="button danger" style={{ flex: 1 }} disabled={loadingStop} type="button" onClick={onStop}>
          <Square size={14} strokeWidth={1.8} aria-hidden />{loadingStop ? '停止中…' : '停止全部任务'}
        </button>
      </div>
      {specsNote && <p className="muted small" role="status">{specsNote}</p>}
      {specs.length > 0 && <div className={styles.workspaceWide}><ComputeSpecs specs={specs} /></div>}
      {error && <p className="status error" role="alert">{error}</p>}
    </div>
  );
}

export function ThemePicker() {
  const [theme, setTheme] = useState(() => localStorage.getItem('studio-theme') || 'dark');
  function change(value: string) {
    setTheme(value);
    localStorage.setItem('studio-theme', value);
    document.documentElement.dataset.theme = value;
  }
  return (
    <div className="field">
      <label htmlFor="theme-select">界面颜色</label>
      <select id="theme-select" value={theme} onChange={(e) => change(e.target.value)}>
        <option value="dark">深色（黑底）</option>
        <option value="light">浅色（白底）</option>
        <option value="system">跟随手机系统</option>
      </select>
    </div>
  );
}
