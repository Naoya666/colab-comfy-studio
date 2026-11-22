import { ModelSourceSettings, ModelStorageSettings } from './ModelLibraryPage';
import { useState } from 'react';
import { Settings, ChevronLeft, Monitor, Gauge, Plug, LogOut, ArrowUpRight } from 'lucide-react';
import { ThemePicker } from './ComputePage';
import type { GpuSettings, GpuUsage } from '../lib/gpu-api';
import { GoogleOAuthSetup } from '../components/GoogleOAuthSetup';
import { AccountDirectory, type GpuAccount } from '../components/AccountDirectory';
import { CuGuardSettings } from '../components/CuGuardSettings';
import { GpuOrderSettings } from '../components/GpuOrderSettings';
import { Card } from '../components/ui';
import styles from '../gpu.module.css';
import { cuNumber } from '../lib/ui-helpers';

export function SettingsPanel({
  settings, usage, canEditGuard, onRefreshUsage, error, accounts = [], onAccountsChange, onOpenCompute, onLogout, onBack,
}: {
  settings: GpuSettings | null;
  usage?: GpuUsage | null;
  canEditGuard: boolean;
  onRefreshUsage: () => Promise<void>;
  error: string;
  onOpenCompute?: () => void;
  onLogout: () => void;
  onBack?: () => void;
  accounts?: GpuAccount[];
  onAccountsChange?: (accounts: GpuAccount[]) => void;
}) {
  const [oauthConfigured, setOauthConfigured] = useState(false);

  const activeCount = Number(usage?.active_runtimes ?? 0);
  const dailyCap = usage?.guard?.daily_limit_cu;
  const used = usage?.today_actual_cu;
  const pct = dailyCap != null && used != null
    ? (dailyCap === 0 ? '已达上限' : `${Math.round((used / dailyCap) * 100)}%`)
    : '—';

  return (
    <div className={`${styles.stack} ${styles.workspaceGrid}`}>
      <div className={`${styles.subHead} ${styles.workspaceWide}`}>
        <button type="button" className={styles.backBtn} onClick={onBack}><ChevronLeft size={18} strokeWidth={1.6} aria-hidden />返回</button>
        <h2><Settings size={20} strokeWidth={1.6} aria-hidden />设置</h2>
      </div>

      <Card icon={<Monitor size={17} strokeWidth={1.6} aria-hidden />} title="外观">
        <ThemePicker />
      </Card>

      <Card
        icon={<Gauge size={17} strokeWidth={1.6} aria-hidden />}
        title="用量与额度"
        aside={<span className="pill">{usage?.local_date || '—'}</span>}
      >
        <div className={styles.grid2}>
          <div className={styles.statBox}><small className="muted">运行中机器</small><strong>{activeCount}</strong></div>
          <div className={styles.statBox}><small className="muted">今日总上限</small><strong>{dailyCap == null ? '不限' : <>{cuNumber(dailyCap)}<em> CU</em></>}</strong></div>
          <div className={styles.statBox}><small className="muted">今日扣费</small><strong>{cuNumber(used)}<em> CU</em></strong></div>
          <div className={styles.statBox}><small className="muted">使用率</small><strong>{pct}</strong></div>
        </div>
        {onOpenCompute && (
          <button type="button" className="button secondary small" onClick={onOpenCompute}>
            管理实例、检查 GPU 与急停<ArrowUpRight size={14} aria-hidden />
          </button>
        )}
      </Card>

      <CuGuardSettings guard={usage?.guard} canEdit={canEditGuard} onSaved={onRefreshUsage} />

      <div className={`${styles.stack} ${styles.workspaceWide}`}>
        <GpuOrderSettings canEdit={canEditGuard} />
        <ModelStorageSettings />
        <ModelSourceSettings />
      </div>

      <Card
        icon={<Plug size={17} strokeWidth={1.6} aria-hidden />}
        title="Google 账号"
        aside={settings?.offline_mode ? <span className="pill">离线模式</span> : oauthConfigured ? <span className="pill solid">应用已配置</span> : undefined}
      >
        <AccountDirectory balances={usage?.accounts} accounts={accounts} onAccountsChange={onAccountsChange} />
        {!settings?.offline_mode && (
          <details className="foldable">
            <summary>Google 应用配置（首次使用）</summary>
            <div className="foldContent"><GoogleOAuthSetup onConfigured={() => setOauthConfigured(true)} /></div>
          </details>
        )}
      </Card>

      <div className={styles.workspaceWide}>
        <button className="button secondary big" type="button" onClick={onLogout}><LogOut size={17} strokeWidth={1.6} aria-hidden />退出登录</button>
      </div>
      {error && <p className="status error" role="alert">{error}</p>}
    </div>
  );
}
