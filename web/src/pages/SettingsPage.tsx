import { ModelSourceSettings, ModelStorageSettings } from '../components/ModelLibrarySettings';
import { useEffect, useState } from 'react';
import { Settings, ChevronLeft, Monitor, Moon, Sun, Plug, LogOut } from 'lucide-react';
import type { GpuSettings, GpuSpecKey, GpuUsage } from '../lib/gpu-api';
import { GoogleOAuthSetup } from '../components/GoogleOAuthSetup';
import { AccountDirectory, type GpuAccount } from '../components/AccountDirectory';
import { GpuOrderSettings } from '../components/GpuOrderSettings';
import { AllocationRetrySettings } from '../components/AllocationRetrySettings';
import { Card, Segmented } from '../components/ui';
import { applyTheme, readStoredTheme, type ThemeMode } from '../lib/theme';
import styles from '../gpu.module.css';

export function ThemePicker() {
  const [theme, setTheme] = useState<ThemeMode>(() => readStoredTheme());
  function change(value: ThemeMode) {
    setTheme(value);
    applyTheme(value);
  }
  return (
    <Segmented<ThemeMode>
      ariaLabel="界面颜色"
      value={theme}
      onChange={change}
      options={[
        { value: 'dark', label: '深色', icon: <Moon size={15} strokeWidth={1.6} aria-hidden /> },
        { value: 'light', label: '浅色', icon: <Sun size={15} strokeWidth={1.6} aria-hidden /> },
        { value: 'system', label: '跟随系统', icon: <Monitor size={15} strokeWidth={1.6} aria-hidden /> },
      ]}
    />
  );
}

export function SettingsPanel({
  settings, usage, canEditGuard, desktop = false, focusSection = null, onFocusHandled, error, accounts = [], accountsLoaded = false, onAccountsChange, onGpuOrderChange, onLogout, onBack,
}: {
  settings: GpuSettings | null;
  usage?: GpuUsage | null;
  canEditGuard: boolean;
  desktop?: boolean;
  focusSection?: 'model-storage' | null;
  onFocusHandled?: () => void;
  error: string;
  onLogout: () => void;
  onBack?: () => void;
  accounts?: GpuAccount[];
  accountsLoaded?: boolean;
  onAccountsChange?: (accounts: GpuAccount[]) => void;
  onGpuOrderChange?: (orderedSpecs: GpuSpecKey[]) => void;
}) {
  const [oauthConfigured, setOauthConfigured] = useState(false);
  const [oauthOpen, setOauthOpen] = useState(false);
  const [orderedSpecs, setOrderedSpecs] = useState<GpuSpecKey[] | undefined>(undefined);

  useEffect(() => {
    if (focusSection !== 'model-storage') return;
    const frame = window.requestAnimationFrame(() => {
      document.getElementById('settings-model-storage')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      onFocusHandled?.();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [focusSection, onFocusHandled]);

  return (
    <div className={`${styles.stack} ${styles.workspaceGrid}`}>
      {desktop ? (
        <div className={`${styles.pageTitle} ${styles.workspaceWide}`}>
          <h2><Settings size={20} strokeWidth={1.6} aria-hidden />设置</h2>
          <div className={styles.titleActions}>
            <button className="button secondary small" type="button" onClick={onLogout}>
              <LogOut size={15} strokeWidth={1.6} aria-hidden />退出登录
            </button>
          </div>
        </div>
      ) : (
        <div className={`${styles.subHead} ${styles.workspaceWide}`}>
          <button type="button" className={styles.backBtn} onClick={onBack}><ChevronLeft size={18} strokeWidth={1.6} aria-hidden />返回</button>
          <h2><Settings size={20} strokeWidth={1.6} aria-hidden />设置</h2>
        </div>
      )}

      <section className={`${styles.card} ${styles.workspaceWide} ${styles.appearanceBanner}`} aria-label="外观设置">
        <div className={styles.appearanceIntro}>
          <Monitor size={18} strokeWidth={1.6} aria-hidden />
          <div>
            <h2>外观</h2>
            <p className="muted small">选择工作台界面主题色，修改后立即生效。</p>
          </div>
        </div>
        <div className={styles.appearanceControl}>
          <ThemePicker />
        </div>
      </section>

      <div className={styles.workspaceWide}>
        <GpuOrderSettings
          canEdit={canEditGuard}
          onSaved={(specs) => {
            setOrderedSpecs(specs);
            onGpuOrderChange?.(specs);
          }}
        />
      </div>

      <div className={styles.workspaceWide}>
        <AllocationRetrySettings canEdit={canEditGuard} orderedSpecs={orderedSpecs} />
      </div>

      <ModelStorageSettings id="settings-model-storage" defaultOpen={focusSection === 'model-storage'} accounts={accountsLoaded ? accounts : undefined} />
      <ModelSourceSettings id="settings-model-source" defaultOpen={focusSection === 'model-storage'} />

      <Card
        className={styles.workspaceWide}
        icon={<Plug size={17} strokeWidth={1.6} aria-hidden />}
        title="Google 账号"
        aside={settings?.offline_mode ? <span className="pill">离线模式</span> : oauthConfigured ? <span className="pill solid">应用已配置</span> : undefined}
      >
        <AccountDirectory hideTitle balances={usage?.accounts} accounts={accounts} accountsLoaded={accountsLoaded} onAccountsChange={onAccountsChange} />
        {!settings?.offline_mode && (
          <details className="foldable" open={oauthOpen} onToggle={(e) => setOauthOpen(e.currentTarget.open)}>
            <summary>Google Web OAuth 应用配置（Drive 必填 / Colab 实验性）</summary>
            <div className="foldContent">{oauthOpen && <GoogleOAuthSetup accounts={accountsLoaded ? accounts : undefined} onConfigured={() => setOauthConfigured(true)} />}</div>
          </details>
        )}
      </Card>

      {!desktop && (
        <div className={styles.workspaceWide}>
          <button className="button secondary big" type="button" onClick={onLogout}><LogOut size={17} strokeWidth={1.6} aria-hidden />退出登录</button>
        </div>
      )}
      {error && <p className={`status error ${styles.workspaceWide}`} role="alert">{error}</p>}
    </div>
  );
}
