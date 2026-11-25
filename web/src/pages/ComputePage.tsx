import { useState } from 'react';
import { Cpu, HardDrive, Info, RefreshCw, Search, Square } from 'lucide-react';
import { apiMessage, queryCcuBalance, type ComputeSpec, type GpuSettings, type GpuUsage } from '../lib/gpu-api';
import { CuGuardSettings } from '../components/CuGuardSettings';
import { CuSessionDetails } from '../components/CuSessionDetails';
import { ComputeSpecs } from '../components/ComputeSpecs';
import { InstanceCenter } from '../components/InstanceCenter';
import { RuntimeTelemetryCard } from '../components/RuntimeTelemetryCard';
import type { GpuInstance } from '../components/AccountDirectory';
import type { TransferFocusTarget } from '../lib/domain-transfer';
import { Card } from '../components/ui';
import styles from '../gpu.module.css';
import { cuNumber, formatDateTime, shown } from '../lib/ui-helpers';

function CcuAccountBalanceList({ usage, error }: { usage: GpuUsage | null; error: string }) {
  const colabAccounts = (usage?.accounts || []).filter(snapshot => snapshot.colab_connected !== false);
  const failedCount = colabAccounts.filter(snapshot => Boolean(snapshot.error)).length;
  const totalPositive = colabAccounts.reduce((acc, row) => acc + (row.balance && row.balance > 0 ? row.balance : 0), 0);

  return (
    <div className={styles.cuBalanceBlock}>
      {error && <p className="status error" role="alert">{error}</p>}
      {usage && (
        <>
          <div className={styles.cuBalanceHead}>
            <span className={styles.cuBalanceTitle}>Colab 账号余额</span>
            <div className={styles.cuBalanceBadges}>
              <span className="pill">{colabAccounts.length} 个已授权</span>
              {failedCount > 0 && <span className="pill danger">{failedCount} 个查询失败</span>}
            </div>
          </div>

          {!colabAccounts.length ? (
            <p className="muted small">暂无已授权 Colab 的账号。可在「设置」中完成 Colab 授权后查询余额。</p>
          ) : (
            <div className={styles.cuBalanceList} role="status">
              {colabAccounts.map(snapshot => {
                const tone = snapshot.error ? 'error' : snapshot.below_floor ? 'warn' : snapshot.balance != null ? 'ok' : 'idle';
                const sharePct = colabAccounts.length > 1 && totalPositive > 0 && snapshot.balance != null && snapshot.balance > 0
                  ? Math.max(4, Math.min(100, (snapshot.balance / totalPositive) * 100))
                  : null;
                return (
                  <div key={snapshot.account_id} className={styles.cuBalanceRow}>
                    <div className={styles.cuBalanceIdentity}>
                      <div className={styles.cuBalanceNameRow}>
                        <span className={styles.cuBalanceDot} data-tone={tone} aria-hidden />
                        <strong>{snapshot.alias || snapshot.account_id}</strong>
                        {snapshot.below_floor && <span className="pill danger">触及底线</span>}
                        {snapshot.error && <span className="pill danger">查询失败</span>}
                      </div>
                      <span className={styles.cuBalanceSub}>
                        今日扣费 {cuNumber(snapshot.today_actual_cu)} CU
                        {snapshot.sampled_at != null ? ` · 更新于 ${formatDateTime(snapshot.sampled_at * 1000)}` : ' · 尚未查询'}
                      </span>
                    </div>
                    <div className={styles.cuBalanceRight}>
                      <div className={styles.cuBalanceAmount} data-empty={snapshot.balance == null}>
                        {snapshot.balance == null ? (
                          <span className="muted small">{snapshot.error ? '读取失败' : '未查询'}</span>
                        ) : (
                          <><strong>{cuNumber(snapshot.balance)}</strong><em>CU</em></>
                        )}
                      </div>
                      {sharePct !== null && (
                        <div className={styles.cuBalanceShareTrack} aria-hidden>
                          <span className={styles.cuBalanceShareFill} style={{ width: `${sharePct}%` }} />
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function ComputePage({
  settings, usage, canEditGuard, onRefreshUsage, instances = [], onStop, onCheck, onContinue, onOpenWorkbench, onOpenNative, onRefreshInstances, loadingStop, loadingCheck, specs = [], specsNote = '', error = '', focusTransferTarget = null, onFocusTransferHandled,
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
  onOpenNative?: () => void;
  onRefreshInstances?: () => void;
  loadingStop?: boolean;
  loadingCheck?: boolean;
  specs?: ComputeSpec[];
  specsNote?: string;
  error?: string;
  focusTransferTarget?: TransferFocusTarget | null;
  onFocusTransferHandled?: () => void;
}) {
  const [queryingBalance, setQueryingBalance] = useState(false);
  const [balanceError, setBalanceError] = useState('');
  const realEnabled = Boolean(settings?.real_enabled) && settings?.offline_mode !== true;
  const actual = usage?.today_actual_cu;
  const limit = usage?.guard?.daily_limit_cu;
  const dailyPct = limit != null && actual != null
    ? (limit === 0 ? '已达上限' : `${Math.round((actual / limit) * 100)}%`)
    : '不限';
  const unresolved = (usage?.ccu?.anomalies || []).filter((item) => !item.resolved_at);

  async function handleQueryBalance() {
    setQueryingBalance(true);
    setBalanceError('');
    try {
      await queryCcuBalance();
      await onRefreshUsage();
    } catch (e) {
      setBalanceError(apiMessage(e));
    } finally {
      setQueryingBalance(false);
    }
  }

  return (
    <div className={`${styles.stack} ${styles.workspaceGrid}`}>
      <div className={`${styles.pageHead} ${styles.workspaceWide}`}>
        <h2><Cpu size={20} strokeWidth={1.6} aria-hidden />算力</h2>
        <p className="muted small">任务在后台运行，关闭页面后仍可回来查看进度。</p>
      </div>

      <RuntimeTelemetryCard settings={settings} usage={usage} instances={instances} />

      <Card
        icon={<HardDrive size={17} strokeWidth={1.6} aria-hidden />}
        title="CU 用量"
        aside={
          <button
            type="button"
            className="button secondary small"
            disabled={queryingBalance}
            onClick={() => void handleQueryBalance()}
          >
            <RefreshCw size={14} strokeWidth={1.8} className={queryingBalance ? styles.spin : undefined} aria-hidden />
            {queryingBalance ? '查询中…' : '查询剩余 CU'}
          </button>
        }
      >
        <div className={styles.grid2}>
          <div className={styles.statBox}><small className="muted">今日扣费 · {usage?.local_date || '—'}</small>
            <strong>{cuNumber(actual)}<em> CU</em></strong>
            {limit != null && actual != null && <>
              <small>{cuNumber(actual)} / {cuNumber(limit)} CU</small>
              <progress aria-label="今日 CU 上限使用进度" max={limit || 1} value={limit === 0 ? 1 : Math.min(actual, limit)} />
            </>}
          </div>
          <div className={styles.statBox}>
            <small className="muted">今日额度使用率</small>
            <strong>{dailyPct}</strong>
            <small className="muted">{limit == null ? '未设置今日总上限' : `今日上限 ${cuNumber(limit)} CU`}</small>
          </div>
          <div className={styles.statBox}><small className="muted">历史累计扣费</small><strong>{cuNumber(usage?.history_actual_cu)}<em> CU</em></strong></div>
          <div className={styles.statBox}><small className="muted">当前剩余 CU</small><strong>{cuNumber(usage?.ccu?.total_balance)}<em> CU</em></strong>
            {usage?.ccu?.sampled_at && <small className="muted">{formatDateTime(usage.ccu.sampled_at * 1000)}</small>}
          </div>
        </div>
        <CcuAccountBalanceList usage={usage} error={balanceError} />
        {unresolved.length > 0 && <p className={styles.cuWarning}>
          检测到空转机器 · {unresolved.map(item => `${item.alias || item.account_id}（${shown(item.assignments)} 台）`).join('、')}
        </p>}
      </Card>

      <CuGuardSettings guard={usage?.guard} canEdit={canEditGuard} onSaved={onRefreshUsage} />
      <div className={styles.workspaceWide}><CuSessionDetails usage={usage} /></div>

      <div className={styles.workspaceWide}>
        <InstanceCenter
          instances={instances}
          onContinue={onContinue}
          onOpenWorkbench={onOpenWorkbench}
          onOpenNative={onOpenNative}
          onRefresh={onRefreshInstances}
          focusTransferTarget={focusTransferTarget}
          onFocusTransferHandled={onFocusTransferHandled}
        />
      </div>

      <div className={`${styles.computeFooter} ${styles.workspaceWide}`}>
        <div className={styles.note}><Info size={16} strokeWidth={1.6} aria-hidden /><span>{realEnabled ? '每日扣费来自 Google 账户余额流水，按工作台时区切日。' : '模拟模式的数据仅供演示，不代表真实账单。'}</span></div>
        <div className={styles.computeFooterActions}>
          <button className="button" disabled={loadingCheck} type="button" onClick={onCheck}>
            <Search size={16} strokeWidth={1.8} aria-hidden />{loadingCheck ? '检查中…' : '检查可用 GPU'}
          </button>
          <button className="button danger" disabled={loadingStop} type="button" onClick={onStop}>
            <Square size={14} strokeWidth={1.8} aria-hidden />{loadingStop ? '停止中…' : '停止全部任务'}
          </button>
        </div>
      </div>
      {specsNote && <p className={`muted small ${styles.workspaceWide}`} role="status">{specsNote}</p>}
      {specs.length > 0 && <div className={styles.workspaceWide}><ComputeSpecs specs={specs} /></div>}
      {error && <p className={`status error ${styles.workspaceWide}`} role="alert">{error}</p>}
    </div>
  );
}
