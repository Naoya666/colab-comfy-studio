import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { RefreshCw, Trash2 } from 'lucide-react';
import {
  apiMessage,
  completeColabCliAuth,
  createGoogleAccount,
  deleteGoogleAccount,
  disconnectGoogleAccount,
  getGoogleAccounts,
  importColabCliCredentials,
  startColabCliAuth,
  startGoogleAccountAuth,
  updateGoogleAccount,
  type CuAccount,
} from '../lib/gpu-api';
import type {
  ColabCliAuthCompleteResponse,
  ColabCliAuthStartResponse,
  ColabCliImportRequest,
  GoogleAuthChannel,
} from '../lib/v020-types';
import { cuNumber } from '../lib/ui-helpers';
import { ColabCliAuthDialog } from './ColabCliAuthDialog';
import styles from './v020.module.css';

export type AccountStatus = 'connected' | 'needs_reauth' | 'disabled' | 'error' | string;
export type AuthorizationStatus = 'connected' | 'needs_reauth' | 'disabled' | 'error' | 'not_connected' | string;

export interface GpuAccount {
  account_id: string;
  alias: string;
  status: AccountStatus;
  colab?: { status?: AuthorizationStatus; scopes?: string[]; checked_at?: string; expires_at?: string; auth_channel?: GoogleAuthChannel | null };
  drive?: { status?: AuthorizationStatus; scopes?: string[]; checked_at?: string; expires_at?: string };
  colab_status?: AuthorizationStatus;
  drive_status?: AuthorizationStatus;
  drive_account_id?: string | null;
  drive_account_alias?: string | null;
  drive_dependents?: string[];
  authorization?: {
    colab?: {
      connected?: boolean;
      configured?: boolean;
      expires_at?: string | number | null;
      message?: string;
      auth_channel?: GoogleAuthChannel | null;
      cli_available?: boolean;
    };
    drive?: {
      connected?: boolean;
      configured?: boolean;
      expires_at?: string | number | null;
      message?: string;
      auth_channel?: GoogleAuthChannel | null;
      delegated_to?: string;
      delegated_alias?: string;
    };
  };
  allow_allocation?: boolean;
  enabled?: boolean;
  budget?: number | Record<string, unknown> | null;
  budget_limit?: number | null;
  strategy?: string;
  instance_count?: number;
  active_instance_count?: number;
  active_instances?: number;
  active_tasks?: number;
  last_checked_at?: string;
  last_error?: string;
  email_masked?: string;
  [key: string]: unknown;
}

export interface GpuInstance {
  /** Backend presents the durable resource id as `id`; instance_id is legacy input compatibility. */
  id?: string;
  instance_id?: string;
  instance_label?: string;
  label?: string;
  account_id?: string | null;
  account_alias_snapshot?: string;
  provider?: 'mock' | 'colab' | string;
  instance_pool?: 'native' | 'creation' | 'ephemeral' | string;
  state?: string;
  lifecycle?: string;
  retention_mode?: string;
  retain_idle_minutes?: number | null;
  gpu_reported?: string;
  runtime_spec_reported?: string | Record<string, unknown>;
  creation_runtime?: Record<string, unknown> | null;
  current_job_id?: string | null;
  current_job_title?: string;
  created_at?: string;
  last_seen_at?: string;
  retained_until?: string | null;
  runtime_deleted?: boolean;
  remote_delete_source?: string | null;
  idle_deadline_at?: string | null;
  hard_deadline_at?: string | null;
  estimated_units?: number | null;
  actual_units?: number | null;
  unit_rate?: number | null;
  verified?: boolean;
  runtime_verified?: boolean;
  last_error_code?: string;
  last_error?: string;
  [key: string]: unknown;
}

export interface GpuInstanceJob {
  id: string;
  title?: string;
  state?: string;
  progress?: number;
  created_at?: string;
  updated_at?: string;
  account_alias_snapshot?: string;
  [key: string]: unknown;
}

export interface GpuInstanceEvent {
  id?: string;
  event_type?: string;
  state?: string;
  message?: string;
  created_at?: string;
  [key: string]: unknown;
}

export interface AccountDirectoryApi {
  getGpuAccounts: () => Promise<GpuAccount[]>;
  createGpuAccount: (body: { alias: string }, idempotencyKey: string) => Promise<GpuAccount>;
  updateGpuAccount: (accountId: string, patch: Record<string, unknown>, idempotencyKey: string) => Promise<GpuAccount>;
  disconnectGpuAccount: (
    accountId: string,
    idempotencyKey: string,
    confirmActiveStop: boolean,
    revokeRemote?: boolean,
  ) => Promise<unknown>;
  deleteGpuAccount: (
    accountId: string,
    idempotencyKey: string,
    revokeRemote?: boolean,
  ) => Promise<{ account_id?: string; deleted?: boolean; cleared_defaults?: string[] }>;
  startGpuAccountAuth: (accountId: string, purpose: 'colab' | 'drive', idempotencyKey: string) => Promise<{ authorization_url?: string } | string>;
  startColabCliAuth: (accountId: string, idempotencyKey: string) => Promise<ColabCliAuthStartResponse>;
  completeColabCliAuth: (
    accountId: string,
    body: { state: string; code: string },
    idempotencyKey: string,
  ) => Promise<ColabCliAuthCompleteResponse>;
  importColabCliCredentials: (
    accountId: string,
    body: ColabCliImportRequest,
    idempotencyKey: string,
  ) => Promise<ColabCliAuthCompleteResponse>;
}

export interface AccountDirectoryProps {
  /** Optional externally loaded list. When supplied it is kept in sync with the view. */
  accounts?: GpuAccount[];
  balances?: CuAccount[];
  api?: Partial<AccountDirectoryApi>;
  onAccountsChange?: (accounts: GpuAccount[]) => void;
  /** Hide inner section title when embedded inside a parent Card that already provides a heading. */
  hideTitle?: boolean;
}

let idSequence = 0;
function newIdempotencyKey(prefix: string): string {
  idSequence += 1;
  return `${prefix}-${Date.now().toString(36)}-${idSequence.toString(36)}`;
}

function listPayload(value: unknown): GpuAccount[] {
  if (Array.isArray(value)) return value as GpuAccount[];
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    for (const key of ['accounts', 'items', 'data']) if (Array.isArray(record[key])) return record[key] as GpuAccount[];
  }
  return [];
}

function accountPayload(value: unknown): GpuAccount {
  if (value && typeof value === 'object' && 'account_id' in value) return value as GpuAccount;
  if (value && typeof value === 'object' && 'data' in value) return accountPayload((value as Record<string, unknown>).data);
  return value as GpuAccount;
}

const defaultAccountApi: AccountDirectoryApi = {
  getGpuAccounts: async () => listPayload(await getGoogleAccounts()),
  createGpuAccount: async (body, key) => accountPayload(await createGoogleAccount(body, key)),
  updateGpuAccount: async (accountId, patch, key) => accountPayload(await updateGoogleAccount(accountId, patch as Parameters<typeof updateGoogleAccount>[1], key)),
  disconnectGpuAccount: (accountId, key, _confirmActiveStop, revokeRemote) =>
    disconnectGoogleAccount(accountId, key, typeof revokeRemote === 'boolean' ? { revoke_remote: revokeRemote } : undefined),
  deleteGpuAccount: (accountId, key, revokeRemote) =>
    deleteGoogleAccount(accountId, key, typeof revokeRemote === 'boolean' ? { revoke_remote: revokeRemote } : undefined),
  startGpuAccountAuth: (accountId, purpose, key) => startGoogleAccountAuth(accountId, purpose, key),
  startColabCliAuth: (accountId, key) => startColabCliAuth(accountId, key),
  completeColabCliAuth: (accountId, body, key) => completeColabCliAuth(accountId, body, key),
  importColabCliCredentials: (accountId, body, key) => importColabCliCredentials(accountId, body, key),
};

function authStatus(account: GpuAccount, purpose: 'colab' | 'drive'): AuthorizationStatus {
  const authorization = account.authorization?.[purpose];
  if (authorization) return authorization.connected ? 'connected' : authorization.message ? 'needs_reauth' : 'not_connected';
  return account[purpose]?.status || account[`${purpose}_status`] || 'not_connected';
}

function colabChannel(account: GpuAccount): GoogleAuthChannel | null {
  return account.authorization?.colab?.auth_channel || account.colab?.auth_channel || null;
}

function channelLabel(channel: GoogleAuthChannel | null): string | null {
  if (channel === 'colab_cli') return 'Colab CLI';
  if (channel === 'web_oauth') return 'Web OAuth';
  return null;
}

function statusLabel(status: string): string {
  return ({ connected: '已连接', needs_reauth: '需重新授权', disabled: '已停用', error: '错误', not_connected: '未连接' } as Record<string, string>)[status] || status;
}

function activeCount(account: GpuAccount): number {
  return Number(account.active_instance_count ?? account.active_instances ?? 0);
}

function checkedAt(account: GpuAccount): string {
  const value = account.last_checked_at || account.colab?.checked_at || account.drive?.checked_at;
  if (!value) return '尚未检查';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function safeAuthRedirect(value: { authorization_url?: string } | string): string | null {
  const candidate = typeof value === 'string' ? value : value.authorization_url;
  if (!candidate || typeof window === 'undefined') return null;
  try {
    const url = new URL(candidate, window.location.origin);
    if (url.protocol !== 'https:' && url.origin !== window.location.origin) return null;
    return url.href;
  } catch {
    return null;
  }
}

export function AccountDirectory({ accounts: suppliedAccounts, balances = [], api, onAccountsChange, hideTitle = false }: AccountDirectoryProps) {
  const accountApi = useMemo(() => ({ ...defaultAccountApi, ...api }), [api]);
  const [accounts, setAccounts] = useState<GpuAccount[]>(suppliedAccounts || []);
  const [alias, setAlias] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editAlias, setEditAlias] = useState('');
  const [cliAuthAccount, setCliAuthAccount] = useState<GpuAccount | null>(null);
  const [disconnecting, setDisconnecting] = useState<GpuAccount | null>(null);
  const [disconnectRevokeRemote, setDisconnectRevokeRemote] = useState(true);
  const [deleting, setDeleting] = useState<GpuAccount | null>(null);
  const [deleteRevokeRemote, setDeleteRevokeRemote] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const keyRef = useRef<Record<string, string>>({});
  const loadRef = useRef(false);

  const replaceAccounts = useCallback((next: GpuAccount[]) => {
    setAccounts(next);
    onAccountsChange?.(next);
  }, [onAccountsChange]);

  const refresh = useCallback(async () => {
    if (loadRef.current) return;
    loadRef.current = true;
    setError('');
    try {
      replaceAccounts(await accountApi.getGpuAccounts());
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      loadRef.current = false;
    }
  }, [accountApi, replaceAccounts]);

  useEffect(() => {
    if (suppliedAccounts) setAccounts(suppliedAccounts);
  }, [suppliedAccounts]);
  useEffect(() => { void refresh(); }, [refresh]);

  function keyFor(operation: string): string {
    return keyRef.current[operation] ||= newIdempotencyKey(operation);
  }
  function finishKey(operation: string) { delete keyRef.current[operation]; }

  async function createAccount(event: FormEvent) {
    event.preventDefault();
    const nextAlias = alias.trim();
    if (!nextAlias || busy) return;
    const operation = 'account-create';
    setBusy(operation); setError(''); setStatus('正在创建账号目录…');
    try {
      const created = await accountApi.createGpuAccount({ alias: nextAlias }, keyFor(operation));
      replaceAccounts([...accounts, created]);
      setAlias(''); finishKey(operation); setStatus(`已创建「${created.alias || nextAlias}」，推荐点击「Colab CLI 授权（推荐）」连接算力，并为至少一个账号授权 Drive。`);
    } catch (cause) {
      setError(apiMessage(cause)); setStatus('创建未确认；再次重试会复用同一幂等键。');
    } finally { setBusy(null); }
  }

  async function updateAccount(account: GpuAccount, patch: Record<string, unknown>, operation: string, success: string) {
    if (busy) return;
    setBusy(operation); setError(''); setStatus('正在保存账号设置…');
    try {
      const updated = await accountApi.updateGpuAccount(account.account_id, patch, keyFor(operation));
      replaceAccounts(accounts.map((item) => item.account_id === account.account_id ? updated : item));
      finishKey(operation); setStatus(success);
    } catch (cause) {
      setError(apiMessage(cause)); setStatus('保存未确认；再次重试会复用同一幂等键。');
    } finally { setBusy(null); }
  }

  async function authorize(account: GpuAccount, purpose: 'colab' | 'drive') {
    const operation = `account-auth-${account.account_id}-${purpose}`;
    if (busy) return;
    setBusy(operation); setError(''); setStatus(`正在准备 ${purpose === 'colab' ? 'Colab (Web OAuth 实验性)' : 'Drive'} 授权…`);
    try {
      const response = await accountApi.startGpuAccountAuth(account.account_id, purpose, keyFor(operation));
      const target = safeAuthRedirect(response);
      if (!target) throw new Error('授权地址无效或未返回；没有打开外部地址。');
      finishKey(operation);
      window.location.assign(target);
    } catch (cause) {
      setError(apiMessage(cause)); setStatus('授权尚未开始；再次重试会复用同一幂等键。'); setBusy(null);
    }
  }

  function driveDependentsOf(account: GpuAccount): string[] {
    if (Array.isArray(account.drive_dependents) && account.drive_dependents.length > 0) {
      return account.drive_dependents;
    }
    return accounts
      .filter((item) => item.account_id !== account.account_id && item.drive_account_id === account.account_id)
      .map((item) => item.alias || item.account_id.slice(0, 8));
  }

  function hasOwnAuthorization(account: GpuAccount): boolean {
    const colab = authStatus(account, 'colab');
    const drive = authStatus(account, 'drive');
    const hasDelegatedDrive = Boolean(account.drive_account_id || account.authorization?.drive?.delegated_to);
    const hasOwnColab = colab === 'connected' || colab === 'needs_reauth';
    const hasOwnDrive = !hasDelegatedDrive && (drive === 'connected' || drive === 'needs_reauth');
    return hasOwnColab || hasOwnDrive;
  }

  function openDisconnect(account: GpuAccount) {
    setDisconnectRevokeRemote(colabChannel(account) !== 'colab_cli');
    setDisconnecting(account);
  }

  function openDelete(account: GpuAccount) {
    setDeleteRevokeRemote(colabChannel(account) !== 'colab_cli');
    setDeleting(account);
  }

  async function disconnect() {
    const account = disconnecting;
    if (!account || busy) return;
    const operation = `account-disconnect-${account.account_id}-${disconnectRevokeRemote ? 'revoke' : 'local'}`;
    setBusy(operation); setError(''); setStatus('正在请求解除授权…');
    try {
      await accountApi.disconnectGpuAccount(
        account.account_id,
        keyFor(operation),
        activeCount(account) > 0,
        disconnectRevokeRemote,
      );
      finishKey(operation); setDisconnecting(null);
      await refresh();
      setStatus(
        disconnectRevokeRemote
          ? `已解除并吊销「${account.alias || '该账号'}」的 Google 授权；如需移除卡片可点击删除。`
          : `已仅从本站移除「${account.alias || '该账号'}」的本地授权凭据（未吊销 Google 远端令牌）。`,
      );
    } catch (cause) {
      setError(apiMessage(cause)); setStatus('解除授权未确认；再次重试会复用同一幂等键。');
    } finally { setBusy(null); }
  }

  async function removeAccount() {
    const account = deleting;
    if (!account || busy) return;
    const operation = `account-delete-${account.account_id}-${deleteRevokeRemote ? 'revoke' : 'local'}`;
    setBusy(operation); setError(''); setStatus('正在删除账号…');
    try {
      const result = await accountApi.deleteGpuAccount(
        account.account_id,
        keyFor(operation),
        deleteRevokeRemote,
      );
      replaceAccounts(accounts.filter((item) => item.account_id !== account.account_id));
      finishKey(operation); setDeleting(null);
      const cleared = result?.cleared_defaults?.length
        ? `（已自动清空 ${result.cleared_defaults.join('、')} 的默认账号）`
        : '';
      const revokeNote = hasOwnAuthorization(account) && !deleteRevokeRemote ? '（仅本地移除，未吊销 Google 远端令牌）' : '';
      setStatus(`已删除账号「${account.alias || '未命名账号'}」${cleared}${revokeNote}，历史任务与账单不受影响。`);
    } catch (cause) {
      setError(apiMessage(cause)); setStatus('删除未确认；再次重试会复用同一幂等键。');
    } finally { setBusy(null); }
  }

  return (
    <section className={styles.directory} aria-label="Google 账号目录">
      {hideTitle ? (
        <div className={styles.toolbar}>
          <p className={styles.muted}>Colab 算力推荐使用 <strong>Colab CLI 授权</strong>（免开 GCP Colaboratory API，支持自用与异地交付授权码）；Google Drive 需通过下方自建 Web OAuth 应用授权，多算力号可复用同一个 Drive 账号。</p>
          <button type="button" className={styles.iconButton} onClick={() => void refresh()} disabled={Boolean(busy)} aria-label="刷新账号" title="刷新账号"><RefreshCw size={16} aria-hidden /></button>
        </div>
      ) : (
        <>
          <header className={styles.sectionHeader}>
            <div><p className={styles.eyebrow}>设置 · 单一账号入口</p><h2>账号目录</h2></div>
            <button type="button" className={styles.iconButton} onClick={() => void refresh()} disabled={Boolean(busy)} aria-label="刷新账号"><RefreshCw size={16} aria-hidden /></button>
          </header>
          <p className={styles.muted}>Colab 算力推荐使用 <strong>Colab CLI 授权</strong>（免开 GCP Colaboratory API，支持自用与异地交付授权码）；Google Drive 需通过下方自建 Web OAuth 应用授权，多算力号可复用同一个 Drive 账号。</p>
        </>
      )}
      {status && <p className={styles.status} role="status">{status}</p>}
      {error && <p className={styles.error} role="alert">{error}</p>}

      <form className={styles.createRow} onSubmit={createAccount}>
        <label className={styles.srOnly} htmlFor="new-gpu-account-alias">新账号别名</label>
        <input id="new-gpu-account-alias" value={alias} onChange={(event) => setAlias(event.target.value)} placeholder="例如：主账号 或 Colab算力号" maxLength={80} />
        <button type="submit" className={styles.button} disabled={!alias.trim() || Boolean(busy)}>新增账号</button>
      </form>

      {!accounts.length ? (
        <div className={styles.empty}><strong>还没有账号目录</strong><span>先新增别名，再用「Colab CLI 授权（推荐）」连接算力，并为 Drive 账号授权。</span></div>
      ) : <div className={styles.accountGrid}>
        {accounts.map((account) => {
          const balance = balances.find(row => row.account_id === account.account_id);
          const active = activeCount(account);
          const colab = authStatus(account, 'colab');
          const drive = authStatus(account, 'drive');
          const channel = colabChannel(account);
          const channelText = channelLabel(channel);
          const delegatedAlias = account.drive_account_alias || account.authorization?.drive?.delegated_alias;
          const isBusy = busy?.includes(account.account_id) || false;
          return (
            <article className={styles.accountCard} key={account.account_id}>
              <div className={styles.cardTop}>
                <div className={styles.minWidth}>
                  {editingId === account.account_id ? (
                    <div className={styles.inlineEdit}><input value={editAlias} onChange={(event) => setEditAlias(event.target.value)} autoFocus aria-label="账号别名" /><button type="button" className={styles.button} disabled={!editAlias.trim() || Boolean(busy)} onClick={() => { void updateAccount(account, { alias: editAlias.trim() }, `account-rename-${account.account_id}`, '账号别名已更新。'); setEditingId(null); }}>保存</button><button type="button" className={styles.buttonSecondary} onClick={() => setEditingId(null)}>取消</button></div>
                  ) : <><h3>{account.alias || '未命名账号'}</h3><p className={styles.muted}>{account.email_masked || '身份邮箱已隐藏'} · 最近检查：{checkedAt(account)}</p></>}
                </div>
                <span className={`${styles.badge} ${account.status === 'connected' ? styles.good : account.status === 'error' ? styles.bad : ''}`}>{statusLabel(account.status)}</span>
              </div>
              {account.last_error && <p className={styles.warning}>{account.last_error}</p>}
              <div className={styles.authGrid}>
                <div className={styles.authCell}>
                  <div className={styles.authCellHead}>
                    <span>Colab 算力</span>
                    {channelText && <span className={styles.channelTag}>{channelText}</span>}
                  </div>
                  <b>{statusLabel(colab)}</b>
                  <div className={styles.authButtons}>
                    <button
                      type="button"
                      className={`${colab === 'connected' && channel === 'colab_cli' ? styles.buttonSecondary : styles.button} ${styles.authButtonRow}`}
                      disabled={Boolean(busy)}
                      onClick={() => setCliAuthAccount(account)}
                    >
                      <span>{colab === 'connected' && channel === 'colab_cli' ? '重新 CLI 授权' : 'Colab CLI 授权'}</span>
                      <span className={`${styles.channelTag} ${styles.channelTagRecommended}`}>推荐</span>
                    </button>
                    <button
                      type="button"
                      className={`${styles.buttonSecondary} ${styles.authButtonRow}`}
                      disabled={Boolean(busy)}
                      title="使用下方自建 Web OAuth 应用授权 Colab（需 GCP 开通 Colaboratory API，长驻保活与实时速率待更多真机验证）"
                      onClick={() => void authorize(account, 'colab')}
                    >
                      <span>{colab === 'connected' && channel === 'web_oauth' ? '重新 Web OAuth' : 'Web OAuth 授权'}</span>
                      <span className={`${styles.channelTag} ${styles.channelTagExperimental}`}>实验性</span>
                    </button>
                  </div>
                </div>
                <div className={styles.authCell}>
                  <div className={styles.authCellHead}>
                    <span>Google Drive</span>
                    <span className={styles.channelTag}>Web OAuth</span>
                  </div>
                  <b>{delegatedAlias ? `复用「${delegatedAlias}」(${statusLabel(drive)})` : statusLabel(drive)}</b>
                  <div className={styles.authButtons}>
                    <button
                      type="button"
                      className={`${styles.buttonSecondary} ${styles.authButtonRow}`}
                      disabled={Boolean(busy)}
                      onClick={() => void authorize(account, 'drive')}
                    >
                      <span>{drive === 'connected' && !delegatedAlias ? '重新授权 Drive' : '授权独立 Drive'}</span>
                    </button>
                  </div>
                </div>
              </div>
              <div className={styles.accountControls}>
                <div className={styles.togglePair}>
                  <label className={styles.toggleRow}><input type="checkbox" checked={account.enabled !== false} disabled={Boolean(busy)} onChange={(event) => void updateAccount(account, { enabled: event.target.checked }, `account-enabled-${account.account_id}`, event.target.checked ? '账号已启用。' : '账号已停用，不会分配新任务。')} /><span>允许使用</span></label>
                  <label className={styles.toggleRow}><input type="checkbox" checked={account.allow_allocation === true} disabled={Boolean(busy) || account.enabled === false} onChange={(event) => void updateAccount(account, { allow_allocation: event.target.checked }, `account-allocation-${account.account_id}`, event.target.checked ? '已允许新任务分配。' : '已禁止新任务分配。')} /><span>允许分配</span></label>
                </div>
                {accounts.length > 1 && (
                  <label className={styles.budgetField}>
                    <span>Drive 存储来源</span>
                    <select
                      aria-label="Drive 存储来源"
                      value={account.drive_account_id || ''}
                      disabled={Boolean(busy)}
                      onChange={(event) => {
                        const target = event.target.value || null;
                        void updateAccount(
                          account,
                          { drive_account_id: target },
                          `account-drive-source-${account.account_id}`,
                          target ? '已切换为复用指定账号的 Drive 存储。' : '已切换为使用本账号独立 Drive。',
                        );
                      }}
                    >
                      <option value="">本账号独立 Drive</option>
                      {accounts
                        .filter((item) => item.account_id !== account.account_id && !item.drive_account_id && authStatus(item, 'drive') === 'connected')
                        .map((item) => (
                          <option key={item.account_id} value={item.account_id}>
                            复用：{item.alias || item.account_id.slice(0, 8)}
                          </option>
                        ))}
                    </select>
                  </label>
                )}
                {colab === 'connected' ? (
                  <div className="small">
                    余额 {cuNumber(balance?.balance)} CU · 今日扣费 {cuNumber(balance?.today_actual_cu)} CU
                    {balance?.below_floor && <span className={styles.warningInline}> · 已触及余额底线</span>}
                    {balance?.error && <span className="muted"> · 最近查询失败</span>}
                    {balance?.balance != null && !balance.below_floor && !balance.error && <span className="muted"> · 底线正常</span>}
                  </div>
                ) : (
                  <div className={styles.muted}>
                    {drive === 'connected'
                      ? '未授权 Colab（仅用于 Google Drive 存储，不参与 CU 余额查询）'
                      : '尚未授权 Colab，不参与算力分配与 CU 余额查询'}
                  </div>
                )}
              </div>
              <div className={styles.cardActions}>
                <button type="button" className={styles.buttonSecondary} disabled={Boolean(busy)} onClick={() => { setEditingId(account.account_id); setEditAlias(account.alias || ''); }}>改名</button>
                {hasOwnAuthorization(account) && (
                  <button type="button" className={styles.buttonSecondary} disabled={Boolean(busy)} onClick={() => openDisconnect(account)}>解除授权</button>
                )}
                <button
                  type="button"
                  className={styles.buttonDanger}
                  disabled={Boolean(busy)}
                  onClick={() => openDelete(account)}
                >
                  <Trash2 size={14} strokeWidth={1.8} aria-hidden />删除账号
                </button>
                {active > 0 && <span className={styles.warningInline}>活动资源 {active} 个，停止释放后方可解除或删除</span>}
                {isBusy && <span className={styles.muted}>处理中…</span>}
              </div>
            </article>
          );
        })}
      </div>}

      {cliAuthAccount && (
        <ColabCliAuthDialog
          account={accounts.find((item) => item.account_id === cliAuthAccount.account_id) || cliAuthAccount}
          driveCandidates={accounts.filter(
            (item) =>
              item.account_id !== cliAuthAccount.account_id &&
              !item.drive_account_id &&
              authStatus(item, 'drive') === 'connected',
          )}
          cliAvailable={cliAuthAccount.authorization?.colab?.cli_available !== false}
          onStart={accountApi.startColabCliAuth}
          onComplete={accountApi.completeColabCliAuth}
          onImport={accountApi.importColabCliCredentials}
          onBindDrive={async (targetAccountId, driveAccountId) => {
            const target = accounts.find((item) => item.account_id === targetAccountId) || cliAuthAccount;
            await updateAccount(
              target,
              { drive_account_id: driveAccountId },
              `account-drive-source-${targetAccountId}`,
              '已绑定 Drive 存储来源。',
            );
          }}
          onCompleted={refresh}
          onClose={() => setCliAuthAccount(null)}
        />
      )}

      {disconnecting && (
        <div className={styles.sheetBackdrop} role="presentation" onClick={() => !busy && setDisconnecting(null)}>
          <section className={styles.sheet} role="dialog" aria-modal="true" aria-labelledby="disconnect-title" onClick={(event) => event.stopPropagation()}>
            <h3 id="disconnect-title">确认解除「{disconnecting.alias}」的授权？</h3>
            {activeCount(disconnecting) > 0 ? (
              <p className={styles.warning}>检测到 {activeCount(disconnecting)} 个活动实例/任务。解除前必须由服务端执行安全停机检查；若仍无法安全停机，操作会失败，历史记录不会丢失。</p>
            ) : (
              <p className={styles.muted}>解除后账号卡片会保留为「未连接」状态，不会删除历史任务与作品。</p>
            )}
            <div className={styles.revokeChoice}>
              <label className={styles.toggleRow}>
                <input
                  type="checkbox"
                  checked={disconnectRevokeRemote}
                  disabled={Boolean(busy)}
                  onChange={(event) => setDisconnectRevokeRemote(event.target.checked)}
                />
                <span>同时向 Google 远端吊销令牌</span>
              </label>
              <p className={styles.muted}>
                {colabChannel(disconnecting) === 'colab_cli'
                  ? '检测到此账号使用 Colab CLI 凭据：默认不勾选（仅从本站移除），避免误将第三方交付或本机复用的 CLI 令牌在 Google 侧作废。'
                  : '勾选后将请求 Google 吊销本站持有的刷新令牌；不勾选则仅清除本站加密凭据。'}
              </p>
            </div>
            <div className={styles.sheetActions}>
              <button type="button" className={styles.buttonSecondary} onClick={() => setDisconnecting(null)} disabled={Boolean(busy)}>取消</button>
              <button type="button" className={styles.buttonDanger} onClick={() => void disconnect()} disabled={Boolean(busy)}>{activeCount(disconnecting) ? '确认安全检查后解除' : '确认解除授权'}</button>
            </div>
          </section>
        </div>
      )}

      {deleting && (() => {
        const deps = driveDependentsOf(deleting);
        const active = activeCount(deleting);
        const blocked = deps.length > 0 || active > 0;
        const ownAuth = hasOwnAuthorization(deleting);
        return (
          <div className={styles.sheetBackdrop} role="presentation" onClick={() => !busy && setDeleting(null)}>
            <section className={styles.sheet} role="dialog" aria-modal="true" aria-labelledby="delete-account-title" onClick={(event) => event.stopPropagation()}>
              <h3 id="delete-account-title">确认删除账号「{deleting.alias}」？</h3>
              {deps.length > 0 && (
                <p className={styles.warning}>
                  无法删除：以下账号正在复用此账号的 Google Drive 存储 —— <strong>{deps.join('、')}</strong>。请先将它们的「Drive 存储来源」切换为其他账号或本账号独立 Drive 后再删除。
                </p>
              )}
              {active > 0 && (
                <p className={styles.warning}>
                  无法删除：检测到 {active} 个活动实例/任务，请先在「算力」页停止并确认释放后再删除。
                </p>
              )}
              {!blocked && (
                <>
                  <p className={styles.muted}>
                    {ownAuth
                      ? '将清除此账号在本站的 Google 授权并从账号目录中移除。历史任务、作品和 CU 账单不会丢失。'
                      : '将从账号目录中移除此账号，历史任务、作品和 CU 账单不会丢失。'}
                  </p>
                  {ownAuth && (
                    <div className={styles.revokeChoice}>
                      <label className={styles.toggleRow}>
                        <input
                          type="checkbox"
                          checked={deleteRevokeRemote}
                          disabled={Boolean(busy)}
                          onChange={(event) => setDeleteRevokeRemote(event.target.checked)}
                        />
                        <span>同时向 Google 远端吊销令牌</span>
                      </label>
                      <p className={styles.muted}>
                        {colabChannel(deleting) === 'colab_cli'
                          ? '检测到 Colab CLI 凭据：默认不勾选（仅从本站移除），不影响卖家或本机 colab-cli 继续使用。'
                          : '勾选后会在删除前向 Google 请求吊销刷新令牌。'}
                      </p>
                    </div>
                  )}
                </>
              )}
              <div className={styles.sheetActions}>
                <button type="button" className={styles.buttonSecondary} onClick={() => setDeleting(null)} disabled={Boolean(busy)}>取消</button>
                <button type="button" className={styles.buttonDanger} onClick={() => void removeAccount()} disabled={Boolean(busy) || blocked}>
                  确认删除账号
                </button>
              </div>
            </section>
          </div>
        );
      })()}
    </section>
  );
}

export { defaultAccountApi };
