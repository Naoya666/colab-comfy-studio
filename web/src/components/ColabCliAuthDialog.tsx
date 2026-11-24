import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Copy, ExternalLink, RefreshCw, ShieldCheck, Upload } from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import type {
  ColabCliAuthCompleteResponse,
  ColabCliAuthStartResponse,
  ColabCliImportRequest,
} from '../lib/v020-types';
import { cuNumber } from '../lib/ui-helpers';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import type { GpuAccount } from './AccountDirectory';
import styles from './v020.module.css';

export interface ColabCliAuthDialogProps {
  account: GpuAccount;
  driveCandidates: GpuAccount[];
  cliAvailable?: boolean;
  onStart: (accountId: string, idempotencyKey: string) => Promise<ColabCliAuthStartResponse>;
  onComplete: (
    accountId: string,
    body: { state: string; code: string },
    idempotencyKey: string,
  ) => Promise<ColabCliAuthCompleteResponse>;
  onImport: (
    accountId: string,
    body: ColabCliImportRequest,
    idempotencyKey: string,
  ) => Promise<ColabCliAuthCompleteResponse>;
  onBindDrive: (accountId: string, driveAccountId: string) => Promise<void>;
  onCompleted: () => Promise<void> | void;
  onClose: () => void;
}

interface StoredCliSession {
  authorization_url: string;
  state: string;
  expires_at: number;
}

let cliKeySeq = 0;
function nextKey(prefix: string): string {
  cliKeySeq += 1;
  return `${prefix}-${Date.now().toString(36)}-${cliKeySeq.toString(36)}`;
}

function storageKey(accountId: string): string {
  return `studio-colab-cli-auth:${accountId}`;
}

function readStoredSession(accountId: string): StoredCliSession | null {
  if (typeof window === 'undefined' || !window.sessionStorage) return null;
  try {
    const raw = window.sessionStorage.getItem(storageKey(accountId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredCliSession>;
    if (
      typeof parsed.authorization_url !== 'string' ||
      typeof parsed.state !== 'string' ||
      typeof parsed.expires_at !== 'number'
    ) {
      return null;
    }
    if (parsed.expires_at * 1000 <= Date.now() + 60_000) {
      window.sessionStorage.removeItem(storageKey(accountId));
      return null;
    }
    return {
      authorization_url: parsed.authorization_url,
      state: parsed.state,
      expires_at: parsed.expires_at,
    };
  } catch {
    return null;
  }
}

function writeStoredSession(accountId: string, value: StoredCliSession | null): void {
  if (typeof window === 'undefined' || !window.sessionStorage) return;
  try {
    if (!value) {
      window.sessionStorage.removeItem(storageKey(accountId));
    } else {
      window.sessionStorage.setItem(storageKey(accountId), JSON.stringify(value));
    }
  } catch {
    // Ignore storage quota or private-mode failures.
  }
}

function remainingMinutes(expiresAt: number | null): number {
  if (!expiresAt) return 0;
  return Math.max(0, Math.ceil((expiresAt * 1000 - Date.now()) / 60_000));
}

export function ColabCliAuthDialog({
  account,
  driveCandidates,
  cliAvailable = true,
  onStart,
  onComplete,
  onImport,
  onBindDrive,
  onCompleted,
  onClose,
}: ColabCliAuthDialogProps) {
  const [session, setSession] = useState<StoredCliSession | null>(() => readStoredSession(account.account_id));
  const [code, setCode] = useState('');
  const [importText, setImportText] = useState('');
  const [busy, setBusy] = useState<'start' | 'complete' | 'import' | 'bind' | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [result, setResult] = useState<ColabCliAuthCompleteResponse | null>(null);
  const [selectedDriveId, setSelectedDriveId] = useState(() => driveCandidates[0]?.account_id || '');
  const [driveBoundAlias, setDriveBoundAlias] = useState('');
  const completeKeyRef = useRef<string>('');
  const importKeyRef = useRef<string>('');
  const initiatedRef = useRef(false);

  useBodyScrollLock(true);

  const hasOwnOrDelegatedDrive = useMemo(() => {
    if (account.drive_account_id || account.authorization?.drive?.delegated_to) return true;
    return Boolean(account.authorization?.drive?.connected || account.drive?.status === 'connected');
  }, [account]);

  const generateLink = useCallback(async (forceNew = false) => {
    if (!forceNew) {
      const existing = readStoredSession(account.account_id);
      if (existing) {
        setSession(existing);
        return;
      }
    }
    setBusy('start');
    setError('');
    setNotice('');
    setCopied(false);
    try {
      const started = await onStart(account.account_id, nextKey(`colab-cli-start-${account.account_id}`));
      const saved: StoredCliSession = {
        authorization_url: started.authorization_url,
        state: started.state,
        expires_at: started.expires_at,
      };
      writeStoredSession(account.account_id, saved);
      setSession(saved);
      setCode('');
      completeKeyRef.current = '';
      if (forceNew) setNotice('已生成新的 Colab CLI 授权链接，请将新链接发给授权方或在新窗口打开。');
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(null);
    }
  }, [account.account_id, onStart]);

  useEffect(() => {
    if (initiatedRef.current || !cliAvailable) return;
    initiatedRef.current = true;
    if (!session) {
      void generateLink(false);
    }
  }, [cliAvailable, generateLink, session]);

  async function copyUrl() {
    if (!session?.authorization_url) return;
    try {
      await navigator.clipboard.writeText(session.authorization_url);
      setCopied(true);
      setNotice('已复制授权链接，可直接发给卖家或在新浏览器窗口打开。');
    } catch {
      setNotice('请点击链接输入框全选后手动复制。');
    }
  }

  async function submitCode() {
    const trimmed = code.trim();
    if (!session?.state || !trimmed || busy) return;
    setBusy('complete');
    setError('');
    setNotice('');
    const key = (completeKeyRef.current ||= nextKey(`colab-cli-complete-${account.account_id}`));
    try {
      const completed = await onComplete(account.account_id, { state: session.state, code: trimmed }, key);
      completeKeyRef.current = '';
      writeStoredSession(account.account_id, null);
      setResult(completed);
      await onCompleted();
    } catch (cause) {
      completeKeyRef.current = '';
      setError(apiMessage(cause));
    } finally {
      setBusy(null);
    }
  }

  async function submitImport() {
    const raw = importText.trim();
    if (!raw || busy) return;
    setBusy('import');
    setError('');
    setNotice('');
    const key = (importKeyRef.current ||= nextKey(`colab-cli-import-${account.account_id}`));
    try {
      let body: ColabCliImportRequest;
      if (raw.startsWith('{')) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(raw);
        } catch {
          throw new Error('粘贴内容不是合法的 JSON 格式，请检查后重试。');
        }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          throw new Error('Colab CLI 凭据 JSON 必须是对象。');
        }
        body = { credentials_json: parsed as Record<string, unknown> };
      } else {
        body = { refresh_token: raw };
      }
      const imported = await onImport(account.account_id, body, key);
      importKeyRef.current = '';
      writeStoredSession(account.account_id, null);
      setResult(imported);
      await onCompleted();
    } catch (cause) {
      importKeyRef.current = '';
      setError(apiMessage(cause));
    } finally {
      setBusy(null);
    }
  }

  async function bindSelectedDrive() {
    if (!selectedDriveId || busy) return;
    setBusy('bind');
    setError('');
    try {
      await onBindDrive(account.account_id, selectedDriveId);
      const picked = driveCandidates.find((item) => item.account_id === selectedDriveId);
      setDriveBoundAlias(picked?.alias || selectedDriveId.slice(0, 8));
      await onCompleted();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(null);
    }
  }

  const minsLeft = remainingMinutes(session?.expires_at ?? null);

  return (
    <div className={styles.sheetBackdrop} role="presentation" onClick={() => !busy && onClose()}>
      <section
        className={`${styles.sheet} ${styles.cliWizardSheet}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="colab-cli-auth-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className={styles.cliWizardHeader}>
          <div>
            <p className={styles.eyebrow}>Colab CLI 算力授权 · 推荐</p>
            <h3 id="colab-cli-auth-title">连接「{account.alias || '未命名账号'}」的 Colab 算力</h3>
          </div>
          <span className={`${styles.channelTag} ${styles.channelTagRecommended}`}>推荐</span>
        </div>

        {result ? (
          <div className={styles.stack}>
            <div className={styles.cliSuccessBox} role="status">
              <div className={styles.cliSuccessTitle}>
                <ShieldCheck size={18} strokeWidth={1.8} aria-hidden />
                <strong>Colab CLI 算力已授权并加密保存</strong>
              </div>
              <p className={styles.muted}>
                通道：<strong>Colab CLI 隧道</strong>
                {result.balance != null ? ` · 当前余额 ${cuNumber(result.balance)} CU` : ''}
                {result.eligible_gpus?.length ? ` · 可用 GPU：${result.eligible_gpus.join(' / ')}` : ''}
              </p>
              {result.probe_error && <p className={styles.muted}>{result.probe_error}</p>}
            </div>

            {!hasOwnOrDelegatedDrive && !driveBoundAlias && (
              <div className={styles.cliDriveGuide}>
                <strong>下一步：关联 Google Drive 存储（模型库与持久化必需）</strong>
                {driveCandidates.length > 0 ? (
                  <>
                    <p className={styles.muted}>
                      Colab CLI 凭据仅授权算力（最小权限，不访问对方网盘）。可直接复用你已授权 Drive 的账号：
                    </p>
                    <div className={styles.inlineEdit}>
                      <select
                        aria-label="选择要复用的 Drive 账号"
                        value={selectedDriveId}
                        disabled={Boolean(busy)}
                        onChange={(event) => setSelectedDriveId(event.target.value)}
                      >
                        {driveCandidates.map((item) => (
                          <option key={item.account_id} value={item.account_id}>
                            复用：{item.alias || item.account_id.slice(0, 8)}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        className={styles.button}
                        disabled={!selectedDriveId || Boolean(busy)}
                        onClick={() => void bindSelectedDrive()}
                      >
                        {busy === 'bind' ? '绑定中…' : '立即绑定 Drive'}
                      </button>
                    </div>
                  </>
                ) : (
                  <p className={styles.muted}>
                    当前尚无已授权 Google Drive 的账号。关闭弹窗后，可在下方「Google Web OAuth 应用配置」中启用 Google Drive API 并为一个主账号授权 Drive，再将此算力号的「Drive 存储来源」指向该主账号。
                  </p>
                )}
              </div>
            )}

            {driveBoundAlias && (
              <p className={styles.status} role="status">
                已将 Drive 存储来源绑定至「{driveBoundAlias}」，现在可以直接前往算力页检查可用 GPU 或开始创作。
              </p>
            )}

            {error && <p className={styles.error} role="alert">{error}</p>}

            <div className={styles.sheetActions}>
              <button type="button" className={styles.button} onClick={onClose} disabled={Boolean(busy)}>
                完成
              </button>
            </div>
          </div>
        ) : (
          <div className={styles.stack}>
            <p className={styles.muted}>
              无需安装 <code>google-colab-cli</code>、无需配置 keyring 或环境变量；仅请求 <strong>Colab 最小必要权限</strong>（不申请网盘或 GCP 权限），不会触发 Scope 变更报错，自用或异地交付授权码均可一步完成。
            </p>

            {cliAvailable ? (
              <>
                <div className={styles.cliStepCard}>
                  <div className={styles.cliStepHead}>
                    <span className={styles.cliStepNum}>1</span>
                    <strong>获取并打开授权链接</strong>
                    {minsLeft > 0 && <span className={styles.muted}>约 {minsLeft} 分钟内有效</span>}
                  </div>
                  <p className={styles.muted}>
                    <strong>纯自用</strong>：直接点「新窗口打开」登录自己的 Google 账号；<strong>第三方交付 CU</strong>：点「复制链接」发给卖家，等待对方返回 <code>4/0A...</code> 授权码（切页或刷新不丢失会话）。
                  </p>
                  <input
                    aria-label="Colab CLI 授权链接"
                    readOnly
                    value={busy === 'start' ? '正在生成授权链接…' : session?.authorization_url || ''}
                    onFocus={(event) => event.target.select()}
                  />
                  <div className={styles.cliActionRow}>
                    <button
                      type="button"
                      className={styles.button}
                      disabled={!session?.authorization_url || Boolean(busy)}
                      onClick={() => void copyUrl()}
                    >
                      {copied ? <Check size={15} strokeWidth={2} aria-hidden /> : <Copy size={15} strokeWidth={1.8} aria-hidden />}
                      {copied ? '已复制链接' : '复制授权链接'}
                    </button>
                    {session?.authorization_url && (
                      <a
                        className={styles.buttonSecondary}
                        href={session.authorization_url}
                        target="_blank"
                        rel="noreferrer noopener"
                      >
                        <ExternalLink size={15} strokeWidth={1.8} aria-hidden />
                        新窗口打开
                      </a>
                    )}
                    <button
                      type="button"
                      className={styles.buttonSecondary}
                      disabled={Boolean(busy)}
                      onClick={() => void generateLink(true)}
                    >
                      <RefreshCw size={14} strokeWidth={1.8} aria-hidden />
                      重新生成
                    </button>
                  </div>
                </div>

                <div className={styles.cliStepCard}>
                  <div className={styles.cliStepHead}>
                    <span className={styles.cliStepNum}>2</span>
                    <strong>粘贴 Google 返回的授权码</strong>
                  </div>
                  <input
                    aria-label="Colab CLI 授权码"
                    autoComplete="off"
                    autoCapitalize="none"
                    spellCheck={false}
                    placeholder="粘贴 4/0A... 授权码（输错可直接修正重试，无需重发链接）"
                    value={code}
                    disabled={Boolean(busy)}
                    onChange={(event) => setCode(event.target.value)}
                  />
                  <div className={styles.cliActionRow}>
                    <button
                      type="button"
                      className={styles.button}
                      disabled={!session?.state || !code.trim() || Boolean(busy)}
                      onClick={() => void submitCode()}
                    >
                      {busy === 'complete' ? '正在验证并保存…' : '提交授权码并连接'}
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <p className={styles.warning}>
                当前运行环境未加载 <code>google-colab-cli</code> 客户端配置（需 Python 3.12+ 或使用官方 Docker 镜像），无法直接生成授权链接。你仍可在下方展开导入已有的 CLI 凭据 JSON。
              </p>
            )}

            <details className={styles.cliImportFold}>
              <summary>已有现成 Colab CLI 凭据 JSON 或 Refresh Token？直接导入</summary>
              <div className={styles.stack}>
                <p className={styles.muted}>
                  支持直接粘贴 <code>google-colab-cli</code> 凭据 JSON（含 <code>refresh_token</code>）或单行 Refresh Token；导入时会实时校验有效性并用 <code>ENCRYPTION_KEY</code> 加密存库。
                </p>
                <div className={styles.cliActionRow}>
                  <label className={styles.buttonSecondary}>
                    <Upload size={14} strokeWidth={1.8} aria-hidden />
                    选择 JSON 文件
                    <input
                      hidden
                      type="file"
                      accept="application/json,.json"
                      onChange={async (event) => {
                        const file = event.target.files?.[0];
                        event.target.value = '';
                        if (!file) return;
                        if (file.size > 65536) {
                          setError('凭据 JSON 文件过大');
                          return;
                        }
                        setImportText(await file.text());
                        setError('');
                        setNotice('已读取凭据文件，点击「校验并导入凭据」完成保存。');
                      }}
                    />
                  </label>
                </div>
                <textarea
                  aria-label="Colab CLI 凭据 JSON 或 Refresh Token"
                  rows={4}
                  spellCheck={false}
                  placeholder='{"refresh_token": "1//...", "client_id": "764086051850-...", "client_secret": "..."} 或直接粘贴 refresh_token'
                  value={importText}
                  disabled={Boolean(busy)}
                  onChange={(event) => setImportText(event.target.value)}
                />
                <div className={styles.cliActionRow}>
                  <button
                    type="button"
                    className={styles.buttonSecondary}
                    disabled={!importText.trim() || Boolean(busy)}
                    onClick={() => void submitImport()}
                  >
                    {busy === 'import' ? '校验导入中…' : '校验并导入凭据'}
                  </button>
                </div>
              </div>
            </details>

            {notice && <p className={styles.status} role="status">{notice}</p>}
            {error && <p className={styles.error} role="alert">{error}</p>}

            <div className={styles.sheetActions}>
              <button type="button" className={styles.buttonSecondary} onClick={onClose} disabled={Boolean(busy)}>
                关闭
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
