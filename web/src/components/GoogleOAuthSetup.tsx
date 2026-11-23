import { useEffect, useRef, useState } from 'react';
import {
  apiMessage,
  getGoogleAccounts,
  getGoogleClientConfig,
  saveGoogleClientConfig,
  type GoogleClientConfig,
} from '../lib/gpu-api';
import styles from '../gpu.module.css';

export function GoogleOAuthSetup({
  accounts: suppliedAccounts,
  onConfigured,
}: {
  accounts?: Array<{ account_id: string; alias?: string | null }>;
  onConfigured: () => void;
}) {
  const [accounts, setAccounts] = useState<Array<{ account_id: string; alias?: string | null }>>(suppliedAccounts || []);
  const [accountId, setAccountId] = useState('');
  const [config, setConfig] = useState<GoogleClientConfig | null>(null);
  const [clientId, setClientId] = useState('');
  const [clientSecret, setClientSecret] = useState('');
  const [webClient, setWebClient] = useState<Record<string, unknown> | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const configRequest = useRef(0);
  const targetGeneration = useRef(0);
  const saveRequest = useRef(0);
  const callback = config?.redirect_uri || `${window.location.origin}/api/gpu/auth/google/callback`;

  useEffect(() => {
    if (suppliedAccounts) {
      setAccounts(suppliedAccounts);
      return;
    }
    const generation = targetGeneration.current;
    let alive = true;
    getGoogleAccounts()
      .then((next) => { if (alive && generation === targetGeneration.current) setAccounts(next); })
      .catch((cause) => { if (alive && generation === targetGeneration.current) setError(apiMessage(cause)); });
    return () => { alive = false; };
  }, [suppliedAccounts]);

  useEffect(() => {
    const requestId = ++configRequest.current;
    let alive = true;
    setConfig(null);
    getGoogleClientConfig(accountId || undefined)
      .then((next) => {
        if (alive && requestId === configRequest.current) setConfig(next);
      })
      .catch((cause) => {
        if (alive && requestId === configRequest.current) setError(apiMessage(cause));
      });
    return () => { alive = false; };
  }, [accountId]);

  function clearDraft() {
    setClientId('');
    setClientSecret('');
    setWebClient(null);
    setConfirmed(false);
    setError('');
    setNotice('');
  }

  function changeTarget(nextAccountId: string) {
    // Invalidate in-flight reads before React runs the next effect cleanup.
    targetGeneration.current += 1;
    configRequest.current += 1;
    setAccountId(nextAccountId);
    setConfig(null);
    clearDraft();
  }

  async function save() {
    if (busy || !confirmed) return;
    const targetAccountId = accountId;
    const targetRequestId = configRequest.current;
    const saveId = ++saveRequest.current;
    setBusy(true); setError(''); setNotice('');
    try {
      const body = webClient
        ? { account_id: targetAccountId || undefined, client_json: webClient, confirmed: true as const }
        : { account_id: targetAccountId || undefined, client_id: clientId.trim(), client_secret: clientSecret, confirmed: true as const };
      const next = await saveGoogleClientConfig(body);
      if (saveId !== saveRequest.current || targetRequestId !== configRequest.current) return;
      setConfig(next); setClientId(''); setClientSecret(''); setWebClient(null); setConfirmed(false);
      setNotice('应用配置已加密保存。下一步点击“连接 Google Colab”和“连接 Google Drive”完成本人授权；不会自动开 GPU。');
      onConfigured();
    } catch (cause) {
      if (saveId === saveRequest.current && targetRequestId === configRequest.current) setError(apiMessage(cause));
    } finally {
      if (saveId === saveRequest.current) setBusy(false);
    }
  }

  return <section className={styles.card} aria-label="Google OAuth 应用配置"><h2>Google Web OAuth 应用配置</h2>
    <p className="muted">{config?.configured ? 'Web OAuth 应用已配置，可用于 Google Drive 授权及 Colab Web OAuth（实验性）。常规 Colab 算力推荐直接在上方账号卡片点击「Colab CLI 授权（推荐）」。' : 'Google Drive 模型库与持久化需要自建 Web OAuth 应用（仅需启用 Google Drive API）；常规 Colab 算力无需此配置，可直接在上方点击「Colab CLI 授权（推荐）」。'}</p>
    <label className="field">配置目标<select aria-label="OAuth 配置目标" value={accountId} disabled={busy} onChange={(event) => changeTarget(event.target.value)}>
      <option value="">共享默认应用</option>
      {accounts.map((account) => <option key={account.account_id} value={account.account_id}>{account.alias || account.account_id}</option>)}
    </select></label>
    <p className="muted">账号专用应用只影响这个 Google 账号，不会影响其他账号；共享默认应用仅保护依赖共享配置的连接（不影响 Colab CLI 算力凭据）。</p>
    <ol className={styles.oauthSteps}>
      <li>打开 <a href="https://console.cloud.google.com/" target="_blank" rel="noreferrer">Google Cloud 控制台</a>，创建或选择自己的项目。</li>
      <li>在“API 和服务 → 库”启用 <strong>Google Drive API</strong>（仅使用「Web OAuth 授权（实验性）」连接 Colab 时才需额外启用 Colaboratory API）。</li>
      <li>配置 Google Auth Platform：应用名称、受众选“外部”；测试模式把自己的 Google 邮箱加入测试用户。</li>
      <li>在“客户端”创建 OAuth 客户端，类型选“Web 应用”，填入下方回调地址，下载客户端 JSON。</li>
      <li>在这里导入 JSON 并保存，然后在上方账号卡片点击 <strong>Drive 授权</strong>；Colab 算力优先使用 <strong>Colab CLI 授权（推荐）</strong>。</li>
    </ol>
    <label className="field">授权回调地址<input aria-label="Google OAuth 回调地址" readOnly value={callback} onFocus={(event) => event.target.select()} /></label>
    <button className="button secondary" type="button" onClick={async () => { try { await navigator.clipboard.writeText(callback); setNotice('已复制回调地址'); } catch { setNotice('请点击回调地址输入框，全选后复制'); } }}>复制回调地址</button>
    <details><summary>需要的 OAuth 权限</summary><p className="muted">Google Drive 必填 <code>drive.file</code>（仅访问本站创建的目录，且按客户端隔离）；<code>colaboratory</code> 仅在选用「Web OAuth 授权（实验性）」连接 Colab 时需要。</p><code className={styles.oauthScope}>https://www.googleapis.com/auth/drive.file</code><code className={styles.oauthScope}>https://www.googleapis.com/auth/colaboratory</code></details>
    <p className="muted">如填写“已获授权的 JavaScript 来源”，使用 {window.location.origin}。授权不会产生 GPU 费用；不要把客户端 JSON 或 Secret 发到群里。</p>
    <details><summary>导入 / 更换 OAuth 客户端</summary><div className={styles.stack}>
      <label className="button secondary">选择 Google 客户端 JSON<input data-testid="google-client-file" hidden type="file" accept="application/json,.json" onChange={async (event) => { const file = event.target.files?.[0]; event.target.value = ''; if (!file) return; try { if (file.size > 65536) throw new Error('客户端 JSON 文件过大'); const value: unknown = JSON.parse(await file.text()); if (!value || typeof value !== 'object' || !('web' in value)) throw new Error('请选择 Web 应用客户端 JSON，不是桌面客户端或服务账号密钥'); setWebClient(value as Record<string, unknown>); setClientId(''); setClientSecret(''); setNotice('已读取文件，尚未保存'); setError(''); } catch (cause) { setWebClient(null); setError(apiMessage(cause)); } }} /></label>
      {!webClient && <><label className="field">或手动填写 Client ID<input autoComplete="off" autoCapitalize="none" spellCheck={false} value={clientId} onChange={(event) => setClientId(event.target.value)} /></label><label className="field">Client Secret<input type="password" autoComplete="new-password" value={clientSecret} onChange={(event) => setClientSecret(event.target.value)} /></label></>}
      <label className={styles.oauthConfirm}><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>这是我自己的 Google 应用，确认加密保存到本站后端。</span></label>
      <button className="button" type="button" disabled={busy || !confirmed || (!webClient && (!clientId || !clientSecret))} onClick={() => void save()}>{busy ? '保存中…' : '保存 OAuth 应用配置'}</button>
    </div></details>
    {error && <p className="status error" role="alert">{error}</p>}{notice && <p className="status" role="status">{notice}</p>}
  </section>;
}
