import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Copy,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  XCircle,
} from 'lucide-react';
import { apiFetch, apiMessage } from '../lib/gpu-api';
import {
  RECOMMENDED_COMFY_COMMIT,
  buildCaddyProxyConfig,
  buildNginxProxyConfig,
  extractComfyCommit,
  extractTunnelToken,
  isValidComfyCommit,
  parseAccessJsonInput,
  previewTunnelToken,
  sourceLabel,
} from '../lib/native-setup';
import styles from '../gpu.module.css';

export type NativeSetupPayload = {
  configured: boolean;
  web_ui_configured: boolean;
  has_db_override: boolean;
  updated_at: string | null;
  recommended_commit: string;
  site_url: string;
  bind_port: number;
  active_session_locked: boolean;
  values: {
    native_comfy_url: string;
    native_comfy_backend_url: string;
    native_comfy_commit: string;
    access_client_id: string;
  };
  secrets: {
    access_client_secret_configured: boolean;
    tunnel_token_configured: boolean;
    tunnel_info: {
      valid_cf_format: boolean;
      tunnel_id: string | null;
      account_id_masked: string | null;
    };
  };
  sources: {
    native_comfy_url: string;
    native_comfy_backend_url: string;
    native_comfy_commit: string;
    access_client_id: string;
    access_client_secret: string;
    tunnel_token: string;
  };
  prerequisites: {
    real_gpu_enabled: boolean;
    oauth_configured: boolean;
    total_accounts: number;
    eligible_accounts: number;
    colab_connected_accounts: number;
    ready_accounts: number;
  };
};

type DiagnoseCheck = {
  id: string;
  label: string;
  status: 'pass' | 'warn' | 'fail' | 'skip';
  detail: string;
  hint?: string;
};

type DiagnoseResult = {
  ok: boolean;
  has_warnings: boolean;
  checks: DiagnoseCheck[];
  checked_at: string;
};

type StepId = 1 | 2 | 3 | 4 | 5;

const STEPS: { id: StepId; short: string; title: string }[] = [
  { id: 1, short: '1. 前置检查', title: '检查真实 GPU 开关与 Google 账号' },
  { id: 2, short: '2. CF 隧道', title: '配置 Cloudflare Named Tunnel' },
  { id: 3, short: '3. CF Access', title: '配置 Cloudflare Access Service Auth' },
  { id: 4, short: '4. 入口与版本', title: '配置网页反代入口与 ComfyUI 版本' },
  { id: 5, short: '5. 保存预检', title: '保存加密配置并执行免开机诊断' },
];

export function NativeSetupWizard({
  configured,
  webUiConfigured,
  openRequest,
  onSaved,
  onOpenSettings,
}: {
  configured?: boolean;
  webUiConfigured?: boolean;
  openRequest?: { seq: number; step?: StepId } | null;
  onSaved?: () => void;
  onOpenSettings?: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [step, setStep] = useState<StepId>(1);
  const [data, setData] = useState<NativeSetupPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [diagnosing, setDiagnosing] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  // Form fields
  const [backendUrl, setBackendUrl] = useState('');
  const [tunnelTokenInput, setTunnelTokenInput] = useState('');
  const [accessClientId, setAccessClientId] = useState('');
  const [accessClientSecretInput, setAccessClientSecretInput] = useState('');
  const [accessJsonPaste, setAccessJsonPaste] = useState('');
  const [showJsonPaste, setShowJsonPaste] = useState(false);
  const [entryMode, setEntryMode] = useState<'web_and_api' | 'api_only'>('web_and_api');
  const [publicUrl, setPublicUrl] = useState('');
  const [commitInput, setCommitInput] = useState(RECOMMENDED_COMFY_COMMIT);
  const [showCustomCommit, setShowCustomCommit] = useState(false);
  const [proxyTab, setProxyTab] = useState<'nginx' | 'caddy'>('nginx');
  const [confirmClear, setConfirmClear] = useState(false);
  const [diag, setDiag] = useState<DiagnoseResult | null>(null);

  const hydrate = useCallback((loaded: NativeSetupPayload) => {
    setData(loaded);
    setBackendUrl(loaded.values.native_comfy_backend_url || '');
    setTunnelTokenInput('');
    setAccessClientId(loaded.values.access_client_id || '');
    setAccessClientSecretInput('');
    setAccessJsonPaste('');
    const urlVal = loaded.values.native_comfy_url || '';
    setPublicUrl(urlVal);
    setEntryMode(loaded.configured && !urlVal ? 'api_only' : 'web_and_api');
    const recommended = loaded.recommended_commit || RECOMMENDED_COMFY_COMMIT;
    const effectiveCommit = loaded.values.native_comfy_commit || recommended;
    setCommitInput(effectiveCommit);
    setShowCustomCommit(effectiveCommit !== recommended);
  }, []);

  const loadSetup = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const loaded = await apiFetch<NativeSetupPayload>('/api/gpu/native/setup');
      hydrate(loaded);
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setLoading(false);
    }
  }, [hydrate]);

  const openWizard = useCallback((targetStep?: StepId) => {
    setStep(targetStep || 1);
    setError('');
    setNotice('');
    setConfirmClear(false);
    setDiag(null);
    dialog.current?.showModal();
    void loadSetup();
  }, [loadSetup]);

  useEffect(() => {
    if (openRequest && openRequest.seq > 0) {
      openWizard(openRequest.step);
    }
  }, [openRequest, openWizard]);

  const tokenPreview = useMemo(() => previewTunnelToken(tunnelTokenInput), [tunnelTokenInput]);
  const recommendedCommit = data?.recommended_commit || RECOMMENDED_COMFY_COMMIT;
  const normalizedCommit = useMemo(() => extractComfyCommit(commitInput), [commitInput]);
  const commitValid = isValidComfyCommit(commitInput);

  const stepReady: Record<StepId, boolean> = {
    1: Boolean(data?.prerequisites.real_gpu_enabled && (data?.prerequisites.ready_accounts || 0) > 0),
    2: Boolean(backendUrl.trim() && (tunnelTokenInput.trim() || data?.secrets.tunnel_token_configured)),
    3: Boolean(accessClientId.trim() && (accessClientSecretInput.trim() || data?.secrets.access_client_secret_configured)),
    4: Boolean(commitValid && (entryMode === 'api_only' || publicUrl.trim())),
    5: Boolean(diag?.ok),
  };

  const proxySnippet = useMemo(() => {
    const port = data?.bind_port || 8788;
    return proxyTab === 'nginx'
      ? buildNginxProxyConfig(publicUrl, port)
      : buildCaddyProxyConfig(publicUrl, port);
  }, [data?.bind_port, proxyTab, publicUrl]);

  async function copyText(text: string, label: string) {
    try {
      await navigator.clipboard.writeText(text);
      setNotice(`已复制${label}`);
    } catch {
      setError('无法访问剪贴板，请选中内容后手动复制。');
    }
  }

  function applyAccessJson(raw: string) {
    setAccessJsonPaste(raw);
    const parsed = parseAccessJsonInput(raw);
    if (parsed) {
      setAccessClientId(parsed.clientId);
      setAccessClientSecretInput(parsed.clientSecret);
      setNotice('已从 JSON 自动填入 Client ID 和 Client Secret');
      setError('');
    }
  }

  async function runDiagnose() {
    setDiagnosing(true);
    setError('');
    try {
      const result = await apiFetch<DiagnoseResult>('/api/gpu/native/setup/diagnose', {
        method: 'POST',
        body: '{}',
      }, 30_000);
      setDiag(result);
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setDiagnosing(false);
    }
  }

  async function saveAndDiagnose(withDiagnose: boolean) {
    setSaving(true);
    setError('');
    setNotice('');
    try {
      const body: Record<string, string> = {
        native_comfy_url: entryMode === 'api_only' ? '' : publicUrl.trim(),
        native_comfy_backend_url: backendUrl.trim(),
        native_comfy_commit: normalizedCommit || recommendedCommit,
        access_client_id: accessClientId.trim(),
      };
      if (accessClientSecretInput.trim()) {
        body.access_client_secret = accessClientSecretInput.trim();
      }
      if (tunnelTokenInput.trim()) {
        body.tunnel_token = extractTunnelToken(tunnelTokenInput);
      }
      const saved = await apiFetch<NativeSetupPayload>('/api/gpu/native/setup', {
        method: 'PUT',
        body: JSON.stringify(body),
      });
      hydrate(saved);
      setNotice('配置已加密保存并即时同步至后端，无需重启服务。');
      onSaved?.();
      if (withDiagnose) {
        await runDiagnose();
      }
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setSaving(false);
    }
  }

  async function clearDbConfig() {
    setSaving(true);
    setError('');
    setNotice('');
    try {
      const next = await apiFetch<NativeSetupPayload>('/api/gpu/native/setup', { method: 'DELETE' });
      hydrate(next);
      setConfirmClear(false);
      setDiag(null);
      setNotice('已清除网页保存的配置，当前已回退使用服务器 .env 与凭据文件。');
      onSaved?.();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setSaving(false);
    }
  }

  const busy = loading || saving || diagnosing;
  const triggerLabel = configured === false
    ? '配置向导'
    : webUiConfigured === false
      ? '隧道配置 · 仅 API'
      : '隧道与域名配置';

  return <>
    <button
      className={configured === false ? 'button' : 'button secondary'}
      type="button"
      onClick={() => openWizard(configured === false ? 1 : 5)}
    >
      <SlidersHorizontal size={15} strokeWidth={1.8} aria-hidden />
      {triggerLabel}
    </button>

    <dialog
      className="nativeRestartDialog nativeSetupDialog"
      ref={dialog}
      aria-labelledby="native-setup-title"
      onCancel={(event) => { if (busy) event.preventDefault(); }}
    >
      <div className="actions nativeApiHeading">
        <div>
          <h2 id="native-setup-title">原生 ComfyUI 配置向导</h2>
          <p className="muted small">凭据使用服务器 ENCRYPTION_KEY 加密存库，即时同步至 API / Worker / Watchdog，并兼容已有 .env 配置。</p>
        </div>
        <button className="button ghost" type="button" disabled={busy} onClick={() => dialog.current?.close()}>关闭</button>
      </div>

      <nav className={styles.wizardTabs} aria-label="配置步骤">
        {STEPS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={styles.wizardTab}
            data-active={step === item.id}
            data-done={stepReady[item.id]}
            onClick={() => setStep(item.id)}
          >
            <span>{item.short}</span>
            {stepReady[item.id] && <CheckCircle2 size={13} aria-hidden />}
          </button>
        ))}
      </nav>

      {data?.active_session_locked && (
        <p className={styles.cuWarning} role="status">
          当前有运行中或启动中的原生实例：修改 Cloudflare 隧道域名、Tunnel Token 或 Access 凭据前需先停止旧实例；补充网页域名或更新下次启动的 ComfyUI 版本可直接保存。
        </p>
      )}

      {data && !data.has_db_override && data.configured && (
        <p className="status" role="status">
          当前正在使用服务器 .env 与凭据文件配置（已兼容生效）。在向导中保存后，网页加密配置将优先于 .env，可随时一键清除并回退。
        </p>
      )}

      {error && <p className="status error" role="alert">{error}</p>}
      {notice && <p className="status" role="status">{notice}</p>}
      {loading && !data && <p className="muted" role="status">正在读取当前配置…</p>}

      {data && (
        <div className={styles.wizardBody}>
          {step === 1 && (
            <section className={styles.stack} aria-label={STEPS[0].title}>
              <h3>第一步：确认服务器开关与 Google 账号</h3>
              <p className="muted small">
                原生 ComfyUI 会在你的 Google Colab GPU 上启动官方界面，并通过 Google Drive 持久化工作流、插件归档与环境缓存（即使选择本地模型模式也会挂载 Drive）。
              </p>

              <div className={styles.wizardCheckList}>
                <div className={styles.wizardCheckRow}>
                  <div>
                    <strong>真实 GPU 开关（GPU_REAL_ENABLED）</strong>
                    <p className="muted small">
                      {data.prerequisites.real_gpu_enabled
                        ? '已开启，允许分配真实 Google Colab GPU。'
                        : '尚未开启。请在 VPS 的 .env 文件中设置 GPU_REAL_ENABLED=1 并重启服务（为防误扣费，该开关仅由 .env 控制）。'}
                    </p>
                  </div>
                  <span className={`pill ${data.prerequisites.real_gpu_enabled ? 'solid' : 'danger'}`}>
                    {data.prerequisites.real_gpu_enabled ? '已开启' : '未开启'}
                  </span>
                </div>

                <div className={styles.wizardCheckRow}>
                  <div>
                    <strong>Google OAuth 应用客户端</strong>
                    <p className="muted small">
                      {data.prerequisites.oauth_configured
                        ? '已配置 Google Web OAuth 客户端。'
                        : '尚未配置。请在 Google Cloud 创建 Web 应用 OAuth 客户端 JSON，并在「设置 → Google 账号」导入。'}
                    </p>
                  </div>
                  <span className={`pill ${data.prerequisites.oauth_configured ? 'solid' : 'danger'}`}>
                    {data.prerequisites.oauth_configured ? '已配置' : '未配置'}
                  </span>
                </div>

                <div className={styles.wizardCheckRow}>
                  <div>
                    <strong>可用 Google 账号（同时连接 Colab 与 Drive）</strong>
                    <p className="muted small">
                      {data.prerequisites.ready_accounts > 0
                        ? `已有 ${data.prerequisites.ready_accounts} 个账号就绪（启用、允许分配 GPU，且已授权 Colab 与 Drive）。`
                        : `当前就绪账号 0 个（总账号 ${data.prerequisites.total_accounts} 个，已连 Colab ${data.prerequisites.colab_connected_accounts} 个）。请确保至少一个账号同时授权了 Colab 和 Drive（或已绑定共享 Drive 账号）并开启「允许分配 GPU」。`}
                    </p>
                  </div>
                  <span className={`pill ${data.prerequisites.ready_accounts > 0 ? 'solid' : 'danger'}`}>
                    {data.prerequisites.ready_accounts > 0 ? `${data.prerequisites.ready_accounts} 个就绪` : '未就绪'}
                  </span>
                </div>
              </div>

              <div className="actions">
                <button type="button" className="button secondary small" disabled={busy} onClick={() => void loadSetup()}>
                  <RefreshCw size={14} aria-hidden />刷新检查状态
                </button>
                {onOpenSettings && (!data.prerequisites.oauth_configured || data.prerequisites.ready_accounts === 0) && (
                  <button
                    type="button"
                    className="button small"
                    onClick={() => {
                      dialog.current?.close();
                      onOpenSettings();
                    }}
                  >
                    前往「设置」配置 Google 账号
                  </button>
                )}
              </div>
            </section>
          )}

          {step === 2 && (
            <section className={styles.stack} aria-label={STEPS[1].title}>
              <h3>第二步：创建 Cloudflare Named Tunnel（内网数据通道）</h3>
              <ol className={styles.oauthSteps}>
                <li>
                  打开 <a href="https://one.dash.cloudflare.com/" target="_blank" rel="noreferrer">Cloudflare Zero Trust</a> → <strong>Networks → Tunnels</strong> → <strong>Create a tunnel</strong> → 选择 <strong>Cloudflared</strong>，输入名称后点 <strong>Save tunnel</strong>。
                </li>
                <li>
                  进入连接器页面后，<strong>只复制命令里的 Token</strong>（可直接把整条 <code>cloudflared ...</code> 命令粘贴到下方输入框，向导会自动提取）。
                </li>
                <li>
                  <strong>绕过“下一步”灰置的方法</strong>：复制完 Token 后，<strong>直接退出创建向导返回 Tunnels 列表页</strong>（刚建的隧道会显示为 <code>INACTIVE</code>），点击它的 <strong>Configure → Published application routes（或 Public Hostname）→ Add</strong>：
                  <ul>
                    <li>子域名填一个专用内部域名（例如 <code>colab-backend.example.com</code>，不要与 Studio 或原生网页域名相同）。</li>
                    <li>Service Type 选 <code>HTTP</code>，URL 填 <code>127.0.0.1:8188</code>，点保存即可。</li>
                  </ul>
                </li>
              </ol>

              <div className={styles.cuWarning} role="note">
                <strong>重要避坑：切勿在 VPS 上运行 Cloudflare 页面的安装命令！</strong>
                <br />
                该隧道由 Colab GPU 开机时自动连接。如果在 VPS 上安装为常驻服务，后续启动 Colab 时两边会同时抢占同一条隧道，导致一半请求随机 502。
              </div>

              <label className="field">
                <span className={styles.wizardFieldHead}>
                  <span>内部隧道域名（NATIVE_COMFY_BACKEND_URL）</span>
                  <span className="pill">{sourceLabel(data.sources.native_comfy_backend_url)}</span>
                </span>
                <input
                  type="url"
                  placeholder="https://colab-backend.example.com"
                  autoCapitalize="none"
                  spellCheck={false}
                  value={backendUrl}
                  onChange={(e) => setBackendUrl(e.target.value)}
                />
                <span className="muted small">必须为独立 HTTPS 源地址（不带路径），且不能与 Studio 域名或原生网页域名相同。</span>
              </label>

              <label className="field">
                <span className={styles.wizardFieldHead}>
                  <span>Cloudflare Tunnel Token</span>
                  <span className="pill">{sourceLabel(data.sources.tunnel_token)}</span>
                </span>
                <textarea
                  rows={2}
                  spellCheck={false}
                  autoCapitalize="none"
                  placeholder={data.secrets.tunnel_token_configured
                    ? '已配置（留空则保持原 Token 不变；粘贴新 Token 或完整 cloudflared 命令可覆盖）'
                    : '直接粘贴 Token（eyJ...）或整条 cloudflared service install eyJ... 命令'}
                  value={tunnelTokenInput}
                  onChange={(e) => setTunnelTokenInput(e.target.value)}
                />
                {tunnelTokenInput.trim() ? (
                  tokenPreview.validCfFormat ? (
                    <span className="muted small">
                      已识别标准 Cloudflare Named Tunnel Token · Tunnel ID: <code>{tokenPreview.tunnelId}</code>（账号 {tokenPreview.accountIdMasked}）
                    </span>
                  ) : (
                    <span className="muted small">
                      已提取字符长度 {tokenPreview.extracted.length}（未解析出标准 Base64 JSON，请核对是否复制完整）。
                    </span>
                  )
                ) : data.secrets.tunnel_token_configured && data.secrets.tunnel_info.tunnel_id ? (
                  <span className="muted small">
                    当前已保存 Tunnel ID: <code>{data.secrets.tunnel_info.tunnel_id}</code>（账号 {data.secrets.tunnel_info.account_id_masked}）
                  </span>
                ) : null}
              </label>
            </section>
          )}

          {step === 3 && (
            <section className={styles.stack} aria-label={STEPS[2].title}>
              <h3>第三步：用 Cloudflare Access Service Auth 锁住内部隧道</h3>
              <ol className={styles.oauthSteps}>
                <li>
                  在 Cloudflare Zero Trust 进入 <strong>Access → Service Auth → Create Service Token</strong>，创建后立刻保存显示的 <strong>Client ID</strong> 和 <strong>Client Secret</strong>（关闭后无法再次查看 Secret）。
                </li>
                <li>
                  进入 <strong>Access → Applications → Add an application → Self-hosted</strong>：
                  <ul>
                    <li>Application domain 填上一步的内部隧道域名（例如 <code>{backendUrl ? backendUrl.replace(/^https?:\/\//, '') : 'colab-backend.example.com'}</code>），<strong>Path 留空</strong>以保护全路径。</li>
                    <li>在 Policies 页面新增规则：Action 必须选 <strong>Service Auth</strong>（不要选 Allow 或 Bypass），Include 选 <strong>Service Token</strong> 并选中刚才创建的 Token。</li>
                  </ul>
                </li>
                <li>
                  <strong>安全要求</strong>：不要给 <code>/manager/*</code>、<code>/v2/*</code>、<code>/customnode/*</code> 或 <code>/api/*</code> 添加 Bypass；开机及预检会主动验证未携带 Service Token 的请求必须被拦截。
                </li>
              </ol>

              <div className="actions">
                <button type="button" className="button secondary small" onClick={() => setShowJsonPaste((v) => !v)}>
                  {showJsonPaste ? '收起 JSON 粘贴框' : '粘贴 comfy-access.json 快捷导入'}
                </button>
              </div>

              {showJsonPaste && (
                <label className="field">
                  <span>粘贴 JSON（自动拆填下方两项）</span>
                  <textarea
                    rows={3}
                    spellCheck={false}
                    placeholder='{"client_id": "xxxx.access", "client_secret": "xxxx"}'
                    value={accessJsonPaste}
                    onChange={(e) => applyAccessJson(e.target.value)}
                  />
                </label>
              )}

              <label className="field">
                <span className={styles.wizardFieldHead}>
                  <span>CF-Access-Client-Id</span>
                  <span className="pill">{sourceLabel(data.sources.access_client_id)}</span>
                </span>
                <input
                  type="text"
                  autoComplete="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  placeholder="例如：1234567890abcdef.access"
                  value={accessClientId}
                  onChange={(e) => setAccessClientId(e.target.value)}
                />
              </label>

              <label className="field">
                <span className={styles.wizardFieldHead}>
                  <span>CF-Access-Client-Secret</span>
                  <span className="pill">{sourceLabel(data.sources.access_client_secret)}</span>
                </span>
                <input
                  type="password"
                  autoComplete="new-password"
                  placeholder={data.secrets.access_client_secret_configured
                    ? '已配置（留空保持不变；若更换了 Client ID 则需同时填写新 Secret）'
                    : '粘贴创建 Service Token 时获得的 Client Secret'}
                  value={accessClientSecretInput}
                  onChange={(e) => setAccessClientSecretInput(e.target.value)}
                />
              </label>
            </section>
          )}

          {step === 4 && (
            <section className={styles.stack} aria-label={STEPS[3].title}>
              <h3>第四步：选择访问入口与 ComfyUI 版本</h3>

              <div className={styles.segmented} role="tablist" aria-label="原生访问模式">
                <button
                  type="button"
                  role="tab"
                  aria-selected={entryMode === 'web_and_api'}
                  data-active={entryMode === 'web_and_api'}
                  onClick={() => setEntryMode('web_and_api')}
                >
                  <span className={styles.segmentedText}>
                    <span>原生网页 UI + API</span>
                    <small>打开官方节点编辑器（需配独立域名）</small>
                  </span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={entryMode === 'api_only'}
                  data-active={entryMode === 'api_only'}
                  onClick={() => setEntryMode('api_only')}
                >
                  <span className={styles.segmentedText}>
                    <span>仅外部 API / 酒馆模式</span>
                    <small>留空网页域名，用子路径短码或独立端口</small>
                  </span>
                </button>
              </div>

              {entryMode === 'web_and_api' ? (
                <>
                  <label className="field">
                    <span className={styles.wizardFieldHead}>
                      <span>原生 ComfyUI 网页域名（NATIVE_COMFY_URL）</span>
                      <span className="pill">{sourceLabel(data.sources.native_comfy_url)}</span>
                    </span>
                    <input
                      type="url"
                      autoCapitalize="none"
                      spellCheck={false}
                      placeholder="https://comfy.example.com（本机调试可用 http://comfy.localhost:8788）"
                      value={publicUrl}
                      onChange={(e) => setPublicUrl(e.target.value)}
                    />
                    <span className="muted small">
                      独立 HTTPS 源地址，不能与 Studio 域名（{data.site_url || 'SITE_URL'}）或内部隧道域名相同。将 DNS 指向本 VPS，与 Studio 转发到同一个本地端口 <code>{data.bind_port}</code>。
                    </span>
                  </label>

                  <div className={styles.wizardCodeBox}>
                    <div className={styles.wizardCodeHead}>
                      <div className="actions">
                        <button
                          type="button"
                          className={`button small ${proxyTab === 'nginx' ? '' : 'secondary'}`}
                          onClick={() => setProxyTab('nginx')}
                        >
                          Nginx 配置
                        </button>
                        <button
                          type="button"
                          className={`button small ${proxyTab === 'caddy' ? '' : 'secondary'}`}
                          onClick={() => setProxyTab('caddy')}
                        >
                          Caddy 配置
                        </button>
                      </div>
                      <button
                        type="button"
                        className="button secondary small"
                        onClick={() => void copyText(proxySnippet, `${proxyTab === 'nginx' ? 'Nginx' : 'Caddy'} 反代配置`)}
                      >
                        <Copy size={13} aria-hidden />复制配置
                      </button>
                    </div>
                    <pre className={styles.diagnostics}>{proxySnippet}</pre>
                    <p className="muted small">
                      整个域名所有路径都直接转给 Studio API（不要把 <code>/api/*</code> 单独拆出去），务必保留原始 <code>Host</code> 头并关闭代理缓冲。
                    </p>
                  </div>
                </>
              ) : (
                <div className="status">
                  <strong>仅 API 模式说明</strong>
                  <p className="muted small">
                    此模式下 <code>NATIVE_COMFY_URL</code> 将留空，无需配置第二个公网域名和反向代理。保存并启动原生 GPU 后，直接在主页面右上角「API 设置」开启「弱鉴权（URL 路径短码）」或「独立监听端口」，把生成的 <code>{data.site_url || 'https://你的站点'}/comfy/n/&lt;短码&gt;</code> 填入酒馆即可。日后想用官方网页界面，随时回来切回左栏补充域名。
                  </p>
                </div>
              )}

              <div className={styles.inlinePanel}>
                <div className={styles.wizardFieldHead}>
                  <strong>ComfyUI 锁定版本（NATIVE_COMFY_COMMIT）</strong>
                  <span className="pill">{sourceLabel(data.sources.native_comfy_commit)}</span>
                </div>
                <p className="muted small">
                  每次开机都会从官方仓库拉取这个固定版本，确保内置 ComfyUI-Manager 可用、与内置工作流版本匹配，并稳定命中启动加速缓存。新手保持默认推荐版本即可。
                </p>
                <div className="actions">
                  <span className="pill solid">
                    当前：{normalizedCommit ? normalizedCommit.slice(0, 12) : '未填写'}
                    {normalizedCommit === recommendedCommit ? '（推荐稳定版）' : ''}
                  </span>
                  {normalizedCommit !== recommendedCommit && (
                    <button
                      type="button"
                      className="button secondary small"
                      onClick={() => {
                        setCommitInput(recommendedCommit);
                        setShowCustomCommit(false);
                      }}
                    >
                      恢复推荐版本 ({recommendedCommit.slice(0, 12)})
                    </button>
                  )}
                  <button
                    type="button"
                    className="button ghost small"
                    onClick={() => setShowCustomCommit((v) => !v)}
                  >
                    {showCustomCommit ? '收起自定义版本' : '自定义 Commit（进阶）'}
                  </button>
                </div>
                {showCustomCommit && (
                  <label className="field">
                    <span>官方 ComfyUI 40 位 Commit SHA（或粘贴 GitHub commit 链接）</span>
                    <input
                      type="text"
                      spellCheck={false}
                      autoCapitalize="none"
                      value={commitInput}
                      placeholder={recommendedCommit}
                      onChange={(e) => setCommitInput(e.target.value)}
                    />
                    {!commitValid && (
                      <span className={styles.errorText}>必须为 40 位十六进制提交哈希，且需支持 --enable-manager 与 manager_requirements.txt。</span>
                    )}
                  </label>
                )}
              </div>
            </section>
          )}

          {step === 5 && (
            <section className={styles.stack} aria-label={STEPS[4].title}>
              <h3>第五步：保存配置并运行零成本预检</h3>
              <p className="muted small">
                预检不会申请 Colab GPU，也不会消耗任何 CU。它会在开机前提前验证 Cloudflare Access 是否拦截了未授权路径、Service Token 能否穿透 Access，以及公网反代是否把 Host 正确透传到本站。
              </p>

              <dl className={styles.kv}>
                <dt>内部隧道域名</dt><dd>{backendUrl.trim() || '未填写'}</dd>
                <dt>Tunnel Token</dt><dd>{tunnelTokenInput.trim() ? '将更新为新输入的 Token' : data.secrets.tunnel_token_configured ? '保持已配置值' : '未填写'}</dd>
                <dt>Access Client ID</dt><dd>{accessClientId.trim() || '未填写'}</dd>
                <dt>Access Secret</dt><dd>{accessClientSecretInput.trim() ? '将更新为新输入的 Secret' : data.secrets.access_client_secret_configured ? '保持已配置值' : '未填写'}</dd>
                <dt>原生网页域名</dt><dd>{entryMode === 'api_only' ? '留空（仅 API 模式）' : publicUrl.trim() || '未填写'}</dd>
                <dt>ComfyUI 版本</dt><dd>{normalizedCommit ? normalizedCommit.slice(0, 12) : '未填写'}</dd>
              </dl>

              <div className="actions">
                <button
                  type="button"
                  className="button"
                  disabled={busy}
                  onClick={() => void saveAndDiagnose(true)}
                >
                  <ShieldCheck size={16} aria-hidden />
                  {saving ? '正在保存…' : diagnosing ? '正在运行预检…' : '保存配置并运行预检'}
                </button>
                <button
                  type="button"
                  className="button secondary"
                  disabled={busy}
                  onClick={() => void saveAndDiagnose(false)}
                >
                  仅保存
                </button>
                <button
                  type="button"
                  className="button secondary"
                  disabled={busy}
                  onClick={() => void runDiagnose()}
                >
                  <RefreshCw size={14} aria-hidden />
                  {diagnosing ? '预检中…' : '仅运行预检'}
                </button>
              </div>

              {diag && (
                <div className={styles.inlinePanel}>
                  <div className="actions nativeApiHeading">
                    <strong>预检结果</strong>
                    <span className={`pill ${diag.ok ? 'solid' : 'danger'}`}>
                      {diag.ok ? (diag.has_warnings ? '通过（含提示）' : '全部通过') : '存在未通过项'}
                    </span>
                  </div>
                  <div className={styles.wizardCheckList}>
                    {diag.checks.map((item) => (
                      <div key={item.id} className={styles.wizardCheckRow}>
                        <div>
                          <div className={styles.wizardCheckTitle}>
                            {item.status === 'pass' && <CheckCircle2 size={15} aria-hidden />}
                            {item.status === 'warn' && <AlertTriangle size={15} aria-hidden />}
                            {item.status === 'fail' && <XCircle size={15} aria-hidden />}
                            <strong>{item.label}</strong>
                          </div>
                          <p className="muted small">{item.detail}</p>
                          {item.hint && <p className={item.status === 'fail' ? styles.errorText : 'muted small'}>{item.hint}</p>}
                        </div>
                        <span className={`pill ${item.status === 'pass' ? 'solid' : item.status === 'fail' ? 'danger' : ''}`}>
                          {item.status === 'pass' ? '通过' : item.status === 'warn' ? '提示' : item.status === 'fail' ? '未通过' : '跳过'}
                        </span>
                      </div>
                    ))}
                  </div>
                  {diag.ok && (
                    <p className="status" role="status">
                      {data.web_ui_configured
                        ? '环境与隧道预检全部通过！现在可以关闭向导，直接点击「启动 / 复用 GPU」开启原生 ComfyUI。'
                        : '隧道预检全部通过（仅 API 模式）！启动 GPU 后，请点主页面右上角「API 设置」开启路径短码或独立端口给酒馆直连。'}
                    </p>
                  )}
                </div>
              )}

              {data.has_db_override && (
                <div className={styles.inlinePanel}>
                  <div className="actions">
                    {confirmClear ? (
                      <>
                        <span className="muted small">确认清除网页保存的加密配置并回退使用服务器 .env / 文件？</span>
                        <button type="button" className="button danger small" disabled={busy} onClick={() => void clearDbConfig()}>
                          确认清除
                        </button>
                        <button type="button" className="button ghost small" disabled={busy} onClick={() => setConfirmClear(false)}>
                          取消
                        </button>
                      </>
                    ) : (
                      <button type="button" className="button ghost small" disabled={busy} onClick={() => setConfirmClear(true)}>
                        清除网页配置（回退 .env / 文件）
                      </button>
                    )}
                  </div>
                </div>
              )}
            </section>
          )}

          <div className={styles.wizardFooter}>
            <button
              type="button"
              className="button secondary"
              disabled={step === 1 || busy}
              onClick={() => setStep((s) => Math.max(1, s - 1) as StepId)}
            >
              <ChevronLeft size={15} aria-hidden />上一步
            </button>
            <span className="muted small">第 {step} / 5 步</span>
            {step < 5 ? (
              <button
                type="button"
                className="button"
                disabled={busy}
                onClick={() => setStep((s) => Math.min(5, s + 1) as StepId)}
              >
                下一步<ChevronRight size={15} aria-hidden />
              </button>
            ) : (
              <button
                type="button"
                className="button secondary"
                disabled={busy}
                onClick={() => dialog.current?.close()}
              >
                完成并关闭
              </button>
            )}
          </div>
        </div>
      )}
    </dialog>
  </>;
}
