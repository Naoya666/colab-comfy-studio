import { useCallback, useEffect, useMemo, useState } from 'react';
import { Bell, RefreshCw, RotateCcw, Save, Send, Trash2 } from 'lucide-react';
import {
  apiMessage,
  getTelegramDeliveries,
  getTelegramSettings,
  saveTelegramSettings,
  testTelegramSettings,
  type TelegramDeliveryItem,
  type TelegramEventCatalogItem,
  type TelegramNotificationSettings,
  type TelegramPriority,
} from '../lib/gpu-api';
import { formatDateTime } from '../lib/ui-helpers';
import { Card } from './ui';
import styles from '../gpu.module.css';

const DEFAULT_API_BASE = 'https://api.telegram.org';

const GROUP_META: Record<TelegramPriority, { title: string; subtitle: string; badge: string }> = {
  P0: {
    title: 'P0 · 高危止损与人工介入',
    subtitle: '默认勾选：关系 CU 止损、防空烧与账号可用性，建议始终开启。',
    badge: '默认开启',
  },
  P1: {
    title: 'P1 · 长耗时运维与实例生命周期',
    subtitle: '默认勾选：原生 ComfyUI、抢卡、自动恢复、跨域转移与模型库后台传输结果。',
    badge: '默认开启',
  },
  P2: {
    title: 'P2 · 任务产出与定时摘要',
    subtitle: '默认不勾选：按需开启批次/创作任务结果、到期结算与每日 CU 简报。',
    badge: '可选',
  },
};

const DELIVERY_STATE_LABEL: Record<string, string> = {
  sent: '已发送',
  pending: '待发送',
  sending: '发送中',
  failed: '失败',
  skipped: '已跳过',
};

export function TelegramSettings({ canEdit }: { canEdit: boolean }) {
  const [saved, setSaved] = useState<TelegramNotificationSettings | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [botTokenInput, setBotTokenInput] = useState('');
  const [clearToken, setClearToken] = useState(false);
  const [chatId, setChatId] = useState('');
  const [threadIdText, setThreadIdText] = useState('');
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE);
  const [events, setEvents] = useState<Record<string, boolean>>({});
  const [includeCreationApiSuccess, setIncludeCreationApiSuccess] = useState(false);
  const [expiryWarnText, setExpiryWarnText] = useState('15');
  const [dailyHourText, setDailyHourText] = useState('23');
  const [drivePctText, setDrivePctText] = useState('10');

  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const [logsOpen, setLogsOpen] = useState(false);
  const [logsLoading, setLogsLoading] = useState(false);
  const [deliveries, setDeliveries] = useState<TelegramDeliveryItem[]>([]);

  const apply = useCallback((cfg: TelegramNotificationSettings) => {
    setSaved(cfg);
    setEnabled(Boolean(cfg.enabled));
    setBotTokenInput('');
    setClearToken(false);
    setChatId(cfg.chat_id || '');
    setThreadIdText(cfg.message_thread_id != null ? String(cfg.message_thread_id) : '');
    setApiBaseUrl(cfg.api_base_url || DEFAULT_API_BASE);
    setEvents({ ...(cfg.events || {}) });
    setIncludeCreationApiSuccess(Boolean(cfg.include_creation_api_success));
    setExpiryWarnText(String(cfg.expiry_warn_minutes ?? 15));
    setDailyHourText(String(cfg.daily_report_hour ?? 23));
    setDrivePctText(String(cfg.drive_low_quota_percent ?? 10));
  }, []);

  useEffect(() => {
    let alive = true;
    setError('');
    getTelegramSettings()
      .then((cfg) => {
        if (alive) apply(cfg);
      })
      .catch((cause) => {
        if (alive) setError(apiMessage(cause));
      });
    return () => { alive = false; };
  }, [apply]);

  const loadDeliveries = useCallback(async () => {
    setLogsLoading(true);
    try {
      const res = await getTelegramDeliveries(20);
      setDeliveries(res.items || []);
    } catch {
      // Ignore transient delivery history load errors; main settings remain usable.
    } finally {
      setLogsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (logsOpen) {
      void loadDeliveries();
    }
  }, [logsOpen, loadDeliveries]);

  const catalogByGroup = useMemo(() => {
    const groups: Record<TelegramPriority, TelegramEventCatalogItem[]> = { P0: [], P1: [], P2: [] };
    for (const item of saved?.catalog || []) {
      if (item.priority === 'P0' || item.priority === 'P1' || item.priority === 'P2') {
        groups[item.priority].push(item);
      }
    }
    return groups;
  }, [saved?.catalog]);

  const parsedExpiryWarn = Number(expiryWarnText);
  const parsedDailyHour = Number(dailyHourText);
  const parsedDrivePct = Number(drivePctText);
  const validExpiryWarn = Number.isInteger(parsedExpiryWarn) && parsedExpiryWarn >= 5 && parsedExpiryWarn <= 60;
  const validDailyHour = Number.isInteger(parsedDailyHour) && parsedDailyHour >= 0 && parsedDailyHour <= 23;
  const validDrivePct = Number.isInteger(parsedDrivePct) && parsedDrivePct >= 1 && parsedDrivePct <= 50;

  const trimmedThread = threadIdText.trim();
  const parsedThreadId = trimmedThread === '' ? null : Number(trimmedThread);
  const validThreadId = parsedThreadId === null || (Number.isInteger(parsedThreadId) && parsedThreadId > 0);

  const hasEffectiveToken = Boolean(botTokenInput.trim()) || (Boolean(saved?.bot_token_configured) && !clearToken);
  const hasEffectiveChatId = Boolean(chatId.trim());
  const canEnableOrTest = hasEffectiveToken && hasEffectiveChatId;
  const formValid = validExpiryWarn && validDailyHour && validDrivePct && validThreadId && (!enabled || canEnableOrTest);

  const dirty = useMemo(() => {
    if (!saved) return false;
    const savedEvents = saved.events || {};
    const eventsChanged = Object.keys({ ...savedEvents, ...events }).some(
      (k) => Boolean(events[k]) !== Boolean(savedEvents[k]),
    );
    return (
      enabled !== Boolean(saved.enabled)
      || Boolean(botTokenInput.trim())
      || clearToken
      || chatId.trim() !== (saved.chat_id || '')
      || parsedThreadId !== (saved.message_thread_id ?? null)
      || (apiBaseUrl.trim() || DEFAULT_API_BASE) !== (saved.api_base_url || DEFAULT_API_BASE)
      || eventsChanged
      || includeCreationApiSuccess !== Boolean(saved.include_creation_api_success)
      || parsedExpiryWarn !== saved.expiry_warn_minutes
      || parsedDailyHour !== saved.daily_report_hour
      || parsedDrivePct !== saved.drive_low_quota_percent
    );
  }, [
    saved, enabled, botTokenInput, clearToken, chatId, parsedThreadId, apiBaseUrl,
    events, includeCreationApiSuccess, parsedExpiryWarn, parsedDailyHour, parsedDrivePct,
  ]);

  function toggleEvent(key: string) {
    setEvents((curr) => ({ ...curr, [key]: !curr[key] }));
    setStatus('');
  }

  function setGroupAll(priority: TelegramPriority, value: boolean) {
    const items = catalogByGroup[priority];
    setEvents((curr) => {
      const next = { ...curr };
      for (const item of items) next[item.key] = value;
      return next;
    });
    setStatus('');
  }

  function resetDefaults() {
    const next: Record<string, boolean> = {};
    for (const item of saved?.catalog || []) {
      next[item.key] = Boolean(item.default);
    }
    setEvents(next);
    setIncludeCreationApiSuccess(false);
    setExpiryWarnText('15');
    setDailyHourText('23');
    setDrivePctText('10');
    setStatus('');
  }

  async function save() {
    if (!canEdit || busy || !formValid) return;
    setBusy(true);
    setError('');
    setStatus('');
    try {
      const next = await saveTelegramSettings({
        enabled,
        ...(botTokenInput.trim() ? { bot_token: botTokenInput.trim() } : {}),
        ...(clearToken ? { clear_bot_token: true } : {}),
        chat_id: chatId.trim(),
        message_thread_id: parsedThreadId,
        api_base_url: apiBaseUrl.trim() || DEFAULT_API_BASE,
        events,
        include_creation_api_success: includeCreationApiSuccess,
        expiry_warn_minutes: parsedExpiryWarn,
        daily_report_hour: parsedDailyHour,
        drive_low_quota_percent: parsedDrivePct,
      });
      apply(next);
      setStatus(next.enabled ? 'Telegram 推送配置已保存并生效。' : 'Telegram 推送配置已保存（当前未启用推送）。');
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function sendTest() {
    if (!canEdit || testing || !canEnableOrTest || !validThreadId) return;
    setTesting(true);
    setError('');
    setStatus('');
    try {
      const res = await testTelegramSettings({
        ...(botTokenInput.trim() ? { bot_token: botTokenInput.trim() } : {}),
        chat_id: chatId.trim(),
        message_thread_id: parsedThreadId,
        api_base_url: apiBaseUrl.trim() || DEFAULT_API_BASE,
        events,
      });
      setStatus(`测试消息已发送至 Telegram（message_id: ${res.message_id ?? 'ok'}）。`);
      if (logsOpen) {
        void loadDeliveries();
      }
    } catch (cause) {
      setError(apiMessage(cause));
      if (logsOpen) {
        void loadDeliveries();
      }
    } finally {
      setTesting(false);
    }
  }

  const enabledCount = useMemo(
    () => (saved?.catalog || []).filter((item) => Boolean(events[item.key])).length,
    [saved?.catalog, events],
  );
  const totalCount = saved?.catalog?.length || 0;

  return (
    <Card
      icon={<Bell size={17} strokeWidth={1.6} aria-hidden />}
      title="Telegram 消息推送"
      aside={
        saved ? (
          <span className={`pill ${saved.enabled ? 'solid' : ''}`}>
            {saved.enabled ? `已启用 · ${enabledCount}/${totalCount}` : saved.bot_token_configured ? '已配置未启用' : '未启用'}
          </span>
        ) : undefined
      }
    >
      <p className="muted small">
        通过 Telegram Bot 异步接收 CU 熔断、实例保护隔离、清理失败、原生 ComfyUI 就绪、抢卡/自动恢复与模型库传输通知。所有推送走独立后台队列，不阻塞 GPU 调度，且凭据加密存库。
      </p>
      {status && <p className="status" role="status">{status}</p>}
      {error && <p className="status error" role="alert">{error}</p>}

      {!saved ? (
        !error && <p className="muted small">正在读取 Telegram 推送配置…</p>
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
              <strong>启用 Telegram 消息推送</strong>（配置 Bot Token 与 Chat ID 后生效）
            </span>
          </label>

          <div className={styles.retryFieldsRow}>
            <div className={styles.tgFieldCol}>
              <label className="field">
                Bot Token
                <input
                  type="password"
                  autoComplete="new-password"
                  disabled={!canEdit || busy}
                  value={botTokenInput}
                  placeholder={
                    saved.bot_token_configured && !clearToken
                      ? `已加密保存（${saved.bot_token_hint || '****'}），留空保持不变`
                      : '从 @BotFather 获取，如 123456789:ABCdef...'
                  }
                  onChange={(e) => {
                    setBotTokenInput(e.target.value);
                    if (clearToken) setClearToken(false);
                    setStatus('');
                  }}
                />
              </label>
              {canEdit && saved.bot_token_configured && !clearToken && !botTokenInput && (
                <button
                  type="button"
                  className="button ghost small"
                  style={{ justifySelf: 'start' }}
                  disabled={busy}
                  onClick={() => {
                    setClearToken(true);
                    setEnabled(false);
                    setStatus('');
                  }}
                >
                  <Trash2 size={13} aria-hidden />清除已保存的 Token
                </button>
              )}
              {clearToken && <span className="muted small">保存后将清除已存 Bot Token。</span>}
            </div>

            <label className="field">
              Chat ID
              <input
                type="text"
                disabled={!canEdit || busy}
                value={chatId}
                placeholder="个人数字 ID、群组 -100... 或 @频道名"
                onChange={(e) => {
                  setChatId(e.target.value);
                  setStatus('');
                }}
              />
            </label>
          </div>

          <details className="foldable">
            <summary>高级连接选项（话题群 Thread ID / 自定义反代地址）</summary>
            <div className="foldContent">
              <div className={styles.retryFieldsRow}>
                <label className="field">
                  话题 ID（message_thread_id，可选）
                  <input
                    type="number"
                    min={1}
                    step={1}
                    disabled={!canEdit || busy}
                    value={threadIdText}
                    placeholder="普通私聊或普通群留空即可"
                    onChange={(e) => {
                      setThreadIdText(e.target.value);
                      setStatus('');
                    }}
                  />
                  {!validThreadId && <span className="status error">请输入正整数话题 ID</span>}
                </label>

                <label className="field">
                  Telegram Bot API 地址
                  <input
                    type="url"
                    disabled={!canEdit || busy}
                    value={apiBaseUrl}
                    placeholder={DEFAULT_API_BASE}
                    onChange={(e) => {
                      setApiBaseUrl(e.target.value);
                      setStatus('');
                    }}
                  />
                </label>
              </div>
            </div>
          </details>

          {(['P0', 'P1', 'P2'] as TelegramPriority[]).map((priority) => {
            const items = catalogByGroup[priority];
            const meta = GROUP_META[priority];
            const checkedInGroup = items.filter((item) => Boolean(events[item.key])).length;
            return (
              <fieldset key={priority} className={styles.tgGroupFieldset} disabled={!canEdit || busy}>
                <div className={styles.retryPoolHead}>
                  <legend className={styles.tgGroupLegend}>
                    <span>{meta.title}</span>
                    <span className="pill">{checkedInGroup}/{items.length}</span>
                  </legend>
                  {canEdit && (
                    <div className={styles.retryPoolQuickActions}>
                      <button
                        type="button"
                        className="button ghost small"
                        disabled={busy || checkedInGroup === items.length}
                        onClick={() => setGroupAll(priority, true)}
                      >
                        全选
                      </button>
                      <button
                        type="button"
                        className="button ghost small"
                        disabled={busy || checkedInGroup === 0}
                        onClick={() => setGroupAll(priority, false)}
                      >
                        全不选
                      </button>
                    </div>
                  )}
                </div>
                <p className="muted small" style={{ margin: 0 }}>{meta.subtitle}</p>
                <div className={styles.tgEventList}>
                  {items.map((item) => {
                    const checked = Boolean(events[item.key]);
                    return (
                      <label
                        key={item.key}
                        className={styles.tgEventItem}
                        data-checked={checked ? 'true' : undefined}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleEvent(item.key)}
                        />
                        <span className={styles.tgEventCopy}>
                          <strong>{item.label}</strong>
                          <span className="muted small">{item.description}</span>
                        </span>
                      </label>
                    );
                  })}
                </div>

                {priority === 'P2' && (
                  <div className={styles.tgP2Options}>
                    <label className={styles.retryToggleRow}>
                      <input
                        type="checkbox"
                        checked={includeCreationApiSuccess}
                        disabled={!canEdit || busy || !events.single_job}
                        onChange={(e) => {
                          setIncludeCreationApiSuccess(e.target.checked);
                          setStatus('');
                        }}
                      />
                      <span className="small">
                        同时推送 Creation API / 酒馆的高频成功生图（默认仅推失败以防刷屏）
                      </span>
                    </label>
                    <div className={styles.retryFieldsRow}>
                      <label className="field">
                        实例到期提前提醒（5 ~ 60 分钟）
                        <input
                          type="number"
                          min={5}
                          max={60}
                          step={1}
                          disabled={!canEdit || busy || !events.instance_expiry_billing}
                          value={expiryWarnText}
                          onChange={(e) => {
                            setExpiryWarnText(e.target.value);
                            setStatus('');
                          }}
                        />
                        {!validExpiryWarn && <span className="status error">请输入 5 ~ 60 之间的整数</span>}
                      </label>
                      <label className="field">
                        每日简报发送时刻（0 ~ 23 点，{saved.timezone || 'Asia/Shanghai'}）
                        <input
                          type="number"
                          min={0}
                          max={23}
                          step={1}
                          disabled={!canEdit || busy || !events.daily_report}
                          value={dailyHourText}
                          onChange={(e) => {
                            setDailyHourText(e.target.value);
                            setStatus('');
                          }}
                        />
                        {!validDailyHour && <span className="status error">请输入 0 ~ 23 之间的整数小时</span>}
                      </label>
                      <label className="field">
                        Drive 剩余空间预警阈值（1% ~ 50%）
                        <input
                          type="number"
                          min={1}
                          max={50}
                          step={1}
                          disabled={!canEdit || busy || !events.daily_report}
                          value={drivePctText}
                          onChange={(e) => {
                            setDrivePctText(e.target.value);
                            setStatus('');
                          }}
                        />
                        {!validDrivePct && <span className="status error">请输入 1 ~ 50 之间的整数</span>}
                      </label>
                    </div>
                  </div>
                )}
              </fieldset>
            );
          })}

          {enabled && !canEnableOrTest && (
            <p className={styles.retryInlineNote}>
              请先填写有效的 Bot Token 和 Chat ID 后再保存启用。
            </p>
          )}

          {canEdit && (
            <div className="actions">
              <button
                type="button"
                className="button small"
                disabled={busy || !dirty || !formValid}
                onClick={() => void save()}
              >
                <Save size={14} aria-hidden />
                {busy ? '保存中…' : '保存配置'}
              </button>
              <button
                type="button"
                className="button secondary small"
                disabled={testing || busy || !canEnableOrTest || !validThreadId}
                onClick={() => void sendTest()}
              >
                <Send size={14} aria-hidden />
                {testing ? '发送中…' : '发送测试消息'}
              </button>
              <button
                type="button"
                className="button ghost small"
                disabled={busy}
                onClick={resetDefaults}
              >
                <RotateCcw size={14} aria-hidden />
                恢复默认勾选
              </button>
            </div>
          )}

          <details
            className="foldable"
            open={logsOpen}
            onToggle={(e) => setLogsOpen(e.currentTarget.open)}
          >
            <summary>最近推送记录</summary>
            <div className="foldContent">
              <div className={styles.localStorageQuickActions} style={{ justifyContent: 'space-between' }}>
                <span className="muted small">展示最近 20 条推送任务（保留 7 天）。</span>
                <button
                  type="button"
                  className="button ghost small"
                  disabled={logsLoading}
                  onClick={() => void loadDeliveries()}
                >
                  <RefreshCw size={13} aria-hidden />
                  {logsLoading ? '刷新中…' : '刷新'}
                </button>
              </div>
              {deliveries.length === 0 ? (
                <p className="muted small" style={{ marginTop: 8 }}>
                  {logsLoading ? '正在加载推送记录…' : '暂无推送记录。'}
                </p>
              ) : (
                <ul className={styles.retryHistoryList}>
                  {deliveries.map((item) => (
                    <li key={item.id} className={styles.retryHistoryItem}>
                      <div className={styles.retryHistoryHead}>
                        <span className={styles.retryHistoryTitle}>
                          [{item.priority || 'INFO'}] {item.title || item.event_key || '通知'}
                        </span>
                        <span className={`pill ${item.state === 'sent' ? 'solid' : ''}`}>
                          {DELIVERY_STATE_LABEL[item.state] || item.state}
                        </span>
                      </div>
                      <p className={styles.retryHistoryDetail}>
                        {formatDateTime(item.created_at ?? undefined)}
                        {item.attempts > 1 ? ` · 重试 ${item.attempts} 次` : ''}
                        {item.last_error ? ` · ${item.last_error}` : ''}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </details>
        </div>
      )}
    </Card>
  );
}
