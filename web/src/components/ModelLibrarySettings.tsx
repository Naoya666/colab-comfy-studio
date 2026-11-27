import { useEffect, useState } from 'react';
import { HardDrive, KeyRound, X } from 'lucide-react';
import { apiFetch, apiMessage } from '../lib/gpu-api';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import { libraryApi } from '../lib/model-library';
import styles from '../gpu.module.css';

export type ModelStorageAccountOption = {
  id?: string;
  account_id?: string;
  alias?: string;
};

const SOURCE_META = [
  {
    key: 'huggingface',
    label: 'Hugging Face Token',
    placeholder: 'hf_...（访问受限仓库或提升速率时填写）',
  },
  {
    key: 'civitai',
    label: 'Civitai API Key',
    placeholder: '在 Civitai 账号设置中生成（下载需登录模型时填写）',
  },
] as const;

function ModelSourceSettingsPanel({ active }: { active: boolean }) {
  const [status, setStatus] = useState<Record<string, boolean>>({});
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [busyKey, setBusyKey] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!active) return;
    let alive = true;
    setLoading(true);
    setError('');
    setMessage('');
    void libraryApi<Record<string, boolean>>('/source-settings')
      .then((next) => {
        if (!alive) return;
        setStatus(next);
      })
      .catch((e) => {
        if (alive) setError(apiMessage(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => { alive = false; };
  }, [active]);

  async function saveToken(sourceKey: string, label: string) {
    const token = (values[sourceKey] || '').trim();
    if (!token || busyKey) return;
    setBusyKey(sourceKey);
    setError('');
    setMessage('');
    try {
      const next = await libraryApi<Record<string, boolean>>(`/source-settings/${sourceKey}`, { token }, 'PUT');
      setStatus(next);
      setValues((curr) => ({ ...curr, [sourceKey]: '' }));
      setMessage(`${label} 已加密保存`);
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setBusyKey('');
    }
  }

  async function clearToken(sourceKey: string, label: string) {
    if (busyKey) return;
    setBusyKey(sourceKey);
    setError('');
    setMessage('');
    try {
      const next = await libraryApi<Record<string, boolean>>(`/source-settings/${sourceKey}`, { token: '' }, 'PUT');
      setStatus(next);
      setValues((curr) => ({ ...curr, [sourceKey]: '' }));
      setMessage(`${label} 已清除`);
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setBusyKey('');
    }
  }

  return (
    <div className={styles.modelSourceList}>
      {SOURCE_META.map(({ key, label, placeholder }) => {
        const configured = Boolean(status[key]);
        const inputVal = values[key] || '';
        const isBusy = busyKey === key;
        return (
          <div key={key} className={styles.modelSourceItem}>
            <div className={styles.modelSourceHead}>
              <span>{label}</span>
              <span className={`pill ${configured ? 'solid' : ''}`}>
                {loading && !configured ? '读取中…' : configured ? '已配置（已脱敏）' : '未配置'}
              </span>
            </div>
            <label className="field">
              <input
                type="password"
                autoComplete="new-password"
                aria-label={label}
                disabled={Boolean(busyKey)}
                value={inputVal}
                placeholder={configured ? '已加密保存，输入新值可覆盖' : placeholder}
                onChange={(e) => {
                  setValues((curr) => ({ ...curr, [key]: e.target.value }));
                  setMessage('');
                  setError('');
                }}
              />
            </label>
            <div className="actions">
              <button
                type="button"
                className="button secondary small"
                disabled={Boolean(busyKey) || !inputVal.trim()}
                onClick={() => void saveToken(key, label)}
              >
                {isBusy ? '保存中…' : '保存凭据'}
              </button>
              <button
                type="button"
                className="button ghost small"
                disabled={Boolean(busyKey) || !configured}
                onClick={() => void clearToken(key, label)}
              >
                清除
              </button>
            </div>
          </div>
        );
      })}
      {message && <p className="status" role="status">{message}</p>}
      {error && <p className="status error" role="alert">{error}</p>}
    </div>
  );
}

function ModelStorageSettingsPanel({
  active,
  accounts: suppliedAccounts,
}: {
  active: boolean;
  accounts?: ModelStorageAccountOption[];
}) {
  const [accounts, setAccounts] = useState<ModelStorageAccountOption[]>(suppliedAccounts || []);
  const [account, setAccount] = useState('');
  const [savedAccount, setSavedAccount] = useState('');
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (suppliedAccounts) setAccounts(suppliedAccounts);
  }, [suppliedAccounts]);

  useEffect(() => {
    if (!active) return;
    let alive = true;
    setLoading(true);
    setError('');
    setMessage('');
    const reqs = suppliedAccounts
      ? Promise.all([Promise.resolve(suppliedAccounts), libraryApi<{ account_id?: string }>('/storage-settings')])
      : Promise.all([
          apiFetch<ModelStorageAccountOption[]>('/api/gpu/accounts'),
          libraryApi<{ account_id?: string }>('/storage-settings'),
        ]);
    void reqs
      .then(([a, s]) => {
        if (!alive) return;
        setAccounts(a);
        const current = s.account_id || '';
        setAccount(current);
        setSavedAccount(current);
      })
      .catch((e) => {
        if (alive) setError(apiMessage(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => { alive = false; };
  }, [active, suppliedAccounts]);

  async function saveStorage() {
    if (busy) return;
    setBusy(true);
    setError('');
    setMessage('');
    try {
      await libraryApi('/storage-settings', { account_id: account || null }, 'PUT');
      setSavedAccount(account);
      setMessage('存储账号已保存');
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.stack} style={{ gap: 10 }}>
      <label className="field">
        长期存储账号
        <select
          value={account}
          disabled={loading || busy}
          onChange={(e) => {
            setAccount(e.target.value);
            setMessage('');
            setError('');
          }}
        >
          <option value="">历史默认 Drive 账号</option>
          {accounts.map((a) => (
            <option key={a.account_id || a.id} value={a.account_id || a.id}>
              {a.alias || a.account_id || a.id}
            </option>
          ))}
        </select>
      </label>
      <p className="muted small">
        所有 GPU 账号共享此模型库。已有模型后禁止直接切换存储账号，防止索引指向另一块 Drive。
      </p>
      <div className="actions">
        <button
          type="button"
          className="button secondary small"
          disabled={loading || busy || account === savedAccount}
          onClick={() => void saveStorage()}
        >
          {busy ? '保存中…' : '保存存储账号'}
        </button>
      </div>
      {message && <p className="status" role="status">{message}</p>}
      {error && <p className="status error" role="alert">{error}</p>}
    </div>
  );
}

export function ModelSourceSettings({ id, defaultOpen = false }: { id?: string; defaultOpen?: boolean } = {}) {
  const [open, setOpen] = useState(defaultOpen);
  useEffect(() => { if (defaultOpen) setOpen(true); }, [defaultOpen]);
  return (
    <details id={id} className="foldable" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>模型下载源凭据</summary>
      <div className="foldContent">
        <ModelSourceSettingsPanel active={open} />
      </div>
    </details>
  );
}

export function ModelStorageSettings({
  id,
  defaultOpen = false,
  accounts,
}: {
  id?: string;
  defaultOpen?: boolean;
  accounts?: ModelStorageAccountOption[];
} = {}) {
  const [open, setOpen] = useState(defaultOpen);
  useEffect(() => { if (defaultOpen) setOpen(true); }, [defaultOpen]);
  return (
    <details id={id} className="foldable" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>模型库 Drive 存储账号</summary>
      <div className="foldContent">
        <ModelStorageSettingsPanel active={open} accounts={accounts} />
      </div>
    </details>
  );
}

export function ModelLibrarySettingsDialog({
  open,
  accounts,
  onClose,
}: {
  open: boolean;
  accounts?: ModelStorageAccountOption[];
  onClose: () => void;
}) {
  useBodyScrollLock(open);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <section
        className={`${styles.dialog} ${styles.modelSettingsDialog}`}
        role="dialog"
        aria-modal="true"
        aria-label="存储与下载源设置"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.cardHead}>
          <h2>存储与下载源设置</h2>
          <button type="button" className={styles.iconGhost} aria-label="关闭" onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        </div>

        <section className={styles.modelSettingsSection} aria-label="模型库 Drive 存储账号">
          <div className={styles.modelSettingsSectionHead}>
            <h3>
              <HardDrive size={15} strokeWidth={1.7} aria-hidden />
              模型库 Drive 存储账号
            </h3>
          </div>
          <ModelStorageSettingsPanel active={open} accounts={accounts} />
        </section>

        <section className={styles.modelSettingsSection} aria-label="模型下载源凭据">
          <div className={styles.modelSettingsSectionHead}>
            <h3>
              <KeyRound size={15} strokeWidth={1.7} aria-hidden />
              模型下载源凭据
            </h3>
          </div>
          <p className="muted small">
            下载受限的 Hugging Face 仓库或需登录的 Civitai 模型时自动携带凭据，服务端加密保存。
          </p>
          <ModelSourceSettingsPanel active={open} />
        </section>
      </section>
    </div>
  );
}
