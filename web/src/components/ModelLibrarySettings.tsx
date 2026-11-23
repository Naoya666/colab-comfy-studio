import { useEffect, useState } from 'react';
import { apiFetch, apiMessage } from '../lib/gpu-api';
import { libraryApi } from '../lib/model-library';
import styles from '../gpu.module.css';

export function ModelSourceSettings({ id, defaultOpen = false }: { id?: string; defaultOpen?: boolean } = {}) {
  const [status, setStatus] = useState<Record<string, boolean>>({});
  const [values, setValues] = useState<Record<string, string>>({});
  const [message, setMessage] = useState('');
  const [open, setOpen] = useState(defaultOpen);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => { if (defaultOpen) setOpen(true); }, [defaultOpen]);
  useEffect(() => {
    if (!open || loaded) return;
    let alive = true;
    void libraryApi<Record<string, boolean>>('/source-settings').then((next) => {
      if (!alive) return;
      setStatus(next);
      setLoaded(true);
    }).catch((e) => { if (alive) setMessage(apiMessage(e)); });
    return () => { alive = false; };
  }, [open, loaded]);
  return <details id={id} className="foldable" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}><summary>模型下载源凭据</summary><div className="foldContent">
    {['huggingface', 'civitai'].map((source) => <div key={source} className={styles.stack}>
      <label className="field">{source === 'huggingface' ? 'Hugging Face Token' : 'Civitai API Key'} · {status[source] ? '已配置（已脱敏）' : '未配置'}
        <input type="password" autoComplete="new-password" value={values[source] || ''} onChange={(e) => setValues({ ...values, [source]: e.target.value })} /></label>
      <div className="actions"><button className="button secondary small" onClick={async () => { try { setStatus(await libraryApi('/source-settings/' + source, { token: values[source] }, 'PUT')); setValues({ ...values, [source]: '' }); setMessage('凭据已加密保存'); } catch (e) { setMessage(apiMessage(e)); } }} disabled={!values[source]}>保存凭据</button>
      <button className="button secondary small" onClick={async () => { try { setStatus(await libraryApi('/source-settings/' + source, { token: '' }, 'PUT')); } catch (e) { setMessage(apiMessage(e)); } }}>清除</button></div>
    </div>)}
    {message && <p role="status">{message}</p>}
  </div></details>;
}

export function ModelStorageSettings({
  id,
  defaultOpen = false,
  accounts: suppliedAccounts,
}: {
  id?: string;
  defaultOpen?: boolean;
  accounts?: { id?: string; account_id?: string; alias?: string }[];
} = {}) {
  const [accounts, setAccounts] = useState<{ id?: string; account_id?: string; alias?: string }[]>(suppliedAccounts || []);
  const [account, setAccount] = useState('');
  const [message, setMessage] = useState('');
  const [open, setOpen] = useState(defaultOpen);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => { if (defaultOpen) setOpen(true); }, [defaultOpen]);
  useEffect(() => { if (suppliedAccounts) setAccounts(suppliedAccounts); }, [suppliedAccounts]);
  useEffect(() => {
    if (!open || loaded) return;
    let alive = true;
    const reqs = suppliedAccounts
      ? Promise.all([Promise.resolve(suppliedAccounts), libraryApi<{ account_id?: string }>('/storage-settings')])
      : Promise.all([apiFetch<{ id?: string; account_id?: string; alias?: string }[]>('/api/gpu/accounts'), libraryApi<{ account_id?: string }>('/storage-settings')]);
    void reqs.then(([a, s]) => {
      if (!alive) return;
      setAccounts(a);
      setAccount(s.account_id || '');
      setLoaded(true);
    }).catch((e) => { if (alive) setMessage(apiMessage(e)); });
    return () => { alive = false; };
  }, [open, loaded, suppliedAccounts]);
  return <details id={id} className="foldable" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}><summary>模型库 Drive 存储账号</summary><div className="foldContent"><label className="field">长期存储账号<select value={account} onChange={(e) => setAccount(e.target.value)}><option value="">历史默认 Drive 账号</option>{accounts.map((a) => <option key={a.account_id || a.id} value={a.account_id || a.id}>{a.alias || a.account_id || a.id}</option>)}</select></label><div className="actions"><button className="button secondary small" onClick={async () => { try { await libraryApi('/storage-settings', { account_id: account || null }, 'PUT'); setMessage('存储账号已保存'); } catch (e) { setMessage(apiMessage(e)); } }}>保存存储账号</button></div><p className="muted small">所有 GPU 账号共享此模型库。已有模型后禁止直接切换存储账号，防止索引指向另一块 Drive。</p>{message && <p role="status">{message}</p>}</div></details>;
}
