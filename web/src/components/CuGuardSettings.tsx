import { useEffect, useState, type FormEvent } from 'react';
import { apiMessage, saveCuGuard, type CuGuard } from '../lib/gpu-api';
import { Card } from './ui';
import styles from '../gpu.module.css';

export function CuGuardStatus({ guard }: { guard?: CuGuard }) {
  if (!guard) return null;
  return <div role="status" className={styles.stack}>
    {guard.status === 'ok' && <span className="pill">CU 熔断保护正常</span>}
    {guard.hard_tripped_accounts.length > 0 && <p className="status error">
      强制熔断 · {guard.message}
    </p>}
    {guard.soft_tripped && <p className={styles.cuWarning}>
      温和熔断 · {guard.daily_message}。本地次日 00:00 恢复，也可调高或清空今日总上限。
    </p>}
  </div>;
}

const asText = (value: number | null | undefined) => value == null ? '' : String(value);
const toLimit = (value: string) => value.trim() === '' ? null : Number(value);
const valid = (value: number | null) => value === null || (Number.isFinite(value) && value >= 0);

export function CuGuardSettings({ guard, canEdit, onSaved }: {
  guard?: CuGuard;
  canEdit: boolean;
  onSaved: () => Promise<void>;
}) {
  const daily = asText(guard?.daily_limit_cu);
  const floor = asText(guard?.min_balance_cu);
  const [draft, setDraft] = useState<{ daily: string; floor: string }>();
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [error, setError] = useState('');
  const fields = draft || { daily, floor };
  const dirty = fields.daily !== daily || fields.floor !== floor;
  useEffect(() => {
    if (draft && !dirty) setDraft(undefined);
  }, [draft, dirty]);

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const daily_limit_cu = toLimit(fields.daily);
    const min_balance_cu = toLimit(fields.floor);
    if (!valid(daily_limit_cu) || !valid(min_balance_cu)) {
      setError('限额必须为空或非负数。');
      return;
    }
    setBusy(true);
    setError('');
    setFeedback('');
    try {
      const result = await saveCuGuard({ daily_limit_cu, min_balance_cu });
      setDraft({ daily: asText(result.daily_limit_cu), floor: asText(result.min_balance_cu) });
      setFeedback('已保存，保护配置已生效。');
      await onSaved();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return <div id="cu-guard">
    <Card title="CU 熔断保护">
      <CuGuardStatus guard={guard} />
      <form onSubmit={(event) => void save(event)} className={styles.stack}>
        <label className="field">今日总上限（CU）
          <input type="number" min="0" step="any" placeholder="留空表示不限制" value={fields.daily}
            disabled={!canEdit || busy || !guard}
            onChange={(event) => { setFeedback(''); setDraft({ ...fields, daily: event.target.value }); }} />
          <small className="muted">本地自然日 00:00 重置。达到上限后暂停新任务，创作任务完成后关机，原生 ComfyUI 队列与同步排空后关机。</small>
        </label>
        <label className="field">余额底线（CU）
          <input type="number" min="0" step="any" placeholder="留空表示不限制" value={fields.floor}
            disabled={!canEdit || busy || !guard}
            onChange={(event) => { setFeedback(''); setDraft({ ...fields, floor: event.target.value }); }} />
          <small className="muted">账号余额降至底线或以下时，停止该账号的在线实例并暂停分配。</small>
        </label>
        {canEdit && <div className="actions">
          <button type="submit" className="button" disabled={!dirty || busy || !guard}>{busy ? '保存中…' : '保存保护配置'}</button>
          {dirty && !busy && !feedback && <span className="muted small">有未保存的修改</span>}
        </div>}
        {feedback && <p role="status" className="small">{feedback}</p>}
        {error && <p role="alert" className="status error">{error}</p>}
      </form>
    </Card>
  </div>;
}
