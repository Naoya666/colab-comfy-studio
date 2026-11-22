import { useState, type FormEvent } from 'react';
import { Triangle } from 'lucide-react';
import { apiMessage, login } from '../lib/gpu-api';

export function LoginScreen({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      await login(password);
      onLoggedIn();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="loginCard" onSubmit={submit}>
        <div className="brand">
          <span className="brandMark"><Triangle size={16} strokeWidth={1.6} aria-hidden /></span>
          <h1>Colab GPU Studio</h1>
        </div>
        <p className="muted">输入访问密码进入工作台</p>
        <label>
          访问密码
          <input
            autoFocus
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            placeholder="请输入密码"
            required
          />
        </label>
        {error && <p className="status error" role="alert">{error}</p>}
        <button className="button big" disabled={busy} type="submit">
          {busy ? '登录中…' : '进入工作台'}
        </button>
      </form>
    </main>
  );
}
