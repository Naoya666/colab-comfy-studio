import type { Provider } from './gpu-api';
import type { GpuInstance } from '../components/AccountDirectory';
import type { RuntimeChoice } from '../pages/CreatePage';

const PROJECT_KEY = 'colab-gpu-studio-project';
const CREATE_RUNTIME_PREFS_KEY = 'studio-create-runtime-prefs';

export type SavedCreateRuntimePrefs = {
  runtime: Provider;
  accountId: string;
  lifecycle: RuntimeChoice['lifecycle'];
  retainIdleMinutes: number;
  maxRuntime: number;
};

export function record(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

export function uniqueBy<T>(items: T[], keyOf: (item: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = keyOf(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function instanceIdOf(instance: Pick<GpuInstance, 'id' | 'instance_id'>): string {
  return instance.id || instance.instance_id || '';
}

export function keyId(): string {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `gpu-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function shown(value: unknown): string {
  return value === null || value === undefined || value === ''
    ? '未知'
    : typeof value === 'object'
      ? JSON.stringify(value)
      : String(value);
}

export function assetIdsFromParams(params: Record<string, unknown>, validIds: Set<string>): string[] {
  return Object.values(params).filter((value): value is string => typeof value === 'string' && validIds.has(value));
}

export function extractPromptText(params?: Record<string, unknown>): string {
  if (!params) return '';
  if (typeof params.prompt === 'string' && params.prompt.trim()) return params.prompt;
  for (const [k, v] of Object.entries(params)) {
    if (/prompt|caption/i.test(k) && !/negative/i.test(k) && typeof v === 'string' && v.trim()) return v;
  }
  return '';
}

export function loadCreateRuntimePrefs(): SavedCreateRuntimePrefs {
  const fallback: SavedCreateRuntimePrefs = {
    runtime: 'mock',
    accountId: '',
    lifecycle: 'save',
    retainIdleMinutes: 15,
    maxRuntime: 30,
  };
  try {
    const raw = localStorage.getItem(CREATE_RUNTIME_PREFS_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return fallback;
    const runtime: Provider = parsed.runtime === 'colab' ? 'colab' : 'mock';
    const accountId = typeof parsed.accountId === 'string' ? parsed.accountId : '';
    const rawLifecycle = parsed.lifecycle;
    const lifecycle: RuntimeChoice['lifecycle'] = rawLifecycle === 'batch' || rawLifecycle === 'smart' || rawLifecycle === 'save'
      ? rawLifecycle
      : rawLifecycle === 'retain' ? 'smart' : 'save';
    const retainIdleMinutes = typeof parsed.retainIdleMinutes === 'number' && Number.isInteger(parsed.retainIdleMinutes) && parsed.retainIdleMinutes >= 1
      ? parsed.retainIdleMinutes : 15;
    const maxRuntime = typeof parsed.maxRuntime === 'number' && Number.isInteger(parsed.maxRuntime) && parsed.maxRuntime >= 1
      ? parsed.maxRuntime : 30;
    return { runtime, accountId, lifecycle, retainIdleMinutes, maxRuntime };
  } catch {
    return fallback;
  }
}

export function saveCreateRuntimePrefs(prefs: SavedCreateRuntimePrefs): void {
  try { localStorage.setItem(CREATE_RUNTIME_PREFS_KEY, JSON.stringify(prefs)); } catch { /* storage can be unavailable */ }
}

export function savedProjectId(): string {
  try { return localStorage.getItem(PROJECT_KEY) || ''; } catch { return ''; }
}

export function saveProjectId(id: string): void {
  try { if (id) localStorage.setItem(PROJECT_KEY, id); } catch { /* storage can be unavailable */ }
}
