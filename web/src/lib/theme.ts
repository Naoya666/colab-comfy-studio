export type ThemeMode = 'dark' | 'light' | 'system';

const LIGHT_BG = '#ffffff';
const DARK_BG = '#000000';

function resolveEffectiveDark(mode: ThemeMode): boolean {
  if (mode === 'light') return false;
  if (mode === 'dark') return true;
  if (typeof window === 'undefined' || !window.matchMedia) return true;
  return !window.matchMedia('(prefers-color-scheme: light)').matches;
}

export function syncThemeColorMeta(mode: ThemeMode): void {
  if (typeof document === 'undefined') return;
  const color = resolveEffectiveDark(mode) ? DARK_BG : LIGHT_BG;
  const metas = document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]');
  if (metas.length === 0) {
    const meta = document.createElement('meta');
    meta.name = 'theme-color';
    meta.content = color;
    document.head.appendChild(meta);
    return;
  }
  metas.forEach((meta) => {
    meta.content = color;
  });
}

export function readStoredTheme(): ThemeMode {
  if (typeof window === 'undefined') return 'dark';
  try {
    const saved = window.localStorage.getItem('studio-theme');
    return saved === 'light' || saved === 'system' ? saved : 'dark';
  } catch {
    return 'dark';
  }
}

export function applyTheme(mode: ThemeMode, persist = true): void {
  if (persist && typeof window !== 'undefined') {
    try {
      window.localStorage.setItem('studio-theme', mode);
    } catch {
      // Ignore storage failures in private browsing modes.
    }
  }
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = mode;
  }
  syncThemeColorMeta(mode);
}

export function initThemeSync(): void {
  const mode = readStoredTheme();
  applyTheme(mode, false);
  if (typeof window === 'undefined' || !window.matchMedia) return;
  const mql = window.matchMedia('(prefers-color-scheme: light)');
  const onChange = () => {
    if (readStoredTheme() === 'system') {
      syncThemeColorMeta('system');
    }
  };
  mql.addEventListener('change', onChange);
}
