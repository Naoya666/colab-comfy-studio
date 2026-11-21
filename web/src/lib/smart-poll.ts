import { useEffect, useRef } from 'react';

export interface SmartPollOptions {
  enabled: boolean;
  /** Whether there are active in-flight items needing faster updates. */
  active: boolean;
  activeMs?: number;
  idleMs?: number;
  /** Run immediately on mount / dependency change. Defaults to true. */
  immediate?: boolean;
}

/**
 * Visibility-aware adaptive polling:
 * - Pauses completely while the browser tab is hidden.
 * - Refreshes immediately when the tab becomes visible again.
 * - Uses activeMs when work is in flight, and backs off to idleMs when idle.
 */
export function useSmartPoll(callback: () => void | Promise<void>, options: SmartPollOptions): void {
  const { enabled, active, activeMs = 4000, idleMs = 25000, immediate = true } = options;
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let timer: number | undefined;

    const interval = active ? activeMs : idleMs;

    const clearTimer = () => {
      if (timer !== undefined) {
        window.clearTimeout(timer);
        timer = undefined;
      }
    };

    const schedule = () => {
      clearTimer();
      if (!alive || (typeof document !== 'undefined' && document.hidden)) return;
      timer = window.setTimeout(() => {
        void run();
      }, interval);
    };

    const run = async () => {
      clearTimer();
      if (!alive) return;
      try {
        await callbackRef.current();
      } finally {
        if (alive) schedule();
      }
    };

    const onVisibilityChange = () => {
      if (!alive) return;
      if (document.hidden) {
        clearTimer();
      } else {
        void run();
      }
    };

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibilityChange);
    }

    if (immediate) {
      if (typeof document === 'undefined' || !document.hidden) {
        void run();
      }
    } else {
      schedule();
    }

    return () => {
      alive = false;
      clearTimer();
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisibilityChange);
      }
    };
  }, [enabled, active, activeMs, idleMs, immediate]);
}
