import { useEffect } from 'react';

let activeLockCount = 0;

export function useBodyScrollLock(locked: boolean): void {
  useEffect(() => {
    if (!locked || typeof document === 'undefined') return;
    activeLockCount += 1;
    document.body.dataset.modalOpen = 'true';
    return () => {
      activeLockCount = Math.max(0, activeLockCount - 1);
      if (activeLockCount === 0) {
        delete document.body.dataset.modalOpen;
      }
    };
  }, [locked]);
}
