import { useSyncExternalStore } from 'react';

// Keep in sync with the desktop shell breakpoint in styles.css.
const desktop = window.matchMedia('(min-width: 800px)');

function subscribe(onChange: () => void) {
  desktop.addEventListener('change', onChange);
  return () => desktop.removeEventListener('change', onChange);
}

export function useDesktopLayout() {
  return useSyncExternalStore(subscribe, () => desktop.matches);
}
