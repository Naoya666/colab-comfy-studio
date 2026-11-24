import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { build } from 'esbuild';

const webRoot = fileURLToPath(new URL('..', import.meta.url));
const readSrc = (rel) => readFileSync(join(webRoot, rel), 'utf8');

const bundled = await build({
  stdin: {
    contents: `
      export {
        readStoredTheme,
        applyTheme,
        syncThemeColorMeta,
        initThemeSync,
      } from './src/lib/theme.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'theme-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const { readStoredTheme, applyTheme, syncThemeColorMeta, initThemeSync } = await import(
  'data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64')
);

test('theme helper persists theme selection and syncs theme-color metas for dark, light, and system modes', () => {
  const store = new Map();
  const metas = [
    { name: 'theme-color', content: '#000000' },
    { name: 'theme-color', content: '#ffffff' },
  ];
  let prefersLight = true;
  let changeHandler = null;

  const prevWindow = globalThis.window;
  const prevDocument = globalThis.document;
  globalThis.window = {
    localStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => { store.set(k, String(v)); },
    },
    matchMedia: () => ({
      get matches() { return prefersLight; },
      addEventListener: (_evt, cb) => { changeHandler = cb; },
    }),
  };
  globalThis.document = {
    documentElement: { dataset: {} },
    querySelectorAll: () => metas,
  };

  try {
    assert.equal(readStoredTheme(), 'dark');
    initThemeSync();
    assert.equal(globalThis.document.documentElement.dataset.theme, 'dark');
    assert.deepEqual(metas.map((m) => m.content), ['#000000', '#000000']);
    assert.equal(store.has('studio-theme'), false);

    applyTheme('light');
    assert.equal(store.get('studio-theme'), 'light');
    assert.equal(globalThis.document.documentElement.dataset.theme, 'light');
    assert.deepEqual(metas.map((m) => m.content), ['#ffffff', '#ffffff']);

    applyTheme('system');
    assert.equal(store.get('studio-theme'), 'system');
    assert.deepEqual(metas.map((m) => m.content), ['#ffffff', '#ffffff']);

    prefersLight = false;
    changeHandler?.();
    assert.deepEqual(metas.map((m) => m.content), ['#000000', '#000000']);

    syncThemeColorMeta('light');
    assert.deepEqual(metas.map((m) => m.content), ['#ffffff', '#ffffff']);
  } finally {
    globalThis.window = prevWindow;
    globalThis.document = prevDocument;
  }
});

test('global styles prevent mobile horizontal overflow, iOS input zoom, sticky clipping, and modal scroll bleed', () => {
  const styles = readSrc('src/styles.css');
  assert.match(styles, /html,\s*body\s*\{[\s\S]*overflow-x:\s*clip/);
  assert.doesNotMatch(styles, /body\s*\{[^}]*overflow-wrap:\s*anywhere/);
  assert.match(styles, /body\s*\{[^}]*overflow-wrap:\s*break-word/);
  assert.match(styles, /body\[data-modal-open="true"\]\s*\{[^}]*overflow:\s*hidden/);
  assert.match(styles, /\.topbar\s*\{[^}]*width:\s*100%/);
  assert.match(styles, /\.mobileSettings\[aria-current="page"\]/);
  assert.match(styles, /details\.foldable\s*\{[^}]*overflow:\s*visible/);
  assert.match(styles, /\.nativeRestartDialog\s*\{[\s\S]*overflow-y:\s*auto[\s\S]*overscroll-behavior:\s*contain/);
  assert.match(styles, /@media\s*\(max-width:\s*799px\)[\s\S]*font-size:\s*16px/);
  assert.match(styles, /\.desktopOnly\s*\{\s*display:\s*none\s*!important;/);
});

test('gpu.module.css fixes mobile stepper clipping, filter chip collapse, lightbox fallback, toast placement, and touch targets', () => {
  const css = readSrc('src/gpu.module.css');
  assert.match(css, /\.stepper\s*\{[^}]*flex:\s*none/);
  assert.match(css, /\.filterBar\s*\{[^}]*flex-wrap:\s*wrap/);
  assert.match(css, /\.chip\s*\{[^}]*white-space:\s*nowrap/);
  assert.match(css, /\.miniSelect\s*\{[^}]*max-width:\s*100%/);
  assert.match(css, /\.dialog\[data-lightbox="true"\]/);
  assert.match(css, /\.tileCheck::after\s*\{[^}]*inset:\s*-10px/);
  assert.match(css, /\.tipBubble\s*\{[^}]*position:\s*fixed/);
  assert.match(css, /\.toastStack\s*\{[^}]*top:\s*calc\(64px \+ var\(--safe-t\)\)/);
  assert.match(css, /\.cuDatePopover\s*\{[^}]*left:\s*0;\s*right:\s*0;\s*width:\s*auto/);
  assert.match(css, /\.drawer\s*\{[^}]*overscroll-behavior:\s*contain/);
  assert.match(css, /\.segmentedStackMobile/);
});

test('WorkflowGraphPreview and v020 stylesheets avoid undefined theme tokens and narrow-screen overflow', () => {
  const graphCss = readSrc('src/components/WorkflowGraphPreview.module.css');
  assert.doesNotMatch(graphCss, /var\(--surface\b/);
  assert.match(graphCss, /background:\s*var\(--card,\s*#101722\)/);
  assert.match(graphCss, /overscroll-behavior:\s*contain/);
  assert.match(graphCss, /var\(--safe-t\b/);
  assert.match(graphCss, /var\(--safe-b\b/);

  const v020Css = readSrc('src/components/v020.module.css');
  assert.match(v020Css, /\.toolbar\s*\{[^}]*flex-wrap:\s*wrap/);
  assert.match(v020Css, /\.sheet\s*\{[\s\S]*max-height:\s*88dvh[\s\S]*overflow-y:\s*auto/);
  assert.match(v020Css, /\.detailGrid\s*>\s*strong\s*\{[^}]*grid-column:\s*1\s*\/\s*-1/);
});

test('TSX components implement mobile lightbox fallback, scroll lock, touch dismissal, and proper styling hooks', () => {
  const viewer = readSrc('src/components/GalleryViewer.tsx');
  assert.match(viewer, /panel\.dataset\.lightbox/);
  assert.match(viewer, /useBodyScrollLock\(lightbox\)/);
  assert.match(viewer, /lastTapRef/);

  const app = readSrc('src/App.tsx');
  assert.match(app, /useBodyScrollLock\(/);
  assert.match(app, /data-lightbox="true"/);
  assert.match(app, /mobileSettings[\s\S]*?aria-current=\{tab === '设置' \? 'page' : undefined\}/);

  const galleryPage = readSrc('src/pages/GalleryPage.tsx');
  assert.doesNotMatch(galleryPage, /if\s*\(inspectedId === undefined\)\s*setSelectedId\(asset\.id\)/);

  const jobsPage = readSrc('src/pages/JobsPage.tsx');
  assert.match(jobsPage, /styles\.jobMachineBtn/);

  const historyDetails = readSrc('src/components/JobHistoryDetails.tsx');
  assert.doesNotMatch(historyDetails, /<dl className="kv">/);
  assert.match(historyDetails, /<dl className=\{styles\.kv\}>/);

  const creationApi = readSrc('src/pages/CreationApiPage.tsx');
  assert.match(creationApi, /<pre className=\{styles\.diagnostics\}>/);

  const modelLib = readSrc('src/pages/ModelLibraryPage.tsx');
  assert.match(modelLib, /addEventListener\('pointerdown'/);
  assert.match(modelLib, /openMenuUp/);

  const chart = readSrc('src/components/CuDailyChart.tsx');
  assert.doesNotMatch(chart, /translateX\(-100%\)/);
  assert.match(chart, /document\.addEventListener\('pointerdown'/);

  const graphPreview = readSrc('src/components/WorkflowGraphPreview.tsx');
  assert.match(graphPreview, /useBodyScrollLock\(true\)/);
  assert.match(graphPreview, /pinchRef/);

  const projectDialogs = readSrc('src/components/ProjectDialogs.tsx');
  assert.match(projectDialogs, /useBodyScrollLock\(open\)/);
  assert.match(projectDialogs, /<\/section>\s*<\/div>\s*<ConfirmDialog/);

  const createPage = readSrc('src/pages/CreatePage.tsx');
  assert.match(createPage, /aside=\{desktop \? <span className="muted small">Ctrl\+Enter 生成<\/span> : undefined\}/);

  const wizard = readSrc('src/components/NativeSetupWizard.tsx');
  assert.match(wizard, /styles\.segmentedStackMobile/);
});
