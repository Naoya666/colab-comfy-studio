import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { build } from 'esbuild';

const webRoot = fileURLToPath(new URL('..', import.meta.url));

const bundled = await build({
  stdin: {
    contents: `
      export {
        buildDailySeries,
        buildSmoothLinePath,
        computeYAxisScale,
        niceMaxValue,
        shiftCalendarDate,
      } from './src/lib/cu-chart.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'cu-chart-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  buildDailySeries,
  buildSmoothLinePath,
  computeYAxisScale,
  niceMaxValue,
  shiftCalendarDate,
} = await import(
  'data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64')
);

test('buildDailySeries zero-fills missing dates in ascending chronological order', () => {
  const { points, clipped } = buildDailySeries(
    [
      { date: '2026-11-22', actual_cu: 4.5, sessions: 2 },
      { date: '2026-11-19', actual_cu: 1.2, sessions: 1 },
    ],
    { today: '2026-11-22', startDate: '2026-11-19', endDate: '2026-11-22' },
  );
  assert.equal(clipped, false);
  assert.deepEqual(
    points.map((p) => [p.date, p.actual_cu, p.sessions]),
    [
      ['2026-11-19', 1.2, 1],
      ['2026-11-20', 0, 0],
      ['2026-11-21', 0, 0],
      ['2026-11-22', 4.5, 2],
    ],
  );
});

test('buildDailySeries clips long date ranges to trailing maxPoints window', () => {
  const { points, clipped, rangeStart, rangeEnd } = buildDailySeries(
    [{ date: '2026-09-01', actual_cu: 2, sessions: 1 }],
    { today: '2026-11-22', startDate: '2026-09-01', endDate: '2026-11-22', maxPoints: 30 },
  );
  assert.equal(clipped, true);
  assert.equal(points.length, 30);
  assert.equal(rangeEnd, '2026-11-22');
  assert.equal(rangeStart, shiftCalendarDate('2026-11-22', -29));
});

test('niceMaxValue rounds up to crisp chart tick bounds', () => {
  assert.equal(niceMaxValue(0), 1);
  assert.equal(niceMaxValue(1.3), 2);
  assert.equal(niceMaxValue(3.4), 5);
  assert.equal(niceMaxValue(7.8), 10);
  assert.equal(niceMaxValue(14), 20);
});

test('computeYAxisScale adapts to real data unless limit is toggled or approached', () => {
  const snug = computeYAxisScale(17.83, 300, null);
  assert.equal(snug.hasLimit, true);
  assert.equal(snug.autoLimitVisible, false);
  assert.equal(snug.showLimitInPlot, false);
  assert.equal(snug.yMax, 20);
  assert.deepEqual(snug.ticks, [0, 5, 10, 15, 20]);

  const toggledOn = computeYAxisScale(17.83, 300, true);
  assert.equal(toggledOn.showLimitInPlot, true);
  assert.equal(toggledOn.yMax, 400);
  assert.deepEqual(toggledOn.ticks, [0, 100, 200, 300, 400]);

  const nearLimit = computeYAxisScale(180, 300, null);
  assert.equal(nearLimit.autoLimitVisible, true);
  assert.equal(nearLimit.showLimitInPlot, true);
  assert.equal(nearLimit.yMax, 400);
});

test('buildSmoothLinePath builds monotone cubic SVG curves without undershooting baseline', () => {
  assert.deepEqual(buildSmoothLinePath([], 150), { linePath: '', areaPath: '' });
  assert.deepEqual(buildSmoothLinePath([{ x: 50, y: 80 }], 150), { linePath: '', areaPath: '' });

  const baselineY = 150;
  const coords = [
    { x: 40, y: 150 },
    { x: 120, y: 140 },
    { x: 200, y: 150 },
    { x: 280, y: 140 },
    { x: 360, y: 140 },
    { x: 440, y: 30 },
  ];
  const { linePath, areaPath } = buildSmoothLinePath(coords, baselineY);
  assert.match(linePath, /^M 40 150 C /);
  assert.match(areaPath, /L 440 150 L 40 150 Z$/);

  // Extract all C control-point Y values and confirm none dip below baselineY or overshoot peak.
  const cubicMatches = [...linePath.matchAll(/C ([\d.-]+) ([\d.-]+), ([\d.-]+) ([\d.-]+), ([\d.-]+) ([\d.-]+)/g)];
  assert.equal(cubicMatches.length, coords.length - 1);
  for (const m of cubicMatches) {
    const cp1y = Number(m[2]);
    const cp2y = Number(m[4]);
    assert.ok(cp1y <= baselineY && cp2y <= baselineY, `control point dipped below baseline: ${m[0]}`);
    assert.ok(cp1y >= 30 && cp2y >= 30, `control point overshot peak: ${m[0]}`);
  }
});

test('RuntimeTelemetryCard is wired to ComputePage and excluded from NativeComfyPage', () => {
  const compute = readFileSync(join(webRoot, 'src/pages/ComputePage.tsx'), 'utf8');
  const nativePage = readFileSync(join(webRoot, 'src/pages/NativeComfyPage.tsx'), 'utf8');
  assert.match(compute, /RuntimeTelemetryCard/);
  assert.doesNotMatch(nativePage, /RuntimeTelemetryCard/);
});

test('PWA manifest and icons exist and are linked from index.html', () => {
  const html = readFileSync(join(webRoot, 'index.html'), 'utf8');
  assert.match(html, /rel="manifest"\s+href="\/manifest\.webmanifest"/);
  assert.match(html, /rel="apple-touch-icon"/);
  assert.match(html, /apple-mobile-web-app-capable/);
  const manifest = JSON.parse(readFileSync(join(webRoot, 'public/manifest.webmanifest'), 'utf8'));
  assert.equal(manifest.display, 'standalone');
  for (const icon of manifest.icons) {
    assert.equal(existsSync(join(webRoot, 'public', icon.src.replace(/^\//, ''))), true);
  }
});

test('SettingsPage removes duplicate CU usage/guard cards, uses full-width appearance banner and two-column accounts', () => {
  const settings = readFileSync(join(webRoot, 'src/pages/SettingsPage.tsx'), 'utf8');
  const app = readFileSync(join(webRoot, 'src/App.tsx'), 'utf8');
  const accounts = readFileSync(join(webRoot, 'src/components/AccountDirectory.tsx'), 'utf8');
  const v020Css = readFileSync(join(webRoot, 'src/components/v020.module.css'), 'utf8');
  assert.doesNotMatch(settings, /CuGuardSettings/);
  assert.doesNotMatch(settings, /用量与额度/);
  assert.match(settings, /appearanceBanner/);
  assert.match(settings, /跟随系统/);
  assert.doesNotMatch(settings, /跟随手机系统/);
  assert.match(settings, /className=\{styles\.workspaceWide\}[\s\S]*title="Google 账号"/);
  assert.match(accounts, /styles\.accountGrid/);
  assert.match(v020Css, /\.accountGrid[\s\S]*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(app, /onEditGuard=\{\(\) => \{\s*navigate\('算力'\)/);
});

test('Studio Gallery shows workflows, supports workflow filtering, and GalleryViewer displays metadata everywhere', () => {
  const gallery = readFileSync(join(webRoot, 'src/pages/GalleryPage.tsx'), 'utf8');
  const viewer = readFileSync(join(webRoot, 'src/components/GalleryViewer.tsx'), 'utf8');
  const galleryState = readFileSync(join(webRoot, 'src/hooks/useGalleryState.ts'), 'utf8');
  const labels = readFileSync(join(webRoot, 'src/lib/labels.ts'), 'utf8');
  assert.match(gallery, /aria-label="按工作流筛选"/);
  assert.match(gallery, /workflowTitle/);
  assert.match(viewer, /<summary>作品信息<\/summary>/);
  assert.match(viewer, /<dt>工作流<\/dt>/);
  assert.doesNotMatch(viewer, /item\.kind === 'studio' && inlineViewer && viewerJobId/);
  assert.doesNotMatch(galleryState, /if \(!inlineViewer \|\| !viewerJobId/);
  assert.match(labels, /正向提示词/);
});

test('CreationApiPage uses CSS module stack and styles.css defines valid progress accent & theme vars', () => {
  const creationApi = readFileSync(join(webRoot, 'src/pages/CreationApiPage.tsx'), 'utf8');
  const stylesCss = readFileSync(join(webRoot, 'src/styles.css'), 'utf8');
  const gpuCss = readFileSync(join(webRoot, 'src/gpu.module.css'), 'utf8');
  assert.doesNotMatch(creationApi, /className="stack"/);
  assert.match(creationApi, /className=\{styles\.stack\}/);
  assert.doesNotMatch(stylesCss, /var\(--accent\)/);
  assert.doesNotMatch(stylesCss, /light-dark\(/);
  assert.doesNotMatch(gpuCss, /light-dark\(/);
});

test('API key management collapses in weak/open modes, CU UI uses compact popover & structured lists, and fullscreen uses dark lightbox', () => {
  const creationApi = readFileSync(join(webRoot, 'src/pages/CreationApiPage.tsx'), 'utf8');
  const nativeApi = readFileSync(join(webRoot, 'src/components/NativeApiSettings.tsx'), 'utf8');
  const compute = readFileSync(join(webRoot, 'src/pages/ComputePage.tsx'), 'utf8');
  const cuDetails = readFileSync(join(webRoot, 'src/components/CuSessionDetails.tsx'), 'utf8');
  const accounts = readFileSync(join(webRoot, 'src/components/AccountDirectory.tsx'), 'utf8');
  const viewer = readFileSync(join(webRoot, 'src/components/GalleryViewer.tsx'), 'utf8');
  const app = readFileSync(join(webRoot, 'src/App.tsx'), 'utf8');
  const gpuCss = readFileSync(join(webRoot, 'src/gpu.module.css'), 'utf8');

  assert.match(creationApi, /isStrictMode \?/);
  assert.match(creationApi, /历史专用 Key 仍在后台生效/);
  assert.match(nativeApi, /isStrictMode \?/);
  assert.match(nativeApi, /历史专用 Key 仍在后台生效/);

  assert.match(compute, /colab_connected !== false/);
  assert.match(compute, /Colab 账号余额/);

  assert.match(cuDetails, /CuDateRangePicker/);
  assert.match(cuDetails, /presetDates\(localTodayFallback\(\), '7d'\)/);
  assert.match(cuDetails, /styles\.cuList/);
  assert.match(cuDetails, /styles\.cuSessionRow/);
  assert.match(cuDetails, /styles\.cuDailyRow/);

  assert.match(accounts, /deleteGoogleAccount/);
  assert.match(accounts, /driveDependentsOf/);
  assert.match(accounts, /确认删除账号/);

  assert.match(viewer, /fullscreenchange/);
  assert.match(viewer, /退出全屏/);
  assert.match(viewer, /styles\.viewerStage/);
  assert.match(viewer, /styles\.viewerViewport/);
  assert.match(viewer, /zoomAtPoint/);
  assert.match(viewer, /toggleDoubleClickZoom/);
  assert.match(app, /if \(document\.fullscreenElement\) return;/);
  assert.match(gpuCss, /\.dialog:fullscreen,[\s\S]*\.assetInspector:fullscreen[\s\S]*--text: #ffffff/);
  assert.match(gpuCss, /\.dialog:fullscreen \.viewerViewport,[\s\S]*position: absolute;[\s\S]*inset: 0;/);
});
