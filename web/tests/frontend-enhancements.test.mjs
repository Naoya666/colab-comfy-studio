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
      export { buildDailySeries, niceMaxValue, shiftCalendarDate } from './src/lib/cu-chart.ts';
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

const { buildDailySeries, niceMaxValue, shiftCalendarDate } = await import(
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
