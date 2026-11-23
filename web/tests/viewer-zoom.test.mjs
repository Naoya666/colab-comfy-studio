import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const webRoot = fileURLToPath(new URL('..', import.meta.url));

const bundled = await build({
  stdin: {
    contents: `
      export {
        clampPan,
        clampScale,
        computeNativeScale,
        IDENTITY_TRANSFORM,
        MAX_ZOOM,
        MIN_ZOOM,
        panBy,
        toggleDoubleClickZoom,
        zoomAtPoint,
      } from './src/lib/viewer-zoom.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'viewer-zoom-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  clampPan,
  computeNativeScale,
  IDENTITY_TRANSFORM,
  MAX_ZOOM,
  panBy,
  toggleDoubleClickZoom,
  zoomAtPoint,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('zoomAtPoint keeps pointer anchor fixed and snaps back to identity at 1x', () => {
  const portraitBase = { width: 667, height: 1000 };
  const anchorX = 120;
  const anchorY = -180;
  const z1 = zoomAtPoint(IDENTITY_TRANSFORM, 2, anchorX, anchorY, portraitBase);
  assert.equal(z1.scale, 2);
  // The image point at (anchorX, anchorY) before zoom stays at (anchorX, anchorY) after zoom:
  // p = (anchorX - 0)/1 = 120 => p * 2 + z1.x = 120 => z1.x = -120
  assert.equal(z1.x, -120);
  assert.equal(z1.y, 180);

  const reset = zoomAtPoint(z1, 1, anchorX, anchorY, portraitBase);
  assert.deepEqual(reset, IDENTITY_TRANSFORM);
});

test('panning a corner detail to screen center then zooming in keeps that detail centered', () => {
  const portraitBase = { width: 600, height: 900 };
  // Start at 2x, pan so top-left corner (-300, -450 in base coords => -600, -900 scaled) sits at screen center (0, 0).
  const panned = panBy({ scale: 2, x: 0, y: 0 }, 600, 900, portraitBase);
  assert.equal(panned.x, 600);
  assert.equal(panned.y, 900);

  // Zoom deeper from 2x to 4x around screen center (0, 0) -> top-left corner stays centered at (1200, 1800).
  const deeper = zoomAtPoint(panned, 4, 0, 0, portraitBase);
  assert.equal(deeper.scale, 4);
  assert.equal(deeper.x, 1200);
  assert.equal(deeper.y, 1800);

  // Attempting to pan further past the corner is clamped so image never leaves the viewport.
  const over = clampPan(9999, -9999, 4, portraitBase);
  assert.equal(over.x, 1200);
  assert.equal(over.y, -1800);
});

test('toggleDoubleClickZoom centers clicked detail at 2.5x and resets on second double click', () => {
  const base = { width: 667, height: 1000 };
  const zoomed = toggleDoubleClickZoom(IDENTITY_TRANSFORM, 50, -100, base);
  assert.equal(zoomed.scale, 2.5);
  assert.equal(zoomed.x, -125);
  assert.equal(zoomed.y, 250);

  const back = toggleDoubleClickZoom(zoomed, 0, 0, base);
  assert.deepEqual(back, IDENTITY_TRANSFORM);
});

test('computeNativeScale calculates 1:1 pixel ratio for high-res portrait images and respects MAX_ZOOM', () => {
  // 1024x1536 portrait fitted to 667x1000 viewport => ~1.536x for 1:1 pixels
  const ratio = computeNativeScale({ width: 1024, height: 1536 }, { width: 667, height: 1000 });
  assert.ok(ratio !== null && Math.abs(ratio - 1.536) < 0.01);

  // Already fits at 1:1 or upscaled => null
  assert.equal(computeNativeScale({ width: 512, height: 512 }, { width: 512, height: 512 }), null);

  // Huge 12k image clamps to MAX_ZOOM
  assert.equal(computeNativeScale({ width: 12000, height: 12000 }, { width: 800, height: 800 }), MAX_ZOOM);
});
