import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const webRoot = fileURLToPath(new URL('..', import.meta.url));

const bundled = await build({
  stdin: {
    contents: `
      export {
        IMAGE_DOWNLOAD_MODE_KEY,
        isStrippableImageMime,
        cleanDownloadFilename,
        loadImageDownloadMode,
        saveImageDownloadMode,
        buildGalleryDownloadTarget,
        imageDownloadButtonLabel,
      } from './src/lib/download-prefs.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'download-prefs-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  IMAGE_DOWNLOAD_MODE_KEY,
  isStrippableImageMime,
  cleanDownloadFilename,
  loadImageDownloadMode,
  saveImageDownloadMode,
  buildGalleryDownloadTarget,
  imageDownloadButtonLabel,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

function createMemoryStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
  };
}

test('cleanDownloadFilename adds _clean suffix without duplicating underscores or existing suffixes', () => {
  assert.equal(cleanDownloadFilename('portrait.png'), 'portrait_clean.png');
  assert.equal(cleanDownloadFilename('ComfyUI_00001_.png'), 'ComfyUI_00001_clean.png');
  assert.equal(cleanDownloadFilename('shot-02-.webp'), 'shot-02-clean.webp');
  assert.equal(cleanDownloadFilename('角色立绘.png'), '角色立绘_clean.png');
  assert.equal(cleanDownloadFilename('my.v2.pic.jpeg'), 'my.v2.pic_clean.jpeg');
  assert.equal(cleanDownloadFilename('already_clean.png'), 'already_clean.png');
  assert.equal(cleanDownloadFilename('already-CLEAN.jpg'), 'already-CLEAN.jpg');
  assert.equal(cleanDownloadFilename('noext'), 'noext_clean');
  assert.equal(cleanDownloadFilename(''), 'image_clean');
});

test('isStrippableImageMime only accepts supported image types', () => {
  for (const mime of ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'IMAGE/PNG']) {
    assert.equal(isStrippableImageMime(mime), true);
  }
  for (const mime of ['video/mp4', 'video/webm', 'application/json', '', undefined, null]) {
    assert.equal(isStrippableImageMime(mime), false);
  }
});

test('download mode preference remembers user choice and falls back safely', () => {
  const storage = createMemoryStorage();
  assert.equal(loadImageDownloadMode(storage), 'original');

  saveImageDownloadMode('clean', storage);
  assert.equal(storage.getItem(IMAGE_DOWNLOAD_MODE_KEY), 'clean');
  assert.equal(loadImageDownloadMode(storage), 'clean');
  assert.equal(imageDownloadButtonLabel('clean'), '无元数据下载');

  saveImageDownloadMode('original', storage);
  assert.equal(loadImageDownloadMode(storage), 'original');
  assert.equal(imageDownloadButtonLabel('original'), '下载原图');

  // Unknown stored value or throwing storage falls back cleanly to original.
  assert.equal(loadImageDownloadMode(createMemoryStorage({ [IMAGE_DOWNLOAD_MODE_KEY]: 'invalid' })), 'original');
  const brokenStorage = {
    getItem() { throw new Error('blocked'); },
    setItem() { throw new Error('blocked'); },
  };
  assert.equal(loadImageDownloadMode(brokenStorage), 'original');
  assert.doesNotThrow(() => saveImageDownloadMode('clean', brokenStorage));
});

test('buildGalleryDownloadTarget produces strip_metadata URLs and _clean filenames for Studio and Drive images only', () => {
  const studioImg = {
    kind: 'studio',
    asset: { id: 'a1', mime: 'image/png', filename: 'ComfyUI_00001_.png' },
  };
  const studioOrig = buildGalleryDownloadTarget(studioImg, 'original');
  assert.equal(studioOrig.strippable, true);
  assert.equal(studioOrig.effectiveMode, 'original');
  assert.equal(studioOrig.href, '/api/assets/a1/download');
  assert.equal(studioOrig.filename, 'ComfyUI_00001_.png');

  const studioClean = buildGalleryDownloadTarget(studioImg, 'clean');
  assert.equal(studioClean.strippable, true);
  assert.equal(studioClean.effectiveMode, 'clean');
  assert.equal(studioClean.href, '/api/assets/a1/download?strip_metadata=true');
  assert.equal(studioClean.filename, 'ComfyUI_00001_clean.png');

  const studioCustomUrl = buildGalleryDownloadTarget({
    kind: 'studio',
    asset: { id: 'a2', mime: 'image/jpeg', filename: 'photo.jpg', download_url: '/api/assets/a2/download?v=2' },
  }, 'clean');
  assert.equal(studioCustomUrl.href, '/api/assets/a2/download?v=2&strip_metadata=true');
  assert.equal(studioCustomUrl.filename, 'photo_clean.jpg');

  const nativeImg = {
    kind: 'native',
    asset: { id: 'f1', mime: 'image/webp', filename: 'render.webp', content_url: '/api/gpu/native/outputs/f1/content' },
  };
  const nativeClean = buildGalleryDownloadTarget(nativeImg, 'clean');
  assert.equal(nativeClean.strippable, true);
  assert.equal(nativeClean.effectiveMode, 'clean');
  assert.equal(nativeClean.href, '/api/gpu/native/outputs/f1/content?download=true&strip_metadata=true');
  assert.equal(nativeClean.filename, 'render_clean.webp');

  // Videos must ignore clean mode and stay untouched.
  const studioVideo = {
    kind: 'studio',
    asset: { id: 'v1', mime: 'video/mp4', filename: 'clip.mp4' },
  };
  const studioVidTarget = buildGalleryDownloadTarget(studioVideo, 'clean');
  assert.equal(studioVidTarget.strippable, false);
  assert.equal(studioVidTarget.effectiveMode, 'original');
  assert.equal(studioVidTarget.href, '/api/assets/v1/download');
  assert.equal(studioVidTarget.filename, 'clip.mp4');

  const nativeVideo = {
    kind: 'native',
    asset: { id: 'nv1', mime: 'video/mp4', filename: 'clip.mp4', content_url: '/api/gpu/native/outputs/nv1/content' },
  };
  const nativeVidTarget = buildGalleryDownloadTarget(nativeVideo, 'clean');
  assert.equal(nativeVidTarget.strippable, false);
  assert.equal(nativeVidTarget.effectiveMode, 'original');
  assert.equal(nativeVidTarget.href, '/api/gpu/native/outputs/nv1/content?download=true');
  assert.equal(nativeVidTarget.filename, 'clip.mp4');
});

test('GalleryViewer wires remembered split download button in both inline and fullscreen views without touching CreateResults', () => {
  const viewer = readFileSync(join(webRoot, 'src/components/GalleryViewer.tsx'), 'utf8');
  const button = readFileSync(join(webRoot, 'src/components/GalleryDownloadButton.tsx'), 'utf8');
  const createResults = readFileSync(join(webRoot, 'src/components/CreateResults.tsx'), 'utf8');

  assert.match(viewer, /loadImageDownloadMode/);
  assert.match(viewer, /saveImageDownloadMode/);
  const matches = viewer.match(/<GalleryDownloadButton\b/g) || [];
  assert.equal(matches.length, 3, 'GalleryViewer should render GalleryDownloadButton in fullscreen, studio, and native action bars');

  assert.match(button, /role="menu"/);
  assert.match(button, /role="menuitemradio"/);
  assert.match(button, /无元数据下载/);
  assert.match(button, /下载原图/);
  assert.match(button, /addEventListener\('pointerdown'/);
  assert.match(button, /event\.key === 'Escape'/);

  // Create page results card must remain untouched.
  assert.doesNotMatch(createResults, /GalleryDownloadButton|strip_metadata|download-prefs/);
});
