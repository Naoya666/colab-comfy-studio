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
        tokenizeGalleryQuery,
        galleryAssetSearchText,
        matchesGalleryQuery,
        galleryPromptPreview,
      } from './src/lib/gallery-search.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'gallery-search-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  tokenizeGalleryQuery,
  matchesGalleryQuery,
  galleryPromptPreview,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('tokenizeGalleryQuery splits, lowercases, and deduplicates multi-keyword input', () => {
  assert.deepEqual(tokenizeGalleryQuery(''), []);
  assert.deepEqual(tokenizeGalleryQuery('  Cyberpunk   雨夜  cyberpunk '), ['cyberpunk', '雨夜']);
});

test('matchesGalleryQuery supports multi-keyword AND search across prompts, negative prompt, model, seed, and metadata', () => {
  const asset = {
    id: 'a-1',
    project_id: 'p-1',
    mime: 'image/png',
    filename: 'ComfyUI_0001.png',
    workflow_key: 'krea2-moss-dual-t2i',
    model_id: 'flux-krea-2',
    resolved_seed: 987654321,
    prompt: 'masterpiece, cyberpunk neon street in pouring rain',
    main_prompt: 'cyberpunk neon street in pouring rain',
    negative_prompt: 'blurry, lowres, watermark',
    source_job_id: 'job-abc123',
  };

  assert.equal(matchesGalleryQuery(asset, '', 'Krea 双采', '默认项目'), true);
  assert.equal(matchesGalleryQuery(asset, 'cyberpunk rain', 'Krea 双采', '默认项目'), true);
  assert.equal(matchesGalleryQuery(asset, 'flux-krea-2 987654321', 'Krea 双采', '默认项目'), true);
  assert.equal(matchesGalleryQuery(asset, 'watermark', 'Krea 双采', '默认项目'), true);
  assert.equal(matchesGalleryQuery(asset, 'Krea 默认项目', 'Krea 双采', '默认项目'), true);
  assert.equal(matchesGalleryQuery(asset, 'cyberpunk snow', 'Krea 双采', '默认项目'), false);
});

test('galleryPromptPreview prefers main_prompt, indicates negative matches, and centers long query hits', () => {
  const asset = {
    id: 'a-2',
    project_id: 'p-1',
    mime: 'image/png',
    prompt: 'masterpiece, ultra detailed, cinematic lighting, ' + 'background filler '.repeat(6) + 'golden phoenix rising over jade temple',
    main_prompt: 'golden phoenix rising over jade temple',
    negative_prompt: 'extra fingers, watermark',
  };

  assert.equal(galleryPromptPreview(asset, ''), 'golden phoenix rising over jade temple');
  assert.equal(galleryPromptPreview(asset, 'watermark'), '负面词：extra fingers, watermark');

  const longAsset = {
    id: 'a-3',
    project_id: 'p-1',
    mime: 'image/png',
    prompt: 'very long prefix description about landscape mountains rivers clouds trees ' + 'and a glowing crystal lantern at midnight in the snowy forest',
  };
  const centered = galleryPromptPreview(longAsset, 'lantern', 56);
  assert.match(centered, /lantern/);
  assert.ok(centered.startsWith('…'));
});

test('gallery pages and settings page wire batch ZIP download and local storage governance cleanly', () => {
  const galleryPage = readFileSync(join(webRoot, 'src/pages/GalleryPage.tsx'), 'utf8');
  const nativeGallery = readFileSync(join(webRoot, 'src/components/NativeGalleryPanel.tsx'), 'utf8');
  const settingsPage = readFileSync(join(webRoot, 'src/pages/SettingsPage.tsx'), 'utf8');
  const governance = readFileSync(join(webRoot, 'src/components/LocalStorageGovernance.tsx'), 'utf8');
  const batchBtn = readFileSync(join(webRoot, 'src/components/BatchZipDownloadButton.tsx'), 'utf8');

  assert.match(galleryPage, /matchesGalleryQuery/);
  assert.match(galleryPage, /galleryPromptPreview/);
  assert.match(galleryPage, /<BatchZipDownloadButton\b/);
  assert.match(nativeGallery, /<BatchZipDownloadButton\b/);
  assert.match(batchBtn, /无元数据打包 ZIP/);
  assert.match(batchBtn, /原图打包 ZIP/);

  assert.match(settingsPage, /<LocalStorageGovernance\b/);
  assert.match(governance, /getLocalStorageOverview/);
  assert.match(governance, /previewLocalStorageCleanup/);
  assert.match(governance, /executeLocalStorageCleanup/);
  assert.match(governance, /vacuumLocalSqlite/);
  assert.match(governance, /createLocalSqliteSnapshot/);
});
