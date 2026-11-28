import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const webRoot = fileURLToPath(new URL('..', import.meta.url));
const readSrc = (rel) => readFileSync(join(webRoot, rel), 'utf8');

const bundled = await build({
  stdin: {
    contents: `
      export {
        formatModelCategoryBadge,
        splitModelName,
        sortPreparedModels,
        summarizePreparation,
        countExplicitDeclarations,
        formatMiB,
        formatKiB,
        formatDuration,
      } from './src/lib/native-model-prep.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'native-model-prep-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  formatModelCategoryBadge,
  splitModelName,
  sortPreparedModels,
  summarizePreparation,
  countExplicitDeclarations,
  formatMiB,
  formatKiB,
  formatDuration,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('formatModelCategoryBadge maps ComfyUI directories to short uppercase badges', () => {
  assert.deepEqual(formatModelCategoryBadge('diffusion_models'), {
    short: 'UNET',
    raw: 'diffusion_models',
    tone: 'core',
  });
  assert.deepEqual(formatModelCategoryBadge('text_encoders'), {
    short: 'CLIP',
    raw: 'text_encoders',
    tone: 'core',
  });
  assert.deepEqual(formatModelCategoryBadge('vae'), {
    short: 'VAE',
    raw: 'vae',
    tone: 'core',
  });
  assert.deepEqual(formatModelCategoryBadge('loras'), {
    short: 'LoRA',
    raw: 'loras',
    tone: 'lora',
  });
  assert.deepEqual(formatModelCategoryBadge('checkpoints'), {
    short: 'CKPT',
    raw: 'checkpoints',
    tone: 'core',
  });
  assert.deepEqual(formatModelCategoryBadge('controlnet'), {
    short: 'CN',
    raw: 'controlnet',
    tone: 'aux',
  });
  assert.deepEqual(formatModelCategoryBadge('upscale_models'), {
    short: 'UPSCALE',
    raw: 'upscale_models',
    tone: 'aux',
  });
  assert.deepEqual(formatModelCategoryBadge('custom_depth_models'), {
    short: 'CUSTOM_D',
    raw: 'custom_depth_models',
    tone: 'aux',
  });
});

test('splitModelName separates nested folder path from filename', () => {
  assert.deepEqual(splitModelName('qwen_image_vae.safetensors'), {
    dir: '',
    file: 'qwen_image_vae.safetensors',
  });
  assert.deepEqual(splitModelName('styles/krea/aurora.safetensors'), {
    dir: 'styles/krea/',
    file: 'aurora.safetensors',
  });
});

test('sortPreparedModels stably prioritizes failed > downloading > pending > ready', () => {
  const models = [
    { category: 'vae', name: 'a.safetensors', state: 'ready' },
    { category: 'loras', name: 'b.safetensors', state: 'pending' },
    { category: 'loras', name: 'c.safetensors', state: 'downloading' },
    { category: 'diffusion_models', name: 'd.safetensors', state: 'failed' },
    { category: 'loras', name: 'e.safetensors', state: 'ready' },
  ];
  const sorted = sortPreparedModels(models);
  assert.deepEqual(
    sorted.map((m) => `${m.state}:${m.name}`),
    [
      'failed:d.safetensors',
      'downloading:c.safetensors',
      'pending:b.safetensors',
      'ready:a.safetensors',
      'ready:e.safetensors',
    ],
  );
});

test('summarizePreparation aggregates counts, sources, bytes, duration, and headlines', () => {
  const readySummary = summarizePreparation({
    state: 'ready',
    models: [
      { category: 'diffusion_models', name: 'unet.safetensors', state: 'ready', source: 'local', seconds: 0 },
      { category: 'loras', name: 'style.safetensors', state: 'ready', source: 'local', seconds: 0 },
    ],
  });
  assert.equal(readySummary.total, 2);
  assert.equal(readySummary.ready, 2);
  assert.equal(readySummary.sources.local, 2);
  assert.equal(readySummary.percent, 100);
  assert.equal(readySummary.elapsedSeconds, 0);
  assert.equal(readySummary.headline, '2 / 2 个模型已就绪');

  const downloadingSummary = summarizePreparation({
    state: 'preparing',
    models: [
      { category: 'vae', name: 'vae.safetensors', state: 'ready', source: 'local', bytes: 100, total: 100, seconds: 0 },
      { category: 'loras', name: 'l1.safetensors', state: 'downloading', source: 'drive', bytes: 50, total: 100, seconds: 1.5 },
      { category: 'loras', name: 'l2.safetensors', state: 'pending' },
    ],
  });
  assert.equal(downloadingSummary.ready, 1);
  assert.equal(downloadingSummary.downloading, 1);
  assert.equal(downloadingSummary.pending, 1);
  assert.equal(downloadingSummary.bytes, 150);
  assert.equal(downloadingSummary.totalBytes, 200);
  assert.equal(downloadingSummary.percent, 75);
  assert.equal(downloadingSummary.elapsedSeconds, 1.5);
  assert.equal(downloadingSummary.headline, '正在准备模型（1 / 3）');

  const failedSummary = summarizePreparation({
    state: 'failed',
    models: [
      { category: 'vae', name: 'vae.safetensors', state: 'ready', source: 'local' },
      { category: 'loras', name: 'missing.safetensors', state: 'failed', error: 'Not found' },
    ],
  });
  assert.equal(failedSummary.failed, 1);
  assert.equal(failedSummary.headline, '1 / 2 就绪，1 个失败');

  const emptySummary = summarizePreparation({ state: 'ready', models: [] });
  assert.equal(emptySummary.headline, '此工作流未识别出模型文件，可补充显式依赖。');
});

test('countExplicitDeclarations validates JSON arrays and ignores blank text', () => {
  assert.equal(countExplicitDeclarations('   '), 0);
  assert.equal(countExplicitDeclarations('[{"category":"loras","name":"a.safetensors"}]'), 1);
  assert.equal(countExplicitDeclarations('{"not":"an array"}'), null);
  assert.equal(countExplicitDeclarations('[broken'), null);
});

test('formatting helpers render compact sizes and durations', () => {
  assert.equal(formatKiB(512), '512 B');
  assert.equal(formatKiB(32 * 1024), '32.0 KiB');
  assert.equal(formatMiB(150 * 1024 * 1024), '150.0 MiB');
  assert.equal(formatMiB(2 * 1024 * 1024 * 1024), '2.00 GiB');
  assert.equal(formatDuration(0), '0s');
  assert.equal(formatDuration(4.2), '4.2s');
  assert.equal(formatDuration(75), '1m 15s');
});

test('NativeModelPreparation uses custom progress bars and internal scrollable list', () => {
  const tsx = readSrc('src/components/NativeModelPreparation.tsx');
  const css = readSrc('src/components/NativeModelPreparation.module.css');

  assert.equal(tsx.includes('<progress'), false);
  assert.match(tsx, /role="progressbar"/);
  assert.match(tsx, /formatModelCategoryBadge/);
  assert.match(tsx, /sortPreparedModels/);
  assert.match(css, /\.modelList\s*\{[^}]*max-height:\s*\d+px;[^}]*overflow-y:\s*auto;/s);
});
