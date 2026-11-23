import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { build } from 'esbuild';

const webRoot = fileURLToPath(new URL('..', import.meta.url));

const bundled = await build({
  stdin: {
    contents: `
      export { crc32 } from './src/lib/png-workflow.ts';
      export { buildCatalogIndex, reconcileDependencyWithCatalog } from './src/lib/model-library.ts';
      export { normalisePreviewGraph } from './src/components/workflow-graph-preview-data.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'batch4-perf-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  define: { 'import.meta.env': '{}' },
  logLevel: 'silent',
});

const mod = await import(
  'data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64')
);

function slowBitwiseCrc32(bytes) {
  let crc = 0xffffffff;
  for (const b of bytes) {
    crc ^= b;
    for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

test('table-driven crc32 matches bit-by-bit reference exactly and runs fast on 4 MiB buffer', () => {
  const sample = new TextEncoder().encode('123456789');
  assert.equal(mod.crc32(sample), 0xcbf43926);
  assert.equal(mod.crc32(sample), slowBitwiseCrc32(sample));

  const buf = new Uint8Array(4 * 1024 * 1024);
  for (let i = 0; i < buf.length; i++) buf[i] = (i * 31 + 7) & 0xff;
  const t0 = performance.now();
  const fast = mod.crc32(buf);
  const fastMs = performance.now() - t0;
  const t1 = performance.now();
  const slow = slowBitwiseCrc32(buf);
  const slowMs = performance.now() - t1;
  assert.equal(fast, slow);
  assert.ok(fastMs < slowMs, `expected table crc32 (${fastMs.toFixed(1)}ms) < bitwise (${slowMs.toFixed(1)}ms)`);
});

test('buildCatalogIndex produces identical reconcileDependencyWithCatalog results to linear scan', () => {
  const models = [
    { id: 'm1', display_name: 'A', filename: 'a.safetensors', model_type: 'checkpoints', architecture: 'sdxl', state: 'ready', sha256: 'aa' },
    { id: 'm2', display_name: 'B', filename: 'b.safetensors', model_type: 'loras', architecture: 'sdxl', state: 'downloading', download_url: 'https://huggingface.co/x/y/resolve/main/b.safetensors' },
    { id: 'm3', display_name: 'C', filename: 'c.safetensors', model_type: 'vae', architecture: 'sdxl', state: 'ready' },
  ];
  const tasks = [{ id: 't1', model_id: 'm2', state: 'downloading', bytes: 10, total: 100 }];
  const deps = [
    { id: 'd1', node_id: '1', node_type: 'CheckpointLoaderSimple', field: 'ckpt_name', reference: 'a.safetensors', model_type: 'checkpoints', architecture: 'sdxl', status: 'unresolved', required: true, expected_sha256: 'AA' },
    { id: 'd2', node_id: '2', node_type: 'LoraLoader', field: 'lora_name', reference: 'b.safetensors', model_type: 'loras', architecture: 'sdxl', status: 'unresolved', required: true, source_hint: 'https://huggingface.co/x/y/resolve/main/b.safetensors' },
    { id: 'd3', node_id: '3', node_type: 'VAELoader', field: 'vae_name', reference: 'sub/c.safetensors', model_type: 'vae', architecture: 'sdxl', status: 'unresolved', required: true },
  ];
  const index = mod.buildCatalogIndex(models, tasks);
  for (const dep of deps) {
    const linear = mod.reconcileDependencyWithCatalog(dep, models, tasks);
    const indexed = mod.reconcileDependencyWithCatalog(dep, models, tasks, null, index);
    assert.deepEqual(indexed, linear);
  }
});

test('normalisePreviewGraph lays out multi-node graphs deterministically without duplicate UI edges', () => {
  const graph = {
    nodes: [
      { id: 1, type: 'Load', inputs: {} },
      { id: 2, type: 'MidA', inputs: {} },
      { id: 3, type: 'MidB', inputs: {} },
      { id: 4, type: 'Save', inputs: {} },
    ],
    links: [
      [10, 1, 0, 2, 0],
      [11, 1, 0, 3, 0],
      [12, 1, 0, 2, 1], // duplicate source->target pair
      [13, 2, 0, 4, 0],
    ],
  };
  const out = mod.normalisePreviewGraph(graph, 'ui');
  assert.equal(out.edges.length, 3);
  const byId = new Map(out.nodes.map((n) => [n.id, n]));
  assert.equal(byId.get('1').x, 40);
  assert.equal(byId.get('2').x, 300);
  assert.equal(byId.get('3').x, 300);
  assert.notEqual(byId.get('2').y, byId.get('3').y);
  assert.equal(byId.get('4').x, 560);
});

test('App code-splits non-initial routes, NativeComfyPage isolates timer, and CSS removes tile backdrop blur', () => {
  const app = readFileSync(join(webRoot, 'src/App.tsx'), 'utf8');
  const nativePage = readFileSync(join(webRoot, 'src/pages/NativeComfyPage.tsx'), 'utf8');
  const css = readFileSync(join(webRoot, 'src/gpu.module.css'), 'utf8');
  const vite = readFileSync(join(webRoot, 'vite.config.ts'), 'utf8');

  assert.match(app, /lazy\(\(\) => import\('\.\/pages\/ModelLibraryPage'\)/);
  assert.match(app, /lazy\(\(\) => import\('\.\/pages\/NativeComfyPage'\)/);
  assert.match(app, /lazy\(\(\) => import\('\.\/components\/GalleryViewer'\)/);
  assert.match(app, /window\.setTimeout\(\(\) => \{\s*writeWorkflowDraft/);
  assert.match(nativePage, /const NativeStageTimelines = memo\(/);
  assert.match(css, /\.tile\s*\{[^}]*content-visibility:\s*auto/);
  assert.doesNotMatch(css, /\.tileType\s*\{[^}]*backdrop-filter:/);
  assert.doesNotMatch(css, /\.stackBadge\s*\{[^}]*backdrop-filter:/);
  assert.match(vite, /manualChunks/);
});
