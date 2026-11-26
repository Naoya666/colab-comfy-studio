import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: {
    contents: `
      export {
        defaultSearchQuery,
        dependencySourceUrl,
        pickUniqueResolvedFile,
        readStoredWorkflowInspect,
        reconcileDependencyWithCatalog,
        safeRelativePathFor,
        shortHash,
        sourceHostLabel,
        writeStoredWorkflowInspect,
      } from './src/lib/model-library.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    sourcefile: 'model-dep-inspector-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  define: { 'import.meta.env': '{}' },
  logLevel: 'silent',
});

const {
  defaultSearchQuery,
  dependencySourceUrl,
  pickUniqueResolvedFile,
  reconcileDependencyWithCatalog,
  safeRelativePathFor,
  shortHash,
  sourceHostLabel,
  readStoredWorkflowInspect,
  writeStoredWorkflowInspect,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('defaultSearchQuery strips subdirectories and weight extensions cleanly', () => {
  assert.equal(defaultSearchQuery('flux/krea2_turbo_fp8.safetensors'), 'krea2_turbo_fp8');
  assert.equal(defaultSearchQuery('loras\\film_v1.GGUF'), 'film_v1');
  assert.equal(defaultSearchQuery('4x-UltraSharp.pth'), '4x-UltraSharp');
  assert.equal(defaultSearchQuery('.safetensors'), '.safetensors');
});

test('dependencySourceUrl resolves direct source_hint and HF weight manifests', () => {
  assert.equal(
    dependencySourceUrl({
      id: 'd1', node_id: '1', node_type: 'UNETLoader', field: 'unet_name',
      reference: 'm.safetensors', model_type: 'diffusion_models', architecture: 'unknown',
      status: 'unresolved', required: true,
      source_hint: 'https://huggingface.co/org/repo/resolve/main/m.safetensors',
    }),
    'https://huggingface.co/org/repo/resolve/main/m.safetensors',
  );
  assert.equal(
    dependencySourceUrl({
      id: 'd2', node_id: '2', node_type: 'LoraLoader', field: 'lora_name',
      reference: 'sub/a.safetensors', model_type: 'loras', architecture: 'unknown',
      status: 'unresolved', required: true,
      weight: { repo_id: 'org/ds', repo_type: 'datasets', revision: 'main', filename: 'sub/a.safetensors' },
    }),
    'https://huggingface.co/datasets/org/ds/resolve/main/sub/a.safetensors',
  );
  assert.equal(
    dependencySourceUrl({
      id: 'd3', node_id: '3', node_type: 'VAELoader', field: 'vae_name',
      reference: 'ae.safetensors', model_type: 'vae', architecture: 'unknown',
      status: 'unresolved', required: true,
    }),
    null,
  );
});

test('reconcileDependencyWithCatalog tracks live downloads, source URLs, SHA, and same-name candidates', () => {
  const models = [
    {
      id: 'm1', display_name: 'Krea2', filename: 'krea2.safetensors',
      model_type: 'diffusion_models', architecture: 'FLUX', state: 'downloading',
      original_source_url: 'https://huggingface.co/org/krea2/resolve/main/krea2.safetensors',
    },
    {
      id: 'm2', display_name: 'Film LoRA', filename: 'film.safetensors',
      model_type: 'loras', architecture: 'FLUX', state: 'ready',
    },
  ];
  const tasks = [{ id: 't1', model_id: 'm1', state: 'downloading', bytes: 500, total: 1000, speed: 1024 ** 2 }];

  const live = reconcileDependencyWithCatalog(
    {
      id: 'dep1', node_id: '10', node_type: 'UNETLoader', field: 'unet_name',
      reference: 'krea2.safetensors', model_type: 'diffusion_models', architecture: 'unknown',
      status: 'unresolved', required: true,
      source_hint: 'https://huggingface.co/org/krea2/resolve/main/krea2.safetensors',
    },
    models,
    tasks,
  );
  assert.equal(live.bound?.id, 'm1');
  assert.equal(live.task?.id, 't1');
  assert.equal(live.effectiveStatus, 'downloading');

  // Unknown-type fallback still discovers same-name candidates in library
  const candidate = reconcileDependencyWithCatalog(
    {
      id: 'dep2', node_id: '20', node_type: 'CustomLoraNode', field: 'widgets[0]',
      reference: 'dir/film.safetensors', model_type: 'unknown', architecture: 'unknown',
      status: 'unresolved', required: true,
    },
    models,
    tasks,
  );
  assert.equal(candidate.bound, undefined);
  assert.equal(candidate.effectiveStatus, 'needs_confirmation');
  assert.deepEqual(candidate.candidates.map((m) => m.id), ['m2']);
});

test('pickUniqueResolvedFile handles exact single file, SHA match, filename match, and ambiguity', () => {
  const baseDep = {
    id: 'd1', node_id: '1', node_type: 'UNETLoader', field: 'unet_name',
    reference: 'sub/target.safetensors', model_type: 'diffusion_models', architecture: 'unknown',
    status: 'unresolved', required: true,
  };

  const single = pickUniqueResolvedFile(baseDep, [{ filename: 'only.safetensors' }]);
  assert.equal(single.ok, true);
  assert.equal(single.file.filename, 'only.safetensors');

  const byName = pickUniqueResolvedFile(baseDep, [
    { filename: 'other.safetensors' },
    { filename: 'target.safetensors' },
  ]);
  assert.equal(byName.ok, true);
  assert.equal(byName.file.filename, 'target.safetensors');

  const bySha = pickUniqueResolvedFile(
    { ...baseDep, expected_sha256: 'b'.repeat(64) },
    [
      { filename: 'v1.safetensors', expected_sha256: 'a'.repeat(64) },
      { filename: 'v2.safetensors', expected_sha256: 'b'.repeat(64) },
    ],
  );
  assert.equal(bySha.ok, true);
  assert.equal(bySha.file.filename, 'v2.safetensors');

  const ambiguous = pickUniqueResolvedFile(baseDep, [
    { filename: 'a.safetensors' },
    { filename: 'b.safetensors' },
  ]);
  assert.equal(ambiguous.ok, false);
  assert.match(ambiguous.reason, /2 个文件/);

  const empty = pickUniqueResolvedFile(baseDep, []);
  assert.equal(empty.ok, false);
});

test('safeRelativePathFor, shortHash, and sourceHostLabel helpers behave safely', () => {
  assert.equal(safeRelativePathFor('flux/krea2.safetensors', 'krea2.safetensors'), 'flux/krea2.safetensors');
  assert.equal(safeRelativePathFor('krea2.safetensors', 'krea2.safetensors'), undefined);
  assert.equal(safeRelativePathFor('../evil/krea2.safetensors', 'krea2.safetensors'), undefined);
  assert.equal(safeRelativePathFor('flux/other.safetensors', 'krea2.safetensors'), undefined);
  assert.equal(shortHash('0123456789abcdef0123456789abcdef'), '0123456789…abcdef');
  assert.equal(sourceHostLabel('https://huggingface.co/org/repo'), 'Hugging Face');
  assert.equal(sourceHostLabel('https://civitai.com/models/123'), 'Civitai');
  assert.equal(sourceHostLabel('https://civitai.red/models/2771337?modelVersionId=3270971'), 'Civitai');
  assert.equal(sourceHostLabel('https://www.civitai.red/api/download/models/3270971'), 'Civitai');
  assert.equal(sourceHostLabel('https://evil-civitai.red/models/1'), 'evil-civitai.red');
});

test('sessionStorage persistence retains workflow inspect state across tab switches and clears cleanly', () => {
  const store = new Map();
  globalThis.sessionStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => { store.set(k, String(v)); },
    removeItem: (k) => { store.delete(k); },
  };

  assert.equal(readStoredWorkflowInspect().result, null);
  writeStoredWorkflowInspect({
    open: true,
    fileName: 'demo.json',
    result: {
      format: 'visual',
      node_count: 2,
      dependencies: [],
      models_ready: true,
      warnings: [],
      summary: { total: 0, ready: 0, in_progress: 0, needs_confirmation: 0, unresolved: 0, manual_review: 0, with_source_hint: 0, optional: 0 },
    },
    importedMap: { dep1: 'model-1' },
  });
  const restored = readStoredWorkflowInspect();
  assert.equal(restored.open, true);
  assert.equal(restored.fileName, 'demo.json');
  assert.deepEqual(restored.importedMap, { dep1: 'model-1' });

  writeStoredWorkflowInspect({ open: false, fileName: '', result: null, importedMap: {} });
  assert.equal(store.size, 0);
});
