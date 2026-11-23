import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: {
    contents: `
      export {
        computeCategorySlots,
        displayModelTitles,
        findCatalogDuplicates,
        orderedTypeSegments,
        planBatchDeletion,
        sortModels,
      } from './src/lib/model-library.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    sourcefile: 'model-library-ui-helpers-entry.ts',
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
  computeCategorySlots,
  displayModelTitles,
  findCatalogDuplicates,
  orderedTypeSegments,
  planBatchDeletion,
  sortModels,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('orderedTypeSegments and computeCategorySlots produce matching deterministic slots', () => {
  const catalog = {
    items: [],
    types: { diffusion_models: '', loras: '', vae: '', checkpoints: '' },
    statistics: {
      count: 4,
      bytes: 1000,
      by_type: {
        vae: { count: 1, bytes: 100 },
        diffusion_models: { count: 2, bytes: 600 },
        loras: { count: 1, bytes: 300 },
        checkpoints: { count: 0, bytes: 0 },
      },
    },
  };
  const { entries, totalBytes } = orderedTypeSegments(catalog);
  assert.equal(totalBytes, 1000);
  assert.deepEqual(entries.map((e) => [e.type, e.slot]), [
    ['diffusion_models', 0],
    ['loras', 1],
    ['vae', 2],
  ]);
  assert.deepEqual(computeCategorySlots(catalog), {
    diffusion_models: 0,
    loras: 1,
    vae: 2,
  });
});

test('displayModelTitles deduplicates identical names and splits path-prefixed titles', () => {
  assert.deepEqual(
    displayModelTitles({
      display_name: 'RealESRGAN_x4plus_anime_6B.pth',
      filename: 'RealESRGAN_x4plus_anime_6B.pth',
    }),
    { primary: 'RealESRGAN_x4plus_anime_6B.pth', secondary: null },
  );

  assert.deepEqual(
    displayModelTitles({
      display_name: 'split_files/diffusion_models/anima-aesthetic-v1.1.safetensors',
      filename: 'anima-aesthetic-v1.1.safetensors',
      relative_path: 'split_files/diffusion_models/anima-aesthetic-v1.1.safetensors',
    }),
    {
      primary: 'anima-aesthetic-v1.1.safetensors',
      secondary: 'split_files/diffusion_models/anima-aesthetic-v1.1.safetensors',
    },
  );

  assert.deepEqual(
    displayModelTitles({
      display_name: 'Style:Ningen mame [Anima&NoobXL]',
      filename: 'ningenmame_v1_epoch22.safetensors',
    }),
    {
      primary: 'Style:Ningen mame [Anima&NoobXL]',
      secondary: 'ningenmame_v1_epoch22.safetensors',
    },
  );
});

test('sortModels pins active and failed models first in smart mode and sorts ready items by size desc', () => {
  const items = [
    { id: 'small-ready', display_name: 'Small', filename: 's.safetensors', model_type: 'loras', architecture: 'SDXL', state: 'ready', size: 100 },
    { id: 'big-ready', display_name: 'Big', filename: 'b.safetensors', model_type: 'diffusion_models', architecture: 'FLUX', state: 'ready', size: 5000 },
    { id: 'failed-item', display_name: 'Broken', filename: 'x.safetensors', model_type: 'vae', architecture: 'unknown', state: 'failed', size: 50 },
    { id: 'downloading-item', display_name: 'Active', filename: 'a.safetensors', model_type: 'loras', architecture: 'FLUX', state: 'downloading', size: 200 },
  ];
  const tasks = [{ id: 't1', model_id: 'downloading-item', state: 'downloading', bytes: 10, total: 200 }];

  assert.deepEqual(
    sortModels(items, tasks, 'smart').map((m) => m.id),
    ['downloading-item', 'failed-item', 'big-ready', 'small-ready'],
  );
  assert.deepEqual(
    sortModels(items, tasks, 'size_desc').map((m) => m.id),
    ['big-ready', 'downloading-item', 'small-ready', 'failed-item'],
  );
  assert.deepEqual(
    sortModels(items, tasks, 'name_asc').map((m) => m.id),
    ['downloading-item', 'big-ready', 'failed-item', 'small-ready'],
  );
});

test('findCatalogDuplicates detects exact SHA/URL/Drive/type+size matches and softer same-name collisions', () => {
  const sha = 'a'.repeat(64);
  const catalog = [
    {
      id: 'm1',
      display_name: 'Anima Aesthetic',
      filename: 'anima-aesthetic-v1.1.safetensors',
      model_type: 'diffusion_models',
      architecture: 'Anima',
      state: 'ready',
      size: 3900000000,
      sha256: sha,
      drive_file_id: 'drv-1',
      original_source_url: 'https://huggingface.co/org/anima/resolve/main/anima-aesthetic-v1.1.safetensors',
    },
    {
      id: 'm2',
      display_name: 'Old VAE',
      filename: 'ae.safetensors',
      model_type: 'vae',
      architecture: 'FLUX',
      state: 'ready',
      size: 335544320,
    },
  ];

  // Exact match by SHA-256 even with renamed file
  const bySha = findCatalogDuplicates({ filename: 'renamed.safetensors', expected_sha256: sha.toUpperCase() }, catalog);
  assert.equal(bySha.level, 'exact');
  assert.equal(bySha.exactMatches[0].id, 'm1');

  // Exact match by Drive file ID
  const byDrive = findCatalogDuplicates({ drive_file_id: 'drv-1', filename: 'other.safetensors' }, catalog);
  assert.equal(byDrive.level, 'exact');
  assert.equal(byDrive.exactMatches[0].id, 'm1');

  // Exact match by URL
  const byUrl = findCatalogDuplicates({
    filename: 'other.safetensors',
    original_source_url: 'https://huggingface.co/org/anima/resolve/main/anima-aesthetic-v1.1.safetensors',
  }, catalog);
  assert.equal(byUrl.level, 'exact');

  // Exact match by same type + same name + same size
  const bySizeAndName = findCatalogDuplicates({
    filename: 'AE.SAFETENSORS',
    model_type: 'vae',
    expected_size: 335544320,
  }, catalog);
  assert.equal(bySizeAndName.level, 'exact');
  assert.equal(bySizeAndName.exactMatches[0].id, 'm2');

  // Name-only collision when size or type differs
  const byNameOnly = findCatalogDuplicates({
    filename: 'ae.safetensors',
    model_type: 'vae',
    expected_size: 999999,
  }, catalog);
  assert.equal(byNameOnly.level, 'name');
  assert.equal(byNameOnly.nameMatches[0].id, 'm2');

  // Clean new file
  const clean = findCatalogDuplicates({
    filename: 'brand-new-lora.safetensors',
    model_type: 'loras',
    expected_size: 12345,
  }, catalog);
  assert.equal(clean.level, 'none');
});

test('planBatchDeletion handles active tasks, unmanaged fallback, shared Drive files, and reclaimable bytes', () => {
  const mActive = { id: 'm-active', display_name: 'Downloading', filename: 'd.safetensors', model_type: 'loras', architecture: 'FLUX', state: 'downloading', size: 1000, managed_file: true, drive_file_id: 'f-active', reference_count: 0 };
  const mUnmanaged = { id: 'm-unmanaged', display_name: 'External', filename: 'e.safetensors', model_type: 'loras', architecture: 'FLUX', state: 'ready', size: 2000, managed_file: false, drive_file_id: 'f-ext', reference_count: 1 };
  const mSharedA = { id: 'm-shared-a', display_name: 'Shared A', filename: 's1.safetensors', model_type: 'checkpoints', architecture: 'SDXL', state: 'ready', size: 4000, managed_file: true, drive_file_id: 'f-shared', reference_count: 0 };
  const mSharedB = { id: 'm-shared-b', display_name: 'Shared B', filename: 's2.safetensors', model_type: 'checkpoints', architecture: 'SDXL', state: 'ready', size: 4000, managed_file: true, drive_file_id: 'f-shared', reference_count: 2 };
  const mOutsideShared = { id: 'm-out-a', display_name: 'Out A', filename: 'o1.safetensors', model_type: 'vae', architecture: 'SDXL', state: 'ready', size: 500, managed_file: true, drive_file_id: 'f-out', reference_count: 0 };
  const mOutsideKeeper = { id: 'm-out-b', display_name: 'Out B', filename: 'o2.safetensors', model_type: 'vae', architecture: 'SDXL', state: 'ready', size: 500, managed_file: true, drive_file_id: 'f-out', reference_count: 0 };

  const all = [mActive, mUnmanaged, mSharedA, mSharedB, mOutsideShared, mOutsideKeeper];
  const selected = [mActive, mUnmanaged, mSharedA, mSharedB, mOutsideShared];
  const tasks = [{ id: 't1', model_id: 'm-active', state: 'downloading', bytes: 50, total: 1000 }];

  const plan = planBatchDeletion(selected, all, tasks, 'delete_file');
  assert.deepEqual(plan.activeBlocked.map((m) => m.id), ['m-active']);
  assert.deepEqual(plan.fallbackToIndexOnly.map((m) => m.id), ['m-unmanaged', 'm-out-a']);
  assert.deepEqual(plan.referencedModels.map((m) => m.id), ['m-unmanaged', 'm-shared-b']);
  // Only f-shared (4000) is actually physically reclaimed; f-shared is deleted ONCE after its sibling index is removed first.
  assert.equal(plan.reclaimableBytes, 4000);
  // Ordering check: deletePhysical === false entries come BEFORE deletePhysical === true
  const physicalFlags = plan.deletable.map((d) => [d.model.id, d.deletePhysical]);
  assert.deepEqual(physicalFlags, [
    ['m-unmanaged', false],
    ['m-shared-a', false],
    ['m-out-a', false],
    ['m-shared-b', true],
  ]);

  // Index-only mode never deletes physical files
  const indexPlan = planBatchDeletion(selected, all, tasks, 'remove_index');
  assert.equal(indexPlan.reclaimableBytes, 0);
  assert.ok(indexPlan.deletable.every((d) => d.deletePhysical === false));
});
