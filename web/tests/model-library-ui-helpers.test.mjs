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
        formatModelCreatedFull,
        formatModelCreatedShort,
        MODEL_SORT_STORAGE_KEY,
        orderedTypeSegments,
        planBatchDeletion,
        pruneSessionAddedModels,
        readStoredModelSortMode,
        recordSessionAddedModels,
        sortModels,
        writeStoredModelSortMode,
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
  formatModelCreatedFull,
  formatModelCreatedShort,
  MODEL_SORT_STORAGE_KEY,
  orderedTypeSegments,
  planBatchDeletion,
  pruneSessionAddedModels,
  readStoredModelSortMode,
  recordSessionAddedModels,
  sortModels,
  writeStoredModelSortMode,
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

test('sortModels in created_desc pins active items on top and orders the rest by newest created_at', () => {
  const items = [
    { id: 'old-ready', display_name: 'Old Ready', filename: 'old.safetensors', model_type: 'checkpoints', architecture: 'SDXL', state: 'ready', size: 9000, created_at: '2026-11-20T08:00:00+00:00' },
    { id: 'new-ready', display_name: 'New Small LoRA', filename: 'lora.safetensors', model_type: 'loras', architecture: 'FLUX', state: 'ready', size: 50, created_at: '2026-11-26T10:00:00+00:00' },
    { id: 'mid-failed', display_name: 'Mid Failed', filename: 'fail.safetensors', model_type: 'vae', architecture: 'unknown', state: 'failed', size: 300, created_at: '2026-11-24T12:00:00+00:00' },
    { id: 'active-old', display_name: 'Active Retried', filename: 'act1.safetensors', model_type: 'loras', architecture: 'FLUX', state: 'downloading', size: 100, created_at: '2026-11-18T00:00:00+00:00' },
    { id: 'active-new', display_name: 'Active Fresh', filename: 'act2.safetensors', model_type: 'loras', architecture: 'FLUX', state: 'pending', size: 100, created_at: '2026-11-26T11:00:00+00:00' },
    { id: 'no-date-b', display_name: 'Undated B', filename: 'b.safetensors', model_type: 'loras', architecture: 'unknown', state: 'ready', size: 200 },
    { id: 'no-date-a', display_name: 'Undated A', filename: 'a.safetensors', model_type: 'loras', architecture: 'unknown', state: 'ready', size: 200, created_at: 'invalid-date' },
  ];
  const tasks = [
    { id: 't1', model_id: 'active-old', state: 'downloading', bytes: 10, total: 100 },
    { id: 't2', model_id: 'active-new', state: 'pending', bytes: 0, total: 100 },
  ];

  assert.deepEqual(
    sortModels(items, tasks, 'created_desc').map((m) => m.id),
    ['active-new', 'active-old', 'new-ready', 'mid-failed', 'old-ready', 'no-date-a', 'no-date-b'],
  );
});

test('readStoredModelSortMode and writeStoredModelSortMode persist valid choices and fall back safely', () => {
  const bag = new Map();
  const fakeStorage = {
    getItem: (key) => (bag.has(key) ? bag.get(key) : null),
    setItem: (key, val) => { bag.set(key, String(val)); },
  };

  assert.equal(readStoredModelSortMode(fakeStorage), 'smart');

  writeStoredModelSortMode('created_desc', fakeStorage);
  assert.equal(bag.get(MODEL_SORT_STORAGE_KEY), 'created_desc');
  assert.equal(readStoredModelSortMode(fakeStorage), 'created_desc');

  bag.set(MODEL_SORT_STORAGE_KEY, 'unknown_mode');
  assert.equal(readStoredModelSortMode(fakeStorage), 'smart');

  const throwingStorage = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('quota'); },
  };
  assert.equal(readStoredModelSortMode(throwingStorage), 'smart');
  assert.doesNotThrow(() => writeStoredModelSortMode('created_desc', throwingStorage));
});

test('formatModelCreatedShort and formatModelCreatedFull format local dates cleanly', () => {
  const now = new Date(2026, 10, 26, 22, 50, 0);
  const today = new Date(2026, 10, 26, 9, 5, 8);
  const yesterday = new Date(2026, 10, 25, 18, 30, 0);
  const earlierThisYear = new Date(2026, 8, 3, 7, 45, 0);
  const lastYear = new Date(2025, 11, 31, 23, 59, 0);

  assert.equal(formatModelCreatedShort(today.toISOString(), now.getTime()), '今天 09:05');
  assert.equal(formatModelCreatedShort(yesterday.toISOString(), now.getTime()), '昨天 18:30');
  assert.equal(formatModelCreatedShort(earlierThisYear.toISOString(), now.getTime()), '09-03 07:45');
  assert.equal(formatModelCreatedShort(lastYear.toISOString(), now.getTime()), '2025-12-31');
  assert.equal(formatModelCreatedShort('', now.getTime()), '');
  assert.equal(formatModelCreatedShort('not-a-date', now.getTime()), '');

  assert.equal(formatModelCreatedFull(today.toISOString()), '2026-11-26 09:05:08');
  assert.equal(formatModelCreatedFull(null), '');
});

test('recordSessionAddedModels tracks sequential and batch additions with deduplication and race-safe pruning', () => {
  const m1 = { id: 'm1', display_name: 'First LoRA', filename: 'first.safetensors' };
  const m2 = { id: 'm2', display_name: 'Second VAE', filename: 'vae.safetensors' };
  const m3 = { id: 'm3', display_name: 'Third UNet', filename: 'unet.safetensors' };

  let list = recordSessionAddedModels([], [m1], 100);
  list = recordSessionAddedModels(list, [m2], 200);
  assert.deepEqual(list.map((e) => e.modelId), ['m2', 'm1']);

  // Batch add containing a new item and a re-added duplicate puts batch at front without duplicates
  list = recordSessionAddedModels(list, [m3, m1], 300);
  assert.deepEqual(list.map((e) => e.modelId), ['m3', 'm1', 'm2']);

  // Stale poll missing newly added m3 (not yet in seenCatalogIds) must NOT prune m3
  const seen = new Set(['m1', 'm2']);
  const staleCatalog = new Set(['m1', 'm2']);
  const afterStale = pruneSessionAddedModels(list, staleCatalog, seen);
  assert.equal(afterStale, list);

  // Once m3 has been observed in catalog, deleting m2 prunes only m2
  seen.add('m3');
  const afterDelete = pruneSessionAddedModels(list, new Set(['m1', 'm3']), seen);
  assert.deepEqual(afterDelete.map((e) => e.modelId), ['m3', 'm1']);
});
