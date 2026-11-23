import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: {
    contents: `
      export {
        groupJobsByBatch,
        partitionJobEntries,
        groupGalleryAssets,
        jobDateBucket,
        isDiagnosticAsset,
      } from './src/lib/history-grouping.ts';
      export { parseHashRoute, formatHashRoute } from './src/lib/route-state.ts';
      export { pickInitialProjectId, projectTitleById, isSystemJob } from './src/lib/project-label.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    sourcefile: 'ux-optimizations-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  groupJobsByBatch,
  partitionJobEntries,
  groupGalleryAssets,
  jobDateBucket,
  isDiagnosticAsset,
  parseHashRoute,
  formatHashRoute,
  pickInitialProjectId,
  projectTitleById,
  isSystemJob,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('groupJobsByBatch stacks multi-job batches and leaves single jobs flat', () => {
  const jobs = [
    { id: 'j1', project_id: 'p1', batch_id: 'b1', state: 'completed', workflow_key: 'flux', provider: 'colab', created_at: '2026-11-21T10:00:00Z' },
    { id: 'j2', project_id: 'p1', batch_id: 'b1', state: 'failed', workflow_key: 'flux', provider: 'colab', created_at: '2026-11-21T09:59:00Z' },
    { id: 'j3', project_id: 'p1', batch_id: 'single-batch', state: 'completed', workflow_key: 'flux', provider: 'mock', created_at: '2026-11-21T09:00:00Z' },
    { id: 'j4', project_id: 'p1', state: 'running', workflow_key: 'i2v', provider: 'colab', created_at: '2026-11-21T10:05:00Z' },
  ];
  const entries = groupJobsByBatch(jobs);
  assert.equal(entries.length, 3);
  assert.equal(entries[0].kind, 'batch');
  assert.equal(entries[0].jobs.length, 2);
  assert.equal(entries[0].doneCount, 1);
  assert.equal(entries[0].failedCount, 1);
  assert.equal(entries[0].allTerminal, true);
  assert.equal(entries[1].kind, 'single');
  assert.equal(entries[1].job.id, 'j3');
  assert.equal(entries[2].kind, 'single');
  assert.equal(entries[2].allTerminal, false);
});

test('partitionJobEntries pins active items and buckets history by date', () => {
  const now = new Date(2026, 10, 21, 15, 0, 0).getTime();
  const todayIso = new Date(2026, 10, 21, 12, 0, 0).toISOString();
  const yesterdayIso = new Date(2026, 10, 20, 12, 0, 0).toISOString();
  const olderIso = new Date(2026, 10, 10, 12, 0, 0).toISOString();

  assert.equal(jobDateBucket(todayIso, now), 'today');
  assert.equal(jobDateBucket(yesterdayIso, now), 'yesterday');
  assert.equal(jobDateBucket(olderIso, now), 'older');

  const entries = groupJobsByBatch([
    { id: 'run', project_id: 'p1', state: 'running', created_at: olderIso },
    { id: 't1', project_id: 'p1', state: 'completed', created_at: todayIso },
    { id: 'y1', project_id: 'p1', state: 'failed', created_at: yesterdayIso },
    { id: 'o1', project_id: 'p1', state: 'cancelled', created_at: olderIso },
  ]);
  const { active, buckets } = partitionJobEntries(entries, now);
  assert.equal(active.length, 1);
  assert.equal(active[0].job.id, 'run');
  assert.deepEqual(buckets.map((b) => b.key), ['today', 'yesterday', 'older']);

  const running = entries.find((entry) => entry.job.id === 'run');
  const limited = partitionJobEntries([...entries.filter((entry) => entry !== running), running], now, 1);
  assert.deepEqual(limited.active.map((entry) => entry.job.id), ['run']);
  assert.deepEqual(limited.buckets.flatMap((bucket) => bucket.entries.map((entry) => entry.job.id)), ['t1']);
});

test('groupGalleryAssets stacks same-batch assets and prefers favorited cover', () => {
  const assets = [
    { id: 'a1', project_id: 'p1', mime: 'image/png', batch_id: 'b1', favorite: false },
    { id: 'a2', project_id: 'p1', mime: 'image/png', batch_id: 'b1', favorite: true },
    { id: 'a3', project_id: 'p1', mime: 'application/json', filename: 'diag.json' },
  ];
  assert.equal(isDiagnosticAsset(assets[2]), true);
  assert.equal(isDiagnosticAsset(assets[0]), false);

  const stacks = groupGalleryAssets(assets);
  assert.equal(stacks.length, 2);
  assert.equal(stacks[0].items.length, 2);
  assert.equal(stacks[0].cover.id, 'a2');
  assert.equal(stacks[1].items.length, 1);
});

test('hash route parse and format preserve tab, sub-tab, and selection state', () => {
  const routes = [
    { tab: '创作', assetsSubTab: 'workflows', creationApiOpen: false, jobId: '', assetId: '' },
    { tab: '创作', assetsSubTab: 'workflows', creationApiOpen: true, jobId: '', assetId: '' },
    { tab: '任务', assetsSubTab: 'workflows', creationApiOpen: false, jobId: 'job-42', assetId: '' },
    { tab: '作品', assetsSubTab: 'workflows', creationApiOpen: false, jobId: '', assetId: 'asset-9' },
    { tab: '资产', assetsSubTab: 'models', creationApiOpen: false, jobId: '', assetId: '' },
    { tab: '算力', assetsSubTab: 'workflows', creationApiOpen: false, jobId: '', assetId: '' },
    { tab: '设置', assetsSubTab: 'workflows', creationApiOpen: false, jobId: '', assetId: '' },
  ];
  for (const state of routes) {
    const hash = formatHashRoute(state);
    assert.deepEqual(parseHashRoute(hash), state);
  }
});

test('project label helpers pick default project and detect system jobs', () => {
  const projects = [
    { id: 'p-new', title: '新项目', is_default: false },
    { id: 'p-def', title: '默认项目', is_default: true },
  ];
  assert.equal(pickInitialProjectId(projects, 'p-new'), 'p-new');
  assert.equal(pickInitialProjectId(projects, 'missing-id'), 'p-def');
  assert.equal(pickInitialProjectId(projects, ''), 'p-def');

  assert.equal(projectTitleById('p-def', projects), '默认项目');
  assert.equal(projectTitleById('native-comfy-owner', projects), '原生 ComfyUI（系统）');
  assert.equal(projectTitleById('system-smoke-owner', projects), 'GPU 可行性检查（系统）');

  assert.equal(isSystemJob({ system: true }), true);
  assert.equal(isSystemJob({ workflow_key: 'system/native-comfy' }), true);
  assert.equal(isSystemJob({ workflow_key: 'flux-dev', system: false }), false);
});
