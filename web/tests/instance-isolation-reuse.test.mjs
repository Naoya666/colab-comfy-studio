import assert from 'node:assert/strict';
import { test, before } from 'node:test';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { tmpdir } from 'node:os';
import { transform } from 'esbuild';

let specOrder;
let submission;

async function loadTs(relPath, tag) {
  const source = readFileSync(join(process.cwd(), relPath), 'utf8');
  const compiled = await transform(source, {
    loader: 'ts',
    format: 'esm',
    target: 'es2020',
    sourcemap: false,
  });
  const out = join(tmpdir(), `${tag}-${Date.now()}-${Math.random()}.mjs`);
  writeFileSync(out, compiled.code);
  return import(`${pathToFileURL(out).href}?t=${Math.random()}`);
}

before(async () => {
  specOrder = await loadTs('src/lib/gpu-spec-order.ts', 'gpu-spec-order');
  submission = await loadTs('src/lib/instance-submission.ts', 'instance-submission');
});

test('gpu-spec-order enforces default low-to-high ranking, L4 highmem normalization, and unranked safety', () => {
  const T4 = { variant: 'VARIANT_GPU', accelerator: 'T4', shape: 'SHAPE_STANDARD' };
  const T4_HM = { variant: 'VARIANT_GPU', accelerator: 'T4', shape: 'SHAPE_HIGHMEM' };
  const L4 = { variant: 'VARIANT_GPU', accelerator: 'L4', shape: 'SHAPE_STANDARD' };
  const L4_HM = { variant: 'VARIANT_GPU', accelerator: 'L4', shape: 'SHAPE_HIGHMEM' };
  const A100 = { variant: 'VARIANT_GPU', accelerator: 'A100', shape: 'SHAPE_STANDARD' };
  const A100_HM = { variant: 'VARIANT_GPU', accelerator: 'A100', shape: 'SHAPE_HIGHMEM' };
  const B200 = { variant: 'VARIANT_GPU', accelerator: 'B200', shape: 'SHAPE_STANDARD' };

  assert.equal(specOrder.compareGpuSpecs(T4, T4_HM), 'lower');
  assert.equal(specOrder.compareGpuSpecs(T4_HM, L4), 'lower');
  assert.equal(specOrder.compareGpuSpecs(L4_HM, L4), 'equal');
  assert.equal(specOrder.compareGpuSpecs(L4, A100), 'lower');
  assert.equal(specOrder.compareGpuSpecs(A100, A100_HM), 'lower');
  assert.equal(specOrder.compareGpuSpecs(A100, L4), 'higher');
  assert.equal(specOrder.compareGpuSpecs(B200, B200), 'equal');
  assert.equal(specOrder.compareGpuSpecs(B200, L4), 'unknown');

  assert.equal(specOrder.isBelowWorkflowDefault(T4, L4), true);
  assert.equal(specOrder.isBelowWorkflowDefault(L4, L4), false);
  assert.equal(specOrder.isBelowWorkflowDefault(A100, L4), false);
  assert.equal(specOrder.isBelowWorkflowDefault(B200, L4), false);
});

test('instanceReusable excludes native and ephemeral pools', () => {
  assert.equal(submission.instanceReusable({ id: 'c1', instance_pool: 'creation', state: 'retained_idle' }), true);
  assert.equal(submission.instanceReusable({ id: 'n1', instance_pool: 'native', state: 'retained_idle' }), false);
  assert.equal(submission.instanceReusable({ id: 'e1', instance_pool: 'ephemeral', state: 'retained_idle' }), false);
});

test('evaluateLifetimeShortening applies dual criteria A (policy) and B (actual expiry)', () => {
  const now = 10_000;

  // Criterion A1: smart -> save always shortens
  const a1 = submission.evaluateLifetimeShortening(
    { retention_mode: 'smart', retain_idle_minutes: 30, idle_deadline_at: now + 300, state: 'retained_idle' },
    'save',
    15,
    now,
  );
  assert.equal(a1.shortens, true);
  assert.ok(a1.reasons.length >= 1);

  // Criterion A2: smart 30m -> smart 15m shortens even if remaining time is only 10m
  const a2 = submission.evaluateLifetimeShortening(
    { retention_mode: 'smart', retain_idle_minutes: 30, idle_deadline_at: now + 600, state: 'retained_idle' },
    'smart',
    15,
    now,
  );
  assert.equal(a2.shortens, true);

  // Criterion B: retain_idle_minutes unchanged/missing, but idle_deadline_at still has 25m remaining while new is 15m
  const b1 = submission.evaluateLifetimeShortening(
    { retention_mode: 'smart', retain_idle_minutes: 15, idle_deadline_at: now + 25 * 60, state: 'retained_idle' },
    'smart',
    15,
    now,
  );
  assert.equal(b1.shortens, true);

  // Criterion B on busy instance: estimated expiry uses now + old_retain_idle_minutes
  const bBusy = submission.evaluateLifetimeShortening(
    { retention_mode: 'retain', retain_idle_minutes: 30, state: 'busy', is_busy: true },
    'smart',
    10,
    now,
  );
  assert.equal(bBusy.shortens, true);

  // Extending retention (smart 15m with 10m left -> smart 30m) does NOT shorten
  const okExtend = submission.evaluateLifetimeShortening(
    { retention_mode: 'smart', retain_idle_minutes: 15, idle_deadline_at: now + 600, state: 'retained_idle' },
    'smart',
    30,
    now,
  );
  assert.equal(okExtend.shortens, false);
  assert.deepEqual(okExtend.reasons, []);
});

test('shouldBypassSkipConfirm strictly overrides skipConfirmSig for forced scenarios', () => {
  for (const decision of [
    'unsupported_spec',
    'confirm_shorten',
    'confirm_busy_same_spec',
    'confirm_higher_spec',
    'confirm_lower_spec',
  ]) {
    assert.equal(submission.shouldBypassSkipConfirm({ decision }), true, `${decision} must bypass skipConfirmSig`);
  }
  assert.equal(submission.shouldBypassSkipConfirm({ decision: 'reuse_direct', shortens_lifetime: true }), true);
  assert.equal(submission.shouldBypassSkipConfirm({ decision: 'create_new', force_confirm: true }), true);
  assert.equal(submission.shouldBypassSkipConfirm({ decision: 'reuse_direct', shortens_lifetime: false, force_confirm: false }), false);
  assert.equal(submission.shouldBypassSkipConfirm({ decision: 'create_new', shortens_lifetime: false, force_confirm: false }), false);
});

test('buildConfirmActions produces exact button counts and actions for higher/lower/busy/shorten decisions', () => {
  // Higher spec -> 3 buttons: reuse, replace, new
  const higher = submission.buildConfirmActions({
    decision: 'confirm_higher_spec',
    target_instance: { instance_id: 'inst-l4', gpu_reported: 'L4' },
  });
  assert.equal(higher.length, 3);
  assert.deepEqual(higher.map((a) => a.reuseAction), ['reuse', 'replace', 'new']);
  assert.equal(higher[0].instanceId, 'inst-l4');
  assert.equal(higher[1].replaceInstanceId, 'inst-l4');
  assert.equal(higher[2].instanceId, null);

  // Lower spec -> 2 buttons: replace, new
  const lower = submission.buildConfirmActions({
    decision: 'confirm_lower_spec',
    target_instance: { instance_id: 'inst-t4', gpu_reported: 'T4' },
  });
  assert.equal(lower.length, 2);
  assert.deepEqual(lower.map((a) => a.reuseAction), ['replace', 'new']);
  assert.equal(lower[0].replaceInstanceId, 'inst-t4');

  // Busy same spec -> 2 buttons: reuse, new
  const busy = submission.buildConfirmActions({
    decision: 'confirm_busy_same_spec',
    target_instance: { instance_id: 'inst-busy', gpu_reported: 'L4', is_busy: true },
  });
  assert.equal(busy.length, 2);
  assert.deepEqual(busy.map((a) => a.reuseAction), ['reuse', 'new']);
  assert.equal(busy[0].instanceId, 'inst-busy');

  // Shorten -> 1 button
  const shorten = submission.buildConfirmActions({
    decision: 'confirm_shorten',
    target_instance: { instance_id: 'inst-idle', gpu_reported: 'L4' },
  });
  assert.equal(shorten.length, 1);
  assert.equal(shorten[0].reuseAction, 'reuse');

  // Direct reuse & create_new -> 1 button
  assert.equal(submission.buildConfirmActions({ decision: 'reuse_direct' }).length, 1);
  assert.equal(submission.buildConfirmActions({ decision: 'create_new' }).length, 1);

  // Unsupported spec -> 0 buttons
  assert.equal(submission.buildConfirmActions({ decision: 'unsupported_spec' }).length, 0);
});

test('UI source wires reuse-preview, skipConfirmSig bypass, GPU order editor, and pool_instances', () => {
  const root = join(process.cwd(), 'src');
  const appSource = readFileSync(join(root, 'App.tsx'), 'utf8');
  const settingsSource = readFileSync(join(root, 'pages/SettingsPage.tsx'), 'utf8');
  const creationApiSource = readFileSync(join(root, 'pages/CreationApiPage.tsx'), 'utf8');
  const gpuSelectSource = readFileSync(join(root, 'components/WorkflowGpuSelection.tsx'), 'utf8');

  assert.match(appSource, /previewGpuJobReuse/);
  assert.match(appSource, /shouldBypassSkipConfirm/);
  assert.match(appSource, /buildConfirmActions/);
  assert.match(appSource, /reuse_action/);
  assert.match(appSource, /replace_instance_id/);
  assert.match(settingsSource, /GpuOrderSettings/);
  assert.match(creationApiSource, /pool_instances/);
  assert.match(gpuSelectSource, /isBelowWorkflowDefault/);
  assert.match(gpuSelectSource, /低于工作流默认规格/);
});
