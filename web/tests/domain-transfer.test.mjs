import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  buildTransferSteps,
  canShowTransferButton,
  domainBadgeLabel,
  domainBadgeTone,
  findTransferableInstancesForCreation,
  findTransferableInstancesForNative,
  formatVmElapsed,
  isCpuInstance,
  isGpuInstance,
  transferDisabledReason,
} from '../src/lib/domain-transfer.ts';
import { smokeIdempotencyKey } from '../src/lib/activation-resume.ts';

const root = join(process.cwd(), 'src');
const read = (name) => readFileSync(join(root, name), 'utf8');

const L4 = { variant: 'VARIANT_GPU', accelerator: 'L4', shape: 'SHAPE_STANDARD' };
const T4 = { variant: 'VARIANT_GPU', accelerator: 'T4', shape: 'SHAPE_STANDARD' };
const A100 = { variant: 'VARIANT_GPU', accelerator: 'A100', shape: 'SHAPE_STANDARD' };
const CPU = { variant: 'VARIANT_CPU', accelerator: 'NONE', shape: 'SHAPE_STANDARD' };

const NOW = 1_750_000_000;

function makeInstance(overrides = {}) {
  return {
    id: 'inst-1',
    instance_id: 'inst-1',
    provider: 'colab',
    instance_pool: 'creation',
    state: 'retained_idle',
    gpu_reported: 'NVIDIA L4',
    runtime_spec_requested: { key: L4 },
    runtime_spec_reported: { key: L4 },
    venv_isolated: true,
    vm_allocated_at: NOW - 1500,
    idle_deadline_at: String(new Date((NOW + 600) * 1000).toISOString()),
    ...overrides,
  };
}

test('domain badges distinguish creation, native, ephemeral GPU, and ephemeral CPU', () => {
  const creation = makeInstance({ instance_pool: 'creation' });
  const nativeInst = makeInstance({ instance_pool: 'native' });
  const smokeGpu = makeInstance({ instance_pool: 'ephemeral' });
  const cpuPrep = makeInstance({
    instance_pool: 'ephemeral',
    gpu_reported: 'CPU',
    runtime_spec_requested: { key: CPU },
    runtime_spec_reported: { key: CPU },
  });

  assert.equal(domainBadgeLabel(creation), '创作池');
  assert.equal(domainBadgeTone(creation), 'creation');
  assert.equal(domainBadgeLabel(nativeInst), '原生 ComfyUI');
  assert.equal(domainBadgeTone(nativeInst), 'native');
  assert.equal(domainBadgeLabel(smokeGpu), '冒烟域');
  assert.equal(domainBadgeTone(smokeGpu), 'ephemeral');
  assert.equal(domainBadgeLabel(cpuPrep), '一次性（CPU）');
  assert.equal(domainBadgeTone(cpuPrep), 'cpu');
  assert.equal(isCpuInstance(cpuPrep), true);
  assert.equal(isGpuInstance(cpuPrep), false);
  assert.equal(isGpuInstance(creation), true);
  assert.equal(formatVmElapsed(NOW - 1500, NOW), '25 分钟');
  assert.equal(formatVmElapsed(NOW - 5400, NOW), '1 小时 30 分钟');
});

test('transfer button visibility and venv_isolated=false disabled reason', () => {
  const readyGpu = makeInstance({ state: 'ready' });
  const idleGpu = makeInstance({ state: 'retained_idle' });
  const busyGpu = makeInstance({ state: 'busy' });
  const cpuInst = makeInstance({
    instance_pool: 'ephemeral',
    gpu_reported: 'CPU',
    runtime_spec_requested: { key: CPU },
    runtime_spec_reported: { key: CPU },
  });
  const legacyGpu = makeInstance({ venv_isolated: false });

  assert.equal(canShowTransferButton(readyGpu), true);
  assert.equal(canShowTransferButton(idleGpu), true);
  assert.equal(canShowTransferButton(busyGpu), false);
  assert.equal(canShowTransferButton(cpuInst), false);
  assert.equal(canShowTransferButton(legacyGpu), true);
  assert.equal(transferDisabledReason(readyGpu), null);
  assert.match(transferDisabledReason(legacyGpu) || '', /旧版运行时环境/);
});

test('buildTransferSteps tracks cleaning, verifying, native_bootstrapping and failure states', () => {
  const stepsToNative = buildTransferSteps({ to_pool: 'native', phase: 'native_bootstrapping' });
  assert.equal(stepsToNative.length, 3);
  assert.deepEqual(
    stepsToNative.map((s) => s.status),
    ['done', 'done', 'active'],
  );

  const stepsToCreation = buildTransferSteps({ to_pool: 'creation', phase: 'verifying' });
  assert.equal(stepsToCreation.length, 2);
  assert.deepEqual(
    stepsToCreation.map((s) => s.status),
    ['done', 'active'],
  );
});

test('cross-page transfer hints appear only when current domain is empty and spec is compatible', () => {
  const idleL4Creation = makeInstance({ id: 'c1', instance_pool: 'creation' });
  const legacyL4Creation = makeInstance({ id: 'c-old', instance_pool: 'creation', venv_isolated: false });
  const idleT4Native = makeInstance({
    id: 'n-t4',
    instance_pool: 'native',
    gpu_reported: 'NVIDIA T4',
    runtime_spec_requested: { key: T4 },
    runtime_spec_reported: { key: T4 },
  });
  const idleA100Ephemeral = makeInstance({
    id: 'e-a100',
    instance_pool: 'ephemeral',
    gpu_reported: 'NVIDIA A100',
    runtime_spec_requested: { key: A100 },
    runtime_spec_reported: { key: A100 },
  });

  // Native page hint: returns isolated creation GPU only when no native/transferring-in exists.
  assert.deepEqual(
    findTransferableInstancesForNative([idleL4Creation, legacyL4Creation], NOW).map((i) => i.id),
    ['c1'],
  );
  assert.deepEqual(
    findTransferableInstancesForNative([idleL4Creation, idleT4Native], NOW),
    [],
  );
  assert.deepEqual(
    findTransferableInstancesForNative(
      [idleL4Creation, makeInstance({ id: 'tr1', state: 'transferring', transfer_state: { to_pool: 'native' } })],
      NOW,
    ),
    [],
  );

  // Creation page hint: requires creation domain empty AND candidate spec >= required spec (no downgrade!).
  assert.deepEqual(
    findTransferableInstancesForCreation([idleT4Native, idleA100Ephemeral], L4, undefined, NOW).map((i) => i.id),
    ['e-a100'],
  );
  assert.deepEqual(
    findTransferableInstancesForCreation([idleT4Native], L4, undefined, NOW),
    [],
  );
  // Hidden when creation domain already has an active instance.
  assert.deepEqual(
    findTransferableInstancesForCreation([idleL4Creation, idleA100Ephemeral], L4, undefined, NOW),
    [],
  );
});

test('smokeIdempotencyKey separates retainAfterSmoke and requestedInstanceId variants', () => {
  const values = new Map();
  const prev = globalThis.sessionStorage;
  globalThis.sessionStorage = {
    getItem: (k) => values.get(k) ?? null,
    setItem: (k, v) => values.set(k, v),
  };
  try {
    const kDefault = smokeIdempotencyKey('wf', 'proj', ['a'], null, { retainAfterSmoke: false, requestedInstanceId: null });
    const kRetain = smokeIdempotencyKey('wf', 'proj', ['a'], null, { retainAfterSmoke: true, requestedInstanceId: null });
    const kBound = smokeIdempotencyKey('wf', 'proj', ['a'], null, { retainAfterSmoke: true, requestedInstanceId: 'inst-9' });
    assert.notEqual(kDefault, kRetain);
    assert.notEqual(kRetain, kBound);
    assert.equal(
      smokeIdempotencyKey('wf', 'proj', ['a'], null, { retainAfterSmoke: true, requestedInstanceId: 'inst-9' }),
      kBound,
    );
  } finally {
    if (prev === undefined) delete globalThis.sessionStorage;
    else globalThis.sessionStorage = prev;
  }
});

test('UI components wire transfer modal, unchecked smoke retain default, 409 handling, and cross-page navigation', () => {
  const center = read('components/InstanceCenter.tsx');
  assert.match(center, /getInstanceTransferPreview/);
  assert.match(center, /transferGpuInstance/);
  assert.match(center, /expected_version/);
  assert.match(center, /domainBadgeLabel/);
  assert.match(center, /formatVmElapsed/);
  assert.match(center, /native_bootstrap_failed/);
  assert.match(center, /quarantine_reason/);
  assert.match(center, /focusTransferTarget/);

  const activation = read('components/WorkflowActivation.tsx');
  assert.match(activation, /useState\(false\).*retainAfterSmoke|const \[retainAfterSmoke, setRetainAfterSmoke\] = useState\(false\)/);
  assert.match(activation, /retain_after_smoke:\s*retainAfterSmoke/);
  assert.match(activation, /requested_instance_id/);
  assert.match(activation, /EPHEMERAL_INSTANCE_UNAVAILABLE/);
  assert.match(activation, /onTransferToEphemeral/);

  const nativePage = read('pages/NativeComfyPage.tsx');
  assert.match(nativePage, /standby_instance/);
  assert.match(nativePage, /transferring_in/);
  assert.match(nativePage, /transferred_out/);
  assert.match(nativePage, /原地重试启动（复用原卡）/);
  assert.match(nativePage, /findTransferableInstancesForNative/);

  const createPage = read('pages/CreatePage.tsx');
  assert.match(createPage, /findTransferableInstancesForCreation/);
  assert.match(createPage, /去实例中心转移至创作池/);

  const cuDetails = read('components/CuSessionDetails.tsx');
  assert.match(cuDetails, /multi_domain:\s*'跨域复用'/);

  const app = read('App.tsx');
  assert.match(app, /focusTransferTarget/);
  assert.match(app, /jumpToTransfer/);
  assert.match(app, /'transferring'/);
});
