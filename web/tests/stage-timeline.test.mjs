import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { transform } from 'esbuild';

async function loadTimelineModule() {
  const stageRows = readFileSync('src/lib/job-stage-rows.ts', 'utf8');
  const timeline = readFileSync('src/lib/stage-timeline.ts', 'utf8')
    .replace(/^import .*?;\n/gm, '');
  const { code } = await transform(`${stageRows}\n${timeline}`, { loader: 'ts', format: 'esm' });
  return import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
}

test('stage heat ratio increases monotonically and shifts hue from green toward red', async () => {
  const { stageHeatRatio, stageHeatLevel, stageHeatColor } = await loadTimelineModule();
  const r0 = stageHeatRatio(0);
  const r2 = stageHeatRatio(2);
  const r12 = stageHeatRatio(12);
  const r40 = stageHeatRatio(40);
  const r90 = stageHeatRatio(90);
  assert.equal(r0, 0);
  assert.ok(r0 < r2 && r2 < r12 && r12 < r40 && r40 < r90 && r90 === 1);
  assert.equal(stageHeatLevel(2), 'fast');
  assert.equal(stageHeatLevel(10), 'normal');
  assert.equal(stageHeatLevel(24), 'slow');
  assert.equal(stageHeatLevel(50), 'hot');

  const hue = (css) => Number(css.match(/^hsl\((\d+)/)?.[1]);
  assert.ok(hue(stageHeatColor(1)) > hue(stageHeatColor(15)));
  assert.ok(hue(stageHeatColor(15)) > hue(stageHeatColor(60)));
  assert.equal(stageHeatColor(1, 'failed'), 'var(--danger)');
});

test('buildNativeStartupStages orders chronologically, denoises fast tail stages, and ticks active stage live', async () => {
  const { buildNativeStartupStages, computeTimelineSegments } = await loadTimelineModule();
  const fastTail = buildNativeStartupStages({
    state: 'completed',
    timingDetails: {
      comfy_start: { seconds: 24.5, source: 'worker', status: 'completed' },
      queue: { seconds: 2.0, source: 'events', status: 'completed' },
      native_system: { seconds: 6.1, source: 'worker', status: 'completed' },
      upload: { seconds: 0.3, source: 'events', status: 'completed' },
      cleanup: { seconds: 0.4, source: 'events', status: 'completed' },
    },
  });
  assert.deepEqual(fastTail.map((s) => s.key), ['queue', 'native_system', 'comfy_start']);

  const slowTail = buildNativeStartupStages({
    state: 'completed',
    timingDetails: {
      native_system: { seconds: 5.0, source: 'worker', status: 'completed' },
      upload: { seconds: 1.2, source: 'events', status: 'completed' },
      cleanup: { seconds: 1.3, source: 'events', status: 'completed' },
    },
  });
  assert.deepEqual(slowTail.map((s) => [s.key, s.label, s.seconds]), [
    ['native_system', '安装系统依赖', 5.0],
    ['native_finalize', '服务就绪收尾', 2.5],
  ]);

  const running = buildNativeStartupStages({
    state: 'running',
    stage: 'native_env_cache',
    timingDetails: {
      allocate: { seconds: 4.0, source: 'events', status: 'completed' },
      native_env_cache: { seconds: 10.0, source: 'worker', status: 'in_progress' },
    },
    elapsedDeltaSeconds: 3.4,
  });
  assert.equal(running[1].key, 'native_env_cache');
  assert.equal(running[1].status, 'in_progress');
  assert.equal(running[1].seconds, 13.4);

  const { segments, totalSeconds, hasInProgress } = computeTimelineSegments(running);
  assert.equal(hasInProgress, true);
  assert.equal(totalSeconds, 17.4);
  assert.equal(segments.length, 2);
  assert.ok(segments[1].percent > segments[0].percent);

  const failedAlloc = buildNativeStartupStages({
    state: 'failed',
    stage: null,
    timingDetails: {
      queue: { seconds: 0.6, source: 'events', status: 'completed' },
      allocate: { seconds: 14.1, source: 'events', status: 'completed' },
      cleanup: { seconds: 65.6, source: 'events', status: 'completed' },
    },
  });
  assert.deepEqual(failedAlloc.map((s) => [s.key, s.label, s.status]), [
    ['queue', '排队等待', 'completed'],
    ['allocate', '分配算力', 'failed'],
    ['cleanup', '释放算力', 'completed'],
  ]);
});

test('buildNativeRestartStages exposes four restart phases with live elapsed and failed phase highlight', async () => {
  const { buildNativeRestartStages } = await loadTimelineModule();
  const active = buildNativeRestartStages({
    restart: {
      restart_id: 'r1',
      instance_id: 'i1',
      phase: 'recovering',
      timing_details: {
        restart_checking: { seconds: 0.4, source: 'events', status: 'completed' },
        restart_syncing: { seconds: 2.1, source: 'events', status: 'completed' },
        restart_rebooting: { seconds: 0.8, source: 'events', status: 'completed' },
        restart_recovering: { seconds: 9.0, source: 'events', status: 'in_progress' },
      },
    },
    elapsedDeltaSeconds: 2.5,
  });
  assert.deepEqual(active.map((s) => [s.key, s.status, s.seconds]), [
    ['restart_checking', 'completed', 0.4],
    ['restart_syncing', 'completed', 2.1],
    ['restart_rebooting', 'completed', 0.8],
    ['restart_recovering', 'in_progress', 11.5],
  ]);

  const failed = buildNativeRestartStages({
    restart: {
      restart_id: 'r2',
      instance_id: 'i1',
      phase: 'failed',
      failed_phase: 'syncing',
      timing_details: {
        restart_checking: { seconds: 0.3, source: 'events', status: 'completed' },
        restart_syncing: { seconds: 4.2, source: 'events', status: 'completed' },
      },
    },
  });
  assert.equal(failed[1].key, 'restart_syncing');
  assert.equal(failed[1].status, 'failed');
  assert.equal(failed[1].seconds, 4.2);
});
