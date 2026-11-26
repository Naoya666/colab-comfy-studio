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
        buildGpuStatusSummary,
        shortGpuName,
        formatCountdownLong,
        formatCountdownCompact,
      } from './src/lib/gpu-status-summary.ts';
    `,
    resolveDir: webRoot,
    sourcefile: 'gpu-status-summary-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  buildGpuStatusSummary,
  shortGpuName,
  formatCountdownLong,
  formatCountdownCompact,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

const sampleWorkflows = [
  {
    id: 'wf-1',
    workflow_key: 'krea2-moss-dual-t2i',
    workflow_version: '1.1',
    model_id: 'krea2',
    state: 'published',
    active: true,
    title: 'Krea-2 二次元厚涂 5-LoRA',
  },
];

const baseUsage = {
  today_actual_cu: 1.25,
  active_runtimes: 0,
  guard: { status: 'ok', soft_tripped: false, hard_tripped_accounts: [], daily_limit_cu: null, min_balance_cu: null, message: null, daily_message: null },
  ccu: { total_balance: 48.5 },
};

test('shortGpuName and countdown formatters produce concise labels', () => {
  assert.equal(shortGpuName('NVIDIA L4'), 'L4');
  assert.equal(shortGpuName('Tesla T4'), 'T4');
  assert.equal(shortGpuName('NVIDIA A100-SXM4-40GB'), 'A100');
  assert.equal(shortGpuName(''), 'GPU');

  const now = 1700000000;
  assert.equal(formatCountdownLong(now + 75, now), '1 分 15 秒');
  assert.equal(formatCountdownLong(now + 3900, now), '1 小时 5 分');
  assert.equal(formatCountdownLong(now - 5, now), '即将到期');
  assert.equal(formatCountdownCompact(now + 45, now), '45s');
  assert.equal(formatCountdownCompact(now + 600, now), '10m');
});

test('idle state reports zero CU burn and recent batch summary', () => {
  const summary = buildGpuStatusSummary({
    settings: { real_enabled: true, offline_mode: false },
    instances: [],
    jobs: [
      {
        id: 'j-1',
        batch_id: 'b-1',
        project_id: 'p-1',
        workflow_key: 'krea2-moss-dual-t2i',
        workflow_version: '1.1',
        state: 'completed',
        output_asset_ids: ['a-1'],
        created_at: '2026-11-26T10:00:00Z',
      },
      {
        id: 'j-2',
        batch_id: 'b-1',
        project_id: 'p-1',
        workflow_key: 'krea2-moss-dual-t2i',
        workflow_version: '1.1',
        state: 'completed',
        output_asset_ids: ['a-2'],
        created_at: '2026-11-26T10:00:01Z',
      },
    ],
    usage: baseUsage,
    workflows: sampleWorkflows,
    nowSec: 1700000000,
  });

  assert.equal(summary.phase, 'idle');
  assert.equal(summary.tone, 'idle');
  assert.equal(summary.pulsing, false);
  assert.equal(summary.pillLabel, '无活动 GPU');
  assert.match(summary.bannerText, /0 消耗/);
  assert.equal(summary.rows.find((r) => r.key === 'gpu')?.value, '已释放（0 消耗）');
  assert.equal(summary.rows.find((r) => r.key === 'batch')?.value, '已完成 2/2');
  assert.match(summary.rows.find((r) => r.key === 'workflow')?.value || '', /Krea-2 二次元厚涂 5-LoRA/);
  assert.equal(summary.rows.find((r) => r.key === 'cu')?.value, '1.25 CU · 余 48.50 CU');
  assert.deepEqual(summary.primaryAction, { kind: 'gallery', label: '查看最近作品' });
  assert.deepEqual(summary.secondaryAction, { kind: 'compute', label: '算力中心' });
});

test('allocation retry waiting shows retry count, live countdown, and jumps to job log', () => {
  const now = 1700000000;
  const batchJobs = Array.from({ length: 4 }, (_, idx) => ({
    id: `job-${idx + 1}`,
    batch_id: 'batch-retry',
    project_id: 'p-1',
    provider: 'colab',
    workflow_key: 'krea2-moss-dual-t2i',
    workflow_version: '1.1',
    state: idx === 0 ? 'waiting_for_capacity' : 'queued',
    created_at: `2026-11-26T10:00:0${idx}Z`,
    allocation_retry: idx === 0
      ? {
          enabled: true,
          active: true,
          phase: 'waiting',
          attempt: 5,
          retries_used: 4,
          max_retries: 25,
          interval_seconds: 60,
          next_retry_at: now + 24,
          next_spec: { variant: 'VARIANT_GPU', accelerator: 'L4', shape: 'SHAPE_STANDARD' },
          next_account_alias: '主号',
          last_error_message: '当前时段无可用 L4',
        }
      : null,
  }));

  const summary = buildGpuStatusSummary({
    settings: { real_enabled: true, offline_mode: false },
    instances: [],
    jobs: batchJobs,
    usage: baseUsage,
    workflows: sampleWorkflows,
    nowSec: now,
  });

  assert.equal(summary.phase, 'retrying');
  assert.equal(summary.tone, 'retrying');
  assert.equal(summary.pulsing, true);
  assert.equal(summary.pillLabel, '排队抢 L4 (4/25)');
  assert.match(summary.bannerText, /第 4\/25 次重试/);
  assert.match(summary.bannerText, /24 秒后尝试/);
  assert.equal(summary.progress.active, true);
  assert.equal(summary.progress.indeterminate, false);
  assert.equal(summary.progress.percent, 60);
  assert.equal(summary.needsCountdownTick, true);
  assert.equal(summary.rows.find((r) => r.key === 'gpu')?.value, '已释放（0 消耗）');
  assert.equal(summary.rows.find((r) => r.key === 'batch')?.value, '进行中 0/4');
  assert.deepEqual(summary.primaryAction, { kind: 'job', jobId: 'job-1', label: '查看抢卡日志' });
  assert.deepEqual(summary.secondaryAction, { kind: 'compute', label: '算力中心' });
});

test('running batch shows batch progress, GPU reported, and workflow title', () => {
  const summary = buildGpuStatusSummary({
    settings: { real_enabled: true, offline_mode: false },
    instances: [
      {
        id: 'inst-l4',
        provider: 'colab',
        instance_pool: 'creation',
        state: 'busy',
        gpu_reported: 'NVIDIA L4',
        runtime_deleted: false,
      },
    ],
    jobs: [
      {
        id: 'j-done',
        batch_id: 'b-run',
        project_id: 'p-1',
        provider: 'colab',
        workflow_key: 'krea2-moss-dual-t2i',
        workflow_version: '1.1',
        state: 'completed',
        created_at: '2026-11-26T10:00:00Z',
      },
      {
        id: 'j-run',
        batch_id: 'b-run',
        project_id: 'p-1',
        provider: 'colab',
        workflow_key: 'krea2-moss-dual-t2i',
        workflow_version: '1.1',
        state: 'running',
        stage: 'sampling',
        progress: 0.5,
        instance_id: 'inst-l4',
        gpu_reported: 'NVIDIA L4',
        account_alias_snapshot: '主号',
        created_at: '2026-11-26T10:00:01Z',
      },
      {
        id: 'j-q1',
        batch_id: 'b-run',
        project_id: 'p-1',
        provider: 'colab',
        workflow_key: 'krea2-moss-dual-t2i',
        workflow_version: '1.1',
        state: 'queued',
        created_at: '2026-11-26T10:00:02Z',
      },
      {
        id: 'j-q2',
        batch_id: 'b-run',
        project_id: 'p-1',
        provider: 'colab',
        workflow_key: 'krea2-moss-dual-t2i',
        workflow_version: '1.1',
        state: 'queued',
        created_at: '2026-11-26T10:00:03Z',
      },
    ],
    usage: { ...baseUsage, active_runtimes: 1 },
    workflows: sampleWorkflows,
    nowSec: 1700000000,
  });

  assert.equal(summary.phase, 'running');
  assert.equal(summary.tone, 'live');
  assert.equal(summary.pulsing, true);
  assert.equal(summary.pillLabel, '生成中 1/4');
  assert.match(summary.bannerText, /NVIDIA L4/);
  assert.match(summary.bannerText, /采样生成/);
  assert.match(summary.bannerText, /第 2\/4 个/);
  assert.equal(summary.progress.percent, 38);
  assert.equal(summary.rows.find((r) => r.key === 'gpu')?.value, '1 台运行中（L4 · 创作池）');
  assert.equal(summary.rows.find((r) => r.key === 'batch')?.value, '进行中 1/4');
  assert.deepEqual(summary.primaryAction, { kind: 'job', jobId: 'j-run', label: '查看任务日志' });
});

test('retained idle GPU surfaces billing warning tone, live countdown, and multi-instance list', () => {
  const now = 1700000000;
  const singleRetained = buildGpuStatusSummary({
    settings: { real_enabled: true, offline_mode: false },
    instances: [
      {
        id: 'inst-t4',
        provider: 'colab',
        instance_pool: 'creation',
        state: 'retained_idle',
        gpu_reported: 'Tesla T4',
        retain_idle_minutes: 20,
        idle_deadline_at: String(now + 600),
        runtime_deleted: false,
      },
    ],
    jobs: [],
    usage: { ...baseUsage, active_runtimes: 1 },
    workflows: sampleWorkflows,
    nowSec: now,
  });

  assert.equal(singleRetained.phase, 'retained');
  assert.equal(singleRetained.tone, 'retained');
  assert.equal(singleRetained.pillLabel, 'T4 保留中 · 10m');
  assert.match(singleRetained.bannerText, /10 分 0 秒/);
  assert.equal(singleRetained.progress.percent, 50);
  assert.equal(singleRetained.needsCountdownTick, true);
  assert.equal(singleRetained.onlineInstances.length, 0);

  const multiOnline = buildGpuStatusSummary({
    settings: { real_enabled: true, offline_mode: false },
    instances: [
      {
        id: 'inst-native',
        provider: 'colab',
        instance_pool: 'native',
        state: 'ready',
        gpu_reported: 'NVIDIA L4',
        account_alias_snapshot: '主号',
        runtime_deleted: false,
      },
      {
        id: 'inst-t4',
        provider: 'colab',
        instance_pool: 'creation',
        state: 'retained_idle',
        gpu_reported: 'Tesla T4',
        account_alias_snapshot: '副号',
        idle_deadline_at: String(now + 300),
        runtime_deleted: false,
      },
    ],
    jobs: [],
    usage: { ...baseUsage, active_runtimes: 2 },
    workflows: sampleWorkflows,
    nowSec: now,
  });

  assert.equal(multiOnline.phase, 'native_online');
  assert.equal(multiOnline.pillLabel, '2 台 GPU 在线');
  assert.equal(multiOnline.onlineInstances.length, 2);
  assert.equal(multiOnline.onlineInstances[0].gpu, 'L4');
  assert.equal(multiOnline.onlineInstances[1].countdown, '5m');
  assert.deepEqual(multiOnline.primaryAction, { kind: 'native', label: '打开原生 ComfyUI' });
});

test('domain transfer, native retry, CU guard tripped, and offline mode map cleanly', () => {
  const transferToCreation = buildGpuStatusSummary({
    settings: { real_enabled: true, offline_mode: false },
    instances: [
      {
        id: 'inst-tr',
        provider: 'colab',
        instance_pool: 'native',
        state: 'transferring',
        gpu_reported: 'NVIDIA L4',
        transfer_state: { from_pool: 'native', to_pool: 'creation', phase: 'stopping_source' },
        runtime_deleted: false,
      },
    ],
    jobs: [],
    usage: baseUsage,
    workflows: sampleWorkflows,
    nowSec: 1700000000,
  });
  assert.equal(transferToCreation.phase, 'transferring');
  assert.equal(transferToCreation.pillLabel, '域转移 · L4');
  assert.notEqual(transferToCreation.primaryAction.kind, transferToCreation.secondaryAction.kind);

  const guardHard = buildGpuStatusSummary({
    settings: { real_enabled: true, offline_mode: false },
    instances: [],
    jobs: [],
    usage: {
      ...baseUsage,
      guard: {
        ...baseUsage.guard,
        status: 'hard_balance_floor',
        message: '主号余额不足 2.00 CU',
      },
    },
    workflows: sampleWorkflows,
    nowSec: 1700000000,
  });
  assert.equal(guardHard.phase, 'guard_tripped');
  assert.equal(guardHard.tone, 'danger');
  assert.equal(guardHard.pillLabel, 'CU 余额熔断');
  assert.match(guardHard.bannerText, /余额不足/);

  const offlineIdle = buildGpuStatusSummary({
    settings: { real_enabled: false, offline_mode: true },
    instances: [],
    jobs: [],
    usage: baseUsage,
    workflows: sampleWorkflows,
    nowSec: 1700000000,
  });
  assert.equal(offlineIdle.phase, 'offline');
  assert.equal(offlineIdle.pillLabel, '离线演示');
  assert.equal(offlineIdle.rows.find((r) => r.key === 'cu'), undefined);
});

test('GpuStatusPopover component and CSS enforce click-only trigger, portal layering, mobile sheet, and valid theme tokens', () => {
  const component = readSrc('src/components/GpuStatusPopover.tsx');
  const css = readSrc('src/components/GpuStatusPopover.module.css');
  const app = readSrc('src/App.tsx');

  assert.match(component, /createPortal/);
  assert.match(component, /useBodyScrollLock\(open && !desktop\)/);
  assert.match(component, /aria-haspopup="dialog"/);
  assert.match(component, /aria-expanded=\{open\}/);
  assert.doesNotMatch(component, /onMouseEnter|onMouseLeave/);

  // Pure dashboard: must not import or call stop/cancel/release mutating APIs.
  assert.doesNotMatch(component, /stopInstance|stopGpuCompute|cancelGpuJob|setInstanceRetention/);

  // Mobile sheet + desktop popover layering and scroll safety.
  assert.match(css, /\.popover\s*\{[\s\S]*position:\s*fixed[\s\S]*overscroll-behavior:\s*contain/);
  assert.match(css, /\.sheet\s*\{[\s\S]*overscroll-behavior:\s*contain[\s\S]*var\(--safe-b\)/);
  assert.match(css, /\.dot\.mobileCornerDot\s*\{\s*display:\s*none;/);
  assert.match(css, /@media\s*\(max-width:\s*519px\)[\s\S]*\.dot\.triggerInlineDot\s*\{\s*display:\s*none;/);

  // App integrates popover, Escape close, and active-job polling.
  assert.match(app, /<GpuStatusPopover/);
  assert.match(app, /else if \(state\.showGpuStatus\) setShowGpuStatus\(false\)/);
  assert.match(app, /hasAnyActiveJob/);
});
