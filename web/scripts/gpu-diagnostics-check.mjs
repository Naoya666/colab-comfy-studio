import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { build } from 'esbuild';

// Exercise real task-card rendering as well as the shared labels, without network or GPU calls.
const { outputFiles } = await build({
  stdin: { contents: `
    import { createElement } from 'react';
    import { renderToStaticMarkup } from 'react-dom/server';
    import { TasksPanel } from './src/pages/JobsPage';
    import { JobDetailInspector } from './src/components/JobDetailInspector';
    import { gpuLabel, canRetainJobInstance } from './src/lib/gpu-label';
    module.exports = { gpuLabel, canRetainJobInstance, render(job) {
      return renderToStaticMarkup(createElement(TasksPanel, {
        jobs: [job], error: '', refreshing: false,
        onRefresh() {}, onCancel() {}, onRetry() {}, onDetail() {}, onRetain() {}
      }));
    }, renderDetail(job) {
      return renderToStaticMarkup(createElement(JobDetailInspector, {
        detailId: job.id || 'test', detailError: '', inspectedJob: job,
        onClose() {}
      }));
    }};
  `, resolveDir: process.cwd(), loader: 'tsx' },
  bundle: true, platform: 'node', format: 'cjs', write: false, outfile: 'gpu-check.cjs',
  external: ['react', 'react-dom/server'], loader: { '.css': 'empty' },
});
const module = { exports: {} };
new Function('require', 'module', 'exports', outputFiles[0].text)(createRequire(import.meta.url), module, module.exports);
const { gpuLabel, canRetainJobInstance, render, renderDetail } = module.exports;
const base = { id: 'test', project_id: 'example', provider: 'colab', state: 'completed',
  instance_id: 'instance', instance_state: 'deleted', runtime_deleted: true,
  runtime_spec_requested: { accelerator: 'L4' } };
assert.equal(gpuLabel(base), '申请 GPU：L4（未实测）');
assert.equal(gpuLabel({ ...base, gpu_reported: 'NVIDIA A100-SXM4-40GB' }), '实测 GPU：NVIDIA A100-SXM4-40GB');
assert.equal(gpuLabel({ provider: 'mock' }), '模拟 GPU');
assert.equal(gpuLabel({ prep: true }), 'CPU 预热');
assert.equal(canRetainJobInstance(base), false);
assert.equal(canRetainJobInstance({ ...base, instance_state: 'unknown', runtime_deleted: false }), false);
assert.doesNotMatch(render(base), /保留这台机器|等待分配机器|GPU 未确认/);
assert.doesNotMatch(render({ ...base, retained_until: '2099-01-01' }), /已保留|保留这台机器/);
assert.match(render({ ...base, instance_state: 'busy', runtime_deleted: false }), /保留这台机器/);
const cancelled = render({ ...base, state: 'cancelled', instance_id: null, instance_state: null });
assert.match(cancelled, /未分配机器/);
assert.doesNotMatch(cancelled, /等待分配机器|保留这台机器/);

const failedDetail = renderDetail({
  ...base,
  state: 'failed',
  error_code: 'colab_upstream_unavailable',
  error_message: 'Colab 服务端暂时不可用（HTTP 503，等待 32.9s）',
  error_details: {
    error_code: 'colab_upstream_unavailable',
    category: 'upstream',
    phase: 'assign_post',
    http_status: 503,
    elapsed_seconds: 32.9,
    reconciling: false,
    reconciliation_result: 'confirmed_unassigned',
  },
  logs: [{ id: 'ev1', state: 'failed', message: '已确认 Colab 远端未分配实例，租约已释放，可以重试' }],
});
assert.match(failedDetail, /失败诊断/);
assert.match(failedDetail, /colab_upstream_unavailable/);
assert.match(failedDetail, /申请算力（assign_post）/);
assert.match(failedDetail, /HTTP 503/);
assert.match(failedDetail, /32\.9 秒/);
assert.match(failedDetail, /已确认远端未创建实例，租约已释放/);
assert.equal((failedDetail.match(/失败诊断/g) || []).length, 1);

const legacyDetail = renderDetail({
  ...base,
  state: 'failed',
  error_code: 'gpu_allocation_aborted',
  error_message: 'GPU allocation entered cleanup before execution; retry after release',
});
assert.match(legacyDetail, /失败诊断/);
assert.match(legacyDetail, /gpu_allocation_aborted/);
assert.doesNotMatch(legacyDetail, /HTTP 状态码|请求耗时/);
console.log('GPU evidence labels and task-card lifecycle checks passed');
