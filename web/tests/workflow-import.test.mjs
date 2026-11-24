import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: {
    contents: `
      export {
        classifyWorkflowJson,
        validateWorkflowUrlInput,
        PLAIN_COMFYUI_WORKFLOW_NOTICE,
      } from './src/lib/workflow-import.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    sourcefile: 'workflow-import-test-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  classifyWorkflowJson,
  validateWorkflowUrlInput,
  PLAIN_COMFYUI_WORKFLOW_NOTICE,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('classifyWorkflowJson accepts Studio bundles and image_extraction wrappers', () => {
  const studioBundle = {
    workflow_key: 'demo',
    manifest: { version: '1.0.0' },
    api_graph: { '1': { class_type: 'VAELoader', inputs: {} } },
    object_info: { VAELoader: {} },
  };
  assert.equal(classifyWorkflowJson(studioBundle).kind, 'studio_bundle');
  assert.equal(classifyWorkflowJson({ workflow: studioBundle }).kind, 'studio_bundle');
  assert.equal(
    classifyWorkflowJson({ workflow_key: 'img-1', image_extraction: { prompt: {} } }).kind,
    'studio_bundle',
  );
});

test('classifyWorkflowJson conservatively flags plain ComfyUI visual and API JSONs', () => {
  const visual = { nodes: [{ id: 1, type: 'UNETLoader', widgets_values: ['flux.safetensors'] }], links: [] };
  const apiPrompt = { '1': { class_type: 'UNETLoader', inputs: { unet_name: 'flux.safetensors' } } };

  for (const sample of [visual, apiPrompt, { workflow: visual }, { prompt: apiPrompt }]) {
    const res = classifyWorkflowJson(sample);
    assert.equal(res.kind, 'plain_comfyui');
    assert.equal(res.message, PLAIN_COMFYUI_WORKFLOW_NOTICE);
  }
});

test('classifyWorkflowJson rejects non-objects and unrelated JSON objects', () => {
  assert.equal(classifyWorkflowJson([]).kind, 'invalid');
  assert.equal(classifyWorkflowJson('str').kind, 'invalid');
  assert.equal(classifyWorkflowJson({ foo: 'bar' }).kind, 'invalid');
});

test('validateWorkflowUrlInput accepts public HTTPS URLs (including temporary signed queries) and rejects unsafe inputs', () => {
  const okSigned = validateWorkflowUrlInput('  https://cdn.discordapp.com/attachments/1/2/wf.png?ex=1&hm=2  ');
  assert.equal(okSigned.ok, true);
  assert.equal(okSigned.url, 'https://cdn.discordapp.com/attachments/1/2/wf.png?ex=1&hm=2');

  assert.equal(validateWorkflowUrlInput('').ok, false);
  assert.equal(validateWorkflowUrlInput('http://example.com/wf.json').ok, false);
  assert.equal(validateWorkflowUrlInput('https://user:pass@example.com/wf.json').ok, false);
  assert.equal(validateWorkflowUrlInput('https://example.com:8443/wf.json').ok, false);
  assert.equal(validateWorkflowUrlInput('not-a-url').ok, false);
});
