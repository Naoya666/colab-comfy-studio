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
        isValidWorkflowKey,
        isValidSemver,
        parseWorkflowVersionConflict,
        applyBundleIdentityOverride,
        formatExportSummaryMessage,
      } from './src/lib/workflow-import.ts';
      export { dependencySourceUrl } from './src/lib/model-library.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    sourcefile: 'workflow-import-test-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  define: { 'import.meta.env.VITE_API_BASE_URL': '""' },
  logLevel: 'silent',
});

const {
  classifyWorkflowJson,
  validateWorkflowUrlInput,
  PLAIN_COMFYUI_WORKFLOW_NOTICE,
  isValidWorkflowKey,
  isValidSemver,
  parseWorkflowVersionConflict,
  applyBundleIdentityOverride,
  formatExportSummaryMessage,
  dependencySourceUrl,
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

test('workflow key and SemVer validators accept valid values and reject invalid ones', () => {
  assert.equal(isValidWorkflowKey('moss-image-sft-t2i'), true);
  assert.equal(isValidWorkflowKey('my.wf_2'), true);
  assert.equal(isValidWorkflowKey(''), false);
  assert.equal(isValidWorkflowKey('-bad'), false);
  assert.equal(isValidWorkflowKey('has space'), false);

  assert.equal(isValidSemver('1.0.1'), true);
  assert.equal(isValidSemver('2.3.4-custom.1'), true);
  assert.equal(isValidSemver('1.0'), false);
  assert.equal(isValidSemver('v1.0.0'), false);
});

test('parseWorkflowVersionConflict extracts structured 409 detail and ignores unrelated errors', () => {
  const conflict = parseWorkflowVersionConflict({
    code: 'WORKFLOW_VERSION_CONFLICT',
    detail: {
      workflow_key: 'demo',
      workflow_version: '1.0.0',
      existing_state: 'published',
      existing_versions: ['1.0.0'],
      suggested_version: '1.0.1',
      suggested_workflow_key: 'demo-imported',
    },
  });
  assert.deepEqual(conflict, {
    workflow_key: 'demo',
    workflow_version: '1.0.0',
    existing_state: 'published',
    existing_versions: ['1.0.0'],
    suggested_version: '1.0.1',
    suggested_workflow_key: 'demo-imported',
  });
  assert.equal(parseWorkflowVersionConflict({ code: 'VALIDATION_ERROR', detail: {} }), null);
  assert.equal(parseWorkflowVersionConflict(null), null);
});

test('applyBundleIdentityOverride updates flat and wrapped bundles immutably', () => {
  const flat = {
    workflow_key: 'orig',
    manifest: { workflow_key: 'orig', version: '1.0.0', parameters: {} },
    api_graph: {},
    object_info: {},
  };
  const nextFlat = applyBundleIdentityOverride(flat, 'orig-copy', '1.0.2');
  assert.equal(flat.workflow_key, 'orig');
  assert.equal(flat.manifest.version, '1.0.0');
  assert.equal(nextFlat.workflow_key, 'orig-copy');
  assert.equal(nextFlat.manifest.workflow_key, 'orig-copy');
  assert.equal(nextFlat.manifest.version, '1.0.2');

  const wrapped = { workflow: flat };
  const nextWrapped = applyBundleIdentityOverride(wrapped, 'wrapped-key', '2.0.0');
  assert.equal(nextWrapped.workflow.workflow_key, 'wrapped-key');
  assert.equal(nextWrapped.workflow.manifest.version, '2.0.0');
});

test('formatExportSummaryMessage highlights Civitai, manual rebind, and stripped Drive IDs', () => {
  const msg = formatExportSummaryMessage({
    workflow_key: 'style-wf',
    workflow_version: '1.0.0',
    model_id: 'm1',
    has_model_metadata: true,
    total_weights: 3,
    huggingface_weights: 1,
    restored_huggingface_weights: 1,
    civitai_weights: 1,
    manual_rebind_weights: 1,
    stripped_drive_ids: 3,
    warnings: [],
    rebind_models: [],
  });
  assert.match(msg, /style-wf/);
  assert.match(msg, /1 个 Civitai 来源/);
  assert.match(msg, /1 个需重绑模型/);
});

test('dependencySourceUrl synthesizes Civitai page URL from source_identity when source_url is absent', () => {
  const url = dependencySourceUrl({
    id: 'd1',
    node_id: '2',
    node_type: 'LoraLoaderModelOnly',
    field: 'lora_name',
    reference: 'style.safetensors',
    model_type: 'loras',
    architecture: 'sdxl',
    status: 'unresolved',
    required: true,
    weight: {
      source: 'civitai',
      source_identity: { model_id: 12345, version_id: 67890, file_id: 999 },
    },
  });
  assert.equal(url, 'https://civitai.com/models/12345?modelVersionId=67890');
});
