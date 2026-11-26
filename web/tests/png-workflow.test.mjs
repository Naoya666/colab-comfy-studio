import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { transform } from 'esbuild';
const { code } = await transform(readFileSync('src/lib/png-workflow.ts', 'utf8'), { loader: 'ts', format: 'esm' });
const { extractPng, parseWorkflowJson, crc32 } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
const workflow = { nodes: [{ id: 1, type: 'CustomLoader', widgets_values: ['模型.safetensors'] }], links: [] };
const prompt = { '1': { class_type: 'CustomLoader', inputs: { ckpt_name: '模型.safetensors' } } };
function chunk(kind, payload) {
  const data = Buffer.concat([Buffer.from(kind), payload]);
  const n = Buffer.alloc(4), crc = Buffer.alloc(4); n.writeUInt32BE(payload.length); crc.writeUInt32BE(crc32(data));
  return Buffer.concat([n, data, crc]);
}
function text(key, value, kind = 'tEXt', packed = true) {
  const data = Buffer.from(typeof value === 'string' ? value : JSON.stringify(value));
  const payload = kind === 'zTXt' ? Buffer.concat([Buffer.from([0]), deflateSync(data)]) : kind === 'iTXt' ? Buffer.concat([Buffer.from([packed ? 1 : 0, 0, 0, 0]), packed ? deflateSync(data) : data]) : data;
  return chunk(kind, Buffer.concat([Buffer.from(key + '\0'), payload]));
}
function png(...texts) {
  const head = Buffer.alloc(13); head.writeUInt32BE(1); head.writeUInt32BE(1, 4); head[8] = 8; head[9] = 2;
  return Buffer.concat([Buffer.from('\x89PNG\r\n\x1a\n', 'latin1'), chunk('IHDR', head), ...texts, chunk('IDAT', deflateSync(Buffer.from([0, 0, 0, 0]))), chunk('IEND', Buffer.alloc(0))]);
}
test('extract both formats, UTF-8 filenames and node types without execution', async () => {
  const result = await extractPng(png(text('workflow', workflow), text('prompt', prompt)));
  assert.deepEqual(result.workflow, workflow); assert.deepEqual(result.prompt, prompt);
  assert.deepEqual(result.model_files, ['模型.safetensors']); assert.deepEqual(result.node_types, ['CustomLoader']); assert.equal(result.node_count, 1);
});
for (const kind of ['zTXt', 'iTXt']) test(`compressed ${kind} metadata`, async () => {
  const result = await extractPng(png(text('workflow', workflow, kind), text('prompt', prompt, kind)));
  assert.deepEqual(result.workflow, workflow); assert.deepEqual(result.prompt, prompt);
});
test('uncompressed international UTF-8', async () => assert.deepEqual((await extractPng(png(text('workflow', workflow, 'iTXt', false)))).workflow, workflow));
test('workflow only is retained and requires API format', async () => {
  const result = await extractPng(png(text('workflow', workflow))); assert.deepEqual(result.workflow, workflow); assert.equal(result.prompt, undefined); assert.match(result.warnings[0], /API/);
});
test('prompt only is retained', async () => assert.deepEqual((await extractPng(png(text('prompt', prompt)))).prompt, prompt));
test('invalid workflow does not discard valid prompt', async () => {
  const result = await extractPng(png(text('workflow', '{'), text('prompt', prompt)));
  assert.deepEqual(result.prompt, prompt); assert.match(result.warnings[0], /workflow 无效/);
});
test('format confusion is rejected', async () => {
  await assert.rejects(extractPng(png(text('prompt', workflow))), /不是 API/);
  await assert.rejects(extractPng(png(text('workflow', prompt))), /不是可视化/);
});
test('no metadata and A1111 explanations', async () => {
  await assert.rejects(extractPng(png()), /社交平台/);
  await assert.rejects(extractPng(png(text('parameters', 'Steps: 20'))), /A1111/);
});
test('bad JSON, wrong format, truncation and CRC corruption', async () => {
  await assert.rejects(extractPng(png(text('workflow', '{oops'))), /无效/);
  await assert.rejects(extractPng(Buffer.from('JPEG')), /PNG 原图/);
  await assert.rejects(extractPng(png(text('prompt', prompt)).subarray(0, -4)), /截断|损坏|完整/);
  const bytes = png(text('prompt', prompt)); bytes[45] ^= 1;
  await assert.rejects(extractPng(bytes), /校验/);
});
test('decompression bomb and oversized image are bounded', async () => {
  await assert.rejects(extractPng(png(text('workflow', 'x'.repeat(4 * 1024 * 1024 + 1), 'zTXt'))), /超过/);
  await assert.rejects(extractPng(new Uint8Array(32 * 1024 * 1024 + 1)), /32 MiB/);
});
test('duplicate metadata is ambiguous', async () => await assert.rejects(extractPng(png(text('prompt', prompt), text('prompt', prompt))), /重复/));
test('parseWorkflowJson sanitizes non-standard NaN/Infinity/overflowing floats while keeping strings intact', () => {
  const raw = '{"a":NaN,"b":Infinity,"c":-Infinity,"d":1e500,"e":-1e500,"str":"keep \\"NaN\\" and Infinity -Infinity","arr":[NaN,1,-Infinity]}';
  const { value, sanitizedNonFinite } = parseWorkflowJson(raw);
  assert.equal(sanitizedNonFinite, true);
  assert.deepEqual(value, {
    a: null,
    b: null,
    c: null,
    d: null,
    e: null,
    str: 'keep "NaN" and Infinity -Infinity',
    arr: [null, 1, null],
  });

  const clean = parseWorkflowJson('{"str":"NaN Infinity -Infinity","n":1.5}');
  assert.equal(clean.sanitizedNonFinite, false);
  assert.deepEqual(clean.value, { str: 'NaN Infinity -Infinity', n: 1.5 });

  for (const bad of ['{NaN: 1}', '{"a": NaNfoo}', '{"a": -NaN}', '{"a": +Infinity}', '{"a": 1-Infinity}']) {
    assert.throws(() => parseWorkflowJson(bad), SyntaxError);
  }
});
test('extractPng tolerates ComfyUI PNG metadata containing NaN/Infinity and emits warnings', async () => {
  const rawWf = '{"nodes":[{"id":1,"type":"CustomLoader","pos":[NaN,-Infinity],"size":[1e500,100],"widgets_values":["模型.safetensors"],"title":"literal NaN"}],"links":[]}';
  const rawPrompt = '{"1":{"class_type":"CustomLoader","inputs":{"ckpt_name":"模型.safetensors","cfg":NaN,"strength":Infinity,"note":"keep Infinity"}}}';
  const result = await extractPng(png(text('workflow', rawWf), text('prompt', rawPrompt)));
  assert.deepEqual(result.workflow.nodes[0].pos, [null, null]);
  assert.deepEqual(result.workflow.nodes[0].size, [null, 100]);
  assert.equal(result.workflow.nodes[0].title, 'literal NaN');
  assert.equal(result.prompt['1'].inputs.cfg, null);
  assert.equal(result.prompt['1'].inputs.strength, null);
  assert.equal(result.prompt['1'].inputs.note, 'keep Infinity');
  assert.deepEqual(result.model_files, ['模型.safetensors']);
  assert.ok(result.warnings.some((w) => w.includes('workflow 包含非标准数值')));
  assert.ok(result.warnings.some((w) => w.includes('prompt 包含非标准数值')));
});
