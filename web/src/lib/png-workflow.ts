/** Local, bounded PNG metadata extraction. Never evaluates graph content. */
export const MAX_IMAGE = 32 * 1024 * 1024;
const MAX_TEXT = 4 * 1024 * 1024;
const MAX_CHUNKS = 10000;
export type ExtractedWorkflow = {
  workflow?: Record<string, unknown>;
  prompt?: Record<string, unknown>;
  metadata?: Array<{keyword:string; text:string}>;
  warnings: string[];
  node_count: number;
  node_types: string[];
  model_files: string[];
};
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const utf8 = (v: Uint8Array) => new TextDecoder('utf-8', { fatal: true }).decode(v);
const ascii = (v: Uint8Array) => Array.from(v, c => String.fromCharCode(c)).join('');
export function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const b of bytes) {
    crc ^= b;
    for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
async function inflate(bytes: Uint8Array, limit: number): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') throw new Error('此浏览器不支持压缩元数据，请使用新版浏览器');
  const reader = new Blob([new Uint8Array(bytes)]).stream().pipeThrough(new DecompressionStream('deflate')).getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.length;
      if (length > limit) throw new Error('元数据解压后超过 4 MiB 限制');
      chunks.push(value);
    }
  } finally { await reader.cancel().catch(() => {}); }
  const out = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}
export async function extractPng(bytes: Uint8Array): Promise<ExtractedWorkflow> {
  if (bytes.length > MAX_IMAGE) throw new Error('图片不能超过 32 MiB');
  if (ascii(bytes.subarray(0, 8)) !== '\x89PNG\r\n\x1a\n') throw new Error('请选择 PNG 原图，不支持此文件格式');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const texts = new Map<string, string>();
  const metadata: Array<{keyword:string; text:string}> = [];
  let offset = 8, chunks = 0, total = 0, compressed = 0, ended = false, imageData = false;
  while (offset < bytes.length) {
    if (++chunks > MAX_CHUNKS || offset + 12 > bytes.length) throw new Error('PNG 损坏或数据块过多');
    const length = view.getUint32(offset);
    if (length > bytes.length - offset - 12) throw new Error('PNG 数据块被截断');
    const kind = ascii(bytes.subarray(offset + 4, offset + 8));
    const data = bytes.subarray(offset + 8, offset + 8 + length);
    if (crc32(bytes.subarray(offset + 4, offset + 8 + length)) !== view.getUint32(offset + 8 + length)) throw new Error('PNG 校验失败，文件可能损坏');
    if (chunks === 1 && (kind !== 'IHDR' || length !== 13)) throw new Error('PNG 缺少有效文件头');
    if (kind === 'IHDR') {
      if (chunks !== 1 || !view.getUint32(offset + 8) || !view.getUint32(offset + 12)) throw new Error('PNG 文件头无效');
      if (view.getUint32(offset + 8) * view.getUint32(offset + 12) > 100_000_000) throw new Error('图片像素超过限制');
    }
    if (kind === 'IDAT') imageData = true;
    if (['tEXt', 'iTXt', 'zTXt'].includes(kind)) {
      compressed += length;
      if (compressed > MAX_TEXT) throw new Error('元数据超过 4 MiB 限制');
      const split = data.indexOf(0);
      if (split < 1 || split > 79) throw new Error('PNG 文本块无效');
      const key = ascii(data.subarray(0, split));
      let payload = data.subarray(split + 1);
      let packed = false;
      if (kind === 'zTXt') {
        if (payload[0] !== 0) throw new Error('PNG 文本压缩方式不支持');
        packed = true; payload = payload.subarray(1);
      } else if (kind === 'iTXt') {
        if (![0, 1].includes(payload[0]) || payload[1] !== 0) throw new Error('PNG 国际文本块无效');
        packed = payload[0] === 1; payload = payload.subarray(2);
        for (let i = 0; i < 2; i++) {
          const end = payload.indexOf(0);
          if (end < 0) throw new Error('PNG 国际文本块被截断');
          payload = payload.subarray(end + 1);
        }
      }
      if (packed) payload = await inflate(payload, MAX_TEXT - total);
      total += payload.length;
      if (total > MAX_TEXT) throw new Error('元数据解压后超过 4 MiB 限制');
      let originalText: string;
      try { originalText = utf8(payload); } catch { originalText = new TextDecoder('windows-1252').decode(payload); }
      metadata.push({keyword:key, text:originalText});
      if (['workflow', 'prompt', 'parameters'].includes(key)) {
        if (texts.has(key)) throw new Error(`重复的 ${key} 元数据，无法确定应使用哪份`);
        texts.set(key, utf8(payload));
      }
    }
    offset += length + 12;
    if (kind === 'IEND') {
      if (length !== 0 || offset !== bytes.length || !imageData) throw new Error('PNG 结束块无效');
      ended = true; break;
    }
  }
  if (!ended) throw new Error('PNG 文件不完整');
  const result: ExtractedWorkflow = { metadata, warnings: [], node_count: 0, node_types: [], model_files: [] };
  for (const key of ['workflow', 'prompt'] as const) {
    if (!texts.has(key)) continue;
    try {
      const value: unknown = JSON.parse(texts.get(key)!);
      if (!object(value)) throw new Error('必须为 JSON 对象');
      if (key === 'workflow') {
        if (!Array.isArray(value.nodes) || !value.nodes.every(n => object(n) && (typeof n.id === 'number' || typeof n.id === 'string') && typeof n.type === 'string') || (value.links !== undefined && !Array.isArray(value.links))) throw new Error('不是可视化节点工作流');
      } else if (!Object.keys(value).length || !Object.values(value).every(n => object(n) && typeof n.class_type === 'string' && object(n.inputs))) throw new Error('不是 API prompt 工作流');
      result[key] = value;
    } catch (e) { result.warnings.push(`${key} 无效：${e instanceof Error ? e.message : 'JSON 格式错误'}`); }
  }
  if (!result.workflow && !result.prompt) {
    throw new Error(result.warnings.join('；') || (texts.has('parameters') ? '检测到 A1111 / Forge parameters，不包含完整 ComfyUI 节点工作流' : '没有找到 ComfyUI 元数据。图片可能被社交平台压缩或转码，请选择 PNG 原图'));
  }
  if (!result.prompt) result.warnings.push('仅有可视化 workflow，无法可靠自动转换为 API prompt；运行前需要提供 API 格式。');
  const nodes = result.prompt ? Object.values(result.prompt) : result.workflow!.nodes as unknown[];
  if (nodes.length > 10000) throw new Error('节点数量超过 10000 限制');
  result.node_count = nodes.length;
  result.node_types = [...new Set(nodes.filter(object).map(n => String(n.class_type || n.type)))].sort();
  // Filename discovery is only a hint, never a download source or installation instruction.
  const files = new Set<string>();
  const walk = (value: unknown, depth = 0) => {
    if (depth > 64) throw new Error('元数据嵌套过深');
    if (typeof value === 'string' && /\.(safetensors|ckpt|pt|pth|bin|gguf|onnx)$/i.test(value)) files.add(value);
    else if (Array.isArray(value)) value.forEach(v => walk(v, depth + 1));
    else if (object(value)) Object.values(value).forEach(v => walk(v, depth + 1));
  };
  walk(result.workflow); walk(result.prompt);
  result.model_files = [...files].sort();
  return result;
}
