export type PreparedModel = {
  category: string;
  name: string;
  state: 'pending' | 'downloading' | 'ready' | 'failed';
  source?: 'external' | 'drive' | 'local';
  host?: string;
  error?: string;
  fallback?: string;
  bytes?: number;
  total?: number;
  seconds?: number;
  attempt?: number;
  integrity?: string;
};

export type DriveWorkflowItem = {
  source: 'comfy' | 'warmup';
  path: string;
  size: number;
  modified_at: number;
};

export type Preparation = {
  state: 'preparing' | 'ready' | 'failed';
  models: PreparedModel[];
  message?: string;
};

export type CategoryTone = 'core' | 'lora' | 'aux';

export type CategoryBadge = {
  short: string;
  raw: string;
  tone: CategoryTone;
};

export type PrepSummary = {
  total: number;
  ready: number;
  downloading: number;
  pending: number;
  failed: number;
  sources: {
    local: number;
    drive: number;
    external: number;
  };
  bytes: number;
  totalBytes: number;
  percent: number | null;
  elapsedSeconds: number | null;
  headline: string;
};

const CATEGORY_MAP: Record<string, { short: string; tone: CategoryTone }> = {
  diffusion_models: { short: 'UNET', tone: 'core' },
  unet: { short: 'UNET', tone: 'core' },
  unets: { short: 'UNET', tone: 'core' },
  checkpoints: { short: 'CKPT', tone: 'core' },
  text_encoders: { short: 'CLIP', tone: 'core' },
  clip: { short: 'CLIP', tone: 'core' },
  t5: { short: 'CLIP', tone: 'core' },
  clip_vision: { short: 'CLIP-V', tone: 'core' },
  vae: { short: 'VAE', tone: 'core' },
  vae_approx: { short: 'VAE', tone: 'core' },
  llm: { short: 'LLM', tone: 'core' },
  loras: { short: 'LoRA', tone: 'lora' },
  lora: { short: 'LoRA', tone: 'lora' },
  controlnet: { short: 'CN', tone: 'aux' },
  controlnets: { short: 'CN', tone: 'aux' },
  upscale_models: { short: 'UPSCALE', tone: 'aux' },
  embeddings: { short: 'EMB', tone: 'aux' },
  ipadapter: { short: 'IP-A', tone: 'aux' },
  style_models: { short: 'STYLE', tone: 'aux' },
  inpaint: { short: 'INPAINT', tone: 'aux' },
  gligen: { short: 'GLIGEN', tone: 'aux' },
  audio_encoders: { short: 'AUDIO', tone: 'aux' },
  sams: { short: 'SAM', tone: 'aux' },
  ultralytics: { short: 'DET', tone: 'aux' },
  facerestore_models: { short: 'FACE', tone: 'aux' },
  insightface: { short: 'FACE', tone: 'aux' },
};

const STATE_PRIORITY: Record<PreparedModel['state'], number> = {
  failed: 0,
  downloading: 1,
  pending: 2,
  ready: 3,
};

/** Map a ComfyUI model directory category to a compact badge label and tone. */
export function formatModelCategoryBadge(category: string): CategoryBadge {
  const raw = String(category || '').trim();
  const key = raw.toLowerCase();
  const mapped = CATEGORY_MAP[key];
  if (mapped) {
    return { short: mapped.short, raw, tone: mapped.tone };
  }
  const normalized = key.replace(/_models?$/, '').toUpperCase().slice(0, 8);
  return {
    short: normalized || 'MODEL',
    raw: raw || 'MODEL',
    tone: 'aux',
  };
}

/** Split a model relative path into muted directory prefix and emphasized filename. */
export function splitModelName(name: string): { dir: string; file: string } {
  const normalized = String(name || '');
  const slashIdx = Math.max(normalized.lastIndexOf('/'), normalized.lastIndexOf('\\'));
  if (slashIdx < 0) {
    return { dir: '', file: normalized };
  }
  return {
    dir: normalized.slice(0, slashIdx + 1),
    file: normalized.slice(slashIdx + 1),
  };
}

/** Stable sort putting failures and active downloads first, ready items last. */
export function sortPreparedModels(models: PreparedModel[]): PreparedModel[] {
  return models
    .map((model, index) => ({ model, index }))
    .sort((a, b) => {
      const diff = (STATE_PRIORITY[a.model.state] ?? 9) - (STATE_PRIORITY[b.model.state] ?? 9);
      return diff !== 0 ? diff : a.index - b.index;
    })
    .map((entry) => entry.model);
}

/** Aggregate counts, transfer progress, sources, and summary headline for a preparation job. */
export function summarizePreparation(result: Preparation): PrepSummary {
  const models = Array.isArray(result.models) ? result.models : [];
  let ready = 0;
  let downloading = 0;
  let pending = 0;
  let failed = 0;
  const sources = { local: 0, drive: 0, external: 0 };
  let bytes = 0;
  let totalBytes = 0;
  let elapsedSum = 0;
  let hasElapsed = false;

  for (const model of models) {
    if (model.state === 'ready') ready += 1;
    else if (model.state === 'downloading') downloading += 1;
    else if (model.state === 'pending') pending += 1;
    else if (model.state === 'failed') failed += 1;

    if (model.source === 'local') sources.local += 1;
    else if (model.source === 'drive') sources.drive += 1;
    else if (model.source === 'external') sources.external += 1;

    if (typeof model.total === 'number' && model.total > 0) {
      totalBytes += model.total;
      const modelBytes = model.state === 'ready' ? model.total : Math.min(model.total, Math.max(0, model.bytes || 0));
      bytes += modelBytes;
    }

    if (typeof model.seconds === 'number' && Number.isFinite(model.seconds)) {
      elapsedSum += model.seconds;
      hasElapsed = true;
    }
  }

  const total = models.length;
  let percent: number | null = null;
  if (totalBytes > 0) {
    percent = Math.min(100, Math.max(0, Math.round((bytes / totalBytes) * 100)));
  } else if (total > 0 && downloading === 0) {
    percent = Math.round((ready / total) * 100);
  }

  let headline = result.message?.trim() || '';
  if (!headline) {
    if (result.state === 'preparing') {
      headline = total > 0 ? `正在准备模型（${ready} / ${total}）` : '正在解析并检查模型…';
    } else if (result.state === 'ready') {
      headline = total > 0 ? `${ready} / ${total} 个模型已就绪` : '此工作流未识别出模型文件，可补充显式依赖。';
    } else {
      headline = total > 0 ? `${ready} / ${total} 就绪，${failed} 个失败` : '模型准备失败，请核对模型库和下方原因。';
    }
  }

  return {
    total,
    ready,
    downloading,
    pending,
    failed,
    sources,
    bytes,
    totalBytes,
    percent,
    elapsedSeconds: hasElapsed ? Number(elapsedSum.toFixed(1)) : null,
    headline,
  };
}

/** Count entries in explicit model dependency JSON; returns null when non-empty JSON is invalid or not an array. */
export function countExplicitDeclarations(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return 0;
  try {
    const parsed = JSON.parse(trimmed);
    return Array.isArray(parsed) ? parsed.length : null;
  } catch {
    return null;
  }
}

export function formatMiB(bytes: number): string {
  if (bytes >= 1024 ** 3) {
    return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  }
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

export function formatKiB(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KiB`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '0s';
  if (seconds < 60) {
    return Number.isInteger(seconds) ? `${seconds}s` : `${seconds.toFixed(1)}s`;
  }
  const rounded = Math.round(seconds);
  const m = Math.floor(rounded / 60);
  const s = rounded % 60;
  return `${m}m ${s}s`;
}
