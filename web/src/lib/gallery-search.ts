import type { GalleryAsset } from './gpu-api';

export function tokenizeGalleryQuery(query: string): string[] {
  const tokens = String(query || '')
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  return [...new Set(tokens)];
}

export function galleryAssetSearchText(
  asset: GalleryAsset,
  workflowLabel: string,
  projectTitle: string,
): string {
  const parts = [
    asset.prompt || '',
    asset.main_prompt || '',
    asset.negative_prompt || '',
    asset.model_id || '',
    asset.resolved_seed != null ? String(asset.resolved_seed) : '',
    asset.filename || '',
    asset.source_job_id || '',
    asset.gpu_job_id || '',
    asset.workflow_key || '',
    workflowLabel || '',
    projectTitle || '',
    asset.gpu_reported || '',
  ];
  return parts.join('\n').toLowerCase();
}

export function matchesGalleryQuery(
  asset: GalleryAsset,
  query: string,
  workflowLabel: string,
  projectTitle: string,
): boolean {
  const tokens = tokenizeGalleryQuery(query);
  if (!tokens.length) return true;
  const haystack = galleryAssetSearchText(asset, workflowLabel, projectTitle);
  return tokens.every((token) => haystack.includes(token));
}

function collapseWhitespace(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

export function galleryPromptPreview(
  asset: GalleryAsset,
  query = '',
  maxLen = 84,
): string {
  const main = collapseWhitespace(asset.main_prompt || '');
  const positive = collapseWhitespace(asset.prompt || '');
  const negative = collapseWhitespace(asset.negative_prompt || '');
  const tokens = tokenizeGalleryQuery(query);

  let source = main || positive;
  let prefix = '';

  if (tokens.length > 0) {
    const mainLower = main.toLowerCase();
    const posLower = positive.toLowerCase();
    const negLower = negative.toLowerCase();
    if (main && tokens.some((t) => mainLower.includes(t))) {
      source = main;
    } else if (positive && tokens.some((t) => posLower.includes(t))) {
      source = positive;
    } else if (negative && tokens.some((t) => negLower.includes(t))) {
      source = negative;
      prefix = '负面词：';
    }
  }

  if (!source) return '';
  const limit = Math.max(20, maxLen - prefix.length);
  if (source.length <= limit) return `${prefix}${source}`;

  if (tokens.length > 0) {
    const lower = source.toLowerCase();
    const matchIdx = tokens
      .map((t) => lower.indexOf(t))
      .filter((idx) => idx >= 0)
      .sort((a, b) => a - b)[0];
    if (matchIdx !== undefined && matchIdx > Math.floor(limit * 0.45)) {
      const start = Math.max(0, matchIdx - Math.floor(limit * 0.3));
      const slice = source.slice(start, start + limit - 1).trim();
      const tail = start + limit - 1 < source.length ? '…' : '';
      return `${prefix}…${slice}${tail}`;
    }
  }
  return `${prefix}${source.slice(0, limit - 1).trim()}…`;
}
