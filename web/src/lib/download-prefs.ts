import type { GalleryAsset, NativeOutputAsset } from './gpu-api';

export type ImageDownloadMode = 'original' | 'clean';

export const IMAGE_DOWNLOAD_MODE_KEY = 'studio-gallery-download-mode';

const STRIPPABLE_IMAGE_MIMES = new Set([
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
]);

const ALREADY_CLEAN_RE = /(?:^|[_-])clean$/i;

export type DownloadableGalleryItem =
  | { kind: 'studio'; asset: Pick<GalleryAsset, 'id' | 'mime' | 'filename' | 'download_url'> }
  | { kind: 'native'; asset: Pick<NativeOutputAsset, 'id' | 'mime' | 'filename' | 'content_url'> };

export interface GalleryDownloadTarget {
  href: string;
  filename: string;
  strippable: boolean;
  effectiveMode: ImageDownloadMode;
}

export function isStrippableImageMime(mime?: string | null): boolean {
  if (!mime) return false;
  return STRIPPABLE_IMAGE_MIMES.has(mime.trim().toLowerCase());
}

export function cleanDownloadFilename(filename?: string | null): string {
  const raw = String(filename || '').trim();
  const segments = raw.replace(/\\/g, '/').split('/');
  const name = segments[segments.length - 1] || '';
  if (!name || name === '.' || name === '..') return 'image_clean';

  const dot = name.lastIndexOf('.');
  const hasExt = dot > 0 && dot < name.length - 1;
  const stem = hasExt ? name.slice(0, dot) : name;
  const ext = hasExt ? name.slice(dot) : '';

  if (ALREADY_CLEAN_RE.test(stem)) return `${stem}${ext}`;
  if (stem.endsWith('_') || stem.endsWith('-')) return `${stem}clean${ext}`;
  return `${stem}_clean${ext}`;
}

export function loadImageDownloadMode(storage?: Pick<Storage, 'getItem'> | null): ImageDownloadMode {
  try {
    const store = storage === undefined
      ? (typeof localStorage !== 'undefined' ? localStorage : null)
      : storage;
    const saved = store?.getItem(IMAGE_DOWNLOAD_MODE_KEY);
    return saved === 'clean' ? 'clean' : 'original';
  } catch {
    return 'original';
  }
}

export function saveImageDownloadMode(
  mode: ImageDownloadMode,
  storage?: Pick<Storage, 'setItem'> | null,
): void {
  try {
    const store = storage === undefined
      ? (typeof localStorage !== 'undefined' ? localStorage : null)
      : storage;
    store?.setItem(IMAGE_DOWNLOAD_MODE_KEY, mode);
  } catch {
    // Ignore restricted storage environments.
  }
}

function appendQuery(url: string, key: string, value: string): string {
  const hashIndex = url.indexOf('#');
  const base = hashIndex >= 0 ? url.slice(0, hashIndex) : url;
  const hash = hashIndex >= 0 ? url.slice(hashIndex) : '';
  const sep = base.includes('?') ? '&' : '?';
  return `${base}${sep}${encodeURIComponent(key)}=${encodeURIComponent(value)}${hash}`;
}

export function buildGalleryDownloadTarget(
  item: DownloadableGalleryItem,
  mode: ImageDownloadMode,
): GalleryDownloadTarget {
  const rawTitle = item.asset.filename || item.asset.id;
  const strippable = isStrippableImageMime(item.asset.mime);
  const effectiveMode: ImageDownloadMode = strippable && mode === 'clean' ? 'clean' : 'original';

  let href = item.kind === 'studio'
    ? (item.asset.download_url || `/api/assets/${item.asset.id}/download`)
    : appendQuery(item.asset.content_url, 'download', 'true');

  if (effectiveMode === 'clean') {
    href = appendQuery(href, 'strip_metadata', 'true');
  }

  return {
    href,
    filename: effectiveMode === 'clean' ? cleanDownloadFilename(rawTitle) : rawTitle,
    strippable,
    effectiveMode,
  };
}

export function imageDownloadButtonLabel(mode: ImageDownloadMode): string {
  return mode === 'clean' ? '无元数据下载' : '下载原图';
}
