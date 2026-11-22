import { useCallback, useEffect, useRef, useState } from 'react';
import { Film, Folder, Image as ImageIcon, Play, RefreshCw, Search, Star } from 'lucide-react';
import {
  apiMessage,
  getNativeOutputs,
  type NativeOutputAsset,
  type NativeOutputsPage,
} from '../lib/gpu-api';
import { bytesLabel } from '../lib/model-library';
import styles from '../gpu.module.css';

type Filter = 'all' | 'image' | 'video' | 'favorite' | 'unfavorited';

const PAGE_SIZE = 48;

function dimsLabel(item: NativeOutputAsset): string {
  if (item.width && item.height) return `${item.width} × ${item.height}`;
  return item.size ? bytesLabel(item.size) : '';
}

export function NativeGalleryPanel({
  inspectedId,
  refreshToken,
  favoriteOverrides,
  deletedIds,
  onOpen,
  onFavorite,
}: {
  inspectedId?: string;
  refreshToken?: number;
  favoriteOverrides: Record<string, boolean>;
  deletedIds: Set<string>;
  onOpen: (asset: NativeOutputAsset, ordered: NativeOutputAsset[]) => void;
  onFavorite: (asset: NativeOutputAsset) => void;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const [subfolder, setSubfolder] = useState('');
  const [page, setPage] = useState<NativeOutputsPage | null>(null);
  const [items, setItems] = useState<NativeOutputAsset[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState('');
  const [brokenThumbs, setBrokenThumbs] = useState<Set<string>>(() => new Set());
  const seqRef = useRef(0);

  const load = useCallback(async (opts: { refresh?: boolean; append?: boolean; offset?: number } = {}) => {
    const seq = ++seqRef.current;
    const append = Boolean(opts.append);
    if (append) setLoadingMore(true);
    else setLoading(true);
    setError('');
    try {
      const mediaType = filter === 'image' ? 'image' : filter === 'video' ? 'video' : 'all';
      const favorite = filter === 'favorite' ? 'only' : filter === 'unfavorited' ? 'unfavorited' : 'all';
      const data = await getNativeOutputs({
        refresh: Boolean(opts.refresh),
        limit: PAGE_SIZE,
        offset: opts.offset ?? 0,
        mediaType,
        favorite,
        q: query,
        subfolder,
      });
      if (seq !== seqRef.current) return;
      setPage(data);
      setItems((curr) => (append ? [...curr, ...data.items.filter((x) => !curr.some((c) => c.id === x.id))] : data.items));
    } catch (cause) {
      if (seq === seqRef.current) setError(apiMessage(cause));
    } finally {
      if (seq === seqRef.current) {
        setLoading(false);
        setLoadingMore(false);
      }
    }
  }, [filter, query, subfolder]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load({ offset: 0 }), query ? 220 : 0);
    return () => window.clearTimeout(timer);
  }, [load, query, refreshToken]);

  const visible = items
    .filter((item) => !deletedIds.has(item.id))
    .map((item) => (
      Object.prototype.hasOwnProperty.call(favoriteOverrides, item.id)
        ? { ...item, favorite: favoriteOverrides[item.id] }
        : item
    ))
    .filter((item) => {
      if (filter === 'favorite' && !item.favorite) return false;
      if (filter === 'unfavorited' && item.favorite) return false;
      return true;
    });

  const chip = (value: Filter, label: string, icon?: React.ReactNode) => (
    <button type="button" className={styles.chip} data-active={filter === value} onClick={() => setFilter(value)}>
      {icon}{label}
    </button>
  );

  return (
    <div className={styles.stack}>
      <div className={styles.pageTitle}>
        <h2>
          <ImageIcon size={20} strokeWidth={1.6} aria-hidden />
          原生作品（Drive）
          {page?.connected && <span className="pill">{page.total} 件</span>}
        </h2>
        <div className={styles.titleActions}>
          <button
            type="button"
            className="button secondary small"
            disabled={loading}
            onClick={() => void load({ refresh: true, offset: 0 })}
          >
            <RefreshCw size={14} className={loading ? styles.spin : undefined} aria-hidden />
            {loading ? '读取中…' : '刷新 Drive'}
          </button>
        </div>
      </div>

      {page?.connected && (
        <p className="muted small">
          直读 Google Drive `native-comfy/output/`{page.account_alias ? `（${page.account_alias}）` : ''}，无需开机即可预览、收藏、导入参考素材或提取 PNG 工作流。
        </p>
      )}

      {error && <p className="status error" role="alert">{error}</p>}

      {page && !page.connected ? (
        <div className="empty">
          <strong>无法读取 Drive 原生作品</strong>
          <span>{page.message || '请先在「设置」中连接 Google Drive 账号。'}</span>
        </div>
      ) : (
        <>
          <label className={styles.searchBox}>
            <Search size={17} strokeWidth={1.6} aria-hidden />
            <input
              aria-label="搜索原生作品"
              placeholder="搜索原生作品（文件名、子目录）"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>

          <div className={styles.filterBar}>
            <div className={styles.chipRow}>
              {chip('all', '全部')}
              {chip('image', '图片')}
              {chip('video', '视频')}
              {chip('favorite', '收藏', <Star size={13} strokeWidth={1.8} aria-hidden />)}
              {chip('unfavorited', '未收藏')}
            </div>
            {Boolean(page?.subfolders?.length) && (
              <div className={styles.filterRight}>
                <label className="small muted" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <Folder size={13} aria-hidden />
                  <select
                    aria-label="按子目录筛选"
                    className={styles.miniSelect}
                    value={subfolder}
                    onChange={(e) => setSubfolder(e.target.value)}
                  >
                    <option value="">全部目录</option>
                    {page!.subfolders.map((dir) => (
                      <option key={dir} value={dir}>{dir}</option>
                    ))}
                  </select>
                </label>
              </div>
            )}
          </div>

          {loading && !visible.length ? (
            <p className="muted small" role="status">正在读取 Drive 原生作品目录…</p>
          ) : visible.length ? (
            <div className={styles.galleryGrid}>
              {visible.map((asset) => {
                const isVideo = asset.mime.startsWith('video/');
                const selected = inspectedId === asset.id;
                const hud = dimsLabel(asset);
                const thumbBroken = brokenThumbs.has(asset.id);
                return (
                  <article className={styles.tile} key={asset.id} data-selected={selected}>
                    <div className={styles.tileMedia}>
                      <button
                        className={styles.tileCover}
                        type="button"
                        aria-label={`查看 ${asset.filename}`}
                        aria-pressed={inspectedId === undefined ? undefined : selected}
                        onClick={() => onOpen(asset, visible)}
                      >
                        {isVideo && thumbBroken ? (
                          <video src={asset.content_url} preload="metadata" muted playsInline aria-label={asset.filename} />
                        ) : (
                          <img
                            loading="lazy"
                            src={thumbBroken ? asset.content_url : asset.thumbnail_url}
                            alt={asset.filename}
                            onError={() => {
                              setBrokenThumbs((curr) => {
                                if (curr.has(asset.id)) return curr;
                                const next = new Set(curr);
                                next.add(asset.id);
                                return next;
                              });
                            }}
                          />
                        )}
                      </button>
                      <span className={styles.tileType}>
                        {isVideo ? <Film size={12} strokeWidth={1.8} aria-hidden /> : <ImageIcon size={12} strokeWidth={1.8} aria-hidden />}
                        {isVideo ? '视频' : '图片'}
                      </span>
                      {(hud || isVideo) && (
                        <span className={styles.tileHud}>
                          {isVideo && <Play size={10} strokeWidth={2} aria-hidden />}
                          {hud || '视频'}
                        </span>
                      )}
                    </div>
                    <div className={styles.tileMeta}>
                      <div>
                        <strong title={asset.path}>{asset.filename}</strong>
                        <small className="muted">{asset.subfolder || 'output/'}</small>
                      </div>
                      <button
                        type="button"
                        className={styles.starBtn}
                        data-on={Boolean(asset.favorite)}
                        aria-label={asset.favorite ? '取消收藏' : '收藏'}
                        onClick={() => onFavorite(asset)}
                      >
                        <Star size={17} strokeWidth={1.6} fill={asset.favorite ? 'currentColor' : 'none'} aria-hidden />
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <div className="empty">
              <strong>暂无符合条件的原生作品</strong>
              <span>原生 ComfyUI 保存到 Google Drive 的图片与视频会自动出现在这里。</span>
            </div>
          )}

          {page?.has_more && (
            <button
              className="button secondary block"
              type="button"
              disabled={loadingMore}
              onClick={() => void load({ append: true, offset: items.length })}
            >
              {loadingMore ? '加载中…' : '加载更多原生作品'}
            </button>
          )}
        </>
      )}
    </div>
  );
}
