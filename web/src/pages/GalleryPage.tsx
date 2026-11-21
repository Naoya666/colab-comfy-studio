import { useMemo, useState } from 'react';
import {
  Image as ImageIcon, Film, Search, Star, Check, Eye, RotateCcw, Trash2, Play,
  Layers, CheckSquare, X, FileJson,
} from 'lucide-react';
import { assetUrl, type GalleryAsset } from '../lib/gpu-api';
import { groupGalleryAssets, isDiagnosticAsset } from '../lib/history-grouping';
import { ConfirmDialog } from '../components/Feedback';
import styles from '../gpu.module.css';

type Filter = 'all' | 'image' | 'video' | 'favorite' | 'unfavorited';

function dims(asset: GalleryAsset): string {
  const meta = (asset.metadata || {}) as Record<string, unknown>;
  const w = Number(meta.width ?? asset.width);
  const h = Number(meta.height ?? asset.height);
  return w > 0 && h > 0 ? `${w} × ${h}` : '';
}

export function GalleryPanel({
  gallery, error, onFavorite, onDelete, onBatchDelete, onOpen, onReuse, onGoCreate, inspectedId,
}: {
  gallery: GalleryAsset[];
  error?: string;
  onFavorite: (asset: GalleryAsset) => void;
  onDelete: (asset: GalleryAsset) => void;
  onBatchDelete?: (ids: string[]) => Promise<void> | void;
  onOpen: (asset: GalleryAsset, ordered?: GalleryAsset[]) => void;
  onReuse?: (asset: GalleryAsset) => void;
  onGoCreate?: () => void;
  inspectedId?: string;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [stackedView, setStackedView] = useState(true);
  const [expandedStacks, setExpandedStacks] = useState<Set<string>>(() => new Set());
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(24);
  const [batchMode, setBatchMode] = useState(false);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(() => new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [confirmBatchDelete, setConfirmBatchDelete] = useState<{ ids: string[]; title: string; description: string } | null>(null);
  const [deletingBatch, setDeletingBatch] = useState(false);

  const diagnosticAssets = useMemo(() => gallery.filter(isDiagnosticAsset), [gallery]);
  const mediaCount = gallery.length - diagnosticAssets.length;

  const filtered = useMemo(() => gallery.filter((asset) => {
    const isDiag = isDiagnosticAsset(asset);
    if (isDiag && !showDiagnostics) return false;
    const isVideo = asset.mime.startsWith('video/');
    const isImage = asset.mime.startsWith('image/');
    if (filter === 'image' && !isImage) return false;
    if (filter === 'video' && !isVideo) return false;
    if (filter === 'favorite' && !asset.favorite) return false;
    if (filter === 'unfavorited' && asset.favorite) return false;
    return `${asset.filename || ''} ${asset.source_job_id || ''} ${asset.workflow_key || ''}`.toLowerCase().includes(query.toLowerCase());
  }), [gallery, filter, query, showDiagnostics]);

  const stacks = useMemo(() => groupGalleryAssets(filtered), [filtered]);
  const hasMultiStacks = useMemo(() => stacks.some((s) => s.items.length > 1), [stacks]);

  const selected = useMemo(() => gallery.find((a) => a.id === selectedId) || null, [gallery, selectedId]);
  const activeCheckedIds = useMemo(() => {
    const valid = new Set(gallery.map((a) => a.id));
    return [...checkedIds].filter((id) => valid.has(id));
  }, [checkedIds, gallery]);

  function toggleChecked(ids: string[]) {
    setBatchMode(true);
    setCheckedIds((curr) => {
      const next = new Set(curr);
      const allChecked = ids.every((id) => next.has(id));
      if (allChecked) ids.forEach((id) => next.delete(id));
      else ids.forEach((id) => next.add(id));
      return next;
    });
  }

  function exitBatchMode() {
    setBatchMode(false);
    setCheckedIds(new Set());
  }

  function toggleStack(key: string) {
    setExpandedStacks((curr) => {
      const next = new Set(curr);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  async function executeBatchDelete() {
    if (!confirmBatchDelete) return;
    setDeletingBatch(true);
    try {
      const ids = confirmBatchDelete.ids;
      if (onBatchDelete) {
        await onBatchDelete(ids);
      } else {
        for (const id of ids) {
          const item = gallery.find((a) => a.id === id);
          if (item) onDelete(item);
        }
      }
      setCheckedIds((curr) => {
        const next = new Set(curr);
        ids.forEach((id) => next.delete(id));
        return next;
      });
      if (selectedId && ids.includes(selectedId)) setSelectedId(null);
      setConfirmBatchDelete(null);
    } finally {
      setDeletingBatch(false);
    }
  }

  const chip = (value: Filter, label: string, icon?: React.ReactNode) => (
    <button type="button" className={styles.chip} data-active={filter === value} onClick={() => setFilter(value)}>{icon}{label}</button>
  );

  function renderTile(asset: GalleryAsset, stackInfo?: { key: string; count: number; items: GalleryAsset[]; expanded: boolean }) {
    const isVideo = asset.mime.startsWith('video/');
    const isJson = isDiagnosticAsset(asset);
    const idsForCheck = stackInfo && !stackInfo.expanded ? stackInfo.items.map((i) => i.id) : [asset.id];
    const isChecked = idsForCheck.every((id) => checkedIds.has(id));
    const on = batchMode ? isChecked : (inspectedId ?? selectedId) === asset.id;
    const size = dims(asset);
    const isStackedCover = Boolean(stackInfo && stackInfo.count > 1 && !stackInfo.expanded);

    return (
      <article
        className={`${styles.tile} ${isStackedCover ? styles.tileStacked : ''}`}
        key={stackInfo ? `${stackInfo.key}:${asset.id}` : asset.id}
        data-selected={on}
      >
        <div className={styles.tileMedia}>
          <button
            className={styles.tileCover}
            type="button"
            aria-label={`查看 ${asset.filename || '作品'}`}
            aria-pressed={inspectedId === undefined ? undefined : on}
            onClick={() => {
              if (batchMode) {
                toggleChecked(idsForCheck);
                return;
              }
              if (inspectedId === undefined) setSelectedId(asset.id);
              onOpen(asset, stackInfo && stackInfo.count > 1 ? stackInfo.items : filtered);
            }}
          >
            {isVideo ? (
              <video src={assetUrl(asset)} preload="metadata" muted playsInline aria-label={asset.filename || '视频'} />
            ) : isJson ? (
              <span className={styles.tileJson}>JSON<small>诊断文件</small></span>
            ) : (
              <img loading="lazy" src={assetUrl(asset, true)} alt={asset.filename || '作品'} />
            )}
          </button>
          <span className={styles.tileType}>
            {isVideo ? <Film size={12} strokeWidth={1.8} aria-hidden /> : <ImageIcon size={12} strokeWidth={1.8} aria-hidden />}
            {isVideo ? '视频' : isJson ? '诊断' : '图片'}
          </span>
          {(batchMode || inspectedId === undefined) && (
            <button
              type="button"
              className={styles.tileCheck}
              data-on={batchMode ? isChecked : on}
              aria-label={isChecked ? '取消选中' : '选中'}
              aria-pressed={batchMode ? isChecked : on}
              onClick={() => {
                if (batchMode) toggleChecked(idsForCheck);
                else setSelectedId(on ? null : asset.id);
              }}
            >
              {(batchMode ? isChecked : on) && <Check size={13} strokeWidth={2.6} aria-hidden />}
            </button>
          )}
          {(size || isVideo) && (
            <span className={styles.tileHud}>
              {isVideo && <Play size={10} strokeWidth={2} aria-hidden />}
              {size || '视频'}
            </span>
          )}
          {isStackedCover && stackInfo && (
            <button
              type="button"
              className={styles.stackBadge}
              title={`展开同批次 ${stackInfo.count} 件作品`}
              onClick={(e) => {
                e.stopPropagation();
                toggleStack(stackInfo.key);
              }}
            >
              <Layers size={11} strokeWidth={2} aria-hidden />×{stackInfo.count} 展开
            </button>
          )}
        </div>
        <div className={styles.tileMeta}>
          <div>
            <strong title={asset.filename || asset.id}>{asset.filename || `作品 ${asset.id.slice(0, 6)}`}</strong>
            <small className="muted">
              {isStackedCover && stackInfo
                ? `同批共 ${stackInfo.count} 件 · ${asset.workflow_key || '点击展开'}`
                : asset.source_job_id ? `任务 ${asset.source_job_id.slice(0, 6)}` : '上传素材'}
            </small>
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
  }

  return (
    <div className={styles.stack}>
      <div className={styles.pageTitle}>
        <h2>
          <ImageIcon size={20} strokeWidth={1.6} aria-hidden />
          作品
          <span className="pill">{mediaCount} 件作品</span>
        </h2>
        <div className={styles.titleActions}>
          {gallery.length > 0 && (
            <button
              type="button"
              className={`button small ${batchMode ? '' : 'secondary'}`}
              onClick={() => (batchMode ? exitBatchMode() : setBatchMode(true))}
            >
              {batchMode ? <><X size={14} aria-hidden />完成管理</> : <><CheckSquare size={14} aria-hidden />批量管理</>}
            </button>
          )}
        </div>
      </div>
      {error && <p className="status error" role="alert">{error}</p>}

      {batchMode && (
        <div className={styles.cleanupBar} role="toolbar" aria-label="批量管理作品">
          <strong>已选 {activeCheckedIds.length} 件</strong>
          <div className="actions">
            <button
              type="button"
              className="button secondary small"
              onClick={() => setCheckedIds(new Set(filtered.map((a) => a.id)))}
            >
              全选当前 ({filtered.length})
            </button>
            <button
              type="button"
              className="button secondary small"
              onClick={() => setCheckedIds(new Set(filtered.filter((a) => !a.favorite).map((a) => a.id)))}
            >
              选中未收藏 ({filtered.filter((a) => !a.favorite).length})
            </button>
            <button
              type="button"
              className="button ghost small"
              disabled={!activeCheckedIds.length}
              onClick={() => setCheckedIds(new Set())}
            >
              清空选择
            </button>
            <button
              type="button"
              className="button danger small"
              disabled={!activeCheckedIds.length || deletingBatch}
              onClick={() => setConfirmBatchDelete({
                ids: activeCheckedIds,
                title: `删除选中的 ${activeCheckedIds.length} 件作品？`,
                description: '文件将从存储中永久移除，删除后无法恢复。',
              })}
            >
              <Trash2 size={14} aria-hidden />批量删除 ({activeCheckedIds.length})
            </button>
          </div>
        </div>
      )}

      <label className={styles.searchBox}>
        <Search size={17} strokeWidth={1.6} aria-hidden />
        <input aria-label="搜索作品" placeholder="搜索作品（文件名、工作流、任务编号）" value={query} onChange={(e) => { setQuery(e.target.value); setLimit(24); }} />
      </label>

      <div className={styles.filterBar}>
        <div className={styles.chipRow}>
          {chip('all', '全部')}
          {chip('image', '图片')}
          {chip('video', '视频')}
          {chip('favorite', '收藏', <Star size={13} strokeWidth={1.8} aria-hidden />)}
          {chip('unfavorited', '未收藏')}
        </div>
        <div className={styles.filterRight}>
          {hasMultiStacks && (
            <button
              type="button"
              className={styles.chip}
              data-active={stackedView}
              onClick={() => setStackedView((v) => !v)}
            >
              <Layers size={13} strokeWidth={1.7} aria-hidden />{stackedView ? '批次已堆叠' : '按批次堆叠'}
            </button>
          )}
          {diagnosticAssets.length > 0 && (
            <>
              <button
                type="button"
                className={styles.chip}
                data-active={showDiagnostics}
                onClick={() => setShowDiagnostics((v) => !v)}
              >
                <FileJson size={13} strokeWidth={1.7} aria-hidden />诊断文件 {diagnosticAssets.length}
              </button>
              <button
                type="button"
                className="button ghost small"
                onClick={() => setConfirmBatchDelete({
                  ids: diagnosticAssets.map((a) => a.id),
                  title: `清理 ${diagnosticAssets.length} 个诊断 JSON 文件？`,
                  description: '仅清理系统冒烟产生的诊断 JSON 文件，不影响你的图片和视频作品。',
                })}
              >
                清理诊断
              </button>
            </>
          )}
        </div>
      </div>

      {filtered.length ? (
        stackedView && hasMultiStacks ? (
          <div className={styles.stack}>
            <div className={styles.galleryGrid}>
              {stacks.slice(0, limit).flatMap((stack) => {
                const expanded = expandedStacks.has(stack.key);
                if (stack.items.length <= 1 || !expanded) {
                  return [renderTile(stack.cover, stack.items.length > 1 ? { key: stack.key, count: stack.items.length, items: stack.items, expanded } : undefined)];
                }
                return [
                  <div key={`${stack.key}:header`} className={styles.expandedStackBar}>
                    <span><Layers size={14} aria-hidden />同批次共 {stack.items.length} 件{stack.workflowKey ? ` · ${stack.workflowKey}` : ''}</span>
                    <div className="actions">
                      {batchMode && (
                        <button type="button" className="button ghost small" onClick={() => toggleChecked(stack.items.map((i) => i.id))}>
                          {stack.items.every((i) => checkedIds.has(i.id)) ? '取消全选本批' : '全选本批'}
                        </button>
                      )}
                      <button type="button" className="button secondary small" onClick={() => toggleStack(stack.key)}>收起此批</button>
                    </div>
                  </div>,
                  ...stack.items.map((item) => renderTile(item)),
                ];
              })}
            </div>
          </div>
        ) : (
          <div className={styles.galleryGrid}>
            {filtered.slice(0, limit).map((asset) => renderTile(asset))}
          </div>
        )
      ) : (
        <div className="empty">
          <strong>暂无生成作品</strong>
          <span>生成成功的图片和视频会自动保存在这里。</span>
          {onGoCreate && <button className="button small" type="button" onClick={onGoCreate}>去创作</button>}
        </div>
      )}

      {(stackedView && hasMultiStacks ? stacks.length : filtered.length) > limit && (
        <button className="button secondary block" type="button" onClick={() => setLimit(limit + 24)}>加载更多作品</button>
      )}

      {!batchMode && selected && inspectedId === undefined && (
        <div className={styles.floatBar} role="toolbar" aria-label="作品操作">
          <button type="button" onClick={() => onOpen(selected, filtered)}><Eye size={17} strokeWidth={1.6} aria-hidden />预览</button>
          <i />
          <button type="button" disabled={!onReuse || !selected.source_job_id} onClick={() => onReuse && onReuse(selected)}><RotateCcw size={17} strokeWidth={1.6} aria-hidden />再次生成</button>
          <i />
          <button type="button" onClick={() => setConfirmBatchDelete({
            ids: [selected.id],
            title: '删除此作品？',
            description: `将永久删除「${selected.filename || selected.id}」，删除后无法恢复。`,
          })}><Trash2 size={17} strokeWidth={1.6} aria-hidden />删除</button>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(confirmBatchDelete)}
        title={confirmBatchDelete?.title || ''}
        description={confirmBatchDelete?.description}
        confirmLabel="确认删除"
        danger
        busy={deletingBatch}
        onCancel={() => setConfirmBatchDelete(null)}
        onConfirm={() => void executeBatchDelete()}
      />
    </div>
  );
}
