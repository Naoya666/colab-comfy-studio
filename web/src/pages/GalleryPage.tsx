import { useCallback, useMemo, useState } from 'react';
import {
  Image as ImageIcon, Film, Search, Star, Check, Eye, RotateCcw, Trash2, Play,
  Layers, CheckSquare, X, FileJson, RefreshCw, Workflow, Folder, FolderInput,
} from 'lucide-react';
import { assetUrl, type GalleryAsset, type GpuWorkflow, type Project } from '../lib/gpu-api';
import { groupGalleryAssets, isDiagnosticAsset } from '../lib/history-grouping';
import { projectOptionLabel, projectTitleById } from '../lib/project-label';
import { workflowTitle } from '../lib/workflow-copy';
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
  gallery,
  allGallery,
  projects = [],
  projectFilter = '',
  onProjectFilterChange,
  workflows = [],
  error,
  refreshing = false,
  onRefresh,
  onFavorite,
  onBatchDelete,
  onRequestMove,
  onOpen,
  onReuse,
  onGoCreate,
  inspectedId,
}: {
  gallery: GalleryAsset[];
  allGallery?: GalleryAsset[];
  projects?: Project[];
  projectFilter?: string;
  onProjectFilterChange?: (projectId: string) => void;
  workflows?: GpuWorkflow[];
  error?: string;
  refreshing?: boolean;
  onRefresh?: () => void;
  onFavorite: (asset: GalleryAsset) => void;
  onBatchDelete: (ids: string[]) => Promise<{ removedIds: string[]; error?: string }>;
  onRequestMove?: (ids: string[], opts?: { currentProjectId?: string; onMoved?: (movedIds: string[]) => void }) => void;
  onOpen: (asset: GalleryAsset, ordered?: GalleryAsset[]) => void;
  onReuse?: (asset: GalleryAsset) => void;
  onGoCreate?: () => void;
  inspectedId?: string;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [workflowFilter, setWorkflowFilter] = useState('');
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [stackedView, setStackedView] = useState(true);
  const [expandedStacks, setExpandedStacks] = useState<Set<string>>(() => new Set());
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(24);
  const [batchMode, setBatchMode] = useState(false);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(() => new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [confirmBatchDelete, setConfirmBatchDelete] = useState<{ ids: string[]; title: string; description: string; error?: string } | null>(null);
  const [deletingBatch, setDeletingBatch] = useState(false);

  const workflowByKey = useMemo(() => {
    const map = new Map<string, GpuWorkflow>();
    for (const wf of workflows) {
      if (!map.has(wf.workflow_key) || wf.active) map.set(wf.workflow_key, wf);
    }
    return map;
  }, [workflows]);

  const workflowLabel = useCallback((key?: string) => {
    if (!key) return '';
    return workflowTitle(workflowByKey.get(key) || { workflow_key: key });
  }, [workflowByKey]);

  const diagnosticAssets = useMemo(() => gallery.filter(isDiagnosticAsset), [gallery]);
  const mediaCount = gallery.length - diagnosticAssets.length;
  const projectCounts = useMemo(() => {
    const source = allGallery || gallery;
    const map = new Map<string, number>();
    let totalMedia = 0;
    for (const asset of source) {
      if (isDiagnosticAsset(asset)) continue;
      totalMedia += 1;
      if (asset.project_id) map.set(asset.project_id, (map.get(asset.project_id) || 0) + 1);
    }
    return { map, totalMedia };
  }, [allGallery, gallery]);
  const extraProjectIds = useMemo(() => {
    const known = new Set(projects.map((p) => p.id));
    return [...projectCounts.map.keys()].filter((id) => !known.has(id));
  }, [projects, projectCounts.map]);
  const showProjectTag = !projectFilter;

  const workflowOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const asset of gallery) {
      if (isDiagnosticAsset(asset) || !asset.workflow_key) continue;
      counts.set(asset.workflow_key, (counts.get(asset.workflow_key) || 0) + 1);
    }
    return [...counts.entries()]
      .map(([key, count]) => ({ key, label: workflowLabel(key), count }))
      .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
  }, [gallery, workflowLabel]);

  const activeWorkflowFilter = workflowOptions.some((o) => o.key === workflowFilter) ? workflowFilter : '';

  const filtered = useMemo(() => gallery.filter((asset) => {
    const isDiag = isDiagnosticAsset(asset);
    if (isDiag && !showDiagnostics) return false;
    const isVideo = asset.mime.startsWith('video/');
    const isImage = asset.mime.startsWith('image/');
    if (filter === 'image' && !isImage) return false;
    if (filter === 'video' && !isVideo) return false;
    if (filter === 'favorite' && !asset.favorite) return false;
    if (filter === 'unfavorited' && asset.favorite) return false;
    if (activeWorkflowFilter && asset.workflow_key !== activeWorkflowFilter) return false;
    const wfName = workflowLabel(asset.workflow_key);
    const projName = projectTitleById(asset.project_id, projects);
    return `${asset.filename || ''} ${asset.source_job_id || ''} ${asset.workflow_key || ''} ${wfName} ${projName}`.toLowerCase().includes(query.toLowerCase());
  }), [gallery, filter, activeWorkflowFilter, query, showDiagnostics, workflowLabel, projects]);

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
    if (!confirmBatchDelete || deletingBatch) return;
    setDeletingBatch(true);
    try {
      const { removedIds, error } = await onBatchDelete(confirmBatchDelete.ids);
      const removed = new Set(removedIds);
      setCheckedIds((curr) => {
        const next = new Set(curr);
        removed.forEach((id) => next.delete(id));
        return next;
      });
      if (selectedId && removed.has(selectedId)) setSelectedId(null);
      if (error) {
        const ids = confirmBatchDelete.ids.filter((id) => !removed.has(id));
        setConfirmBatchDelete({
          ...confirmBatchDelete,
          ids,
          title: removed.size ? `删除剩余的 ${ids.length} 件作品？` : confirmBatchDelete.title,
          error,
        });
      } else {
        setConfirmBatchDelete(null);
      }
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
            {(() => {
              const wfName = workflowLabel(asset.workflow_key);
              const projName = showProjectTag ? projectTitleById(asset.project_id, projects) : '';
              const primary = wfName || asset.filename || `作品 ${asset.id.slice(0, 6)}`;
              const versionTag = asset.workflow_version != null ? `v${asset.workflow_version}` : '';
              const jobTag = asset.source_job_id ? `任务 ${asset.source_job_id.slice(0, 6)}` : '';
              const secondary = isStackedCover && stackInfo
                ? [projName, `同批共 ${stackInfo.count} 件`, versionTag, jobTag].filter(Boolean).join(' · ')
                : wfName
                  ? [projName, versionTag, jobTag || '作品', asset.filename].filter(Boolean).join(' · ')
                  : [projName, jobTag || '上传素材'].filter(Boolean).join(' · ');
              const fullTitle = [
                wfName ? `${wfName}${versionTag ? ` ${versionTag}` : ''}` : '',
                asset.filename || asset.id,
              ].filter(Boolean).join(' · ');
              return (
                <>
                  <strong title={fullTitle}>{primary}</strong>
                  <small className="muted" title={secondary}>{secondary}</small>
                </>
              );
            })()}
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
          {onRefresh && (
            <button
              type="button"
              className="button secondary small"
              disabled={refreshing}
              onClick={onRefresh}
            >
              <RefreshCw size={14} className={refreshing ? styles.spin : undefined} aria-hidden />
              {refreshing ? '刷新中…' : '刷新'}
            </button>
          )}
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
            {onRequestMove && projects.length > 0 && (
              projects.length > 1 || activeCheckedIds.some((id) => gallery.find((a) => a.id === id)?.project_id !== projects[0]?.id)
            ) && (
              <button
                type="button"
                className="button secondary small"
                disabled={!activeCheckedIds.length}
                onClick={() => onRequestMove(activeCheckedIds, {
                  currentProjectId: activeCheckedIds.length === 1
                    ? gallery.find((a) => a.id === activeCheckedIds[0])?.project_id
                    : undefined,
                  onMoved: (movedIds) => {
                    const moved = new Set(movedIds);
                    setCheckedIds((curr) => {
                      const next = new Set(curr);
                      moved.forEach((id) => next.delete(id));
                      return next;
                    });
                  },
                })}
              >
                <FolderInput size={14} aria-hidden />移动到项目 ({activeCheckedIds.length})
              </button>
            )}
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
          {onProjectFilterChange && (projects.length > 0 || extraProjectIds.length > 0) && (
            <label className={`small muted ${styles.filterSelectWrap}`}>
              <Folder size={13} aria-hidden />
              <select
                aria-label="按项目筛选作品"
                className={styles.miniSelect}
                value={projectFilter}
                onChange={(e) => { onProjectFilterChange(e.target.value); setLimit(24); }}
              >
                <option value="">全部项目 ({projectCounts.totalMedia})</option>
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>
                    {projectOptionLabel(p, projects)}{p.is_default ? ' · 默认' : ''} ({projectCounts.map.get(p.id) || 0})
                  </option>
                ))}
                {extraProjectIds.map((pid) => (
                  <option key={pid} value={pid}>
                    {projectTitleById(pid, projects)} ({projectCounts.map.get(pid) || 0})
                  </option>
                ))}
              </select>
            </label>
          )}
          {workflowOptions.length > 0 && (
            <label className={`small muted ${styles.filterSelectWrap}`}>
              <Workflow size={13} aria-hidden />
              <select
                aria-label="按工作流筛选"
                className={styles.miniSelect}
                value={activeWorkflowFilter}
                onChange={(e) => { setWorkflowFilter(e.target.value); setLimit(24); }}
              >
                <option value="">全部工作流</option>
                {workflowOptions.map((opt) => (
                  <option key={opt.key} value={opt.key}>{opt.label} ({opt.count})</option>
                ))}
              </select>
            </label>
          )}
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
                    <span><Layers size={14} aria-hidden />同批次共 {stack.items.length} 件{stack.workflowKey ? ` · ${workflowLabel(stack.workflowKey)}` : ''}</span>
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
          {onRequestMove && projects.some((p) => p.id !== selected.project_id) && (
            <>
              <i />
              <button type="button" onClick={() => onRequestMove([selected.id], { currentProjectId: selected.project_id })}><FolderInput size={17} strokeWidth={1.6} aria-hidden />移动</button>
            </>
          )}
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
        error={confirmBatchDelete?.error}
        onCancel={() => setConfirmBatchDelete(null)}
        onConfirm={() => void executeBatchDelete()}
      />
    </div>
  );
}
