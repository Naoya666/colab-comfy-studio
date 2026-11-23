import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  apiMessage,
  BATCH_DELETE_ASSET_LIMIT,
  BATCH_MOVE_ASSET_LIMIT,
  batchDeleteAssets,
  batchMoveAssets,
  deleteAsset,
  deleteNativeOutput,
  favoriteNativeOutput,
  getGpuGallery,
  getGpuJob,
  importNativeOutputAsReference,
  updateAsset,
  type GalleryAsset,
  type GpuJob,
  type NativeOutputAsset,
  type Project,
} from '../lib/gpu-api';
import { projectTitleById } from '../lib/project-label';
import type { ViewerItem } from '../components/GalleryViewer';
import type { MainTab } from '../lib/route-state';

export type GallerySource = 'studio' | 'native';

export function useGalleryState({
  projectId,
  projects,
  loggedIn,
  tab,
  desktop,
  galleryOutputs,
  initialRouteAssetId,
  pushToast,
  onLoadedAgainJob,
  onUseReference,
}: {
  projectId: string;
  projects: Project[];
  loggedIn: boolean;
  tab: MainTab;
  desktop: boolean;
  galleryOutputs: string;
  initialRouteAssetId: string;
  pushToast: (message: string, tone?: 'info' | 'success' | 'error') => void;
  onLoadedAgainJob: (job: GpuJob) => void;
  onUseReference: (asset: GalleryAsset) => void;
}) {
  const [gallerySource, setGallerySource] = useState<GallerySource>('studio');
  const [galleryProjectFilter, setGalleryProjectFilter] = useState('');
  const [gallery, setGallery] = useState<GalleryAsset[]>([]);
  const [galleryError, setGalleryError] = useState('');
  const [galleryRevision, setGalleryRevision] = useState(0);
  const [refreshingGallery, setRefreshingGallery] = useState(false);

  const [viewerSelection, setViewer] = useState<GalleryAsset | null>(null);
  const [viewerList, setViewerList] = useState<GalleryAsset[]>([]);
  const [pendingRouteAssetId, setPendingRouteAssetId] = useState(initialRouteAssetId);
  const [confirmDeleteViewer, setConfirmDeleteViewer] = useState<GalleryAsset | null>(null);
  const [viewerJob, setViewerJob] = useState<GpuJob | null>(null);
  const [viewerJobError, setViewerJobError] = useState('');

  const [nativeViewerSelection, setNativeViewer] = useState<NativeOutputAsset | null>(null);
  const [nativeViewerList, setNativeViewerList] = useState<NativeOutputAsset[]>([]);
  const [nativeFavOverrides, setNativeFavOverrides] = useState<Record<string, boolean>>({});
  const [nativeDeletedIds, setNativeDeletedIds] = useState<Set<string>>(() => new Set());
  const [nativeRefreshToken, setNativeRefreshToken] = useState(0);
  const [importingNativeRef, setImportingNativeRef] = useState(false);
  const [confirmDeleteNative, setConfirmDeleteNative] = useState<NativeOutputAsset | null>(null);

  const createProjectGallery = useMemo(
    () => (projectId ? gallery.filter((asset) => asset.project_id === projectId) : []),
    [gallery, projectId],
  );
  const projectGallery = useMemo(
    () => (galleryProjectFilter ? gallery.filter((asset) => asset.project_id === galleryProjectFilter) : gallery),
    [gallery, galleryProjectFilter],
  );
  const viewer = viewerSelection
    ? gallery.find((asset) => asset.id === viewerSelection.id) ?? viewerSelection
    : null;
  const nativeViewer = useMemo(() => {
    if (!nativeViewerSelection || nativeDeletedIds.has(nativeViewerSelection.id)) return null;
    return Object.prototype.hasOwnProperty.call(nativeFavOverrides, nativeViewerSelection.id)
      ? { ...nativeViewerSelection, favorite: nativeFavOverrides[nativeViewerSelection.id] }
      : nativeViewerSelection;
  }, [nativeViewerSelection, nativeFavOverrides, nativeDeletedIds]);

  const activeViewerItem: ViewerItem | null = tab === '作品' && gallerySource === 'native'
    ? (nativeViewer ? { kind: 'native', asset: nativeViewer } : null)
    : (viewer ? { kind: 'studio', asset: viewer } : null);

  const routeAssetId = tab === '作品' && gallerySource === 'studio' ? pendingRouteAssetId || viewer?.id || '' : '';
  const viewerJobId = viewer?.source_job_id || viewer?.gpu_job_id;
  const inlineViewer = desktop && tab === '作品';

  const studioNavList = useMemo(() => {
    const base = viewerList.length ? viewerList.filter((a) => gallery.some((g) => g.id === a.id)) : projectGallery;
    return base.map((item) => gallery.find((g) => g.id === item.id) || item);
  }, [viewerList, gallery, projectGallery]);

  const nativeNavList = useMemo(
    () => nativeViewerList
      .filter((a) => !nativeDeletedIds.has(a.id))
      .map((a) => (Object.prototype.hasOwnProperty.call(nativeFavOverrides, a.id) ? { ...a, favorite: nativeFavOverrides[a.id] } : a)),
    [nativeViewerList, nativeDeletedIds, nativeFavOverrides],
  );

  const activeNavCount = activeViewerItem?.kind === 'native' ? nativeNavList.length : studioNavList.length;
  const activeNavIndex = activeViewerItem?.kind === 'native'
    ? nativeNavList.findIndex((a) => a.id === activeViewerItem.asset.id)
    : activeViewerItem?.kind === 'studio'
      ? studioNavList.findIndex((a) => a.id === activeViewerItem.asset.id)
      : -1;

  useEffect(() => {
    if (!pendingRouteAssetId || !gallery.length) return;
    const found = gallery.find((a) => a.id === pendingRouteAssetId);
    if (found) {
      setGallerySource('studio');
      if (galleryProjectFilter && found.project_id !== galleryProjectFilter) {
        setGalleryProjectFilter('');
      }
      setViewer(found);
      setViewerList(galleryProjectFilter && found.project_id === galleryProjectFilter ? projectGallery : gallery);
      setPendingRouteAssetId('');
    }
  }, [pendingRouteAssetId, gallery, projectGallery, galleryProjectFilter]);

  const lastLoadedKey = useRef<string | null>(null);
  const needsGallery = tab === '作品' || (tab === '创作' && desktop) || Boolean(pendingRouteAssetId);

  useEffect(() => {
    if (!loggedIn) {
      lastLoadedKey.current = null;
      return;
    }
    if (!needsGallery) return;
    const fetchKey = `${galleryOutputs}|${galleryRevision}`;
    if (lastLoadedKey.current === fetchKey) return;
    const controller = new AbortController();
    let alive = true;
    setGalleryError('');
    setRefreshingGallery(true);
    getGpuGallery(undefined, controller.signal).then((next) => {
      if (alive) {
        setGallery(next);
        lastLoadedKey.current = fetchKey;
      }
    }).catch((cause) => {
      if (alive && !controller.signal.aborted) setGalleryError(apiMessage(cause));
    }).finally(() => {
      if (alive) setRefreshingGallery(false);
    });
    return () => {
      alive = false;
      controller.abort();
    };
  }, [loggedIn, needsGallery, galleryOutputs, galleryRevision]);

  const refreshGallery = useCallback(() => {
    setGalleryRevision((revision) => revision + 1);
  }, []);

  useEffect(() => {
    if (!viewerJobId || !loggedIn) {
      setViewerJob(null);
      setViewerJobError('');
      return;
    }
    let alive = true;
    setViewerJobError('');
    getGpuJob(viewerJobId).then((job) => {
      if (alive && job.id === viewerJobId) setViewerJob(job);
    }).catch((cause) => {
      if (alive) setViewerJobError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [viewerJobId, loggedIn]);

  const closeViewers = useCallback(() => {
    setViewer(null);
    setNativeViewer(null);
  }, []);

  const stepActiveViewer = useCallback((delta: number) => {
    if (!activeViewerItem || activeNavIndex < 0 || activeNavCount < 2) return;
    const nextIndex = (activeNavIndex + delta + activeNavCount) % activeNavCount;
    if (activeViewerItem.kind === 'native') {
      setNativeViewer(nativeNavList[nextIndex]);
    } else {
      setViewer(studioNavList[nextIndex]);
    }
  }, [activeViewerItem, activeNavIndex, activeNavCount, nativeNavList, studioNavList]);

  async function reuseAsset(asset: GalleryAsset) {
    const sourceJobId = asset.source_job_id || asset.gpu_job_id;
    if (!sourceJobId) {
      setGalleryError('该素材没有来源任务记录。');
      return;
    }
    setGalleryError('');
    try {
      const job = await getGpuJob(sourceJobId);
      onLoadedAgainJob(job);
      setViewer((current) => (current?.id === asset.id ? null : current));
    } catch (cause) {
      setGalleryError(`再次生成：${apiMessage(cause)}`);
    }
  }

  async function favorite(asset: GalleryAsset) {
    setGalleryError('');
    try {
      const next = await updateAsset(asset.id, !asset.favorite);
      setGallery((curr) => curr.map((i) => (i.id === next.id ? { ...i, ...next } : i)));
      setViewer((current) => (current?.id === next.id ? { ...current, ...next } : current));
      pushToast(next.favorite ? '已收藏作品' : '已取消收藏', 'info');
    } catch (cause) {
      setGalleryError(apiMessage(cause));
    }
  }

  async function remove(asset: GalleryAsset) {
    setGalleryError('');
    try {
      await deleteAsset(asset.id);
      setGallery((curr) => curr.filter((i) => i.id !== asset.id));
      setViewer((current) => (current?.id === asset.id ? null : current));
      pushToast('已删除作品', 'success');
    } catch (cause) {
      setGalleryError(`删除失败：${apiMessage(cause)}`);
    }
  }

  async function batchRemove(ids: string[]) {
    const removedIds = new Set<string>();
    let batchError = '';
    setGalleryError('');
    try {
      for (let offset = 0; offset < ids.length; offset += BATCH_DELETE_ASSET_LIMIT) {
        const batch = ids.slice(offset, offset + BATCH_DELETE_ASSET_LIMIT);
        const result = await batchDeleteAssets(batch);
        result.ids.forEach((id) => removedIds.add(id));
      }
    } catch (cause) {
      const remaining = await getGpuGallery().catch(() => null);
      if (remaining) {
        const existing = new Set(remaining.map((asset) => asset.id));
        ids.filter((id) => !existing.has(id)).forEach((id) => removedIds.add(id));
      }
      if (removedIds.size < ids.length) batchError = `批量删除失败：${apiMessage(cause)}`;
    }
    if (removedIds.size > 0) {
      setGallery((current) => current.filter((asset) => !removedIds.has(asset.id)));
      setViewer((current) => (current && removedIds.has(current.id) ? null : current));
      setGalleryRevision((revision) => revision + 1);
    }
    if (!batchError) pushToast(`已批量删除 ${removedIds.size} 件作品`, 'success');
    return { removedIds: [...removedIds], error: batchError };
  }

  async function moveAssets(ids: string[], targetProjectId: string) {
    const movedIds = new Set<string>();
    let moveError = '';
    setGalleryError('');
    try {
      for (let offset = 0; offset < ids.length; offset += BATCH_MOVE_ASSET_LIMIT) {
        const batch = ids.slice(offset, offset + BATCH_MOVE_ASSET_LIMIT);
        const result = await batchMoveAssets(batch, targetProjectId);
        result.ids.forEach((id) => movedIds.add(id));
      }
    } catch (cause) {
      moveError = `移动失败：${apiMessage(cause)}`;
    }
    if (movedIds.size > 0) {
      setGallery((current) => current.map((asset) => (movedIds.has(asset.id) ? { ...asset, project_id: targetProjectId } : asset)));
      setViewer((current) => (current && movedIds.has(current.id) ? { ...current, project_id: targetProjectId } : current));
      setGalleryRevision((revision) => revision + 1);
    }
    if (!moveError && movedIds.size > 0) {
      const label = projectTitleById(targetProjectId, projects) || '目标项目';
      pushToast(`已将 ${movedIds.size} 件作品移动到「${label}」`, 'success');
    }
    return { movedIds: [...movedIds], error: moveError };
  }

  async function favoriteNative(asset: NativeOutputAsset) {
    setGalleryError('');
    const nextVal = !asset.favorite;
    setNativeFavOverrides((curr) => ({ ...curr, [asset.id]: nextVal }));
    try {
      const updated = await favoriteNativeOutput(asset.id, nextVal);
      setNativeFavOverrides((curr) => ({ ...curr, [updated.id]: updated.favorite }));
      setNativeViewer((curr) => (curr?.id === updated.id ? { ...curr, favorite: updated.favorite } : curr));
      pushToast(updated.favorite ? '已收藏原生作品' : '已取消收藏', 'info');
    } catch (cause) {
      setNativeFavOverrides((curr) => ({ ...curr, [asset.id]: asset.favorite }));
      const msg = apiMessage(cause);
      setGalleryError(msg);
      pushToast(msg, 'error');
    }
  }

  async function importNativeRef(asset: NativeOutputAsset) {
    if (!projectId || importingNativeRef) {
      if (!projectId) pushToast('请先选择一个项目', 'error');
      return;
    }
    setImportingNativeRef(true);
    setGalleryError('');
    try {
      const imported = await importNativeOutputAsReference(asset.id, projectId);
      closeViewers();
      onUseReference(imported);
    } catch (cause) {
      const msg = `导入参考素材失败：${apiMessage(cause)}`;
      setGalleryError(msg);
      pushToast(msg, 'error');
    } finally {
      setImportingNativeRef(false);
    }
  }

  async function removeNative(asset: NativeOutputAsset) {
    setGalleryError('');
    try {
      await deleteNativeOutput(asset.id);
      setNativeDeletedIds((curr) => new Set([...curr, asset.id]));
      setNativeViewer((curr) => (curr?.id === asset.id ? null : curr));
      setNativeRefreshToken((n) => n + 1);
      pushToast('已将原生作品移入 Google Drive 回收站', 'success');
    } catch (cause) {
      const msg = `删除失败：${apiMessage(cause)}`;
      setGalleryError(msg);
      pushToast(msg, 'error');
    }
  }

  async function batchRemoveNative(ids: string[]) {
    const removedIds = new Set<string>();
    let batchError = '';
    setGalleryError('');
    for (const id of ids) {
      try {
        await deleteNativeOutput(id);
        removedIds.add(id);
      } catch (cause) {
        batchError = `批量移至回收站失败：${apiMessage(cause)}`;
        break;
      }
    }
    if (removedIds.size > 0) {
      setNativeDeletedIds((curr) => new Set([...curr, ...removedIds]));
      setNativeViewer((curr) => (curr && removedIds.has(curr.id) ? null : curr));
      setNativeRefreshToken((n) => n + 1);
    }
    if (!batchError) pushToast(`已将 ${removedIds.size} 件原生作品移入 Google Drive 回收站`, 'success');
    return { removedIds: [...removedIds], error: batchError };
  }

  return {
    gallerySource,
    setGallerySource,
    galleryProjectFilter,
    setGalleryProjectFilter,
    allGallery: gallery,
    createProjectGallery,
    projectGallery,
    galleryError,
    setGalleryError,
    refreshingGallery,
    refreshGallery,
    viewer,
    setViewer,
    setViewerList,
    nativeViewer,
    setNativeViewer,
    setNativeViewerList,
    nativeFavOverrides,
    nativeDeletedIds,
    nativeRefreshToken,
    importingNativeRef,
    activeViewerItem,
    activeNavCount,
    activeNavIndex,
    routeAssetId,
    inlineViewer,
    viewerJob,
    viewerJobError,
    confirmDeleteViewer,
    setConfirmDeleteViewer,
    confirmDeleteNative,
    setConfirmDeleteNative,
    setPendingRouteAssetId,
    closeViewers,
    stepActiveViewer,
    reuseAsset,
    favorite,
    remove,
    batchRemove,
    moveAssets,
    favoriteNative,
    importNativeRef,
    removeNative,
    batchRemoveNative,
  };
}
