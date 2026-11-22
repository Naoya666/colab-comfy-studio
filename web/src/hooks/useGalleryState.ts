import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  apiMessage,
  BATCH_DELETE_ASSET_LIMIT,
  batchDeleteAssets,
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
} from '../lib/gpu-api';
import type { ViewerItem } from '../components/GalleryViewer';
import type { MainTab } from '../lib/route-state';

export type GallerySource = 'studio' | 'native';

export function useGalleryState({
  projectId,
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
  loggedIn: boolean;
  tab: MainTab;
  desktop: boolean;
  galleryOutputs: string;
  initialRouteAssetId: string;
  pushToast: (message: string, tone?: 'info' | 'success' | 'error') => void;
  onLoadedAgainJob: (job: GpuJob) => void;
  onUseReference: (asset: GalleryAsset) => void;
}) {
  const currentProjectId = useRef(projectId);
  currentProjectId.current = projectId;

  const [gallerySource, setGallerySource] = useState<GallerySource>('studio');
  const [gallery, setGallery] = useState<GalleryAsset[]>([]);
  const [galleryError, setGalleryError] = useState('');
  const [galleryRevision, setGalleryRevision] = useState(0);

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

  const projectGallery = useMemo(
    () => gallery.filter((asset) => asset.project_id === projectId),
    [gallery, projectId],
  );
  const viewer = viewerSelection?.project_id === projectId
    ? projectGallery.find((asset) => asset.id === viewerSelection.id) ?? viewerSelection
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
    const base = viewerList.length ? viewerList.filter((a) => projectGallery.some((g) => g.id === a.id)) : projectGallery;
    return base.map((item) => projectGallery.find((g) => g.id === item.id) || item);
  }, [viewerList, projectGallery]);

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
    if (!pendingRouteAssetId || !projectGallery.length) return;
    const found = projectGallery.find((a) => a.id === pendingRouteAssetId);
    if (found) {
      setGallerySource('studio');
      setViewer(found);
      setViewerList(projectGallery);
      setPendingRouteAssetId('');
    }
  }, [pendingRouteAssetId, projectGallery]);

  useEffect(() => {
    if (!projectId || !loggedIn) return;
    let alive = true;
    setGalleryError('');
    getGpuGallery(projectId).then((next) => {
      if (alive) setGallery(next.map((asset) => (asset.project_id ? asset : { ...asset, project_id: projectId })));
    }).catch((cause) => {
      if (alive) setGalleryError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [projectId, loggedIn, tab, galleryOutputs, galleryRevision]);

  useEffect(() => {
    if (!inlineViewer || !viewerJobId || !projectId || !loggedIn) return;
    let alive = true;
    setViewerJobError('');
    getGpuJob(viewerJobId).then((job) => {
      if (alive && job.project_id === projectId) setViewerJob(job);
    }).catch((cause) => {
      if (alive) setViewerJobError(apiMessage(cause));
    });
    return () => { alive = false; };
  }, [inlineViewer, viewerJobId, projectId, loggedIn]);

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
      if (currentProjectId.current !== asset.project_id) return;
      onLoadedAgainJob(job);
      setViewer((current) => (current?.id === asset.id ? null : current));
    } catch (cause) {
      if (currentProjectId.current === asset.project_id) setGalleryError(`再次生成：${apiMessage(cause)}`);
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
      if (currentProjectId.current === asset.project_id) setGalleryError(apiMessage(cause));
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
      if (currentProjectId.current === asset.project_id) setGalleryError(`删除失败：${apiMessage(cause)}`);
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
      const remaining = await getGpuGallery(projectId).catch(() => null);
      if (remaining) {
        const existing = new Set(remaining.map((asset) => asset.id));
        ids.filter((id) => !existing.has(id)).forEach((id) => removedIds.add(id));
      }
      if (removedIds.size < ids.length) batchError = `批量删除失败：${apiMessage(cause)}`;
    }
    if (removedIds.size > 0) {
      setGallery((current) => current.filter((asset) => !removedIds.has(asset.id)));
      setViewer((current) => (current && removedIds.has(current.id) ? null : current));
      if (currentProjectId.current === projectId) setGalleryRevision((revision) => revision + 1);
    }
    if (!batchError) pushToast(`已批量删除 ${removedIds.size} 件作品`, 'success');
    return { removedIds: [...removedIds], error: batchError };
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

  return {
    gallerySource,
    setGallerySource,
    projectGallery,
    galleryError,
    setGalleryError,
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
    favoriteNative,
    importNativeRef,
    removeNative,
  };
}
