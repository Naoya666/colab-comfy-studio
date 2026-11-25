import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowUpRight, ChevronLeft, ChevronRight, Copy, FileCode2, FolderInput, ImagePlus, Maximize2, Minimize2, RotateCcw, Star, Trash2, ZoomIn, ZoomOut,
} from 'lucide-react';
import {
  apiMessage, assetUrl, fetchNativeOutputBytes,
  type GalleryAsset, type GpuJob, type GpuWorkflow, type NativeOutputAsset,
} from '../lib/gpu-api';
import { extractPromptText } from '../lib/create-runtime-prefs';
import {
  loadImageDownloadMode,
  saveImageDownloadMode,
  type ImageDownloadMode,
} from '../lib/download-prefs';
import { formatInstanceTime } from '../lib/instance-time';
import { paramRows, promptCompositionRows } from '../lib/job-detail-copy';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import { bytesLabel } from '../lib/model-library';
import { extractPng, MAX_IMAGE, type ExtractedWorkflow } from '../lib/png-workflow';
import { GalleryDownloadButton } from './GalleryDownloadButton';
import {
  computeNativeScale,
  IDENTITY_TRANSFORM,
  MAX_ZOOM,
  MIN_ZOOM,
  panBy,
  toggleDoubleClickZoom,
  zoomAtPoint,
  ZOOM_STEP_FACTOR,
  type Size2D,
  type ViewTransform,
} from '../lib/viewer-zoom';
import { workflowTitle } from '../lib/workflow-copy';
import styles from '../gpu.module.css';

export type ViewerItem =
  | { kind: 'studio'; asset: GalleryAsset }
  | { kind: 'native'; asset: NativeOutputAsset };

export function GalleryViewer({
  item,
  workflows = [],
  projectName,
  navCount,
  navIndex,
  inlineViewer,
  viewerJob,
  viewerJobError,
  error,
  importingRef,
  onStep,
  onClose,
  onError,
  onToast,
  onOpenJob,
  onFavoriteStudio,
  onReuseStudio,
  onUseStudioAsReference,
  onMoveStudio,
  onDeleteStudio,
  onFavoriteNative,
  onImportNativeReference,
  onImportWorkflowBundle,
  onDeleteNative,
}: {
  item: ViewerItem;
  workflows?: GpuWorkflow[];
  projectName?: string;
  navCount: number;
  navIndex: number;
  inlineViewer: boolean;
  viewerJob: GpuJob | null;
  viewerJobError: string;
  error: string;
  importingRef?: boolean;
  onStep: (delta: number) => void;
  onClose: () => void;
  onError: (msg: string) => void;
  onToast: (message: string, tone?: 'info' | 'success' | 'error') => void;
  onOpenJob?: (jobId: string) => void;
  onFavoriteStudio: (asset: GalleryAsset) => void;
  onReuseStudio: (asset: GalleryAsset) => void;
  onUseStudioAsReference: (asset: GalleryAsset) => void;
  onMoveStudio?: (asset: GalleryAsset) => void;
  onDeleteStudio: (asset: GalleryAsset) => void;
  onFavoriteNative?: (asset: NativeOutputAsset) => void;
  onImportNativeReference?: (asset: NativeOutputAsset) => Promise<void>;
  onImportWorkflowBundle?: (bundleJson: string) => Promise<void>;
  onDeleteNative?: (asset: NativeOutputAsset) => void;
}) {
  const [extracting, setExtracting] = useState(false);
  const [extracted, setExtracted] = useState<ExtractedWorkflow | null>(null);
  const [extractError, setExtractError] = useState('');
  const [selectedJson, setSelectedJson] = useState<'workflow' | 'prompt'>('workflow');
  const [importingDraft, setImportingDraft] = useState(false);
  const [downloadMode, setDownloadMode] = useState<ImageDownloadMode>(() => loadImageDownloadMode());
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [lightbox, setLightbox] = useState(false);
  const [view, setView] = useState<ViewTransform>(IDENTITY_TRANSFORM);
  const [dragging, setDragging] = useState(false);
  const [smoothTransform, setSmoothTransform] = useState(false);
  const [naturalSize, setNaturalSize] = useState<Size2D>({ width: 0, height: 0 });
  const [stageSize, setStageSize] = useState<Size2D>({ width: 0, height: 0 });
  const headRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const pointersRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinchRef = useRef<{ dist: number; scale: number; midX: number; midY: number } | null>(null);
  const pointerStartRef = useRef<{ id: number; x: number; y: number; moved: boolean } | null>(null);
  const lastTapRef = useRef<{ time: number; x: number; y: number } | null>(null);
  const moveRafRef = useRef<number | null>(null);
  const pendingPanRef = useRef<{ dx: number; dy: number }>({ dx: 0, dy: 0 });
  const pendingPinchRef = useRef<{ dx: number; dy: number; targetScale: number; anchorX: number; anchorY: number } | null>(null);

  useEffect(() => () => {
    if (moveRafRef.current !== null) {
      window.cancelAnimationFrame(moveRafRef.current);
      moveRafRef.current = null;
    }
  }, []);

  const assetId = item.asset.id;
  const mime = item.asset.mime || '';
  const isImage = mime.startsWith('image/');
  const isExpanded = isFullscreen || lightbox;
  const canZoom = isExpanded && isImage;

  useBodyScrollLock(lightbox);

  useEffect(() => {
    const panel = headRef.current?.closest('section') ?? null;
    if (!panel) return;
    if (lightbox) {
      panel.dataset.lightbox = 'true';
    } else {
      delete panel.dataset.lightbox;
    }
    return () => {
      delete panel.dataset.lightbox;
    };
  }, [lightbox]);

  useEffect(() => {
    function syncFullscreen() {
      const panel = headRef.current?.closest('section') ?? null;
      const active = Boolean(panel && document.fullscreenElement === panel);
      setIsFullscreen(active);
      if (!active) {
        setView(IDENTITY_TRANSFORM);
        setDragging(false);
        setSmoothTransform(false);
        pointersRef.current.clear();
        pinchRef.current = null;
        pointerStartRef.current = null;
        lastTapRef.current = null;
      }
    }
    syncFullscreen();
    document.addEventListener('fullscreenchange', syncFullscreen);
    return () => document.removeEventListener('fullscreenchange', syncFullscreen);
  }, []);

  useEffect(() => {
    if (isExpanded) return;
    setView(IDENTITY_TRANSFORM);
    setDragging(false);
    setSmoothTransform(false);
    pointersRef.current.clear();
    pinchRef.current = null;
    pointerStartRef.current = null;
    lastTapRef.current = null;
  }, [isExpanded]);

  useEffect(() => {
    if (!lightbox) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        setLightbox(false);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [lightbox]);

  useEffect(() => {
    setExtracted(null);
    setExtractError('');
    setExtracting(false);
    setView(IDENTITY_TRANSFORM);
    setDragging(false);
    setSmoothTransform(false);
    pointersRef.current.clear();
    pinchRef.current = null;
    pointerStartRef.current = null;
    lastTapRef.current = null;
    if (imgRef.current?.complete && imgRef.current.naturalWidth > 0) {
      setNaturalSize({ width: imgRef.current.naturalWidth, height: imgRef.current.naturalHeight });
    } else {
      setNaturalSize({ width: 0, height: 0 });
    }
  }, [assetId, item.kind]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el || !isExpanded) return;
    if (imgRef.current?.complete && imgRef.current.naturalWidth > 0) {
      setNaturalSize({ width: imgRef.current.naturalWidth, height: imgRef.current.naturalHeight });
    }
    const measure = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setStageSize({ width: rect.width, height: rect.height });
      }
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [isExpanded]);

  const baseImageSize = useMemo<Size2D>(() => {
    if (naturalSize.width > 0 && naturalSize.height > 0 && stageSize.width > 0 && stageSize.height > 0) {
      const fit = Math.min(stageSize.width / naturalSize.width, stageSize.height / naturalSize.height);
      return { width: naturalSize.width * fit, height: naturalSize.height * fit };
    }
    return stageSize;
  }, [naturalSize, stageSize]);

  const getBaseImageSize = useCallback((): Size2D => {
    if (baseImageSize.width > 0 && baseImageSize.height > 0) {
      return baseImageSize;
    }
    const img = imgRef.current;
    if (img && img.offsetWidth > 0 && img.offsetHeight > 0) {
      return { width: img.offsetWidth, height: img.offsetHeight };
    }
    return stageSize;
  }, [baseImageSize, stageSize]);

  const nativeScale = useMemo(
    () => (canZoom ? computeNativeScale(naturalSize, baseImageSize) : null),
    [canZoom, naturalSize, baseImageSize],
  );

  const zoomCenterBy = useCallback((factor: number) => {
    setSmoothTransform(true);
    setView((curr) => zoomAtPoint(curr, curr.scale * factor, 0, 0, getBaseImageSize()));
  }, [getBaseImageSize]);

  const zoomToScale = useCallback((target: number) => {
    setSmoothTransform(true);
    setView((curr) => zoomAtPoint(curr, target, 0, 0, getBaseImageSize()));
  }, [getBaseImageSize]);

  const resetZoom = useCallback(() => {
    setSmoothTransform(true);
    setView(IDENTITY_TRANSFORM);
  }, []);

  // Non-passive wheel zoom anchored at cursor position in fullscreen image mode.
  useEffect(() => {
    const el = stageRef.current;
    if (!el || !canZoom) return;
    function onWheel(event: WheelEvent) {
      event.preventDefault();
      const rect = el!.getBoundingClientRect();
      const cx = event.clientX - (rect.left + rect.width / 2);
      const cy = event.clientY - (rect.top + rect.height / 2);
      const clampedDelta = Math.max(-240, Math.min(240, event.deltaY));
      const factor = Math.exp(-clampedDelta * 0.0022);
      setSmoothTransform(false);
      setView((curr) => zoomAtPoint(curr, curr.scale * factor, cx, cy, getBaseImageSize()));
    }
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [canZoom, getBaseImageSize]);

  // Keyboard zoom shortcuts (+, -, 0) in fullscreen image mode.
  useEffect(() => {
    if (!canZoom) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        zoomCenterBy(ZOOM_STEP_FACTOR);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        zoomCenterBy(1 / ZOOM_STEP_FACTOR);
      } else if (event.key === '0') {
        event.preventDefault();
        resetZoom();
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [canZoom, zoomCenterBy, resetZoom]);

  function handleStagePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (!canZoom) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    const el = stageRef.current;
    if (!el) return;
    el.setPointerCapture(event.pointerId);
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    setSmoothTransform(false);
    if (pointersRef.current.size === 2) {
      pointerStartRef.current = null;
      lastTapRef.current = null;
      const [a, b] = [...pointersRef.current.values()];
      pinchRef.current = {
        dist: Math.hypot(b.x - a.x, b.y - a.y) || 1,
        scale: view.scale,
        midX: (a.x + b.x) / 2,
        midY: (a.y + b.y) / 2,
      };
      setDragging(true);
    } else if (pointersRef.current.size === 1) {
      if (event.pointerType === 'touch') {
        pointerStartRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
      } else {
        pointerStartRef.current = null;
      }
      if (view.scale > MIN_ZOOM + 0.01) {
        setDragging(true);
      }
    }
  }

  function schedulePointerFlush() {
    if (moveRafRef.current !== null) return;
    moveRafRef.current = window.requestAnimationFrame(() => {
      moveRafRef.current = null;
      const pinch = pendingPinchRef.current;
      if (pinch) {
        pendingPinchRef.current = null;
        pendingPanRef.current = { dx: 0, dy: 0 };
        const baseSize = getBaseImageSize();
        setView((curr) => {
          const shifted = panBy(curr, pinch.dx, pinch.dy, baseSize);
          return zoomAtPoint(shifted, pinch.targetScale, pinch.anchorX, pinch.anchorY, baseSize);
        });
        return;
      }
      const { dx, dy } = pendingPanRef.current;
      if (dx !== 0 || dy !== 0) {
        pendingPanRef.current = { dx: 0, dy: 0 };
        setView((curr) => panBy(curr, dx, dy, getBaseImageSize()));
      }
    });
  }

  function handleStagePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!canZoom || !pointersRef.current.has(event.pointerId)) return;
    const prev = pointersRef.current.get(event.pointerId)!;
    const currPt = { x: event.clientX, y: event.clientY };
    pointersRef.current.set(event.pointerId, currPt);

    if (pointersRef.current.size === 2 && pinchRef.current && stageRef.current) {
      const [a, b] = [...pointersRef.current.values()];
      const dist = Math.hypot(b.x - a.x, b.y - a.y) || 1;
      const midX = (a.x + b.x) / 2;
      const midY = (a.y + b.y) / 2;
      const rect = stageRef.current.getBoundingClientRect();
      const anchorX = midX - (rect.left + rect.width / 2);
      const anchorY = midY - (rect.top + rect.height / 2);
      const dx = midX - pinchRef.current.midX;
      const dy = midY - pinchRef.current.midY;
      const targetScale = pinchRef.current.scale * (dist / pinchRef.current.dist);
      pinchRef.current.midX = midX;
      pinchRef.current.midY = midY;
      const prevPinch = pendingPinchRef.current;
      pendingPinchRef.current = {
        dx: (prevPinch?.dx || 0) + dx,
        dy: (prevPinch?.dy || 0) + dy,
        targetScale,
        anchorX,
        anchorY,
      };
      schedulePointerFlush();
      return;
    }

    if (pointersRef.current.size === 1) {
      if (
        pointerStartRef.current
        && pointerStartRef.current.id === event.pointerId
        && Math.hypot(currPt.x - pointerStartRef.current.x, currPt.y - pointerStartRef.current.y) > 10
      ) {
        pointerStartRef.current.moved = true;
      }
      const dx = currPt.x - prev.x;
      const dy = currPt.y - prev.y;
      if (dx !== 0 || dy !== 0) {
        pendingPanRef.current.dx += dx;
        pendingPanRef.current.dy += dy;
        schedulePointerFlush();
      }
    }
  }

  function handleStagePointerUp(event: React.PointerEvent<HTMLDivElement>) {
    const tapStart = pointerStartRef.current;
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) {
      pinchRef.current = null;
      pendingPinchRef.current = null;
    }
    if (pointersRef.current.size === 0) {
      pendingPanRef.current = { dx: 0, dy: 0 };
      if (moveRafRef.current !== null) {
        window.cancelAnimationFrame(moveRafRef.current);
        moveRafRef.current = null;
      }
      setDragging(false);
      pointerStartRef.current = null;
      if (
        event.type === 'pointerup'
        && event.pointerType === 'touch'
        && canZoom
        && stageRef.current
        && tapStart
        && tapStart.id === event.pointerId
        && !tapStart.moved
      ) {
        const now = Date.now();
        const prevTap = lastTapRef.current;
        if (prevTap && now - prevTap.time < 300 && Math.hypot(event.clientX - prevTap.x, event.clientY - prevTap.y) < 28) {
          lastTapRef.current = null;
          const rect = stageRef.current.getBoundingClientRect();
          const tapX = event.clientX - (rect.left + rect.width / 2);
          const tapY = event.clientY - (rect.top + rect.height / 2);
          setSmoothTransform(true);
          setView((curr) => toggleDoubleClickZoom(curr, tapX, tapY, getBaseImageSize()));
        } else {
          lastTapRef.current = { time: now, x: event.clientX, y: event.clientY };
        }
      }
    }
  }

  function handleStageDoubleClick(event: React.MouseEvent<HTMLDivElement>) {
    if (!canZoom || !stageRef.current) return;
    const rect = stageRef.current.getBoundingClientRect();
    const clickX = event.clientX - (rect.left + rect.width / 2);
    const clickY = event.clientY - (rect.top + rect.height / 2);
    setSmoothTransform(true);
    setView((curr) => toggleDoubleClickZoom(curr, clickX, clickY, getBaseImageSize()));
  }

  const title = item.asset.filename || item.asset.id;
  const mediaSrc = item.kind === 'studio' ? assetUrl(item.asset) : item.asset.content_url;

  const handleDownloadModeChange = useCallback((next: ImageDownloadMode) => {
    setDownloadMode(next);
    saveImageDownloadMode(next);
  }, []);

  const viewerJobId = item.kind === 'studio' ? (item.asset.source_job_id || item.asset.gpu_job_id) : undefined;
  const matchedViewerJob = item.kind === 'studio' && viewerJob && viewerJob.id === viewerJobId ? viewerJob : null;
  const viewerPrompt = matchedViewerJob
    ? extractPromptText(matchedViewerJob.resolved_params || matchedViewerJob.params)
    : '';
  const studioWfKey = item.kind === 'studio' ? (matchedViewerJob?.workflow_key || item.asset.workflow_key || '') : '';
  const studioWfVersion = item.kind === 'studio' ? (matchedViewerJob?.workflow_version ?? item.asset.workflow_version) : undefined;
  const studioWfObj = studioWfKey
    ? (workflows.find((w) => w.workflow_key === studioWfKey && w.active) || workflows.find((w) => w.workflow_key === studioWfKey))
    : undefined;
  const studioWfName = studioWfKey ? workflowTitle(studioWfObj || { workflow_key: studioWfKey }) : '';
  const studioMeta = item.kind === 'studio' ? ((item.asset.metadata || {}) as Record<string, unknown>) : {};
  const studioWidth = Number(studioMeta.width ?? (item.kind === 'studio' ? item.asset.width : 0));
  const studioHeight = Number(studioMeta.height ?? (item.kind === 'studio' ? item.asset.height : 0));
  const studioGpu = item.kind === 'studio'
    ? (matchedViewerJob?.provider === 'mock' || item.asset.mock
      ? '离线模拟'
      : (matchedViewerJob?.gpu_reported || item.asset.gpu_reported || ''))
    : '';
  const studioCreatedAt = item.kind === 'studio' ? (item.asset.created_at || matchedViewerJob?.created_at || '') : '';
  const hasStudioInfo = Boolean(projectName || studioWfName || (studioWidth > 0 && studioHeight > 0) || studioGpu || studioCreatedAt || viewerJobId);

  async function handleExtractPng() {
    if (item.kind !== 'native' || extracting) return;
    setExtracting(true);
    setExtractError('');
    setExtracted(null);
    try {
      if (item.asset.size > MAX_IMAGE) {
        throw new Error('图片超过 32 MiB，暂不支持在线解析工作流');
      }
      const bytes = await fetchNativeOutputBytes(item.asset.content_url);
      const parsed = await extractPng(bytes);
      setExtracted(parsed);
      setSelectedJson(parsed.workflow ? 'workflow' : 'prompt');
    } catch (cause) {
      setExtractError(apiMessage(cause));
    } finally {
      setExtracting(false);
    }
  }

  function downloadExtracted(key: 'workflow' | 'prompt') {
    if (!extracted?.[key]) return;
    const blobUrl = URL.createObjectURL(new Blob([JSON.stringify(extracted[key], null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = `${title.replace(/\.png$/i, '')}-${key}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  }

  async function importExtractedDraft() {
    if (!extracted || !onImportWorkflowBundle || importingDraft) return;
    setImportingDraft(true);
    try {
      const safeStem = title.replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 32) || 'native';
      const suffix = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID().slice(0, 8)
        : Date.now().toString(36);
      const payload = JSON.stringify({
        workflow_key: `native-${safeStem}-${suffix}`,
        image_extraction: {
          workflow: extracted.workflow,
          prompt: extracted.prompt,
          metadata: extracted.metadata,
        },
      });
      await onImportWorkflowBundle(payload);
    } finally {
      setImportingDraft(false);
    }
  }

  return (
    <>
      <div ref={headRef} className={`${styles.cardHead} ${styles.viewerHead}`}>
        <h2>{title}</h2>
        <div className={styles.viewerHeadActions}>
          {navCount > 1 && navIndex >= 0 && (
            <div className={styles.viewerNavGroup}>
              <button type="button" className="button secondary small" aria-label="上一张" title="上一张 (←)" onClick={() => onStep(-1)}>
                <ChevronLeft size={14} aria-hidden />
              </button>
              <span className="muted small">{navIndex + 1}/{navCount}</span>
              <button type="button" className="button secondary small" aria-label="下一张" title="下一张 (→)" onClick={() => onStep(1)}>
                <ChevronRight size={14} aria-hidden />
              </button>
            </div>
          )}
          {canZoom && (
            <div className={styles.viewerZoomGroup} role="group" aria-label="图片缩放控制">
              <button
                type="button"
                className="button secondary small"
                aria-label="缩小"
                title="缩小 (-)"
                disabled={view.scale <= MIN_ZOOM + 0.01}
                onClick={() => zoomCenterBy(1 / ZOOM_STEP_FACTOR)}
              >
                <ZoomOut size={14} strokeWidth={1.7} aria-hidden />
              </button>
              <button
                type="button"
                className={`button secondary small ${styles.viewerZoomPct}`}
                aria-label="重置缩放为适应屏幕"
                title="点击复位适应屏幕 (0)"
                onClick={resetZoom}
              >
                {Math.round(view.scale * 100)}%
              </button>
              <button
                type="button"
                className="button secondary small"
                aria-label="放大"
                title="以屏幕中心放大 (+)"
                disabled={view.scale >= MAX_ZOOM - 0.01}
                onClick={() => zoomCenterBy(ZOOM_STEP_FACTOR)}
              >
                <ZoomIn size={14} strokeWidth={1.7} aria-hidden />
              </button>
              {nativeScale !== null && (
                <button
                  type="button"
                  className="button secondary small"
                  title={Math.abs(view.scale - nativeScale) < 0.08 ? '恢复适应屏幕' : '查看 1:1 原始像素尺寸'}
                  onClick={() => (Math.abs(view.scale - nativeScale) < 0.08 ? resetZoom() : zoomToScale(nativeScale))}
                >
                  {Math.abs(view.scale - nativeScale) < 0.08 ? '适应' : '1:1'}
                </button>
              )}
            </div>
          )}
          {isExpanded && (
            <>
              {item.kind === 'studio' ? (
                <button className="button secondary small" type="button" onClick={() => onFavoriteStudio(item.asset)}>
                  <Star size={14} strokeWidth={1.6} fill={item.asset.favorite ? 'currentColor' : 'none'} aria-hidden />
                  {item.asset.favorite ? '已收藏' : '收藏'}
                </button>
              ) : onFavoriteNative ? (
                <button className="button secondary small" type="button" onClick={() => onFavoriteNative(item.asset)}>
                  <Star size={14} strokeWidth={1.6} fill={item.asset.favorite ? 'currentColor' : 'none'} aria-hidden />
                  {item.asset.favorite ? '已收藏' : '收藏'}
                </button>
              ) : null}
              <GalleryDownloadButton
                item={item}
                mode={downloadMode}
                onModeChange={handleDownloadModeChange}
                placement="down"
              />
            </>
          )}
          <button
            className="button secondary small"
            type="button"
            onClick={async (event) => {
              const panel = event.currentTarget.closest('section');
              if (document.fullscreenElement) {
                try {
                  await document.exitFullscreen();
                } catch (cause) {
                  onError(apiMessage(cause));
                }
                return;
              }
              if (lightbox) {
                setLightbox(false);
                return;
              }
              if (panel?.requestFullscreen) {
                try {
                  await panel.requestFullscreen();
                  return;
                } catch {
                  // Fall back to CSS lightbox when Fullscreen API is unavailable or rejected (e.g. iOS Safari).
                }
              }
              setLightbox(true);
            }}
          >
            {isExpanded ? (
              <><Minimize2 size={14} strokeWidth={1.6} aria-hidden />退出全屏</>
            ) : (
              <><Maximize2 size={14} strokeWidth={1.6} aria-hidden />全屏查看</>
            )}
          </button>
          <button
            className="button secondary small"
            type="button"
            onClick={() => {
              if (document.fullscreenElement) void document.exitFullscreen().catch(() => {});
              setLightbox(false);
              onClose();
            }}
          >
            关闭
          </button>
        </div>
      </div>

      <div
        ref={stageRef}
        className={styles.viewerStage}
        data-zoomable={canZoom}
        data-zoomed={canZoom && view.scale > MIN_ZOOM + 0.01}
        data-dragging={canZoom && dragging}
        onPointerDown={handleStagePointerDown}
        onPointerMove={handleStagePointerMove}
        onPointerUp={handleStagePointerUp}
        onPointerCancel={handleStagePointerUp}
        onDoubleClick={handleStageDoubleClick}
      >
        <div className={styles.viewerViewport}>
          {mime.startsWith('video/') ? (
            <video className={styles.viewerVideo} src={mediaSrc} controls autoPlay playsInline />
          ) : isImage ? (
            <img
              ref={imgRef}
              className={styles.viewerImage}
              src={mediaSrc}
              alt=""
              draggable={false}
              data-smooth={canZoom && smoothTransform}
              style={canZoom ? { transform: `translate3d(${view.x}px, ${view.y}px, 0) scale(${view.scale})` } : undefined}
              onLoad={(event) => {
                const target = event.currentTarget;
                setNaturalSize({ width: target.naturalWidth, height: target.naturalHeight });
              }}
            />
          ) : (
            <p className="muted">这是诊断文件，不能直接预览，请下载查看。</p>
          )}
        </div>
      </div>

      {!isExpanded && item.kind === 'studio' && hasStudioInfo && (
        <details className="foldable" open={inlineViewer}>
          <summary>作品信息</summary>
          <div className="foldContent">
            <dl className={styles.kv}>
              {projectName && (
                <><dt>所属项目</dt><dd>{projectName}</dd></>
              )}
              {studioWfName && (
                <>
                  <dt>工作流</dt>
                  <dd title={studioWfKey}>
                    {studioWfName}
                    {studioWfVersion != null && ` · v${studioWfVersion}`}
                    {studioWfKey && studioWfName !== studioWfKey && <span className="muted small"> ({studioWfKey})</span>}
                  </dd>
                </>
              )}
              {studioWidth > 0 && studioHeight > 0 && (
                <><dt>分辨率</dt><dd>{studioWidth} × {studioHeight}</dd></>
              )}
              {studioGpu && (
                <><dt>运行环境</dt><dd>{studioGpu}</dd></>
              )}
              {studioCreatedAt && (
                <><dt>生成时间</dt><dd>{formatInstanceTime(studioCreatedAt)}</dd></>
              )}
              {viewerJobId && (
                <>
                  <dt>来源任务</dt>
                  <dd>
                    {onOpenJob ? (
                      <button type="button" className={styles.inlineLinkBtn} onClick={() => onOpenJob(viewerJobId)}>
                        #{viewerJobId.slice(0, 8)}<ArrowUpRight size={13} aria-hidden />
                      </button>
                    ) : `#${viewerJobId.slice(0, 8)}`}
                  </dd>
                </>
              )}
            </dl>
          </div>
        </details>
      )}

      {!isExpanded && item.kind === 'studio' && viewerJobId && (
        matchedViewerJob ? (
          <details className="foldable" open={inlineViewer}>
            <summary>生成参数</summary>
            <div className="foldContent">
              {viewerPrompt && (
                <button
                  type="button"
                  className="button secondary small"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(viewerPrompt);
                      onToast('已复制提示词', 'success');
                    } catch {
                      onToast('无法访问剪贴板，请手动复制', 'error');
                    }
                  }}
                >
                  <Copy size={13} aria-hidden />复制提示词
                </button>
              )}
              <dl className={styles.kv}>
                {promptCompositionRows(matchedViewerJob.prompt_composition).map((row) => (
                  <Fragment key={row.key}><dt>{row.label}</dt><dd>{row.value}</dd></Fragment>
                ))}
                {paramRows(matchedViewerJob.resolved_params || matchedViewerJob.params || {}).map((row) => (
                  <Fragment key={row.key}><dt>{row.label}</dt><dd>{row.value}</dd></Fragment>
                ))}
              </dl>
            </div>
          </details>
        ) : (
          <p className="muted small">{viewerJobError || '正在读取生成参数…'}</p>
        )
      )}

      {!isExpanded && item.kind === 'native' && (
        <details className="foldable" open={inlineViewer}>
          <summary>原生作品信息（Drive）</summary>
          <div className="foldContent">
            <dl className={styles.kv}>
              <dt>来源目录</dt><dd>{item.asset.subfolder ? `output/${item.asset.subfolder}` : 'output/'}</dd>
              {item.asset.width && item.asset.height && <><dt>分辨率</dt><dd>{item.asset.width} × {item.asset.height}</dd></>}
              <dt>文件大小</dt><dd>{bytesLabel(item.asset.size)}</dd>
              {item.asset.modified_at && <><dt>修改时间</dt><dd>{formatInstanceTime(item.asset.modified_at)}</dd></>}
              {item.asset.account_alias && <><dt>Drive 账号</dt><dd>{item.asset.account_alias}</dd></>}
            </dl>
          </div>
        </details>
      )}

      {!isExpanded && <div className={styles.viewerActions}>
        {item.kind === 'studio' ? (
          <>
            <button className="button secondary small" type="button" onClick={() => onFavoriteStudio(item.asset)}>
              <Star size={14} strokeWidth={1.6} fill={item.asset.favorite ? 'currentColor' : 'none'} aria-hidden />
              {item.asset.favorite ? '取消收藏' : '收藏'}
            </button>
            <button className="button secondary small" type="button" disabled={!viewerJobId} onClick={() => onReuseStudio(item.asset)}>
              <RotateCcw size={14} strokeWidth={1.6} aria-hidden />用同样参数再来一次
            </button>
            {(mime.startsWith('image/') || mime.startsWith('video/')) && (
              <button className="button secondary small" type="button" onClick={() => onUseStudioAsReference(item.asset)}>
                <ImagePlus size={14} strokeWidth={1.6} aria-hidden />用作参考图
              </button>
            )}
            {onMoveStudio && (
              <button className="button secondary small" type="button" onClick={() => onMoveStudio(item.asset)}>
                <FolderInput size={14} strokeWidth={1.6} aria-hidden />移动到项目
              </button>
            )}
            <GalleryDownloadButton
              item={item}
              mode={downloadMode}
              onModeChange={handleDownloadModeChange}
            />
            <button className="button danger small" type="button" onClick={() => onDeleteStudio(item.asset)}>
              <Trash2 size={14} strokeWidth={1.6} aria-hidden />删除
            </button>
          </>
        ) : (
          <>
            {onFavoriteNative && (
              <button className="button secondary small" type="button" onClick={() => onFavoriteNative(item.asset)}>
                <Star size={14} strokeWidth={1.6} fill={item.asset.favorite ? 'currentColor' : 'none'} aria-hidden />
                {item.asset.favorite ? '取消收藏' : '收藏'}
              </button>
            )}
            {onImportNativeReference && (mime.startsWith('image/') || mime.startsWith('video/')) && (
              <button
                className="button secondary small"
                type="button"
                disabled={importingRef}
                onClick={() => void onImportNativeReference(item.asset)}
              >
                <ImagePlus size={14} strokeWidth={1.6} aria-hidden />
                {importingRef ? '导入中…' : '导入为参考素材'}
              </button>
            )}
            {mime === 'image/png' && (
              <button
                className="button secondary small"
                type="button"
                disabled={extracting}
                onClick={() => void handleExtractPng()}
              >
                <FileCode2 size={14} strokeWidth={1.6} aria-hidden />
                {extracting ? '解析中…' : '提取 PNG 工作流'}
              </button>
            )}
            <GalleryDownloadButton
              item={item}
              mode={downloadMode}
              onModeChange={handleDownloadModeChange}
            />
            {onDeleteNative && (
              <button className="button danger small" type="button" onClick={() => onDeleteNative(item.asset)}>
                <Trash2 size={14} strokeWidth={1.6} aria-hidden />移至回收站
              </button>
            )}
          </>
        )}
      </div>}

      {!isExpanded && extractError && <p className="status error" role="alert">{extractError}</p>}

      {!isExpanded && extracted && (
        <div className={styles.inlinePanel} aria-label="PNG 工作流提取结果">
          <div className={styles.cardHead}>
            <strong>已提取 ComfyUI 工作流</strong>
            <span className="pill">{extracted.node_count} 节点</span>
          </div>
          <p className="muted small">
            workflow：{extracted.workflow ? '已检测' : '未检测'} · API prompt：{extracted.prompt ? '已检测' : '未检测'}
          </p>
          {extracted.model_files.length > 0 && (
            <p className="small">模型文件线索：{extracted.model_files.join('、')}</p>
          )}
          {extracted.warnings.map((w) => <p key={w} className="status">{w}</p>)}
          <div className="actions">
            {(['workflow', 'prompt'] as const).filter((k) => extracted[k]).map((k) => (
              <button key={k} type="button" className="button secondary small" onClick={() => downloadExtracted(k)}>
                下载 {k}.json
              </button>
            ))}
            <button
              type="button"
              className="button secondary small"
              onClick={async () => {
                const payload = extracted[selectedJson] ? JSON.stringify(extracted[selectedJson], null, 2) : '';
                try {
                  await navigator.clipboard.writeText(payload);
                  onToast('已复制工作流 JSON', 'success');
                } catch {
                  onToast('无法访问剪贴板，请手动下载 JSON', 'error');
                }
              }}
            >
              <Copy size={13} aria-hidden />复制 JSON
            </button>
            {onImportWorkflowBundle && (
              <button
                type="button"
                className="button small"
                disabled={importingDraft}
                onClick={() => void importExtractedDraft()}
              >
                {importingDraft ? '导入中…' : '导入为待配置草稿'}
              </button>
            )}
          </div>
        </div>
      )}

      {error && <p className="status error" role="alert">{error}</p>}
    </>
  );
}
