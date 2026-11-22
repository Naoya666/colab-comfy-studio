import { Fragment, useEffect, useState } from 'react';
import {
  ArrowUpRight, ChevronLeft, ChevronRight, Copy, Download, FileCode2, ImagePlus, Maximize2, RotateCcw, Star, Trash2,
} from 'lucide-react';
import {
  apiMessage, assetUrl, fetchNativeOutputBytes,
  type GalleryAsset, type GpuJob, type GpuWorkflow, type NativeOutputAsset,
} from '../lib/gpu-api';
import { extractPromptText } from '../lib/create-runtime-prefs';
import { formatInstanceTime } from '../lib/instance-time';
import { paramRows } from '../lib/job-detail-copy';
import { bytesLabel } from '../lib/model-library';
import { extractPng, MAX_IMAGE, type ExtractedWorkflow } from '../lib/png-workflow';
import { workflowTitle } from '../lib/workflow-copy';
import styles from '../gpu.module.css';

export type ViewerItem =
  | { kind: 'studio'; asset: GalleryAsset }
  | { kind: 'native'; asset: NativeOutputAsset };

export function GalleryViewer({
  item,
  workflows = [],
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
  onDeleteStudio,
  onFavoriteNative,
  onImportNativeReference,
  onImportWorkflowBundle,
  onDeleteNative,
}: {
  item: ViewerItem;
  workflows?: GpuWorkflow[];
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

  const assetId = item.asset.id;
  useEffect(() => {
    setExtracted(null);
    setExtractError('');
    setExtracting(false);
  }, [assetId, item.kind]);

  const title = item.asset.filename || item.asset.id;
  const mime = item.asset.mime || '';
  const mediaSrc = item.kind === 'studio' ? assetUrl(item.asset) : item.asset.content_url;
  const downloadHref = item.kind === 'studio'
    ? assetUrl(item.asset)
    : `${item.asset.content_url}${item.asset.content_url.includes('?') ? '&' : '?'}download=true`;

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
  const hasStudioInfo = Boolean(studioWfName || (studioWidth > 0 && studioHeight > 0) || studioGpu || studioCreatedAt || viewerJobId);

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
      <div className={`${styles.cardHead} ${styles.viewerHead}`}>
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
          <button
            className="button secondary small"
            type="button"
            onClick={async (event) => {
              const panel = event.currentTarget.closest('section');
              try {
                if (document.fullscreenElement) await document.exitFullscreen();
                else if (panel?.requestFullscreen) await panel.requestFullscreen();
                else throw new Error('当前浏览器不支持全屏');
              } catch (cause) {
                onError(apiMessage(cause));
              }
            }}
          >
            <Maximize2 size={14} strokeWidth={1.6} aria-hidden />全屏查看
          </button>
          <button
            className="button secondary small"
            type="button"
            onClick={() => {
              if (document.fullscreenElement) void document.exitFullscreen().catch(() => {});
              onClose();
            }}
          >
            关闭
          </button>
        </div>
      </div>

      {mime.startsWith('video/') ? (
        <video className={styles.viewerVideo} src={mediaSrc} controls autoPlay playsInline />
      ) : mime.startsWith('image/') ? (
        <img className={styles.viewerImage} src={mediaSrc} alt="" />
      ) : (
        <p className="muted">这是诊断文件，不能直接预览，请下载查看。</p>
      )}

      {item.kind === 'studio' && hasStudioInfo && (
        <details className="foldable" open={inlineViewer}>
          <summary>作品信息</summary>
          <div className="foldContent">
            <dl className={styles.kv}>
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

      {item.kind === 'studio' && viewerJobId && (
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

      {item.kind === 'native' && (
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

      <div className={styles.viewerActions}>
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
            <a className="button secondary small" href={downloadHref} download={title} target="_blank" rel="noreferrer">
              <Download size={14} strokeWidth={1.6} aria-hidden />下载
            </a>
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
            <a className="button secondary small" href={downloadHref} download={title} target="_blank" rel="noreferrer">
              <Download size={14} strokeWidth={1.6} aria-hidden />下载
            </a>
            {onDeleteNative && (
              <button className="button danger small" type="button" onClick={() => onDeleteNative(item.asset)}>
                <Trash2 size={14} strokeWidth={1.6} aria-hidden />移至回收站
              </button>
            )}
          </>
        )}
      </div>

      {extractError && <p className="status error" role="alert">{extractError}</p>}

      {extracted && (
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
