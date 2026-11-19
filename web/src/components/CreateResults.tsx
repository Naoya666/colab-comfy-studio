import { useState } from 'react';
import { ArrowUpRight, Film, Image as ImageIcon, Maximize2 } from 'lucide-react';
import { assetUrl, type GalleryAsset, type GpuJob } from '../lib/gpu-api';
import { jobStageLabel, jobStateLabel } from '../lib/job-stage-rows';
import { progressPercent, terminalStates } from '../lib/ui-helpers';
import styles from '../gpu.module.css';

function isMedia(asset: GalleryAsset) {
  return asset.mime.startsWith('image/') || asset.mime.startsWith('video/');
}

export function CreateResults({ jobs, count, gallery, error, onOpen, onJobs }: {
  jobs: GpuJob[];
  count: number;
  gallery: GalleryAsset[];
  error: string;
  onOpen: (asset: GalleryAsset) => void;
  onJobs: () => void;
}) {
  const [selectedId, setSelectedId] = useState('');
  const jobIds = new Set(jobs.map(job => job.id));
  const outputIds = new Set(jobs.flatMap(job => job.output_asset_ids || []));
  const outputs = gallery.filter(asset => isMedia(asset) && (
    outputIds.has(asset.id) || jobIds.has(asset.source_job_id || asset.gpu_job_id || '')
  ));
  const latest = outputs.find(asset => asset.id === selectedId) ?? outputs[0];
  const previous = !latest ? gallery.find(isMedia) : undefined;
  const preview = latest ?? previous;
  const running = jobs.find(job => !terminalStates.has(String(job.state).toLowerCase()));
  const finished = jobs.filter(job => terminalStates.has(String(job.state).toLowerCase())).length;
  const progress = progressPercent(running?.progress);
  const status = running
    ? `${running.stage ? jobStageLabel(running.stage) : jobStateLabel(running.state)}${progress === null ? '' : ` · ${progress}%`}`
    : finished === count
      ? count === 1 ? jobStateLabel(jobs[0]?.state) : `${finished}/${count} 个任务已结束`
      : '等待任务状态更新';

  return (
    <section className={`${styles.card} ${styles.createResults}`} aria-label="结果预览">
      <div className={styles.cardHead}>
        <h2>结果预览</h2>
        <button type="button" className={styles.linkButton} onClick={onJobs}>查看任务<ArrowUpRight size={15} aria-hidden /></button>
      </div>
      {error && <p className="status error" role="alert">{error}</p>}
      {count > 0 && <div className={styles.resultStatus} role="status">
        <span>{status}</span>
        {count > 1 && finished < count && <span>已结束 {finished}/{count}</span>}
      </div>}
      <div className={styles.resultCanvas}>
        {preview ? preview.mime.startsWith('video/') ? (
          <video key={preview.id} src={assetUrl(preview)} controls playsInline preload="metadata" aria-label={preview.filename || '结果视频'} />
        ) : (
          <button type="button" onClick={() => onOpen(preview)} aria-label={`查看 ${preview.filename || '结果图片'}`}>
            <img src={assetUrl(preview)} alt={preview.filename || '结果图片'} />
          </button>
        ) : (
          <div className={styles.resultEmpty}>
            <ImageIcon size={28} strokeWidth={1.4} aria-hidden />
            <span>{!count ? '生成后在这里预览' : finished === count ? '本次没有可预览的作品' : '等待首个结果'}</span>
          </div>
        )}
      </div>
      {preview && <div className={styles.resultStatus}>
        <span>{previous && count > 0 ? '此前的作品' : preview.filename || '生成作品'}</span>
        <button type="button" className={styles.linkButton} onClick={() => onOpen(preview)}><Maximize2 size={14} aria-hidden />查看作品</button>
      </div>}
      {outputs.length > 1 && <div className={styles.resultStrip} aria-label="本批次作品">
        {outputs.map((asset, index) => (
          <button key={asset.id} type="button" aria-pressed={preview?.id === asset.id}
            aria-label={`结果 ${index + 1}${asset.filename ? ` · ${asset.filename}` : ''}`} onClick={() => setSelectedId(asset.id)}>
            {asset.mime.startsWith('video/') ? <Film size={18} aria-hidden /> : <img src={assetUrl(asset, true)} alt="" loading="lazy" />}
          </button>
        ))}
      </div>}
    </section>
  );
}
