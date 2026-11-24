import { useState } from 'react';
import { Upload, Check } from 'lucide-react';
import type { GalleryAsset, GpuManifest, Project } from '../lib/gpu-api';
import { projectTitleById } from '../lib/project-label';
import styles from '../gpu.module.css';

export function referenceAssets(assets: GalleryAsset[]): GalleryAsset[] {
  return assets.filter((asset) => !asset.mock && asset.metadata?.mock !== true && /^(image|video|audio)\//.test(asset.mime));
}

export function supportsReferences(manifest: GpuManifest | null): boolean {
  if (!manifest) return false;
  const declaration = manifest.assets as { required?: unknown[]; optional?: unknown[] } | undefined;
  if (declaration?.required?.length || declaration?.optional?.length) return true;
  const params = Array.isArray(manifest.parameters) ? manifest.parameters : Object.values(manifest.parameters || {});
  return params.some((p) => ['image', 'video', 'file', 'audio'].includes(String(p.type)) || p.widget === 'file');
}

export function ReferencePicker({ manifest, assets, selected, currentProjectId, projects = [], onChange, onUpload }: {
  manifest: GpuManifest | null; assets: GalleryAsset[]; selected: string[];
  currentProjectId?: string; projects?: Project[];
  onChange: (ids: string[]) => void; onUpload: (file: File) => Promise<void>;
}) {
  const [dragging, setDragging] = useState(false);

  if (!manifest || !supportsReferences(manifest)) return null;
  const available = referenceAssets(assets);
  const toggle = (id: string) => onChange(selected.includes(id) ? selected.filter((item) => item !== id) : [...new Set([...selected, id])]);

  async function handleFiles(files: FileList | null) {
    if (!files?.length) return;
    for (const file of Array.from(files)) {
      if (/^(image|video|audio)\//.test(file.type)) {
        await onUpload(file);
      }
    }
  }

  return (
    <div
      role="group"
      aria-label="参考素材"
      className={`${styles.refBox} ${dragging ? styles.dropActive : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false); }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        void handleFiles(e.dataTransfer.files);
      }}
    >
      <div className={styles.refHead}>
        <span className="muted small">点选<span className="desktopOnly">、拖拽或 Ctrl+V 粘贴</span>参考图</span>
        <label className="button secondary small">
          <Upload size={14} strokeWidth={1.6} aria-hidden />上传
          <input hidden type="file" multiple accept="image/*,video/*,audio/*" onChange={async (event) => {
            const files = event.target.files;
            await handleFiles(files);
            event.target.value = '';
          }} />
        </label>
      </div>
      {available.length ? (
        <div className={styles.refGrid}>
          {available.map((asset) => {
            const on = selected.includes(asset.id);
            const fromOther = Boolean(currentProjectId && asset.project_id && asset.project_id !== currentProjectId);
            const projLabel = fromOther ? projectTitleById(asset.project_id, projects) : '';
            const label = projLabel ? `${asset.filename || asset.id}（${projLabel}）` : (asset.filename || asset.id);
            return (
              <label className={styles.refItem} data-on={on} key={asset.id} title={label}>
                <input type="checkbox" hidden aria-label={`选用 ${label}`} checked={on} onChange={() => toggle(asset.id)} />
                {asset.mime.startsWith('image/')
                  ? <img src={asset.thumbnail_url || `/api/assets/${asset.id}/thumbnail`} alt="" loading="lazy" />
                  : <span className={styles.refMedia}>{asset.mime.startsWith('video/') ? '视频' : '音频'}</span>}
                {on && <span className={styles.refCheck}><Check size={12} strokeWidth={2.4} aria-hidden /></span>}
              </label>
            );
          })}
        </div>
      ) : <p className="muted small">还没有素材，<span className="desktopOnly">拖入文件、Ctrl+V 粘贴，或</span>点右上角「上传」添加。</p>}
    </div>
  );
}
