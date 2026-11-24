import { useRef, useState } from 'react';
import { FileCode2, Link2, Loader2, Upload } from 'lucide-react';
import styles from '../gpu.module.css';

type SecondaryMode = 'none' | 'url' | 'paste';

export interface WorkflowSourceDropzoneProps {
  title?: string;
  description: string;
  busy?: boolean;
  busyLabel?: string;
  urlPlaceholder?: string;
  urlSubmitLabel?: string;
  urlBusyLabel?: string;
  pasteLabel?: string;
  pastePlaceholder?: string;
  pasteSubmitLabel?: string;
  pasteBusyLabel?: string;
  onSelectFile: (file: File) => void | Promise<void>;
  onSubmitUrl: (url: string) => boolean | void | Promise<boolean | void>;
  onSubmitPaste: (text: string) => boolean | void | Promise<boolean | void>;
}

export function WorkflowSourceDropzone({
  title = '拖入或选择 ComfyUI 工作流文件（.json / .png 原图）',
  description,
  busy = false,
  busyLabel = '解析中…',
  urlPlaceholder = 'https://...（支持 .json 或含元数据的 PNG 原图直链，自动转换 GitHub / HF / Civitai 常见链接）',
  urlSubmitLabel = '解析链接',
  urlBusyLabel = '拉取解析中…',
  pasteLabel = '粘贴 JSON 文本',
  pastePlaceholder = '粘贴工作流 JSON',
  pasteSubmitLabel = '解析 JSON',
  pasteBusyLabel = '处理中…',
  onSelectFile,
  onSubmitUrl,
  onSubmitPaste,
}: WorkflowSourceDropzoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [mode, setMode] = useState<SecondaryMode>('none');
  const [urlText, setUrlText] = useState('');
  const [pasteText, setPasteText] = useState('');

  async function submitUrl(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !urlText.trim()) return;
    const ok = await onSubmitUrl(urlText);
    if (ok !== false) {
      setUrlText('');
      setMode('none');
    }
  }

  async function submitPaste() {
    if (busy || !pasteText.trim()) return;
    const ok = await onSubmitPaste(pasteText);
    if (ok !== false) {
      setPasteText('');
      setMode('none');
    }
  }

  return (
    <div className={styles.stack}>
      <div
        className={`${styles.depDropzone} ${dragActive ? styles.dropActive : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          const file = e.dataTransfer.files?.[0];
          if (file) void onSelectFile(file);
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          hidden
          accept=".json,application/json,.png,image/png"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            if (file) void onSelectFile(file);
          }}
        />
        <Upload size={20} strokeWidth={1.6} aria-hidden />
        <strong>{title}</strong>
        <span className="muted small">{description}</span>
        <div className="actions">
          <button
            type="button"
            className="button small"
            disabled={busy}
            onClick={() => fileInputRef.current?.click()}
          >
            {busy ? <><Loader2 size={13} className={styles.spin} aria-hidden /> {busyLabel}</> : '选择 JSON / PNG 文件'}
          </button>
          <button
            type="button"
            className="button secondary small"
            disabled={busy}
            aria-expanded={mode === 'url'}
            onClick={() => setMode((curr) => (curr === 'url' ? 'none' : 'url'))}
          >
            <Link2 size={13} aria-hidden />
            {mode === 'url' ? '收起 URL 输入' : '从 URL 导入'}
          </button>
          <button
            type="button"
            className="button secondary small"
            disabled={busy}
            aria-expanded={mode === 'paste'}
            onClick={() => setMode((curr) => (curr === 'paste' ? 'none' : 'paste'))}
          >
            <FileCode2 size={13} aria-hidden />
            {mode === 'paste' ? '收起粘贴框' : pasteLabel}
          </button>
        </div>
      </div>

      {mode === 'url' && (
        <div className={styles.depSubPanel}>
          <form className={`${styles.depInlineUrlForm} ${styles.wfImportUrlRow}`} onSubmit={(e) => void submitUrl(e)}>
            <input
              type="url"
              inputMode="url"
              autoFocus
              aria-label="工作流 JSON 或 PNG 原图 HTTPS 链接"
              placeholder={urlPlaceholder}
              value={urlText}
              disabled={busy}
              onChange={(e) => setUrlText(e.target.value)}
            />
            <button
              type="submit"
              className="button small"
              disabled={busy || !urlText.trim()}
            >
              {busy ? <><Loader2 size={13} className={styles.spin} aria-hidden /> {urlBusyLabel}</> : urlSubmitLabel}
            </button>
          </form>
          <span className="muted small">
            由服务端安全拉取并仅在内存中解析，不经本地设备中转，不落盘且不保存原始链接或原图。
          </span>
        </div>
      )}

      {mode === 'paste' && (
        <div className={styles.depSubPanel}>
          <textarea
            className={styles.code}
            aria-label={pasteLabel}
            placeholder={pastePlaceholder}
            value={pasteText}
            disabled={busy}
            onChange={(e) => setPasteText(e.target.value)}
          />
          <div className="actions">
            <button
              type="button"
              className="button small"
              disabled={busy || !pasteText.trim()}
              onClick={() => void submitPaste()}
            >
              {busy ? pasteBusyLabel : pasteSubmitLabel}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
