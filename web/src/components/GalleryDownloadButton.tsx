import { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, Download } from 'lucide-react';
import {
  buildGalleryDownloadTarget,
  imageDownloadButtonLabel,
  type DownloadableGalleryItem,
  type ImageDownloadMode,
} from '../lib/download-prefs';
import styles from '../gpu.module.css';

const MENU_ESTIMATED_HEIGHT = 136;
const MENU_ESTIMATED_WIDTH = 244;

export function GalleryDownloadButton({
  item,
  mode,
  onModeChange,
  placement = 'auto',
}: {
  item: DownloadableGalleryItem;
  mode: ImageDownloadMode;
  onModeChange: (next: ImageDownloadMode) => void;
  placement?: 'down' | 'auto';
}) {
  const [open, setOpen] = useState(false);
  const [openUp, setOpenUp] = useState(false);
  const [alignRight, setAlignRight] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const primary = buildGalleryDownloadTarget(item, mode);
  const originalTarget = buildGalleryDownloadTarget(item, 'original');
  const cleanTarget = buildGalleryDownloadTarget(item, 'clean');

  useEffect(() => {
    setOpen(false);
  }, [item.kind, item.asset.id, placement]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        setOpen(false);
      }
    }
    document.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('keydown', onKeyDown, true);
    };
  }, [open]);

  if (!primary.strippable) {
    return (
      <a
        className="button secondary small"
        href={originalTarget.href}
        download={originalTarget.filename}
        target="_blank"
        rel="noreferrer"
      >
        <Download size={14} strokeWidth={1.6} aria-hidden />下载
      </a>
    );
  }

  function toggleMenu() {
    if (!open && wrapRef.current) {
      const rect = wrapRef.current.getBoundingClientRect();
      const boundsEl = wrapRef.current.closest('section, [role="dialog"]');
      const bounds = boundsEl?.getBoundingClientRect();
      const containerTop = bounds ? Math.max(0, bounds.top) : 0;
      const containerBottom = bounds ? Math.min(window.innerHeight, bounds.bottom) : window.innerHeight;
      const containerLeft = bounds ? Math.max(0, bounds.left) : 0;
      const containerRight = bounds ? Math.min(window.innerWidth, bounds.right) : window.innerWidth;

      const spaceBelow = containerBottom - rect.bottom;
      const spaceAbove = rect.top - containerTop;
      setOpenUp(placement === 'auto' && spaceBelow < MENU_ESTIMATED_HEIGHT && spaceAbove > spaceBelow);

      const overflowsRight = rect.left + MENU_ESTIMATED_WIDTH > containerRight - 8;
      const fitsRightAligned = rect.right - MENU_ESTIMATED_WIDTH >= containerLeft + 8;
      setAlignRight(overflowsRight && fitsRightAligned);
    }
    setOpen((curr) => !curr);
  }

  function handleOptionSelect(next: ImageDownloadMode) {
    onModeChange(next);
    setOpen(false);
  }

  const primaryTitle = primary.effectiveMode === 'clean'
    ? '下载不含工作流与 EXIF 元数据的图片（文件名自动加 _clean 后缀）'
    : '下载原始图片（保留工作流与元数据）';

  return (
    <div ref={wrapRef} className={styles.splitDownload} data-open={open}>
      <a
        className={`button secondary small ${styles.splitDownloadMain}`}
        href={primary.href}
        download={primary.filename}
        target="_blank"
        rel="noreferrer"
        title={primaryTitle}
      >
        <Download size={14} strokeWidth={1.6} aria-hidden />
        {imageDownloadButtonLabel(primary.effectiveMode)}
      </a>
      <button
        type="button"
        className={`button secondary small ${styles.splitDownloadToggle}`}
        aria-label="选择下载方式"
        aria-haspopup="menu"
        aria-expanded={open}
        title="切换下载方式（记住本次选择）"
        onClick={toggleMenu}
      >
        <ChevronDown size={14} strokeWidth={1.8} aria-hidden />
      </button>

      {open && (
        <div
          className={styles.splitDownloadMenu}
          role="menu"
          aria-label="图片下载方式"
          data-up={openUp ? 'true' : 'false'}
          data-align={alignRight ? 'right' : 'left'}
        >
          <a
            role="menuitemradio"
            aria-checked={primary.effectiveMode === 'clean'}
            className={styles.splitDownloadOption}
            data-active={primary.effectiveMode === 'clean'}
            href={cleanTarget.href}
            download={cleanTarget.filename}
            target="_blank"
            rel="noreferrer"
            onClick={() => handleOptionSelect('clean')}
          >
            <span className={styles.splitDownloadOptionBody}>
              <strong>无元数据下载</strong>
              <small>去除工作流与 EXIF · 文件名加 _clean</small>
            </span>
            <span className={styles.splitDownloadCheck} aria-hidden>
              {primary.effectiveMode === 'clean' && <Check size={14} strokeWidth={2.2} />}
            </span>
          </a>
          <a
            role="menuitemradio"
            aria-checked={primary.effectiveMode === 'original'}
            className={styles.splitDownloadOption}
            data-active={primary.effectiveMode === 'original'}
            href={originalTarget.href}
            download={originalTarget.filename}
            target="_blank"
            rel="noreferrer"
            onClick={() => handleOptionSelect('original')}
          >
            <span className={styles.splitDownloadOptionBody}>
              <strong>下载原图</strong>
              <small>保留完整工作流与元数据</small>
            </span>
            <span className={styles.splitDownloadCheck} aria-hidden>
              {primary.effectiveMode === 'original' && <Check size={14} strokeWidth={2.2} />}
            </span>
          </a>
        </div>
      )}
    </div>
  );
}
