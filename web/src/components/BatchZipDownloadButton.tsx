import { useEffect, useRef, useState } from 'react';
import { Archive, Check, ChevronDown, Loader2 } from 'lucide-react';
import type { ImageDownloadMode } from '../lib/download-prefs';
import styles from '../gpu.module.css';

const MENU_ESTIMATED_HEIGHT = 136;
const MENU_ESTIMATED_WIDTH = 256;

export function batchZipButtonLabel(mode: ImageDownloadMode, hasStrippable: boolean, count: number): string {
  if (!hasStrippable) return `打包下载 ZIP (${count})`;
  return mode === 'clean' ? `无元数据 ZIP (${count})` : `原图 ZIP (${count})`;
}

export function BatchZipDownloadButton({
  count,
  hasStrippable,
  disabled = false,
  busy = false,
  mode,
  onModeChange,
  onDownload,
}: {
  count: number;
  hasStrippable: boolean;
  disabled?: boolean;
  busy?: boolean;
  mode: ImageDownloadMode;
  onModeChange: (next: ImageDownloadMode) => void;
  onDownload: (effectiveMode: ImageDownloadMode) => void;
}) {
  const [open, setOpen] = useState(false);
  const [openUp, setOpenUp] = useState(false);
  const [alignRight, setAlignRight] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const effectiveMode: ImageDownloadMode = hasStrippable ? mode : 'original';
  const isDisabled = disabled || busy || count <= 0;

  useEffect(() => {
    if (isDisabled) setOpen(false);
  }, [isDisabled]);

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

  if (!hasStrippable) {
    return (
      <button
        type="button"
        className="button secondary small"
        disabled={isDisabled}
        onClick={() => onDownload('original')}
      >
        {busy ? <Loader2 size={14} className={styles.spin} aria-hidden /> : <Archive size={14} strokeWidth={1.6} aria-hidden />}
        {busy ? '正在打包…' : batchZipButtonLabel('original', false, count)}
      </button>
    );
  }

  function toggleMenu() {
    if (!open && wrapRef.current) {
      const rect = wrapRef.current.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;
      setOpenUp(spaceBelow < MENU_ESTIMATED_HEIGHT && spaceAbove > spaceBelow);

      const overflowsRight = rect.left + MENU_ESTIMATED_WIDTH > window.innerWidth - 8;
      const fitsRightAligned = rect.right - MENU_ESTIMATED_WIDTH >= 8;
      setAlignRight(overflowsRight && fitsRightAligned);
    }
    setOpen((curr) => !curr);
  }

  function selectAndDownload(next: ImageDownloadMode) {
    onModeChange(next);
    setOpen(false);
    onDownload(next);
  }

  const primaryTitle = effectiveMode === 'clean'
    ? '将选中作品打包为 ZIP，自动去除图片工作流与 EXIF 元数据（视频保持原文件）'
    : '将选中作品打包为 ZIP，保留完整工作流与元数据';

  return (
    <div ref={wrapRef} className={styles.splitDownload} data-open={open}>
      <button
        type="button"
        className={`button secondary small ${styles.splitDownloadMain}`}
        disabled={isDisabled}
        title={primaryTitle}
        onClick={() => onDownload(effectiveMode)}
      >
        {busy ? <Loader2 size={14} className={styles.spin} aria-hidden /> : <Archive size={14} strokeWidth={1.6} aria-hidden />}
        {busy ? '正在打包…' : batchZipButtonLabel(effectiveMode, true, count)}
      </button>
      <button
        type="button"
        className={`button secondary small ${styles.splitDownloadToggle}`}
        disabled={isDisabled}
        aria-label="选择批量打包方式"
        aria-haspopup="menu"
        aria-expanded={open}
        title="切换批量打包方式（记住本次选择）"
        onClick={toggleMenu}
      >
        <ChevronDown size={14} strokeWidth={1.8} aria-hidden />
      </button>

      {open && (
        <div
          className={styles.splitDownloadMenu}
          role="menu"
          aria-label="批量 ZIP 下载方式"
          data-up={openUp ? 'true' : 'false'}
          data-align={alignRight ? 'right' : 'left'}
        >
          <button
            type="button"
            role="menuitemradio"
            aria-checked={effectiveMode === 'clean'}
            className={styles.splitDownloadOption}
            data-active={effectiveMode === 'clean'}
            onClick={() => selectAndDownload('clean')}
          >
            <span className={styles.splitDownloadOptionBody}>
              <strong>无元数据打包 ZIP</strong>
              <small>去除图片工作流与 EXIF · 视频保留原文件</small>
            </span>
            <span className={styles.splitDownloadCheck} aria-hidden>
              {effectiveMode === 'clean' && <Check size={14} strokeWidth={2.2} />}
            </span>
          </button>
          <button
            type="button"
            role="menuitemradio"
            aria-checked={effectiveMode === 'original'}
            className={styles.splitDownloadOption}
            data-active={effectiveMode === 'original'}
            onClick={() => selectAndDownload('original')}
          >
            <span className={styles.splitDownloadOptionBody}>
              <strong>原图打包 ZIP</strong>
              <small>保留完整工作流与元数据</small>
            </span>
            <span className={styles.splitDownloadCheck} aria-hidden>
              {effectiveMode === 'original' && <Check size={14} strokeWidth={2.2} />}
            </span>
          </button>
        </div>
      )}
    </div>
  );
}
