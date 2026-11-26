import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Cpu, X } from 'lucide-react';
import type { GpuInstance } from './AccountDirectory';
import type { GpuJob, GpuSettings, GpuUsage, GpuWorkflow } from '../lib/gpu-api';
import { buildGpuStatusSummary, type GpuStatusActionTarget } from '../lib/gpu-status-summary';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import styles from './GpuStatusPopover.module.css';

export interface GpuStatusPopoverProps {
  open: boolean;
  desktop: boolean;
  settings: GpuSettings | null;
  instances: GpuInstance[];
  jobs: GpuJob[];
  usage: GpuUsage | null;
  workflows: GpuWorkflow[];
  onToggle: () => void;
  onClose: () => void;
  onAction: (action: GpuStatusActionTarget) => void;
}

export function GpuStatusPopover({
  open,
  desktop,
  settings,
  instances,
  jobs,
  usage,
  workflows,
  onToggle,
  onClose,
  onAction,
}: GpuStatusPopoverProps) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLElement>(null);
  const [nowSec, setNowSec] = useState(() => Math.floor(Date.now() / 1000));
  const [anchorPos, setAnchorPos] = useState<{ top: number; right: number }>({ top: 64, right: 24 });

  const summary = useMemo(
    () => buildGpuStatusSummary({ settings, instances, jobs, usage, workflows, nowSec }),
    [settings, instances, jobs, usage, workflows, nowSec],
  );

  useBodyScrollLock(open && !desktop);

  const syncAnchorPosition = useCallback(() => {
    const el = triggerRef.current;
    if (!el || typeof window === 'undefined') return;
    const rect = el.getBoundingClientRect();
    setAnchorPos({
      top: Math.round(rect.bottom + 8),
      right: Math.max(12, Math.round(window.innerWidth - rect.right)),
    });
  }, []);

  useLayoutEffect(() => {
    if (!open || !desktop) return;
    syncAnchorPosition();
    window.addEventListener('resize', syncAnchorPosition);
    window.addEventListener('scroll', syncAnchorPosition, true);
    return () => {
      window.removeEventListener('resize', syncAnchorPosition);
      window.removeEventListener('scroll', syncAnchorPosition, true);
    };
  }, [open, desktop, syncAnchorPosition, summary.pillLabel]);

  // Tick local countdowns every second while open, or every 15s while closed so retention minutes stay current.
  useEffect(() => {
    if (!summary.needsCountdownTick) return;
    setNowSec(Math.floor(Date.now() / 1000));
    const intervalMs = open ? 1000 : 15000;
    const timer = window.setInterval(() => {
      setNowSec(Math.floor(Date.now() / 1000));
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [open, summary.needsCountdownTick]);

  // Close desktop popover on outside pointer interaction.
  useEffect(() => {
    if (!open || !desktop) return;
    function onPointerDown(event: PointerEvent) {
      const target = event.target as Node | null;
      if (!target) return;
      if (triggerRef.current?.contains(target)) return;
      if (popoverRef.current?.contains(target)) return;
      onClose();
    }
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [open, desktop, onClose]);

  function handleAction(action: GpuStatusActionTarget) {
    onClose();
    onAction(action);
  }

  const panelContent = (
    <div className={styles.panelBody}>
      <div className={styles.panelHead}>
        <div className={styles.panelTitle}>
          <span
            className={styles.dot}
            data-tone={summary.tone}
            data-pulsing={summary.pulsing || undefined}
            aria-hidden
          />
          <span>{summary.panelTitle}</span>
        </div>
        <button
          type="button"
          className={styles.closeBtn}
          aria-label="关闭实时状态"
          onClick={onClose}
        >
          <X size={16} strokeWidth={1.7} aria-hidden />
        </button>
      </div>

      <div className={styles.banner} data-tone={summary.tone} role="status" aria-live="polite">
        <span className={styles.bannerText}>{summary.bannerText}</span>
        {summary.bannerSub && <span className={styles.bannerSub}>{summary.bannerSub}</span>}
      </div>

      <div
        className={styles.progressTrack}
        data-indeterminate={summary.progress.indeterminate || undefined}
        aria-hidden="true"
      >
        {summary.progress.active && (
          <span
            className={styles.progressFill}
            data-tone={summary.tone}
            style={
              summary.progress.indeterminate
                ? undefined
                : { width: `${Math.max(0, Math.min(100, summary.progress.percent ?? 0))}%` }
            }
          />
        )}
      </div>

      <dl className={styles.kvList}>
        {summary.rows.map((row) => (
          <div key={row.key} className={styles.kvRow}>
            <dt>{row.label}</dt>
            <dd data-tone={row.tone || 'default'} title={row.title || row.value}>
              {row.value}
            </dd>
          </div>
        ))}
      </dl>

      {summary.onlineInstances.length >= 2 && (
        <ul className={styles.instanceList} aria-label="在线 GPU 实例">
          {summary.onlineInstances.map((item) => (
            <li key={item.id} className={styles.instanceItem}>
              <span className={styles.instanceLeft}>
                <strong>{item.gpu}</strong>
                <span className={styles.instancePool}>
                  {item.pool}
                  {item.account ? ` · ${item.account}` : ''}
                </span>
              </span>
              <span className={styles.instanceRight}>
                {item.stateText}
                {item.countdown ? ` · 剩 ${item.countdown}` : ''}
              </span>
            </li>
          ))}
        </ul>
      )}

      <div className={styles.actionRow}>
        <button
          type="button"
          className="button"
          onClick={() => handleAction(summary.primaryAction)}
        >
          {summary.primaryAction.label}
        </button>
        <button
          type="button"
          className="button secondary"
          onClick={() => handleAction(summary.secondaryAction)}
        >
          {summary.secondaryAction.label}
        </button>
      </div>

      <p className={styles.footerNote}>{summary.footerNote}</p>
    </div>
  );

  const portalRoot = typeof document !== 'undefined' ? document.body : null;

  return (
    <div className={styles.wrapper}>
      <button
        ref={triggerRef}
        type="button"
        className={styles.trigger}
        data-tone={summary.tone}
        data-phase={summary.phase}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? 'gpu-status-popover' : undefined}
        aria-label={`GPU 实时状态：${summary.pillLabel}`}
        title={summary.pillTitle}
        onClick={onToggle}
      >
        <span
          className={`${styles.dot} ${styles.triggerInlineDot}`}
          data-tone={summary.tone}
          data-pulsing={summary.pulsing || undefined}
          aria-hidden
        />
        <Cpu size={15} strokeWidth={1.65} className={styles.triggerIcon} aria-hidden />
        <span className={styles.triggerText}>{summary.pillLabel}</span>
        <ChevronDown size={13} strokeWidth={1.7} className={styles.triggerChevron} aria-hidden />
        <span
          className={`${styles.dot} ${styles.mobileCornerDot}`}
          data-tone={summary.tone}
          data-pulsing={summary.pulsing || undefined}
          aria-hidden
        />
      </button>

      {open && desktop && portalRoot && createPortal(
        <section
          ref={popoverRef}
          id="gpu-status-popover"
          className={styles.popover}
          style={{ top: anchorPos.top, right: anchorPos.right }}
          role="dialog"
          aria-modal="false"
          aria-label="算力与任务实时状态"
        >
          {panelContent}
        </section>,
        portalRoot,
      )}

      {open && !desktop && portalRoot && createPortal(
        <div
          className={styles.sheetBackdrop}
          role="presentation"
          onClick={onClose}
        >
          <section
            ref={popoverRef}
            id="gpu-status-popover"
            className={styles.sheet}
            role="dialog"
            aria-modal="true"
            aria-label="算力与任务实时状态"
            onClick={(event) => event.stopPropagation()}
          >
            {panelContent}
          </section>
        </div>,
        portalRoot,
      )}
    </div>
  );
}
