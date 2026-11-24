import { useEffect } from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import styles from '../gpu.module.css';

export interface ToastItem {
  id: string;
  tone: 'info' | 'success' | 'error';
  message: string;
}

export function ToastStack({ items, onDismiss }: { items: ToastItem[]; onDismiss: (id: string) => void }) {
  if (!items.length) return null;
  return (
    <div className={styles.toastStack} role="region" aria-label="操作提示" aria-live="polite">
      {items.map((item) => {
        const Icon = item.tone === 'success' ? CheckCircle2 : item.tone === 'error' ? AlertCircle : Info;
        return (
          <div key={item.id} className={styles.toast} data-tone={item.tone} role={item.tone === 'error' ? 'alert' : 'status'}>
            <Icon size={16} strokeWidth={1.8} aria-hidden />
            <span>{item.message}</span>
            <button type="button" className={styles.iconGhost} aria-label="关闭提示" onClick={() => onDismiss(item.id)}>
              <X size={14} aria-hidden />
            </button>
          </div>
        );
      })}
    </div>
  );
}

export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  busy?: boolean;
  error?: string;
  children?: React.ReactNode;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = '确认',
  cancelLabel = '取消',
  danger = false,
  busy = false,
  error = '',
  children,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  useBodyScrollLock(open);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) {
        e.stopPropagation();
        onCancel();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, busy, onCancel]);

  if (!open) return null;

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={(e) => { e.stopPropagation(); if (!busy) onCancel(); }}>
      <section
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.cardHead}>
          <h2>{title}</h2>
          <button type="button" className={styles.iconGhost} disabled={busy} aria-label="关闭" onClick={onCancel}>
            <X size={18} aria-hidden />
          </button>
        </div>
        {description && <p className="muted small">{description}</p>}
        {children}
        {error && <p className="status error" role="alert">{error}</p>}
        <div className="actions">
          <button type="button" className="button secondary" disabled={busy} onClick={onCancel}>{cancelLabel}</button>
          <button type="button" className={`button ${danger ? 'danger' : ''}`} disabled={busy} onClick={onConfirm}>
            {busy ? '处理中…' : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}
