import { useEffect, useState, type FormEvent } from 'react';
import { Check, FolderInput, Pencil, Search, Star, Trash2, X } from 'lucide-react';
import type { Project } from '../lib/gpu-api';
import { projectOptionLabel } from '../lib/project-label';
import { ConfirmDialog } from './Feedback';
import styles from '../gpu.module.css';

export function ProjectSheet({
  open,
  projects,
  projectId,
  onClose,
  onSelect,
  onRename,
  onSetDefault,
  onDelete,
}: {
  open: boolean;
  projects: Project[];
  projectId: string;
  onClose: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => Promise<boolean>;
  onSetDefault?: (id: string) => Promise<boolean>;
  onDelete: (project: Project) => Promise<boolean>;
}) {
  const [query, setQuery] = useState('');
  const [editingId, setEditingId] = useState('');
  const [editingTitle, setEditingTitle] = useState('');
  const [busyId, setBusyId] = useState('');
  const [confirmDelete, setConfirmDelete] = useState<Project | null>(null);

  if (!open) return null;

  const visible = projects.filter((p) => !query.trim() || `${p.title} ${p.id}`.toLowerCase().includes(query.trim().toLowerCase()));

  async function saveEdit(id: string) {
    const trimmed = editingTitle.trim();
    if (!trimmed || busyId) return;
    setBusyId(id);
    try {
      if (await onRename(id, trimmed)) {
        setEditingId('');
        setEditingTitle('');
      }
    } finally {
      setBusyId('');
    }
  }

  async function markDefault(id: string) {
    if (!onSetDefault || busyId) return;
    setBusyId(id);
    try {
      await onSetDefault(id);
    } finally {
      setBusyId('');
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="切换与管理项目" onClick={(e) => e.stopPropagation()}>
        <div className={styles.cardHead}>
          <h2>切换与管理项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
        <p className="muted small">顶栏选中的项目用于「创作」页保存新任务与上传参考图；「作品」与「任务」页可按全部或单个项目筛选。</p>
        {projects.length > 5 && (
          <label className={styles.searchBox}>
            <Search size={16} strokeWidth={1.6} aria-hidden />
            <input aria-label="搜索项目" placeholder="搜索项目名称" value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>
        )}
        {projects.length === 0 && <p className="muted">还没有项目，点右上角的 ＋ 新建一个。</p>}
        <ul className="projectList">
          {visible.map((p) => {
            const isEditing = editingId === p.id;
            const isDefault = Boolean(p.is_default);
            const deleteDisabled = projects.length <= 1 || isDefault || busyId === p.id;
            return (
              <li key={p.id} className={styles.projectManageRow}>
                {isEditing ? (
                  <form
                    className={styles.inlineForm}
                    style={{ width: '100%' }}
                    onSubmit={(e) => { e.preventDefault(); void saveEdit(p.id); }}
                  >
                    <input
                      autoFocus
                      maxLength={80}
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      aria-label="项目新名称"
                    />
                    <button className="button small" type="submit" disabled={busyId === p.id || !editingTitle.trim()}>保存</button>
                    <button className="button secondary small" type="button" onClick={() => setEditingId('')}>取消</button>
                  </form>
                ) : (
                  <>
                    <button
                      type="button"
                      className="projectRow"
                      aria-current={p.id === projectId ? 'true' : undefined}
                      onClick={() => { onSelect(p.id); onClose(); }}
                    >
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                        <span>{projectOptionLabel(p, projects)}</span>
                        {isDefault && <span className="pill">默认</span>}
                      </span>
                      {p.id === projectId && <Check size={16} strokeWidth={1.8} aria-label="当前项目" />}
                    </button>
                    {onSetDefault && (
                      <button
                        type="button"
                        className={styles.iconGhost}
                        title={isDefault ? '已是默认项目' : '设为默认项目'}
                        aria-label={isDefault ? `${p.title} 已是默认项目` : `将 ${p.title} 设为默认项目`}
                        disabled={isDefault || busyId === p.id}
                        onClick={() => void markDefault(p.id)}
                      >
                        <Star size={15} strokeWidth={1.6} fill={isDefault ? 'currentColor' : 'none'} aria-hidden />
                      </button>
                    )}
                    <button
                      type="button"
                      className={styles.iconGhost}
                      title="重命名项目"
                      aria-label={`重命名 ${p.title}`}
                      onClick={() => { setEditingId(p.id); setEditingTitle(p.title); }}
                    >
                      <Pencil size={15} strokeWidth={1.6} aria-hidden />
                    </button>
                    <button
                      type="button"
                      className={styles.iconGhost}
                      title={isDefault ? '默认项目不可直接移除，请先将其他项目设为默认' : '归档移除项目'}
                      aria-label={`移除 ${p.title}`}
                      disabled={deleteDisabled}
                      onClick={() => setConfirmDelete(p)}
                    >
                      <Trash2 size={15} strokeWidth={1.6} aria-hidden />
                    </button>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      <ConfirmDialog
        open={Boolean(confirmDelete)}
        title="归档移除项目？"
        description={confirmDelete ? `将从项目列表移除「${confirmDelete.title}」。若该项目有正在运行的任务或正被创作 API 绑定，会阻止移除；历史账单记录仍会保留。` : ''}
        confirmLabel="确认移除"
        danger
        busy={Boolean(busyId)}
        onCancel={() => setConfirmDelete(null)}
        onConfirm={async () => {
          if (!confirmDelete) return;
          setBusyId(confirmDelete.id);
          try {
            if (await onDelete(confirmDelete)) setConfirmDelete(null);
          } finally {
            setBusyId('');
          }
        }}
      />
    </div>
  );
}

export function MoveProjectDialog({
  open,
  assetIds,
  currentProjectId,
  projects,
  onClose,
  onConfirm,
}: {
  open: boolean;
  assetIds: string[];
  currentProjectId?: string;
  projects: Project[];
  onClose: () => void;
  onConfirm: (assetIds: string[], targetProjectId: string) => Promise<{ movedIds: string[]; error?: string }>;
}) {
  const [targetId, setTargetId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!open) return;
    setError('');
    const candidate = projects.find((p) => !currentProjectId || p.id !== currentProjectId) || projects[0];
    setTargetId(candidate?.id || '');
  }, [open, projects, currentProjectId]);

  if (!open || !assetIds.length) return null;

  const isSingle = assetIds.length === 1;
  const selectable = projects.filter((p) => !(isSingle && currentProjectId && p.id === currentProjectId));

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!targetId || busy) return;
    setBusy(true);
    setError('');
    try {
      const result = await onConfirm(assetIds, targetId);
      if (result.error) setError(result.error);
      else onClose();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={busy ? undefined : onClose}>
      <form className={styles.dialog} role="dialog" aria-modal="true" aria-label="移动作品到其他项目" onClick={(e) => e.stopPropagation()} onSubmit={handleSubmit}>
        <div className={styles.cardHead}>
          <h2><FolderInput size={18} strokeWidth={1.6} aria-hidden /> 移动到项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" disabled={busy} onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
        <p className="muted small">
          将选中的 <strong>{assetIds.length}</strong> 件作品移动到目标项目。来源任务记录、生成参数与原文件均会保留。
        </p>
        {selectable.length ? (
          <div className="field">
            <label htmlFor="move-target-project">目标项目</label>
            <select
              id="move-target-project"
              value={targetId}
              disabled={busy}
              onChange={(e) => setTargetId(e.target.value)}
            >
              {projects.map((p) => {
                const sameAsSingle = Boolean(isSingle && currentProjectId && p.id === currentProjectId);
                return (
                  <option key={p.id} value={p.id} disabled={sameAsSingle}>
                    {projectOptionLabel(p, projects)}{p.is_default ? '（默认）' : ''}{sameAsSingle ? '（当前所在）' : ''}
                  </option>
                );
              })}
            </select>
          </div>
        ) : (
          <p className="status" role="status">目前没有其他可移入的项目，请先点右上角 ＋ 新建一个项目。</p>
        )}
        {error && <p className="status error" role="alert">{error}</p>}
        <div className="actions">
          <button className="button secondary" type="button" disabled={busy} onClick={onClose}>取消</button>
          <button className="button" type="submit" disabled={busy || !targetId || !selectable.length}>
            {busy ? '移动中…' : `确认移动 (${assetIds.length})`}
          </button>
        </div>
      </form>
    </div>
  );
}

export function NewProjectDialog({
  open,
  onClose,
  onCreate,
}: {
  open: boolean;
  onClose: () => void;
  onCreate: (title: string) => Promise<boolean>;
}) {
  const [title, setTitle] = useState('');
  const [creating, setCreating] = useState(false);
  if (!open) return null;

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    const name = title.trim();
    if (!name || creating) return;
    setCreating(true);
    try {
      if (await onCreate(name)) {
        setTitle('');
        onClose();
      }
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <form className={styles.dialog} role="dialog" aria-modal="true" aria-label="新建项目" onClick={(e) => e.stopPropagation()} onSubmit={handleCreate}>
        <div className={styles.cardHead}>
          <h2>新建项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
        <div className="field">
          <label htmlFor="project-new">项目名称</label>
          <input id="project-new" autoFocus maxLength={80} placeholder="例如：符玄和青雀" value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
        <div className={styles.inlineForm}>
          <button className="button secondary" type="button" onClick={onClose}>取消</button>
          <button className="button" disabled={creating || !title.trim()} type="submit">{creating ? '创建中…' : '创建'}</button>
        </div>
      </form>
    </div>
  );
}
