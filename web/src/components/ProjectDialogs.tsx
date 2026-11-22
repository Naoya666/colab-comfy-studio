import { useState, type FormEvent } from 'react';
import { Check, Pencil, Search, Trash2, X } from 'lucide-react';
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
  onDelete,
}: {
  open: boolean;
  projects: Project[];
  projectId: string;
  onClose: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => Promise<boolean>;
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

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <section className={styles.dialog} role="dialog" aria-modal="true" aria-label="切换与管理项目" onClick={(e) => e.stopPropagation()}>
        <div className={styles.cardHead}>
          <h2>切换与管理项目</h2>
          <button className={styles.iconGhost} type="button" aria-label="关闭" onClick={onClose}><X size={18} strokeWidth={1.6} aria-hidden /></button>
        </div>
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
                      <span>{projectOptionLabel(p, projects)}</span>
                      {p.id === projectId && <Check size={16} strokeWidth={1.8} aria-label="当前项目" />}
                    </button>
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
                      title="归档移除项目"
                      aria-label={`移除 ${p.title}`}
                      disabled={projects.length <= 1 || busyId === p.id}
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
        description={confirmDelete ? `将从项目列表移除「${confirmDelete.title}」。若该项目有正在运行的任务会阻止移除；历史账单记录仍会保留。` : ''}
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
