import { useEffect, useRef, useState } from 'react';
import {
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  HardDrive,
  RefreshCw,
  ShieldCheck,
  Trash2,
  X,
} from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import {
  ACTIVE_MODEL_STATES,
  bytesLabel,
  displayModelTitles,
  libraryApi,
  modelStatus,
  shortHash,
  type LibraryModel,
  type ModelTask,
} from '../lib/model-library';
import { ConfirmDialog } from './Feedback';
import styles from '../gpu.module.css';

type LibraryModelDetail = LibraryModel & { references?: { workflow_id: string }[] };

export function ModelDetailDrawer({
  model,
  task,
  categorySlot,
  hasPrev,
  hasNext,
  onPrev,
  onNext,
  onClose,
  onRefresh,
  onProgress,
}: {
  model: LibraryModel | null;
  task?: ModelTask;
  categorySlot: number;
  hasPrev: boolean;
  hasNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
  onRefresh: () => Promise<void>;
  onProgress: (message: string) => void;
}) {
  const [detail, setDetail] = useState<LibraryModelDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [displayName, setDisplayName] = useState('');
  const [architecture, setArchitecture] = useState('');
  const [tagsText, setTagsText] = useState('');
  const [triggerWordsText, setTriggerWordsText] = useState('');
  const [externalSource, setExternalSource] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [confirmAction, setConfirmAction] = useState<'remove_index' | 'delete_file' | null>(null);
  const [registerOpen, setRegisterOpen] = useState(false);
  const [regKey, setRegKey] = useState('');
  const [regLicense, setRegLicense] = useState('');
  const [regVram, setRegVram] = useState('');
  const [regError, setRegError] = useState('');

  const modelId = model?.id || null;

  useBodyScrollLock(Boolean(modelId));

  useEffect(() => {
    if (!modelId) {
      setDetail(null);
      return;
    }
    let active = true;
    setLoading(true);
    setError('');
    setConfirmAction(null);
    setRegisterOpen(false);
    void libraryApi<LibraryModelDetail>('/' + modelId).then((loaded) => {
      if (!active) return;
      setDetail(loaded);
      setDisplayName(loaded.display_name || '');
      setArchitecture(loaded.architecture || 'unknown');
      setTagsText((loaded.tags || []).join(', '));
      setTriggerWordsText((loaded.trigger_words || []).join(', '));
      setExternalSource(loaded.external_source?.url || '');
    }).catch((cause) => {
      if (active) setError(apiMessage(cause));
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [modelId]);

  const keyStateRef = useRef({ busy, confirmAction, registerOpen, hasPrev, hasNext, onPrev, onNext, onClose });
  keyStateRef.current = { busy, confirmAction, registerOpen, hasPrev, hasNext, onPrev, onNext, onClose };

  useEffect(() => {
    if (!modelId) return;
    const onKey = (e: KeyboardEvent) => {
      const s = keyStateRef.current;
      if (s.confirmAction || s.registerOpen) return;
      if (e.key === 'Escape' && !s.busy) {
        e.stopPropagation();
        s.onClose();
        return;
      }
      const target = e.target as HTMLElement | null;
      const inFormField = Boolean(target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
      if (inFormField) return;
      if (e.key === 'ArrowUp' && s.hasPrev) {
        e.preventDefault();
        s.onPrev();
      } else if (e.key === 'ArrowDown' && s.hasNext) {
        e.preventDefault();
        s.onNext();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [modelId]);

  if (!model) return null;

  const activeModel = detail && detail.id === model.id ? detail : model;
  const titles = displayModelTitles(activeModel);
  const isActiveTask = Boolean(task && ACTIVE_MODEL_STATES.has(task.state));
  const statusTone = isActiveTask
    ? 'active'
    : activeModel.state === 'ready'
      ? 'ready'
      : activeModel.state === 'failed' || activeModel.state === 'missing'
        ? 'danger'
        : 'warn';
  const canDeletePhysical = Boolean(activeModel.drive_file_id && activeModel.managed_file && !isActiveTask);
  const refCount = detail?.id === model.id ? (detail.references?.length || 0) : (model.reference_count || 0);

  async function copyText(key: string, value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedField(key);
      window.setTimeout(() => setCopiedField((curr) => (curr === key ? null : curr)), 1600);
    } catch {
      // Clipboard API unavailable in restricted contexts.
    }
  }

  async function runAction(fn: () => Promise<unknown>, successMessage?: string) {
    setBusy(true);
    setError('');
    try {
      await fn();
      await onRefresh();
      if (modelId) {
        const refreshed = await libraryApi<LibraryModelDetail>('/' + modelId).catch(() => null);
        if (refreshed) {
          setDetail(refreshed);
          setDisplayName(refreshed.display_name || '');
          setArchitecture(refreshed.architecture || 'unknown');
          setTagsText((refreshed.tags || []).join(', '));
          setTriggerWordsText((refreshed.trigger_words || []).join(', '));
          setExternalSource(refreshed.external_source?.url || '');
        }
      }
      if (successMessage) onProgress(successMessage);
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function saveMetadata() {
    if (!detail) return;
    const tags = tagsText.split(',').map((s) => s.trim()).filter(Boolean);
    const trigger_words = triggerWordsText
      .split(/[,\n，]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    await runAction(
      () => libraryApi('/' + detail.id, {
        display_name: displayName.trim() || detail.filename,
        architecture: architecture.trim() || 'unknown',
        tags,
        trigger_words,
      }, 'PATCH'),
      '模型信息已保存',
    );
  }

  async function saveExternalSource() {
    if (!detail) return;
    await runAction(
      () => libraryApi('/' + detail.id, {
        external_source_url: externalSource.trim() || null,
      }, 'PATCH'),
      externalSource.trim() ? '外部来源已保存' : '已解除外部来源绑定',
    );
  }

  async function submitRegister() {
    if (!detail) return;
    const key = regKey.trim();
    const license = regLicense.trim();
    const vram = Number(regVram);
    if (!key) { setRegError('请输入唯一模型标识（字母、数字、短横线）。'); return; }
    if (!license) { setRegError('请输入模型的实际许可证名称。'); return; }
    if (!Number.isFinite(vram) || vram <= 0) { setRegError('请输入有效的显存 GB 正数。'); return; }
    setRegError('');
    await runAction(async () => {
      await libraryApi(`/${detail.id}/workflow-model`, { model_key: key, license, resource_profile: { vram_gb: vram } });
      setRegisterOpen(false);
      setRegKey('');
      setRegLicense('');
      setRegVram('');
    }, '已登记为未启用模型，请在工作流配置中关联该标识并完成审核与真实 GPU 验证');
  }

  async function executeDelete(deleteFile: boolean) {
    if (!modelId) return;
    setConfirmAction(null);
    setBusy(true);
    setError('');
    try {
      const qs = deleteFile ? '?confirm_references=true&delete_file=true' : '?confirm_references=true';
      await libraryApi(`/${modelId}${qs}`, undefined, 'DELETE');
      await onRefresh();
      onProgress(deleteFile ? `已删除「${titles.primary}」及 Drive 物理文件` : `已移除「${titles.primary}」索引`);
      onClose();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
    <div className={styles.drawerBackdrop} role="presentation" onClick={() => { if (!busy) onClose(); }}>
      <aside
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-label={`模型详情：${titles.primary}`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className={styles.drawerHeader}>
          <div className={styles.drawerTitleGroup}>
            <div className={styles.drawerTitleRow}>
              <i className={styles.legendDot} data-slot={categorySlot} aria-hidden />
              <h3>{titles.primary}</h3>
            </div>
            <div className={styles.depMetaRow}>
              <span className={styles.depStatusBadge} data-tone={statusTone}>
                {modelStatus[isActiveTask && task ? task.state : activeModel.state] || activeModel.state}
              </span>
              <span className={styles.depTag}>{activeModel.model_type}</span>
              <span className={styles.depTag}>{bytesLabel(activeModel.size)}</span>
              {activeModel.architecture && activeModel.architecture !== 'unknown' && (
                <span className={styles.depTag}>{activeModel.architecture}</span>
              )}
              <span className={styles.depTag}>
                {activeModel.managed_file ? '应用托管' : '外部 / 缓存'}
              </span>
            </div>
          </div>

          <div className={styles.drawerHeaderActions}>
            <button
              type="button"
              className={styles.iconGhost}
              disabled={!hasPrev || busy}
              title="上一个模型（↑）"
              aria-label="上一个模型"
              onClick={onPrev}
            >
              <ChevronUp size={16} aria-hidden />
            </button>
            <button
              type="button"
              className={styles.iconGhost}
              disabled={!hasNext || busy}
              title="下一个模型（↓）"
              aria-label="下一个模型"
              onClick={onNext}
            >
              <ChevronDown size={16} aria-hidden />
            </button>
            <button
              type="button"
              className={styles.iconGhost}
              disabled={busy}
              aria-label="关闭详情"
              onClick={onClose}
            >
              <X size={18} aria-hidden />
            </button>
          </div>
        </header>

        <div className={styles.drawerBody}>
          {/* Quick Maintenance Actions */}
          <div className={styles.drawerQuickBar}>
            {isActiveTask && task ? (
              <button
                type="button"
                className="button secondary small"
                disabled={busy}
                onClick={() => void runAction(() => libraryApi(`/tasks/${task.id}/cancel`, {}), '已请求取消模型准备')}
              >
                取消准备
              </button>
            ) : (
              <button
                type="button"
                className="button secondary small"
                disabled={busy}
                onClick={() => void runAction(() => libraryApi(`/${activeModel.id}/prepare`, { action: 'prepare' }), '已加入准备队列')}
              >
                <RefreshCw size={13} aria-hidden />
                准备 / 重试
              </button>
            )}
            <button
              type="button"
              className="button secondary small"
              disabled={busy || !activeModel.drive_file_id || isActiveTask}
              onClick={() => void runAction(() => libraryApi(`/${activeModel.id}/prepare`, { action: 'verify' }), '已加入重新校验队列')}
            >
              <ShieldCheck size={13} aria-hidden />
              重新校验
            </button>
            <button
              type="button"
              className="button secondary small"
              disabled={busy || !activeModel.drive_file_id || isActiveTask}
              onClick={() => void runAction(() => libraryApi(`/${activeModel.id}/check`, {}), 'Drive 文件状态已检查')}
            >
              <HardDrive size={13} aria-hidden />
              检查 Drive
            </button>
          </div>

          {isActiveTask && task && (
            <div className={styles.depProgressBox}>
              <div className={styles.progressText}>
                <span>{modelStatus[task.state] || task.state}</span>
                <span>
                  {task.bytes ? bytesLabel(task.bytes) : '0 B'} / {bytesLabel(task.total)}
                  {task.speed ? ` · ${(task.speed / 1024 ** 2).toFixed(1)} MB/s` : ''}
                </span>
              </div>
              <div className={styles.progress} data-indeterminate={!task.total}>
                <span style={{ width: task.total ? `${Math.min(100, (task.bytes / task.total) * 100)}%` : undefined }} />
              </div>
              {!!task.attempt && task.attempt > 1 && (
                <small className="muted">
                  第 {task.attempt} 次尝试
                  {task.resume_phase && task.resume_offset !== undefined
                    && ` · ${modelStatus[task.resume_phase] || task.resume_phase}从 ${task.resume_offset.toLocaleString()} 字节继续`}
                </small>
              )}
              {task.restart_reason && <small className="muted">{task.restart_reason}</small>}
            </div>
          )}

          {activeModel.error && <p className="status error" role="alert">{activeModel.error}</p>}
          {error && <p className="status error" role="alert">{error}</p>}
          {loading && !detail && <p className="muted small" role="status">正在加载完整详情…</p>}

          {/* Summary & Integrity */}
          <section className={styles.drawerSection}>
            <h4>文件与校验</h4>
            <dl className={styles.kv}>
              <dt>文件路径</dt>
              <dd className={styles.drawerCopyVal}>
                <span className={styles.monoInline}>{activeModel.relative_path || activeModel.filename}</span>
                <button
                  type="button"
                  className={styles.depCopyBtn}
                  title="复制路径"
                  aria-label="复制文件路径"
                  onClick={() => void copyText('path', activeModel.relative_path || activeModel.filename)}
                >
                  {copiedField === 'path' ? <Check size={13} aria-hidden /> : <Copy size={13} aria-hidden />}
                </button>
              </dd>
              <dt>SHA-256</dt>
              <dd className={styles.drawerCopyVal}>
                <span className={styles.monoInline} title={activeModel.sha256 || undefined}>
                  {activeModel.sha256 ? shortHash(activeModel.sha256) : '未计算'}
                </span>
                {activeModel.sha256 && (
                  <button
                    type="button"
                    className={styles.depCopyBtn}
                    title="复制完整 SHA-256"
                    aria-label="复制 SHA-256"
                    onClick={() => void copyText('sha', activeModel.sha256!)}
                  >
                    {copiedField === 'sha' ? <Check size={13} aria-hidden /> : <Copy size={13} aria-hidden />}
                  </button>
                )}
              </dd>
              {activeModel.expected_sha256 && (
                <>
                  <dt>预期 SHA-256</dt>
                  <dd className={styles.monoInline} title={activeModel.expected_sha256}>
                    {shortHash(activeModel.expected_sha256)}
                  </dd>
                </>
              )}
              <dt>来源与版本</dt>
              <dd>{activeModel.source || '未知'}{activeModel.version_name ? ` · ${activeModel.version_name}` : ''}</dd>
              <dt>外部来源</dt>
              <dd>
                {activeModel.external_source
                  ? (activeModel.external_source.status === 'verified' ? '已验证' : '待验证')
                  : '未绑定'}
              </dd>
            </dl>
            {activeModel.trigger_words && activeModel.trigger_words.length > 0 && (
              <div className={styles.stack} style={{ gap: '6px' }}>
                <small className="muted">触发词</small>
                <div className={styles.depMetaRow}>
                  {activeModel.trigger_words.map((tw) => (
                    <span key={tw} className={styles.depTag}>{tw}</span>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* Workflow References */}
          <section className={styles.drawerSection}>
            <div className={styles.storageRowHead}>
              <h4>工作流引用</h4>
              <span className="muted small">{refCount} 个</span>
            </div>
            {detail && detail.id === model.id ? (
              detail.references && detail.references.length > 0 ? (
                <div className={styles.depMetaRow}>
                  {detail.references.map((r) => (
                    <span key={r.workflow_id} className={styles.depTag}>{r.workflow_id}</span>
                  ))}
                </div>
              ) : (
                <p className="muted small">暂无工作流引用此模型。</p>
              )
            ) : (
              <p className="muted small">正在加载引用列表…</p>
            )}
          </section>

          {/* External Source Binding */}
          {detail && detail.id === model.id && (
            <>
              <section className={styles.drawerSection}>
                <h4>外部来源链接</h4>
                <label className="field">
                  <input
                    type="url"
                    value={externalSource}
                    disabled={busy}
                    placeholder="https://huggingface.co/... 或 Civitai 链接"
                    onChange={(e) => setExternalSource(e.target.value)}
                  />
                </label>
                <p className="muted small">可填写无凭据的 HTTPS 来源链接，凭据在「设置 › 模型下载源凭据」中配置。留空并保存可解除绑定。</p>
                <div className="actions">
                  <button
                    type="button"
                    className="button secondary small"
                    disabled={busy}
                    onClick={() => void saveExternalSource()}
                  >
                    保存外部来源
                  </button>
                </div>
              </section>

              {/* Editable Metadata */}
              <section className={styles.drawerSection}>
                <h4>编辑信息</h4>
                <label className="field">
                  显示名称
                  <input value={displayName} disabled={busy} onChange={(e) => setDisplayName(e.target.value)} />
                </label>
                <div className={styles.grid2}>
                  <label className="field">
                    架构
                    <input value={architecture} disabled={busy} onChange={(e) => setArchitecture(e.target.value)} />
                  </label>
                  <label className="field">
                    标签（逗号分隔）
                    <input value={tagsText} disabled={busy} onChange={(e) => setTagsText(e.target.value)} placeholder="anime, style" />
                  </label>
                </div>
                <label className="field">
                  触发词（逗号或换行分隔）
                  <textarea
                    rows={2}
                    value={triggerWordsText}
                    disabled={busy}
                    onChange={(e) => setTriggerWordsText(e.target.value)}
                    placeholder="例如：camera motion, Anime-Motion"
                  />
                </label>
                <div className="actions">
                  <button
                    type="button"
                    className="button small"
                    disabled={busy}
                    onClick={() => void saveMetadata()}
                  >
                    保存信息
                  </button>
                  <button
                    type="button"
                    className="button secondary small"
                    disabled={busy || detail.state !== 'ready'}
                    onClick={() => { setRegError(''); setRegisterOpen(true); }}
                  >
                    登记为工作流模型
                  </button>
                </div>
              </section>

              {/* Danger Zone */}
              <section className={`${styles.drawerSection} ${styles.drawerDangerSection}`}>
                <h4>清理与删除</h4>
                <p className="muted small">
                  {activeModel.managed_file
                    ? '移除索引会保留 Drive 物理文件；删除 Drive 文件会把文件移至回收站并同步移除索引。'
                    : '该模型为外部导入或历史缓存文件，仅支持移除索引；物理文件请在 Google Drive 中手动管理。'}
                </p>
                <div className="actions">
                  <button
                    type="button"
                    className="button danger small"
                    disabled={busy || isActiveTask}
                    onClick={() => setConfirmAction('remove_index')}
                  >
                    移除索引
                  </button>
                  <button
                    type="button"
                    className="button danger small"
                    disabled={busy || !canDeletePhysical}
                    title={!activeModel.managed_file ? '外部导入/历史缓存文件仅允许移除索引' : undefined}
                    onClick={() => setConfirmAction('delete_file')}
                  >
                    <Trash2 size={13} aria-hidden />
                    删除 Drive 文件
                  </button>
                </div>
              </section>
            </>
          )}
        </div>
      </aside>
    </div>

        <ConfirmDialog
          open={confirmAction === 'remove_index'}
          title="移除模型索引？"
          description={`当前有 ${refCount} 个工作流引用此模型，移除后相关依赖将失效（不会删除 Drive 物理文件）。`}
          confirmLabel="确认移除"
          danger
          busy={busy}
          onCancel={() => setConfirmAction(null)}
          onConfirm={() => void executeDelete(false)}
        />

        <ConfirmDialog
          open={confirmAction === 'delete_file'}
          title="删除 Drive 物理文件并移除索引？"
          description={`将把 Drive 上的「${titles.primary}」（${bytesLabel(activeModel.size)}）移至回收站并移除索引，当前引用工作流 ${refCount} 个。`}
          confirmLabel="确认删除"
          danger
          busy={busy}
          onCancel={() => setConfirmAction(null)}
          onConfirm={() => void executeDelete(true)}
        />

        <ConfirmDialog
          open={registerOpen}
          title="登记为工作流模型"
          description="登记后仍需在工作流配置中关联该标识，并完成人工接受许可与真实 GPU 验证。"
          confirmLabel="提交登记"
          busy={busy}
          error={regError}
          onCancel={() => setRegisterOpen(false)}
          onConfirm={() => void submitRegister()}
        >
          <div className="field">
            <label htmlFor="reg-model-key">唯一模型标识（字母、数字、短横线）</label>
            <input id="reg-model-key" value={regKey} placeholder="例如：flux-dev-fp8" onChange={(e) => setRegKey(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="reg-model-license">模型许可证名称</label>
            <input id="reg-model-license" value={regLicense} placeholder="例如：Apache-2.0 / Non-Commercial" onChange={(e) => setRegLicense(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="reg-model-vram">所需显存（GB）</label>
            <input id="reg-model-vram" type="number" min={1} step={0.5} value={regVram} placeholder="例如：16" onChange={(e) => setRegVram(e.target.value)} />
          </div>
        </ConfirmDialog>
    </>
  );
}
