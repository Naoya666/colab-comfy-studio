import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileCode2,
  FileSearch,
  Loader2,
  RefreshCw,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import {
  ACTIVE_MODEL_STATES,
  buildCatalogIndex,
  bytesLabel,
  inspectWorkflowDependencies,
  libraryApi,
  modelBasename,
  pickUniqueResolvedFile,
  reconcileDependencyWithCatalog,
  safeRelativePathFor,
  shortHash,
  sourceHostLabel,
  type Dependency,
  type LibraryModel,
  type ModelTask,
  type ReconciledDependency,
  type ResolvedModelFile,
  type StoredInspectorState,
  type WorkflowInspectResult,
} from '../lib/model-library';
import { extractPng, MAX_IMAGE } from '../lib/png-workflow';
import { DependencyCard } from './ModelDependencyItem';
import styles from '../gpu.module.css';

type InspectFilter = 'all' | 'missing' | 'with_link' | 'ready' | 'candidates';

const FORMAT_LABELS: Record<WorkflowInspectResult['format'], string> = {
  visual: 'ComfyUI 可视化工作流',
  api: 'ComfyUI API Prompt',
  both: 'PNG 内嵌完整工作流',
  bundle: 'Studio 工作流包',
};

interface BatchPreviewEntry {
  key: string;
  dep: Dependency;
  sourceUrl: string;
  nodeCount: number;
  hasLocalCandidate: boolean;
  optional: boolean;
  phase: 'resolving' | 'ready' | 'skipped' | 'importing' | 'done' | 'failed';
  file?: ResolvedModelFile;
  matchNote?: string;
  targetType: string;
  selected: boolean;
  reason?: string;
  error?: string;
  importedModel?: LibraryModel;
}

function BatchImportPreviewDialog({
  open,
  items,
  availableTypes,
  onClose,
  onImported,
}: {
  open: boolean;
  items: ReconciledDependency[];
  availableTypes: string[];
  onClose: () => void;
  onImported: (imported: Array<{ depId: string; model: LibraryModel }>) => Promise<void>;
}) {
  const [entries, setEntries] = useState<BatchPreviewEntry[]>([]);
  const [committing, setCommitting] = useState(false);
  const [globalError, setGlobalError] = useState('');
  const runToken = useRef(0);
  useBodyScrollLock(open);
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const itemsSignature = useMemo(
    () => items
      .filter((item) => Boolean(item.sourceUrl))
      .map((item) => `${item.sourceUrl}::${item.dep.reference.toLowerCase()}`)
      .sort()
      .join('|'),
    [items],
  );

  // Deduplicate by (sourceUrl, reference) and resolve concurrently when dialog opens.
  useEffect(() => {
    if (!open) return;
    const token = ++runToken.current;
    setGlobalError('');
    setCommitting(false);

    const grouped = new Map<string, { dep: Dependency; sourceUrl: string; nodeCount: number; hasLocalCandidate: boolean; optional: boolean }>();
    for (const item of itemsRef.current) {
      if (!item.sourceUrl) continue;
      const key = `${item.sourceUrl}::${item.dep.reference.toLowerCase()}`;
      const existing = grouped.get(key);
      const isOptional = !item.dep.required || Boolean(item.dep.disabled) || Boolean(item.dep.bypassed);
      if (existing) {
        existing.nodeCount += 1;
        existing.optional = existing.optional && isOptional;
        existing.hasLocalCandidate = existing.hasLocalCandidate || item.candidates.length > 0;
      } else {
        grouped.set(key, {
          dep: item.dep,
          sourceUrl: item.sourceUrl,
          nodeCount: 1,
          hasLocalCandidate: item.candidates.length > 0,
          optional: isOptional,
        });
      }
    }

    const initial: BatchPreviewEntry[] = [...grouped.entries()].map(([key, g]) => ({
      key,
      dep: g.dep,
      sourceUrl: g.sourceUrl,
      nodeCount: g.nodeCount,
      hasLocalCandidate: g.hasLocalCandidate,
      optional: g.optional,
      phase: 'resolving',
      targetType: g.dep.model_type !== 'unknown' ? g.dep.model_type : '',
      selected: false,
    }));
    setEntries(initial);

    let cursor = 0;
    const concurrency = Math.min(3, initial.length);
    async function worker() {
      while (cursor < initial.length) {
        const index = cursor++;
        const current = initial[index];
        if (!current || token !== runToken.current) return;
        try {
          const { files } = await libraryApi<{ files: ResolvedModelFile[] }>('/resolve', {
            url: current.sourceUrl,
            model_type: current.dep.model_type !== 'unknown' ? current.dep.model_type : undefined,
            filename: modelBasename(current.dep.reference),
            expected_sha256: current.dep.expected_sha256 || undefined,
          });
          if (token !== runToken.current) return;
          const picked = pickUniqueResolvedFile(current.dep, files || []);
          setEntries((prev) => prev.map((entry) => {
            if (entry.key !== current.key) return entry;
            if (!picked.ok) {
              return { ...entry, phase: 'skipped', reason: picked.reason, selected: false };
            }
            const resolvedType = (
              (entry.dep.model_type !== 'unknown' ? entry.dep.model_type : '')
              || (picked.file.model_type && picked.file.model_type !== 'unknown' ? picked.file.model_type : '')
            );
            const autoSelect = Boolean(resolvedType) && !entry.optional && !entry.hasLocalCandidate;
            return {
              ...entry,
              phase: 'ready',
              file: picked.file,
              matchNote: picked.matchNote,
              targetType: resolvedType,
              selected: autoSelect,
            };
          }));
        } catch (cause) {
          if (token !== runToken.current) return;
          setEntries((prev) => prev.map((entry) => (
            entry.key === current.key
              ? { ...entry, phase: 'skipped', reason: apiMessage(cause), selected: false }
              : entry
          )));
        }
      }
    }
    for (let i = 0; i < concurrency; i++) {
      void worker();
    }
  }, [open, itemsSignature]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !committing) {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, committing, onClose]);

  if (!open) return null;

  const resolvingCount = entries.filter((e) => e.phase === 'resolving').length;
  const readyEntries = entries.filter((e) => e.phase === 'ready' || e.phase === 'importing' || e.phase === 'done' || e.phase === 'failed');
  const skippedEntries = entries.filter((e) => e.phase === 'skipped');
  const selectedEntries = entries.filter((e) => e.phase === 'ready' && e.selected);
  const hasMissingType = selectedEntries.some((e) => !e.targetType);
  const allDone = entries.length > 0 && entries.every((e) => e.phase === 'done' || e.phase === 'skipped');

  async function confirmImport() {
    if (committing || !selectedEntries.length || hasMissingType) return;
    setCommitting(true);
    setGlobalError('');
    const created: Array<{ depId: string; model: LibraryModel }> = [];

    for (const entry of selectedEntries) {
      if (!entry.file) continue;
      setEntries((prev) => prev.map((item) => (
        item.key === entry.key ? { ...item, phase: 'importing', error: undefined } : item
      )));
      try {
        const relPath = safeRelativePathFor(entry.dep.reference, entry.file.filename);
        const model = await libraryApi<LibraryModel>('', {
          ...entry.file,
          model_type: entry.targetType,
          ...(relPath ? { relative_path: relPath } : {}),
        });
        await libraryApi(`/${model.id}/prepare`, {});
        created.push({ depId: entry.dep.id, model });
        setEntries((prev) => prev.map((item) => (
          item.key === entry.key ? { ...item, phase: 'done', importedModel: model } : item
        )));
      } catch (cause) {
        setEntries((prev) => prev.map((item) => (
          item.key === entry.key ? { ...item, phase: 'failed', error: apiMessage(cause) } : item
        )));
      }
    }

    if (created.length) {
      try {
        await onImported(created);
      } catch (cause) {
        setGlobalError(apiMessage(cause));
      }
    }
    setCommitting(false);
    if (created.length === selectedEntries.length) {
      onClose();
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={() => { if (!committing) onClose(); }}>
      <section
        className={`${styles.dialog} ${styles.depPreviewDialog}`}
        role="dialog"
        aria-modal="true"
        aria-label="一键入库预览确认"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.cardHead}>
          <h2>一键入库预览确认</h2>
          <button type="button" className={styles.iconGhost} disabled={committing} aria-label="关闭" onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        </div>

        <p className="muted small">
          {resolvingCount > 0
            ? `正在解析自带来源链接（已完成 ${entries.length - resolvingCount} / ${entries.length}）…`
            : `已解析 ${entries.length} 个自带链接的模型：${readyEntries.length} 个可入库，${skippedEntries.length} 个需手动处理。核对后点击确认加入后台下载队列。`}
        </p>

        {resolvingCount > 0 && (
          <div className={styles.depLoadingLine} role="status">
            <Loader2 size={14} className={styles.spin} aria-hidden />
            正在读取文件体积、版本与 SHA-256 预览信息…
          </div>
        )}

        {readyEntries.length > 0 && (
          <div className={styles.stack}>
            <div className={styles.depPreviewSectionHead}>
              <strong className="small">可直接入库（已选 {selectedEntries.length} / {readyEntries.length}）</strong>
              {!committing && resolvingCount === 0 && (
                <div className="actions">
                  <button
                    type="button"
                    className="button ghost small"
                    onClick={() => setEntries((prev) => prev.map((e) => (
                      e.phase === 'ready' ? { ...e, selected: true } : e
                    )))}
                  >
                    全选
                  </button>
                  <button
                    type="button"
                    className="button ghost small"
                    onClick={() => setEntries((prev) => prev.map((e) => (
                      e.phase === 'ready' ? { ...e, selected: false } : e
                    )))}
                  >
                    取消全选
                  </button>
                </div>
              )}
            </div>

            <div className={styles.depPreviewList}>
              {readyEntries.map((entry) => {
                const file = entry.file!;
                const sizeVal = Number(file.expected_size || file.reported_size || entry.dep.expected_size || 0);
                const shaVal = (typeof file.expected_sha256 === 'string' && file.expected_sha256) || entry.dep.expected_sha256 || '';
                const host = sourceHostLabel(entry.sourceUrl, file.source);
                return (
                  <div className={styles.depPreviewRow} key={entry.key} data-phase={entry.phase}>
                    <label className={styles.depPreviewCheck}>
                      <input
                        type="checkbox"
                        checked={entry.selected}
                        disabled={committing || entry.phase !== 'ready'}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          setEntries((prev) => prev.map((item) => (
                            item.key === entry.key ? { ...item, selected: checked } : item
                          )));
                        }}
                      />
                    </label>
                    <div className={styles.depPreviewBody}>
                      <div className={styles.depNameRow}>
                        <code className={styles.depFileName}>{file.filename}</code>
                        <span className={styles.depSourceBadge}>{host}</span>
                        {entry.optional && <span className={styles.depTag}>可选 / 已停用</span>}
                        {entry.hasLocalCandidate && <span className={styles.depWarnTag}>库内已有同名</span>}
                        {entry.matchNote && <span className={styles.depExactBadge}>{entry.matchNote}</span>}
                        {entry.nodeCount > 1 && <span className="muted small">被 {entry.nodeCount} 个节点引用</span>}
                      </div>
                      <div className={styles.depMetaRow}>
                        {entry.targetType ? (
                          <span className={styles.depTag}>{entry.targetType}</span>
                        ) : (
                          <select
                            className={styles.depCompactSelect}
                            aria-label={`选择 ${file.filename} 的入库目录`}
                            value={entry.targetType}
                            disabled={committing || entry.phase !== 'ready'}
                            onChange={(e) => {
                              const nextType = e.target.value;
                              setEntries((prev) => prev.map((item) => (
                                item.key === entry.key ? { ...item, targetType: nextType } : item
                              )));
                            }}
                          >
                            <option value="">请选择入库目录…</option>
                            {availableTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                          </select>
                        )}
                        {file.architecture && file.architecture !== 'unknown' && (
                          <span className={styles.depTag}>{String(file.architecture)}</span>
                        )}
                        <span className="muted small">{bytesLabel(sizeVal)}</span>
                        {shaVal && <span className="muted small" title={shaVal}>SHA {shortHash(shaVal)}</span>}
                        {file.version_name && <span className="muted small">{String(file.version_name)}</span>}
                      </div>
                      {entry.phase === 'importing' && (
                        <span className={styles.depLoadingLine}><Loader2 size={12} className={styles.spin} aria-hidden /> 正在提交入库…</span>
                      )}
                      {entry.phase === 'done' && (
                        <span className={styles.depDoneLine}><CheckCircle2 size={12} aria-hidden /> 已加入后台准备队列</span>
                      )}
                      {entry.error && <span className={styles.errorText}>{entry.error}</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {skippedEntries.length > 0 && (
          <div className={styles.stack}>
            <strong className="small muted">需手动处理 / 已跳过（{skippedEntries.length}）</strong>
            <div className={styles.depPreviewList}>
              {skippedEntries.map((entry) => (
                <div className={styles.depPreviewRow} key={entry.key} data-phase="skipped">
                  <AlertTriangle size={15} className={styles.depWarnIcon} aria-hidden />
                  <div className={styles.depPreviewBody}>
                    <div className={styles.depNameRow}>
                      <code className={styles.depFileName}>{entry.dep.reference}</code>
                    </div>
                    <p className="muted small">{entry.reason}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {hasMissingType && (
          <p className="status error" role="alert">有已勾选模型尚未选择入库目录，请先指定目标目录后再确认。</p>
        )}
        {globalError && <p className="status error" role="alert">{globalError}</p>}

        <div className={styles.depPreviewFooter}>
          <button type="button" className="button secondary" disabled={committing} onClick={onClose}>
            {allDone ? '关闭' : '取消'}
          </button>
          <button
            type="button"
            className="button"
            disabled={committing || resolvingCount > 0 || !selectedEntries.length || hasMissingType}
            onClick={() => void confirmImport()}
          >
            {committing
              ? '入库提交中…'
              : `确认入库并准备（${selectedEntries.length} 项）`}
          </button>
        </div>
      </section>
    </div>
  );
}

export function WorkflowDependencyInspector({
  open,
  fileName,
  result,
  importedMap,
  models,
  tasks,
  availableTypes,
  onStateChange,
  onClose,
  onRefreshLibrary,
  onLocateModel,
}: {
  open: boolean;
  fileName: string;
  result: WorkflowInspectResult | null;
  importedMap: Record<string, string>;
  models: LibraryModel[];
  tasks: ModelTask[];
  availableTypes: string[];
  onStateChange: (patch: Partial<StoredInspectorState>) => void;
  onClose: () => void;
  onRefreshLibrary: () => Promise<void>;
  onLocateModel: (model: LibraryModel) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pasteText, setPasteText] = useState('');
  const [showPaste, setShowPaste] = useState(false);
  const [showDropzone, setShowDropzone] = useState(!result);
  const [dragActive, setDragActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<InspectFilter>('all');
  const [previewOpen, setPreviewOpen] = useState(false);

  useEffect(() => {
    if (!result) setShowDropzone(true);
  }, [result]);

  const catalogIndex = useMemo(() => buildCatalogIndex(models, tasks), [models, tasks]);

  const reconciled = useMemo(() => {
    if (!result) return [];
    return result.dependencies.map((dep) => reconcileDependencyWithCatalog(dep, models, tasks, importedMap[dep.id], catalogIndex));
  }, [result, models, tasks, importedMap, catalogIndex]);

  const stats = useMemo(() => {
    let ready = 0;
    let inProgress = 0;
    let candidates = 0;
    let missing = 0;
    let withLink = 0;
    for (const item of reconciled) {
      if (item.effectiveStatus === 'matched' || item.effectiveStatus === 'ready') ready += 1;
      else if (ACTIVE_MODEL_STATES.has(item.effectiveStatus)) inProgress += 1;
      else if (item.effectiveStatus === 'needs_confirmation') candidates += 1;
      else missing += 1;
      if (item.sourceUrl) withLink += 1;
    }
    return { total: reconciled.length, ready, inProgress, candidates, missing, withLink };
  }, [reconciled]);

  const batchImportTargets = useMemo(
    () => reconciled.filter(
      (item) => Boolean(item.sourceUrl)
        && !item.bound
        && item.effectiveStatus !== 'manual_review',
    ),
    [reconciled],
  );

  const visibleItems = useMemo(() => {
    if (filter === 'all') return reconciled;
    if (filter === 'ready') return reconciled.filter((i) => i.effectiveStatus === 'matched' || i.effectiveStatus === 'ready' || ACTIVE_MODEL_STATES.has(i.effectiveStatus));
    if (filter === 'candidates') return reconciled.filter((i) => i.effectiveStatus === 'needs_confirmation');
    if (filter === 'with_link') return reconciled.filter((i) => Boolean(i.sourceUrl));
    return reconciled.filter((i) => i.effectiveStatus !== 'matched' && i.effectiveStatus !== 'ready' && !ACTIVE_MODEL_STATES.has(i.effectiveStatus));
  }, [reconciled, filter]);

  if (!open) return null;

  async function runInspect(payload: Record<string, unknown>, label: string, extraWarnings: string[] = []) {
    setBusy(true);
    setError('');
    try {
      const inspected = await inspectWorkflowDependencies(payload);
      const mergedWarnings = [...extraWarnings, ...(inspected.warnings || [])];
      onStateChange({
        open: true,
        fileName: label,
        result: { ...inspected, warnings: mergedWarnings },
        importedMap: {},
      });
      setShowDropzone(false);
      setShowPaste(false);
      setPasteText('');
      setFilter('all');
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function handleFile(file?: File) {
    if (!file) return;
    setError('');
    const lower = file.name.toLowerCase();
    try {
      if (lower.endsWith('.png') || file.type === 'image/png') {
        if (file.size > MAX_IMAGE) throw new Error('图片不能超过 32 MiB');
        const extracted = await extractPng(new Uint8Array(await file.arrayBuffer()));
        await runInspect(
          { workflow: extracted.workflow, prompt: extracted.prompt },
          file.name,
          extracted.warnings.filter((w) => !w.includes('无法可靠自动转换为 API prompt')),
        );
        return;
      }
      const text = await file.text();
      const parsed: unknown = JSON.parse(text);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('JSON 根节点必须是对象');
      }
      await runInspect(parsed as Record<string, unknown>, file.name);
    } catch (cause) {
      setError(cause instanceof SyntaxError ? 'JSON 格式错误，请检查文件内容' : apiMessage(cause));
    }
  }

  async function handlePasteSubmit() {
    if (!pasteText.trim()) return;
    setError('');
    try {
      const parsed: unknown = JSON.parse(pasteText);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('JSON 根节点必须是对象');
      }
      await runInspect(parsed as Record<string, unknown>, '粘贴的工作流 JSON');
    } catch (cause) {
      setError(cause instanceof SyntaxError ? 'JSON 格式错误，请检查粘贴内容' : apiMessage(cause));
    }
  }

  function clearResult() {
    setError('');
    setPasteText('');
    setShowDropzone(true);
    setFilter('all');
    onStateChange({ fileName: '', result: null, importedMap: {} });
  }

  return (
    <section className={`${styles.card} ${styles.depWorkspace}`} aria-label="工作流模型依赖检测">
      <div className={styles.depWorkspaceHead}>
        <div className={styles.depWorkspaceTitle}>
          <h3>
            <FileSearch size={17} strokeWidth={1.7} aria-hidden />
            工作流模型依赖检测
          </h3>
          <span className="muted small">仅分析模型与来源，不创建创作模式草稿；切页自动保留当前结果。</span>
        </div>
        <div className={styles.titleActions}>
          {result && (
            <>
              <button
                type="button"
                className="button ghost small"
                onClick={() => setShowDropzone((v) => !v)}
              >
                <RefreshCw size={13} aria-hidden />
                {showDropzone ? '收起导入' : '更换工作流'}
              </button>
              <button
                type="button"
                className="button ghost small"
                onClick={clearResult}
              >
                <Trash2 size={13} aria-hidden />
                清空
              </button>
            </>
          )}
          <button type="button" className={styles.iconGhost} aria-label="收起检测面板" onClick={onClose}>
            <X size={17} aria-hidden />
          </button>
        </div>
      </div>

      {/* Import dropzone + paste JSON */}
      {showDropzone && (
        <div className={styles.stack}>
          <div
            className={`${styles.depDropzone} ${dragActive ? styles.dropActive : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              void handleFile(e.dataTransfer.files?.[0]);
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
                void handleFile(file);
              }}
            />
            <Upload size={20} strokeWidth={1.6} aria-hidden />
            <strong>拖入或选择 ComfyUI 工作流文件（.json / .png 原图）</strong>
            <span className="muted small">支持可视化工作流 JSON（可自动提取节点内嵌下载链接）、API Prompt JSON 及 PNG 原图（本地解包不上传原图）</span>
            <div className="actions">
              <button
                type="button"
                className="button small"
                disabled={busy}
                onClick={() => fileInputRef.current?.click()}
              >
                {busy ? <><Loader2 size={13} className={styles.spin} aria-hidden /> 解析检测中…</> : '选择 JSON / PNG 文件'}
              </button>
              <button
                type="button"
                className="button secondary small"
                disabled={busy}
                onClick={() => setShowPaste((v) => !v)}
              >
                <FileCode2 size={13} aria-hidden />
                {showPaste ? '收起粘贴框' : '粘贴 JSON 文本'}
              </button>
            </div>
          </div>

          {showPaste && (
            <div className={styles.depSubPanel}>
              <textarea
                className={styles.code}
                aria-label="粘贴 ComfyUI 工作流 JSON"
                placeholder="粘贴 ComfyUI 工作流 JSON 或 API Prompt JSON"
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
              />
              <div className="actions">
                <button
                  type="button"
                  className="button small"
                  disabled={busy || !pasteText.trim()}
                  onClick={() => void handlePasteSubmit()}
                >
                  {busy ? '检测中…' : '检测模型依赖'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {error && <p role="alert" className="status error">{error}</p>}

      {/* Inspection results */}
      {result && (
        <div className={styles.stack}>
          <div className={styles.depSummaryBar}>
            <div className={styles.depSummaryMeta}>
              <strong>{fileName || '已导入工作流'}</strong>
              <span className={styles.depTag}>{FORMAT_LABELS[result.format] || result.format}</span>
              <span className="muted small">{result.node_count} 个节点 · 检出 {stats.total} 项模型依赖</span>
            </div>

            {batchImportTargets.length > 0 && (
              <button
                type="button"
                className="button small"
                onClick={() => setPreviewOpen(true)}
              >
                <Download size={13} aria-hidden />
                一键入库自带链接模型（{batchImportTargets.length}）
              </button>
            )}
          </div>

          {result.warnings.length > 0 && (
            <div className={styles.depNoticeStrip}>
              {result.warnings.map((w) => <p key={w} className="muted small">{w}</p>)}
            </div>
          )}

          {/* Filter pills */}
          {stats.total > 0 && (
            <div className={styles.depFilterBar}>
              {([
                ['all', `全部 ${stats.total}`],
                ['missing', `待处理 ${stats.missing + stats.candidates}`],
                ['with_link', `自带链接 ${stats.withLink}`],
                ['candidates', `库内有同名 ${stats.candidates}`],
                ['ready', `已就绪 / 准备中 ${stats.ready + stats.inProgress}`],
              ] as const).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  className={styles.depFilterChip}
                  data-active={filter === key}
                  onClick={() => setFilter(key)}
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          {!stats.total ? (
            <p className="muted small">未在该工作流中发现可识别的模型依赖。</p>
          ) : !visibleItems.length ? (
            <p className="muted small">当前筛选条件下暂无模型依赖。</p>
          ) : (
            <div className={styles.depList}>
              {visibleItems.map((item) => (
                <DependencyCard
                  key={item.dep.id}
                  mode="library"
                  dep={item.dep}
                  models={models}
                  tasks={tasks}
                  availableTypes={availableTypes}
                  boundModel={item.bound}
                  candidateModels={item.candidates}
                  effectiveStatus={item.effectiveStatus}
                  refresh={onRefreshLibrary}
                  onLibraryModelImported={(depId, model) => {
                    onStateChange({ importedMap: { ...importedMap, [depId]: model.id } });
                  }}
                  onLocateLibraryModel={onLocateModel}
                />
              ))}
            </div>
          )}
        </div>
      )}

      <BatchImportPreviewDialog
        open={previewOpen}
        items={batchImportTargets}
        availableTypes={availableTypes}
        onClose={() => setPreviewOpen(false)}
        onImported={async (created) => {
          const nextMap = { ...importedMap };
          for (const item of created) {
            nextMap[item.depId] = item.model.id;
          }
          onStateChange({ importedMap: nextMap });
          await onRefreshLibrary();
        }}
      />
    </section>
  );
}
