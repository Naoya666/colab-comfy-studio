import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Copy, Download, Globe, Image as ImageIcon, RefreshCw, Upload, X } from 'lucide-react';
import { ApiError, apiMessage, fetchWorkflowFromUrl, type WorkflowVersionConflictDetail } from '../lib/gpu-api';
import { extractPng, MAX_IMAGE, parseWorkflowJson, type ExtractedWorkflow } from '../lib/png-workflow';
import {
  applyBundleIdentityOverride,
  classifyWorkflowJson,
  isValidSemver,
  isValidWorkflowKey,
  parseWorkflowVersionConflict,
  validateWorkflowUrlInput,
} from '../lib/workflow-import';
import { WorkflowSourceDropzone } from './WorkflowSourceDropzone';
import styles from '../gpu.module.css';

export type WorkflowImportResult =
  | { ok: true }
  | { ok: false; conflict: WorkflowVersionConflictDetail }
  | { ok: false; error?: string };

interface PendingPngPreview {
  sourceKind: 'local' | 'remote';
  label: string;
  sourceHost?: string;
  note?: string;
  file?: File;
  result: ExtractedWorkflow;
}

interface PendingVersionConflict {
  bundle: Record<string, unknown>;
  detail: WorkflowVersionConflictDetail;
  workflowKey: string;
  version: string;
}

export function WorkflowImportWorkspace({
  open,
  onClose,
  onImport,
  onOpenModelLibrary,
}: {
  open: boolean;
  onClose: () => void;
  onImport: (bundle: string) => Promise<boolean | WorkflowImportResult> | void;
  onOpenModelLibrary?: () => void;
}) {
  const revision = useRef(0);
  const [pending, setPending] = useState<PendingPngPreview | null>(null);
  const [conflict, setConflict] = useState<PendingVersionConflict | null>(null);
  const [thumbUrl, setThumbUrl] = useState('');
  const [selected, setSelected] = useState<'workflow' | 'prompt'>('workflow');
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [plainComfyNotice, setPlainComfyNotice] = useState('');
  const [copyFeedback, setCopyFeedback] = useState('');

  useEffect(() => {
    const file = pending?.sourceKind === 'local' ? pending.file : undefined;
    if (!file) {
      setThumbUrl('');
      return;
    }
    const next = URL.createObjectURL(file);
    setThumbUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [pending]);

  if (!open) return null;

  function resetPreview() {
    revision.current += 1;
    setPending(null);
    setConflict(null);
    setError('');
    setPlainComfyNotice('');
    setCopyFeedback('');
  }

  async function importBundlePayload(bundle: Record<string, unknown>): Promise<boolean> {
    setImporting(true);
    setError('');
    try {
      const outcome = await onImport(JSON.stringify(bundle));
      if (outcome && typeof outcome === 'object' && 'conflict' in outcome && outcome.conflict) {
        setConflict({
          bundle,
          detail: outcome.conflict,
          workflowKey: outcome.conflict.workflow_key,
          version: outcome.conflict.suggested_version,
        });
        setError('');
        return false;
      }
      if (outcome === false || (outcome && typeof outcome === 'object' && outcome.ok === false)) {
        if (outcome && typeof outcome === 'object' && 'error' in outcome && outcome.error) {
          setError(outcome.error);
        }
        return false;
      }
      resetPreview();
      onClose();
      return true;
    } catch (cause) {
      const parsedConflict = parseWorkflowVersionConflict(cause);
      if (parsedConflict) {
        setConflict({
          bundle,
          detail: parsedConflict,
          workflowKey: parsedConflict.workflow_key,
          version: parsedConflict.suggested_version,
        });
        setError('');
        return false;
      }
      setError(apiMessage(cause));
      return false;
    } finally {
      setImporting(false);
    }
  }

  async function submitConflictResolution() {
    if (!conflict || importing) return;
    const nextKey = conflict.workflowKey.trim();
    const nextVersion = conflict.version.trim();
    if (!isValidWorkflowKey(nextKey)) {
      setError('工作流标识格式无效：仅支持字母、数字、点、下划线和短横线，且需以字母或数字开头（最多 120 字符）');
      return;
    }
    if (!isValidSemver(nextVersion)) {
      setError('版本号必须符合 SemVer 格式，例如 1.0.1 或 1.2.0-custom.1');
      return;
    }
    const updated = applyBundleIdentityOverride(conflict.bundle, nextKey, nextVersion);
    await importBundlePayload(updated);
  }

  async function handleJsonObject(parsed: unknown): Promise<boolean> {
    const classified = classifyWorkflowJson(parsed);
    if (classified.kind === 'plain_comfyui') {
      setPlainComfyNotice(classified.message);
      setError('');
      return false;
    }
    if (classified.kind === 'invalid') {
      setError(classified.message);
      setPlainComfyNotice('');
      return false;
    }
    setError('');
    setPlainComfyNotice('');
    return importBundlePayload(classified.bundle);
  }

  async function handleFile(file: File) {
    const token = ++revision.current;
    setError('');
    setPlainComfyNotice('');
    setCopyFeedback('');
    setBusy(true);
    try {
      const lower = file.name.toLowerCase();
      if (lower.endsWith('.png') || file.type === 'image/png') {
        if (file.size > MAX_IMAGE) throw new Error('图片不能超过 32 MiB');
        const data = await extractPng(new Uint8Array(await file.arrayBuffer()));
        if (token !== revision.current) return;
        setPending({ sourceKind: 'local', label: file.name, file, result: data });
        setSelected(data.workflow ? 'workflow' : 'prompt');
        return;
      }
      const text = await file.text();
      let parsed: unknown;
      try {
        parsed = parseWorkflowJson(text).value;
      } catch {
        throw new Error('JSON 格式错误，请检查文件内容');
      }
      if (token !== revision.current) return;
      await handleJsonObject(parsed);
    } catch (cause) {
      if (token === revision.current) {
        setError(apiMessage(cause));
      }
    } finally {
      if (token === revision.current) setBusy(false);
    }
  }

  async function handlePaste(text: string): Promise<boolean> {
    setError('');
    setPlainComfyNotice('');
    setCopyFeedback('');
    setBusy(true);
    try {
      let parsed: unknown;
      try {
        parsed = parseWorkflowJson(text).value;
      } catch {
        setError('JSON 格式错误，请检查粘贴内容');
        return false;
      }
      return await handleJsonObject(parsed);
    } finally {
      setBusy(false);
    }
  }

  async function handleUrl(rawUrl: string): Promise<boolean> {
    setError('');
    setPlainComfyNotice('');
    setCopyFeedback('');
    const checked = validateWorkflowUrlInput(rawUrl);
    if (!checked.ok) {
      setError(checked.error);
      return false;
    }
    const token = ++revision.current;
    setBusy(true);
    try {
      const fetched = await fetchWorkflowFromUrl(checked.url);
      if (token !== revision.current) return false;
      if (fetched.kind === 'png') {
        const data = fetched.extraction;
        setPending({
          sourceKind: 'remote',
          label: fetched.source_name || fetched.source_host || '远程 PNG 图片',
          sourceHost: fetched.source_host,
          note: fetched.note,
          result: data,
        });
        setSelected(data.workflow ? 'workflow' : 'prompt');
        return true;
      }
      return await importBundlePayload(fetched.bundle);
    } catch (cause) {
      if (token === revision.current) {
        if (cause instanceof ApiError && cause.code === 'UNSUPPORTED_WORKFLOW_FORMAT') {
          setPlainComfyNotice(cause.message);
        } else {
          setError(apiMessage(cause));
        }
      }
      return false;
    } finally {
      if (token === revision.current) setBusy(false);
    }
  }

  const previewJson = pending?.result?.[selected]
    ? JSON.stringify(pending.result[selected], null, 2)
    : '';

  function downloadExtracted(key: 'workflow' | 'prompt') {
    const payload = pending?.result?.[key];
    if (!payload) return;
    const blobUrl = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = `${key}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  }

  async function importPendingPngDraft() {
    if (!pending || importing) return;
    const payload = {
      workflow_key: `image-${crypto.randomUUID()}`,
      image_extraction: {
        workflow: pending.result.workflow,
        prompt: pending.result.prompt,
        metadata: pending.result.metadata,
      },
    };
    await importBundlePayload(payload);
  }

  const anyBusy = busy || importing;

  return (
    <section className={`${styles.card} ${styles.depWorkspace}`} aria-label="导入工作流">
      <div className={styles.depWorkspaceHead}>
        <div className={styles.depWorkspaceTitle}>
          <h3>
            <Upload size={17} strokeWidth={1.7} aria-hidden />
            导入工作流
          </h3>
          <span className="muted small">
            支持 Studio 完整工作流包 JSON（含 manifest、api_graph、object_info）或含 ComfyUI 元数据的 PNG 原图；导入后为草稿，需校验并发布。
          </span>
        </div>
        <div className={styles.titleActions}>
          {(pending || conflict) && (
            <button
              type="button"
              className="button ghost small"
              disabled={anyBusy}
              onClick={resetPreview}
            >
              <RefreshCw size={13} aria-hidden />
              重新选择
            </button>
          )}
          <button
            type="button"
            className={styles.iconGhost}
            aria-label="收起导入面板"
            disabled={anyBusy}
            onClick={onClose}
          >
            <X size={17} aria-hidden />
          </button>
        </div>
      </div>

      {!pending && !conflict && (
        <WorkflowSourceDropzone
          title="拖入或选择工作流文件（.json / .png 原图）"
          description="支持 Studio 工作流包 JSON 及 PNG 原图（本地解包或服务端安全拉取远程链接）；普通 ComfyUI 工作流 JSON 请到「模型库」检测模型依赖。"
          busy={anyBusy}
          busyLabel={importing ? '导入中…' : '解析中…'}
          urlPlaceholder="https://...（支持 Studio 工作流包 .json 或 PNG 原图直链，自动转换 GitHub / HF / Civitai 常见链接）"
          urlSubmitLabel="解析链接"
          urlBusyLabel="拉取解析中…"
          pasteLabel="粘贴 JSON 文本"
          pastePlaceholder="在这里粘贴 Studio 工作流包 JSON（含 manifest、api_graph、object_info）"
          pasteSubmitLabel="导入为草稿"
          pasteBusyLabel="导入中…"
          onSelectFile={handleFile}
          onSubmitUrl={handleUrl}
          onSubmitPaste={handlePaste}
        />
      )}

      {plainComfyNotice && (
        <div className={`${styles.depNoticeStrip} ${styles.wfImportNoticeRow}`} role="status">
          <p className="small">{plainComfyNotice}</p>
          {onOpenModelLibrary && (
            <div className="actions">
              <button
                type="button"
                className="button secondary small"
                onClick={onOpenModelLibrary}
              >
                前往模型库检测模型
              </button>
            </div>
          )}
        </div>
      )}

      {conflict && (
        <div className={`${styles.depNoticeStrip} ${styles.wfConflictPanel}`} role="region" aria-label="解决工作流版本冲突">
          <div className={styles.depSummaryMeta}>
            <AlertTriangle size={15} aria-hidden />
            <strong>
              版本冲突：「{conflict.detail.workflow_key}」v{conflict.detail.workflow_version} 已存在
            </strong>
            {conflict.detail.existing_state === 'deleted' && (
              <span className={styles.depTag}>历史已删除版本仍占号</span>
            )}
          </div>
          <p className="small muted">
            可直接使用建议的新版本号，或修改工作流标识 / 版本号后重新导入为草稿。
            {conflict.detail.existing_versions.length > 0 && (
              <> 已有版本：{conflict.detail.existing_versions.map((v) => `v${v}`).join('、')}。</>
            )}
          </p>
          <div className={styles.wfConflictFields}>
            <label className={styles.wfConflictField}>
              <span className="small">工作流标识（workflow_key）</span>
              <input
                type="text"
                aria-label="工作流标识"
                value={conflict.workflowKey}
                disabled={anyBusy}
                onChange={(e) => {
                  setConflict((curr) => (curr ? { ...curr, workflowKey: e.target.value } : null));
                  setError('');
                }}
              />
            </label>
            <label className={styles.wfConflictField}>
              <span className="small">版本号（SemVer，例如 1.0.1）</span>
              <input
                type="text"
                aria-label="新版本号"
                value={conflict.version}
                disabled={anyBusy}
                onChange={(e) => {
                  setConflict((curr) => (curr ? { ...curr, version: e.target.value } : null));
                  setError('');
                }}
              />
            </label>
          </div>
          <div className="actions">
            {conflict.workflowKey.trim() !== conflict.detail.suggested_workflow_key && (
              <button
                type="button"
                className="button ghost small"
                disabled={anyBusy}
                onClick={() => {
                  setConflict((curr) => (
                    curr ? { ...curr, workflowKey: curr.detail.suggested_workflow_key } : null
                  ));
                  setError('');
                }}
              >
                改用新标识：{conflict.detail.suggested_workflow_key}
              </button>
            )}
            {conflict.version.trim() !== conflict.detail.suggested_version && (
              <button
                type="button"
                className="button ghost small"
                disabled={anyBusy}
                onClick={() => {
                  setConflict((curr) => (
                    curr ? { ...curr, version: curr.detail.suggested_version } : null
                  ));
                  setError('');
                }}
              >
                恢复建议版本：v{conflict.detail.suggested_version}
              </button>
            )}
          </div>
          <div className="actions">
            <button
              type="button"
              className="button small"
              disabled={anyBusy}
              onClick={() => void submitConflictResolution()}
            >
              {importing ? '导入中…' : '按新版本导入为草稿'}
            </button>
            <button
              type="button"
              className="button ghost small"
              disabled={anyBusy}
              onClick={resetPreview}
            >
              取消
            </button>
          </div>
        </div>
      )}

      {error && <p role="alert" className="status error">{error}</p>}

      {pending && (
        <div className={styles.wfImportPreview}>
          <div className={styles.depSummaryBar}>
            <div className={styles.depSummaryMeta}>
              {pending.sourceKind === 'remote' ? (
                <Globe size={15} aria-hidden />
              ) : (
                <ImageIcon size={15} aria-hidden />
              )}
              <strong>{pending.label}</strong>
              {pending.sourceKind === 'remote' && pending.sourceHost && (
                <span className={styles.depTag}>来源：{pending.sourceHost}</span>
              )}
              <span className={styles.depTag}>
                workflow：{pending.result.workflow ? '已检测' : '未检测'}
              </span>
              <span className={styles.depTag}>
                API prompt：{pending.result.prompt ? '已检测' : '未检测'}
              </span>
              <span className="muted small">{pending.result.node_count} 个节点</span>
            </div>
          </div>

          {pending.note && (
            <div className={styles.depInfoStrip}>
              <p className="muted small">{pending.note}</p>
            </div>
          )}

          {thumbUrl && (
            <img
              src={thumbUrl}
              alt="导入图片缩略图"
              className={styles.wfImportThumb}
            />
          )}

          <p className="small">
            模型 / LoRA / VAE / ControlNet 文件线索：{pending.result.model_files.join('、') || '未识别'}
          </p>
          <p className="small">
            节点类型（含待审核自定义节点，不能仅凭类型确认来源）：{pending.result.node_types.join('、') || '未识别'}
          </p>

          {pending.result.warnings.length > 0 && (
            <div className={styles.depNoticeStrip}>
              {pending.result.warnings.map((w) => <p key={w} className="muted small">{w}</p>)}
            </div>
          )}

          <p className="muted small">
            草稿仍需补充模型来源与 SHA-256、ComfyUI 版本、节点依赖、object_info、参数映射、GPU 和许可。不会自动安装节点或标记为已验证。
          </p>

          <div className="actions">
            {(['workflow', 'prompt'] as const).filter((k) => pending.result[k]).map((k) => (
              <button
                type="button"
                className="button secondary small"
                key={k}
                onClick={() => downloadExtracted(k)}
              >
                <Download size={13} aria-hidden />
                下载 {k}.json
              </button>
            ))}
            <button
              className="button secondary small"
              type="button"
              onClick={() => {
                void navigator.clipboard.writeText(previewJson).then(
                  () => setCopyFeedback('已复制 JSON'),
                  () => setError('复制失败，请从 JSON 预览中手动复制'),
                );
              }}
            >
              <Copy size={13} aria-hidden />
              {copyFeedback || '复制 JSON'}
            </button>
          </div>

          <div className={styles.depSubPanel}>
            <div className={styles.depSubPanelHead}>
              <strong>JSON 预览</strong>
              <select
                aria-label="JSON 预览格式"
                className={styles.depCompactSelect}
                value={selected}
                onChange={(e) => {
                  setSelected(e.target.value as 'workflow' | 'prompt');
                  setCopyFeedback('');
                }}
              >
                {(['workflow', 'prompt'] as const).filter((k) => pending.result[k]).map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>
            <textarea
              className={styles.code}
              aria-label="提取 JSON 预览"
              readOnly
              rows={10}
              value={previewJson}
            />
          </div>

          <div className="actions">
            <button
              className="button small"
              type="button"
              disabled={anyBusy}
              onClick={() => void importPendingPngDraft()}
            >
              {importing ? '导入中…' : '导入为待配置草稿'}
            </button>
            <button
              className="button ghost small"
              type="button"
              disabled={anyBusy}
              onClick={resetPreview}
            >
              取消 / 重新选择
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
