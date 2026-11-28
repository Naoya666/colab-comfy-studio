import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  HardDrive,
  Link2,
  Loader2,
  Upload,
  X,
} from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import { useBodyScrollLock } from '../lib/modal-scroll-lock';
import {
  bytesLabel,
  displayModelTitles,
  findCatalogDuplicates,
  libraryApi,
  modelStatus,
  shortHash,
  uploadModel,
  type DuplicateMatchResult,
  type LibraryModel,
  type ResolvedModelFile,
} from '../lib/model-library';
import styles from '../gpu.module.css';

type AddTab = 'url' | 'upload' | 'drive';

function DuplicateAlert({
  match,
  onLocate,
}: {
  match: DuplicateMatchResult;
  onLocate: (model: LibraryModel) => void;
}) {
  if (match.level === 'none') return null;
  const targets = match.level === 'exact' ? match.exactMatches : match.nameMatches;
  const isExact = match.level === 'exact';

  return (
    <div className={styles.modelDupBox} data-level={match.level} role="status">
      <div className={styles.modelDupHead}>
        <AlertTriangle size={14} strokeWidth={1.8} aria-hidden />
        <strong>
          {isExact ? `库内已有相同模型（${match.reason}）` : `检测到库内存在同名文件（${targets.length} 个）`}
        </strong>
      </div>
      <div className={styles.modelDupList}>
        {targets.slice(0, 3).map((existing) => {
          const title = displayModelTitles(existing);
          return (
            <div key={existing.id} className={styles.modelDupRow}>
              <div className={styles.modelDupMeta}>
                <span className={styles.modelDupName}>{title.primary}</span>
                <small className="muted">
                  {existing.model_type} · {bytesLabel(existing.size)} · {modelStatus[existing.state] || existing.state}
                </small>
              </div>
              <button
                type="button"
                className="button ghost small"
                onClick={() => onLocate(existing)}
              >
                <ExternalLink size={12} aria-hidden />
                定位查看
              </button>
            </div>
          );
        })}
        {targets.length > 3 && (
          <small className="muted">另有 {targets.length - 3} 个匹配模型</small>
        )}
      </div>
    </div>
  );
}

export function ModelAddDialog({
  open,
  availableTypes,
  defaultType,
  models,
  onClose,
  onAdded,
  onLocateModel,
}: {
  open: boolean;
  availableTypes: string[];
  defaultType?: string;
  models: LibraryModel[];
  onClose: () => void;
  onAdded: (model: LibraryModel, message: string) => Promise<void>;
  onLocateModel: (model: LibraryModel) => void;
}) {
  const [tab, setTab] = useState<AddTab>('url');
  const [type, setType] = useState(defaultType || availableTypes[0] || 'checkpoints');
  const [architecture, setArchitecture] = useState('unknown');
  const [name, setName] = useState('');
  const [relativePath, setRelativePath] = useState('');
  const [expected, setExpected] = useState('');
  const [url, setUrl] = useState('');
  const [driveId, setDriveId] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [choices, setChoices] = useState<ResolvedModelFile[]>([]);
  const [choiceTypes, setChoiceTypes] = useState<Record<number, string>>({});
  const [addedKeys, setAddedKeys] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [statusMsg, setStatusMsg] = useState('');
  const [uploadPct, setUploadPct] = useState<number | null>(null);

  useBodyScrollLock(open);

  useEffect(() => {
    if (!open) return;
    setError('');
    setStatusMsg('');
    setUploadPct(null);
    if (defaultType && availableTypes.includes(defaultType)) {
      setType(defaultType);
    }
  }, [open, defaultType, availableTypes]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, busy, onClose]);

  const uploadDuplicate = useMemo(() => {
    if (!selectedFile) return { level: 'none', exactMatches: [], nameMatches: [], reason: '' } as DuplicateMatchResult;
    return findCatalogDuplicates(
      {
        filename: selectedFile.name,
        expected_size: selectedFile.size,
        expected_sha256: expected.trim() || undefined,
        model_type: type,
      },
      models,
    );
  }, [selectedFile, expected, type, models]);

  const driveDuplicate = useMemo(() => {
    if (!driveId.trim() && !name.trim()) {
      return { level: 'none', exactMatches: [], nameMatches: [], reason: '' } as DuplicateMatchResult;
    }
    return findCatalogDuplicates(
      {
        drive_file_id: driveId.trim() || undefined,
        filename: name.trim() || undefined,
        expected_sha256: expected.trim() || undefined,
        model_type: type,
      },
      models,
    );
  }, [driveId, name, expected, type, models]);

  if (!open) return null;

  async function resolveSource() {
    if (!url.trim() || busy) return;
    setBusy(true);
    setError('');
    setStatusMsg('');
    setChoices([]);
    setChoiceTypes({});
    setAddedKeys({});
    try {
      const resolved = await libraryApi<{ files: ResolvedModelFile[] }>('/resolve', {
        url: url.trim(),
        filename: name.trim() || undefined,
        model_type: type,
        architecture: architecture.trim() || 'unknown',
        expected_sha256: expected.trim() || undefined,
      });
      const files = resolved.files || [];
      setChoices(files);
      const nextTypes: Record<number, string> = {};
      files.forEach((f, idx) => {
        nextTypes[idx] = f.model_type && f.model_type !== 'unknown' && availableTypes.includes(f.model_type)
          ? f.model_type
          : type;
      });
      setChoiceTypes(nextTypes);
      if (!files.length) {
        setStatusMsg('该来源未返回受支持的模型文件');
      }
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function commitAdd(data: Record<string, unknown>, key?: string, keepOpen = false) {
    setBusy(true);
    setError('');
    setStatusMsg('');
    try {
      const targetType = typeof data.model_type === 'string' && data.model_type && data.model_type !== 'unknown'
        ? data.model_type
        : type;
      const targetArch = typeof data.architecture === 'string' && data.architecture && data.architecture !== 'unknown'
        ? data.architecture
        : (architecture.trim() || 'unknown');
      const model = await libraryApi<LibraryModel>('', {
        ...data,
        relative_path: relativePath.trim() || undefined,
        model_type: targetType,
        architecture: targetArch,
      });
      await libraryApi(`/${model.id}/prepare`, {});
      const msg = `「${model.display_name || model.filename}」已入库并加入准备队列`;
      await onAdded(model, msg);
      if (key) {
        setAddedKeys((curr) => ({ ...curr, [key]: true }));
      }
      if (keepOpen) {
        setStatusMsg(msg);
      } else {
        setUrl('');
        setName('');
        setRelativePath('');
        setExpected('');
        setDriveId('');
        setSelectedFile(null);
        setChoices([]);
        setUploadPct(null);
        onClose();
      }
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  async function submitLocalUpload() {
    if (!selectedFile || busy) return;
    setBusy(true);
    setError('');
    setStatusMsg('');
    setUploadPct(0);
    try {
      const uploadedDriveId = await uploadModel(selectedFile, (sent) => {
        const pct = selectedFile.size > 0 ? Math.min(100, Math.round((sent / selectedFile.size) * 100)) : 100;
        setUploadPct(pct);
        setStatusMsg(`正在分块上传到 Drive… ${pct}%`);
      }, type);
      const model = await libraryApi<LibraryModel>('', {
        source: 'upload',
        drive_file_id: uploadedDriveId,
        filename: selectedFile.name,
        relative_path: relativePath.trim() || undefined,
        model_type: type,
        architecture: architecture.trim() || 'unknown',
        expected_sha256: expected.trim() || undefined,
      });
      await libraryApi(`/${model.id}/prepare`, {});
      const msg = `「${model.display_name || model.filename}」已上传并加入校验队列`;
      await onAdded(model, msg);
      setSelectedFile(null);
      setRelativePath('');
      setExpected('');
      setUploadPct(null);
      onClose();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function handleLocate(model: LibraryModel) {
    onClose();
    onLocateModel(model);
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={() => { if (!busy) onClose(); }}>
      <section
        className={`${styles.dialog} ${styles.modelAddDialog}`}
        role="dialog"
        aria-modal="true"
        aria-label="添加模型"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.cardHead}>
          <h2>添加模型</h2>
          <button type="button" className={styles.iconGhost} disabled={busy} aria-label="关闭" onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        </div>

        <div className={styles.segmented} role="tablist" aria-label="添加模型方式">
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'url'}
            data-active={tab === 'url'}
            disabled={busy}
            onClick={() => { setTab('url'); setError(''); setStatusMsg(''); }}
          >
            <Link2 size={14} strokeWidth={1.8} aria-hidden />
            <span>链接下载</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'upload'}
            data-active={tab === 'upload'}
            disabled={busy}
            onClick={() => { setTab('upload'); setError(''); setStatusMsg(''); }}
          >
            <Upload size={14} strokeWidth={1.8} aria-hidden />
            <span>本地上传</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'drive'}
            data-active={tab === 'drive'}
            disabled={busy}
            onClick={() => { setTab('drive'); setError(''); setStatusMsg(''); }}
          >
            <HardDrive size={14} strokeWidth={1.8} aria-hidden />
            <span>导入 Drive</span>
          </button>
        </div>

        {tab === 'url' && (
          <div className={styles.stack}>
            <div className={styles.modelAddPrimaryGrid}>
              <label className="field">
                模型页面或 HTTPS 下载链接
                <input
                  value={url}
                  disabled={busy}
                  placeholder="https://huggingface.co/... 或 Civitai 页面 / 直链"
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && url.trim() && !busy) void resolveSource(); }}
                />
              </label>
              <label className="field">
                模型类型
                <select value={type} disabled={busy} onChange={(e) => setType(e.target.value)}>
                  {availableTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
            </div>

            <p className="muted small">
              {type === 'upscale_models'
                ? '超分模型（upscale_models）支持 safetensors、gguf、onnx，以及来自 Hugging Face / 受信任 GitHub Release 的受控 .pt/.pth（仅做字节与安全结构校验）。'
                : type === 'ultralytics_bbox' || type === 'ultralytics_segm'
                  ? 'Ultralytics 模型（ultralytics_bbox / ultralytics_segm）支持经安全校验的 .pt 权重以及 safetensors、gguf、onnx，映射至 models/ultralytics/bbox/ 或 segm/。'
                  : '默认支持 safetensors、gguf、onnx；若导入超分模型 .pt/.pth 或 Ultralytics .pt，请切换至对应分类。'}
            </p>

            <details className="foldable">
              <summary>高级选项（架构 / 相对路径 / 预期 SHA-256 / 文件名）</summary>
              <div className="foldContent">
                <div className={styles.grid2}>
                  <label className="field">
                    模型架构
                    <input value={architecture} disabled={busy} onChange={(e) => setArchitecture(e.target.value)} placeholder="unknown" />
                  </label>
                  <label className="field">
                    文件名（通用直链可选）
                    <input value={name} disabled={busy} onChange={(e) => setName(e.target.value)} placeholder="model.safetensors" />
                  </label>
                  <label className="field">
                    工作流引用的相对路径（可选）
                    <input value={relativePath} disabled={busy} onChange={(e) => setRelativePath(e.target.value)} placeholder="子目录/model.safetensors" />
                  </label>
                  <label className="field">
                    预期 SHA-256（可选）
                    <input value={expected} disabled={busy} onChange={(e) => setExpected(e.target.value)} placeholder="64 位十六进制" />
                  </label>
                </div>
              </div>
            </details>

            <div className="actions">
              <button
                type="button"
                disabled={busy || !url.trim()}
                className="button"
                onClick={() => void resolveSource()}
              >
                {busy && !choices.length ? <Loader2 size={14} className={styles.spin} aria-hidden /> : null}
                解析来源并检测重复
              </button>
            </div>

            {choices.length > 0 && (
              <div className={styles.modelChoiceList}>
                <div className={styles.storageRowHead}>
                  <strong className="small">解析结果（{choices.length} 个文件）</strong>
                  <span className="muted small">已自动比对现有模型库</span>
                </div>
                {choices.map((c, idx) => {
                  const chosenType = choiceTypes[idx] || type;
                  const choiceKey = `${idx}:${c.filename}`;
                  const dup = findCatalogDuplicates({ ...c, model_type: chosenType }, models);
                  const alreadyAdded = Boolean(addedKeys[choiceKey]);
                  const sizeVal = Number(c.expected_size || c.reported_size || 0);

                  return (
                    <div key={choiceKey} className={styles.modelChoiceCard} data-dup={dup.level}>
                      <div className={styles.modelChoiceHead}>
                        <div className={styles.modelChoiceMain}>
                          <strong className={styles.depFileName}>{c.filename}</strong>
                          <div className={styles.depMetaRow}>
                            {c.version_name && <span className={styles.depTag}>{c.version_name}</span>}
                            {c.source && <span className={styles.depTag}>{c.source}</span>}
                            {c.architecture && c.architecture !== 'unknown' && <span className={styles.depTag}>{c.architecture}</span>}
                            {sizeVal > 0 && <span className={styles.depTag}>{bytesLabel(sizeVal)}</span>}
                            {typeof c.expected_sha256 === 'string' && c.expected_sha256 && (
                              <span className={styles.depHashChip} title={c.expected_sha256}>
                                SHA {shortHash(c.expected_sha256)}
                              </span>
                            )}
                          </div>
                        </div>
                        <select
                          aria-label={`${c.filename} 模型类型`}
                          className={styles.miniSelect}
                          value={chosenType}
                          disabled={busy || alreadyAdded}
                          onChange={(e) => setChoiceTypes((prev) => ({ ...prev, [idx]: e.target.value }))}
                        >
                          {availableTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                        </select>
                      </div>

                      {!alreadyAdded && <DuplicateAlert match={dup} onLocate={handleLocate} />}

                      <div className={styles.modelChoiceFooter}>
                        {alreadyAdded ? (
                          <span className={styles.depDoneLine}>
                            <CheckCircle2 size={14} aria-hidden />
                            已入库并加入准备队列
                          </span>
                        ) : dup.level === 'exact' ? (
                          <>
                            <button
                              type="button"
                              className="button small"
                              onClick={() => handleLocate(dup.exactMatches[0])}
                            >
                              定位已有模型
                            </button>
                            <button
                              type="button"
                              disabled={busy}
                              className="button secondary small"
                              onClick={() => void commitAdd({ ...c, model_type: chosenType }, choiceKey, choices.length > 1)}
                            >
                              仍要重复入库
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            disabled={busy}
                            className="button small"
                            onClick={() => void commitAdd({ ...c, model_type: chosenType }, choiceKey, choices.length > 1)}
                          >
                            {dup.level === 'name' ? '确认非重复并准备' : '选择此文件并准备'}
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {tab === 'upload' && (
          <div className={styles.stack}>
            <div className={styles.modelAddPrimaryGrid}>
              <label className="field">
                选择本地权重文件（直传 Drive）
                <input
                  disabled={busy}
                  type="file"
                  accept={type === 'ultralytics_bbox' || type === 'ultralytics_segm' ? '.pt,.safetensors,.gguf,.onnx' : '.safetensors,.gguf,.onnx'}
                  onChange={(e) => {
                    const f = e.target.files?.[0] || null;
                    setSelectedFile(f);
                    setError('');
                    setStatusMsg('');
                  }}
                />
              </label>
              <label className="field">
                模型类型
                <select value={type} disabled={busy} onChange={(e) => setType(e.target.value)}>
                  {availableTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
            </div>

            {selectedFile && (
              <div className={styles.modelChoiceCard} data-dup={uploadDuplicate.level}>
                <div className={styles.modelChoiceMain}>
                  <strong className={styles.depFileName}>{selectedFile.name}</strong>
                  <small className="muted">待上传大小：{bytesLabel(selectedFile.size)} · 目标分类：{type}</small>
                </div>
                <DuplicateAlert match={uploadDuplicate} onLocate={handleLocate} />
                {uploadPct !== null && (
                  <div className={styles.progressBlock}>
                    <div className={styles.progressText}>
                      <span>上传进度</span>
                      <span>{uploadPct}%</span>
                    </div>
                    <div className={styles.progress}>
                      <span style={{ width: `${uploadPct}%` }} />
                    </div>
                  </div>
                )}
              </div>
            )}

            <details className="foldable">
              <summary>高级选项（架构 / 相对路径 / 预期 SHA-256）</summary>
              <div className="foldContent">
                <div className={styles.grid2}>
                  <label className="field">
                    模型架构
                    <input value={architecture} disabled={busy} onChange={(e) => setArchitecture(e.target.value)} placeholder="unknown" />
                  </label>
                  <label className="field">
                    预期 SHA-256（可选）
                    <input value={expected} disabled={busy} onChange={(e) => setExpected(e.target.value)} placeholder="64 位十六进制" />
                  </label>
                </div>
                <label className="field">
                  工作流引用的相对路径（可选）
                  <input value={relativePath} disabled={busy} onChange={(e) => setRelativePath(e.target.value)} placeholder="子目录/model.safetensors" />
                </label>
              </div>
            </details>

            <p className="muted small">
              Drive OAuth 的 drive.file 权限仅可读取应用可见文件。上传失败会在当前页面自动重试分块；刷新页面后需重新上传。
            </p>

            <div className="actions">
              {selectedFile && uploadDuplicate.level === 'exact' ? (
                <>
                  <button type="button" className="button" onClick={() => handleLocate(uploadDuplicate.exactMatches[0])}>
                    定位已有模型
                  </button>
                  <button type="button" disabled={busy} className="button secondary" onClick={() => void submitLocalUpload()}>
                    仍要重复上传
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  disabled={busy || !selectedFile}
                  className="button"
                  onClick={() => void submitLocalUpload()}
                >
                  {busy ? <Loader2 size={14} className={styles.spin} aria-hidden /> : <Upload size={14} aria-hidden />}
                  {uploadDuplicate.level === 'name' ? '确认非重复并开始上传' : '开始上传并校验'}
                </button>
              )}
            </div>
          </div>
        )}

        {tab === 'drive' && (
          <div className={styles.stack}>
            <div className={styles.grid2}>
              <label className="field">
                已有 Drive 文件 ID
                <input
                  value={driveId}
                  disabled={busy}
                  placeholder="粘贴 Google Drive File ID"
                  onChange={(e) => setDriveId(e.target.value)}
                />
              </label>
              <label className="field">
                文件名
                <input
                  value={name}
                  disabled={busy}
                  placeholder="model.safetensors"
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              <label className="field">
                模型类型
                <select value={type} disabled={busy} onChange={(e) => setType(e.target.value)}>
                  {availableTypes.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
              <label className="field">
                模型架构
                <input value={architecture} disabled={busy} onChange={(e) => setArchitecture(e.target.value)} placeholder="unknown" />
              </label>
            </div>

            <DuplicateAlert match={driveDuplicate} onLocate={handleLocate} />

            <details className="foldable">
              <summary>高级选项（相对路径 / 预期 SHA-256）</summary>
              <div className="foldContent">
                <div className={styles.grid2}>
                  <label className="field">
                    工作流引用的相对路径（可选）
                    <input value={relativePath} disabled={busy} onChange={(e) => setRelativePath(e.target.value)} placeholder="子目录/model.safetensors" />
                  </label>
                  <label className="field">
                    预期 SHA-256（可选）
                    <input value={expected} disabled={busy} onChange={(e) => setExpected(e.target.value)} placeholder="64 位十六进制" />
                  </label>
                </div>
              </div>
            </details>

            <p className="muted small">
              外部导入的 Drive 文件仅建立索引并校验，删除时只能移除索引（保留 Drive 原文件）。
            </p>

            <div className="actions">
              {driveDuplicate.level === 'exact' ? (
                <>
                  <button type="button" className="button" onClick={() => handleLocate(driveDuplicate.exactMatches[0])}>
                    定位已有模型
                  </button>
                  <button
                    type="button"
                    disabled={busy || !driveId.trim() || !name.trim()}
                    className="button secondary"
                    onClick={() => void commitAdd({
                      source: 'drive',
                      drive_file_id: driveId.trim(),
                      filename: name.trim(),
                      model_type: type,
                      architecture: architecture.trim() || 'unknown',
                      expected_sha256: expected.trim() || undefined,
                    })}
                  >
                    仍要重复导入
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  disabled={busy || !driveId.trim() || !name.trim()}
                  className="button"
                  onClick={() => void commitAdd({
                    source: 'drive',
                    drive_file_id: driveId.trim(),
                    filename: name.trim(),
                    model_type: type,
                    architecture: architecture.trim() || 'unknown',
                    expected_sha256: expected.trim() || undefined,
                  })}
                >
                  {driveDuplicate.level === 'name' ? '确认非重复并导入校验' : '导入已有文件并校验'}
                </button>
              )}
            </div>
          </div>
        )}

        {error && <p className="status error" role="alert">{error}</p>}
        {statusMsg && <p className="status" role="status">{statusMsg}</p>}
      </section>
    </div>
  );
}
