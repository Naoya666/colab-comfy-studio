import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Clock,
  Copy,
  Download,
  ExternalLink,
  HelpCircle,
  Link2,
  Loader2,
  Search,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import {
  ACTIVE_MODEL_STATES,
  bytesLabel,
  defaultSearchQuery,
  dependencyApi,
  dependencySourceUrl,
  libraryApi,
  modelBasename,
  modelStatus,
  safeRelativePathFor,
  shortHash,
  sourceHostLabel,
  type Dependency,
  type LibraryModel,
  type ModelTask,
  type ResolvedModelFile,
  type SourceCandidate,
  type SourceSearchResponse,
} from '../lib/model-library';
import styles from '../gpu.module.css';

type SourceLookup =
  | { kind: 'search' | 'resolve'; pending: true; query?: string }
  | { kind: 'search' | 'resolve'; error: string; query?: string }
  | { kind: 'search'; candidates: SourceCandidate[]; warnings: string[]; query: string }
  | { kind: 'resolve'; files: ResolvedModelFile[]; resolvedUrl: string };

type SourcePlatformFilter = 'all' | 'civitai' | 'huggingface';

function statusTone(status: string, optional: boolean): 'ready' | 'active' | 'warn' | 'danger' | 'muted' {
  if (optional && status !== 'matched' && !ACTIVE_MODEL_STATES.has(status)) return 'muted';
  if (status === 'matched' || status === 'ready') return 'ready';
  if (ACTIVE_MODEL_STATES.has(status)) return 'active';
  if (status === 'failed' || status === 'missing') return 'danger';
  return 'warn';
}

function StatusIcon({ tone }: { tone: ReturnType<typeof statusTone> }) {
  if (tone === 'ready') return <CheckCircle2 size={13} strokeWidth={2} aria-hidden />;
  if (tone === 'active') return <Clock size={13} strokeWidth={2} aria-hidden />;
  if (tone === 'danger') return <AlertTriangle size={13} strokeWidth={2} aria-hidden />;
  if (tone === 'warn') return <AlertTriangle size={13} strokeWidth={2} aria-hidden />;
  return <HelpCircle size={13} strokeWidth={2} aria-hidden />;
}

export interface DependencyCardProps {
  mode: 'workflow' | 'library';
  dep: Dependency;
  models: LibraryModel[];
  tasks?: ModelTask[];
  availableTypes: string[];
  workflowId?: string;
  boundModel?: LibraryModel;
  candidateModels?: LibraryModel[];
  effectiveStatus?: string;
  refresh: () => Promise<void>;
  onBind?: (modelId: string, strengths: Record<string, number>, disabled: boolean) => void;
  onLibraryModelImported?: (depId: string, model: LibraryModel) => void;
  onLocateLibraryModel?: (model: LibraryModel) => void;
}

export function DependencyCard({
  mode,
  dep,
  models,
  tasks = [],
  availableTypes,
  workflowId,
  boundModel,
  candidateModels,
  effectiveStatus,
  refresh,
  onBind,
  onLibraryModelImported,
  onLocateLibraryModel,
}: DependencyCardProps) {
  const sourceUrl = dependencySourceUrl(dep);
  const bound = boundModel ?? models.find((m) => m.id === dep.model_id);
  const candidates = candidateModels ?? (
    dep.candidates?.length
      ? models.filter((m) => dep.candidates?.includes(m.id))
      : []
  );
  const activeTask = bound
    ? tasks.find((t) => t.model_id === bound.id && ACTIVE_MODEL_STATES.has(t.state))
    : undefined;
  const displayStatus = effectiveStatus || (bound
    ? (bound.state === 'ready' ? 'matched' : (activeTask?.state || bound.state))
    : dep.status);
  const isOptional = !dep.required || Boolean(dep.disabled) || Boolean(dep.bypassed);
  const tone = statusTone(displayStatus, isOptional);
  const isManualReview = displayStatus === 'manual_review';
  const isReady = displayStatus === 'matched' || displayStatus === 'ready';

  const [chosen, setChosen] = useState(dep.model_id || '');
  const [strengths, setStrengths] = useState<Record<string, number>>(dep.strengths || {});
  const [disabled, setDisabled] = useState(dep.disabled || false);
  const [targetType, setTargetType] = useState(dep.model_type !== 'unknown' ? dep.model_type : '');
  const [url, setUrl] = useState(sourceUrl || '');
  const [searchQuery, setSearchQuery] = useState(() => defaultSearchQuery(dep.reference));
  const [platformFilter, setPlatformFilter] = useState<SourcePlatformFilter>('all');
  const [lookup, setLookup] = useState<SourceLookup | null>(null);
  const [showManualUrl, setShowManualUrl] = useState(false);
  const [showBindEditor, setShowBindEditor] = useState(false);
  const [submittingFile, setSubmittingFile] = useState(false);
  const [actionError, setActionError] = useState('');
  const [actionNotice, setActionNotice] = useState('');
  const [copiedKey, setCopiedKey] = useState<string>('');

  const pending = lookup !== null && 'pending' in lookup;

  useEffect(() => {
    setChosen(dep.model_id || '');
    setStrengths(dep.strengths || {});
    setDisabled(dep.disabled || false);
  }, [dep]);

  useEffect(() => {
    if (dep.model_type !== 'unknown') {
      setTargetType(dep.model_type);
    }
  }, [dep.model_type]);

  useEffect(() => {
    setUrl(sourceUrl || '');
    setSearchQuery(defaultSearchQuery(dep.reference));
  }, [dep.id, dep.reference, sourceUrl]);

  const compatibleModels = useMemo(
    () => models.filter((m) => dep.model_type === 'unknown' || m.model_type === dep.model_type),
    [models, dep.model_type],
  );

  const searchCounts = useMemo(() => {
    if (lookup?.kind !== 'search' || !('candidates' in lookup)) {
      return { all: 0, civitai: 0, huggingface: 0 };
    }
    let civitai = 0;
    let huggingface = 0;
    for (const item of lookup.candidates) {
      const src = (item.source || sourceHostLabel(item.url)).toLowerCase();
      if (src.includes('civitai')) civitai += 1;
      else if (src.includes('hugging')) huggingface += 1;
    }
    return { all: lookup.candidates.length, civitai, huggingface };
  }, [lookup]);

  const filteredCandidates = useMemo(() => {
    if (lookup?.kind !== 'search' || !('candidates' in lookup)) return [];
    if (platformFilter === 'all') return lookup.candidates;
    return lookup.candidates.filter((item) => {
      const src = (item.source || sourceHostLabel(item.url)).toLowerCase();
      return platformFilter === 'civitai' ? src.includes('civitai') : src.includes('hugging');
    });
  }, [lookup, platformFilter]);

  function copyText(key: string, text: string) {
    void navigator.clipboard?.writeText(text).then(() => {
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey((curr) => (curr === key ? '' : curr)), 1600);
    }).catch(() => {});
  }

  async function runSearch(overrideQuery?: string) {
    const q = (overrideQuery ?? searchQuery).trim();
    if (!q) return;
    setPlatformFilter('all');
    setLookup({ kind: 'search', pending: true, query: q });
    setActionError('');
    setActionNotice('');
    try {
      const res = await libraryApi<SourceSearchResponse>('/search?q=' + encodeURIComponent(q));
      setLookup({
        kind: 'search',
        candidates: res.candidates || [],
        warnings: res.warnings || [],
        query: q,
      });
    } catch (e) {
      setLookup({ kind: 'search', error: apiMessage(e), query: q });
    }
  }

  async function runResolve(overrideUrl?: string) {
    const targetUrl = (overrideUrl ?? url).trim();
    if (!targetUrl) return;
    setUrl(targetUrl);
    setLookup({ kind: 'resolve', pending: true });
    setActionError('');
    setActionNotice('');
    try {
      const resolvedModelType = targetType || (dep.model_type !== 'unknown' ? dep.model_type : undefined);
      const { files } = await libraryApi<{ files: ResolvedModelFile[] }>('/resolve', {
        url: targetUrl,
        model_type: resolvedModelType,
        filename: modelBasename(dep.reference),
        expected_sha256: dep.expected_sha256 || undefined,
      });
      setLookup({ kind: 'resolve', files: files || [], resolvedUrl: targetUrl });
    } catch (e) {
      setLookup({ kind: 'resolve', error: apiMessage(e) });
    }
  }

  async function commitResolvedFile(file: ResolvedModelFile) {
    const resolvedType = (
      (dep.model_type !== 'unknown' ? dep.model_type : '')
      || targetType
      || (file.model_type && file.model_type !== 'unknown' ? file.model_type : '')
    );
    if (!resolvedType) {
      setActionError('请先选择目标模型目录（如 diffusion_models / loras / checkpoints），再入库。');
      return;
    }
    if (mode === 'workflow' && !window.confirm('确认使用此具体文件并绑定到当前工作流？请核对模型版本与架构。')) {
      return;
    }
    setSubmittingFile(true);
    setActionError('');
    setActionNotice('');
    try {
      const relPath = safeRelativePathFor(dep.reference, file.filename);
      const model = await libraryApi<LibraryModel>('', {
        ...file,
        model_type: resolvedType,
        ...(relPath ? { relative_path: relPath } : {}),
      });
      if (mode === 'workflow' && workflowId) {
        await dependencyApi(workflowId, 'bind', {
          dependency_id: dep.id,
          model_id: model.id,
          confirm_unknown: true,
        });
      }
      await libraryApi(`/${model.id}/prepare`, {});
      if (mode === 'library') {
        onLibraryModelImported?.(dep.id, model);
        setActionNotice(`已加入模型库准备队列：${model.filename}`);
      } else {
        setActionNotice(`已绑定并加入准备队列：${model.filename}`);
      }
      await refresh();
      setLookup(null);
      setShowManualUrl(false);
    } catch (e) {
      setActionError(apiMessage(e));
    } finally {
      setSubmittingFile(false);
    }
  }

  const strengthEntries = Object.entries(strengths);
  const effectiveFileSize = dep.expected_size || dep.weight?.size;
  const effectiveSha = dep.expected_sha256 || dep.weight?.sha256;
  const expectedBasename = modelBasename(dep.reference);
  const showInlineUrlForm = showManualUrl || (!sourceUrl && !isReady && !isManualReview && !bound);

  return (
    <article className={styles.depCard} data-tone={tone}>
      {/* Row 1: Title + badges + status */}
      <div className={styles.depHead}>
        <div className={styles.depTitleBlock}>
          <div className={styles.depNameRow}>
            <code className={styles.depFileName} title={dep.reference}>{dep.reference}</code>
            {!isManualReview && (
              <button
                type="button"
                className={styles.depCopyBtn}
                title="复制模型名称"
                aria-label={`复制 ${dep.reference}`}
                onClick={() => copyText('name', dep.reference)}
              >
                {copiedKey === 'name' ? <Check size={12} aria-hidden /> : <Copy size={12} aria-hidden />}
              </button>
            )}
          </div>
          <div className={styles.depMetaRow}>
            <span className={styles.depTag}>{dep.model_type}</span>
            {dep.architecture && dep.architecture !== 'unknown' && (
              <span className={styles.depTag}>{dep.architecture}</span>
            )}
            {isOptional && <span className={styles.depTag}>{dep.disabled ? '已关闭' : dep.bypassed ? '已旁路' : '可选'}</span>}
            {dep.detected_via === 'metadata' && <span className={styles.depTag}>工作流元数据</span>}
            {dep.detected_via === 'heuristic' && <span className={styles.depTag}>推断线索</span>}
            <span className="muted small">节点 {dep.node_id} · {dep.node_type}</span>
            {effectiveFileSize ? <span className="muted small">· {bytesLabel(effectiveFileSize)}</span> : null}
            {effectiveSha ? (
              <button
                type="button"
                className={styles.depHashChip}
                title={`点击复制完整 SHA-256：${effectiveSha}`}
                onClick={() => copyText('sha', effectiveSha)}
              >
                SHA {shortHash(effectiveSha)}
                {copiedKey === 'sha' ? <Check size={11} aria-hidden /> : <Copy size={11} aria-hidden />}
              </button>
            ) : null}
          </div>
        </div>

        <span className={styles.depStatusBadge} data-tone={tone}>
          <StatusIcon tone={tone} />
          {modelStatus[displayStatus] || displayStatus}
        </span>
      </div>

      {/* Bound / matched library model summary */}
      {bound && (
        <div className={styles.depMatchedBar}>
          <div className={styles.depMatchedInfo}>
            <strong>{bound.display_name}</strong>
            <span className="muted small">
              {bound.relative_path || bound.filename} · {modelStatus[bound.state] || bound.state} · {bytesLabel(bound.size)}
              {strengthEntries.length > 0 && ` · ${strengthEntries.map(([k, v]) => `${k}=${v}`).join(' / ')}`}
            </span>
            {bound.trigger_words?.length ? (
              <span className="muted small">触发词：{bound.trigger_words.join(', ')}</span>
            ) : null}
            {bound.error && <span className={styles.errorText}>{bound.error}</span>}
          </div>
          <div className={styles.depMatchedActions}>
            {mode === 'library' && onLocateLibraryModel && (
              <button type="button" className="button ghost small" onClick={() => onLocateLibraryModel(bound)}>
                在库中查看
              </button>
            )}
            {bound.state !== 'ready' && !activeTask && (
              <button
                type="button"
                className="button secondary small"
                onClick={() => void libraryApi(`/${bound.id}/prepare`, {}).then(refresh).catch((e) => setActionError(apiMessage(e)))}
              >
                准备模型
              </button>
            )}
          </div>
        </div>
      )}

      {/* Live download / verify progress */}
      {activeTask && (
        <div className={styles.depProgressBox}>
          <div className={styles.progressText}>
            <span>{modelStatus[activeTask.state] || activeTask.state}</span>
            <span>
              {activeTask.bytes ? bytesLabel(activeTask.bytes) : '0 B'}
              {activeTask.total ? ` / ${bytesLabel(activeTask.total)}` : ''}
              {activeTask.speed ? ` · ${(activeTask.speed / 1024 ** 2).toFixed(1)} MB/s` : ''}
            </span>
          </div>
          <div className={styles.progress} data-indeterminate={!activeTask.total}>
            <span style={activeTask.total ? { width: `${Math.min(100, (activeTask.bytes / activeTask.total) * 100)}%` } : undefined} />
          </div>
        </div>
      )}

      {/* Existing same-filename candidates in local model library */}
      {!bound && candidates.length > 0 && (
        <div className={styles.depCandidateHint}>
          <span className="small">
            模型库已有同名文件（{candidates.length} 项）：
            {candidates.map((m) => `${m.display_name}（${modelStatus[m.state] || m.state}）`).join('、')}
          </span>
          {mode === 'library' && onLocateLibraryModel && (
            <button type="button" className="button ghost small" onClick={() => onLocateLibraryModel(candidates[0])}>
              定位模型
            </button>
          )}
          {mode === 'workflow' && onBind && candidates.length === 1 && (
            <button
              type="button"
              className="button secondary small"
              onClick={() => onBind(candidates[0].id, strengths, disabled)}
            >
              直接绑定此文件
            </button>
          )}
        </div>
      )}

      {/* Embedded workflow source URL (when present) */}
      {sourceUrl && (
        <div className={styles.depSourceRow}>
          <div className={styles.depSourceMain}>
            <span className={styles.depSourceBadge}>
              <Link2 size={12} aria-hidden />
              {sourceHostLabel(sourceUrl)}
            </span>
            <a href={sourceUrl} target="_blank" rel="noreferrer" className={styles.depSourceLink} title={sourceUrl}>
              {sourceUrl}
            </a>
          </div>
          <div className={styles.depSourceBtns}>
            <button
              type="button"
              className="button ghost small"
              title="复制来源链接"
              onClick={() => copyText('sourceUrl', sourceUrl)}
            >
              {copiedKey === 'sourceUrl' ? <Check size={13} aria-hidden /> : <Copy size={13} aria-hidden />}
              {copiedKey === 'sourceUrl' ? '已复制' : '复制'}
            </button>
            <a href={sourceUrl} target="_blank" rel="noreferrer" className="button ghost small" title="新标签页打开链接">
              <ExternalLink size={13} aria-hidden />
              打开
            </a>
            {!isReady && !activeTask && (
              <button
                type="button"
                className="button small"
                disabled={pending || submittingFile}
                onClick={() => void runResolve(sourceUrl)}
              >
                <Download size={13} aria-hidden />
                {mode === 'library' ? '解析并入库准备' : '解析并绑定准备'}
              </button>
            )}
          </div>
        </div>
      )}

      {dep.source_error && <p className={styles.errorText}>{dep.source_error}</p>}
      {dep.source_conflict && (
        <p className="muted small">检测到该节点存在多个冲突的来源元数据，建议核对后再解析入库。</p>
      )}

      {/* Action bar */}
      {!isManualReview && (
        <div className={styles.depActionBar}>
          {dep.model_type === 'unknown' && mode === 'library' && !bound && (
            <label className={styles.depInlineSelect}>
              <span className="muted small">入库目录</span>
              <select
                aria-label={`${dep.reference} 入库目录`}
                value={targetType}
                onChange={(e) => setTargetType(e.target.value)}
              >
                <option value="">选择目录</option>
                {availableTypes.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </label>
          )}

          <button
            type="button"
            className="button secondary small"
            disabled={pending || submittingFile}
            onClick={() => void runSearch()}
          >
            <Search size={13} aria-hidden />
            搜索来源候选
          </button>

          {(!showInlineUrlForm || sourceUrl) && (
            <button
              type="button"
              className="button ghost small"
              onClick={() => setShowManualUrl((open) => !open)}
            >
              <Link2 size={13} aria-hidden />
              {showManualUrl ? '收起链接输入' : sourceUrl ? '更换 / 手动填链接' : '手动填链接'}
            </button>
          )}

          {mode === 'workflow' && onBind && (
            <button
              type="button"
              className="button ghost small"
              aria-expanded={showBindEditor}
              onClick={() => setShowBindEditor((open) => !open)}
            >
              <SlidersHorizontal size={13} aria-hidden />
              {showBindEditor ? '收起绑定配置' : bound ? '替换模型 / 权重' : '从模型库绑定'}
            </button>
          )}
        </div>
      )}

      {/* Workflow bind / replace & LoRA weight editor (collapsible) */}
      {mode === 'workflow' && onBind && showBindEditor && !isManualReview && (
        <div className={styles.depSubPanel}>
          <div className={styles.depBindGrid}>
            <label className="field">
              <span className="label">绑定 / 替换模型库文件</span>
              <select value={chosen} onChange={(e) => setChosen(e.target.value)}>
                <option value="">选择模型库文件</option>
                {compatibleModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name} · {m.architecture} · {modelStatus[m.state] || m.state}
                  </option>
                ))}
              </select>
            </label>
            {strengthEntries.map(([key, value]) => (
              <label key={key} className="field">
                <span className="label">{key}</span>
                <input
                  type="number"
                  min={-3}
                  max={3}
                  step={0.05}
                  value={value}
                  onChange={(e) => setStrengths({ ...strengths, [key]: Number(e.target.value) })}
                />
              </label>
            ))}
          </div>
          <div className={styles.depBindFooter}>
            {strengthEntries.length > 0 ? (
              <label className={styles.checkRow}>
                <input type="checkbox" checked={disabled} onChange={(e) => setDisabled(e.target.checked)} />
                <span>关闭此 LoRA</span>
              </label>
            ) : <span />}
            <button
              type="button"
              className="button small"
              disabled={!chosen}
              onClick={() => onBind(chosen, strengths, disabled)}
            >
              保存绑定与权重
            </button>
          </div>
        </div>
      )}

      {/* Inline manual URL input + resolve */}
      {showInlineUrlForm && !isManualReview && (
        <form
          className={styles.depInlineUrlForm}
          onSubmit={(e) => {
            e.preventDefault();
            void runResolve();
          }}
        >
          <input
            type="url"
            aria-label={`${dep.reference} 模型来源链接`}
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="粘贴 Civitai / Hugging Face / HTTPS 链接"
          />
          <button
            type="submit"
            className="button secondary small"
            disabled={!url.trim() || pending || submittingFile}
          >
            解析链接
          </button>
        </form>
      )}

      {actionError && <p role="alert" className="status error">{actionError}</p>}
      {actionNotice && <p role="status" className="status">{actionNotice}</p>}

      {/* Lookup Panel: Search Source Candidates */}
      {lookup?.kind === 'search' && (
        <section className={styles.depSubPanel} aria-label="搜索来源候选">
          <div className={styles.depSubPanelHead}>
            <strong>来源候选搜索</strong>
            <button
              type="button"
              className={styles.iconGhost}
              aria-label="关闭搜索候选"
              onClick={() => setLookup(null)}
            >
              <X size={15} aria-hidden />
            </button>
          </div>

          <form
            className={styles.depInlineUrlForm}
            onSubmit={(e) => {
              e.preventDefault();
              void runSearch();
            }}
          >
            <input
              aria-label="候选搜索关键词"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="可删减版本后缀微调关键词后重搜"
            />
            <button type="submit" className="button secondary small" disabled={!searchQuery.trim() || pending}>
              <Search size={13} aria-hidden />
              重搜
            </button>
          </form>

          {'pending' in lookup && (
            <p role="status" className={styles.depLoadingLine}>
              <Loader2 size={14} className={styles.spin} aria-hidden />
              正在 Civitai 与 Hugging Face 搜索「{lookup.query}」…
            </p>
          )}
          {'error' in lookup && <p role="alert" className="status error">{lookup.error}</p>}

          {'candidates' in lookup && (
            <>
              {lookup.warnings.length > 0 && (
                <p className="muted small">{lookup.warnings.join('；')}</p>
              )}

              {lookup.candidates.length > 0 ? (
                <>
                  <div className={styles.depFilterBar}>
                    {([
                      ['all', `全部 ${searchCounts.all}`],
                      ['civitai', `Civitai ${searchCounts.civitai}`],
                      ['huggingface', `Hugging Face ${searchCounts.huggingface}`],
                    ] as const).map(([key, label]) => (
                      <button
                        key={key}
                        type="button"
                        className={styles.depFilterChip}
                        data-active={platformFilter === key}
                        onClick={() => setPlatformFilter(key)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  <div className={styles.depCandidateList}>
                    {filteredCandidates.map((candidate, index) => {
                      const hostLabel = sourceHostLabel(candidate.url, candidate.source);
                      const shortUrl = candidate.url.replace(/^https?:\/\//, '');
                      return (
                        <div className={styles.depCandidateItem} key={`${candidate.url}-${index}`}>
                          <div className={styles.depCandidateMain}>
                            <div className={styles.depCandidateTitleRow}>
                              <span
                                className={styles.depSourceBadge}
                                data-platform={hostLabel === 'Civitai' ? 'civitai' : hostLabel === 'Hugging Face' ? 'hf' : 'other'}
                              >
                                {hostLabel === 'Hugging Face' ? 'HF' : hostLabel}
                              </span>
                              <a
                                href={candidate.url}
                                target="_blank"
                                rel="noreferrer"
                                className={styles.depCandidateName}
                                title={candidate.name}
                              >
                                {candidate.name}
                              </a>
                            </div>
                            <span className={styles.depCandidateUrl}>{shortUrl}</span>
                          </div>
                          <div className={styles.depCandidateActions}>
                            <a
                              href={candidate.url}
                              target="_blank"
                              rel="noreferrer"
                              className="button ghost small"
                              title="在新标签页确认"
                            >
                              <ExternalLink size={13} aria-hidden />
                              打开
                            </a>
                            <button
                              type="button"
                              className="button secondary small"
                              disabled={pending || submittingFile}
                              onClick={() => {
                                setShowManualUrl(true);
                                void runResolve(candidate.url);
                              }}
                            >
                              填入并解析
                            </button>
                          </div>
                        </div>
                      );
                    })}
                    {!filteredCandidates.length && (
                      <p className="muted small">当前平台筛选下暂无结果，可切换「全部」或调整关键词。</p>
                    )}
                  </div>
                </>
              ) : (
                <p role="status" className="muted small">
                  未找到匹配候选，可尝试缩短上方关键词（如只保留模型核心名）重新搜索。
                </p>
              )}
            </>
          )}
        </section>
      )}

      {/* Lookup Panel: Resolved Concrete Files */}
      {lookup?.kind === 'resolve' && (
        <section className={styles.depSubPanel} aria-label="解析来源候选">
          <div className={styles.depSubPanelHead}>
            <strong>解析出的模型文件</strong>
            <button
              type="button"
              className={styles.iconGhost}
              aria-label="关闭文件解析列表"
              onClick={() => setLookup(null)}
            >
              <X size={15} aria-hidden />
            </button>
          </div>

          {'pending' in lookup && (
            <p role="status" className={styles.depLoadingLine}>
              <Loader2 size={14} className={styles.spin} aria-hidden />
              正在解析模型链接并读取文件元信息…
            </p>
          )}
          {'error' in lookup && <p role="alert" className="status error">{lookup.error}</p>}

          {'files' in lookup && (
            lookup.files.length > 0 ? (
              <div className={styles.depFileList}>
                {lookup.files.map((file, index) => {
                  const fileSize = Number(file.expected_size || file.reported_size || 0);
                  const isExactName = file.filename.toLowerCase() === expectedBasename.toLowerCase();
                  const fileSha = typeof file.expected_sha256 === 'string' ? file.expected_sha256 : '';
                  const fileTypeLabel = (file.model_type && file.model_type !== 'unknown')
                    ? file.model_type
                    : (targetType || dep.model_type);
                  return (
                    <div className={styles.depFileRow} key={`${file.filename}-${index}`} data-exact={isExactName}>
                      <div className={styles.depFileMain}>
                        <div className={styles.depNameRow}>
                          <code className={styles.depFileName}>{file.filename}</code>
                          {isExactName && <span className={styles.depExactBadge}>文件名匹配</span>}
                        </div>
                        <div className={styles.depMetaRow}>
                          <span className={styles.depTag}>{fileTypeLabel}</span>
                          {file.architecture && file.architecture !== 'unknown' && (
                            <span className={styles.depTag}>{String(file.architecture)}</span>
                          )}
                          {file.version_name && <span className="muted small">{String(file.version_name)}</span>}
                          <span className="muted small">{bytesLabel(fileSize)}</span>
                          {fileSha && (
                            <span className="muted small" title={fileSha}>SHA {shortHash(fileSha)}</span>
                          )}
                        </div>
                      </div>
                      <button
                        type="button"
                        className="button small"
                        disabled={submittingFile}
                        onClick={() => void commitResolvedFile(file)}
                      >
                        {submittingFile
                          ? '提交中…'
                          : mode === 'library'
                            ? '入库并准备'
                            : '绑定并准备'}
                      </button>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p role="status" className="muted small">该来源未返回受支持的模型文件。</p>
            )
          )}
        </section>
      )}
    </article>
  );
}
