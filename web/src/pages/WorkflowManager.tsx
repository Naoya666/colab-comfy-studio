import { ModelDependencies } from '../components/ModelDependencies';
import { WorkflowImportWorkspace, type WorkflowImportResult } from '../components/WorkflowImportWorkspace';
import { ImageDraftConfiguration } from '../components/ImageDraftConfiguration';
import { Fragment, Suspense, lazy, useCallback, useMemo, useState } from 'react';
import { ChevronDown, Download, Search, Trash2, Upload, X } from 'lucide-react';
import type { GpuWorkflow, WorkflowExportResponse } from '../lib/gpu-api';
import { Pager } from '../components/ui';
import { WorkflowActivationPanel } from '../components/WorkflowActivation';
import type { WorkflowGraphPreviewData } from '../components/WorkflowGraphPreview';
const WorkflowGraphPreview = lazy(() => import('../components/WorkflowGraphPreview').then((m) => ({ default: m.WorkflowGraphPreview })));
import { ConfirmDialog } from '../components/Feedback';
import { readActivationWatch, writeActivationWatch, type ActivationWatch } from '../lib/activation-resume';
import { shortHash } from '../lib/model-library';
import { workflowSearchText, workflowTitle, workflowUsage } from '../lib/workflow-copy';
import styles from '../gpu.module.css';
import { workflowState } from '../lib/ui-helpers';

const GROUPS_PER_PAGE = 6;
const VERSIONS_SHOWN = 4;

const cmp = (a: GpuWorkflow, b: GpuWorkflow) =>
  String(b.workflow_version).localeCompare(String(a.workflow_version), undefined, { numeric: true });

async function fetchWorkflowGraphPreview(workflow: Pick<GpuWorkflow, 'workflow_key' | 'workflow_version' | 'state'>): Promise<WorkflowGraphPreviewData> {
  const query = new URLSearchParams({ version: String(workflow.workflow_version) });
  if (workflow.state !== 'published') query.set('draft', 'true');
  const response = await fetch(`/api/gpu/workflows/${encodeURIComponent(workflow.workflow_key)}/preview?${query}`, { credentials: 'include' });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = body && typeof body === 'object' && 'error' in body ? (body as { error?: { message?: string } }).error : undefined;
    throw new Error(error?.message || `预览请求失败（${response.status}）`);
  }
  return (body && typeof body === 'object' && 'data' in body ? body.data : body) as WorkflowGraphPreviewData;
}

export function WorkflowsPanel({
  workflows, projectId = '', projectTitle = '', onImport, onExport, onValidate, onPublish, onRollback, onDeactivate, onDelete, deleting = false, onActivated, onOpenJob, onUseProject, onOpenModelLibrary,
}: {
  workflows: GpuWorkflow[];
  projectId?: string;
  projectTitle?: string;
  onImport: (bundle: string) => Promise<boolean | WorkflowImportResult> | void;
  onExport?: (workflow: GpuWorkflow) => Promise<WorkflowExportResponse | null | void>;
  onValidate: (workflow: GpuWorkflow) => void;
  onPublish: (workflow: GpuWorkflow) => void;
  onRollback: (workflow: GpuWorkflow) => void;
  onDeactivate?: (workflow: GpuWorkflow) => void;
  onDelete?: (workflow: GpuWorkflow) => void;
  deleting?: boolean;
  onActivated: () => void;
  onOpenJob?: (jobId: string, projectId?: string) => void;
  onUseProject?: (projectId: string) => void;
  onOpenModelLibrary?: () => void;
}) {
  const [importOpen, setImportOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(0);
  const [activating, setActivating] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [exportReport, setExportReport] = useState<WorkflowExportResponse | null>(null);
  const [previewing, setPreviewing] = useState<GpuWorkflow | null>(null);
  const [deletingTarget, setDeletingTarget] = useState<GpuWorkflow | null>(null);
  const [openKeys, setOpenKeys] = useState<Set<string>>(new Set());
  const [showAll, setShowAll] = useState<Set<string>>(new Set());
  const [watch, setWatch] = useState<Record<string, ActivationWatch>>(() => readActivationWatch());

  async function handleExport(wf: GpuWorkflow) {
    if (!onExport || exportingId) return;
    setExportingId(wf.id);
    try {
      const res = await onExport(wf);
      if (res && (res.summary.warnings.length > 0 || res.summary.rebind_models.length > 0)) {
        setExportReport(res);
      } else if (res) {
        setExportReport(null);
      }
    } finally {
      setExportingId(null);
    }
  }
  const rememberWatch = useCallback((workflowId: string, next: ActivationWatch | null) => {
    setWatch((current) => {
      const copy = { ...current };
      if (!next) delete copy[workflowId];
      else copy[workflowId] = next;
      writeActivationWatch(copy);
      return copy;
    });
  }, []);

  // 同一个 workflow_key（同一个模型流程）的所有版本收成一组，最新版本排最前
  const groups = useMemo(() => {
    const map = new Map<string, { key: string; modelId: string; items: GpuWorkflow[] }>();
    workflows.forEach((wf) => {
      const g = map.get(wf.workflow_key) || { key: wf.workflow_key, modelId: wf.model_id, items: [] };
      g.items.push(wf);
      map.set(wf.workflow_key, g);
    });
    const list = [...map.values()];
    list.forEach((g) => g.items.sort(cmp));
    return list.sort((a, b) => a.key.localeCompare(b.key));
  }, [workflows]);

  const q = query.trim().toLowerCase();
  const visible = groups.filter((g) => !q || g.items.some((item) => workflowSearchText(item).toLowerCase().includes(q)) || `${g.key} ${g.modelId}`.toLowerCase().includes(q));
  const pages = Math.max(1, Math.ceil(visible.length / GROUPS_PER_PAGE));
  const safePage = Math.min(page, pages - 1);
  const slice = visible.slice(safePage * GROUPS_PER_PAGE, (safePage + 1) * GROUPS_PER_PAGE);

  const toggle = (set: Set<string>, setter: (s: Set<string>) => void, key: string) => {
    const next = new Set(set);
    if (next.has(key)) next.delete(key); else next.add(key);
    setter(next);
  };

  return (
    <div className={styles.stack}>
      <div className={styles.pageTitle}>
        <h2>工作流管理</h2>
        <div className={styles.titleActions}>
          <button
            className={`button ${importOpen ? '' : 'secondary'} small`}
            type="button"
            aria-expanded={importOpen}
            onClick={() => setImportOpen((v) => !v)}
          >
            <Upload size={14} strokeWidth={1.8} aria-hidden />
            导入工作流
          </button>
        </div>
      </div>

      <p className="muted small">
        共 {groups.length} 个工作流、{workflows.length} 个版本。同一个工作流的版本收在一起，点开可查看全部。
      </p>

      <WorkflowImportWorkspace
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImport={onImport}
        onOpenModelLibrary={onOpenModelLibrary}
      />

      {exportReport && (
        <div className={`${styles.depNoticeStrip} ${styles.wfExportSummary}`} role="status">
          <div className={styles.wfExportSummaryHead}>
            <strong>已导出分享包：{exportReport.filename}</strong>
            <button
              type="button"
              className={styles.iconGhost}
              aria-label="关闭导出提示"
              onClick={() => setExportReport(null)}
            >
              <X size={15} aria-hidden />
            </button>
          </div>
          {exportReport.summary.warnings.map((note) => (
            <p key={note} className="small">{note}</p>
          ))}
          {exportReport.summary.rebind_models.length > 0 && (
            <ul className={styles.wfRebindList}>
              {exportReport.summary.rebind_models.map((item) => (
                <li key={`${item.source}:${item.target}:${item.sha256 || item.filename}`} className="small">
                  <span className={styles.depTag}>{item.source === 'civitai' ? 'Civitai' : '需重绑'}</span>
                  <strong>{item.display_name || item.filename}</strong>
                  {item.filename && item.display_name && item.filename !== item.display_name ? (
                    <span className="muted">（{item.filename}）</span>
                  ) : null}
                  {item.sha256 ? <span className="muted"> · SHA {shortHash(item.sha256)}</span> : null}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <label className={styles.searchBox}>
        <Search size={17} strokeWidth={1.6} aria-hidden />
        <input aria-label="搜索工作流" placeholder="搜索工作流名称" value={query} onChange={(e) => { setQuery(e.target.value); setPage(0); }} />
      </label>

      {slice.length ? (
        <div className={styles.list}>
          {slice.map((g) => {
            const current = g.items.find((i) => i.active);
            const sample = current || g.items[0];
            const isOpen = openKeys.has(g.key);
            const all = showAll.has(g.key);
            const shown = all ? g.items : g.items.slice(0, VERSIONS_SHOWN);
            const activeWatch = g.items.map((item) => watch[item.id]).find((item) => item && ['queued', 'running', 'pending'].includes(item.state.toLowerCase()));
            return (
              <article className={styles.wfGroup} key={g.key} data-open={isOpen}>
                <button type="button" className={styles.wfGroupHead} aria-expanded={isOpen} onClick={() => toggle(openKeys, setOpenKeys, g.key)}>
                  <span>
                    <strong>{workflowTitle(sample)}</strong>
                    <small className="muted">{workflowUsage(sample)}</small>
                    <small className="muted">{g.key} · {g.items.length} 个版本</small>
                  </span>
                  {current ? <span className="pill solid">当前 v{current.workflow_version}</span> : g.items.some((item) => item.state === 'published') ? <span className="pill">未设为当前</span> : <span className="pill">未启用</span>}
                  <ChevronDown size={16} strokeWidth={1.6} aria-hidden className={styles.chev} />
                </button>
                {activeWatch && activating !== activeWatch.workflowId ? (
                  <p className="muted small" style={{ padding: '0 12px 10px' }}>
                    启用冒烟仍在进行（最长 {activeWatch.maxRuntimeMinutes ?? '以服务端为准'} 分钟）。收起后点「查看启用进度」即可继续看，不会重新提交。
                    <button className="button ghost small" type="button" onClick={() => { setOpenKeys((curr) => new Set(curr).add(g.key)); setActivating(activeWatch.workflowId); }}>查看启用进度</button>
                  </p>
                ) : null}
                {isOpen && (
                  <div className={styles.versionList}>
                    {shown.map((wf) => {
                      const published = wf.state === 'published';
                      return (
                        <Fragment key={wf.id}>
                        <div className={styles.versionItem}>
                          <span className={styles.versionName}>
                            <strong>v{wf.workflow_version}</strong>
                            <span className="pill" data-state={wf.active ? 'active' : wf.state}>{workflowState(wf.state, wf.active)}</span>
                          </span>
                          <span className={styles.versionActions}>
                            <button className="button secondary small" type="button" onClick={() => setPreviewing(wf)}>离线预览</button>
                            {published && onExport && (
                              <button
                                className="button secondary small"
                                type="button"
                                disabled={exportingId === wf.id}
                                title="导出不含个人 Drive ID 的 Studio 工作流分享包"
                                onClick={() => void handleExport(wf)}
                              >
                                <Download size={14} aria-hidden />
                                {exportingId === wf.id ? '导出中…' : '导出'}
                              </button>
                            )}
                            {!published && (
                              <button className="button small" type="button" disabled={Array.isArray(wf.pending_configuration) && wf.pending_configuration.length > 0} title={Array.isArray(wf.pending_configuration) && wf.pending_configuration.length ? '请先补充图片草稿配置' : undefined} onClick={() => setActivating(activating === wf.id ? null : wf.id)}>
                                {activating === wf.id ? '收起启用向导' : watch[wf.id] && ['queued', 'running', 'pending'].includes(watch[wf.id].state.toLowerCase()) ? '查看启用进度' : '一键启用'}
                              </button>
                            )}
                            {!published && <button className="button secondary small" type="button" onClick={() => onValidate(wf)}>校验</button>}
                            {!published && <button className="button ghost small" type="button" onClick={() => onPublish(wf)} title="发布不会跳过冒烟和模型启用检查">发布</button>}
                            {published && wf.active && onDeactivate ? <button className="button secondary small" type="button" title="不再作为默认版本。已经在跑的任务会继续完成。" onClick={() => onDeactivate(wf)}>停用</button> : null}
                            {published && !wf.active && <button className="button secondary small" type="button" onClick={() => onRollback(wf)}>设为当前</button>}
                            {onDelete && <button className="button danger small" type="button" disabled={deleting} onClick={() => setDeletingTarget(wf)}><Trash2 size={14} aria-hidden />删除</button>}
                          </span>
                        </div>
                        <ModelDependencies workflowId={wf.id} onDraft={onActivated} />
                        {wf.image_extraction && wf.state === 'draft' ? <ImageDraftConfiguration workflow={wf} onDone={onActivated} /> : null}
                        {activating === wf.id && (
                          <div className={styles.versionDetail}>
                            <WorkflowActivationPanel
                              workflow={wf}
                              projectId={projectId}
                              projectTitle={projectTitle}
                              onClose={() => setActivating(null)}
                              onDone={onActivated}
                              onPreview={() => setPreviewing(wf)}
                              onOpenJob={onOpenJob}
                              onUseProject={onUseProject}
                              onWatch={(next) => rememberWatch(wf.id, next)}
                            />
                          </div>
                        )}
                        {!published && activating !== wf.id && (wf.manifest?.parameters && Object.values(wf.manifest.parameters).some((parameter) => parameter && typeof parameter === 'object' && (parameter as { type?: string }).type === 'image')) ? (
                          <p className="muted small" style={{ padding: '0 12px 8px' }}>启用前要在向导里上传必填图片。直接发布不会跳过冒烟，模型未启用时会被拒绝。</p>
                        ) : null}
                        </Fragment>
                      );
                    })}
                    {g.items.length > VERSIONS_SHOWN && (
                      <button type="button" className="button ghost small" onClick={() => toggle(showAll, setShowAll, g.key)}>
                        {all ? '收起旧版本' : `显示全部 ${g.items.length} 个版本`}
                      </button>
                    )}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty"><strong>{workflows.length ? '没有匹配的工作流' : '还没有工作流'}</strong></div>
      )}

      <Pager page={safePage} pages={pages} onChange={setPage} />
      {previewing && (
        <Suspense fallback={null}>
          <WorkflowGraphPreview workflow={previewing} loadPreview={() => fetchWorkflowGraphPreview(previewing)} onClose={() => setPreviewing(null)} />
        </Suspense>
      )}
      <ConfirmDialog
        open={Boolean(deletingTarget)}
        title="删除工作流版本？"
        description={deletingTarget ? `将删除「${workflowTitle(deletingTarget)}」v${deletingTarget.workflow_version}。该版本将无法再用于新任务，已有任务和作品会保留。` : ''}
        confirmLabel="确认删除"
        danger
        busy={deleting}
        onCancel={() => setDeletingTarget(null)}
        onConfirm={() => {
          if (deletingTarget && onDelete) onDelete(deletingTarget);
          setDeletingTarget(null);
        }}
      />
    </div>
  );
}
