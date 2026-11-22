import { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Clock, FolderGit2, RefreshCw, SearchCheck, Wrench } from 'lucide-react';
import { apiMessage } from '../lib/gpu-api';
import {
  ACTIVE_MODEL_STATES,
  dependencyApi,
  libraryApi,
  type Catalog,
  type DependencyRecord,
  type LibraryModel,
  type ModelTask,
} from '../lib/model-library';
import { DependencyCard } from './ModelDependencyItem';
import styles from '../gpu.module.css';

type WorkflowDepFilter = 'all' | 'unbound' | 'ready';

export function ModelDependencies({ workflowId, onDraft }: { workflowId: string; onDraft?: () => void }) {
  const [record, setRecord] = useState<DependencyRecord | null>(null);
  const [models, setModels] = useState<LibraryModel[]>([]);
  const [tasks, setTasks] = useState<ModelTask[]>([]);
  const [types, setTypes] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busyAction, setBusyAction] = useState('');
  const [filter, setFilter] = useState<WorkflowDepFilter>('all');

  const refresh = useCallback(async () => {
    const [r, c, t] = await Promise.all([
      dependencyApi<DependencyRecord>(workflowId),
      libraryApi<Catalog>(),
      libraryApi<ModelTask[]>('/tasks').catch(() => [] as ModelTask[]),
    ]);
    setRecord(r);
    setModels(c.items);
    setTypes(Object.keys(c.types || {}));
    setTasks(t);
  }, [workflowId]);

  useEffect(() => {
    void refresh().catch((e) => setError(apiMessage(e)));
    const id = setInterval(() => {
      void refresh().catch(() => {});
    }, 4000);
    return () => clearInterval(id);
  }, [refresh]);

  async function action(key: string, fn: () => Promise<unknown>) {
    setError('');
    setBusyAction(key);
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      setBusyAction('');
    }
  }

  const stats = useMemo(() => {
    const list = record?.dependencies || [];
    let ready = 0;
    let unbound = 0;
    let inProgress = 0;
    for (const dep of list) {
      if (dep.status === 'matched') ready += 1;
      else if (ACTIVE_MODEL_STATES.has(dep.status)) inProgress += 1;
      if (!dep.model_id) unbound += 1;
    }
    return { total: list.length, ready, unbound, inProgress };
  }, [record]);

  const dependencies = useMemo(() => {
    const list = record?.dependencies || [];
    if (filter === 'unbound') return list.filter((dep) => !dep.model_id);
    if (filter === 'ready') return list.filter((dep) => dep.status === 'matched');
    return list;
  }, [record, filter]);

  return (
    <details className={`foldable ${styles.depFoldable}`}>
      <summary>
        <span className={styles.depFoldSummary}>
          <span>模型依赖</span>
          {record && (
            <span className={styles.depStatusBadge} data-tone={record.models_ready ? 'ready' : stats.inProgress > 0 ? 'active' : 'warn'}>
              {record.models_ready ? <CheckCircle2 size={12} aria-hidden /> : <Clock size={12} aria-hidden />}
              {record.models_ready ? '已就绪' : '待准备'} · {stats.ready}/{stats.total}
            </span>
          )}
        </span>
      </summary>
      <div className="foldContent">
        {/* Top status summary & actions */}
        <div className={styles.depSummaryBar}>
          <div className={styles.depSummaryMeta}>
            <span className={styles.depStatusBadge} data-tone={record?.models_ready ? 'ready' : 'warn'}>
              {record?.models_ready ? '模型依赖已准备完成' : '模型依赖待准备'}
            </span>
            <span className={styles.depTag}>工作流：{record?.workflow_state || '加载中'}</span>
            <span className="muted small">
              {record?.runtime_validated ? '已通过真实运行验证' : '运行环境 / 节点及真实运行仍需验证'}
            </span>
          </div>

          <div className="actions">
            <button
              type="button"
              className="button secondary small"
              disabled={Boolean(busyAction)}
              onClick={() => void action('scan', async () => {
                const res = await dependencyApi<{ unresolved?: string[] }>(workflowId, 'scan');
                setMessage(res.unresolved?.length ? `${res.unresolved.length} 项仍需补充来源或确认候选` : '已重新扫描依赖');
              })}
            >
              <RefreshCw size={13} className={busyAction === 'scan' ? styles.spin : undefined} aria-hidden />
              重新扫描
            </button>
            <button
              type="button"
              className="button secondary small"
              disabled={Boolean(busyAction)}
              onClick={() => void action('match', async () => {
                const res = await dependencyApi<{ unresolved?: string[] }>(workflowId, 'match');
                setMessage(res.unresolved?.length ? `${res.unresolved.length} 项仍需补充来源或确认候选` : '已完成模型库匹配');
              })}
            >
              <SearchCheck size={13} aria-hidden />
              重新匹配
            </button>
            <button
              type="button"
              className="button small"
              disabled={Boolean(busyAction)}
              onClick={() => void action('prepare', async () => {
                const res = await dependencyApi<{ unresolved?: string[] }>(workflowId, 'prepare');
                setMessage(res.unresolved?.length ? `${res.unresolved.length} 项仍需补充来源或确认候选` : '已提交全部缺失模型准备任务');
              })}
            >
              准备全部缺失模型
            </button>
            <button
              type="button"
              className="button secondary small"
              disabled={Boolean(busyAction)}
              onClick={() => void action('adopt-cache', async () => {
                await dependencyApi(workflowId, 'adopt-cache');
                setMessage('已提交旧缓存索引与校验');
              })}
            >
              <Wrench size={13} aria-hidden />
              索引并校验旧缓存
            </button>
            <button
              type="button"
              className="button secondary small"
              disabled={Boolean(busyAction)}
              onClick={() => void action('draft', async () => {
                await dependencyApi(workflowId, 'draft');
                setMessage('已创建绑定草稿，请在工作流列表配置、审核并验证');
                onDraft?.();
              })}
            >
              <FolderGit2 size={13} aria-hidden />
              将当前绑定生成新草稿
            </button>
          </div>
        </div>

        <p className="muted small">
          模型准备、来源可信度和工作流运行验证分别管理。替换已审核模型后，可生成绑定草稿并按现有流程审核、验证和发布。
        </p>

        {error && <p role="alert" className="status error">{error}</p>}
        {message && <p role="status" className="status">{message}</p>}

        {/* Filter chips */}
        {record && stats.total > 0 && (
          <div className={styles.depFilterBar}>
            {([
              ['all', `全部 ${stats.total}`],
              ['unbound', `未绑定 ${stats.unbound}`],
              ['ready', `已就绪 ${stats.ready}`],
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

        {record && !record.dependencies.length && (
          <p className="muted small">未发现可可靠识别的模型引用。可为自定义节点配置识别规则。</p>
        )}
        {filter === 'unbound' && Boolean(record?.dependencies.length) && !dependencies.length && (
          <p className="muted small" role="status">所有模型依赖均已绑定。</p>
        )}

        {dependencies.length > 0 && (
          <div className={styles.depList}>
            {dependencies.map((dep) => (
              <DependencyCard
                key={dep.id}
                mode="workflow"
                dep={dep}
                models={models}
                tasks={tasks}
                availableTypes={types}
                workflowId={workflowId}
                refresh={refresh}
                onBind={(id, strengths, disabled) => {
                  const chosen = models.find((m) => m.id === id);
                  const unknown = dep.architecture === 'unknown' || chosen?.architecture === 'unknown';
                  if (unknown && !window.confirm('依赖或模型架构未知，请确认该模型兼容当前工作流。模型身份变化仍需重新审核。')) {
                    return;
                  }
                  void action('bind', () => dependencyApi(workflowId, 'bind', {
                    dependency_id: dep.id,
                    model_id: id,
                    strengths,
                    disabled,
                    confirm_unknown: unknown,
                  }));
                }}
              />
            ))}
          </div>
        )}
      </div>
    </details>
  );
}

export interface ModelOverride {
  dependency_id: string;
  model_id: string;
  strengths?: Record<string, number>;
  disabled?: boolean;
  confirm_unknown: boolean;
}

export function ModelRunOverrides({
  workflowId,
  value,
  onChange,
}: {
  workflowId: string;
  value: ModelOverride[];
  onChange: (v: ModelOverride[]) => void;
}) {
  const [record, setRecord] = useState<DependencyRecord | null>(null);
  const [models, setModels] = useState<LibraryModel[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    setError('');
    setRecord(null);
    void Promise.all([dependencyApi<DependencyRecord>(workflowId), libraryApi<Catalog>()])
      .then(([r, c]) => {
        setRecord(r);
        setModels(c.items);
      })
      .catch((e) => setError(apiMessage(e)));
  }, [workflowId]);

  const overridable = record?.dependencies.filter((d) => ['checkpoints', 'loras'].includes(d.model_type)) || [];

  return (
    <details className="foldable">
      <summary>本次底模 / LoRA 覆盖（{value.length} 项）</summary>
      <div className="foldContent">
        <p className="muted small">
          仅影响本次任务。新模型需先在同一工作流系列通过审核和真实运行验收，之后可直接切换；未知架构需确认。
        </p>
        {error && <p className="status error">{error}</p>}
        <div className={styles.depList}>
          {overridable.map((dep) => {
            const override = value.find((o) => o.dependency_id === dep.id);
            const chosen = override?.model_id || dep.model_id || '';
            function update(
              model_id: string,
              strengths = override?.strengths || dep.strengths,
              disabled = override?.disabled || false,
            ) {
              const model = models.find((m) => m.id === model_id);
              if (!model) return;
              const unknown = dep.architecture === 'unknown' || model.architecture === 'unknown';
              if (unknown && !window.confirm('模型或依赖架构未知，请确认兼容当前工作流。')) return;
              onChange([
                ...value.filter((v) => v.dependency_id !== dep.id),
                { dependency_id: dep.id, model_id, strengths, disabled, confirm_unknown: unknown },
              ]);
            }
            return (
              <div key={dep.id} className={styles.depCard}>
                <div className={styles.depBindGrid}>
                  <label className="field">
                    <span className="label">{dep.model_type} · 节点 {dep.node_id}（{dep.reference}）</span>
                    <select value={chosen} onChange={(e) => update(e.target.value)}>
                      <option value="">使用工作流原模型</option>
                      {models
                        .filter((m) => m.model_type === dep.model_type && m.state === 'ready')
                        .map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.display_name} · {m.architecture}
                          </option>
                        ))}
                    </select>
                  </label>
                  {Object.entries(override?.strengths || dep.strengths || {}).map(([field, strength]) => (
                    <label key={field} className="field">
                      <span className="label">{field}</span>
                      <input
                        type="number"
                        min={-3}
                        max={3}
                        step={0.05}
                        value={strength}
                        onChange={(e) => update(chosen, { ...(override?.strengths || dep.strengths), [field]: Number(e.target.value) })}
                      />
                    </label>
                  ))}
                </div>
                {!!Object.keys(dep.strengths || {}).length && (
                  <label className={styles.checkRow}>
                    <input
                      type="checkbox"
                      checked={!!override?.disabled}
                      onChange={(e) => update(chosen, undefined, e.target.checked)}
                    />
                    <span>关闭此 LoRA</span>
                  </label>
                )}
              </div>
            );
          })}
        </div>
        {record && !record.dependencies.some((d) => d.model_type === 'loras') && (
          <p className="muted small">此工作流没有可识别的 LoRA 节点，不能直接新增 LoRA。</p>
        )}
        <div className="actions">
          <button type="button" className="button secondary small" disabled={!value.length} onClick={() => onChange([])}>
            清除本次覆盖
          </button>
        </div>
      </div>
    </details>
  );
}
