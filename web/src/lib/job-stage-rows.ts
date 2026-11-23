const stages: Array<[string, string]> = [
  ['queued', '排队等待'], ['queue', '排队等待'], ['prep_cache', '准备共享缓存'],
  ['allocate', '分配算力'],
  ['native_system', '安装系统依赖'], ['native_drive', '挂载 Drive 目录'],
  ['native_models', '准备源码与模型目录'], ['native_plugins_restore', '恢复节点归档'],
  ['native_env_cache', '恢复环境缓存'], ['native_python', '安装基础依赖'],
  ['native_plugins_deps', '安装节点依赖'], ['native_environment', '准备原生环境'],
  ['restore_env', '恢复运行环境'],
  ['creation_python', '安装基础依赖'],
  ['restore_weights', '恢复模型权重'], ['restore_loras', '恢复用户 LoRA'],
  ['warm_reuse', '复用已就绪环境'],
  ['start_comfy', '启动 ComfyUI'], ['comfy_start', '启动 ComfyUI'],
  ['native_tunnel', '建立代理连接'],
  ['native_finalize', '服务就绪收尾'],
  ['creation_warming', '预热加载模型'],
  ['sampling', '采样生成'], ['upload', '保存产物'], ['cleanup', '释放算力'], ['delete', '释放算力'],
  ['prep_env', '准备运行环境'], ['prep_weights', '准备模型权重'],
  ['restart_checking', '检查会话空闲状态'],
  ['restart_syncing', '同步模型索引'],
  ['restart_rebooting', '重启 ComfyUI'],
  ['restart_recovering', '等待服务恢复'],
];

const stageLabels = new Map([
  ...stages,
  ['validating', '校验任务'], ['waiting_for_capacity', '等待算力'],
  ['allocating', '分配算力'], ['preparing', '准备运行环境'], ['loading', '加载运行环境'],
  ['running', '运行中'], ['uploading', '保存产物'], ['cleaning_up', '释放算力'],
  ['cleanup_pending', '等待释放算力'],
]);

const stateLabels = new Map([
  ['queued', '排队中'], ['validating', '校验中'], ['waiting_for_capacity', '等待算力'],
  ['allocating', '分配中'], ['preparing', '准备中'], ['loading', '加载中'],
  ['running', '运行中'], ['uploading', '上传中'], ['cleaning_up', '清理中'],
  ['cleanup_pending', '待清理'], ['completed', '已完成'], ['failed', '失败'],
  ['cancelled', '已取消'], ['canceled', '已取消'], ['cancelling', '取消中'],
]);

export function jobStageLabel(value: unknown): string {
  const key = String(value ?? '');
  return stageLabels.get(key) || key || '阶段未知';
}

export function jobStateLabel(value: unknown): string {
  const raw = String(value ?? '').trim();
  const key = raw.toLowerCase();
  return stateLabels.get(key) || raw || '状态未知';
}

export function jobStageRows(value: unknown): Array<{key: string; label: string; seconds: number}> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
  const data = value as Record<string, unknown>;
  const orderedKeys = stages.map(([key]) => key);
  const knownKeys = new Set(orderedKeys);
  const keys = [...orderedKeys, ...Object.keys(data).filter((key) => !knownKeys.has(key))];
  return keys.flatMap((key) => {
    const seconds = data[key];
    return typeof seconds === 'number' && Number.isFinite(seconds) && seconds >= 0
      ? [{ key, label: jobStageLabel(key), seconds }] : [];
  });
}
