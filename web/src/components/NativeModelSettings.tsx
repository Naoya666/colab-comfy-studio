export type ModelStorage = {
  model_storage_mode: 'drive_mount' | 'local';
  cache_gb: number;
  cache_max_age_minutes: number;
};

export function NativeModelSettings({ value, onChange }: {
  value: ModelStorage;
  onChange: (value: ModelStorage) => void;
}) {
  return <>
    <label className="field">模型存储模式
      <select value={value.model_storage_mode}
        onChange={e => onChange({ ...value, model_storage_mode: e.target.value as ModelStorage['model_storage_mode'] })}>
        <option value="drive_mount">Drive 挂载模式</option>
        <option value="local">本地模式（按需完整下载）</option>
      </select>
    </label>
    {value.model_storage_mode === 'drive_mount' && <>
      <label className="field">挂载缓存容量（GB）
        <input type="number" min={1} max={2147483647} required value={value.cache_gb || ''}
          onChange={e => onChange({ ...value, cache_gb: Number(e.target.value) })} />
      </label>
      <label className="field">挂载缓存未访问时的保留时间（分钟）
        <input type="number" min={1} max={153722867} required value={value.cache_max_age_minutes || ''}
          onChange={e => onChange({ ...value, cache_max_age_minutes: Number(e.target.value) })} />
        <span className="muted small">rclone 根据上次访问时间回收 VFS 缓存。实例空闲回收和会话上限另行设置。</span>
      </label>
    </>}
  </>;
}
