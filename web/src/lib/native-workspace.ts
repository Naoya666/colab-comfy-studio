export type WorkspaceWorkflowRef = { source: string; path: string };
export type WorkspaceUnresolvedNode = { type: string; cnr_id?: string | null; aux_id?: string | null };

export type WorkspaceWorkflowItem = {
  id: string;
  source: 'comfy' | 'archived' | 'warmup' | 'studio_db';
  path: string;
  size_bytes: number;
  modified_at?: number | null;
  node_count: number;
  plugins: string[];
  plugin_nodes: Record<string, string[]>;
  unresolved_nodes: WorkspaceUnresolvedNode[];
  parse_error?: string | null;
  readonly?: boolean;
};

export type WorkspacePluginCategory =
  | 'in_use'
  | 'archived_only'
  | 'unused_ui'
  | 'unused'
  | 'manager_disabled'
  | 'disabled';

export type WorkspacePluginRow = {
  path: string;
  version?: string | null;
  repo?: string | null;
  commit?: string | null;
  enabled: boolean;
  archive: string;
  archive_size_bytes: number;
  archive_present: boolean;
  status_category: WorkspacePluginCategory;
  used_by_active: WorkspaceWorkflowRef[];
  used_by_archived: WorkspaceWorkflowRef[];
  import_seconds?: number | null;
  node_count?: number | null;
  js_count?: number | null;
  has_requirements?: boolean | null;
  has_install_py?: boolean | null;
  disabled_at?: number | null;
  origin?: string | null;
};

export type WorkspaceHistoryRow = {
  generation: string;
  backed_up_at?: number | null;
  plugin_count: number;
  total_archive_bytes: number;
  is_head: boolean;
  is_pending: boolean;
  restorable: boolean;
  missing_archives: string[];
};

export type WorkspaceEnvCacheItem = {
  key: string;
  schema?: number | null;
  size_bytes: number;
  created_at?: number | null;
  complete: boolean;
  flags: string[];
  comfy_commit?: string | null;
  python?: string | null;
  gpu_tag?: string | null;
  added_count?: number | null;
  overlay_count?: number | null;
};

export type WorkspaceOverview = {
  offline_writable: boolean;
  lock_reason?: string | null;
  recovery_plan_id?: string | null;
  drive_connected: boolean;
  drive_account_id?: string | null;
  drive_account_alias?: string | null;
  cached_at?: number;
  message?: string;
  plugin_state: {
    head_generation?: string | null;
    pending_generation?: string | null;
    effective_generation?: string | null;
    has_pending_offline_changes: boolean;
    telemetry_available: boolean;
    telemetry_collected_at?: number | null;
    telemetry_gpu?: string | null;
    telemetry_generation?: string | null;
  };
  workflows: {
    active: WorkspaceWorkflowItem[];
    archived: WorkspaceWorkflowItem[];
    warmup: WorkspaceWorkflowItem[];
    studio_published: WorkspaceWorkflowItem[];
  };
  plugins: {
    active: WorkspacePluginRow[];
    disabled: WorkspacePluginRow[];
    history: WorkspaceHistoryRow[];
    total_active_archive_bytes: number;
    total_active_import_seconds: number;
  };
  env_caches: {
    oversized_threshold_bytes: number;
    current_schema: number;
    total_bytes: number;
    items: WorkspaceEnvCacheItem[];
  };
  orphan_objects: {
    keep_history: number;
    total_objects_count: number;
    total_objects_bytes: number;
    protected_count: number;
    protected_bytes: number;
    orphan_count: number;
    orphan_bytes: number;
    items: { id: string; sha256: string; name: string; size_bytes: number; modified_at?: number | null }[];
  };
};

export function formatWorkspaceBytes(bytes?: number | null): string {
  const n = typeof bytes === 'number' && Number.isFinite(bytes) && bytes >= 0 ? bytes : 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function computeNewlyOrphanedPlugins(
  selectedWorkflows: WorkspaceWorkflowRef[],
  activePlugins: WorkspacePluginRow[],
): WorkspacePluginRow[] {
  const removedKeys = new Set(selectedWorkflows.map(item => `${item.source}:${item.path}`));
  return activePlugins.filter(plugin => {
    if (!plugin.used_by_active.length) return false;
    const remaining = plugin.used_by_active.filter(ref => !removedKeys.has(`${ref.source}:${ref.path}`));
    return remaining.length === 0;
  });
}

export function defaultLinkedPluginSelection(orphaned: WorkspacePluginRow[]): Record<string, boolean> {
  const selection: Record<string, boolean> = {};
  for (const p of orphaned) {
    selection[p.path] = p.js_count === 0;
  }
  return selection;
}
