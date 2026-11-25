export type V020Provider = 'mock' | 'colab';

export type GpuAllocation = 'auto' | 'existing';
export type GpuLifecycle = 'save' | 'batch' | 'smart' | 'retain';
export type RetentionMode = 'save' | 'retain' | 'smart';
export type GpuReuseAction = 'auto' | 'reuse' | 'replace' | 'new';
export type GpuInstancePool = 'native' | 'creation' | 'ephemeral' | string;

/** New execution policy fields are optional so legacy job payloads remain readable. */
export interface GpuExecutionPolicyV020 {
  provider: V020Provider;
  max_runtime_minutes: number;
  compute_spec?: Record<string, unknown> | null;
  allocation?: GpuAllocation;
  account_id?: string | null;
  instance_id?: string | null;
  reuse_action?: GpuReuseAction | null;
  replace_instance_id?: string | null;
  lifecycle?: GpuLifecycle;
  retain_idle_minutes?: number | null;
  /** Legacy fields: when present, the server validates them against lifecycle. */
  shutdown_after_job?: boolean;
  idle_grace_minutes?: number;
}

export interface GpuJobRequestV020 {
  project_id: string;
  model_id: string;
  workflow_key: string;
  workflow_version: string | number;
  params: Record<string, unknown>;
  asset_ids: string[];
  preset_id?: string;
  prompt_composition?: {
    positive?: {
      param_key: string;
      main: string;
      fixed: string;
      enabled: boolean;
      position: 'prefix' | 'suffix';
      separator: 'comma' | 'space' | 'newline';
    } | null;
    negative?: {
      param_key: string;
      main: string;
      fixed: string;
      enabled: boolean;
      position: 'prefix' | 'suffix';
      separator: 'comma' | 'space' | 'newline';
    } | null;
  } | null;
  execution_policy: GpuExecutionPolicyV020;
}

export type GoogleAuthPurpose = 'colab' | 'drive';
export type GoogleAuthChannel = 'colab_cli' | 'web_oauth';
export type GoogleAccountState = 'connected' | 'needs_reauth' | 'disabled' | 'error' | string;

export interface GoogleAuthorization {
  connected: boolean;
  configured?: boolean;
  state?: GoogleAccountState;
  scopes?: string[];
  expires_at?: string | number | null;
  last_checked_at?: string | null;
  error_code?: string | null;
  auth_channel?: GoogleAuthChannel | null;
  cli_available?: boolean;
  message?: string;
  delegated_to?: string;
  delegated_alias?: string;
}

export interface ColabCliAuthStartResponse {
  authorization_url: string;
  state: string;
  expires_at: number;
  expires_in: number;
  auth_channel: 'colab_cli';
  scopes?: string[];
}

export interface ColabCliAuthCompleteResponse {
  account_id: string;
  purpose: 'colab';
  auth_channel: 'colab_cli';
  connected: boolean;
  balance?: number | null;
  eligible_gpus?: string[];
  probe_error?: string | null;
}

export interface ColabCliImportRequest {
  credentials_json?: Record<string, unknown> | string;
  refresh_token?: string;
  client_id?: string;
  client_secret?: string;
}

export interface GoogleAccount {
  account_id: string;
  alias: string;
  state: GoogleAccountState;
  enabled: boolean;
  email_masked?: string | null;
  drive_account_id?: string | null;
  drive_account_alias?: string | null;
  drive_dependents?: string[];
  colab: GoogleAuthorization;
  drive: GoogleAuthorization;
  allowed_project_ids?: string[];
  scheduling_policy?: string | null;
  budget?: Record<string, number | null>;
  active_instance_count?: number;
  last_checked_at?: string | null;
  last_error_code?: string | null;
  can_allocate?: boolean;
  /** Compatibility for older account presenters that used id. */
  id?: string;
}

export interface GoogleAccountCreateRequest {
  alias: string;
  enabled?: boolean;
  drive_account_id?: string | null;
  allowed_project_ids?: string[];
  scheduling_policy?: string;
  budget?: Record<string, number | null>;
}

export interface GoogleAccountUpdateRequest {
  alias?: string;
  enabled?: boolean;
  drive_account_id?: string | null;
  allowed_project_ids?: string[];
  scheduling_policy?: string;
  budget?: Record<string, number | null>;
}

export interface AccountRuntime {
  runtime_id: string;
  instance_id?: string | null;
  state?: string;
  provider?: V020Provider;
  gpu_reported?: string | null;
  created_at?: string | null;
  last_seen_at?: string | null;
  current_job_id?: string | null;
}

export type GpuInstanceState =
  | 'requested' | 'allocating' | 'ready' | 'busy' | 'retained_idle'
  | 'transferring' | 'releasing' | 'deleted' | 'unknown' | 'cleanup_pending' | 'quarantined' | string;

export interface InstanceTransferState {
  transfer_id?: string;
  from_pool?: GpuInstancePool;
  to_pool?: GpuInstancePool;
  phase?: string;
  transfer_job_id?: string;
  native_bootstrap_job_id?: string;
  started_at?: number | null;
  completed_at?: number | null;
  error_code?: string | null;
  error_message?: string | null;
}

export interface InstanceTransferHistoryItem {
  transfer_id: string;
  from_pool: GpuInstancePool;
  to_pool: GpuInstancePool;
  started_at?: number | null;
  completed_at?: number | null;
  transfer_job_id?: string;
  domain_generation?: number;
}

export interface TransferTargetPreview {
  allowed: boolean;
  reason?: string | null;
  defaults?: Record<string, unknown>;
}

export interface TransferPreviewResponse {
  instance_id: string;
  version: number;
  current_pool: GpuInstancePool;
  venv_isolated: boolean;
  native_bootstrap_failed?: boolean;
  vm_allocated_at?: number | null;
  vm_elapsed_seconds?: number | null;
  account_id?: string | null;
  account_alias?: string | null;
  spec?: Record<string, unknown> | null;
  gpu_reported?: string | null;
  transferable: boolean;
  blocked_code?: string | null;
  blocked_reason?: string | null;
  targets: Partial<Record<'creation' | 'native' | 'ephemeral', TransferTargetPreview>>;
}

export interface TransferInstanceRequest {
  target_pool: 'creation' | 'native' | 'ephemeral';
  confirmed: true;
  expected_version?: number;
  key?: string;
  creation_options?: {
    retain_idle_minutes?: number;
  };
  native_options?: {
    model_storage_mode?: 'local' | 'drive_mount';
    cache_gb?: number;
    cache_max_age_minutes?: number;
    retain_idle_minutes?: number;
    max_runtime_minutes?: number;
    plugin_generation?: string | null;
  };
}

export interface TransferInstanceResponse {
  instance: GpuInstance;
  transfer_job_id: string;
  transfer_id: string;
}

export interface GpuInstance {
  instance_id: string;
  label?: string | null;
  account_id?: string | null;
  account_alias_snapshot?: string | null;
  provider?: V020Provider;
  instance_pool?: GpuInstancePool;
  state: GpuInstanceState;
  retention_mode?: RetentionMode;
  retention?: RetentionMode;
  retain_idle_minutes?: number | null;
  version?: number;
  idle_minutes?: number | null;
  retain_requested_at?: string | null;
  idle_deadline_at?: string | number | null;
  hard_deadline_at?: string | number | null;
  manual_stop_requested?: boolean;
  current_job_id?: string | null;
  gpu_reported?: string | null;
  runtime_spec_requested?: Record<string, unknown> | null;
  runtime_spec_reported?: Record<string, unknown> | null;
  creation_runtime?: Record<string, unknown> | null;
  created_at?: string;
  last_seen_at?: string | null;
  retained_until?: string | null;
  estimated_units?: number | null;
  actual_units?: number | null;
  runtime_deleted?: boolean;
  last_error_code?: string | null;
  recovery_group_id?: string | null;
  recovery_generation?: number | null;
  parent_instance_id?: string | null;
  replaced_by_instance_id?: string | null;
  recovery_plan_id?: string | null;
  recovery_source?: string | null;
  venv_isolated?: boolean;
  vm_allocated_at?: number | null;
  domain_generation?: number;
  transfer_state?: InstanceTransferState | null;
  transfer_history?: InstanceTransferHistoryItem[];
  native_bootstrap_failed?: boolean;
  native_bootstrap_error?: string | null;
  ephemeral_standby?: boolean;
  quarantine_reason?: string | null;
}

export interface InstanceRetentionRequest {
  retention_mode: RetentionMode;
  idle_minutes: number;
  expected_version: number;
}

export interface GpuInstanceFilters {
  state?: GpuInstanceState;
  account_id?: string;
  provider?: V020Provider;
}

export type WorkbenchAction =
  | 'health'
  | 'list_allowed_workflows'
  | 'get_reviewed_graph'
  | 'set_reviewed_params'
  | 'queue_reviewed_workflow'
  | 'job_status'
  | 'read_safe_log'
  | 'preview_artifacts'
  | 'stop_current_job';

export interface Workbench {
  instance_id: string;
  state?: string;
  healthy?: boolean;
  service_state?: string;
  capabilities: WorkbenchAction[];
  allowed_workflows?: Array<Record<string, unknown>>;
  cache?: Record<string, unknown>;
  current_job_id?: string | null;
  last_checked_at?: string | null;
  message?: string | null;
}

export interface WorkbenchActionRequest {
  action: WorkbenchAction;
  params?: Record<string, unknown>;
}

export interface WorkbenchActionResult {
  action?: WorkbenchAction;
  state?: string;
  data?: unknown;
  message?: string;
  [key: string]: unknown;
}

/** Safe, user-visible job fields for v0.2.0; provider handles and tokens are never typed here. */
export interface GpuJobSafetyFields {
  account_id?: string | null;
  account_alias_snapshot?: string | null;
  instance_id?: string | null;
  instance_label?: string | null;
  instance_state?: GpuInstanceState | null;
  gpu_reported?: string | null;
  retained_until?: string | null;
  runtime_deleted?: boolean | null;
  stage?: string | null;
  actual_units?: number | null;
}
