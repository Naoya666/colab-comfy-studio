import type { GalleryAsset, GpuJob } from './gpu-api';
import { terminalStates } from './ui-helpers';

const DONE_STATES = new Set(['completed', 'succeeded']);
const FAILED_STATES = new Set(['failed', 'failed_final', 'cancelled', 'cancelled_local', 'blocked']);

export type DateBucketKey = 'today' | 'yesterday' | 'older';

export interface JobBatchEntry {
  kind: 'batch';
  key: string;
  batchId: string;
  jobs: GpuJob[];
  latestCreatedAt: string;
  activeCount: number;
  doneCount: number;
  failedCount: number;
  workflowKey: string;
  provider: string;
  allTerminal: boolean;
}

export interface JobSingleEntry {
  kind: 'single';
  key: string;
  job: GpuJob;
  latestCreatedAt: string;
  allTerminal: boolean;
}

export type JobDisplayEntry = JobBatchEntry | JobSingleEntry;

export interface JobDateBucket {
  key: DateBucketKey;
  label: string;
  entries: JobDisplayEntry[];
}

export interface GalleryStack {
  key: string;
  batchId?: string;
  workflowKey?: string;
  cover: GalleryAsset;
  items: GalleryAsset[];
  createdAt: string;
}

function toTimestamp(value?: string | number | null): number {
  if (value === null || value === undefined || value === '') return 0;
  if (typeof value === 'number') return value < 1e12 ? value * 1000 : value;
  const num = Number(value);
  if (Number.isFinite(num) && String(value).trim() !== '') return num < 1e12 ? num * 1000 : num;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function jobDateBucket(value?: string | number | null, nowMs: number = Date.now()): DateBucketKey {
  const ts = toTimestamp(value);
  if (!ts) return 'older';
  const now = new Date(nowMs);
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfYesterday = startOfToday - 24 * 60 * 60 * 1000;
  if (ts >= startOfToday) return 'today';
  if (ts >= startOfYesterday) return 'yesterday';
  return 'older';
}

const BUCKET_LABELS: Record<DateBucketKey, string> = {
  today: '今天',
  yesterday: '昨天',
  older: '更早',
};

/**
 * Group jobs sharing the same batch_id (>= 2 items) into collapsible batch entries,
 * keeping standalone jobs (or single-item batches) as regular single cards.
 */
export function groupJobsByBatch(jobs: GpuJob[]): JobDisplayEntry[] {
  const byBatch = new Map<string, GpuJob[]>();
  for (const job of jobs) {
    if (job.batch_id) {
      const list = byBatch.get(job.batch_id) || [];
      list.push(job);
      byBatch.set(job.batch_id, list);
    }
  }

  const emittedBatches = new Set<string>();
  const entries: JobDisplayEntry[] = [];

  for (let index = 0; index < jobs.length; index += 1) {
    const job = jobs[index];
    const batchId = job.batch_id;
    const grouped = batchId ? byBatch.get(batchId) : undefined;
    if (batchId && grouped && grouped.length >= 2) {
      if (emittedBatches.has(batchId)) continue;
      emittedBatches.add(batchId);
      const sorted = [...grouped].sort((a, b) => toTimestamp(b.created_at) - toTimestamp(a.created_at));
      let activeCount = 0;
      let doneCount = 0;
      let failedCount = 0;
      for (const item of sorted) {
        const s = String(item.state || 'queued').toLowerCase();
        if (!terminalStates.has(s)) activeCount += 1;
        else if (DONE_STATES.has(s)) doneCount += 1;
        else if (FAILED_STATES.has(s)) failedCount += 1;
      }
      entries.push({
        kind: 'batch',
        key: `batch:${batchId}`,
        batchId,
        jobs: sorted,
        latestCreatedAt: sorted[0]?.created_at || '',
        activeCount,
        doneCount,
        failedCount,
        workflowKey: sorted[0]?.workflow_key || '批量任务',
        provider: String(sorted[0]?.provider || ''),
        allTerminal: activeCount === 0,
      });
    } else {
      const s = String(job.state || 'queued').toLowerCase();
      entries.push({
        kind: 'single',
        key: String(job.id || job.job_id || `job-${index}`),
        job,
        latestCreatedAt: job.created_at || '',
        allTerminal: terminalStates.has(s),
      });
    }
  }

  return entries;
}

/**
 * Pin all active entries and group up to historyLimit history entries by date.
 */
export function partitionJobEntries(
  entries: JobDisplayEntry[],
  nowMs: number = Date.now(),
  historyLimit: number = Infinity,
): { active: JobDisplayEntry[]; buckets: JobDateBucket[] } {
  const active: JobDisplayEntry[] = [];
  const byBucket: Record<DateBucketKey, JobDisplayEntry[]> = {
    today: [],
    yesterday: [],
    older: [],
  };

  let historyCount = 0;
  for (const entry of entries) {
    if (!entry.allTerminal) {
      active.push(entry);
    } else if (historyCount < historyLimit) {
      byBucket[jobDateBucket(entry.latestCreatedAt, nowMs)].push(entry);
      historyCount += 1;
    }
  }

  const buckets: JobDateBucket[] = (['today', 'yesterday', 'older'] as const)
    .filter((key) => byBucket[key].length > 0)
    .map((key) => ({ key, label: BUCKET_LABELS[key], entries: byBucket[key] }));

  return { active, buckets };
}

/**
 * Stack gallery assets that belong to the same generation batch (batch_id) or the same multi-output job.
 */
export function groupGalleryAssets(assets: GalleryAsset[]): GalleryStack[] {
  const byGroup = new Map<string, GalleryAsset[]>();
  const keyOf = (asset: GalleryAsset) => {
    if (asset.batch_id) return `batch:${asset.batch_id}`;
    const jobId = asset.source_job_id || asset.gpu_job_id;
    if (jobId) return `job:${jobId}`;
    return `asset:${asset.id}`;
  };

  for (const asset of assets) {
    const key = keyOf(asset);
    const list = byGroup.get(key) || [];
    list.push(asset);
    byGroup.set(key, list);
  }

  const emitted = new Set<string>();
  const stacks: GalleryStack[] = [];
  for (const asset of assets) {
    const key = keyOf(asset);
    if (emitted.has(key)) continue;
    emitted.add(key);
    const items = byGroup.get(key) || [asset];
    const cover = items.find((item) => item.favorite) || items[0];
    stacks.push({
      key,
      batchId: asset.batch_id,
      workflowKey: asset.workflow_key,
      cover,
      items,
      createdAt: asset.created_at || '',
    });
  }
  return stacks;
}

export function isDiagnosticAsset(asset: GalleryAsset): boolean {
  return asset.mime.includes('json') || Boolean(asset.filename?.toLowerCase().endsWith('.json'));
}
