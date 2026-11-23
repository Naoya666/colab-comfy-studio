import type { TimingDetail } from './job-history';
import { jobStageLabel, jobStageRows } from './job-stage-rows';

export type StageStatus = 'completed' | 'in_progress' | 'failed';
export type StageHeatLevel = 'fast' | 'normal' | 'slow' | 'hot';

export interface StageTimelineItem {
  key: string;
  label: string;
  seconds: number | null;
  status: StageStatus;
  source?: string;
}

export interface StageTimelineSegment extends StageTimelineItem {
  displaySeconds: number;
  percent: number;
  heatRatio: number;
  heatLevel: StageHeatLevel;
  color: string;
}

export interface NativeRestartInfo {
  restart_id: string;
  instance_id: string;
  phase: 'checking' | 'syncing' | 'rebooting' | 'recovering' | 'succeeded' | 'failed';
  failed_phase?: 'checking' | 'syncing' | 'rebooting' | 'recovering' | 'succeeded' | 'failed';
  error?: string;
  timing_details?: Record<string, TimingDetail>;
}

const HEAT_MAX_SECONDS = 65;
const NATIVE_FINALIZE_NOISE_SECONDS = 1.5;
const INTERNAL_FINALIZE_KEYS = new Set(['upload', 'cleanup', 'delete']);
const RESTART_ORDER = ['restart_checking', 'restart_syncing', 'restart_rebooting', 'restart_recovering'] as const;

export function stageHeatRatio(seconds: number | null | undefined): number {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return 0;
  return Math.min(1, Math.max(0, Math.log1p(seconds) / Math.log1p(HEAT_MAX_SECONDS)));
}

export function stageHeatLevel(seconds: number | null | undefined): StageHeatLevel {
  const s = seconds != null && Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  if (s < 5) return 'fast';
  if (s < 15) return 'normal';
  if (s < 35) return 'slow';
  return 'hot';
}

export function stageHeatColor(seconds: number | null | undefined, status: StageStatus = 'completed'): string {
  const ratio = stageHeatRatio(seconds);
  if (status === 'failed' && ratio < 0.8) {
    return 'var(--danger)';
  }
  // Smooth thermal ramp from cool emerald (152deg) -> amber -> orange -> crimson (2deg).
  const hue = Math.round(152 - ratio * 150);
  // Perceptual lightness compensation around yellow-green/amber (45-85deg) so light/dark modes stay balanced.
  const yellowPenalty = hue >= 36 && hue <= 88 ? Math.round(Math.sin(((hue - 36) / 52) * Math.PI) * 7) : 0;
  const satAdjust = ratio > 0.7 ? 6 : 0;
  const lightExpr = yellowPenalty > 0
    ? `calc(var(--stage-heat-l, 48%) - ${yellowPenalty}%)`
    : 'var(--stage-heat-l, 48%)';
  const satExpr = satAdjust > 0
    ? `calc(var(--stage-heat-s, 72%) + ${satAdjust}%)`
    : 'var(--stage-heat-s, 72%)';
  return `hsl(${hue} ${satExpr} ${lightExpr})`;
}

function roundTenths(value: number): number {
  return Math.round(Math.max(0, value) * 10) / 10;
}

function liveSeconds(
  baseSeconds: number | null | undefined,
  status: StageStatus,
  elapsedDeltaSeconds: number,
  fallbackElapsedSeconds?: number,
): number | null {
  const validBase = typeof baseSeconds === 'number' && Number.isFinite(baseSeconds) && baseSeconds >= 0 ? baseSeconds : null;
  if (status !== 'in_progress') return validBase;
  const delta = Number.isFinite(elapsedDeltaSeconds) && elapsedDeltaSeconds > 0 ? elapsedDeltaSeconds : 0;
  if (validBase !== null) return roundTenths(validBase + delta);
  if (typeof fallbackElapsedSeconds === 'number' && Number.isFinite(fallbackElapsedSeconds) && fallbackElapsedSeconds >= 0) {
    return roundTenths(fallbackElapsedSeconds);
  }
  return null;
}

function sortByCanonicalOrder(keys: string[]): string[] {
  const preferred = jobStageRows(Object.fromEntries(keys.map((key) => [key, 0]))).map((row) => row.key);
  const seen = new Set(preferred);
  return [...preferred, ...keys.filter((key) => !seen.has(key))];
}

export function buildNativeStartupStages(params: {
  state?: string | null;
  stage?: string | null;
  timings?: Record<string, number> | null;
  timingDetails?: Record<string, TimingDetail> | null;
  elapsedDeltaSeconds?: number;
  fallbackActiveSeconds?: number;
}): StageTimelineItem[] {
  const {
    state,
    stage,
    timings,
    timingDetails,
    elapsedDeltaSeconds = 0,
    fallbackActiveSeconds,
  } = params;

  const raw = new Map<string, { seconds: number | null; status: StageStatus; source?: string }>();
  for (const [key, seconds] of Object.entries(timings || {})) {
    if (typeof seconds === 'number' && Number.isFinite(seconds) && seconds >= 0) {
      raw.set(key, { seconds: roundTenths(seconds), status: 'completed', source: 'worker' });
    }
  }
  for (const [key, detail] of Object.entries(timingDetails || {})) {
    if (!detail || typeof detail !== 'object') continue;
    const status: StageStatus = detail.status === 'in_progress' ? 'in_progress' : 'completed';
    const validSeconds = typeof detail.seconds === 'number' && Number.isFinite(detail.seconds) && detail.seconds >= 0
      ? detail.seconds
      : null;
    if (status !== 'in_progress' && validSeconds === null) continue;
    raw.set(key, {
      seconds: liveSeconds(validSeconds, status, elapsedDeltaSeconds, fallbackActiveSeconds),
      status,
      source: detail.source,
    });
  }

  // Denoise native internal tail stages (upload of native-ready.json & cleanup/retain mark):
  // hide sub-1.5s framework bookkeeping, or fold slow/active tail into one friendly stage.
  let finalizeSeconds = 0;
  let finalizeHasNumeric = false;
  let finalizeActive = false;
  for (const key of INTERNAL_FINALIZE_KEYS) {
    const entry = raw.get(key);
    if (!entry) continue;
    if (entry.status === 'in_progress') finalizeActive = true;
    if (entry.seconds !== null) {
      finalizeSeconds += entry.seconds;
      finalizeHasNumeric = true;
    }
    raw.delete(key);
  }
  if (finalizeActive || (finalizeHasNumeric && finalizeSeconds >= NATIVE_FINALIZE_NOISE_SECONDS)) {
    raw.set('native_finalize', {
      seconds: finalizeHasNumeric ? roundTenths(finalizeSeconds) : null,
      status: finalizeActive ? 'in_progress' : 'completed',
      source: 'events',
    });
  }

  const normalizedState = String(state || '').toLowerCase();
  const normalizedStage = String(stage || '').trim();
  if (normalizedState === 'failed') {
    const orderedForFailure = sortByCanonicalOrder([...raw.keys()]);
    const failedKey = normalizedStage && !INTERNAL_FINALIZE_KEYS.has(normalizedStage)
      ? normalizedStage
      : orderedForFailure[orderedForFailure.length - 1];
    if (failedKey) {
      const existing = raw.get(failedKey);
      raw.set(failedKey, {
        seconds: existing?.seconds ?? null,
        status: 'failed',
        source: existing?.source,
      });
    }
  }

  const orderedKeys = sortByCanonicalOrder([...raw.keys()]);
  return orderedKeys.flatMap((key) => {
    const entry = raw.get(key);
    if (!entry) return [];
    return [{
      key,
      label: jobStageLabel(key),
      seconds: entry.seconds,
      status: entry.status,
      source: entry.source,
    }];
  });
}

export function buildNativeRestartStages(params: {
  restart?: NativeRestartInfo | null;
  elapsedDeltaSeconds?: number;
  fallbackActiveSeconds?: number;
}): StageTimelineItem[] {
  const { restart, elapsedDeltaSeconds = 0, fallbackActiveSeconds } = params;
  if (!restart) return [];
  const details = restart.timing_details || {};
  const failedKey = restart.phase === 'failed' && restart.failed_phase ? `restart_${restart.failed_phase}` : '';
  const items: StageTimelineItem[] = [];

  for (const key of RESTART_ORDER) {
    const detail = details[key];
    const isFailed = key === failedKey;
    if (!detail && !isFailed) continue;
    let status: StageStatus = detail?.status === 'in_progress' ? 'in_progress' : 'completed';
    if (isFailed) status = 'failed';
    const seconds = liveSeconds(detail?.seconds, status, elapsedDeltaSeconds, fallbackActiveSeconds);
    items.push({
      key,
      label: jobStageLabel(key),
      seconds,
      status,
      source: detail?.source || 'events',
    });
  }
  return items;
}

export function computeTimelineSegments(items: StageTimelineItem[]): {
  segments: StageTimelineSegment[];
  totalSeconds: number;
  hasInProgress: boolean;
} {
  const hasInProgress = items.some((item) => item.status === 'in_progress');
  const totalSeconds = roundTenths(
    items.reduce((sum, item) => sum + (item.seconds !== null && item.seconds > 0 ? item.seconds : 0), 0),
  );
  const segments: StageTimelineSegment[] = items.map((item) => {
    const displaySeconds = item.seconds !== null && item.seconds >= 0 ? item.seconds : 0;
    const percent = totalSeconds > 0
      ? (displaySeconds / totalSeconds) * 100
      : (items.length > 0 ? 100 / items.length : 0);
    return {
      ...item,
      displaySeconds,
      percent,
      heatRatio: stageHeatRatio(item.seconds),
      heatLevel: stageHeatLevel(item.seconds),
      color: stageHeatColor(item.seconds, item.status),
    };
  });
  return { segments, totalSeconds, hasInProgress };
}
