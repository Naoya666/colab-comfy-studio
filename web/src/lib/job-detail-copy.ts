import type { GpuJob } from './gpu-api';
import type { PromptComposition } from './fixed-prompts';
import { friendlyLabel } from './labels';
import { formatDuration } from './job-history';

export function taskStatusNote(job: { state?: string; workflow_smoke?: boolean }): string {
  const done = ['completed', 'succeeded', 'success'].includes(String(job.state || '').toLowerCase());
  if (job.workflow_smoke) {
    return done
      ? '这是工作流启用前的冒烟任务，这次生成已经结束。任务完成不等于工作流已经发布。'
      : '这是工作流启用前的冒烟任务。下面的进度是这次冒烟，不是发布是否完成。';
  }
  return done
    ? '这是一次生成任务。状态「已完成」只表示这次生成结束，不表示工作流已经发布。'
    : '这是一次生成任务的进度。它不表示工作流是否已经发布。';
}

export function paramRows(params: Record<string, unknown> | null | undefined): Array<{ key: string; label: string; value: string }> {
  if (!params) return [];
  return Object.entries(params).flatMap(([key, value]) => {
    if (value === null || value === undefined || value === '') return [];
    const text = typeof value === 'object' ? JSON.stringify(value) : String(value);
    return [{ key, label: friendlyLabel(key, key), value: text }];
  });
}

export function promptCompositionRows(
  composition: PromptComposition | null | undefined,
): Array<{ key: string; label: string; value: string }> {
  if (!composition) return [];
  const rows: Array<{ key: string; label: string; value: string }> = [];
  const pos = composition.positive;
  if (pos && pos.enabled && pos.fixed.trim()) {
    rows.push({
      key: 'composition_fixed_positive',
      label: `固定正面提示词（${pos.position === 'suffix' ? '后置' : '前置'}）`,
      value: pos.fixed.trim(),
    });
    if (pos.main.trim()) {
      rows.push({
        key: 'composition_main_positive',
        label: '本次正面提示词',
        value: pos.main.trim(),
      });
    }
  }
  const neg = composition.negative;
  if (neg && neg.enabled && neg.fixed.trim()) {
    rows.push({
      key: 'composition_fixed_negative',
      label: `固定负面提示词（${neg.position === 'suffix' ? '后置' : '前置'}）`,
      value: neg.fixed.trim(),
    });
    if (neg.main.trim()) {
      rows.push({
        key: 'composition_main_negative',
        label: '本次负面提示词',
        value: neg.main.trim(),
      });
    }
  }
  return rows;
}

export function stageSecondsTotal(timings: Record<string, number> | null | undefined): string {
  if (!timings) return '';
  const values = Object.values(timings).filter((value) => typeof value === 'number' && Number.isFinite(value) && value >= 0);
  if (!values.length) return '';
  return formatDuration(values.reduce((sum, value) => sum + value, 0));
}

export function jobDetailTimings(job: GpuJob): Record<string, number> | undefined {
  const value = job.timings;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  return value as Record<string, number>;
}
