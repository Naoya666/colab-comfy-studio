import type { CuDailyEntry } from './gpu-api';

export interface CuChartPoint {
  date: string;
  actual_cu: number;
  sessions: number;
  hasRecord: boolean;
}

const ISO_DAY_RE = /^\d{4}-\d{2}-\d{2}$/;

export function shiftCalendarDate(isoDate: string, deltaDays: number): string {
  const [y, m, d] = isoDate.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d + deltaDays));
  const yyyy = dt.getUTCFullYear();
  const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(dt.getUTCDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

export function diffCalendarDays(startIso: string, endIso: string): number {
  const [sy, sm, sd] = startIso.split('-').map(Number);
  const [ey, em, ed] = endIso.split('-').map(Number);
  const startMs = Date.UTC(sy, sm - 1, sd);
  const endMs = Date.UTC(ey, em - 1, ed);
  return Math.round((endMs - startMs) / 86_400_000);
}

/**
 * Build an ascending, zero-filled daily CU series suitable for an SVG bar chart.
 * Clamps to at most `maxPoints` trailing days so bars remain legible on mobile.
 */
export function buildDailySeries(
  entries: CuDailyEntry[],
  {
    today,
    startDate,
    endDate,
    maxPoints = 45,
  }: {
    today: string;
    startDate?: string;
    endDate?: string;
    maxPoints?: number;
  },
): { points: CuChartPoint[]; clipped: boolean; rangeStart: string; rangeEnd: string } {
  const byDate = new Map<string, CuDailyEntry>();
  for (const row of entries) {
    if (ISO_DAY_RE.test(row.date)) byDate.set(row.date, row);
  }
  const sortedDates = [...byDate.keys()].sort();

  let rangeEnd = endDate && ISO_DAY_RE.test(endDate)
    ? endDate
    : (sortedDates.length ? (sortedDates[sortedDates.length - 1] > today ? sortedDates[sortedDates.length - 1] : today) : today);
  let rangeStart = startDate && ISO_DAY_RE.test(startDate)
    ? startDate
    : (sortedDates.length ? sortedDates[0] : shiftCalendarDate(rangeEnd, -6));

  if (rangeStart > rangeEnd) {
    return { points: [], clipped: false, rangeStart, rangeEnd };
  }

  // Ensure unfiltered "all" view has at least a 7-day timeline for visual balance.
  if (!startDate && !endDate && diffCalendarDays(rangeStart, rangeEnd) < 6) {
    rangeStart = shiftCalendarDate(rangeEnd, -6);
  }

  let clipped = false;
  const span = diffCalendarDays(rangeStart, rangeEnd) + 1;
  if (span > maxPoints) {
    rangeStart = shiftCalendarDate(rangeEnd, -(maxPoints - 1));
    clipped = true;
  }

  const totalDays = diffCalendarDays(rangeStart, rangeEnd) + 1;
  const points: CuChartPoint[] = [];
  for (let i = 0; i < totalDays; i++) {
    const date = shiftCalendarDate(rangeStart, i);
    const hit = byDate.get(date);
    points.push({
      date,
      actual_cu: hit ? Math.max(0, Number(hit.actual_cu) || 0) : 0,
      sessions: hit ? Math.max(0, Number(hit.sessions) || 0) : 0,
      hasRecord: Boolean(hit),
    });
  }
  return { points, clipped, rangeStart, rangeEnd };
}

export function niceMaxValue(rawMax: number): number {
  if (!Number.isFinite(rawMax) || rawMax <= 0) return 1;
  const exp = Math.floor(Math.log10(rawMax));
  const base = 10 ** exp;
  const frac = rawMax / base;
  const nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  return nice * base;
}
