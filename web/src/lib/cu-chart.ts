import type { CuDailyEntry } from './gpu-api';

export interface CuChartPoint {
  date: string;
  actual_cu: number;
  sessions: number;
  hasRecord: boolean;
}

export interface ChartCoord {
  x: number;
  y: number;
}

export interface YAxisScale {
  yMax: number;
  ticks: number[];
  hasLimit: boolean;
  autoLimitVisible: boolean;
  showLimitInPlot: boolean;
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
 * Build an ascending, zero-filled daily CU series suitable for an SVG trend chart.
 * Clamps to at most `maxPoints` trailing days so labels and points remain legible.
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

function niceStep(roughStep: number): number {
  if (!Number.isFinite(roughStep) || roughStep <= 0) return 0.5;
  const exp = Math.floor(Math.log10(roughStep));
  const base = 10 ** exp;
  const frac = roughStep / base;
  const mult = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 2.5 ? 2.5 : frac <= 5 ? 5 : 10;
  return mult * base;
}

/**
 * Compute crisp Y-axis bounds and tick marks.
 * Only folds `dailyLimitCu` into the axis scale when either the user toggles it on
 * or actual peak usage reaches >= 50% of the daily limit (so low-usage curves aren't flattened).
 */
export function computeYAxisScale(
  maxData: number,
  dailyLimitCu?: number | null,
  showLimitOverride?: boolean | null,
): YAxisScale {
  const safeMax = Number.isFinite(maxData) && maxData > 0 ? maxData : 0;
  const hasLimit = typeof dailyLimitCu === 'number' && Number.isFinite(dailyLimitCu) && dailyLimitCu > 0;
  const autoLimitVisible = hasLimit && safeMax >= dailyLimitCu! * 0.5;
  const showLimitInPlot = hasLimit && (showLimitOverride ?? autoLimitVisible);

  const rawTarget = Math.max(
    safeMax > 0 ? safeMax * 1.12 : 0,
    showLimitInPlot ? dailyLimitCu! * 1.1 : 0,
  );

  if (rawTarget <= 0) {
    return { yMax: 1, ticks: [0, 0.5, 1], hasLimit, autoLimitVisible, showLimitInPlot };
  }

  const step = niceStep(rawTarget / 4);
  const intervals = Math.max(2, Math.ceil(rawTarget / step));
  const yMax = Number((step * intervals).toPrecision(10));
  const ticks: number[] = [];
  for (let i = 0; i <= intervals; i++) {
    ticks.push(Number((step * i).toPrecision(10)));
  }
  return { yMax, ticks, hasLimit, autoLimitVisible, showLimitInPlot };
}

function fmt(n: number): string {
  return Number(n.toFixed(2)).toString();
}

/**
 * Build a taut Fritsch-Carlson monotone cubic spline SVG path + matching closed area path.
 * Strictly monotone between consecutive points: never undershoots below `baselineY` (0 CU)
 * and never overshoots past local peaks.
 */
export function buildSmoothLinePath(
  coords: ChartCoord[],
  baselineY: number,
): { linePath: string; areaPath: string } {
  const n = coords.length;
  if (n < 2) {
    return { linePath: '', areaPath: '' };
  }

  const dx: number[] = new Array(n - 1);
  const secant: number[] = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    dx[i] = Math.max(1e-6, coords[i + 1].x - coords[i].x);
    secant[i] = (coords[i + 1].y - coords[i].y) / dx[i];
  }

  const tangent: number[] = new Array(n).fill(0);
  tangent[0] = secant[0];
  tangent[n - 1] = secant[n - 2];

  for (let i = 1; i < n - 1; i++) {
    const prev = secant[i - 1];
    const next = secant[i];
    if (prev * next <= 0) {
      tangent[i] = 0;
    } else {
      const w1 = 2 * dx[i] + dx[i - 1];
      const w2 = dx[i] + 2 * dx[i - 1];
      tangent[i] = (w1 + w2) / (w1 / prev + w2 / next);
    }
  }

  // Fritsch-Carlson monotonicity limiter.
  for (let i = 0; i < n - 1; i++) {
    const d = secant[i];
    if (Math.abs(d) < 1e-9) {
      tangent[i] = 0;
      tangent[i + 1] = 0;
      continue;
    }
    let a = tangent[i] / d;
    let b = tangent[i + 1] / d;
    if (a < 0) {
      tangent[i] = 0;
      a = 0;
    }
    if (b < 0) {
      tangent[i + 1] = 0;
      b = 0;
    }
    const magSq = a * a + b * b;
    if (magSq > 9) {
      const tau = 3 / Math.sqrt(magSq);
      tangent[i] = tau * a * d;
      tangent[i + 1] = tau * b * d;
    }
  }

  // Mild tautness factor for a crisp, slightly-curved ("微弧") analytical line.
  const TAUT = 0.84;
  const segments: string[] = [`M ${fmt(coords[0].x)} ${fmt(coords[0].y)}`];
  for (let i = 0; i < n - 1; i++) {
    const p0 = coords[i];
    const p1 = coords[i + 1];
    const h = dx[i];
    const minY = Math.min(p0.y, p1.y);
    const maxY = Math.min(baselineY, Math.max(p0.y, p1.y));
    const clampY = (val: number) => Math.max(minY, Math.min(maxY, val));

    const cp1x = p0.x + (h / 3) * TAUT;
    const cp1y = clampY(p0.y + (tangent[i] * h / 3) * TAUT);
    const cp2x = p1.x - (h / 3) * TAUT;
    const cp2y = clampY(p1.y - (tangent[i + 1] * h / 3) * TAUT);

    segments.push(
      `C ${fmt(cp1x)} ${fmt(cp1y)}, ${fmt(cp2x)} ${fmt(cp2y)}, ${fmt(p1.x)} ${fmt(p1.y)}`,
    );
  }

  const linePath = segments.join(' ');
  const first = coords[0];
  const last = coords[n - 1];
  const areaPath = `${linePath} L ${fmt(last.x)} ${fmt(baselineY)} L ${fmt(first.x)} ${fmt(baselineY)} Z`;
  return { linePath, areaPath };
}
