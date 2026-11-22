import { useMemo, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import type { CuDailyEntry } from '../lib/gpu-api';
import { buildDailySeries, niceMaxValue } from '../lib/cu-chart';
import { cuNumber } from '../lib/ui-helpers';
import styles from '../gpu.module.css';

const CHART_W = 640;
const CHART_H = 184;
const PAD_L = 44;
const PAD_R = 14;
const PAD_T = 16;
const PAD_B = 28;
const PLOT_W = CHART_W - PAD_L - PAD_R;
const PLOT_H = CHART_H - PAD_T - PAD_B;

function shortDayLabel(iso: string): string {
  return iso.slice(5);
}

export function CuDailyChart({
  days,
  today,
  startDate,
  endDate,
  dailyLimitCu,
  loading,
  onSelectDate,
}: {
  days: CuDailyEntry[];
  today: string;
  startDate: string;
  endDate: string;
  dailyLimitCu?: number | null;
  loading?: boolean;
  onSelectDate?: (date: string) => void;
}) {
  const { points, clipped } = useMemo(
    () => buildDailySeries(days, { today, startDate, endDate, maxPoints: 45 }),
    [days, today, startDate, endDate],
  );

  const [focusedDate, setFocusedDate] = useState<string | null>(null);

  const maxData = useMemo(
    () => points.reduce((acc, p) => Math.max(acc, p.actual_cu), 0),
    [points],
  );
  const hasLimit = typeof dailyLimitCu === 'number' && Number.isFinite(dailyLimitCu) && dailyLimitCu > 0;
  const yMax = niceMaxValue(Math.max(maxData, hasLimit ? dailyLimitCu! * 1.05 : 0, 0.5));

  const activePoint = useMemo(() => {
    if (!points.length) return null;
    if (focusedDate) {
      const hit = points.find((p) => p.date === focusedDate);
      if (hit) return hit;
    }
    const latestWithSpend = [...points].reverse().find((p) => p.actual_cu > 0 || p.sessions > 0);
    return latestWithSpend || points[points.length - 1];
  }, [points, focusedDate]);

  if (!points.length) return null;

  const slotW = PLOT_W / points.length;
  const barW = Math.max(4, Math.min(28, slotW * 0.64));
  const labelEvery = Math.max(1, Math.ceil(points.length / 7));
  const limitY = hasLimit ? PAD_T + PLOT_H * (1 - Math.min(dailyLimitCu! / yMax, 1)) : null;

  return (
    <div className={styles.chartCard} aria-label="每日 CU 消耗趋势图">
      <div className={styles.chartHead}>
        <div className={styles.chartReadout} aria-live="polite">
          {activePoint ? (
            <>
              <strong>{activePoint.date}</strong>
              <span>扣费 <b>{cuNumber(activePoint.actual_cu)}</b> CU</span>
              <span className="muted">· 开机 {activePoint.sessions} 场</span>
            </>
          ) : (
            <span className="muted small">暂无每日消耗数据</span>
          )}
        </div>
        <div className={styles.chartHeadActions}>
          {loading && <span className="muted small">更新中…</span>}
          {activePoint && onSelectDate && activePoint.sessions > 0 && (
            <button
              type="button"
              className="button ghost small"
              onClick={() => onSelectDate(activePoint.date)}
            >
              查看当日会话<ArrowUpRight size={13} aria-hidden />
            </button>
          )}
        </div>
      </div>

      <svg
        className={styles.chartSvg}
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        role="img"
        aria-label={`每日 CU 消耗柱状图，共 ${points.length} 天`}
      >
        {[0, 0.5, 1].map((ratio) => {
          const y = PAD_T + PLOT_H * (1 - ratio);
          const tick = yMax * ratio;
          return (
            <g key={ratio}>
              <line
                x1={PAD_L}
                x2={CHART_W - PAD_R}
                y1={y}
                y2={y}
                className={styles.chartGridLine}
              />
              <text
                x={PAD_L - 8}
                y={y + 3.5}
                textAnchor="end"
                className={styles.chartAxisText}
              >
                {tick === 0 ? '0' : tick >= 10 ? tick.toFixed(0) : tick.toFixed(1)}
              </text>
            </g>
          );
        })}

        {limitY !== null && (
          <g>
            <line
              x1={PAD_L}
              x2={CHART_W - PAD_R}
              y1={limitY}
              y2={limitY}
              className={styles.chartLimitLine}
            />
            <text
              x={CHART_W - PAD_R}
              y={Math.max(11, limitY - 4)}
              textAnchor="end"
              className={styles.chartLimitText}
            >
              日上限 {cuNumber(dailyLimitCu)} CU
            </text>
          </g>
        )}

        {points.map((point, index) => {
          const cx = PAD_L + slotW * index + slotW / 2;
          const x = cx - barW / 2;
          const ratio = point.actual_cu > 0 ? Math.min(1, point.actual_cu / yMax) : 0;
          const barH = point.actual_cu > 0 ? Math.max(3, ratio * PLOT_H) : 1.5;
          const y = PAD_T + PLOT_H - barH;
          const isFocused = activePoint?.date === point.date;
          const overLimit = hasLimit && point.actual_cu >= dailyLimitCu!;
          const showLabel = index === 0 || index === points.length - 1 || index % labelEvery === 0;
          return (
            <g
              key={point.date}
              className={styles.chartBarGroup}
              onMouseEnter={() => setFocusedDate(point.date)}
              onClick={() => setFocusedDate(point.date)}
            >
              <rect
                x={PAD_L + slotW * index}
                y={PAD_T}
                width={slotW}
                height={PLOT_H}
                fill="transparent"
              />
              <rect
                x={x}
                y={y}
                width={barW}
                height={barH}
                rx={Math.min(3, barW / 2)}
                className={styles.chartBar}
                data-empty={point.actual_cu <= 0}
                data-active={isFocused}
                data-over={overLimit}
              >
                <title>{`${point.date}：${cuNumber(point.actual_cu)} CU（${point.sessions} 场）`}</title>
              </rect>
              {showLabel && (
                <text
                  x={cx}
                  y={CHART_H - 8}
                  textAnchor="middle"
                  className={styles.chartAxisText}
                  data-active={isFocused}
                >
                  {shortDayLabel(point.date)}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {clipped && (
        <p className="muted small">趋势图最多展示区间内最近 45 天；可通过上方日期范围缩小查看。</p>
      )}
    </div>
  );
}
