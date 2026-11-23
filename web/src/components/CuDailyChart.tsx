import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import type { CuDailyEntry } from '../lib/gpu-api';
import {
  buildDailySeries,
  buildSmoothLinePath,
  computeYAxisScale,
} from '../lib/cu-chart';
import { cuNumber } from '../lib/ui-helpers';
import styles from '../gpu.module.css';

function shortDayLabel(iso: string): string {
  return iso.slice(5);
}

function formatAxisTick(value: number): string {
  if (value === 0) return '0';
  if (value >= 100) return value.toFixed(0);
  if (Number.isInteger(value)) return String(value);
  return value >= 10 ? value.toFixed(1) : value.toFixed(2).replace(/0$/, '');
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

  const stageRef = useRef<HTMLDivElement | null>(null);
  const stageLeftRef = useRef<number | null>(null);
  const [stageWidth, setStageWidth] = useState(560);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [touchPinnedIdx, setTouchPinnedIdx] = useState<number | null>(null);
  const [limitOverride, setLimitOverride] = useState<boolean | null>(null);
  const gradId = useId();

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      stageLeftRef.current = rect.left;
      const w = Math.floor(rect.width);
      if (w >= 200) setStageWidth(w);
    };
    update();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Reset pinned index when the series changes length/bounds.
  useEffect(() => {
    setHoveredIdx(null);
    setTouchPinnedIdx(null);
  }, [startDate, endDate, points.length]);

  const summary = useMemo(() => {
    let totalCu = 0;
    let peakPoint = points[0] ?? null;
    for (const p of points) {
      totalCu += p.actual_cu;
      if (!peakPoint || p.actual_cu > peakPoint.actual_cu) {
        peakPoint = p;
      }
    }
    const avgCu = points.length > 0 ? totalCu / points.length : 0;
    const peakCu = peakPoint?.actual_cu ?? 0;
    return { totalCu, avgCu, peakCu, peakPoint };
  }, [points]);

  const scale = useMemo(
    () => computeYAxisScale(summary.peakCu, dailyLimitCu, limitOverride),
    [summary.peakCu, dailyLimitCu, limitOverride],
  );

  const isNarrow = stageWidth < 430;
  const chartW = stageWidth;
  const chartH = isNarrow ? 156 : 188;
  const padL = scale.yMax >= 100 ? 40 : 34;
  const padR = 14;
  const padT = 16;
  const padB = 26;
  const plotW = Math.max(120, chartW - padL - padR);
  const plotH = Math.max(60, chartH - padT - padB);
  const baselineY = padT + plotH;

  const { coords, linePath, areaPath } = useMemo(() => {
    const nextCoords = points.map((point, index) => {
      const x = points.length === 1
        ? padL + plotW / 2
        : padL + (plotW * index) / (points.length - 1);
      const ratio = scale.yMax > 0 ? Math.min(1, Math.max(0, point.actual_cu / scale.yMax)) : 0;
      const y = baselineY - ratio * plotH;
      return { x, y };
    });
    const paths = buildSmoothLinePath(nextCoords, baselineY);
    return { coords: nextCoords, linePath: paths.linePath, areaPath: paths.areaPath };
  }, [points, padL, plotW, plotH, baselineY, scale.yMax]);

  if (!points.length) return null;

  const activeIdx = hoveredIdx ?? touchPinnedIdx;
  const activePoint = activeIdx !== null ? points[activeIdx] ?? null : null;
  const activeCoord = activeIdx !== null ? coords[activeIdx] ?? null : null;

  const maxXLabels = Math.max(2, Math.floor(plotW / 64));
  const labelStep = points.length <= maxXLabels
    ? 1
    : Math.max(1, Math.ceil((points.length - 1) / (maxXLabels - 1)));

  const limitY = scale.showLimitInPlot && dailyLimitCu != null
    ? baselineY - Math.min(1, dailyLimitCu / scale.yMax) * plotH
    : null;

  const peakLimitPct = scale.hasLimit && dailyLimitCu
    ? (summary.peakCu / dailyLimitCu) * 100
    : null;

  function refreshStageLeft() {
    const el = stageRef.current;
    if (el) stageLeftRef.current = el.getBoundingClientRect().left;
  }

  function resolveIndexFromClientX(clientX: number, refresh = false): number {
    const el = stageRef.current;
    if (!el || points.length <= 1) return 0;
    if (refresh || stageLeftRef.current === null) {
      stageLeftRef.current = el.getBoundingClientRect().left;
    }
    const relX = clientX - stageLeftRef.current;
    const ratio = (relX - padL) / plotW;
    const clamped = Math.max(0, Math.min(1, ratio));
    return Math.round(clamped * (points.length - 1));
  }

  function handlePointerEnter() {
    refreshStageLeft();
  }

  function handlePointerMove(event: React.PointerEvent<SVGSVGElement>) {
    if (event.pointerType === 'touch') return;
    const nextIdx = resolveIndexFromClientX(event.clientX);
    setHoveredIdx((curr) => (curr === nextIdx ? curr : nextIdx));
  }

  function handlePointerLeave(event: React.PointerEvent<SVGSVGElement>) {
    if (event.pointerType === 'touch') return;
    setHoveredIdx(null);
  }

  function handlePointerUp(event: React.PointerEvent<SVGSVGElement>) {
    const idx = resolveIndexFromClientX(event.clientX, true);
    const point = points[idx];
    if (!point) return;
    if (event.pointerType === 'touch') {
      if (touchPinnedIdx === idx) {
        if (onSelectDate && point.sessions > 0) {
          onSelectDate(point.date);
        } else {
          setTouchPinnedIdx(null);
        }
      } else {
        setTouchPinnedIdx(idx);
      }
      return;
    }
    if (onSelectDate && point.sessions > 0) {
      onSelectDate(point.date);
    }
  }

  // Position tooltip neatly left or right of active point without overflowing narrow viewports.
  const tooltipPlaceLeft = activeCoord ? chartW - activeCoord.x < 172 : false;
  const tooltipLeftPx = activeCoord
    ? (tooltipPlaceLeft
      ? Math.max(150, activeCoord.x - 12)
      : Math.min(chartW - 150, activeCoord.x + 12))
    : 0;
  const tooltipTopPx = activeCoord
    ? Math.max(4, Math.min(baselineY - 76, activeCoord.y - 32))
    : 8;

  const showRestingDots = points.length <= 16;

  return (
    <div className={styles.chartCard} aria-label="每日 CU 消耗趋势图">
      <div className={styles.chartHead}>
        <div className={styles.chartSummary} aria-label="区间消耗概览">
          <span className={styles.chartStat}>
            <small>合计</small>
            <strong>{cuNumber(summary.totalCu)}</strong>
            <small>CU</small>
          </span>
          <span className={styles.chartStatSep} aria-hidden />
          <span className={styles.chartStat}>
            <small>日均</small>
            <strong>{cuNumber(summary.avgCu)}</strong>
          </span>
          <span className={styles.chartStatSep} aria-hidden />
          <span className={styles.chartStat}>
            <small>峰值</small>
            <strong>{cuNumber(summary.peakCu)}</strong>
            {summary.peakPoint && summary.peakCu > 0 && (
              <small>（{shortDayLabel(summary.peakPoint.date)}）</small>
            )}
          </span>
        </div>

        <div className={styles.chartHeadActions}>
          {loading && <span className="muted small">更新中…</span>}
          {scale.hasLimit && dailyLimitCu != null && (
            <>
              {peakLimitPct !== null && !isNarrow && (
                <span className={styles.chartLimitHint} title={`日上限 ${cuNumber(dailyLimitCu)} CU`}>
                  上限 {cuNumber(dailyLimitCu)} CU · 峰值占 {peakLimitPct < 0.1 && peakLimitPct > 0 ? '<0.1' : peakLimitPct.toFixed(1)}%
                </span>
              )}
              <button
                type="button"
                className={styles.chartToggleBtn}
                data-active={scale.showLimitInPlot}
                aria-pressed={scale.showLimitInPlot}
                onClick={() => setLimitOverride(!scale.showLimitInPlot)}
                title={scale.showLimitInPlot ? '隐藏日上限参考线，按实际消耗自适应缩放' : `在图中显示日上限参考线（${cuNumber(dailyLimitCu)} CU）`}
              >
                <span className={styles.chartToggleDash} aria-hidden />
                显示上限线
              </button>
            </>
          )}
        </div>
      </div>

      <div ref={stageRef} className={styles.chartStage}>
        <svg
          className={styles.chartSvg}
          width={chartW}
          height={chartH}
          viewBox={`0 0 ${chartW} ${chartH}`}
          role="img"
          aria-label={`每日 CU 消耗面积折线图，共 ${points.length} 天`}
          style={{ cursor: activePoint && activePoint.sessions > 0 && onSelectDate ? 'pointer' : 'crosshair' }}
          onPointerEnter={handlePointerEnter}
          onPointerMove={handlePointerMove}
          onPointerLeave={handlePointerLeave}
          onPointerUp={handlePointerUp}
        >
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="currentColor" stopOpacity="0.16" />
              <stop offset="72%" stopColor="currentColor" stopOpacity="0.03" />
              <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Horizontal grid & crisp Y ticks */}
          {scale.ticks.map((tick) => {
            const ratio = scale.yMax > 0 ? tick / scale.yMax : 0;
            const y = baselineY - ratio * plotH;
            const isBase = tick === 0;
            return (
              <g key={tick}>
                <line
                  x1={padL}
                  x2={chartW - padR}
                  y1={y}
                  y2={y}
                  className={isBase ? styles.chartBaseLine : styles.chartGridLine}
                />
                <text
                  x={padL - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  className={styles.chartAxisText}
                >
                  {formatAxisTick(tick)}
                </text>
              </g>
            );
          })}

          {/* Optional daily limit reference line */}
          {limitY !== null && dailyLimitCu != null && (
            <g>
              <line
                x1={padL}
                x2={chartW - padR}
                y1={limitY}
                y2={limitY}
                className={styles.chartLimitLine}
              />
              <text
                x={chartW - padR}
                y={Math.max(11, limitY - 5)}
                textAnchor="end"
                className={styles.chartLimitText}
              >
                日上限 {cuNumber(dailyLimitCu)} CU
              </text>
            </g>
          )}

          {/* Smooth area fill + crisp stroke */}
          {points.length >= 2 && (
            <>
              <path d={areaPath} fill={`url(#${gradId})`} className={styles.chartAreaPath} />
              <path d={linePath} fill="none" className={styles.chartLinePath} />
            </>
          )}

          {/* Single-day graceful stem */}
          {points.length === 1 && coords[0] && (
            <line
              x1={coords[0].x}
              x2={coords[0].x}
              y1={baselineY}
              y2={coords[0].y}
              className={styles.chartSingleStem}
            />
          )}

          {/* Resting subtle markers on sparse timelines */}
          {showRestingDots && points.map((point, index) => {
            if (points.length > 1 && point.actual_cu <= 0) return null;
            if (activeIdx === index) return null;
            const c = coords[index];
            const overLimit = scale.hasLimit && dailyLimitCu != null && point.actual_cu >= dailyLimitCu;
            return (
              <circle
                key={point.date}
                cx={c.x}
                cy={c.y}
                r={points.length === 1 ? 4 : 2.5}
                className={styles.chartDot}
                data-over={overLimit}
              />
            );
          })}

          {/* Hover / pinned vertical crosshair & glowing double-ring node */}
          {activeCoord && activePoint && (
            <g className={styles.chartCrosshairGroup}>
              <line
                x1={activeCoord.x}
                x2={activeCoord.x}
                y1={padT}
                y2={baselineY}
                className={styles.chartCrosshair}
              />
              <circle
                cx={activeCoord.x}
                cy={activeCoord.y}
                r={9}
                className={styles.chartFocusHalo}
                data-over={scale.hasLimit && dailyLimitCu != null && activePoint.actual_cu >= dailyLimitCu}
              />
              <circle
                cx={activeCoord.x}
                cy={activeCoord.y}
                r={4}
                className={styles.chartFocusDot}
                data-over={scale.hasLimit && dailyLimitCu != null && activePoint.actual_cu >= dailyLimitCu}
              />
            </g>
          )}

          {/* Crisp X-axis date ticks */}
          {points.map((point, index) => {
            const isFirst = index === 0;
            const isLast = index === points.length - 1;
            const nearLast = !isLast && points.length - 1 - index < labelStep * 0.6;
            const showLabel = isFirst || isLast || (!nearLast && index % labelStep === 0);
            if (!showLabel) return null;
            const c = coords[index];
            const anchor = points.length === 1
              ? 'middle'
              : isFirst ? 'start' : isLast ? 'end' : 'middle';
            return (
              <text
                key={point.date}
                x={c.x}
                y={chartH - 7}
                textAnchor={anchor}
                className={styles.chartAxisText}
                data-active={activeIdx === index}
              >
                {shortDayLabel(point.date)}
              </text>
            );
          })}
        </svg>

        {/* Floating crosshair tooltip */}
        {activePoint && activeCoord && (
          <div
            className={styles.chartTooltip}
            data-side={tooltipPlaceLeft ? 'left' : 'right'}
            data-interactive={touchPinnedIdx !== null}
            style={{
              transform: tooltipPlaceLeft
                ? `translate3d(${tooltipLeftPx}px, ${tooltipTopPx}px, 0) translateX(-100%)`
                : `translate3d(${tooltipLeftPx}px, ${tooltipTopPx}px, 0)`,
            }}
            role="status"
            aria-live="polite"
          >
            <div className={styles.chartTooltipDate}>
              <strong>{activePoint.date}</strong>
              {scale.hasLimit && dailyLimitCu != null && activePoint.actual_cu >= dailyLimitCu && (
                <span className={styles.chartTooltipWarn}>超上限</span>
              )}
            </div>
            <div className={styles.chartTooltipMetric}>
              <strong>{cuNumber(activePoint.actual_cu)}</strong>
              <span>CU</span>
              <span className="muted">· {activePoint.sessions} 场</span>
            </div>
            {onSelectDate && activePoint.sessions > 0 && (
              touchPinnedIdx !== null ? (
                <button
                  type="button"
                  className={styles.chartTooltipLink}
                  onClick={() => onSelectDate(activePoint.date)}
                >
                  查看当日会话<ArrowUpRight size={12} aria-hidden />
                </button>
              ) : (
                <span className={styles.chartTooltipHint}>点击查看当日会话 ↗</span>
              )
            )}
          </div>
        )}
      </div>

      {clipped && (
        <p className="muted small">趋势图最多展示区间内最近 45 天；可通过上方日期范围缩小查看。</p>
      )}
    </div>
  );
}
