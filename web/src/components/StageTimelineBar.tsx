import { useMemo, useState } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { formatDuration } from '../lib/job-history';
import { computeTimelineSegments, type StageTimelineItem } from '../lib/stage-timeline';
import styles from '../gpu.module.css';

export function StageTimelineBar({
  title,
  items,
  ariaLabel = '阶段耗时统计',
}: {
  title: string;
  items: StageTimelineItem[];
  ariaLabel?: string;
}) {
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const { segments, totalSeconds, hasInProgress } = useMemo(() => computeTimelineSegments(items), [items]);

  if (!segments.length) return null;

  const activeSegment = hoveredKey ? segments.find((seg) => seg.key === hoveredKey) ?? null : null;
  const formatShare = (seg: { displaySeconds: number; percent: number }) => {
    if (totalSeconds <= 0) return '';
    const rounded = Math.round(seg.percent);
    if (rounded === 0 && seg.displaySeconds > 0) return '<1%';
    return `${rounded}%`;
  };

  return (
    <div className={styles.stageTimeline} role="group" aria-label={ariaLabel}>
      <div className={styles.stageTimelineHead}>
        <span className={styles.stageTimelineTitle}>{title}</span>
        <span className={styles.stageTimelineTotal}>
          {hasInProgress ? '已耗时 ' : '总耗时 '}
          <strong>{formatDuration(totalSeconds)}</strong>
        </span>
      </div>

      <div
        className={styles.stageBarTrack}
        role="img"
        aria-label={`${title}，共 ${segments.length} 个阶段，${hasInProgress ? '已耗时' : '总耗时'} ${formatDuration(totalSeconds)}`}
      >
        {segments.map((seg) => {
          const isHighlighted = hoveredKey === seg.key;
          const isDimmed = hoveredKey !== null && !isHighlighted;
          const shareLabel = formatShare(seg);
          const durationText = seg.seconds !== null ? formatDuration(seg.seconds) : '进行中';
          const tooltip = shareLabel ? `${seg.label} · ${durationText}（${shareLabel}）` : `${seg.label} · ${durationText}`;
          return (
            <button
              key={seg.key}
              type="button"
              className={styles.stageBarSegment}
              data-status={seg.status}
              data-highlighted={isHighlighted ? 'true' : undefined}
              data-dimmed={isDimmed ? 'true' : undefined}
              style={{
                flexGrow: Math.max(seg.percent, 0.5),
                backgroundColor: seg.color,
              }}
              title={tooltip}
              aria-label={tooltip}
              onMouseEnter={() => setHoveredKey(seg.key)}
              onMouseLeave={() => setHoveredKey((cur) => (cur === seg.key ? null : cur))}
              onFocus={() => setHoveredKey(seg.key)}
              onBlur={() => setHoveredKey((cur) => (cur === seg.key ? null : cur))}
              onClick={() => setHoveredKey((cur) => (cur === seg.key ? null : seg.key))}
            />
          );
        })}
      </div>

      <div className={styles.stageLegendRow}>
        <span className={styles.stageLegendHint}>
          {activeSegment
            ? `${activeSegment.label} · ${activeSegment.seconds !== null ? formatDuration(activeSegment.seconds) : '进行中'}${formatShare(activeSegment) ? ` (${formatShare(activeSegment)})` : ''}`
            : '颜色越红代表该阶段耗时越久'}
        </span>
        <span className={styles.stageHeatScale} aria-hidden="true">
          <span>快</span>
          <i className={styles.stageHeatGradient} />
          <span>慢</span>
        </span>
      </div>

      <ul className={styles.stageList} aria-label={`${title}明细`}>
        {segments.map((seg) => {
          const isHighlighted = hoveredKey === seg.key;
          const isDimmed = hoveredKey !== null && !isHighlighted;
          const durationText = seg.seconds !== null ? formatDuration(seg.seconds) : '进行中';
          const shareText = seg.seconds !== null ? formatShare(seg) : '';
          return (
            <li
              key={seg.key}
              className={styles.stageListItem}
              data-status={seg.status}
              data-highlighted={isHighlighted ? 'true' : undefined}
              data-dimmed={isDimmed ? 'true' : undefined}
              onMouseEnter={() => setHoveredKey(seg.key)}
              onMouseLeave={() => setHoveredKey((cur) => (cur === seg.key ? null : cur))}
            >
              <span className={styles.stageItemMain}>
                <span className={styles.stageMarker} style={{ color: seg.color }} aria-hidden="true">
                  {seg.status === 'in_progress' ? (
                    <Loader2 size={13} className={styles.spin} />
                  ) : seg.status === 'failed' ? (
                    <AlertCircle size={13} />
                  ) : (
                    '●'
                  )}
                </span>
                <span className={styles.stageItemLabel}>{seg.label}</span>
              </span>
              <span className={styles.stageItemMeta}>
                <span className={styles.stageItemDuration}>{durationText}</span>
                {shareText && <span className={styles.stageItemShare}>{shareText}</span>}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
