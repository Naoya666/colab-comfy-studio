import { useState } from 'react';
import { ChevronDown, Pin, Sparkles } from 'lucide-react';
import {
  SEPARATOR_OPTIONS,
  composePrompt,
  hasFixedPromptText,
  hasTriggerWordInText,
  isFixedPromptActive,
  summarizeFixedPrompts,
  toggleTriggerWordInText,
  type FixedPromptConfig,
  type FixedPromptSlot,
  type PromptPosition,
  type PromptSeparator,
  type TriggerWordChip,
} from '../lib/fixed-prompts';
import styles from '../gpu.module.css';

interface FixedPromptPanelProps {
  config: FixedPromptConfig;
  onChange: (next: FixedPromptConfig) => void;
  disabled?: boolean;
  syncing?: boolean;
  mainPrompt: string;
  negativePrompt: string;
  hasNegative: boolean;
  triggerWords: TriggerWordChip[];
  onSubmitShortcut?: () => void;
}

export function FixedPromptPanel({
  config,
  onChange,
  disabled = false,
  syncing = false,
  mainPrompt,
  negativePrompt,
  hasNegative,
  triggerWords,
  onSubmitShortcut,
}: FixedPromptPanelProps) {
  const [open, setOpen] = useState(false);

  const summary = summarizeFixedPrompts(config, hasNegative);
  const posActive = isFixedPromptActive(config.positive);
  const negActive = hasNegative && isFixedPromptActive(config.negative);

  function updateSlot(kind: 'positive' | 'negative', patch: Partial<FixedPromptSlot>) {
    onChange({
      ...config,
      [kind]: {
        ...config[kind],
        ...patch,
      },
    });
  }

  function toggleAllActive() {
    const nextEnabled = summary.activeCount === 0;
    onChange({
      positive: hasFixedPromptText(config.positive)
        ? { ...config.positive, enabled: nextEnabled }
        : config.positive,
      negative: hasNegative && hasFixedPromptText(config.negative)
        ? { ...config.negative, enabled: nextEnabled }
        : config.negative,
    });
  }

  function handleTriggerClick(word: string) {
    const alreadyIn = hasTriggerWordInText(config.positive.text, word);
    const nextText = toggleTriggerWordInText(config.positive.text, word, config.positive.separator);
    updateSlot('positive', {
      text: nextText,
      enabled: alreadyIn ? config.positive.enabled : true,
    });
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !disabled && onSubmitShortcut) {
      e.preventDefault();
      onSubmitShortcut();
    }
  }

  const positivePreview = posActive ? composePrompt(mainPrompt, config.positive) : '';
  const negativePreview = negActive ? composePrompt(negativePrompt, config.negative) : '';

  return (
    <div className={styles.fixedPromptBox} data-active={summary.activeCount > 0}>
      <div className={styles.fixedPromptHeader}>
        <button
          type="button"
          className={styles.fixedPromptToggle}
          aria-expanded={open}
          disabled={disabled}
          onClick={() => setOpen((curr) => !curr)}
        >
          <span className={styles.fixedPromptTitle}>
            <Pin size={14} strokeWidth={1.7} aria-hidden />
            <strong>固定提示词</strong>
            <span
              className={styles.fixedPromptStatus}
              data-tone={summary.activeCount > 0 ? 'active' : summary.hasAnyText ? 'paused' : 'empty'}
            >
              {summary.summaryText}
            </span>
          </span>
          <ChevronDown size={15} aria-hidden className={open ? styles.chevOpen : styles.chev} />
        </button>
        {summary.hasAnyText && (
          <button
            type="button"
            className="button secondary small"
            disabled={disabled}
            aria-pressed={summary.activeCount > 0}
            title={summary.activeCount > 0 ? '临时暂停固定提示词（保留文本）' : '重新启用固定提示词'}
            onClick={toggleAllActive}
          >
            {summary.activeCount > 0 ? '暂停' : '启用'}
          </button>
        )}
      </div>

      {open && (
        <div className={styles.fixedPromptBody}>
          {/* 正面固定提示词 */}
          <div className={styles.fixedSlot}>
            <div className={styles.fixedSlotBar}>
              <label className={styles.checkRow}>
                <input
                  type="checkbox"
                  checked={config.positive.enabled}
                  disabled={disabled}
                  onChange={(e) => updateSlot('positive', { enabled: e.target.checked })}
                />
                <span>固定正面提示词</span>
              </label>
              <div className={styles.fixedSlotControls}>
                <div className={styles.fixedPosGroup} role="group" aria-label="正面固定提示词位置">
                  {([
                    ['prefix', '前置'],
                    ['suffix', '后置'],
                  ] as const).map(([pos, label]) => (
                    <button
                      key={pos}
                      type="button"
                      className={styles.fixedPosBtn}
                      data-active={config.positive.position === pos}
                      aria-pressed={config.positive.position === pos}
                      disabled={disabled}
                      onClick={() => updateSlot('positive', { position: pos as PromptPosition })}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <select
                  aria-label="正面固定提示词分隔符"
                  className={styles.miniSelect}
                  value={config.positive.separator}
                  disabled={disabled}
                  onChange={(e) => updateSlot('positive', { separator: e.target.value as PromptSeparator })}
                >
                  {SEPARATOR_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
            </div>
            <textarea
              aria-label="正面固定提示词"
              value={config.positive.text}
              disabled={disabled}
              rows={2}
              placeholder="LoRA 触发词、固定画风或角色外貌（按当前项目 + 工作流自动同步）"
              onChange={(e) => updateSlot('positive', { text: e.target.value })}
              onKeyDown={handleKeyDown}
            />
            {triggerWords.length > 0 && (
              <div className={styles.triggerWordSection}>
                <div className={styles.triggerWordHead}>
                  <span className={styles.triggerWordLabel}>
                    <Sparkles size={13} aria-hidden />
                    当前 LoRA 触发词
                  </span>
                  <span className="muted small">点击加入或移出正面固定词</span>
                </div>
                <div className={styles.chipRow} role="group" aria-label="LoRA 触发词快捷胶囊">
                  {triggerWords.map((item) => {
                    const active = hasTriggerWordInText(config.positive.text, item.word);
                    return (
                      <button
                        key={item.word}
                        type="button"
                        className={styles.triggerChip}
                        data-active={active}
                        aria-pressed={active}
                        disabled={disabled}
                        title={`来源：${item.sources.join('、')}`}
                        onClick={() => handleTriggerClick(item.word)}
                      >
                        {item.word}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* 负面固定提示词 */}
          {hasNegative && (
            <div className={styles.fixedSlot}>
              <div className={styles.fixedSlotBar}>
                <label className={styles.checkRow}>
                  <input
                    type="checkbox"
                    checked={config.negative.enabled}
                    disabled={disabled}
                    onChange={(e) => updateSlot('negative', { enabled: e.target.checked })}
                  />
                  <span>固定负面提示词</span>
                </label>
                <div className={styles.fixedSlotControls}>
                  <div className={styles.fixedPosGroup} role="group" aria-label="负面固定提示词位置">
                    {([
                      ['prefix', '前置'],
                      ['suffix', '后置'],
                    ] as const).map(([pos, label]) => (
                      <button
                        key={pos}
                        type="button"
                        className={styles.fixedPosBtn}
                        data-active={config.negative.position === pos}
                        aria-pressed={config.negative.position === pos}
                        disabled={disabled}
                        onClick={() => updateSlot('negative', { position: pos as PromptPosition })}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <select
                    aria-label="负面固定提示词分隔符"
                    className={styles.miniSelect}
                    value={config.negative.separator}
                    disabled={disabled}
                    onChange={(e) => updateSlot('negative', { separator: e.target.value as PromptSeparator })}
                  >
                    {SEPARATOR_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
              </div>
              <textarea
                aria-label="负面固定提示词"
                value={config.negative.text}
                disabled={disabled}
                rows={2}
                placeholder="固定负面提示词，例如：lowres, blurry, bad anatomy..."
                onChange={(e) => updateSlot('negative', { text: e.target.value })}
                onKeyDown={handleKeyDown}
              />
            </div>
          )}

          {/* 最终生效预览 */}
          {(posActive || negActive) && (
            <div className={styles.fixedPromptPreview} aria-live="polite">
              <div className={styles.fixedPromptPreviewHead}>
                <span>最终生效预览</span>
                {syncing && <small className="muted">同步中…</small>}
              </div>
              {posActive && (
                <p className={styles.fixedPromptPreviewLine}>
                  <strong>正面：</strong>
                  <span>{positivePreview}</span>
                </p>
              )}
              {negActive && (
                <p className={styles.fixedPromptPreviewLine}>
                  <strong>负面：</strong>
                  <span>{negativePreview}</span>
                </p>
              )}
            </div>
          )}

          {!posActive && !negActive && syncing && (
            <p className="muted small">正在同步固定提示词…</p>
          )}
        </div>
      )}
    </div>
  );
}
