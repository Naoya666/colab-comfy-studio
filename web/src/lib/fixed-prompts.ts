export type PromptPosition = 'prefix' | 'suffix';
export type PromptSeparator = 'comma' | 'space' | 'newline';

export interface FixedPromptSlot {
  enabled: boolean;
  text: string;
  position: PromptPosition;
  separator: PromptSeparator;
}

export interface FixedPromptConfig {
  positive: FixedPromptSlot;
  negative: FixedPromptSlot;
}

export interface PromptCompositionSlot {
  param_key: string;
  main: string;
  fixed: string;
  enabled: boolean;
  position: PromptPosition;
  separator: PromptSeparator;
}

export interface PromptComposition {
  positive?: PromptCompositionSlot | null;
  negative?: PromptCompositionSlot | null;
}

export interface TriggerWordChip {
  word: string;
  sources: string[];
}

export const SEPARATOR_OPTIONS: ReadonlyArray<{ value: PromptSeparator; label: string }> = [
  { value: 'comma', label: '逗号 (, )' },
  { value: 'space', label: '空格' },
  { value: 'newline', label: '换行' },
];

export function defaultFixedPromptSlot(): FixedPromptSlot {
  return {
    enabled: true,
    text: '',
    position: 'prefix',
    separator: 'comma',
  };
}

export function defaultFixedPromptConfig(): FixedPromptConfig {
  return {
    positive: defaultFixedPromptSlot(),
    negative: defaultFixedPromptSlot(),
  };
}

export function normalizeFixedPromptSlot(raw?: Partial<FixedPromptSlot> | null): FixedPromptSlot {
  const base = defaultFixedPromptSlot();
  if (!raw || typeof raw !== 'object') return base;
  return {
    enabled: typeof raw.enabled === 'boolean' ? raw.enabled : base.enabled,
    text: typeof raw.text === 'string' ? raw.text : base.text,
    position: raw.position === 'suffix' ? 'suffix' : 'prefix',
    separator: raw.separator === 'space' || raw.separator === 'newline' ? raw.separator : 'comma',
  };
}

export function normalizeFixedPromptConfig(raw?: Partial<FixedPromptConfig> | null): FixedPromptConfig {
  if (!raw || typeof raw !== 'object') return defaultFixedPromptConfig();
  return {
    positive: normalizeFixedPromptSlot(raw.positive),
    negative: normalizeFixedPromptSlot(raw.negative),
  };
}

export function hasFixedPromptText(slot?: FixedPromptSlot | null): boolean {
  return Boolean(slot && slot.text.trim().length > 0);
}

export function isFixedPromptActive(slot?: FixedPromptSlot | null): boolean {
  return Boolean(slot && slot.enabled && slot.text.trim().length > 0);
}

export function joinPromptParts(first: string, second: string, separator: PromptSeparator): string {
  const a = first.trim();
  const b = second.trim();
  if (!a) return b;
  if (!b) return a;
  if (separator === 'newline') {
    return `${first.replace(/[ \t\r\n]+$/, '')}\n${second.replace(/^[ \t\r\n]+/, '')}`;
  }
  if (separator === 'space') {
    return `${a} ${b}`;
  }
  if (/\n$/.test(a) || /^\n/.test(b)) {
    return `${a.replace(/[ \t]+$/, '')}\n${b.replace(/^[ \t]+/, '')}`;
  }
  if (/[,;:]$/.test(a)) {
    const cleanB = b.replace(/^[,，、;；\s]+/, '');
    return cleanB ? `${a} ${cleanB}` : a;
  }
  if (/[，、；：]$/.test(a)) {
    const cleanB = b.replace(/^[,，、;；\s]+/, '');
    return cleanB ? `${a}${cleanB}` : a;
  }
  if (/^[,，、;；]/.test(b)) {
    const cleanB = b.replace(/^[,，、;；\s]+/, '');
    return cleanB ? `${a}, ${cleanB}` : a;
  }
  return `${a}, ${b}`;
}

export function composePrompt(main: string, slot?: FixedPromptSlot | null): string {
  if (!isFixedPromptActive(slot)) return main;
  const fixed = slot!.text.trim();
  if (!main.trim()) return fixed;
  return slot!.position === 'suffix'
    ? joinPromptParts(main, fixed, slot!.separator)
    : joinPromptParts(fixed, main, slot!.separator);
}

export function applyFixedPromptsToParams(
  params: Record<string, unknown>,
  fixedPrompts: FixedPromptConfig,
  promptKey: string | null,
  negativePromptKey: string | null,
): {
  params: Record<string, unknown>;
  promptComposition?: PromptComposition;
} {
  const nextParams: Record<string, unknown> = { ...params };
  const positiveActive = Boolean(promptKey && isFixedPromptActive(fixedPrompts.positive));
  const negativeActive = Boolean(negativePromptKey && isFixedPromptActive(fixedPrompts.negative));

  if (promptKey && positiveActive) {
    const main = String(params[promptKey] ?? '');
    nextParams[promptKey] = composePrompt(main, fixedPrompts.positive);
  }
  if (negativePromptKey && negativeActive) {
    const negMain = String(params[negativePromptKey] ?? '');
    nextParams[negativePromptKey] = composePrompt(negMain, fixedPrompts.negative);
  }

  if (!positiveActive && !negativeActive) {
    return { params: nextParams };
  }

  const promptComposition: PromptComposition = {};
  if (promptKey) {
    promptComposition.positive = {
      param_key: promptKey,
      main: String(params[promptKey] ?? ''),
      fixed: fixedPrompts.positive.text,
      enabled: fixedPrompts.positive.enabled,
      position: fixedPrompts.positive.position,
      separator: fixedPrompts.positive.separator,
    };
  }
  if (negativePromptKey) {
    promptComposition.negative = {
      param_key: negativePromptKey,
      main: String(params[negativePromptKey] ?? ''),
      fixed: fixedPrompts.negative.text,
      enabled: fixedPrompts.negative.enabled,
      position: fixedPrompts.negative.position,
      separator: fixedPrompts.negative.separator,
    };
  }
  return { params: nextParams, promptComposition };
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function stripFixedPromptFromFull(
  fullText: string,
  slot: FixedPromptSlot,
): { matched: boolean; main: string } {
  const fixed = slot.text.trim();
  const full = fullText.trim();
  if (!fixed) return { matched: true, main: fullText };
  if (!full) return { matched: false, main: fullText };
  if (full === fixed) return { matched: true, main: '' };

  if (slot.position === 'prefix') {
    if (!full.startsWith(fixed)) return { matched: false, main: fullText };
    const rest = full.slice(fixed.length);
    if (!rest) return { matched: true, main: '' };
    if (/^[\s,，、;；:：]/.test(rest) || /[,，、;；:：\n]$/.test(fixed)) {
      const cleaned = rest.replace(/^[\s,，、;；]+/, '');
      return { matched: true, main: cleaned };
    }
    return { matched: false, main: fullText };
  }

  if (!full.endsWith(fixed)) return { matched: false, main: fullText };
  const head = full.slice(0, full.length - fixed.length);
  if (!head) return { matched: true, main: '' };
  if (/[\s,，、;；:：]$/.test(head) || /^[,，、;；]/.test(fixed)) {
    const cleaned = head.replace(/[\s,，、;；]+$/, '');
    return { matched: true, main: cleaned };
  }
  return { matched: false, main: fullText };
}

export function restoreFromHistoricalJob(
  resolvedParams: Record<string, unknown>,
  promptComposition: PromptComposition | null | undefined,
  currentFixed: FixedPromptConfig,
  promptKey: string | null,
  negativePromptKey: string | null,
): {
  params: Record<string, unknown>;
  fixedPrompts: FixedPromptConfig;
  pausedLegacy: boolean;
} {
  const nextParams: Record<string, unknown> = { ...resolvedParams };
  const nextFixed = normalizeFixedPromptConfig(currentFixed);

  if (promptComposition && (promptComposition.positive || promptComposition.negative)) {
    if (promptComposition.positive) {
      const slot = promptComposition.positive;
      const targetKey = slot.param_key || promptKey;
      if (targetKey) nextParams[targetKey] = slot.main;
      nextFixed.positive = normalizeFixedPromptSlot({
        enabled: slot.enabled,
        text: slot.fixed,
        position: slot.position,
        separator: slot.separator,
      });
    } else {
      nextFixed.positive = { ...nextFixed.positive, enabled: false };
    }

    if (promptComposition.negative) {
      const slot = promptComposition.negative;
      const targetKey = slot.param_key || negativePromptKey;
      if (targetKey) nextParams[targetKey] = slot.main;
      nextFixed.negative = normalizeFixedPromptSlot({
        enabled: slot.enabled,
        text: slot.fixed,
        position: slot.position,
        separator: slot.separator,
      });
    } else if (negativePromptKey) {
      nextFixed.negative = { ...nextFixed.negative, enabled: false };
    }
    return { params: nextParams, fixedPrompts: nextFixed, pausedLegacy: false };
  }

  let pausedLegacy = false;
  if (promptKey && isFixedPromptActive(nextFixed.positive) && typeof nextParams[promptKey] === 'string') {
    const stripped = stripFixedPromptFromFull(nextParams[promptKey] as string, nextFixed.positive);
    if (stripped.matched) {
      nextParams[promptKey] = stripped.main;
    } else {
      nextFixed.positive = { ...nextFixed.positive, enabled: false };
      pausedLegacy = true;
    }
  }
  if (negativePromptKey && isFixedPromptActive(nextFixed.negative) && typeof nextParams[negativePromptKey] === 'string') {
    const stripped = stripFixedPromptFromFull(nextParams[negativePromptKey] as string, nextFixed.negative);
    if (stripped.matched) {
      nextParams[negativePromptKey] = stripped.main;
    } else {
      nextFixed.negative = { ...nextFixed.negative, enabled: false };
      pausedLegacy = true;
    }
  }
  return { params: nextParams, fixedPrompts: nextFixed, pausedLegacy };
}

export function hasTriggerWordInText(text: string, word: string): boolean {
  const target = word.trim().toLowerCase();
  if (!target || !text.trim()) return false;
  const segments = text.split(/[,，、\n]+/).map((part) => part.trim().toLowerCase()).filter(Boolean);
  if (segments.includes(target)) return true;
  const boundaryRe = new RegExp(`(?:^|[\\s,，、;；\\n])${escapeRegExp(target)}(?=$|[\\s,，、;；\\n])`, 'i');
  return boundaryRe.test(text);
}

export function toggleTriggerWordInText(text: string, word: string, separator: PromptSeparator): string {
  const cleanWord = word.trim();
  if (!cleanWord) return text;
  if (!hasTriggerWordInText(text, cleanWord)) {
    return joinPromptParts(text, cleanWord, separator);
  }

  const targetLower = cleanWord.toLowerCase();
  if (separator === 'newline') {
    const lines = text.split(/\r?\n/);
    const filtered = lines.filter((line) => line.trim().toLowerCase() !== targetLower);
    if (filtered.length !== lines.length) {
      return filtered.join('\n').replace(/^\n+|\n+$/g, '');
    }
  } else if (separator === 'comma') {
    const parts = text.split(/[,，、\n]+/).map((part) => part.trim()).filter(Boolean);
    const filtered = parts.filter((part) => part.toLowerCase() !== targetLower);
    if (filtered.length !== parts.length) {
      return filtered.join(', ');
    }
  }

  const re = new RegExp(
    `(^|[\\s,，、;；\\n])${escapeRegExp(cleanWord)}(?=$|[\\s,，、;；\\n])`,
    'ig',
  );
  const removed = text.replace(re, (_match, leading: string) => (leading === '\n' ? '\n' : ' '));
  return removed
    .replace(/\s*[,，、]+\s*[,，、]+\s*/g, ', ')
    .replace(/^[,\s，、;；]+|[,\s，、;；]+$/g, '')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

export interface LoraDependencyLike {
  id: string;
  model_type: string;
  model_id?: string | null;
  disabled?: boolean;
}

export interface LibraryModelLike {
  id: string;
  display_name?: string;
  filename?: string;
  trigger_words?: string[];
}

export interface LoraOverrideLike {
  dependency_id: string;
  model_id: string;
  disabled?: boolean;
}

export function collectActiveLoraTriggerWords(
  dependencies: LoraDependencyLike[] | null | undefined,
  models: LibraryModelLike[] | null | undefined,
  overrides: LoraOverrideLike[] | null | undefined = [],
): TriggerWordChip[] {
  if (!dependencies?.length || !models?.length) return [];
  const modelMap = new Map(models.map((item) => [item.id, item]));
  const overrideMap = new Map((overrides || []).map((item) => [item.dependency_id, item]));
  const byLower = new Map<string, TriggerWordChip>();

  for (const dep of dependencies) {
    if (dep.model_type !== 'loras') continue;
    const override = overrideMap.get(dep.id);
    const isDisabled = override ? Boolean(override.disabled) : Boolean(dep.disabled);
    if (isDisabled) continue;
    const effectiveModelId = override?.model_id || dep.model_id;
    if (!effectiveModelId) continue;
    const model = modelMap.get(effectiveModelId);
    if (!model?.trigger_words?.length) continue;
    const sourceLabel = model.display_name || model.filename || effectiveModelId;
    for (const rawWord of model.trigger_words) {
      const word = String(rawWord || '').trim();
      if (!word) continue;
      const lower = word.toLowerCase();
      const existing = byLower.get(lower);
      if (existing) {
        if (!existing.sources.includes(sourceLabel)) existing.sources.push(sourceLabel);
      } else {
        byLower.set(lower, { word, sources: [sourceLabel] });
      }
    }
  }
  return [...byLower.values()];
}

export function summarizeFixedPrompts(
  config: FixedPromptConfig,
  hasNegative: boolean,
): { activeCount: number; hasAnyText: boolean; summaryText: string } {
  const posHasText = hasFixedPromptText(config.positive);
  const negHasText = hasNegative && hasFixedPromptText(config.negative);
  const posActive = isFixedPromptActive(config.positive);
  const negActive = hasNegative && isFixedPromptActive(config.negative);
  const hasAnyText = posHasText || negHasText;
  const activeCount = (posActive ? 1 : 0) + (negActive ? 1 : 0);

  if (!hasAnyText) {
    return { activeCount: 0, hasAnyText: false, summaryText: '未设置' };
  }
  if (activeCount === 0) {
    return { activeCount: 0, hasAnyText: true, summaryText: '已暂停' };
  }
  const parts: string[] = [];
  if (posActive) parts.push(`正:${config.positive.position === 'suffix' ? '后置' : '前置'}`);
  if (negActive) parts.push(`负:${config.negative.position === 'suffix' ? '后置' : '前置'}`);
  return { activeCount, hasAnyText: true, summaryText: `已启用（${parts.join(' / ')}）` };
}

export function fixedPromptStorageKey(projectId: string, workflowId: string): string {
  return `studio-fixed-prompts:${projectId}:${workflowId}`;
}

export function fixedPromptPendingStorageKey(projectId: string, workflowId: string): string {
  return `studio-fixed-prompts-pending:${projectId}:${workflowId}`;
}

export function readCachedFixedPrompts(projectId: string, workflowId: string): FixedPromptConfig | null {
  if (!projectId || !workflowId) return null;
  try {
    const raw = localStorage.getItem(fixedPromptStorageKey(projectId, workflowId));
    if (!raw) return null;
    return normalizeFixedPromptConfig(JSON.parse(raw) as Partial<FixedPromptConfig>);
  } catch {
    return null;
  }
}

export function writeCachedFixedPrompts(projectId: string, workflowId: string, config: FixedPromptConfig): void {
  if (!projectId || !workflowId) return;
  try {
    localStorage.setItem(fixedPromptStorageKey(projectId, workflowId), JSON.stringify(normalizeFixedPromptConfig(config)));
  } catch {
    /* browser storage unavailable */
  }
}

export function readPendingFixedPrompts(projectId: string, workflowId: string): FixedPromptConfig | null {
  if (!projectId || !workflowId) return null;
  try {
    const raw = localStorage.getItem(fixedPromptPendingStorageKey(projectId, workflowId));
    if (!raw) return null;
    return normalizeFixedPromptConfig(JSON.parse(raw) as Partial<FixedPromptConfig>);
  } catch {
    return null;
  }
}

export function writePendingFixedPrompts(projectId: string, workflowId: string, config: FixedPromptConfig): void {
  if (!projectId || !workflowId) return;
  try {
    localStorage.setItem(fixedPromptPendingStorageKey(projectId, workflowId), JSON.stringify(normalizeFixedPromptConfig(config)));
  } catch {
    /* browser storage unavailable */
  }
}

export function clearPendingFixedPrompts(projectId: string, workflowId: string): void {
  if (!projectId || !workflowId) return;
  try {
    localStorage.removeItem(fixedPromptPendingStorageKey(projectId, workflowId));
  } catch {
    /* browser storage unavailable */
  }
}

export interface PendingFixedPromptEntry {
  projectId: string;
  workflowId: string;
  config: FixedPromptConfig;
  revision: number;
}

export interface FixedPromptSyncStorage {
  readCached: (projectId: string, workflowId: string) => FixedPromptConfig | null;
  writeCached: (projectId: string, workflowId: string, config: FixedPromptConfig) => void;
  readPending: (projectId: string, workflowId: string) => FixedPromptConfig | null;
  writePending: (projectId: string, workflowId: string, config: FixedPromptConfig) => void;
  clearPending: (projectId: string, workflowId: string) => void;
}

export interface FixedPromptSyncer {
  markDirty: (projectId: string, workflowId: string, config: FixedPromptConfig) => PendingFixedPromptEntry;
  getPending: (projectId: string, workflowId: string) => PendingFixedPromptEntry | null;
  keyVersion: (projectId: string, workflowId: string) => number;
  flush: (projectId: string, workflowId: string) => Promise<boolean>;
  flushAll: () => Promise<void>;
  resolveLoaded: (
    projectId: string,
    workflowId: string,
    remote: Partial<FixedPromptConfig> | null,
  ) => { config: FixedPromptConfig; hasPending: boolean };
}

const defaultFixedPromptSyncStorage: FixedPromptSyncStorage = {
  readCached: readCachedFixedPrompts,
  writeCached: writeCachedFixedPrompts,
  readPending: readPendingFixedPrompts,
  writePending: writePendingFixedPrompts,
  clearPending: clearPendingFixedPrompts,
};

export function createFixedPromptSyncer(opts: {
  save: (projectId: string, workflowId: string, config: FixedPromptConfig) => Promise<unknown>;
  storage?: FixedPromptSyncStorage;
  onSyncingChange?: (syncing: boolean) => void;
}): FixedPromptSyncer {
  const storage = opts.storage ?? defaultFixedPromptSyncStorage;
  const pending = new Map<string, PendingFixedPromptEntry>();
  const versions = new Map<string, number>();
  let nextRevision = 0;
  let inFlightCount = 0;

  const bumpVersion = (key: string) => {
    versions.set(key, (versions.get(key) ?? 0) + 1);
  };

  const getPending = (projectId: string, workflowId: string): PendingFixedPromptEntry | null => {
    if (!projectId || !workflowId) return null;
    const key = fixedPromptStorageKey(projectId, workflowId);
    const existing = pending.get(key);
    if (existing) return existing;
    const stored = storage.readPending(projectId, workflowId);
    if (!stored) return null;
    const hydrated: PendingFixedPromptEntry = {
      projectId,
      workflowId,
      config: normalizeFixedPromptConfig(stored),
      revision: ++nextRevision,
    };
    pending.set(key, hydrated);
    return hydrated;
  };

  const markDirty = (projectId: string, workflowId: string, config: FixedPromptConfig): PendingFixedPromptEntry => {
    const normalized = normalizeFixedPromptConfig(config);
    const key = fixedPromptStorageKey(projectId, workflowId);
    const entry: PendingFixedPromptEntry = {
      projectId,
      workflowId,
      config: normalized,
      revision: ++nextRevision,
    };
    pending.set(key, entry);
    bumpVersion(key);
    storage.writeCached(projectId, workflowId, normalized);
    storage.writePending(projectId, workflowId, normalized);
    return entry;
  };

  const flush = async (projectId: string, workflowId: string): Promise<boolean> => {
    if (!projectId || !workflowId) return true;
    const key = fixedPromptStorageKey(projectId, workflowId);
    const target = getPending(projectId, workflowId);
    if (!target) return true;
    inFlightCount += 1;
    if (inFlightCount === 1) opts.onSyncingChange?.(true);
    try {
      await opts.save(target.projectId, target.workflowId, target.config);
      const latest = pending.get(key);
      if (latest && latest.revision === target.revision) {
        pending.delete(key);
        storage.clearPending(target.projectId, target.workflowId);
        bumpVersion(key);
      }
      return true;
    } catch {
      return false;
    } finally {
      inFlightCount = Math.max(0, inFlightCount - 1);
      if (inFlightCount === 0) opts.onSyncingChange?.(false);
    }
  };

  const flushAll = async (): Promise<void> => {
    const targets = [...pending.values()].map((entry) => ({
      projectId: entry.projectId,
      workflowId: entry.workflowId,
    }));
    await Promise.all(targets.map((t) => flush(t.projectId, t.workflowId)));
  };

  const resolveLoaded = (
    projectId: string,
    workflowId: string,
    remote: Partial<FixedPromptConfig> | null,
  ): { config: FixedPromptConfig; hasPending: boolean } => {
    const unsynced = getPending(projectId, workflowId);
    if (unsynced) {
      return { config: unsynced.config, hasPending: true };
    }
    const cached = storage.readCached(projectId, workflowId);
    const resolved = remote
      ? normalizeFixedPromptConfig(remote)
      : (cached ?? defaultFixedPromptConfig());
    storage.writeCached(projectId, workflowId, resolved);
    return { config: resolved, hasPending: false };
  };

  return {
    markDirty,
    getPending,
    keyVersion: (projectId, workflowId) => versions.get(fixedPromptStorageKey(projectId, workflowId)) ?? 0,
    flush,
    flushAll,
    resolveLoaded,
  };
}
