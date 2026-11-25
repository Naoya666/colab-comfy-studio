import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: {
    contents: `
      export * from './src/lib/fixed-prompts.ts';
      export { promptCompositionRows } from './src/lib/job-detail-copy.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    sourcefile: 'fixed-prompts-test-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});

const {
  composePrompt,
  joinPromptParts,
  applyFixedPromptsToParams,
  restoreFromHistoricalJob,
  hasTriggerWordInText,
  toggleTriggerWordInText,
  collectActiveLoraTriggerWords,
  summarizeFixedPrompts,
  defaultFixedPromptConfig,
  createFixedPromptSyncer,
  promptCompositionRows,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

test('composePrompt handles prefix/suffix, separators, and smart punctuation dedup', () => {
  assert.equal(
    composePrompt('a girl in rain', { enabled: true, text: 'anime_style', position: 'prefix', separator: 'comma' }),
    'anime_style, a girl in rain',
  );
  assert.equal(
    composePrompt('a girl in rain', { enabled: true, text: 'masterpiece', position: 'suffix', separator: 'comma' }),
    'a girl in rain, masterpiece',
  );
  assert.equal(
    composePrompt('slow zoom in', { enabled: true, text: 'camera motion,', position: 'prefix', separator: 'comma' }),
    'camera motion, slow zoom in',
  );
  assert.equal(
    composePrompt('Animate <Picture 1>', { enabled: true, text: 'live_wallpaper:', position: 'prefix', separator: 'comma' }),
    'live_wallpaper: Animate <Picture 1>',
  );
  assert.equal(
    composePrompt('少女', { enabled: true, text: '古风，', position: 'prefix', separator: 'comma' }),
    '古风，少女',
  );
  assert.equal(
    composePrompt('a girl', { enabled: true, text: 'trigger', position: 'prefix', separator: 'space' }),
    'trigger a girl',
  );
  assert.equal(
    composePrompt('a girl', { enabled: true, text: 'trigger', position: 'prefix', separator: 'newline' }),
    'trigger\na girl',
  );
  assert.equal(
    composePrompt('', { enabled: true, text: 'only fixed', position: 'prefix', separator: 'comma' }),
    'only fixed',
  );
  assert.equal(
    composePrompt('main only', { enabled: false, text: 'paused fixed', position: 'prefix', separator: 'comma' }),
    'main only',
  );
  assert.equal(joinPromptParts('a', ', b', 'comma'), 'a, b');
});

test('applyFixedPromptsToParams composes positive & negative and omits metadata when inactive', () => {
  const empty = applyFixedPromptsToParams(
    { prompt: 'hello', negative_prompt: 'blurry', width: 768 },
    defaultFixedPromptConfig(),
    'prompt',
    'negative_prompt',
  );
  assert.equal(empty.params.prompt, 'hello');
  assert.equal(empty.promptComposition, undefined);

  const active = applyFixedPromptsToParams(
    { prompt: 'a cat', negative_prompt: 'extra fingers', width: 768 },
    {
      positive: { enabled: true, text: 'style_a', position: 'prefix', separator: 'comma' },
      negative: { enabled: true, text: 'lowres', position: 'suffix', separator: 'comma' },
    },
    'prompt',
    'negative_prompt',
  );
  assert.equal(active.params.prompt, 'style_a, a cat');
  assert.equal(active.params.negative_prompt, 'extra fingers, lowres');
  assert.equal(active.promptComposition.positive.main, 'a cat');
  assert.equal(active.promptComposition.positive.fixed, 'style_a');
  assert.equal(active.promptComposition.negative.main, 'extra fingers');
  assert.equal(active.promptComposition.negative.fixed, 'lowres');
});

test('restoreFromHistoricalJob restores Scheme A exactly and avoids legacy double-stacking', () => {
  const currentFixed = {
    positive: { enabled: true, text: 'current_lora', position: 'prefix', separator: 'comma' },
    negative: { enabled: true, text: 'lowres', position: 'suffix', separator: 'comma' },
  };

  // Scheme A exact restore
  const exact = restoreFromHistoricalJob(
    { prompt: 'old_lora, mountain view', negative_prompt: 'blurry, old_neg', width: 1024 },
    {
      positive: { param_key: 'prompt', main: 'mountain view', fixed: 'old_lora', enabled: true, position: 'prefix', separator: 'comma' },
      negative: { param_key: 'negative_prompt', main: 'blurry', fixed: 'old_neg', enabled: true, position: 'suffix', separator: 'comma' },
    },
    currentFixed,
    'prompt',
    'negative_prompt',
  );
  assert.equal(exact.pausedLegacy, false);
  assert.equal(exact.params.prompt, 'mountain view');
  assert.equal(exact.params.negative_prompt, 'blurry');
  assert.equal(exact.fixedPrompts.positive.text, 'old_lora');
  assert.equal(exact.fixedPrompts.negative.text, 'old_neg');

  // Legacy job matching current fixed prompt strips cleanly without pausing
  const legacyMatch = restoreFromHistoricalJob(
    { prompt: 'current_lora, ocean sunset', negative_prompt: 'noise, lowres' },
    null,
    currentFixed,
    'prompt',
    'negative_prompt',
  );
  assert.equal(legacyMatch.pausedLegacy, false);
  assert.equal(legacyMatch.params.prompt, 'ocean sunset');
  assert.equal(legacyMatch.params.negative_prompt, 'noise');
  assert.equal(legacyMatch.fixedPrompts.positive.enabled, true);

  // Legacy job not matching current fixed prompt pauses fixed prompt while keeping text
  const legacyMismatch = restoreFromHistoricalJob(
    { prompt: 'completely different prompt', negative_prompt: 'noise' },
    null,
    currentFixed,
    'prompt',
    'negative_prompt',
  );
  assert.equal(legacyMismatch.pausedLegacy, true);
  assert.equal(legacyMismatch.params.prompt, 'completely different prompt');
  assert.equal(legacyMismatch.fixedPrompts.positive.enabled, false);
  assert.equal(legacyMismatch.fixedPrompts.positive.text, 'current_lora');
});

test('trigger word detection, toggling, and active LoRA override collection', () => {
  assert.equal(hasTriggerWordInText('camera motion, anime style', 'Camera Motion'), true);
  assert.equal(hasTriggerWordInText('anime style', 'camera motion'), false);

  const added = toggleTriggerWordInText('anime style', 'camera motion', 'comma');
  assert.equal(added, 'anime style, camera motion');
  const removed = toggleTriggerWordInText(added, 'camera motion', 'comma');
  assert.equal(removed, 'anime style');

  const deps = [
    { id: 'd1', model_type: 'checkpoints', model_id: 'ckpt1' },
    { id: 'd2', model_type: 'loras', model_id: 'lora1' },
    { id: 'd3', model_type: 'loras', model_id: 'lora2' },
  ];
  const models = [
    { id: 'ckpt1', display_name: 'Base', trigger_words: ['ignored_ckpt'] },
    { id: 'lora1', display_name: 'Lora A', trigger_words: ['trigger_a', 'shared_word'] },
    { id: 'lora2', display_name: 'Lora B', trigger_words: ['trigger_b', 'Shared_Word'] },
    { id: 'lora3', display_name: 'Lora C', trigger_words: ['trigger_c'] },
  ];

  const baseChips = collectActiveLoraTriggerWords(deps, models, []);
  assert.deepEqual(
    baseChips.map((c) => c.word),
    ['trigger_a', 'shared_word', 'trigger_b'],
  );
  assert.deepEqual(baseChips[1].sources, ['Lora A', 'Lora B']);

  // Override d2 -> lora3 and disable d3
  const overriddenChips = collectActiveLoraTriggerWords(deps, models, [
    { dependency_id: 'd2', model_id: 'lora3', disabled: false },
    { dependency_id: 'd3', model_id: 'lora2', disabled: true },
  ]);
  assert.deepEqual(overriddenChips.map((c) => c.word), ['trigger_c']);
});

test('summarizeFixedPrompts and promptCompositionRows produce accurate labels', () => {
  assert.equal(summarizeFixedPrompts(defaultFixedPromptConfig(), true).summaryText, '未设置');
  assert.equal(
    summarizeFixedPrompts(
      {
        positive: { enabled: false, text: 'abc', position: 'prefix', separator: 'comma' },
        negative: { enabled: true, text: '', position: 'prefix', separator: 'comma' },
      },
      true,
    ).summaryText,
    '已暂停',
  );
  assert.equal(
    summarizeFixedPrompts(
      {
        positive: { enabled: true, text: 'abc', position: 'prefix', separator: 'comma' },
        negative: { enabled: true, text: 'bad', position: 'suffix', separator: 'comma' },
      },
      true,
    ).summaryText,
    '已启用（正:前置 / 负:后置）',
  );
  assert.equal(
    summarizeFixedPrompts(
      {
        positive: { enabled: true, text: 'abc', position: 'prefix', separator: 'comma' },
        negative: { enabled: true, text: 'bad', position: 'suffix', separator: 'comma' },
      },
      false,
    ).summaryText,
    '已启用（正:前置）',
  );

  const rows = promptCompositionRows({
    positive: { param_key: 'prompt', main: 'girl', fixed: 'style_x', enabled: true, position: 'prefix', separator: 'comma' },
    negative: { param_key: 'negative_prompt', main: '', fixed: 'lowres', enabled: true, position: 'suffix', separator: 'comma' },
  });
  assert.deepEqual(rows.map((r) => r.label), [
    '固定正面提示词（前置）',
    '本次正面提示词',
    '固定负面提示词（后置）',
  ]);
});

test('createFixedPromptSyncer preserves unsynced edits across failed PUT, workflow switch, stale GET, and concurrent edits', async () => {
  const cachedMap = new Map();
  const pendingMap = new Map();
  const keyOf = (pid, wid) => `${pid}:${wid}`;
  const storage = {
    readCached: (pid, wid) => cachedMap.get(keyOf(pid, wid)) ?? null,
    writeCached: (pid, wid, cfg) => cachedMap.set(keyOf(pid, wid), cfg),
    readPending: (pid, wid) => pendingMap.get(keyOf(pid, wid)) ?? null,
    writePending: (pid, wid, cfg) => pendingMap.set(keyOf(pid, wid), cfg),
    clearPending: (pid, wid) => pendingMap.delete(keyOf(pid, wid)),
  };

  let failNextA = true;
  const savedCalls = [];
  const syncingStates = [];
  const syncer = createFixedPromptSyncer({
    storage,
    onSyncingChange: (val) => syncingStates.push(val),
    save: async (pid, wid, cfg) => {
      savedCalls.push({ pid, wid, text: cfg.positive.text });
      if (wid === 'wfA' && failNextA) {
        throw new Error('network error');
      }
    },
  });

  const editedA = {
    positive: { enabled: true, text: 'local_unsynced_edit', position: 'prefix', separator: 'comma' },
    negative: { enabled: true, text: '', position: 'prefix', separator: 'comma' },
  };
  syncer.markDirty('p1', 'wfA', editedA);

  // 1. PUT fails -> pending state must stay preserved in both memory and storage.
  const ok1 = await syncer.flush('p1', 'wfA');
  assert.equal(ok1, false);
  assert.equal(syncer.getPending('p1', 'wfA')?.config.positive.text, 'local_unsynced_edit');
  assert.equal(pendingMap.get('p1:wfA')?.positive.text, 'local_unsynced_edit');
  assert.deepEqual(syncingStates, [true, false]);

  // 2. User switches to wfB and edits/saves wfB -> must not clear wfA pending edit.
  const editedB = {
    positive: { enabled: true, text: 'wfB_text', position: 'prefix', separator: 'comma' },
    negative: { enabled: true, text: '', position: 'prefix', separator: 'comma' },
  };
  syncer.markDirty('p1', 'wfB', editedB);
  assert.equal(await syncer.flush('p1', 'wfB'), true);
  assert.equal(syncer.getPending('p1', 'wfB'), null);
  assert.equal(syncer.getPending('p1', 'wfA')?.config.positive.text, 'local_unsynced_edit');

  // 3. User switches back to wfA and GET succeeds with stale server value -> MUST preserve local unsynced edit!
  const staleServerA = {
    positive: { enabled: true, text: 'stale_server_value', position: 'prefix', separator: 'comma' },
    negative: { enabled: true, text: '', position: 'prefix', separator: 'comma' },
  };
  const loadedAfterSwitch = syncer.resolveLoaded('p1', 'wfA', staleServerA);
  assert.equal(loadedAfterSwitch.hasPending, true);
  assert.equal(loadedAfterSwitch.config.positive.text, 'local_unsynced_edit');
  assert.equal(cachedMap.get('p1:wfA')?.positive.text, 'local_unsynced_edit');

  // 4. Retry PUT now succeeds -> pending cleared, subsequent GET uses server value cleanly.
  failNextA = false;
  assert.equal(await syncer.flush('p1', 'wfA'), true);
  assert.equal(syncer.getPending('p1', 'wfA'), null);
  assert.equal(pendingMap.has('p1:wfA'), false);

  const freshServerA = {
    positive: { enabled: true, text: 'fresh_server_value', position: 'prefix', separator: 'comma' },
    negative: { enabled: true, text: '', position: 'prefix', separator: 'comma' },
  };
  const loadedClean = syncer.resolveLoaded('p1', 'wfA', freshServerA);
  assert.equal(loadedClean.hasPending, false);
  assert.equal(loadedClean.config.positive.text, 'fresh_server_value');

  // 5. Newer edit during an in-flight PUT must NOT be cleared when the older PUT completes.
  let resolveFirstPut;
  const concurrentSyncer = createFixedPromptSyncer({
    storage,
    save: () => new Promise((resolve) => {
      if (!resolveFirstPut) {
        resolveFirstPut = resolve;
      } else {
        resolve();
      }
    }),
  });
  concurrentSyncer.markDirty('p1', 'wfA', {
    positive: { enabled: true, text: 'rev1', position: 'prefix', separator: 'comma' },
    negative: { enabled: true, text: '', position: 'prefix', separator: 'comma' },
  });
  const firstFlush = concurrentSyncer.flush('p1', 'wfA');
  concurrentSyncer.markDirty('p1', 'wfA', {
    positive: { enabled: true, text: 'rev2', position: 'prefix', separator: 'comma' },
    negative: { enabled: true, text: '', position: 'prefix', separator: 'comma' },
  });
  resolveFirstPut();
  await firstFlush;
  assert.equal(concurrentSyncer.getPending('p1', 'wfA')?.config.positive.text, 'rev2');
  assert.equal(concurrentSyncer.resolveLoaded('p1', 'wfA', staleServerA).config.positive.text, 'rev2');
  await concurrentSyncer.flush('p1', 'wfA');
  assert.equal(concurrentSyncer.getPending('p1', 'wfA'), null);
});
