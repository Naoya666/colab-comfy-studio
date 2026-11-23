import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeNewlyOrphanedPlugins,
  defaultLinkedPluginSelection,
  formatWorkspaceBytes,
} from '../src/lib/native-workspace.ts';

test('formatWorkspaceBytes formats units cleanly', () => {
  assert.equal(formatWorkspaceBytes(0), '0 B');
  assert.equal(formatWorkspaceBytes(512), '512 B');
  assert.equal(formatWorkspaceBytes(2048), '2.0 KB');
  assert.equal(formatWorkspaceBytes(182.89 * 1024 ** 2), '182.89 MB');
  assert.equal(formatWorkspaceBytes(1.5 * 1024 ** 3), '1.50 GB');
});

test('computeNewlyOrphanedPlugins finds exclusively referenced plugins and protects shared ones', () => {
  const plugins = [
    {
      path: 'LanPaint',
      enabled: true,
      archive: 'a'.repeat(64),
      archive_size_bytes: 180 * 1024 ** 2,
      archive_present: true,
      status_category: 'in_use',
      used_by_active: [{ source: 'comfy', path: '工作流1.json' }],
      used_by_archived: [],
      js_count: 0,
    },
    {
      path: 'prompt-assistant',
      enabled: true,
      archive: 'b'.repeat(64),
      archive_size_bytes: 600 * 1024,
      archive_present: true,
      status_category: 'in_use',
      used_by_active: [{ source: 'comfy', path: '工作流1.json' }],
      used_by_archived: [],
      js_count: 27,
    },
    {
      path: 'comfyui-kjnodes',
      enabled: true,
      archive: 'c'.repeat(64),
      archive_size_bytes: 1024 ** 2,
      archive_present: true,
      status_category: 'in_use',
      used_by_active: [
        { source: 'comfy', path: '工作流1.json' },
        { source: 'comfy', path: 'Anima-修复版.json' },
      ],
      used_by_archived: [],
      js_count: 31,
    },
  ];

  const orphaned = computeNewlyOrphanedPlugins([{ source: 'comfy', path: '工作流1.json' }], plugins);
  assert.deepEqual(orphaned.map(p => p.path), ['LanPaint', 'prompt-assistant']);

  const defaults = defaultLinkedPluginSelection(orphaned);
  assert.equal(defaults.LanPaint, true);
  assert.equal(defaults['prompt-assistant'], false);
});
