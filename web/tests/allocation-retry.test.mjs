import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('allocation retry copy helper and settings UI wiring', async () => {
  const copySource = fs.readFileSync(path.join(root, 'src/lib/allocation-retry-copy.ts'), 'utf8');
  assert.match(copySource, /export function formatAllocationRetryHeadline/);
  assert.match(copySource, /export function countDistinctDriveAccounts/);

  const settingsPage = fs.readFileSync(path.join(root, 'src/pages/SettingsPage.tsx'), 'utf8');
  assert.match(settingsPage, /AllocationRetrySettings/);

  const retrySettings = fs.readFileSync(path.join(root, 'src/components/AllocationRetrySettings.tsx'), 'utf8');
  assert.match(retrySettings, /启用自动排队抢卡/);
  assert.match(retrySettings, /参与重试的显卡池/);
  assert.match(retrySettings, /429/);
  assert.match(retrySettings, /const MIN_INTERVAL = 10;/);
  assert.match(retrySettings, /const DEFAULT_INTERVAL = 15;/);
  assert.match(retrySettings, /自动抵扣/);

  const nativePage = fs.readFileSync(path.join(root, 'src/pages/NativeComfyPage.tsx'), 'utf8');
  assert.match(nativePage, /自动选择账号（多账号轮换）/);
  assert.match(nativePage, /停止排队/);
  assert.match(nativePage, /allocation_retry/);

  const nativeApi = fs.readFileSync(path.join(root, 'src/components/NativeApiSettings.tsx'), 'utf8');
  assert.match(nativeApi, /自动选择账号（多账号轮换）/);

  const creationApi = fs.readFileSync(path.join(root, 'src/pages/CreationApiPage.tsx'), 'utf8');
  assert.match(creationApi, /allocation_retry/);
  assert.match(creationApi, /停止排队/);
});
