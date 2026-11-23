import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const webRoot = fileURLToPath(new URL('..', import.meta.url));
const read = (rel) => readFileSync(join(webRoot, rel), 'utf8');

test('AccountDirectory exposes recommended Colab CLI auth alongside experimental Web OAuth and safe local-only unlink defaults', () => {
  const accounts = read('src/components/AccountDirectory.tsx');
  assert.match(accounts, /Colab CLI 授权/);
  assert.match(accounts, /channelTagRecommended[\s\S]*推荐/);
  assert.match(accounts, /Web OAuth 授权/);
  assert.match(accounts, /channelTagExperimental[\s\S]*实验性/);
  assert.match(accounts, /ColabCliAuthDialog/);
  assert.match(accounts, /setDisconnectRevokeRemote\(colabChannel\(account\) !== 'colab_cli'\)/);
  assert.match(accounts, /setDeleteRevokeRemote\(colabChannel\(account\) !== 'colab_cli'\)/);
  assert.match(accounts, /同时向 Google 远端吊销令牌/);
});

test('ColabCliAuthDialog guides copy-paste authorization, JSON import fallback, and Drive binding', () => {
  const dialog = read('src/components/ColabCliAuthDialog.tsx');
  assert.match(dialog, /studio-colab-cli-auth:/);
  assert.match(dialog, /复制授权链接/);
  assert.match(dialog, /新窗口打开/);
  assert.match(dialog, /提交授权码并连接/);
  assert.match(dialog, /已有现成 Colab CLI 凭据 JSON 或 Refresh Token？直接导入/);
  assert.match(dialog, /立即绑定 Drive/);
  assert.match(dialog, /Colab 最小必要权限/);
});

test('SettingsPage and GoogleOAuthSetup clarify Drive Web OAuth requirement vs experimental Colab Web OAuth', () => {
  const settings = read('src/pages/SettingsPage.tsx');
  const setup = read('src/components/GoogleOAuthSetup.tsx');
  assert.match(settings, /Drive 必填 \/ Colab 实验性/);
  assert.match(setup, /Colab CLI 授权（推荐）/);
  assert.match(setup, /Web OAuth 授权（实验性）/);
});
