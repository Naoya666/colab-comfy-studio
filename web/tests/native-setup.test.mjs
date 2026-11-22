import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

async function loadModule() {
  const source = readFileSync('src/lib/native-setup.ts', 'utf8');
  const { code } = await transform(source, { loader: 'ts', format: 'esm' });
  return import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
}

test('extractTunnelToken and previewTunnelToken parse cloudflared commands and Base64 JSON', async () => {
  const mod = await loadModule();
  assert.equal(mod.RECOMMENDED_COMFY_COMMIT, '423cbd20511becb1f16122ab518e92e58bfa03d9');

  const rawToken = Buffer.from(JSON.stringify({
    a: 'abcdef1234567890fedcba',
    t: '12345678-1234-1234-1234-1234567890ab',
    s: 'secret-tunnel-bytes',
  })).toString('base64');

  assert.equal(mod.extractTunnelToken(`sudo cloudflared service install ${rawToken}`), rawToken);
  assert.equal(mod.extractTunnelToken(`cloudflared tunnel run --token=${rawToken}`), rawToken);

  const preview = mod.previewTunnelToken(`cloudflared.exe service install ${rawToken}`);
  assert.equal(preview.validCfFormat, true);
  assert.equal(preview.tunnelId, '12345678-1234-1234-1234-1234567890ab');
  assert.equal(preview.accountIdMasked, 'abcdef…dcba');
});

test('extractComfyCommit handles raw SHA and GitHub commit URLs', async () => {
  const mod = await loadModule();
  const sha = '423cbd20511becb1f16122ab518e92e58bfa03d9';
  assert.equal(mod.extractComfyCommit(`https://github.com/Comfy-Org/ComfyUI/commit/${sha.toUpperCase()}#files`), sha);
  assert.equal(mod.isValidComfyCommit(sha), true);
  assert.equal(mod.isValidComfyCommit('master'), false);
});

test('parseAccessJsonInput parses valid JSON and rejects incomplete objects', async () => {
  const mod = await loadModule();
  assert.deepEqual(
    mod.parseAccessJsonInput('{"client_id": "abc.access", "client_secret": "xyz-secret"}'),
    { clientId: 'abc.access', clientSecret: 'xyz-secret' },
  );
  assert.equal(mod.parseAccessJsonInput('{"client_id": "abc.access"}'), null);
  assert.equal(mod.parseAccessJsonInput('not-json'), null);
});

test('Nginx and Caddy proxy templates embed domain, bind port, and streaming/WS directives', async () => {
  const mod = await loadModule();
  const nginx = mod.buildNginxProxyConfig('https://comfy.bero.example', 8788);
  assert.match(nginx, /server_name comfy\.bero\.example;/);
  assert.match(nginx, /proxy_pass http:\/\/127\.0\.0\.1:8788;/);
  assert.match(nginx, /proxy_set_header Host \$http_host;/);
  assert.match(nginx, /proxy_buffering off;/);
  assert.match(nginx, /proxy_set_header Upgrade \$http_upgrade;/);

  const caddy = mod.buildCaddyProxyConfig('https://comfy.bero.example', 9001);
  assert.match(caddy, /^comfy\.bero\.example \{/);
  assert.match(caddy, /reverse_proxy 127\.0\.0\.1:9001/);
  assert.match(caddy, /flush_interval -1/);
});
