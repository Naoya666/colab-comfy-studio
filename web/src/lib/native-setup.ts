export const RECOMMENDED_COMFY_COMMIT = '423cbd20511becb1f16122ab518e92e58bfa03d9';

export type TunnelTokenPreview = {
  extracted: string;
  validCfFormat: boolean;
  tunnelId: string | null;
  accountIdMasked: string | null;
};

const TOKEN_CMD_RE = /(?:--token(?:\s+|=)|service\s+install\s+)['"]?([A-Za-z0-9+/=_-]{16,4096})['"]?/;
const COMMIT_URL_RE = /github\.com\/Comfy-Org\/ComfyUI\/commit\/([0-9a-fA-F]{40})(?:[/?#].*)?$/;
const COMMIT_HEX_RE = /^[0-9a-f]{40}$/;

function decodeBase64Json(token: string): Record<string, unknown> | null {
  const normalized = token.replace(/-/g, '+').replace(/_/g, '/');
  const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4);
  try {
    const raw = atob(padded);
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Ignore non-JSON tokens
  }
  return null;
}

export function extractTunnelToken(input: string): string {
  const text = (input || '').trim();
  if (!text) return '';
  const match = TOKEN_CMD_RE.exec(text);
  if (match && match[1]) return match[1].trim();
  return text.replace(/^['"]|['"]$/g, '').trim();
}

export function previewTunnelToken(input: string): TunnelTokenPreview {
  const extracted = extractTunnelToken(input);
  if (!extracted) {
    return { extracted: '', validCfFormat: false, tunnelId: null, accountIdMasked: null };
  }
  const parsed = decodeBase64Json(extracted);
  if (
    parsed
    && typeof parsed.a === 'string' && parsed.a
    && typeof parsed.t === 'string' && parsed.t
    && typeof parsed.s === 'string' && parsed.s
  ) {
    const acc = parsed.a;
    const masked = acc.length > 12 ? `${acc.slice(0, 6)}…${acc.slice(-4)}` : `${acc.slice(0, 4)}…`;
    return { extracted, validCfFormat: true, tunnelId: parsed.t, accountIdMasked: masked };
  }
  return { extracted, validCfFormat: false, tunnelId: null, accountIdMasked: null };
}

export function extractComfyCommit(input: string): string {
  const text = (input || '').trim();
  if (!text) return '';
  const match = COMMIT_URL_RE.exec(text);
  return (match ? match[1] : text).toLowerCase();
}

export function isValidComfyCommit(input: string): boolean {
  return COMMIT_HEX_RE.test(extractComfyCommit(input));
}

export function parseAccessJsonInput(input: string): { clientId: string; clientSecret: string } | null {
  try {
    const parsed: unknown = JSON.parse(input);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    const record = parsed as Record<string, unknown>;
    const clientId = typeof record.client_id === 'string' ? record.client_id.trim() : '';
    const clientSecret = typeof record.client_secret === 'string' ? record.client_secret.trim() : '';
    if (!clientId || !clientSecret) return null;
    return { clientId, clientSecret };
  } catch {
    return null;
  }
}

export function extractHostFromUrl(rawUrl: string, fallback = 'comfy.example.com'): string {
  const text = (rawUrl || '').trim();
  if (!text) return fallback;
  try {
    const url = new URL(text.includes('://') ? text : `https://${text}`);
    return url.host || fallback;
  } catch {
    return fallback;
  }
}

export function buildNginxProxyConfig(nativeComfyUrl: string, bindPort = 8788): string {
  const host = extractHostFromUrl(nativeComfyUrl, 'comfy.example.com');
  const bareHost = host.split(':')[0] || 'comfy.example.com';
  return `# 放到 http {} 上下文中（若已有同名 map 可复用）
map $http_upgrade $native_connection_upgrade {
    default upgrade;
    '' close;
}

server {
    listen 443 ssl http2;
    server_name ${bareHost};
    ssl_certificate /etc/letsencrypt/live/${bareHost}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${bareHost}/privkey.pem;
    client_max_body_size 0;

    location / {
        proxy_pass http://127.0.0.1:${bindPort};
        proxy_set_header Host $http_host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $native_connection_upgrade;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}`;
}

export function buildCaddyProxyConfig(nativeComfyUrl: string, bindPort = 8788): string {
  const host = extractHostFromUrl(nativeComfyUrl, 'comfy.example.com');
  return `${host} {
    reverse_proxy 127.0.0.1:${bindPort} {
        header_up Host {http.request.host}
        flush_interval -1
        transport http {
            read_timeout 3600s
            write_timeout 3600s
        }
    }
}`;
}

export function sourceLabel(source: string | undefined): string {
  switch (source) {
    case 'encrypted_database':
      return '网页加密存储';
    case 'environment':
      return '来自 .env / 文件';
    case 'error':
      return '解密异常';
    default:
      return '未配置';
  }
}
