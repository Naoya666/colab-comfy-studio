"""Render monochrome light and dark README architecture diagrams via Playwright.

Usage:
    uv run python scripts/render_architecture.py
"""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "images"

HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN" data-theme="__THEME__">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {
    --canvas: #0a0a0a;
    --dot: rgba(255, 255, 255, 0.065);
    --surface-1: #111111;
    --surface-2: #181818;
    --surface-3: #222222;
    --line: rgba(255, 255, 255, 0.13);
    --line-strong: rgba(255, 255, 255, 0.28);
    --line-dashed: rgba(255, 255, 255, 0.20);
    --text: #ffffff;
    --text-2: #c6c6c6;
    --muted: #8c8c8c;
    --primary: #ffffff;
    --on-primary: #0a0a0a;
    --accent: #83cdb4;
    --wire: #e5e5e5;
    --wire-bus: #7a7a7a;
    --pill-bg: #121212;
    --pill-border: rgba(255, 255, 255, 0.22);
    --pill-text: #ececec;
    --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.45);
  }

  :root[data-theme="light"] {
    --canvas: #ffffff;
    --dot: rgba(0, 0, 0, 0.065);
    --surface-1: #f6f6f6;
    --surface-2: #ffffff;
    --surface-3: #efefef;
    --line: rgba(0, 0, 0, 0.12);
    --line-strong: rgba(0, 0, 0, 0.28);
    --line-dashed: rgba(0, 0, 0, 0.18);
    --text: #000000;
    --text-2: #3a3a3a;
    --muted: #6e6e6e;
    --primary: #000000;
    --on-primary: #ffffff;
    --accent: #1b634c;
    --wire: #141414;
    --wire-bus: #8a8a8a;
    --pill-bg: #ffffff;
    --pill-border: rgba(0, 0, 0, 0.18);
    --pill-text: #1c1c1c;
    --shadow-card: 0 1px 2px rgba(0, 0, 0, 0.04);
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  html, body {
    width: 1420px;
    height: 660px;
    background: transparent;
    font-family: "Noto Sans CJK SC", "Noto Sans SC", "PingFang SC", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
    text-rendering: geometricPrecision;
  }

  .frame {
    position: relative;
    width: 1420px;
    height: 660px;
    border-radius: 20px;
    border: 1px solid var(--line);
    background-color: var(--canvas);
    background-image: radial-gradient(var(--dot) 1px, transparent 1px);
    background-size: 20px 20px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* ── Top Header ── */
  .topbar {
    height: 58px;
    padding: 0 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--canvas);
    border-bottom: 1px solid var(--line);
    flex: none;
    z-index: 2;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .brand-mark {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    border: 1px solid var(--line-strong);
    background: var(--surface-2);
    display: grid;
    place-items: center;
    color: var(--accent);
  }

  .brand-title {
    display: flex;
    align-items: baseline;
    gap: 10px;
  }

  .brand-title strong {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text);
  }

  .brand-title .sep {
    color: var(--line-strong);
    font-weight: 400;
  }

  .brand-title span {
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
  }

  .tiers {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .tier-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 26px;
    padding: 0 11px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--surface-1);
    font-size: 11.5px;
    font-weight: 500;
    color: var(--text-2);
  }

  .tier-chip b {
    font-family: "Noto Sans Mono CJK SC", ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 10.5px;
    font-weight: 700;
    color: var(--muted);
  }

  /* ── Stage ── */
  .stage {
    position: relative;
    flex: 1;
    width: 100%;
    height: 602px;
  }

  .wires {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 4;
  }

  .pill-layer {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 5;
  }

  .edge-pill {
    position: absolute;
    transform: translate(-50%, -50%);
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid var(--pill-border);
    background: var(--pill-bg);
    color: var(--pill-text);
    font-size: 11px;
    font-weight: 600;
    line-height: 1.35;
    white-space: nowrap;
    text-align: center;
    letter-spacing: -0.005em;
    box-shadow: var(--shadow-card);
  }

  .edge-pill.multiline {
    border-radius: 12px;
    padding: 5px 13px;
    line-height: 1.4;
  }

  /* ── Cards & Containers ── */
  .card {
    position: absolute;
    border-radius: 14px;
    border: 1px solid var(--line);
    background: var(--surface-2);
    box-shadow: var(--shadow-card);
    z-index: 2;
  }

  .card.strong-border {
    border-color: var(--line-strong);
  }

  .host-group {
    position: absolute;
    left: 396px;
    top: 24px;
    width: 472px;
    height: 552px;
    border-radius: 18px;
    border: 1.5px dashed var(--line-dashed);
    background: var(--surface-1);
    z-index: 1;
  }

  .host-header {
    height: 46px;
    padding: 0 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
  }

  .host-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13.5px;
    font-weight: 700;
    color: var(--text);
  }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 22px;
    padding: 0 9px;
    border-radius: 999px;
    border: 1px solid var(--line-strong);
    background: transparent;
    color: var(--text-2);
    font-size: 10.5px;
    font-weight: 600;
    white-space: nowrap;
    flex: none;
  }

  .pill.solid {
    background: var(--primary);
    border-color: var(--primary);
    color: var(--on-primary);
  }

  .pill.mono {
    font-family: "Noto Sans Mono CJK SC", ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 10.5px;
    font-weight: 600;
  }

  .pill.xs {
    height: 19px;
    padding: 0 7px;
    font-size: 10px;
  }

  .icon-box {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: var(--surface-3);
    display: grid;
    place-items: center;
    color: var(--text);
    flex: none;
  }

  .icon-box.dark {
    background: var(--primary);
    border-color: var(--primary);
    color: var(--on-primary);
  }

  /* ── Left Column: Client + Legend ── */
  #node-client {
    left: 28px;
    top: 86px;
    width: 222px;
    height: 286px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 13px;
  }

  .client-head {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .client-head h2 {
    font-size: 14px;
    font-weight: 700;
    line-height: 1.25;
  }

  .client-head small {
    display: block;
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
  }

  .client-list {
    display: flex;
    flex-direction: column;
    gap: 9px;
    margin-top: auto;
  }

  .client-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 11px;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: var(--surface-1);
  }

  .client-item svg {
    color: var(--text);
    flex: none;
  }

  .client-item-text {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .client-item-text strong {
    font-size: 12.5px;
    font-weight: 650;
    color: var(--text);
    line-height: 1.25;
  }

  .client-item-text span {
    font-size: 10.5px;
    color: var(--muted);
    line-height: 1.3;
    margin-top: 2px;
    white-space: nowrap;
  }

  #node-legend {
    left: 28px;
    top: 444px;
    width: 222px;
    height: 114px;
    padding: 13px 15px;
    background: var(--surface-1);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .legend-title {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--muted);
  }

  .legend-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 11.5px;
    color: var(--text-2);
    font-weight: 500;
  }

  .legend-swatch {
    width: 30px;
    height: 14px;
    flex: none;
  }

  /* ── Host Children ── */
  #node-api {
    left: 414px;
    top: 86px;
    width: 436px;
    height: 82px;
    padding: 0 15px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  #node-worker {
    left: 572px;
    top: 188px;
    width: 278px;
    height: 82px;
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 11px;
  }

  #node-watchdog {
    left: 572px;
    top: 290px;
    width: 278px;
    height: 82px;
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 11px;
  }

  .host-lane-sep {
    position: absolute;
    left: 572px;
    top: 404px;
    width: 278px;
    display: flex;
    align-items: center;
    gap: 9px;
    z-index: 2;
  }

  .host-lane-sep::before,
  .host-lane-sep::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--line);
  }

  .host-lane-sep span {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.03em;
    white-space: nowrap;
  }

  #node-model-worker {
    left: 572px;
    top: 462px;
    width: 278px;
    height: 96px;
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 11px;
  }

  .svc-body {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .svc-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }

  .svc-name {
    font-size: 13.5px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.25;
    white-space: nowrap;
  }

  .svc-desc {
    font-size: 11.5px;
    color: var(--muted);
    line-height: 1.35;
    white-space: nowrap;
  }

  .svc-inline-meta {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 1px;
  }

  #node-db {
    left: 414px;
    top: 188px;
    width: 134px;
    height: 370px;
    padding: 18px 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 12px;
  }

  .db-cylinder {
    width: 48px;
    height: 48px;
    border-radius: 13px;
    border: 1px solid var(--line-strong);
    background: var(--surface-3);
    display: grid;
    place-items: center;
    color: var(--text);
  }

  .db-title {
    font-size: 13px;
    font-weight: 700;
    line-height: 1.4;
    color: var(--text);
  }

  .db-sub {
    font-size: 11px;
    line-height: 1.45;
    color: var(--muted);
  }

  .db-divider {
    width: 44px;
    height: 1px;
    background: var(--line);
  }

  .db-meta {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }

  /* ── Right Column: Colab & Drive ── */
  #node-colab {
    left: 1124px;
    top: 86px;
    width: 268px;
    height: 286px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .colab-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }

  .colab-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 26px;
    padding: 0 11px;
    border-radius: 999px;
    background: var(--primary);
    color: var(--on-primary);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: -0.01em;
  }

  .colab-title {
    font-size: 15px;
    font-weight: 750;
    line-height: 1.3;
    color: var(--text);
    letter-spacing: -0.01em;
  }

  .colab-sub {
    font-size: 11.5px;
    color: var(--muted);
    line-height: 1.45;
    margin-top: -3px;
  }

  .colab-specs {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: auto;
  }

  .colab-spec-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 8.5px 11px;
    border-radius: 9px;
    border: 1px solid var(--line);
    background: var(--surface-1);
    font-size: 11.5px;
  }

  .colab-spec-row strong {
    font-weight: 650;
    color: var(--text);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    flex: none;
  }

  .colab-spec-row span {
    font-size: 10.5px;
    color: var(--muted);
    white-space: nowrap;
    font-family: "Noto Sans Mono CJK SC", ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  #node-drive {
    left: 1124px;
    top: 462px;
    width: 268px;
    height: 96px;
    padding: 15px 16px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 8px;
  }

  .drive-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .drive-head-left {
    display: flex;
    align-items: center;
    gap: 9px;
  }

  .drive-head strong {
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
  }

  .drive-desc {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.45;
  }
</style>
</head>
<body>
  <div class="frame" id="frame">
    <!-- Top Header -->
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M13.73 4a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
          </svg>
        </div>
        <div class="brand-title">
          <strong>Colab GPU Studio</strong>
          <span class="sep">/</span>
          <span>系统架构与数据流</span>
        </div>
      </div>
      <div class="tiers">
        <span class="tier-chip"><b>01</b> 客户端</span>
        <span class="tier-chip"><b>02</b> 本机 / VPS 控制面</span>
        <span class="tier-chip"><b>03</b> Google 云端算力与存储</span>
      </div>
    </header>

    <!-- Main Stage -->
    <main class="stage" id="stage">
      <!-- Host Group Container -->
      <section class="host-group" id="group-host">
        <div class="host-header">
          <div class="host-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <rect width="20" height="8" x="2" y="2" rx="2" ry="2"/>
              <rect width="20" height="8" x="2" y="14" rx="2" ry="2"/>
              <line x1="6" x2="6.01" y1="6" y2="6"/>
              <line x1="6" x2="6.01" y1="18" y2="18"/>
            </svg>
            <span>本机 / VPS</span>
            <span class="pill solid">Docker Compose</span>
          </div>
          <span class="pill mono">4 容器 · 共享卷</span>
        </div>
      </section>

      <!-- Left Column: Client -->
      <article class="card strong-border" id="node-client">
        <div class="client-head">
          <div class="icon-box dark">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <rect width="20" height="14" x="2" y="3" rx="2"/>
              <line x1="8" x2="16" y1="21" y2="21"/>
              <line x1="12" x2="12" y1="17" y2="21"/>
            </svg>
          </div>
          <div>
            <h2>客户端</h2>
            <small>多端控制与 API 接入</small>
          </div>
        </div>
        <div class="client-list">
          <div class="client-item">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
              <path d="M2 12h20"/>
            </svg>
            <div class="client-item-text">
              <strong>浏览器</strong>
              <span>桌面 / 手机 Web 工作台</span>
            </div>
          </div>
          <div class="client-item">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
              <rect width="14" height="20" x="5" y="2" rx="2" ry="2"/>
              <path d="M12 18h.01"/>
            </svg>
            <div class="client-item-text">
              <strong>PWA</strong>
              <span>独立安装 · 响应式触控</span>
            </div>
          </div>
          <div class="client-item">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="16 18 22 12 16 6"/>
              <polyline points="8 6 2 12 8 18"/>
            </svg>
            <div class="client-item-text">
              <strong>SillyTavern</strong>
              <span>ComfyUI 兼容 API 接入</span>
            </div>
          </div>
        </div>
      </article>

      <!-- Left Bottom: Legend -->
      <aside class="card" id="node-legend">
        <div class="legend-title">连线图例</div>
        <div class="legend-row">
          <svg class="legend-swatch" viewBox="0 0 30 14">
            <line x1="1" y1="7" x2="23" y2="7" stroke="var(--wire)" stroke-width="1.6"/>
            <polygon points="29,7 22,3.5 22,10.5" fill="var(--wire)"/>
          </svg>
          <span>控制 / 代理 / 流式上传</span>
        </div>
        <div class="legend-row">
          <svg class="legend-swatch" viewBox="0 0 30 14">
            <polygon points="1,7 7.5,3.5 7.5,10.5" fill="var(--wire)"/>
            <line x1="7" y1="7" x2="23" y2="7" stroke="var(--wire)" stroke-width="1.6"/>
            <polygon points="29,7 22.5,3.5 22.5,10.5" fill="var(--wire)"/>
          </svg>
          <span>云端双向挂载与同步</span>
        </div>
        <div class="legend-row">
          <svg class="legend-swatch" viewBox="0 0 30 14">
            <circle cx="4" cy="7" r="2.6" fill="var(--wire-bus)"/>
            <line x1="6" y1="7" x2="24" y2="7" stroke="var(--wire-bus)" stroke-width="1.5"/>
            <circle cx="26" cy="7" r="2.6" fill="var(--wire-bus)"/>
          </svg>
          <span>共享数据卷读写</span>
        </div>
      </aside>

      <!-- Host Service 1: Studio API -->
      <article class="card strong-border" id="node-api">
        <div class="icon-box dark">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2"/>
            <polyline points="2 17 12 22 22 17"/>
            <polyline points="2 12 12 17 22 12"/>
          </svg>
        </div>
        <div class="svc-body">
          <div class="svc-top">
            <span class="svc-name">Studio API</span>
            <div style="display:flex;gap:5px;">
              <span class="pill">FastAPI + React</span>
              <span class="pill mono">api</span>
            </div>
          </div>
          <div class="svc-desc">前端静态资源、创作 / 原生 ComfyUI HTTP &amp; WebSocket 网关</div>
        </div>
      </article>

      <!-- Host Shared DB -->
      <article class="card" id="node-db">
        <div class="db-cylinder">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
            <ellipse cx="12" cy="5" rx="9" ry="3"/>
            <path d="M3 5v14a9 3 0 0 0 18 0V5"/>
            <path d="M3 12a9 3 0 0 0 18 0"/>
          </svg>
        </div>
        <div class="db-title">SQLite /<br/>PostgreSQL</div>
        <div class="db-sub">+ 本地产物与队列状态</div>
        <div class="db-divider"></div>
        <div class="db-meta">
          <span class="pill mono">studio-data</span>
          <span class="db-sub">共享持久化存储</span>
        </div>
      </article>

      <!-- Host Service 2: Task Worker -->
      <article class="card" id="node-worker">
        <div class="icon-box">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
          </svg>
        </div>
        <div class="svc-body">
          <div class="svc-top">
            <span class="svc-name">任务 Worker</span>
            <span class="pill mono">worker</span>
          </div>
          <div class="svc-desc">创作、批次、冒烟与原生调度</div>
        </div>
      </article>

      <!-- Host Service 3: Watchdog -->
      <article class="card" id="node-watchdog">
        <div class="icon-box">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </div>
        <div class="svc-body">
          <div class="svc-top">
            <span class="svc-name">回收与保活 Watchdog</span>
            <span class="pill mono xs">watchdog</span>
          </div>
          <div class="svc-desc">租约巡检、CU 账本与两道熔断</div>
        </div>
      </article>

      <!-- Subtle separation between GPU orchestration lane and CPU-only model preparation -->
      <div class="host-lane-sep"><span>免 GPU 独立模型准备通道</span></div>

      <!-- Host Service 4: Model Worker -->
      <article class="card" id="node-model-worker">
        <div class="icon-box">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" x2="12" y1="15" y2="3"/>
          </svg>
        </div>
        <div class="svc-body">
          <div class="svc-top">
            <span class="svc-name">模型下载 Worker</span>
            <span class="pill mono xs">model-worker</span>
          </div>
          <div class="svc-inline-meta">
            <span class="pill solid xs">CPU-only</span>
            <span class="svc-desc">断点续传与哈希校验</span>
          </div>
        </div>
      </article>

      <!-- Right Top: Google Colab GPU Runtime -->
      <article class="card strong-border" id="node-colab">
        <div class="colab-top">
          <span class="colab-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
            </svg>
            Colab
          </span>
          <span class="pill mono">按需 GPU</span>
        </div>
        <div class="colab-title">Google Colab GPU 运行时</div>
        <div class="colab-sub">按需分配执行真实图像 / 视频推理与原生 ComfyUI 节点编辑</div>
        <div class="colab-specs">
          <div class="colab-spec-row">
            <strong>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect width="16" height="16" x="4" y="4" rx="2"/>
                <rect width="6" height="6" x="9" y="9" rx="1"/>
                <path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>
              </svg>
              三域隔离
            </strong>
            <span>原生 / 创作 / 一次性</span>
          </div>
          <div class="colab-spec-row">
            <strong>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 20V10M18 20V4M6 20v-4"/>
              </svg>
              GPU 档位
            </strong>
            <span>T4 · L4 · A100 · H100</span>
          </div>
          <div class="colab-spec-row">
            <strong>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
              </svg>
              零暴露代理
            </strong>
            <span>CF Tunnel + Access</span>
          </div>
        </div>
      </article>

      <!-- Right Bottom: Google Drive -->
      <article class="card strong-border" id="node-drive">
        <div class="drive-head">
          <div class="drive-head-left">
            <div class="icon-box">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <line x1="22" x2="2" y1="12" y2="12"/>
                <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>
                <line x1="6" x2="6.01" y1="16" y2="16"/>
                <line x1="10" x2="10.01" y1="16" y2="16"/>
              </svg>
            </div>
            <strong>Google Drive</strong>
          </div>
          <span class="pill mono">drive.file</span>
        </div>
        <div class="drive-desc">统一模型库 · 自定义节点 · 环境缓存与输出</div>
      </article>

      <!-- SVG Wires & Pill Overlays (computed from exact measured DOM coordinates) -->
      <svg class="wires" id="wires" viewBox="0 0 1420 602" xmlns="http://www.w3.org/2000/svg"></svg>
      <div class="pill-layer" id="pill-layer"></div>
    </main>
  </div>

<script>
(() => {
  const stage = document.getElementById("stage");
  const svg = document.getElementById("wires");
  const pillLayer = document.getElementById("pill-layer");
  const stageRect = stage.getBoundingClientRect();

  const box = (id) => {
    const r = document.getElementById(id).getBoundingClientRect();
    return {
      left: r.left - stageRect.left,
      right: r.right - stageRect.left,
      top: r.top - stageRect.top,
      bottom: r.bottom - stageRect.top,
      cx: (r.left + r.right) / 2 - stageRect.left,
      cy: (r.top + r.bottom) / 2 - stageRect.top,
    };
  };

  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
  };

  const addPill = (x, y, html, extraClass = "") => {
    const div = document.createElement("div");
    div.className = `edge-pill ${extraClass}`.trim();
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;
    div.innerHTML = html;
    pillLayer.appendChild(div);
  };

  const drawArrowH = (x1, x2, y, label, labelX) => {
    const arrowLen = 7.5;
    svg.appendChild(el("line", {
      x1, y1: y, x2: x2 - arrowLen, y2: y,
      stroke: "var(--wire)", "stroke-width": 1.65,
    }));
    svg.appendChild(el("circle", {
      cx: x1, cy: y, r: 2.8, fill: "var(--wire)",
    }));
    svg.appendChild(el("polygon", {
      points: `${x2},${y} ${x2 - arrowLen - 1},${y - 4.2} ${x2 - arrowLen - 1},${y + 4.2}`,
      fill: "var(--wire)",
    }));
    if (label) addPill(labelX, y, label);
  };

  const drawBidirectionalV = (x, y1, y2, labelHtml) => {
    const arrowLen = 7.5;
    svg.appendChild(el("line", {
      x1: x, y1: y1 + arrowLen, x2: x, y2: y2 - arrowLen,
      stroke: "var(--wire)", "stroke-width": 1.65,
    }));
    svg.appendChild(el("polygon", {
      points: `${x},${y1} ${x - 4.2},${y1 + arrowLen + 1} ${x + 4.2},${y1 + arrowLen + 1}`,
      fill: "var(--wire)",
    }));
    svg.appendChild(el("polygon", {
      points: `${x},${y2} ${x - 4.2},${y2 - arrowLen - 1} ${x + 4.2},${y2 - arrowLen - 1}`,
      fill: "var(--wire)",
    }));
    addPill(x, (y1 + y2) / 2, labelHtml, "multiline");
  };

  const drawBus = (x1, y1, x2, y2) => {
    svg.appendChild(el("line", {
      x1, y1, x2, y2,
      stroke: "var(--wire-bus)", "stroke-width": 1.5,
    }));
    svg.appendChild(el("circle", { cx: x1, cy: y1, r: 2.8, fill: "var(--wire-bus)" }));
    svg.appendChild(el("circle", { cx: x2, cy: y2, r: 2.8, fill: "var(--wire-bus)" }));
  };

  const host = box("group-host");
  const client = box("node-client");
  const api = box("node-api");
  const db = box("node-db");
  const worker = box("node-worker");
  const watchdog = box("node-watchdog");
  const modelWorker = box("node-model-worker");
  const colab = box("node-colab");
  const drive = box("node-drive");

  const leftGapCenter = (client.right + host.left) / 2;
  const rightGapCenter = (host.right + colab.left) / 2;

  // 1. Client -> Studio API
  drawArrowH(client.right, api.left, api.cy, "HTTPS / WebSocket", leftGapCenter);

  // 2. Host Internal DB Bus (4 containers <-> shared DB)
  drawBus(db.cx, api.bottom, db.cx, db.top);
  drawBus(db.right, worker.cy, worker.left, worker.cy);
  drawBus(db.right, watchdog.cy, watchdog.left, watchdog.cy);
  drawBus(db.right, modelWorker.cy, modelWorker.left, modelWorker.cy);

  // 3. Host -> Google Colab (3 parallel clean channels)
  drawArrowH(api.right, colab.left, api.cy, "Cloudflare Named Tunnel + Access", rightGapCenter);
  drawArrowH(worker.right, colab.left, worker.cy, "Colab MCP 控制通道", rightGapCenter);
  drawArrowH(watchdog.right, colab.left, watchdog.cy, "心跳保活 / 空闲与熔断回收", rightGapCenter);

  // 4. Model Worker -> Google Drive
  drawArrowH(modelWorker.right, drive.left, modelWorker.cy, "免 GPU 流式上传权重", rightGapCenter);

  // 5. Google Colab <-> Google Drive (bidirectional vertical sync)
  drawBidirectionalV(
    colab.cx,
    colab.bottom,
    drive.top,
    "rclone VFS / 节点与环境缓存<br/>输出同步",
  );
})();
</script>
</body>
</html>
"""


def _find_chromium() -> str | None:
    if explicit := os.environ.get("OFFLINE_CHROMIUM"):
        return explicit
    cache_root = Path.home() / ".cache" / "ms-playwright"
    if cache_root.is_dir():
        for pattern in ("chromium-*/chrome-linux64/chrome", "chromium-*/chrome-linux/chrome"):
            matches = sorted(cache_root.glob(pattern), reverse=True)
            if matches:
                return str(matches[0])
    return None


def render_diagrams() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=_find_chromium(),
            args=["--no-sandbox"],
        )
        try:
            for theme in ("light", "dark"):
                context = browser.new_context(
                    viewport={"width": 1420, "height": 660},
                    device_scale_factor=2,
                )
                page = context.new_page()
                page.set_content(
                    HTML_TEMPLATE.replace("__THEME__", theme),
                    wait_until="load",
                )
                page.evaluate("() => document.fonts.ready")
                out_path = OUT_DIR / f"architecture-{theme}.png"
                page.locator("#frame").screenshot(
                    path=out_path,
                    omit_background=True,
                )
                outputs.append(out_path)
                context.close()
        finally:
            browser.close()
    return outputs


if __name__ == "__main__":
    for path in render_diagrams():
        print(f"Rendered {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
