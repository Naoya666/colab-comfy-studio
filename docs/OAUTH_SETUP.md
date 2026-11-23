# Google Colab 与 Drive 授权配置指南

本项目不会内置任何人的 Google 账号或令牌。为了同时兼顾 **「纯自用账号」** 与 **「第三方交付 CU（Colab CLI）+ 自有 Drive」** 两种场景，Studio 将 **Google Colab 算力授权** 与 **Google Drive 存储授权** 明确拆分管理：

- **Google Colab 算力**：推荐使用网页内置的 **Colab CLI 授权（推荐）**，免开 GCP Colaboratory API、免安装命令行、自动处理最小权限与加密入库；同时保留自建 Web OAuth 入口（**实验性**）。
- **Google Drive 存储**：统一通过你自己的 **Google Web OAuth 应用** 授权（仅需启用 Google Drive API，申请最小权限 `drive.file`）。多个 Colab 算力号可以通过「Drive 存储来源」共用同一个 Drive 主账号。

> **为什么即使纯自用也建议 Drive 独立使用自己的 Web OAuth 应用？**
> Google Drive 的 `drive.file` 权限是按 **OAuth Client ID 隔离可见性** 的——不同客户端创建的文件互不可见。将模型库、自定义节点与原生输出固定在自己可控的 Web OAuth 应用下，以后无论切换或增删多少个 Colab 算力号，都永远不会丢失已有模型与缓存。

---

## 第一步：连接 Google Colab 算力

### 方式 A：网页内置 Colab CLI 授权（推荐 · 已完整实测）

官方 Docker 镜像与 Python 3.12+ 运行环境已内置 `google-colab-cli` 客户端定义（源码不硬编码任何密钥）。在网页端 2 步即可完成授权：

1. 登录 Studio → 进入 **设置 → Google 账号**，点击 **新增账号**（例如命名 `Colab算力号` 或 `主账号`）。
2. 在该账号卡片中点击 **「Colab CLI 授权（推荐）」** 打开向导，系统会自动生成一条 30 分钟有效的 Google 授权链接（刷新或切页不会丢失会话）：
   - **纯自用**：点击 **「新窗口打开」**，登录自己的 Google 账号并确认授权，复制页面显示的 `4/0A...` 授权码。
   - **第三方 / 异地交付 CU**：点击 **「复制授权链接」** 发送给交付方，等待对方在浏览器完成登录并将 `4/0A...` 授权码发回给你。
3. 将授权码粘贴回向导第 2 步的输入框，点击 **「提交授权码并连接」**。
4. 后端会直接换取刷新令牌、使用 `.env` 中的 `ENCRYPTION_KEY` 加密存库，并自动探测一次实时 CU 余额与可用 GPU 规格。

**相比手工在宿主机终端运行 `google-colab-cli` 的优势**：
- **零环境折腾**：无需手动安装 `uv`、`google-colab-cli` 或 `keyrings.alt`，不产生宿主机明文 keyring 文件；
- **零 `Scope has changed` 报错**：向导从源头仅请求 Colab 最小必要权限（`openid`、`userinfo.profile`、`userinfo.email`、`colaboratory`），**不请求** `drive.file` 与 `cloud-platform`，交付方无需手动取消勾选、更安心；
- **输错可重试、不断线**：PKCE 会话持久化在数据库 30 分钟，手滑贴错授权码可直接修正重试，不必反复麻烦对方重开链接；
- **支持复用已有凭据**：若本机或别处已有 `google-colab-cli` 导出的凭据 JSON（含 `refresh_token`），可在向导底部展开 **「已有现成 Colab CLI 凭据 JSON 或 Refresh Token？直接导入」** 校验入库。

### 方式 B：自建 Web OAuth 授权 Colab（实验性）

如果你已经拥有并开通了 Google Cloud 的 **Colaboratory API**，也可以在完成下方「第二步」后，在账号卡片中点击 **「Web OAuth 授权（实验性）」** 跳转授权。

> 当前原生 ComfyUI 双层保活、实时 CU 消耗速率与 4 MiB 单 WebSocket 流式产物回传均在 **Colab CLI 通道** 上完成了完整真机验收；Web OAuth（`v1beta` REST + MCP）通道代码已保留但仍在持续实测中，如遇保活或余额采样异常，建议优先切换为 **Colab CLI 授权（推荐）**。

---

## 第二步：配置 Google Drive 授权（模型库与持久化必需）

只需要在一个属于你自己的 Google 账号上完成一次 Drive 授权，其余所有 Colab 算力号都可以直接复用该 Drive。

### 1. 在 Google Cloud 创建 Web OAuth 客户端 JSON

1. 打开 [Google Cloud Console](https://console.cloud.google.com/)，新建或选择一个个人项目。
2. 在“API 和服务 → 库”中，搜索并启用 **Google Drive API**（只有在使用「方式 B：Web OAuth 授权 Colab（实验性）」时才需要同时启用 **Colaboratory API**）。
3. 进入“Google Auth Platform / OAuth 同意屏幕”：
   - 受众选“外部”；若处于“测试（Testing）”模式，务必把自己的 Google 邮箱加入 **测试用户（Test users）** 列表。
   - 在数据访问权限（Scopes）中添加：
     - `https://www.googleapis.com/auth/drive.file`（仅允许访问本站创建的缓存目录与作品，不扫描网盘中其他文件）
     - `https://www.googleapis.com/auth/colaboratory`（仅方式 B 实验性 Web OAuth 连接 Colab 时需要）
4. 进入“客户端（Clients）”创建凭据，类型必须选 **Web 应用（Web application）**：
   - **已获授权的 JavaScript 来源**：填你的网站地址，例如 `https://studio.example.com`（本地测试填 `http://127.0.0.1:8788`）
   - **已获授权的重定向 URI**：必须精确填写 `https://studio.example.com/api/gpu/auth/google/callback`（把域名换成你实际配置的 `SITE_URL`）
5. 创建完成后，下载客户端 **JSON 文件**。

### 2. 在网页导入 JSON 并授权 Drive

1. 登录 Studio → **设置 → Google 账号**，展开底部的 **「Google Web OAuth 应用配置（Drive 必填 / Colab 实验性）」**。
2. 上传刚才下载的 Web 客户端 JSON，勾选确认并保存（Client Secret 会用 `.env` 中的 `ENCRYPTION_KEY` 加密存库，前端不回显；保存或更换 Web 客户端不会影响已授权的 Colab CLI 算力凭据）。
3. 在账号卡片中点击 **「授权独立 Drive」**，跳转到 Google 完成授权。
4. 如果你的 Colab 算力号是另一个账号，在算力号卡片的 **「Drive 存储来源」** 下拉框中选择 **「复用：<你的 Drive 主账号>」** 即可（也可在 Colab CLI 授权完成页一键绑定）。

> **备用方式**：也可将 `GPU_GOOGLE_CLIENT_ID`、`GPU_GOOGLE_CLIENT_SECRET` 直接写入 `.env` 再重启容器。

---

## 第三步：开启真实 GPU 与 CU 熔断保护

授权完成后，默认仍保持关闭真实 GPU 防止误操作。确认准备就绪后，编辑服务器的 `.env`：

```ini
GPU_REAL_ENABLED=1
GPU_STORAGE=drive
STUDIO_TIMEZONE=Asia/Shanghai
```

重启服务后：
1. 进入 **算力** 页，点击 **检查可用 GPU** 确认可申请的显卡规格；
2. 在 **CU 熔断保护** 中按需设置 **今日总上限（温和熔断）** 与 **余额底线（强制熔断）**（今日扣费按工作台本地自然日切日，配置持久化至数据库）；
3. 随后即可运行 CUDA 冒烟测试、启用内置工作流或启动原生 ComfyUI。

---

## 安全与运维提示

- **解绑/删除 Colab CLI 账号不会误作废远端令牌**：在账号卡片点击「解除授权」或「删除账号」时，对于 Colab CLI 凭据，**「同时向 Google 远端吊销令牌」默认不勾选**（仅清除本站加密记录），避免误将第三方交付或本机仍在复用的 CLI 凭据在 Google 侧吊销。
- **本机非 Docker 部署要求**：网页内置的 Colab CLI 授权向导依赖 `google-colab-cli`（需 **Python 3.12+**）；使用官方 Docker 镜像（`python:3.12-slim`）或 `uv sync`（Python 3.12）即可开箱即用。如在 Python 3.11 环境运行，仍可通过向导底部的「直接导入 CLI 凭据 JSON」或设置 `GPU_COLAB_CLI_CLIENT_ID` / `GPU_COLAB_CLI_CLIENT_SECRET` 环境变量启用。
