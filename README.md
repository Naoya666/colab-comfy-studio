# Colab GPU Studio

Colab GPU Studio 是一个自托管的 ComfyUI 创作工作台。你可以在手机或电脑上填写工作流参数、提交图片和视频任务、查看进度并下载作品。Web 服务和任务队列运行在本机或 VPS 上，真实生成由 Google Colab GPU 执行。

新部署默认关闭真实 GPU。启动后可用 `demo-mock` 体验提交任务和查看作品的流程。接入 Google 账号后，可按需申请算力、设置预算，并查看实际分配的 GPU 与实例状态。

| 手机创作页 | 桌面创作页 |
| --- | --- |
| <img src="docs/demo/screenshots/mobile-create.png" width="240" alt="手机上的创作页面"> | <img src="docs/demo/screenshots/desktop-create.png" width="640" alt="桌面上的创作页面"> |

截图来自原项目的演示版本。[演示说明与更多截图](docs/DEMO.md)

## 快速开始

### 使用 Docker Compose

需要 Git、Docker、Docker Compose 和 `uv`。

```bash
git clone https://github.com/Naoya666/colab-comfy-study.git
cd colab-comfy-study
uv run python scripts/init_env.py --site-url http://127.0.0.1:8788
cp docker-compose.example.yml docker-compose.yml
docker compose up -d --build
```

打开 `http://127.0.0.1:8788`，使用 `.env` 中的 `STUDIO_PASSWORD` 登录。在「创作」页选择 `demo-mock` 提交第一条任务，再到「任务」和「作品」页查看结果。

Compose 会启动 Web API、任务 Worker、模型下载 Worker 和回收 Watchdog。数据库和本地产物保存在 `studio-data` 卷。部署到 VPS 时，将 `--site-url` 改为实际访问的 HTTPS 地址，并使用反向代理转发默认的本机监听端口。迁移或升级前，备份数据卷和 `.env`，其中的 `ENCRYPTION_KEY` 用于解密已保存的 OAuth 令牌。凭据文件和数据库应留在部署环境中。

### 在本机运行离线演示

需要 Python 3.11+、`uv`、Node.js 和 npm。在仓库根目录运行以下命令。

```bash
uv sync
npm --prefix web ci
npm --prefix web run build
uv run python scripts/offline_server.py --port 18788
```

打开 `http://127.0.0.1:18788`，使用演示密码 `mock-local-only` 登录。演示使用临时数据库和本地文件，无需 Google 账号或云端算力。

## 使用真实 Colab GPU

1. 在「设置」中导入 Google OAuth Web 客户端 JSON，然后分别授权 Colab 和 Google Drive。按实际部署地址配置回调地址。[Google OAuth 与 Drive 配置](docs/OAUTH_SETUP.md)
2. 将 `.env` 中的 `GPU_REAL_ENABLED` 设为 `1`，配置每次任务、日和月预算，再重启服务。Docker Compose 部署可运行 `docker compose up -d` 以应用配置变更。[配置示例](.env.example)
3. 在工作流管理页启用内置草稿，按向导完成依赖校验、费用确认和真实 GPU 冒烟测试。通过后可从「创作」页提交正式任务。[内置工作流说明](docs/BUILTIN_WORKFLOWS.md)

内置模板包括 Krea-2 + MOSS 双采样文生图、MOSS-Image SFT 文生图、MiniMax H3 图生视频和 MiniMax H3 多参考图生视频。也可以导入自己的 ComfyUI 工作流，或从带工作流元数据的 PNG 提取工作流。[工作流审查](docs/WORKFLOW_REVIEW.md) · [PNG 导入与 GPU 选择](docs/runtime-gpu-and-png-import.md)

普通生成任务默认在完成后释放运行时，也可以配置空闲保留以便复用。GPU 规格受 Colab 当时可用容量影响，实际计算单元用量以 Google Colab 账单为准。

## 原生 ComfyUI 与模型库

- **原生编辑器**通过独立 HTTPS 域名打开 Colab 上的 ComfyUI，支持 Manager、HTTP/WebSocket 代理及 Google Drive 持久化。原生会话默认空闲 120 分钟回收，会话上限默认 720 分钟。[部署与验收](docs/NATIVE_COMFYUI.md)
- **统一模型库**由独立 Worker 把权重准备到 Google Drive，模型下载阶段无需申请 GPU。大文件下载可在来源支持且校验条件满足时恢复；原生会话可重启 ComfyUI 进程并同步模型索引，继续使用当前 GPU 实例。[模型库配置](docs/model-library-phase3.md) · [下载恢复与原生重启](docs/specs/NATIVE_COMFYUI_RESTART.md)
- **自定义节点**支持通过 Google Drive 保存归档，并在新的原生会话中恢复已验证版本。[自定义节点持久化](docs/CUSTOM_NODES_PERSISTENCE.md)

## API 接入

| 接口 | 用途 | 文档 |
| --- | --- | --- |
| Studio API | 上传素材、提交单任务或批次、查询进度、下载产物 | [调用文档](docs/API调用文档.md) · [Python 示例](docs/examples/studio_api_example.py) |
| 原生 ComfyUI API | 使用专用 Key 调用 `/prompt`、`/history/{prompt_id}` 和 `/view`，支持自管的 SillyTavern 接入 | [专用 API 接入](docs/NATIVE_COMFYUI.md#专用-api-接入) · [SillyTavern 配置](docs/NATIVE_COMFYUI.md#sillytavern) |

## 开发与验证

后端使用 FastAPI，前端使用 React、TypeScript 和 Vite。以下命令可从项目根目录运行。

```bash
uv sync --extra test
npm --prefix web ci
uv run pytest -q
npm --prefix web run test:contract
npm --prefix web run build
```

需要本地浏览器覆盖时，可运行 `uv run python scripts/verify_offline.py`。离线验收使用 Mock 算力和临时数据，真实云端验证记录见 [真实 GPU 验证](docs/REAL_GPU_VALIDATION.md) 和 [原生 ComfyUI 验收说明](docs/NATIVE_COMFYUI.md#自动测试与真实验收)。

## 上游与许可证

本仓库基于 Colab GPU Studio，沿用 [GNU AGPL-3.0](LICENSE)。ComfyUI、第三方节点和模型权重各有独立许可。[原项目](https://github.com/TizenryA/colab-gpu-studio) · [上游英文说明](README.en.md) · [上游 0.3.1 更新记录](docs/RELEASE_NOTES_0.3.1.md)
