# 第三阶段交付：统一模型库、依赖与 Drive 持久化

本次直接扩展当前本地版本。原有 GPU 选择、PNG 工作流提取、OAuth、CPU prep、审核和发布流程继续保留。模型文件的「完整性就绪」、来源与许可的「人工审核」、工作流的「真实运行验收」相互独立。

## 1. 实际文件与职责

| 文件 | 用途 |
| --- | --- |
| `api/app/gpu/model_sources.py` | HF / Civitai / HTTPS 解析、具体文件候选、完整 Hash 查询、平台搜索、公网 DNS 固定、重定向与凭据隔离 |
| `api/app/gpu/model_library.py` | 模型索引、Drive 存储账号、加密下载源凭据、持久化队列、流式下载、校验、分块上传、旧缓存索引、绑定及草稿生成 |
| `api/app/gpu/model_dependencies.py` | 可扩展节点规则、API prompt / widget 解析、准确匹配、候选、架构约束、运行副本、已验收替换模型集合 |
| `api/app/model_library_routes.py` | 所有者隔离的模型库、任务、存储设置与工作流依赖 HTTP API |
| `api/app/gpu/repository.py`、`migrations/versions/0004_model_library.py` | 三张增量表及兼容初始化；不修改旧工作流或搬迁权重 |
| `api/app/main.py`、`api/app/gpu/registry.py` | 导入与图片配置后的依赖扫描；具有明确旧格式来源的导入自动入准备队列；保留原始图和元数据；补充标准模型节点发布 allowlist |
| `api/app/admission.py`、`api/app/job_schemas.py` | 默认绑定及单任务覆盖、开 GPU 前检查模型、独立 Drive 账号、任务快照中的路径与权重映射 |
| `api/app/review_routes.py` | 模型库文件登记后的多来源人工审核；普通旧模型沿用原有固定 revision 规则 |
| `api/app/gpu/runtime_worker.py` | 仅恢复当前快照权重、本地 SHA 校验复用、磁盘检查、避免为库内 Drive 文件再生成缓存副本 |
| `api/app/worker.py`、`docker-compose.example.yml` | 新增独立 `model-worker`，与 GPU 调度进程分开 |
| `web/src/pages/ModelLibraryPage.tsx`、`web/src/lib/model-library.ts` | 模型库、统计、筛选、详情、来源选择、重试、校验、删除、上传与来源/存储设置 |
| `web/src/components/ModelDependencies.tsx` | 工作流依赖、来源候选、默认绑定、LoRA 权重、绑定草稿和本次任务模型覆盖 |
| `web/src/App.tsx`、`web/src/pages/SettingsPage.tsx`、`web/src/pages/WorkflowManager.tsx`、`web/src/styles.css` | 新入口与移动端布局 |
| `web/src/lib/png-workflow.ts`、`web/src/components/ImageWorkflowImport.tsx`、`api/app/gpu/image_import.py` | 在原有提取功能上保留 PNG 文本元数据，原始 workflow / prompt 仍保留 |
| `tests/test_model_library.py`、`tests/test_model_library_http.py` | 新增单元和模拟集成 / 真实本地 HTTP 测试 |
| `scripts/model_library_browser_check.py` | 手机 / 桌面实际浏览器操作检查 |

## 2. 数据结构和 API

新增 `gpu_model_library`、`gpu_model_downloads`、`gpu_model_dependencies`，均复用 Repository 的所有者隔离、时间戳与 JSON 数据列。数据库迁移是增量迁移。

模型使用独立 ID，显示名称与文件名分开；记录类型、架构（默认 unknown）、来源 URL、真实平台文件身份、预期 SHA-256、实际 SHA-256、预期/实际字节数、Drive ID / 存储账号 / MD5、管理所有权、状态、错误、标签、触发词及最后使用时间字段。同完整 Hash 的同类型添加可共用索引；不同版本/Hash 不以文件名覆盖。

HF 身份是 `repo_id + revision + filename`，revision 解析到 40 位 commit。Civitai 身份是 `model_id + version_id + file_id`，不会写进 HF revision。Civitai `sizeKB` 是显示用近似大小，不作为精确字节校验值。

模型状态包括 pending、downloading、verifying、uploading、ready、failed、missing、unverified；任务另有 cancelled。任务持久化 ID、阶段、传输字节、总大小、阶段平均速度、错误、尝试次数、心跳与取消标记。分块上传会话在服务端加密保存，不进入任务公开响应。

API 基础路径为 `/api/gpu`，沿用现有登录鉴权与 CSRF 检查：

| 方法 / 路径 | 功能 |
| --- | --- |
| `GET /model-library?q=&model_type=&architecture=&state=` | 列表、分类目录和索引统计 |
| `POST /model-library/resolve` | 输入 url，或 repo_id / revision / filename，或完整 sha256；返回具体文件列表 |
| `GET /model-library/search?q=` | HF / Civitai 候选发现，不能自动认定为原模型 |
| `POST /model-library` | 选定文件入库；平台身份在服务端重新解析确认 |
| `GET/PATCH/DELETE /model-library/{id}` | 详情 / 非身份信息 / 删除；详情包含工作流引用 |
| `POST /model-library/{id}/prepare` | 后台准备；`{"action":"verify"}` 重新校验 |
| `POST /model-library/{id}/check` | 检查 Drive 文件是否仍存在且元数据一致 |
| `GET /model-library/tasks`、`POST /model-library/tasks/{id}/cancel` | 队列状态与取消 |
| `POST /model-library/upload-session` | 建立浏览器直接上传 Drive 的 resumable 会话 |
| `GET/PUT /model-library/storage-settings` | 固定长期存储 Drive 账号；有索引后禁止直接切换账号 |
| `GET /model-library/source-settings` | 仅返回凭据是否已配置 |
| `PUT /model-library/source-settings/{huggingface或civitai}` | `{"token":"..."}` 加密保存，空字符串清除 |
| `POST /model-library/{id}/workflow-model` | 管理员将库内模型登记为**未启用**的工作流模型；需提供 model_key、实际 license、resource_profile |
| `GET /workflows/{id}/model-dependencies` | 各节点依赖、匹配、模型准备与工作流验证状态 |
| `POST /workflows/{id}/model-dependencies/{action}` | scan / match / prepare / adopt-cache / bind / draft |

`bind` 请求包含 dependency_id、model_id，以及可选 strengths、disabled、confirm_unknown。`draft` 创建新版本并重置审核状态，不修改原版。

任务提交增加 `model_overrides` 数组，每项含 dependency_id、model_id、strengths、disabled、confirm_unknown，仅影响本次快照。默认绑定位于独立的依赖表，不写回原始 JSON。

## 3. 下载、存储与完整性

新下载由 **VPS 的独立 CPU 模型进程**处理，不申请 Colab GPU，也不假设 CPU runtime 免费。默认同时处理一个大模型，使用跨进程文件锁；API / 浏览器关闭不影响已接受任务。必须部署 model-worker。

文件在 `MODEL_LIBRARY_SCRATCH` 中以临时文件形式下载，按块读取和计算 Hash，不整体加载进内存。下载中检查管理员上限和可用空间。完整性校验通过后，分块写入现有 Drive 模型缓存根目录：

```text
AI-Studio/<owner>/models/
  <原有 target 目录与缓存文件，继续保留>
  by-sha256/<Hash前两位>/<完整SHA-256>
  library-uploads/<浏览器上传的原文件>
```

模型物理身份与文件名、目录分类、工作流、GPU 型号解耦。上传后检查 Drive 文件大小和实际 MD5，确认远端提交后才标记 ready。物理文件的 SHA-256 由下载字节计算；若来源提供预期 SHA-256，必须匹配。`computed_hash_only` 与 `expected_hash_verified` 区分是否具备预期 Hash；这两个标记都不会自动批准来源或许可。

同 Hash 已就绪文件先检查远端元数据再复用；相同平台文件或预期 Hash 的并发请求尽量合并索引和队列。未知 Hash 的同内容文件在最终内容地址上传时去重。中断后安全重试，重试会检查已有内容地址；不会把半成品标记为 ready。VPS 临时文件在本次操作完成/失败后清理，崩溃残留在下次持锁运行时清理。

本地文件在浏览器分块直接上传 Drive，不经过 API 的普通上传体积上限。当前页面内有分块失败探测、进度和重试。导入 Drive 文件优先使用 Drive 原生 `sha256Checksum`；不存在该值时，后台流式读取并计算校验。**不信任外部文件自填的 appProperties Hash 作为最终字节校验结果。**

没有自动淘汰策略。删除前显示引用数；仍有引用必须明确确认。默认只移除索引；外部导入/历史缓存文件不允许通过模型库删除原文件。仍被其他索引共享的物理文件禁止删除。

## 4. 依赖解析、匹配与模型替换

优先解析 API prompt 的已知节点输入，包括 Checkpoint、LoRA、VAE、UNET、单/双/三 CLIP、CLIP Vision、ControlNet、Upscale、Embedding，以及现有视频流程使用的加载节点。补充 rgthree Power LoRA 的已知结构。未知 Loader 显示人工确认项，不把普通提示词、输出文件名或任意扩展名字符串当作依赖。

仅有可视化 workflow 时，只使用已知 widget 顺序；不会把可视化图标记为有效 API prompt。workflow 节点 `properties.models` 中与具体引用一致的 URL、完整 SHA、目录和显式架构可作为补充线索；原始数据保留。

匹配顺序：明确绑定 → 完整 Hash / 固定来源文件身份 → 旧 Drive 缓存 → 明确来源 → 完整 Hash 的 Civitai 查找 → 文件名候选 / 平台搜索 → 人工补充。短 Hash 不充当 SHA-256，唯一文件名候选也必须确认。

一键准备只为必需且启用的资源建立任务。强度为零/关闭的已知 LoRA 不要求恢复；各节点引用关系保留。相同权重通过库索引和内容地址共享。

默认绑定和单任务覆盖均支持已有 LoRA 节点的实际强度字段；不插入任意新节点链。已知架构不兼容直接拒绝，unknown 要求用户确认。底模/LoRA 的**新身份首次使用仍要走原有审核及真实冒烟**；可将绑定生成草稿完成审核。之后，同一工作流系列中环境 pins、图结构、参数定义一致，且有已验收真实运行证据的模型身份，允许通过默认绑定或本次覆盖直接切换。单纯把模型标签改成 SDXL 或模型状态改成 ready 不能取得运行资格。

## 5. 旧格式、GPU 与审核兼容

原有 weight_files / HF 固定 revision / SHA 和 Drive 权重格式继续工作。扫描会把已声明来源与节点引用关联。显式来源的导入可直接创建准备队列；不明来源不猜测。

后台首次准备旧 weight_files 的模型时，先按原 target / Hash / 大小查找现有 DriveCache，找到后读取或使用 Drive 原生 SHA 校验并索引，不重新上传。也可从工作流依赖区手动「索引并校验旧缓存」。不扫描或恢复整个 10TB 模型库，不自动删除历史副本。

GPU 接收的仍是本次不可变快照里的 weight_files。映射为 Drive 文件后按原引用别名恢复到 ComfyUI 对应目录；本地相同 SHA 的文件直接复用。切换 L4/A100 不改变权重物理地址。环境缓存仍沿用既有 GPU 架构区分。恢复库内 Drive 文件不会再向 DriveCache 保存第二份。

模型库长期存储账号与算力账号独立，绑定模型的任务显式携带相应 drive_account_id。现有 OAuth 配置不重构。库内模型登记到旧 registry 时默认 disabled；非 HF 模型不生成虚构 commit。已有 HF 模型的审核路径保持原规则。

## 6. 安全与测试

下载只允许 HTTPS 公网。每一跳重定向均检查 DNS 所有结果，拒绝私有、回环、保留、组播和元数据地址，并固定连接到已检查 IP，保留 TLS SNI / Host。禁用环境代理和自动重定向；Token 仅发往对应平台原主机，不传给 CDN。含 token / key / signature / credential 的链接拒绝持久化。无授权许可、401/403、429 等情况不会尝试绕过。

新模型入口默认只接受 safetensors / gguf / onnx；pickle 类权重拒绝自动入库，旧已审核运行路径保留。文件名和配置路径检查目录穿越。模型库不执行下载的脚本或自定义节点。未知自定义节点仍受既有 pin 和发布 allowlist 约束。

本次全量后端结果为 **608 passed、3 subtests passed**，新增模型测试 **58 项**；只有现有 FastAPI / httpx 测试客户端弃用提示。前端 `npm run build`、`npm run test:contract`、Ruff 与 `git diff --check` 均通过，空 SQLite 数据库实际执行 `alembic upgrade head` 到 `0004_model_library` 成功。全量输出保存在 [model-library-tests.txt](offline-evidence/model-library-tests.txt)，网络审计中真实运行时、MCP、Colab OAuth 调用均为 0。

实际执行的验证代码见 `tests/test_model_library*.py`、`scripts/model_library_browser_check.py`。新增测试覆盖来源解析、SSRF/重定向/凭据、Hash 不匹配、Drive 模拟上传与去重、索引/任务合并、重启重试、取消、架构约束、可选 LoRA、原始 PNG 保存、身份不可被重新校验覆写、运行副本、已审核替换集合及 HTTP 所有权。

浏览器检查在 390×844 和 1280×900 验证模型添加、搜索、编辑、取消、凭据脱敏和无横向溢出；此前 PNG 压缩文本提取、原图导入、JSON 下载、L4/A100 选择与确认页面也再次执行通过。

**所有平台/Drive/传输集成验证均为模拟；没有使用实际平台凭据、真实模型或真实 Colab GPU。** 浏览器使用真实本地 API/数据库，但不启动 model-worker 或任何真实提供商请求。还需在部署环境验收实际 Drive CORS、gated HF、Civitai 不同精度下载、大文件网络中断和真实 GPU 运行。

## 7. 升级和使用

1. 备份现有数据库、数据卷和部署环境配置，保留 ENCRYPTION_KEY（不要重新生成，否则旧 OAuth/新凭据无法解密）。
2. 停止旧进程，使用当前代码重新构建；已有授权、缓存和工作流数据保留。
3. 在同一个数据库配置和数据卷下运行 `alembic upgrade head`。Repository 也支持兼容增量建表，但正式部署建议执行迁移。
4. 在原 api / worker / watchdog 之外启动 `python -m app.worker model-worker`。Docker 示例已包含该服务；自定义 compose 部署需要补上它，所有服务共享相同 DATA_DIR/ENCRYPTION_KEY。
5. 确认模型临时磁盘有足够空间。可按下列变量配置；不要把下载 Token 放进 URL 或日志。
6. 打开「模型库」或设置，先选择已完成 Drive OAuth 的长期存储账号，再配置 HF Token / Civitai Key。添加一个小模型做真实端到端验收，再扩展到大模型。
7. 添加来源并选择具体文件 / 直接上传 / 填已有 Drive 文件 ID；查看队列进度与错误。工作流列表展开「模型依赖」，扫描、匹配或导入旧缓存，必要时确认候选并准备。
8. 新的模型系列可在模型详情「登记为工作流模型」，填写实际许可与资源信息；在工作流配置中关联该 model_key 并完成原有审核、真实冒烟及发布流程。
9. 模型就绪后选择 GPU 开始生成。首次替换新身份时生成绑定草稿并审核；已验收的兼容身份可以直接覆盖本次任务。

```dotenv
# 可选；默认 DATA_DIR/model-scratch。一次只暂存一个模型，仍需模型本体大小的磁盘。
MODEL_LIBRARY_SCRATCH=/data/model-scratch
# 默认 150 GiB 的单文件上限，可按实际磁盘调整。
MODEL_LIBRARY_MAX_FILE_BYTES=161061273600
# 扩展 ComfyUI models 目录内的类别映射；禁止绝对路径和目录穿越。
MODEL_LIBRARY_DIRECTORIES={"custom_weights":"custom_weights"}
# 扩展已知节点的字段映射。识别依赖不会自动授权执行该自定义节点。
MODEL_DEPENDENCY_RULES={"MyReviewedLoader":{"weights_name":"custom_weights"}}
```

本地复验（当前机器 Python 环境路径仅为本次验证环境）：

```bash
python -m pytest -q
python -m ruff check api/app tests/test_model_library*.py scripts/model_library_browser_check.py
cd web
npm run build
npm run test:contract
# 另一个终端启动 scripts/offline_server.py 后运行两个浏览器脚本：
# python scripts/model_library_browser_check.py
# python scripts/runtime_gpu_png_browser_check.py
```

## 8. 当前限制与风险

- 尚未完成真实 Drive / HF / Civitai / Colab 验收，不能将 mock 通过视为生产下载或推理已验证。
- 服务端采用单机、共享 DATA_DIR 的串行传输锁，不是跨主机分布式下载队列。大模型会占用临时磁盘和 VPS 带宽；下载不长期保存，但传输期间仍需要空间。
- 服务端中断使用安全重试，未实现通用 HTTPS Range 字节断点续传；Drive 内容地址用于完成上传后的远端协调。浏览器上传支持当前会话内分块重试，刷新或关闭页面后没有恢复持久化会话；取消后已创建的不完整 Drive 上传会话由平台过期处理。
- Drive 导入目前为文件 ID；没有 Drive 文件选择器或任意路径浏览。drive.file 仅能访问当前 OAuth 应用可见的文件；授权范围之外的历史文件不能仅凭 ID 读取，需要通过现有应用授权/可见机制导入或重新上传。
- 列表统计是「已索引模型」的物理大小，不能代表整个 Google Drive 实际容量；旧缓存中既有重复副本不会自动删除。就绪状态在准备、显式远端检查、校验及运行提交时验证，列表刷新不会逐个遍历远端所有文件。
- 仅有文件名无法可靠确定原模型，搜索结果必须确认。未知架构不猜测；自定义节点识别、widgets 以及新格式需要补充规则。
- 新模型身份仍要先完成原有审核/真实运行验收；切换能力只对已有相匹配证据的模型开放，防止以「兼容」标签绕过审核。
- 新入口不接受 ckpt / pt / pth / bin 等潜在反序列化风险格式。需要时先在可信环境转换，或使用现有独立审核流程；不会为了接入模型库降低原有执行审核。
- 页面目前使用卡片列表；没有独立列表/卡片切换、封面管理、Drive 全库扫描或通用可视化节点编辑器。长列表使用现有轻量 Repository 查询，超大型目录后续应增加数据库分页/索引。
- `last_used_at` 字段保留；目前未据此执行回收策略。没有任何自动删除模型逻辑。

Civitai 实现依据官方 [模型版本与文件接口](https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md) 和 [Bearer 鉴权说明](https://github.com/civitai/civitai-developer-docs/blob/main/site/guide/authentication.md)；HF 使用官方 [Hub API](https://huggingface.co/docs/hub/api) 对仓库 revision 和文件元数据的接口约定。
