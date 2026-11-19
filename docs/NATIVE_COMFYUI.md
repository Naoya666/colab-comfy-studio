# 原生 ComfyUI 接入与验收

本功能在独立域名直接代理 Colab 内的官方 ComfyUI。Studio 的 manifest/api_graph/object_info 导入和原图提取保留，原生入口不经过这些转换。没有复制官方前端、重建节点编辑器或改写官方工作流。

当前交付是代码实现及离线验证，尚未在真实 Colab、Drive FUSE、Cloudflare 或 SillyTavern 上完成端到端验收。下面明确区分自动测试和真实验证。

## 源码分析与复用

分析时本地 HEAD 为 `f003e09`，上游 origin/main 为 `c12f06071cc16c6e2661873f7703b60b30a46a8c`。fetch 后本地领先 4 个提交、落后 0 个：`b909b06`、`a901bca`（GPU 自由选择、原图提取）、`c971e08`（模型管理）、`f003e09`（创作页 GPU 选择）。这些提交全部保留；开始时工作区干净。

已实际 clone 并阅读 [FireTable/comfy-colab-kit](https://github.com/FireTable/comfy-colab-kit)，参考提交 `e642a36306e495cd5d68c4c0030765590d3ac85b`：

- `apps/gate/proxy.py`：Host/Origin 改写、用户凭据过滤、原始编码路径、流式响应、双向文本和二进制 WS、任务结束后的转发清理。按本项目 FastAPI/httpx/websockets 技术栈适配；额外将上传请求也改为流式，不注入 overlay。
- `apps/gate/session.py`、`apps/gate/app.py`：幂等启动、持久会话恢复、控制面与原生代理分离。实际使用本项目已有数据库租约与锁，未引入第二套会话数据库。
- `packages/colab/setup.py::start_tunnel`：named tunnel、HTTP/2、token-file。保留本项目 Colab CLI 隧道作为启动/执行控制通道；它提供 Jupyter/MCP 执行，不是已验证的任意端口 HTTP/WS 通道，所以浏览器数据通道采用参考项目的 Cloudflare 方案。
- `apps/gate/stopper.py`、`docs/AUTO_STOP.md`：以队列活动判断空闲；本实现对探测失败保守保留，同时检查 Drive 写入和上传。
- Manager 后台活动通过官方 [queue/status](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/openapi.yaml) 状态识别，包括节点安装及模型下载；兼容 legacy 和 v2 路径。
- 已阅读 `docs/AGENT_SCAFFOLDING.md`、`docs/SECURITY.md`；MIT 声明保留在 [licenses/comfy-colab-kit.txt](licenses/comfy-colab-kit.txt)。当前仓库和官方 ComfyUI、ComfyUI_frontend、comfy-cli 的许可证均为 GPL-3.0，未把官方源码复制进本仓库。

官方机制依据：[启动参数](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/cli_args.py)、[模型路径配置](https://github.com/Comfy-Org/ComfyUI/blob/master/extra_model_paths.yaml.example)、[内置 Manager 依赖](https://github.com/Comfy-Org/ComfyUI/blob/master/manager_requirements.txt)、[官方前端](https://github.com/Comfy-Org/ComfyUI_frontend)、[comfy-cli](https://github.com/Comfy-Org/comfy-cli)、[官方文档](https://docs.comfy.org/)。部署时必须填完整固定提交，代码不会自动追踪 master。

复用与主要修改文件：

| 文件 | 作用 |
| --- | --- |
| `api/app/gpu/colab.py`、`google.py`、`drive.py` | 复用 OAuth、账号绑定、GPU 分配、连接恢复、已有 Drive 模型目录；未覆盖其实现 |
| `api/app/gpu/native.py` | 幂等启动任务、Drive 运行凭据、健康检查、空闲和关闭围栏 |
| `api/app/gpu/native_runtime.py` | 官方服务、Manager、Drive VFS、用户目录、节点依赖恢复、Colab 侧失联回收 |
| `api/app/native_proxy.py` | 按 Host 分流的完整 HTTP/WS 代理、跨域名登录票据、cookie 与 Bearer 鉴权 |
| `api/app/native_routes.py` | `/api/gpu/native/status`、`start`、`ticket` |
| `api/app/gpu/mcp.py`、`runtime_worker.py` | 复用可信 bootstrap、状态文件和进度；Studio 共享原生进程，不在任务结束后关闭它 |
| `api/app/gpu/instances.py`、`scheduler.py` | 复用实例、预算、租约、CU 统计与回收；防止原生会话在忙时因不兼容而隐式创建第二台 GPU |
| `web/src/pages/NativeComfyPage.tsx`、`App.tsx` | 原生入口、GPU 选择、真实任务事件、就绪后跳转 |
| `tests/test_native_comfy.py` | 代理、鉴权、会话复用、回收和目录恢复模拟测试 |

## 部署

继续使用原有 api/worker/watchdog/model-worker 服务、同一个数据库与数据卷。升级已有部署时运行 `alembic upgrade head`，新增的表保存专用 API Key 和任务归属。以下域名、Access 策略和 tunnel 必须在部署环境配置。

1. 为 VPS 配置 `studio.example.com` 和 `comfy.example.com`，都转发至本项目 API；必须保留原始 Host、编码 URL、查询参数、WebSocket Upgrade，并关闭代理缓冲。示例 [deploy/native-comfy.nginx.conf.example](deploy/native-comfy.nginx.conf.example)。不要把 `/api/*` 单独转到 Studio 路由；整个 Comfy 域名均经过同一网关。当前仅实现独立域名，不提供未经验证的 `/comfy/` 子路径。
2. 在 Cloudflare 创建 **named tunnel**，另用内部域名，例如 `colab-backend.example.com`，origin 为 Colab 中的 `http://127.0.0.1:8188`。不能将其指向 VPS。此域名配置独立 Access Application，策略使用 **Service Auth**，只允许网关的 service token；不要配置公开 Bypass。启动时网关会验证无 service token 的健康检查不能返回 200。
3. 保存 tunnel token 和 Access service token，目录及文件不可进入 Git。Compose 示例将 `./secrets` 只读挂载到 `/run/studio-secrets`，各服务运行 UID 10001，文件必须对该 UID 可读。建议目录 0700、文件 0600，并正确设置所有者。
4. `.env` 增加下表，api、worker、watchdog 使用同一份配置。保留原有 Google OAuth、Drive 授权、真实 GPU 开关、费率与账号预算。完成正常构建、迁移和服务重启。

| 配置 | 示例/默认值 |
| --- | --- |
| `NATIVE_COMFY_URL` | `https://comfy.example.com`，留空禁用；必须为独立 HTTPS 源地址 |
| `NATIVE_COMFY_BACKEND_URL` | `https://colab-backend.example.com`，不能带路径、凭据或查询参数 |
| `NATIVE_COMFY_COMMIT` | 官方 ComfyUI 的 40 位提交，且支持 `--enable-manager` 与 `manager_requirements.txt` |
| `NATIVE_COMFY_ACCESS_FILE` | `/run/studio-secrets/comfy-access.json`，内容 `{"client_id":"…","client_secret":"…"}` |
| `NATIVE_COMFY_TUNNEL_TOKEN_FILE` | `/run/studio-secrets/comfy-tunnel.token`，内容为 tunnel connector token |
| `NATIVE_COMFY_IDLE_MINUTES` | `120`，正整数分钟，无最大值限制，原生队列/写入活动结束后才开始计时 |
| `NATIVE_COMFY_MAX_RUNTIME_MINUTES` | `720`，正整数分钟，无最大值限制，原生会话默认上限时长 |
| `NATIVE_COMFY_CACHE_GB` | `20`，按需 VFS 缓存回收目标；活动文件可能暂时超过该值；另外保留 5GiB 本地可用空间 |
| `NATIVE_COMFY_CACHE_MAX_AGE_MINUTES` | `60`，VFS 缓存自上次访问起的保留分钟数 |
| `NATIVE_COMFY_MAX_UPLOAD_MB` | `2048`，传给 ComfyUI `--max-upload-size`；Cloudflare 套餐仍可能限制单次请求大小 |
| `STUDIO_API_TOKEN` | 标准 API 客户端使用的现有 Bearer Token |
| `GPU_MAX_RUNTIME_MINUTES` | 复用原有会话期限；期限后拒绝新的原生写入，已入队任务和 Drive 同步完成后才释放 |

Colab 初始化安装 fuse3、固定 rclone v1.70.3（核对 [官方 release SHA256SUMS](https://github.com/rclone/rclone/releases/download/v1.70.3/SHA256SUMS)）和 cloudflared；需 Colab 支持 FUSE、系统包安装以及访问官方依赖源。这部分必须在真实 runtime 验证，不支持时启动任务会明确失败并走已有清理流程。GPU 参数保持自动选择，不使用固定 L4/highvram/tmpfs 优化。

### 内置 Manager 启动与验证

实际调用链为 `mcp.build_bootstrap_code` → Colab 内核的 `sys.executable` 启动 `runtime_worker.py` → `run_job` 的 `native_comfy` 分支 → `native_runtime.run`。官方 ComfyUI checkout 位于 `/content/ai-studio/native/ComfyUI`，版本由 `NATIVE_COMFY_COMMIT` 锁定。项目不包含该 checkout，因此 `manager_requirements.txt` 在 Colab checkout 校验后检查；缺少文件会明确失败。

初始化阶段使用运行 worker 的同一个 `sys.executable -m pip install -q -r <ComfyUI>/manager_requirements.txt` 安装依赖；安装前在 `worker.log` 输出实际 Python 路径和 requirements 路径。启动仍为同一个 Python 在该 checkout 下执行 `main.py --listen 127.0.0.1 --port 8188 --enable-manager`，保留用户/输入/输出/临时目录、上传大小、模型路径、预览和禁止自动打开浏览器等原有参数。没有新增端口、GPU 参数、第二个 ComfyUI 进程或旧版 Manager 的 clone/install 步骤。

ComfyUI 健康检查成功后，还要求本次启动的 `comfy.log` 包含官方 `[START] ComfyUI-Manager` 标记，并且本机 `/v2/manager/queue/status`（兼容 `/manager/queue/status`）返回布尔 `is_processing`。通过后 `worker.log` 输出 `[studio-native] Manager loaded: …`，之后才继续 tunnel/watchdog 和就绪报告。校验失败会报错并回收本次启动的 ComfyUI 子进程；历史日志中的标记不能冒充本次成功。日志位于 `/content/ai-studio/<execution_id>/comfy.log`，其路径和本次日志起点保存在 `native/state.json`；复用时校验原进程的日志及实时 Manager API，不另起进程。升级前缺少日志元数据的旧会话需正常停止后重新启动。

Colab 回收后 `/content` 中的状态和 Python 包不再保留；下一次初始化会再次安装官方和 Manager requirements，并携带 `--enable-manager`。Drive 用户数据保持恢复；自定义节点从校验归档恢复到本地 SSD，再通过官方 Manager 恢复依赖，详见 [Custom Nodes 持久化](CUSTOM_NODES_PERSISTENCE.md)。

安全边界为公网 Comfy 域名的 Studio 登录/Bearer 鉴权与写请求 Origin 校验，加内部 tunnel 全路径的 Access Service Auth。Manager 自身的 security level 不能替代这些鉴权。就绪检查除了 `/system_stats`，还对 Manager 队列及安装路径发起**无凭据 GET**，要求得到 401/403/302；不会提交安装操作。不要给 `/manager/*`、`/v2/*`、`/customnode/*` 或 `/api/*` 配置 Access Bypass，也不要降低 Manager 安全等级来绕过安装限制。

真实部署验收：检查上述两类日志证据；从公网未登录访问 Manager 安装路径应被 Studio 返回 401，已登录的跨站写请求应返回 403；内部 tunnel 上同一路径的无 service token 请求必须被 Access 拦截。然后正常释放并在新 Colab VM 再启动，重复检查日志和 Manager UI。当前本地只有模拟测试，不能据此认定真实 GPU、外部 Access 策略及重建后启动已经验收。

2026-11-17 验证：`pytest -q tests/test_native_comfy.py` 为 **69 passed**；完整 `pytest -q` 为 **677 passed、3 subtests passed**（56.23 秒，1 个既有 Starlette/httpx 弃用警告）。改动 Python 文件的 Ruff 检查和 `git diff --check` 通过。新增模拟覆盖 Manager requirements 缺失、同一 Python 安装与启动、启动日志/API 缺失、历史日志隔离、重复调用复用、清除运行时状态后重新安装启用、安装接口鉴权和内部 Access 路径绕过检测。

## 使用与生命周期

在已登录 Studio 点击「原生 ComfyUI」。已有健康会话直接进入；没有会话时选择账号与实际可申请 GPU，提交后查看现有任务事件，健康检查通过后跳转。跨域名通过 60 秒签名票据的 POST 建立专用 HttpOnly/Secure cookie，票据不放进 URL；原生 API 写入和 WS 使用此 cookie。服务不向浏览器发送 Colab、Drive 或 tunnel 凭据。

启动页可设置「空闲回收时间」和「会话上限时间」，单位为正整数分钟，均无最大值限制。空闲回收默认使用 `NATIVE_COMFY_IDLE_MINUTES`（默认 120 分钟）；会话上限默认使用 `NATIVE_COMFY_MAX_RUNTIME_MINUTES`（默认 720 分钟），原生会话可设置任意正整数分钟时长，从提交启动时计时，并按所选时长估算预算，仍需满足已有预算配置。`POST /api/gpu/native/start` 对应可选字段为 `retain_idle_minutes` 和 `max_runtime_minutes`，省略时沿用默认值。参数随会话持久化，重启服务后仍生效；修改已有原生会话的时间参数需先停止旧实例。复用 Studio GPU 时取本次会话截止与原实例截止的较早时间。会话到期后等待生成和同步完成再回收。保留期间看门狗按 60 秒间隔对未到期会话发送 Colab TFE `keep-alive` 与 Runtime Proxy / Jupyter Kernel 保活心跳（每 240 秒附带一次轻量内核空操作执行），一旦触发手动停止、空闲超时或到达会话截止时间即立即停止保活并释放。

API 客户端使用相同外部源地址和 `Authorization: Bearer <STUDIO_API_TOKEN>`。未授权请求返回 401，不跳转 OAuth。所有原生 API、静态资源和扩展路径由完整代理处理，路由不使用 API 白名单。内置 ComfyUI API 版本也可能有 `/api/` 前缀，仍原样转发。

原生启动也是现有调度器里的任务；多标签页、多进程启动以及 VPS 重启使用数据库作唯一来源。正在启动或已有原生实例时不会再创建一个原生 GPU。若已有 Studio 实例，必须等它空闲后复用。切换账号/GPU 要先停止旧实例，待旧租约释放后再选新规格；公网域名不变。会话到运行期限后也需要停止并重启。

Studio 复用原生实例时必须使用同一 ComfyUI 提交，所需受审节点也必须已安装且版本匹配。版本不一致会明确失败，不会在运行中的原生会话里改包或重启服务。原生队列与 Studio 队列最终都提交到该 ComfyUI 的同一个执行队列，原生页面不受 Studio 封装格式约束。

回收检查原生 `/queue`、Manager 节点/模型安装队列、rclone 上传队列/错误/空间、传输活动，以及持久目录的打开写句柄。HTTP 写入有持久活动心跳，关闭围栏与新请求共用 dispatch lock。页面静态请求、打开 WS、查看进度不延长空闲时间。未知/探测失败按忙处理，可能延迟释放，需检查 tunnel/rclone 日志；不会为达到超时强杀未完成视频或持久化写入。

Colab 侧 watchdog 还会按 PID 启动时间检查并恢复退出的 tunnel connector。另有失联保护：网关探测心跳丢失 20 分钟，且连续两次确认无队列及待写入时，尝试 `google.colab.runtime.unassign()`。它只在原生 bootstrap 成功后启动；初始化阶段仍依靠原有 worker/watchdog 恢复与清理。实际 Colab 自释放行为待验证。

## Drive 目录与持久化

沿用现有账号解析和 `AI-Studio/<owner>` 目录，不移动、改名或删除已有模型。

```text
AI-Studio/<owner>/
  models/                    # 原有模型缓存，保留 checkpoint/LoRA 等原生类别目录
    checkpoints/ … loras/
    native-index/<类别>/     # 模型库 ready 文件的 Drive 快捷方式；不复制权重
  native-comfy/
    user/                    # 原生工作流、设置、前端/Manager 用户数据
    input/                   # 上传的素材
    output/                  # 图片、视频等结果
    custom_nodes/            # 旧版本源码，保留用于首次迁移
    plugins/                 # 校验归档、候选/已验证清单、历史版本
  runtime-cache/             # 原有 Studio 环境缓存保持原样
```

`models/native-index` 名称加文件 ID 前缀避免同名模型冲突。模型库索引在原生启动时生成；运行中经 Studio 新导入的库文件需要重启会话刷新索引，Manager 直接下载到原生模型目录的文件不受此限制。OAuth 维持现有 `drive.file` 权限，只能看到此应用可访问的文件；用户 Drive 中其他文件需先通过已有模型库授权/导入。不能据此宣称扫描了全部 10TB。

本地 `/content/ai-studio/native/drive` 为 rclone 挂载，模型按实际读取下载到有上限的本地 VFS 缓存；模型文件在执行时真实可读，不做仅名称占位。不预复制整个 Drive，不强行加载显存。`extra_model_paths.yaml` 保留 `unet/diffusion_models`、`clip/text_encoders` 等别名。

用户目录、输入输出沿用 Drive VFS。Drive 挂载模式下，Manager 下载的模型也保存在挂载目录。自定义节点从本地 SSD 执行，稳定后以压缩对象直接写入 Drive，重启加载验证后更新 VPS 清单；正常释放会等待候选同步完成。依赖通过同一 Python 环境下的官方 Manager restore-dependencies 恢复，同一 VM 指纹不变时复用安装结果，不共享不同 GPU 的整个 site-packages。完整协议、回滚接口和八项验收见 [Custom Nodes 持久化](CUSTOM_NODES_PERSISTENCE.md)。

临时预览/中间文件在本地 `native/temp`，VFS 缓存和未完成下载也属于临时数据。突然被 Colab 强制回收、断电或手工外部删除 VM 时，尚未上传完成的数据仍可能丢失；代码不能保证未落盘数据在基础设施强制终止后恢复。已确认写入 Drive 的工作流、模型与结果保留。不要把配置了挂载等同于已经验证了真实持久化恢复。

## SillyTavern

核查了实际克隆的 [SillyTavern/src/endpoints/stable-diffusion.js](https://github.com/SillyTavern/SillyTavern/blob/06bde939fb1e9c4c8d8641d810f0a916b5bce127/src/endpoints/stable-diffusion.js)，其中 ComfyUI 的 ping、object_info、prompt、history、view 等 fetch 多数不带 Authorization；仅部分取消请求带 auth。因此不能要求其现有 ComfyUI 页面完成本站网页登录，或声称在一个 auth 输入框里填 Bearer 就都能成功。

在「原生 ComfyUI → API 设置」为自管的酒馆创建专用接入 Key，并按需启用开机权限。用 [同机中继配置](deploy/sillytavern-comfy.nginx.conf.example) 让中继监听 `127.0.0.1:8189` 并转发至本机 Studio API 端口。固定 `Host` 为原生 ComfyUI 域名，网关据此选择原生代理。把 Key 放到权限受限的 nginx include 文件，并在柏宝绘填写 `http://127.0.0.1:8189`。如果酒馆在另一台服务器，在那里运行中继并通过 HTTPS 请求原生域名。容器场景应使用仅酒馆容器可访问的私网中继。

实际酒馆生成、轮询和取图仍需真实环境验证。离线测试覆盖管理员透明转发、专用 Key 的任务和产物归属，以及 API 冷启动等待超时。

## 专用 API 接入

- 服务端客户端直接使用公网原生 HTTPS 域名，并在每个请求带上 `Authorization: Bearer <专用 Key>`。Key 仅在创建时显示，撤销后立即拒绝后续请求。
- 专用 Key 可用 `GET /_studio/api/status` 检查网关及会话就绪状态，也可获取 `/system_stats`、`/object_info` 和模型列表。生成顺序为 `POST /prompt`、`GET /history/{prompt_id}`，再按历史返回的 `filename`、`subfolder`、`type` 请求 `GET /view`。状态和取图权限归属提交 Key 及原生会话。该模式的共享队列、取消、Manager 和 WebSocket 操作需要管理员原生会话。
- 「限定已批准的工作流」要求节点图和固定输入与保存的模板一致。只在管理员已确认可变的输入位置放完整占位符。支持 `%prompt%`、`%negative_prompt%`、`%nl%`、`%seed%`、`%width%`、`%height%`。管理员应审核模板节点及固定输入，文本占位符适用于提示词输入。自管客户端可使用「信任客户端提交的工作流」权限。每把专用 Key 最多提交 20 次/分钟，最多同时有 4 个未完成任务。
- 自动启动默认为关闭。开启后，获准开机的 Key 发送有效 `POST /prompt` 可按保存的账号、GPU 和回收参数启动。连接探测与轮询不会开机。启动请求共用现有调度和预算，已有实例沿用自己的配置。网关最多等待启动 25 秒。若返回 `503 NATIVE_STARTING`，说明工作流尚未提交，后台仍在启动。调用方可等待就绪后重发。未修改的柏宝绘需手动重试。初次配置柏宝绘时可先在原生页启动实例，再测试连接和加载模型。

## 自动测试与真实验收

新增测试使用 MockTransport、ASGI WebSocket、真实 SQLite 调度器及模拟 GPU，不申请 GPU、不调用 Google、不启动真实 tunnel。

- `pytest -q tests/test_native_comfy.py`：代理所有主要路径和任意扩展路由、编码路径、分块媒体响应、大请求体、任意 HTTP 方法、文本/二进制 WS 与断开、cookie 登录、跨站拒绝、未授权拒绝、会话期限和停止围栏。
- 实例测试：重复启动、服务重启后复用同一启动任务、GPU 切换冲突、Studio 不另起/关闭原生进程、mock bootstrap→保留→watchdog→释放；视频忙时即使过硬期限也保留。
- 本地目录重建模拟：已保存工作流和模型在新 checkout 下仍可读；不是 Drive FUSE 实测。
- 原有后端测试、前端 TypeScript/构建、前端 contract 和 Node 测试继续执行。执行结果如下；不把模拟测试当作真实 GPU 通过。

部署后按以下 16 项验收，以下均尚未在真实 GPU 验证：

| 场景 | 当前证据 / 真实环境动作 |
| --- | --- |
| 1. 未启动时选 GPU 启动 | 幂等 admission/mock 生命周期覆盖；实测账号分配、FUSE、依赖及 tunnel |
| 2. 已启动时不重复创建 | 数据库幂等与重启模拟覆盖；实测多标签页和已有 Studio 实例 |
| 3. 官方节点编辑器 | HTML/资源透明代理覆盖；浏览器打开官方 UI |
| 4. 普通 UI Workflow JSON | 直接使用官方导入；用真实 UI JSON 导入 |
| 5. 元数据图片拖入 | 未改官方导入；拖入 PNG 验证 |
| 6. Drive checkpoint/LoRA | 目录/索引代码及模拟恢复；实测已校验文件可选且可执行 |
| 7. 出图和实时预览 | WS 二进制模拟覆盖；实测 GPU 采样预览 |
| 8. 上传/图片/视频下载 | 流式及 Range 模拟覆盖；实测 Cloudflare 套餐限制和大文件 |
| 9. 自定义节点 | 官方 Manager 启用与 requirements 恢复；安装并执行至少一个节点 |
| 10. 释放后持久化恢复 | 本地目录重建覆盖；实测等待上传后释放、新 VM 恢复工作流与模型 |
| 11. GPU 切换固定域名 | 冲突拒绝模拟覆盖；停止→换规格→原域名恢复 |
| 12. 酒馆 API | 标准 API 代理模拟覆盖；配置 loopback relay 后实际生成取图 |
| 13. Studio/封装导入 | 原有测试继续通过；真实共享进程上跑已有工作流 |
| 14. 视频不会误回收 | 队列、同步、超时保护模拟覆盖；实测无浏览器长视频任务 |
| 15. VPS 重启不重复分配 | 持久 job/lease 模拟覆盖；重启 api/worker/watchdog 后核对 Colab assignments |
| 16. 未授权不能访问或启动 | 代理/cookie/WS 拒绝模拟覆盖；公网检查 Comfy、内部 tunnel、启动 API |


2026-11-16 本次执行结果：

| 检查 | 结果 |
| --- | --- |
| `/tmp/colab-studio-venv/bin/pytest -q` | **646 passed，3 subtests passed**；1 个既有 Starlette/httpx 弃用警告；56.25 秒 |
| 新增 `tests/test_native_comfy.py` | **38 passed**，包含启动失败释放、Manager 队列及 rclone 校验测试 |
| `npm run build --prefix web` | TypeScript 与 Vite 构建通过 |
| `npm run test:contract --prefix web` | 两项契约检查通过 |
| 在 `web/` 执行 `node --test tests/*.test.mjs` | **57 passed，0 failed** |
| Ruff 检查全部改动 Python 模块和新增测试 | 通过 |
| `git diff --check` | 通过 |

上述测试由仓库的离线网络拦截器禁止真实 Google/Colab 请求；未申请 GPU，未消费 CU。未构建或发布 Docker 镜像，未配置外部 DNS、TLS 或 Cloudflare 账号。真实 UI、FUSE、模型读取/生成、Manager 安装、Drive 恢复和酒馆端到端状态均仍为上表的待验收项。

## 模型存储模式

在「原生 ComfyUI」启动页选择模式。「API 设置」也可以为 API 自动开机保存独立的启动配置。现有实例继续使用启动时的参数。切换模式或修改挂载缓存参数需停止实例并重新启动。同实例的「重启原生 ComfyUI」只重启 ComfyUI 进程。

| 模式 | 运行方式 |
| --- | --- |
| Drive 挂载模式（默认） | 模型沿用 Drive FUSE 路径和 rclone VFS 按需缓存。模型库已有 Drive 快捷方式继续可用。 |
| 本地模式 | 权重保存在 Colab 的 `/content/ai-studio/native/ComfyUI/models`。完整下载并校验后才出现在实际文件列表。模型目录和本项目的模型路径配置均使用本地磁盘。工作流、设置、输入输出和自定义节点持久化继续使用现有 Drive 机制。 |

启动 API 可传 `model_storage_mode`（`drive_mount` 或 `local`）、`cache_gb`、`cache_max_age_minutes`。省略时使用挂载模式、`NATIVE_COMFY_CACHE_GB`（默认 20）与 `NATIVE_COMFY_CACHE_MAX_AGE_MINUTES`（默认 60）。容量按 rclone 的二进制 GB 计，属于缓存清理目标。保留时间从 VFS 文件上次访问起算，另受容量和磁盘空间回收影响。原生页面在运行时可获取指标时展示 VFS 占用与本地磁盘剩余。升级前启动的旧会话继续沿用其挂载，重新启动后可查看保存的实际配置。空闲回收和会话上限由各自的原生会话参数控制。

本地模式先检查同类别、相对路径与可用内容校验信息是否匹配。命中时无需查询 Drive。缺失时 Colab 优先从绑定的外部来源或工作流的明确模型下载元数据读取，接着尝试已登记的 Drive 文件。对于旧的 Drive 模型目录，还会按准确类别及相对路径查询。外链失败时，只有存在符合所需 SHA-256 的 Drive 副本才尝试回退。多个版本或不确定来源会列出具体路径和原因。对于已经存在的本地同路径文件，版本冲突时应给所需版本选用不同路径。实例内的完整文件可重复使用，新 Colab 实例按需重新下载。

每个实例合并同一路径和身份的并发下载，在模型目录外写临时文件并原子发布，为运行预留 5 GiB，并阻止空闲回收中断正在准备的模型。支持强 ETag 的来源可在短暂连接中断后续传。模型字节由 Colab 直接读取。运行日志与准备状态记录所用来源类别、域名、进度、耗时、回退和失败原因。可用现有模型库凭据设置配置 Hugging Face Token 或 Civitai Key。临时签名重定向由下载器处理，凭据仅发送给相应服务的首跳主机。

从「创作」页提交的作业使用固定工作流的来源和模型身份。复用原生会话时需要兼容的本地模型目录，原生 Drive 挂载实例会被跳过或提示不兼容。原生页面继续采用上表的模型存储策略。[创作模式的本地模型恢复](RUNTIME_PROTOCOL.md#创作模式的本地模型恢复)

### 在模型库登记外部来源

导入外链时先解析并选择具体文件。准备任务从创建起保存输入的原始链接，成功后将其关联至最终 Drive 文件 ID、文件大小和 SHA-256。模型库可填写类别下的相对路径，包含文件名，例如 `styles/pencil.safetensors`。相同文件内容可以共享 Drive 副本，并保有各自的类别和路径记录。

对已有的 Drive 模型，打开「详情与引用 → 外部来源链接」新增或修改链接。清空并保存可移除绑定，Drive 文件继续可用。可取得完整来源 SHA-256 时与已登记身份比较，同时检查可用的精确大小。通用直链或缺少可比对的哈希时显示「外部来源待验证」，执行下载仍检查记录中的哈希或大小。链接须为无凭据的 HTTPS 链接。

### 在酒馆首次调用前准备

1. 启动本地模式的原生 ComfyUI。在 Studio 原生页打开「检查并下载此工作流所需模型」。
2. 上传或粘贴 ComfyUI 的普通工作流 JSON 或 API 工作流 JSON。对于动态或自定义加载器，按需填写显式依赖，例如 `[{"category":"loras","name":"styles/pencil.safetensors"}]`。可以通过模型库绑定准确路径，也可为显式依赖补上 `url`、`sha256` 和字节数 `size`。
3. 查看缺失项、下载进度和失败信息。准备完成后，酒馆按原方式提交工作流。

Studio 预准备接口为 `POST /api/gpu/native/model-preparation`，请求字段为 `workflow` 和可选的 `dependencies`；随后 `GET /api/gpu/native/model-preparation/{id}` 获取状态。可用 `GET /api/gpu/native/model-preparation/latest` 查看最近一次准备，包括自动提交的来源和回退原因。预准备不把任务放入 ComfyUI 队列。历史状态保存在当前 ComfyUI 进程。再次浏览 `/object_info` 或模型列表时可看到模型库中可用的登记名称，加载时仍使用经过校验的实际文件。

网页、专用 API Key、管理员 API 和共享该实例的 Studio 工作流都经过 Colab 的准备逻辑。投递前刷新 ComfyUI 文件列表，再由原生处理器校验、入队并返回真实 `prompt_id`。后续队列、WebSocket、history、取图保持原协议。客户端首个冷请求仍可能因客户端、中继或网关的等待限制而断开。准备阶段断开时不会代为入队，已经开始的共享下载可继续，后续提交可复用文件。一旦请求交给 ComfyUI，则沿用其取消和队列语义。重复请求共享权重下载；相互独立且成功送达的提交仍按 ComfyUI 原规则分别入队。

自动识别使用已有加载器字段规则和目录别名，覆盖 checkpoint、LoRA、VAE、diffusion model、text encoder 等。原生工作流顶层 `models` 与节点 `properties.models` 中类别和文件名匹配的条目可以提供下载身份。自动准备不会把远端 API 节点的模型选项、普通字符串或提示词当成权重。自定义动态节点、视觉 JSON 的自定义小部件和运行期生成的依赖可用显式声明；管理员还可配置 `MODEL_DEPENDENCY_RULES` 与 `MODEL_LIBRARY_DIRECTORIES`。自动下载沿用模型库安全扩展名，未提供哈希的来源需要可校验的文件大小和支持的格式。只使用工作流 JSON 所能静态识别的依赖，并保留 ComfyUI 自身对节点、模型与图的正式校验。
