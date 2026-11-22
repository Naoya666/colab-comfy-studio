# 原生 ComfyUI 启动耗时分析与优化设计

## 1. 背景与实测基线

2026-11-22 在 `bero` 生产环境（源码提交 `3cba89e`）对原生 ComfyUI 进行了一次完整冷启动压测与内部取证，目标是量化端到端各个环节的真实耗时，定位瓶颈，并制定可分阶段落地、不降低安全与一致性保障的优化方案。

### 1.1 实测配置与样本

- **账号**：`colab cli账号`（CLI tunnel 分配）
- **GPU 规格**：`L4 · SHAPE_STANDARD`（实测 `NVIDIA L4 23034 MiB`，12 vCPU，54 GB RAM，Python `3.13.16`，PyTorch `2.14.1+cu130`，预装 `/usr/local/bin/uv` 与 `/usr/bin/pigz`）
- **模型存储模式**：`local`（本地按需完整下载）
- **会话策略**：空闲回收 `120` 分钟，会话上限 `720` 分钟
- **归档插件规模**：19 项（18 个自定义节点包 + `websocket_image_save.py`），压缩包总计 **299.4 MB**，解压后约 **446 MB**
- **实测任务**：`e508d30a-93f9-4185-9294-b07076e9ddbd`（`13:06:19.920` 提交 → `13:19:11.679` 完成，Asia/Shanghai）
- **端到端总耗时**：**771.8 秒（12 分 51.8 秒）**

> 同期对照历史 L4 样本 `ebf2f527-9e83-4df4-b929-2c0006377153` 总耗时 753 秒（`native_environment=549.4s`、`comfy_start=97.1s`），本次实测结构高度稳定、具有代表性。

### 1.2 全链路分阶段耗时拆解

注：当前 [native_runtime.py](../../api/app/gpu/native_runtime.py) 在第 526 行与第 564 行两次调用 `stages.start("native_environment")`，[runtime_worker.py](../../api/app/gpu/runtime_worker.py) 的 `_Stages.end()` 直接按阶段名覆写字典，导致第一段系统依赖耗时（23.7s）在 `worker_timings` 中被覆盖隐藏。下表结合控制面事件与 Colab 内部文件时间戳还原了完整细分耗时：

| 序号 | 阶段 | 细分步骤 | 耗时 (s) | 占比 | 代码入口与实测发现 |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 控制面与分配 | `queued` → `validating` → `waiting_for_capacity` → `allocating` → `preparing` → `loading` → Worker 拉起 | **21.7** | 2.8% | 6 次状态推进穿插 [worker.py](../../api/app/worker.py) 固定 `sleep(2)`；`allocating` 6.4s、`loading`（`runtime_secrets` + 创建 Jupyter Kernel + 下发 bootstrap）6.9s |
| 2 | 系统依赖（原第 1 段） | `apt-get update` (13.4s) + `apt-get install fuse3 mesa-vulkan-drivers` (8.0s) + `rclone` (1.6s) + `cloudflared` (0.7s) | **23.7** | 3.1% | 全程串行；耗时在 `worker_timings` 中被覆盖 |
| 3 | Drive 挂载 | `mount_drive()`（`rclone mount` FUSE） | **0.6** | 0.1% | 已极快 |
| 4 | 源码与目录 | ComfyUI shallow checkout (3.2s) + `prepare_paths` / 复制本地模块与配置 (3.1s) | **6.3** | 0.8% | 可与阶段 2 并发 |
| 5 | 基础 Python 依赖 | 4 次独立 `pip install`（`jedi` 3.3s、`requirements.txt` 23.9s、`httpx` + `manager_requirements.txt` 5.7s） | **32.9** | 4.3% | Colab 已预装 `uv`，但仍用 `pip` 分 4 次串行安装 |
| 6 | 插件归档下载与解压 | `pending.json` (4.6s) + 19 个插件归档串行 `rclone copyto` 与 `tarfile.extractall` (142.9s) | **147.5** | 19.1% | 每个归档单独拉起一次 `rclone` 进程（855B 小文件亦耗 5.2s），299.4 MB 有效吞吐仅 ~2.1 MB/s；`LanPaint` 单包占 191.8 MB（64%） |
| 7 | 插件全量扫描 #1 | `persistence.restore()` 末尾 `inventory(stage, snapshot_databases=False)` | **41.5** | 5.4% | 单线程对 446 MB 文件（含大量 GIF/PNG/MP4/CSV/字体）逐块执行 Python `SECRET` 正则扫描 |
| 8 | 插件全量扫描 #2 | `persistence.dependencies()` 开头 `inventory(self.nodes)` | **42.3** | 5.5% | **纯重复扫描**：`stage.rename(self.nodes)` 导致路径变化、`_cached_checked_source` 命中率为 0，0ms 后再扫 446 MB |
| 9 | 插件依赖安装 | `cm-cli restore-dependencies` + `protect_environment` + `pip check` | **283.2** | **36.7%** | **最大单点瓶颈**：`comfyui-impact-pack` 依赖 `git+https://github.com/facebookresearch/sam2`，每次冷启现场 `nvcc` 编译 `sam2/_C.so` 独占 **174.8s**；其余依赖由 Manager 逐项安装耗 **104.3s**（`ultralytics` 31.7s、`uuid7` 15.4s、`aiohttp-socks` 12.5s、`smp` 11.1s、`clip_interrogator` 10.4s 等）；`pip check` 等 4.1s |
| 10 | ComfyUI 启动 | 进程引导 + Manager prestartup (6.3s)<br>**`studio_native_guard` 全量扫描 #3 (42.4s)**<br>核心/CUDA/aimdo 初始化 (~21.0s)<br>19 个插件导入 (~30.5s) + 健康/Manager 校验 (0.9s) | **101.1** | 13.1% | **第 3 次纯重复扫描**：ComfyUI 新进程空缓存下 `prestartup()` 再次全量扫 446 MB 耗 **42.4s**，只为命中 6 秒前刚写入的 `plugin-dependencies.json`！节点导入中 `prompt-assistant` 8.1s、`comfyui-easy-use` 7.8s、`forbidden-vision` 5.4s（每次冷启从 HF 现下 3 个默认模型） |
| 11 | 隧道就绪与收尾 | `cloudflared` 启动 (0.0s，1s 连上 Cloudflare)<br>调度检出完成与 `native.ready()` 延迟 (**57.9s**)<br>`uploading` 持久化 `native-ready.json` + `metadata.json` 至 Drive (**10.9s**)<br>`cleaning_up` → `completed` (**2.3s**) | **71.1** | 9.2% | `status.json` 于 `05:18:00.5Z` 完成，worker 至 `05:18:58.5Z` 才推进（受主循环同步 CCU 采样、失效账号 OAuth 报错、30s WebSocket 超时重试与 `native.ready()` 串行 7 次 HTTP 探针影响）；211B + 443B 报告上传并回读校验 Drive 耗 10.9s |
| | **合计** | | **771.8** | **100.0%** | |

### 1.3 实测细分数据佐证

1. **`inventory()` 冷扫描耗时分布（实测总计 40.45s，热缓存 0.58s）**：
   - `LanPaint`：**17.49s**（193 文件 / 194.8 MB，主要是 `examples/` 下 GIF/PNG/MP4 动图）
   - `comfyui-autocomplete-aaalice`：**9.38s**（54 文件 / 106.2 MB，单文件 `danbooru_tags_cooccurrence.csv` 占 98.5 MB）
   - `zml-image`：**5.05s**（139 文件 / 58.5 MB，主要是 `.ttf/.otf` 字体与 6.8 MB 词库文本）
   - `comfyui-lora-manager`：**2.74s**（576 文件 / 33.3 MB）
   - 其余 15 个插件合计：**5.79s**
2. **启动后后台副作用**：
   - `weilin-comfyui-tools` 启动时执行内置 SQLite 完整性检查，文件字节发生变化，导致启动后约 40 秒后台 `node_persistence` 自动触发一次 `save-snapshot`（`05:18:40Z`）并重新打包上传 `1.8 MB` 归档到 Drive（`05:18:54Z`），占用启动后期的 CPU/IO。

---

## 2. 优化目标与安全不变量

### 2.1 目标

| 阶段 | 范围 | 目标冷启动耗时 | 较基线（771.8s） |
| --- | --- | --- | --- |
| **Phase 1（低风险快赢）** | 不新增 Drive 缓存格式，纯代码重构：消除 3× 重复/低效扫描 + 插件并发下载解压 + `uv` 合并基础依赖 + 精简收尾与调度延迟 + 修复阶段埋点 | **430–470s（约 7.5 分钟）** | **-40%（省 ~5 分钟）** |
| **Phase 2（核心缓存加速）** | 叠加原生模式 Drive 增量 Python 环境缓存包（免每次编译 `sam2` 与全量重装依赖）+ 系统依赖并发安装 | **110–150s（约 2 分钟）** | **-82%（省 ~10.5 分钟）** |
| **Phase 3（长尾打磨）** | `forbidden-vision` 默认模型持久化 + 治理 `weilin-comfyui-tools` 启动后重复备份 + 可选插件归档瘦身 | **95–130s（约 1.5–2 分钟）** | **-85%** |

### 2.2 必须坚守的不变量（不可降级）

1. **PyTorch / CUDA 保护**：`protect_environment()` 与 `PIP_CONSTRAINT` / `UV_CONSTRAINT` 始终生效；任何缓存恢复或依赖安装均不得破坏镜像内置 `torch`/`torchvision`/`torchaudio`/`triton`/`nvidia-*` 版本。
2. **归档完整性与防凭据泄漏**：插件归档下载后必须校验 SHA-256 与路径安全边界（拒绝绝对路径、`..`、符号链接、超 4 GiB 解压炸弹）；插件备份前仍必须拦截文本/源码/配置文件中的硬编码凭据与私钥。
3. **环境缓存强隔离**：环境缓存 key 必须同时绑定 `ComfyUI commit`、Python 版本、基础镜像完整包清单哈希、GPU 架构/驱动标识与 `plugin_generation`，杜绝二进制扩展（如 `sam2/_C.so`）跨 CUDA/GPU 架构错配。
4. **隧道鉴权核验不放松**：[native.py](../../api/app/gpu/native.py) `ready()` 对 5 条未鉴权路径的 Access Service Token 防护探针必须全部执行并通过，只改并发方式，不删减探针项。
5. **本地模型语义不变**：`local` 模式下 `ComfyUI/models` 仍保持无软链接、按需完整下载与校验语义。

---

## 3. 分阶段优化详细设计

### Phase 1：低风险快赢优化（纯代码重构）

#### 1.1 拆分启动阶段埋点（可观测性前置）
- **改动文件**：`api/app/gpu/native_runtime.py`、`api/app/gpu/scheduler.py`、`api/app/job_history.py`、`web/src/lib/job-stage-rows.ts`
- **方案**：
  - 将原两次 `stages.start("native_environment")` 拆分为互不覆盖的细分阶段，并将无 Python 依赖的插件归档恢复前置至 Python 安装前（以便 `env-cache` 精确计算依赖指纹并合并恢复全部环境）：
    - `native_system`（`apt` + `rclone` + `cloudflared` 等并发初始化）
    - `native_drive`（Drive 挂载）
    - `native_models`（ComfyUI checkout 校验 + 目录与本地模型配置）
    - `native_plugins_restore`（并发下载、解压、校验插件归档）
    - `native_env_cache`（Phase 2 命中时：原子恢复 `site-packages` 增量环境缓存，跳过下方两步）
    - `native_python`（未命中缓存时：ComfyUI 与 Manager 基础 Python 依赖合并安装）
    - `native_plugins_deps`（未命中缓存时：安装插件 Python 依赖与 `pip check`）
    - `comfy_start`（拉起 ComfyUI、prestartup、节点导入、Manager 校验）
    - `native_tunnel`（启动 `cloudflared` 并等待本地就绪探针）
  - 同步补充 `scheduler.py` 的 `_STAGE_TEXT`、`job_history.py` 的 `_WORKER_KEYS` 白名单以及前端 `job-stage-rows.ts` 阶段中文映射，确保 API 历史详情与前端 UI 完整展示各细分阶段耗时。

#### 1.2 消除 3 次重复与低效的 `inventory()` 扫描（预计省 **115–125s**）
- **改动文件**：`api/app/gpu/node_persistence.py`、`api/app/gpu/native_runtime.py`
- **方案**：
  1. **非文本二进制资产免跑正则凭据扫描**：
     - 在 `checked_source()` / `_cached_checked_source()` 中，针对纯二进制扩展名（`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.ico`, `.mp4`, `.webm`, `.mov`, `.mp3`, `.wav`, `.ogg`, `.ttf`, `.otf`, `.woff`, `.woff2`, `.eot`, `.bin`, `.onnx`, `.pt`, `.pth`, `.safetensors`, `.pyc`, `.so`）以及头部含 `\x00` 的不可打印二进制文件，直接用 C 实现的 `hashlib.file_digest(f, "sha256")` 计算校验和，跳过逐块 Python `SECRET` 正则扫描。
     - 保留对 `.py/.js/.ts/.json/.yaml/.yml/.toml/.ini/.cfg/.txt/.md/.sh/.csv/.env*` 等文本与配置文件的完整 `SECRET` 正则 + AST 检查。
     - 对超大只读数据 CSV（如 `>10 MiB`）保留 SHA-256 校验，并在首次扫描通过后按 `(ino, size, mtime_ns, ctime_ns)` 持久化缓存结果到运行时本地索引（`/content/ai-studio/native/scan-cache.json`），供同一 VM 内不同进程复用。
  2. **合并/跳过 3 次冗余全量扫描**：
     - **扫描 #1（`restore()` 验证）**：`restore()` 解压的是已校验归档 SHA-256 的受控对象，验证语义一致性时新增 `check_secrets=False` 参数（仅算文件 SHA-256，不跑凭据正则），并返回计算出的 `plugins` 清单。
     - **扫描 #2（`dependencies()`）**：`dependencies(plugins=None)` 支持直接接收 `restore()` 刚算出的 `plugins` 清单，彻底消除 `rename` 后立即重扫一遍的 42.3s。
     - **扫描 #3（`studio_native_guard/prestartup_script.py`）**：`native_runtime.py` 在 `dependencies()` 成功并启动 ComfyUI 前，写入带快速指纹（各插件目录及顶层文件数、总字节数、最新 `mtime_ns`）的 `plugin-dependencies.json` 与一次性启动标记（含时间戳，TTL 60s）；`prestartup()` 检测到该标记且快速指纹未变化、且不是 Manager 触发的重启时，直接跳过全量 `inventory()`（仅在快速指纹变化或 Manager 重启时回退全量校验）。
  3. **多核并行扫描、C 级前缀过滤与消除隐藏的第 4/5 次扫描**：
     - 在 `inventory()` 中使用 `ThreadPoolExecutor(max_workers=min(8, len(candidates), os.cpu_count() or 4))` 并行扫描各个插件目录（`hashlib` 释放 GIL，可充分利用 Colab 12 vCPU）。
     - **新增 C 级子串预筛（零漏报）**：原 `SECRET` 正则含首部大小写不敏感分支 `(?i:...)`，导致 Python `re` 无法启用前缀快跳，在 `danbooru_tags_cooccurrence.csv`（98.5 MB 纯文本 CSV，无法走二进制白名单）上逐字节回溯耗 9.38s。现先用 C 实现的 `bytes` 子串包含检查（`access_token`/`refresh_token`/`client_secret`/`api_key`/`password`/`gh[pousr]_`/`hf_`/`-----BEGIN `）预筛 1 MiB 块，仅命中候选词时才运行完整正则与 AST 校验，使 98.5 MB CSV 扫描降至 `<0.1s`。
     - **消除 ComfyUI 扩展导入与状态探针中的隐藏重复扫描**：通过跨进程持久化的 `scan-cache.json`，令 ComfyUI 子进程内 `studio_native_state.py` 的 `install()`（原隐藏的第 4 次扫描）直接命中缓存；同时修复 `/studio_native/state` 在单次请求中串联调用 `probe()` 与 `PERSISTENCE.status()` 重复执行两次 `inventory()` 的问题。

#### 1.3 插件归档并发下载与并行解压（预计省 **125–138s**）
- **改动文件**：`api/app/gpu/node_persistence.py`
- **方案**：
  1. **单次批量并发下载**：
     - `restore()` 先筛出本地 `plugin-cache/` 缺失或 SHA-256 不匹配的归档列表 `missing_names`。
     - 将 `missing_names` 写入临时文件，仅调用 **1 次** `rclone copy`：
       ```bash
       rclone --config rclone.conf copy \
         --files-from <missing_list> \
         --ignore-times --fast-list \
         --transfers 8 --checkers 8 \
         studio:native-comfy/plugins/objects <plugin-cache>
       ```
     - 下载结束后对每个目标文件仍逐一校验 `file_hash(archive) == item['archive']`；若有缺失立即抛出明确缺失插件名错误。
  2. **多线程/多核安全解压**：
     - 保留对每个 tar 成员的安全检查（总大小 ≤ 4 GiB、无绝对路径/符号链接/`..`、重置 uid/gid 与权限掩码）。
     - 使用 `ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))` 并行校验与解压 19 个归档；对单个大包（如 >20 MB 的 `LanPaint`、`autocomplete`、`zml-image`）优先通过 `pigz -dc` 管道流式喂给 `tarfile.open(fileobj=..., mode="r|")` 或多线程解压，充分利用多核 CPU。

#### 1.4 合并基础 Python 依赖安装并优先使用 `uv`（预计省 **18–25s**）
- **改动文件**：`api/app/gpu/native_runtime.py`
- **方案**：
  - 将原本 4 次串行 `python -m pip install`（`jedi`、`ComfyUI/requirements.txt`、`httpx`、`manager_requirements.txt`）合并为 **1 次安装命令**。
  - 若 `uv_bin := shutil.which("uv")` 存在，执行：
    ```bash
    uv pip install --system --python <sys.executable> \
      --constraint <ROOT/gpu-constraints.txt> \
      "jedi>=0.16,<0.20" "httpx>=0.27,<1" \
      -r <comfy/requirements.txt> -r <comfy/manager_requirements.txt>
    ```
  - 若 `uv` 不可用，自动回退到合并后的单次 `python -m pip install -q ...`。

#### 1.5 缩短控制面轮询、收尾检测与报告上传耗时（预计省 **20–30s 确定耗时 + 消除 58s 抖动**）
- **改动文件**：`api/app/worker.py`、`api/app/gpu/scheduler.py`、`api/app/gpu/native.py`、`api/app/gpu/native_runtime.py`
- **方案**：
  1. **原生启动报告免同步上传 Drive（省 ~10.9s）**：
     - 在 `Scheduler._upload()` 中，当 `(job.get("snapshot") or {}).get("native_comfy")` 为 True 时，强制使用本地存储写入 `native-ready.json` 与 `metadata.json`（不触发 `DriveStorage.put` + `verify` 的双次网络往返），因为原生会话产物与工作流本身已由 Drive FUSE 持久化。
  2. **推进态免 `sleep(2)`（省 ~10s）**：
     - 在 [worker.py](../../api/app/worker.py#L80-L87) 主循环中，当 `result.get("advanced")` 为 True 时跳过本轮 `time.sleep(args.interval)`，直接进入下一状态跃迁（`queued → validating → waiting_for_capacity → allocating → preparing → loading → running` 连续推进，无活跃变更时才按 2s 休眠）。
  3. **解耦 CCU 采样并跳过无效账号**：
     - 将每 60s 的全账号 `tick_ccu` 从 `worker.py` 主同步循环抽离到独立后台线程（或交由独立轻量计时循环），避免 OAuth 刷新/网络超时阻塞正在轮询启动的 job；同时对未绑定有效 Colab 凭据或已失效的账号做退避，避免日志每分钟持续刷 `GoogleOAuthError`。
  4. **加速 `native.ready()` 与消除隧道竞争**：
     - 在 [native.py](../../api/app/gpu/native.py#L625-L643) 中使用 `ThreadPoolExecutor`（或 `httpx.AsyncClient`）并发执行 5 条未鉴权 Access 探针，单次 `ready()` 耗时从 ~2.8s 压至 `<1s`。
     - 在 [native_runtime.py](../../api/app/gpu/native_runtime.py#L634-L648) 启动 `cloudflared` 后，轮询 `http://127.0.0.1:20241/ready`（cloudflared 自带本地指标端口，上限等待 10s，间隔 0.3s）确认至少 1 条边缘连接已注册后再写 `status.json = completed`，减少控制面在隧道尚未就绪时撞上超时重试的概率。

---

### Phase 2：核心大提速（原生环境增量缓存包 + Phase 1 并行）

#### 2.1 原生模式 Drive 增量环境缓存包（预计省 **280–305s**）
- **改动文件**：`api/app/gpu/native_runtime.py`、`api/app/gpu/node_persistence.py`、`api/app/gpu/runtime_worker.py`（复用 `_site_state` / `_gpu_cache_tag` 逻辑）
- **核心思路**：
  将当前 VM 上 **ComfyUI 基础依赖 + Manager + 全部 19 项插件依赖（含已编译好的 `sam2/_C.so` 等原生扩展）** 对 `site-packages` 的增量改动打成一个压缩包存到 Drive `native-comfy/env-cache/`。下次全新 L4 冷启动命中同一缓存 key 时，直接流式解压覆盖 `site-packages` 增量，完全跳过 `pip`/`uv` 解析与 `sam2` 174.8s 的 `nvcc` 源码编译！

- **缓存 Key 设计（强绑定、零错配）**：
  ```text
  native-env-v1-{comfy_commit[:12]}-py{major}{minor}-img{base_pkg_sha[:12]}-gpu{gpu_tag}-plug{plugin_gen[:16]}
  ```
  - `comfy_commit`：锁定的 40 位 ComfyUI commit
  - `py{major}{minor}`：Python 主次版本（如 `py313`）
  - `base_pkg_sha`：安装任何包之前 Colab 基础镜像 `importlib.metadata.distributions()` 排序列表的 SHA-256（涵盖 `torch`、CUDA 库与全部预装依赖）
  - `gpu_tag`：`_gpu_cache_tag()`（`nvidia-smi` 的 `name,compute_cap,driver_version` 哈希，保证 `sam2/_C.so` 等 CUDA 扩展严格匹配同型号 GPU 与驱动）
  - `plugin_gen`：`manifest["generation"]`（64 位插件代次哈希；任何插件新增、升级、删除、禁用都会自动换 key 失效）
  - 附加：`comfy/requirements.txt` 与 `manager_requirements.txt` 内容哈希纳入 key 摘要（防止极端情况下的本地变更）

- **缓存包结构与完整性校验**：
  - Drive 路径：
    - `studio:native-comfy/env-cache/{key}.tar.gz`（使用 `pigz -1` 快速压缩）
    - `studio:native-comfy/env-cache/{key}.meta.json`（包含 `schema=1`、`key`、`archive_sha256`、`size`、`removed`、`protected_packages`、`dependency_key`、`created_at`）
  - 仅在 `.meta.json` 和 `.tar.gz` 均存在且实际下载文件的 SHA-256 严格等于 `meta["archive_sha256"]` 时才视为命中；任何校验失败自动回退到正常安装流程（fail-open to clean build，不阻断启动）。

- **执行流程**：
  1. **记录基线**：在安装任何 Python 包前，调用 `before_site = _site_state(site)` 并计算 `env_key`。
  2. **尝试命中缓存**：
     - `rclone` 拉取 `env-cache/{key}.meta.json`，若存在则拉取 `{key}.tar.gz` 并验证 SHA-256。
     - 使用 `pigz -dc` 解包 `site/` 增量到 `site-packages`（按 `meta["removed"]` 清理被替换条目），写入对应的 `plugin-dependencies.json` 与已验证标记。
     - 二次核验 `protect_environment()` 受保护包（`torch`/`nvidia-*` 等）与镜像基线完全一致，并执行 `pip check`（实测 1.3s）。
     - **命中时总耗时**：下载（~150–250 MB 约 3–6s）+ `pigz` 解包（约 3–5s）+ 校验（~1.5s）= **约 8–15s**，替代原来的 **316s**！
  3. **未命中时（首次构建或插件依赖变更后）**：
     - 走优化后的常规安装：汇总启用插件的 `requirements.txt` 单次 `uv pip install --system` 批量预装，并自动探测当前 GPU `compute_cap`（如 L4 为 `8.9`）注入 `TORCH_CUDA_ARCH_LIST` 与 `MAX_JOBS`，使冷构建下 `sam2/_C.so` 仅针对当前 GPU 架构编译（冷构建 `nvcc` 耗时从 174.8s 大幅压减至 ~25–35s），随后执行官方 `cm-cli restore-dependencies` 兜底运行 `install.py` 脚本。
     - 安装与 `pip check` 通过后立即快照 `after_site` 增量（含新增 `.dist-info/RECORD` 引用的顶层命名空间包目录）生成 `env-cache-plan.json`；待 ComfyUI 与隧道就绪后，拉起**独立后台子进程**（`start_new_session=True`，避免 `runtime_worker.py` 写完 `status.json` 退出时连带终止守护线程！）用 `pigz -1` 打包上传至 Drive `env-cache/`（先传 `.tar.gz`，最后原子写入并校验 `.meta.json`），同时按 LRU 保留最近 3 个缓存包。
     - 缓存命中解包时在 `site.parent` 同文件系统下创建临时目录与回滚目录，以 `O(1)` 原子 `rename` 替换并在 `protect_environment()` 或 `pip check` 异常时无损回滚基线环境。

- **额外收益：冷构建期的 `requirements.txt` 批量合并**：
  - 观察本次实测：18 个插件里只有 `comfyui-impact-pack` 有 `install.py`，其余 11 个有依赖的插件都只有纯 `requirements.txt`。
  - 当前 `cm-cli restore-dependencies` 逐个插件反复启动安装器，导致除 `sam2` 之外的 10 多个小包（`ultralytics`、`uuid7`、`aiohttp-socks`、`smp`、`clip_interrogator`、`onnxruntime`、`color-matcher`、`mss` 等）耗时 **104.3s**。
  - 若在未命中 env-cache 时，先汇总全部启用插件的 `requirements.txt`（跳过已在基线满足的版本，约束 `gpu-constraints.txt`）用一次 `uv pip install --system` 批量安装，再仅对含 `install.py` 的插件执行脚本，即使缓存完全未命中也能把这部分从 104s 压至 ~20–30s。

#### 2.2 系统依赖、二进制与 ComfyUI 源码并发准备（预计省 **10–15s**）
- **改动文件**：`api/app/gpu/native_runtime.py`
- **方案**：
  - 用 `ThreadPoolExecutor(max_workers=4)` 并行执行 4 项互不依赖的初始化任务：
    1. `apt-get update -qq && apt-get install -y -qq --no-install-recommends fuse3 mesa-vulkan-drivers`
    2. `install_rclone(ROOT, worker)`
    3. 下载 `cloudflared`
    4. `_checkout_comfy(ROOT / "ComfyUI", commit, 180)` + `_verify_checkout`
  - 4 项完成后再执行 `mount_drive()`（依赖 `fuse3` + `rclone`）与 `prepare_paths()`（依赖 `mount` + `ComfyUI`）。
  - 预计将原串行 `23.7s + 6.3s = 30.0s` 降至 **~15–18s**（受限于 `apt` 单项耗时）。
  - 补充检查：若 `fuse3` 需等 `apt`，可在 `mount_drive` 前只等待 `apt` + `rclone`，而 `cloudflared` 实际到最后一步才用，完全可放到后台线程与整个环境准备并行！

---

### Phase 3：启动期副作用治理与可选深度优化

1. **`forbidden-vision` 每次冷启下载默认模型（省 ~5.4s）**：
   - 实测日志显示 `forbidden-vision` 导入时每次均向 HuggingFace 无鉴权下载 `face_detect`、`face_segment`、`neural_corrector` 3 个模型。
   - 排查该插件模型默认存放路径，在 `prepare_paths()` 中将该目录持久化或缓存到 Drive `native-comfy/node-assets/forbidden-vision/`，冷启直接复用。
2. **治理 `weilin-comfyui-tools` 启动后重复触发备份**：
   - 实测 `weilin-comfyui-tools` 启动时执行 SQLite 自检导致文件内容变更，启动后 40s 触发 `node_persistence` 后台重跑 `save-snapshot` 并重传 `1.8 MB` 归档，不仅消耗启动后 CPU/IO，还会导致 `plugin_generation` 每次重启无故漂移（从而使 Phase 2 的 `env-cache` 缓存 key 失效！）。
   - **重要联动**：在落地 Phase 2 `env-cache` 前，必须同步核查 `weilin-comfyui-tools/user_data/*.db` 为什么会在启动后改变校验和；可考虑在 `env-cache` 的 key 中仅纳入影响依赖的清单字段（`path` + `enabled` + `requirements.txt` 哈希 + `install.py` 哈希 + `pyproject.toml` 版本，而非整个插件数据文件的全量 `generation`），这样即使某个插件修改了运行时 SQLite 数据或配置，也**不会让环境缓存失效**！
3. **超大示例媒体瘦身（可选）**：
   - `LanPaint/examples/` 占全部 19 个插件归档的 64%（191.8 MB / 299.4 MB）。在 Phase 1 并发下载 + 二进制跳过正则后，其耗时已从 ~26s 降至 ~4–6s；是否需要在归档排除列表中排除纯示例大媒体可作为后续可选项，不强依赖。

---

## 4. 关键风险与设计细节核对

| 风险点 | 影响 | 缓解措施 |
| --- | --- | --- |
| `env-cache` key 绑定了完整 `plugin_generation`，某个插件写内置 DB（如 `weilin-comfyui-tools`）导致缓存频繁失效 | 命中率骤降 | `env-cache` key 改用**依赖指纹**（所有启用插件的 `path`、`requirements.txt` 内容哈希、`install.py` 内容哈希、`comfy_commit`、镜像基线与 GPU 架构），与业务数据文件解耦 |
| 跳过二进制文件的 `SECRET` 正则扫描后，是否会漏检伪装后缀的凭据 | 安全边界回归 | 仅对明确白名单二进制后缀 **且** 文件头检测确为二进制魔数（PNG/GIF/JPEG/WEBP/MP4/TTF/OTF/WOFF 等）的文件跳过正则；后缀与魔数不符时仍按文本全量扫描 |
| `rclone copy --files-from` 部分文件缺失或 Drive 限流 | 恢复不完整 | `--transfers 8` 保守并发并加 `--retries 3 --low-level-retries 10`；下载后逐一强制核对每个归档的 SHA-256，缺一即报错 |
| 后台异步上传 `env-cache` 期间用户立即点击停止实例 | 上传中断或残留半成品 | 先传至临时对象名或利用 Drive 原子上传 + 最后再写 `.meta.json`（`.meta.json` 存在才视为有效缓存）；且上传线程持锁或注册到 `/studio_native/state` 时需注意**不能阻塞用户手动停止**（手动停止可直接中断后台缓存构建，保证释放速度优先） |
| `uv` 与旧版 `pip` 在某些 `git+https://` 或构建隔离行为上的差异 | 冷构建失败 | 保留环境变量开关 `NATIVE_USE_UV=1`（默认开启，失败可自动回退 `pip`），并在实机覆盖 `sam2` 编译回归测试 |

---

## 5. 验收计划与回滚策略

### 5.1 自动化单测（部署前）
- `node_persistence`：
  - 二进制媒体/字体魔数校验通过时跳过正则，伪装成 `.png` 的纯文本秘钥仍能被拦截；
  - `restore()` → `dependencies()` → `prestartup()` 仅执行 1 次完整目录哈希；Manager 重启后 `prestartup()` 仍正常执行校验；
  - `rclone copy --files-from` 批量恢复与 SHA-256 校验失败回退路径。
- `native_runtime`：
  - 细分阶段 `worker_timings` 不再互相覆盖；
  - `env-cache` key 在 Python/Torch/GPU/Comfy/插件依赖任一变更时必然失配；校验和不匹配时安全回退常规安装。
- `scheduler` / `worker` / `native`：
  - `native_comfy` 任务 `_upload()` 不调用 Drive 存储；
  - `worker.py` 在 `advanced=True` 时免 sleep；
  - `native.ready()` 并发 5 路未鉴权探针任一返回 200 时仍 fail-closed 返回 `False`。

### 5.2 真实 L4 验收（两阶段实测对比）
1. **Phase 1 部署后实测**：全新 L4 冷启动，核对细分 `worker_timings` 与端到端总耗时应降至 **~7.5 分钟以内**，19 个插件 0 导入失败，正常加载并出图通过。
2. **Phase 2 部署后两轮实测**：
   - **第 1 轮（冷缓存构建）**：验证常规冷启动耗时、后台异步写入 Drive `env-cache/` 成功且不阻塞就绪。
   - **第 2 轮（全新 L4 命中缓存）**：停止旧实例并重开全新 L4，验证 `env-cache` 命中、`sam2` 免重编、端到端总耗时降至 **~2 分钟左右**，`pip check` 通过、19 项插件 0 导入失败、Anima 完整工作流出图通过。
   - **变更回归**：确认停止实例后零残留租约，修改/新增插件依赖时自动失效旧缓存并正常重建。

### 5.3 回滚与独立环境变量开关
- 所有优化项均提供独立环境变量开关（默认均为 `1` 开启，可在 VPS `.env` 中独立配置，并由 `mcp.py` 随启动载荷自动透传同步至远端 Colab 运行时，切换验收无需改代码）：
  - `NATIVE_FAST_SCAN=1|0`（Phase 1.2）：二进制魔数快速哈希、C 级凭据预筛、并行 `inventory()`、`scan-cache.json` 跨进程复用及 `restore -> dependencies -> prestartup` 三重扫描去重。
  - `NATIVE_PARALLEL_RESTORE=1|0`（Phase 1.3）：`rclone copy --files-from-raw` 批量并发拉取与 `pigz`/多线程安全解压（关闭则回退串行 `rclone copyto`）。
  - `NATIVE_USE_UV=1|0`（Phase 1.4）：合并基础与 Manager 依赖安装并优先使用 `uv pip install --system`（失败或关闭自动回退合并 `python -m pip`）。
  - `NATIVE_FAST_CONTROL_PLANE=1|0`（Phase 1.5）：原生启动报告本地化存储、`worker.py` 推进态免 `sleep(2)`、后台异步 `tick_ccu` 与无凭据/失效账号退避、`native.ready()` 7 路并发探针及 `cloudflared` `:20241/ready` 就绪等待。
  - `NATIVE_ENV_CACHE=1|0`（Phase 2.1）：Drive `native-comfy/env-cache/` 增量 `site-packages` 环境缓存读写与原子回滚（关闭则跳过缓存读写，回退纯现场安装）。
  - `NATIVE_BATCH_PLUGIN_PIP=1|0`（Phase 2.1 冷构建加速）：未命中 `env-cache` 时批量合并插件 `requirements.txt` 预安装 + 注入当前 GPU `TORCH_CUDA_ARCH_LIST` / `MAX_JOBS` 加速源码扩展编译。
  - `NATIVE_PARALLEL_BOOTSTRAP=1|0`（Phase 2.2）：`apt-get`、`rclone`、`cloudflared`、ComfyUI checkout 与模型索引并发初始化。
- Drive 既有 `native-comfy/plugins/objects/` 与 `head.json`/`pending.json` 格式保持 **100% 向后兼容**，任何时候均可在 `bero` 一键切回 `3cba89e` 镜像。
