# 原生 ComfyUI 启动耗时二次诊断与优化设计（V2）

## 1. 背景与当前实测基线

在完成 [NATIVE_COMFYUI_STARTUP_OPTIMIZATION.md](NATIVE_COMFYUI_STARTUP_OPTIMIZATION.md) 第一批与第二批优化并部署至 `bero`（镜像 `colab-gpu-studio:92405569`）后，已验证多项优化显著生效：

- `native_plugins_restore`：从 **188.9s → 18.0–20.6s**（提速约 9 倍）
- `native_python`：从 **32.9s → 7.9s**（提速约 4 倍）
- `studio_native_guard` prestartup：从 **42.4s → 0.9s**（提速约 47 倍）
- `uploading` 报告落盘：从 **10.9s → 0.25s**

但在 2026-11-22 晚间实际使用中，**最新一次冷启动（`c1566a79-f03b-4603-8b11-a1476219e14a`）仍耗时 526.6 秒（8 分 46.6 秒）**，且此前 3 次**命中环境缓存**的启动（`ecda5563` / `7bd249b6` / `1639fdd7`）端到端仍需 **~224 秒（3 分 44 秒）**，距离目标（热缓存 ~2 分钟、冷构建 ~4–5 分钟）仍有明显差距。

本文基于 `bero` 调度事件记录与运行中 L4 Colab 实例的内部取证，系统梳理当前残留的 **5 类核心问题** 并给出第二轮针对性优化方案。

### 1.1 三组 L4 实测样本对照

配置一致：`colab cli账号` / `L4 · SHAPE_STANDARD` / `local` 模式 / 19 项归档插件。

| 阶段 | 优化前基线 `3cba89e`<br>(`e508d30a`) | 新版命中缓存<br>(`1639fdd7`) | 新版最新冷构建 `92405569`<br>(`c1566a79`) | 问题定位摘要 |
| --- | ---: | ---: | ---: | --- |
| 控制面分配 + Worker 启动 | 21.7s | 10.4s | 14.5s | 正常 |
| `native_system` | 23.7s | 24.4s | 23.7s | 受 `apt-get` 主导 |
| `native_drive` | 0.6s | 0.6s | 0.6s | 极快 |
| `native_models` | 6.3s | 3.3s | 3.9s | 正常 |
| `native_plugins_restore` | 188.9s | 18.0s | 20.6s | 优化已见效 |
| **`native_env_cache`** | — | **67.1s** | **5.0s (Miss)** | **问题 2**：缓存包异常膨大至 1.58 GiB；本次因 key 变更 miss |
| `native_python` | 32.9s | 0s (跳过) | 7.9s | 优化已见效 |
| **`native_plugins_deps`** | 325.5s | 0s (跳过) | **339.6s** | **问题 1**：`uv` 批量装完后仍无条件重跑 `cm-cli` 重复编译 `sam2` |
| **`comfy_start`** | 101.1s | **71.0s** | **74.0s** | **问题 3**：`studio_native_state.py` 同步冷扫占 13.9s；`forbidden-vision` 现下模型 8.0s |
| `native_tunnel` | 0.0s | 0.8s | 1.0s | 极快 |
| **完成到 `uploading` 检出空窗** | 57.9s | **28.2s** | **34.6s** | **问题 4**：冷/热启动均稳定出现 28–35s 收尾空窗 |
| `uploading` + `cleaning_up` | 13.2s | 0.5s | 1.0s | 优化已见效 |
| **端到端总耗时** | **771.8s (12m52s)** | **224.3s (3m44s)** | **526.6s (8m47s)** | |

---

## 2. 现存问题清单与实测根因

### 问题 1（P0 · 冷构建最大卡点）：`dependencies()` 双重安装依赖并重复编译 `sam2`（多耗 **~223 秒**）

- **现象与时间线证据（任务 `c1566a79`）**：
  - `13:06:44.661Z`：`native_plugins_deps` 开始（写入 `plugin-install-notes.json`）。
  - `13:06:44.7Z` → `13:08:40.041Z`（**115.4 秒**）：`NATIVE_BATCH_PLUGIN_PIP` 调用 `uv pip install -r ...` **已将全部 23 个插件包（含 `sam_2-1.0` 编译 `sam2/_C.so`、`ultralytics` 等）一次性安装完成**（所有 `.dist-info` 写入完成时间戳均为 `1795352920.03–.04`）。
  - `13:08:40.041Z` → `13:12:23.198Z`（**223.2 秒 / 3 分 43 秒黑洞**）：所有依赖其实已经装好，但流程又卡了 3 分 43 秒才写入 `plugin-dependencies.json`。
- **代码根因**：
  - [node_persistence.py](../../api/app/gpu/node_persistence.py) `Persistence.dependencies()`（第 871–895 行）在 `uv` 批量安装成功之后，**仍然无条件执行了 `cm-cli.py restore-dependencies`**。
  - `comfyui-impact-pack/requirements.txt` 包含 VCS 依赖 `git+https://github.com/facebookresearch/sam2`。pip/Manager 对 VCS URL **不会判定为已满足**，导致 `cm-cli` 再次拉起 `pip` 联网解析并重跑安装/编译（实测 `13:09:15Z`–`13:10:07Z` 期间 `/root/.cache/pip/http-v2` 写入上百个缓存条目）。
  - 紧接着在 `13:11:14Z` 执行 `comfyui-impact-pack/install.py`，现场下载 **375 MB** 的 `ComfyUI/models/sams/sam_vit_b_01ec64.pth`；随后继续遍历其他插件检查直到 `13:12:23Z`。
- **额外发现（行为不一致隐患）**：
  - 命中 `env-cache` 时不会执行 `install.py`，因此新 VM 上 `ComfyUI/models/sams/sam_vit_b_01ec64.pth` 并不存在；而冷构建时会由 `install.py` 现下 375 MB 到本地临时盘（停机即丢）。

---

### 问题 2（P0 · 缓存命中最大卡点）：`env-cache` 增量包被错误膨大至 `1.58 GiB`（多耗 **~50–55 秒**，并拖慢收尾）

- **现象与证据**：
  - Drive 上现有两个缓存包体积均高达 **`1.58 GiB`**（`1,657,288,993` 字节与 `1,657,298,860` 字节）。
  - 导致前 3 次命中缓存时，单 `native_env_cache` 下载 + 解压就占了 **65.0–67.8 秒**（远高于正常 150–250 MB 增量包的 10–15 秒）。
  - 本次冷构建在 `13:13:38.6Z` 启动后台打包上传 `1.58 GiB`（`env-cache.log` 于 `13:15:10Z` 结束，耗时 **91.4 秒**），高并发 `pigz` + `rclone` 立即打满 CPU 与带宽，直接叠加了就绪检出延迟。
- **代码根因**：
  - 检查 Colab 上 `env-cache-plan.json` 的 `added`（共 141 项），赫然包含 Colab 镜像原本预装且受保护的巨型包：
    - `"torch"`
    - `"torchvision"`
    - `"triton"`
    - 以及 `"sympy"`、`"mpmath"`、`"PIL"`、`"pip"`、`"aiohttp"`、`"huggingface_hub"`、`"tabulate"`、`"toml"` 等预装包目录。
  - 根源在于 [node_persistence.py](../../api/app/gpu/node_persistence.py) `site_snapshot()`（第 1091–1100 行）与 `prepare_env_cache_plan()`（第 1233–1276 行）通过对比顶层目录的 `lstat()`（`st_ino, st_size, st_mtime_ns, st_ctime_ns`）判断变更：
    ```python
    changed = {name for name, st in after_site.items() if before_site.get(name) != st}
    ```
    当 `uv`/`cm-cli`/`install.py` 运行过程中 Python `import torch / torchvision / triton / sympy / PIL` 时，解释器会在这些预装目录内生成或更新 `__pycache__/*.pyc`，**改变了目录本身的 `mtime_ns` / `ctime_ns`**，从而被误判为“新安装/已修改”，把 **超 1.4 GiB 的镜像预装包整目录打进了增量缓存包**！
  - 此外，由于 `torch`/`torchvision`/`triton` 被打入缓存包，每次命中缓存解包时还会先将原系统目录移入 rollback 再移入解压副本，白白增加大量磁盘 I/O 与失败回滚风险。

---

### 问题 3（P1 · ComfyUI 启动卡点）：`studio_native_state.py` 同步导入期冷跑 `inventory()` 占 **13.9 秒**

- **现象与证据**：
  - `comfy.log` 显示 `studio_native_guard` prestartup 耗时已压降至 **`0.9 seconds`**。
  - 但在 `Import times for custom nodes:` 中，自家扩展 `studio_native_state.py` 耗时 **`13.9 seconds`**，成为全部 20 个扩展中**最慢的第一名**（超过 `comfyui-easy-use` 8.4s、`forbidden-vision` 8.0s、`prompt-assistant` 8.0s）。
- **代码根因**：
  1. `restore()` 调用 `inventory(stage, snapshot_databases=False, check_secrets=False)` 时（第 800–804 行），因 `check_secrets=False` 导致 `cache_path = None`，**未预热写入 `scan-cache.json`**。
  2. 随后 `dependencies(plugins=restored_plugins)` 与 `prestartup()` 快速路径均跳过了 `inventory()`。
  3. 当 ComfyUI 同步导入 `studio_native_state.py` 时，[node_persistence.py](../../api/app/gpu/node_persistence.py) `install()`（第 1372 行）同步执行 `boot_fingerprint = digest(inventory(persistence.nodes))`，成为整台 VM 上**第一次**带 `check_secrets=True` 的冷扫描，直接阻塞 ComfyUI HTTP 服务监听 **13.9 秒**（尤其是 98.5 MB `danbooru_tags_cooccurrence.csv` 在 `_may_contain_secret()` 中逐 MiB 做 `chunk.lower()` 拷贝与子串扫描 + 3 个 SQLite 数据库快照）。
  4. 实际上，`boot_fingerprint` **仅在 `async def loop()` 等待 `healthy()` 通过之后（第 1400 行）才被赋值使用**，完全没有必要在 `install()` 同步导入阶段阻塞 ComfyUI 启动。

---

### 问题 4（P1 · 稳定复现的收尾延迟）：`status.json = completed` 后存在 **28–35 秒**检出空窗

- **现象与证据**：
  - **热缓存样本 `1639fdd7`**（无后台 `env-cache` 上传）：Worker 阶段于 `10:10:33.8Z` 完成，但直到 `10:11:01.98Z` 才进入 `uploading`（空窗 **28.2 秒**）。
  - **冷构建样本 `c1566a79`**：`exec/status.json` 写入完成时间戳为 `13:13:38.516Z`，`tunnel.log` 于 `13:13:38Z` 已连上 `sin13`，但直到 `13:14:13.087Z` 才进入 `uploading`（空窗 **34.6 秒**）。
  - 同次任务在运行期也出现多次 17–21 秒的状态推送延迟（如 `native_system` 晚 21.3s、`native_plugins_restore` 晚 17.6s 才写入事件）。
  - 且上述样本均**未**触发 `等待代理健康检查及内部隧道鉴权验证` 事件，说明耗时发生在 `Scheduler._run()` 的单次 `poll()` 或首次 `native.ready()` 调用内部（未抛错返回 False，而是慢返回或内部重试后成功）。
- **需重点排查与治理的环节**：
  1. **后台 `env-cache` 打包上传抢占资源**：[native_runtime.py](../../api/app/gpu/native_runtime.py) 第 806–812 行在写入 `status.json = completed` **之前**就立即拉起了全核 `pigz` + `rclone` 上传进程，与控制面最终 poll、artifact 拉取及 `native.ready()` 争抢 CPU 与网络。
  2. **`GET /studio_native/state` 首次探针开销**：[native.py](../../api/app/gpu/native.py) `ready()` 并发请求 `/studio_native/state`；而 `studio_native_state.py` 收到 `/studio_native/state` 时会调用 `probe()`（内部扫一次 `inventory(nodes)`）以及 `PERSISTENCE.status()`（内部又扫一次 `inventory(nodes)`），同一请求内重复扫目录；在 `plugins-status.json` 与 `scan-cache.json` 尚未完全稳定时容易导致长耗时。
  3. **Colab 隧道轮询通道开销**：[mcp.py](../../api/app/gpu/mcp.py) `_execute_tunnel_code()` 每次 `poll()` 都重新 `GET /api/kernels/{id}` 并新建一次 WebSocket 连接执行短脚本，缺少单次 `poll()` 与 `native.ready()` 的分项耗时观测日志，遇到偶发 30s WebSocket 超时重试时无法从日志一眼定位。

---

### 问题 5（P2 · 缓存 Key 治理与插件启动期外网下载）

1. **`env-cache` Key 变更导致全量重建**：
   - 本次 `92405569` 在 `spec` 新增 `base_packages` 字段导致已有 1.58 GiB 缓存失效（属代码演进引起的一次性 miss），但说明后续任何非必要改动 `spec` 结构都会直接引发 5–8 分钟冷重建；需要固化 `spec` 版本策略并在文档中明确仅在依赖语义变化时升版。
2. **`forbidden-vision` 每次冷启现下 3 个默认模型（占 `comfy_start` 8.0 秒）**：
   - `comfy.log` 显示每次冷启动导入 `forbidden-vision` 时均向 HuggingFace 无鉴权下载 `face_detect`、`face_segment`、`neural_corrector` 3 个模型，本次耗时 **8.0s**（上午基线 5.4s，受 HF 匿名限速波动影响）。
3. **`weilin-comfyui-tools` 启动后仍触发重复持久化扫描**：
   - 虽已通过 `plugin_dep_fingerprint` 隔离了 `env-cache` Key 防止其失效，但 `weilin-comfyui-tools` 启动后仍会导致 `plugins-pending.json` / `plugins-verified.json` 重新写入与后台扫描，可在后续一并排查其 SQLite `integrity_check` 写入原因。

---

## 3. 第二轮针对性优化方案

### 方案 A（P0）：彻底瘦身 `env-cache`（`1.58 GiB` → `~150–250 MB`，命中缓存省 **~50–55 秒**）

- **改动位置**：`api/app/gpu/node_persistence.py`（`site_snapshot`、`prepare_env_cache_plan`、`try_restore_env_cache`、`upload_env_cache_from_plan`）
- **设计方案**：
  1. **改用 `.dist-info` 语义差集，彻底抛弃目录 `mtime` 启发式**：
     - 安装前记录基线快照：`before_dists = {d.name: dist_fingerprint(d)}`（仅扫描 `*.dist-info` 目录及其 `METADATA`/`RECORD` 哈希）以及 `before_entries = set(os.listdir(site))`。
     - 安装后计算：
       - `new_or_changed_dists`：新增或指纹发生变化的 `*.dist-info` 集合；
       - `new_top_entries`：`set(os.listdir(site)) - before_entries`（捕捉无 dist-info 的纯新增顶层文件/目录）。
     - 根据 `new_or_changed_dists` 的 `RECORD` 文件解析其拥有的顶层包/模块名，结合 `new_top_entries` 生成 `added` 列表。
  2. **显式硬过滤受保护与未变更基础包**：
     - 严禁将 `baseline_protected`（`torch`、`torchvision`、`torchaudio`、`triton`、`nvidia-*`）或版本未发生变化的基线包目录加入 `added`。
  3. **打包排除 `__pycache__` 与 `.pyc`**：
     - `tar.add(..., filter=...)` 跳过所有 `__pycache__` 和 `.pyc` 文件，进一步缩小归档体积与解压 I/O。
  4. **升级缓存 Schema 版本号（`native-env-v2-...`）**：
     - 新格式生效后自动与旧 1.58 GiB 膨大包隔离，并通过 `prune_env_cache` 清理旧包，释放 Drive 空间。
- **预期效果**：
  - 缓存包体积从 **1,580 MiB → ~150–250 MiB**（缩小 **6–10 倍**）；
  - 缓存命中恢复 `native_env_cache` 从 **67s → ~10–15s**；
  - 后台打包上传从 **91s → ~8–15s**，大幅降低对 CPU/网络的冲击。

---

### 方案 B（P0）：消除 `dependencies()` 双重安装与重复编译 `sam2`（冷构建省 **~180–210 秒**）

- **改动位置**：`api/app/gpu/node_persistence.py`（`Persistence.dependencies`）
- **设计方案**：
  1. **批量 `uv` 成功后，不再对全部插件重跑 `cm-cli restore-dependencies`**：
     - 拆分为两步：
       - **Step 1（全部 `requirements.txt` 批量去重安装）**：汇总全部启用插件的 `requirements.txt`，规范化去重后通过单次 `uv pip install --system --constraint gpu-constraints.txt` 安装（保留 `TORCH_CUDA_ARCH_LIST` 与 `MAX_JOBS`）。
       - **Step 2（仅执行含 `install.py` 的插件脚本）**：当 Step 1 返回码为 0 时，**仅对存在 `install.py` 的启用插件**（当前 19 项中仅 `comfyui-impact-pack` 1 个）直接受控运行其 `install.py`（继承 `COMFYUI_PATH`、`PIP_CONSTRAINT`、`UV_CONSTRAINT`），跳过 `cm-cli restore-dependencies` 对全部 `requirements.txt`（特别是 `git+https://github.com/facebookresearch/sam2`）的重复解析与二次编译。
     - **兜底回退**：仅当 Step 1 批量安装失败（或 `NATIVE_BATCH_PLUGIN_PIP=0`）时，才回退执行完整的 `cm-cli restore-dependencies`。
  2. **核查并隔离 `comfyui-impact-pack/install.py` 的默认模型下载**：
     - 核查 `impact-pack/install.py` 逻辑（如环境变量或配置项控制跳过/重定向 `sam_vit_b_01ec64.pth` 375 MB 自动下载，或将其持久化至 Drive 模型库），避免冷构建每次白下 375 MB 且与缓存命中态行为不一致。
- **预期效果**：
  - 冷构建 `native_plugins_deps` 从 **339.6s → ~120–140s**（省 **~200 秒**）。

---

### 方案 C（P1）：消除 `studio_native_state.py` 同步导入 13.9 秒阻塞（省 **~13 秒**）

- **改动位置**：`api/app/gpu/node_persistence.py`（`install`、`checked_source`、`Persistence.restore`）
- **设计方案**：
  1. **将 `boot_fingerprint` 计算移出 `install()` 同步导入路径**：
     - 在 `install()` 中仅完成轻量初始化与事件循环任务注册；
     - 在 `async def loop()` 中，通过 `asyncio.to_thread(lambda: digest(inventory(persistence.nodes)))` 异步计算 `boot_fingerprint`（与 ComfyUI 启动并发，或在 `healthy()` 后完成，不阻塞 `/system_stats` 就绪）。
     - 保持安全语义不变：`boot_fingerprint` 未完成前 `persistence.boot` 保持 `None`，绝不会提前晋升 `head.json`。
  2. **修复 `/studio_native/state` 单次请求内双扫 `inventory()`**：
     - `probe()` 与 `PERSISTENCE.status()` 在单次 `/studio_native/state` 响应中复用同一份 `inventory()` 结果（或以后台 `loop()` 每 10 秒刷新的 `plugins-status.json` + 目录快指纹判断脏状态），避免每次健康/空闲探针都触发两次全量扫描。
  3. **优化大文本/CSV 扫描拷贝开销**：
     - 对 `_may_contain_secret()` 避免对 1 MiB chunk 反复整块 `chunk.lower()` 分配临时内存（可改用预编译轻量字节正则或分块搜索），并在启动后后台预热 `scan-cache.json`。
- **预期效果**：
  - `studio_native_state.py` 导入时间从 **13.9s → <0.2s**，`comfy_start` 从 **71–74s → ~57–60s**。

---

### 方案 D（P1）：消除 `status.json = completed` 后 28–35 秒收尾空窗（省 **~20–30 秒**）

- **改动位置**：`api/app/gpu/native_runtime.py`、`api/app/gpu/scheduler.py`、`api/app/gpu/mcp.py`、`api/app/gpu/native.py`
- **设计方案**：
  1. **延迟并限流后台 `env-cache` 打包上传**：
     - 后台上传进程启动后先 `sleep(15)`（等待 VPS 完成 `poll` + `native.ready()` 并点亮前端就绪状态），并以低优先级（`os.nice(15)`、`pigz -1 -p 4`、`rclone --bwlimit` 可选）运行，绝不与就绪握手争抢 CPU/带宽。
  2. **补齐控制面轮询与 `native.ready()` 细粒度耗时观测日志**：
     - 在 `worker` 日志中记录单次 `poll()` 耗时（拆出 kernel check / WebSocket exec / chunk read）与 `native.ready()` 各子请求耗时（7 路探针最大耗时与 `/studio_native/state` 耗时），彻底定位热缓存样本 `1639fdd7` 中 28.2s 空窗的确切来源。
  3. **降低无意义的轮询连接开销（按需跟进）**：
     - 复用 `httpx.Client` 连接池并在 `poll()` 读取完 `status.json=completed` 的同一次脚本执行中直接内联返回极小的 `native-ready.json`（仅 ~300 字节），减少一次独立的 WebSocket 往返（`build_chunk_code`）。

---

### 方案 E（P2）：`forbidden-vision` 默认模型持久化（省 **~8 秒**）

- **改动位置**：`api/app/gpu/native_runtime.py`（`prepare_paths`）
- **设计方案**：
  - 确认 `forbidden-vision` 默认模型下载目录，将其缓存至 Drive `native-comfy/node-assets/forbidden-vision/` 并映射至节点预期目录（或仅在首次缺失时下载并回写 Drive），消除每次冷启向 HF 匿名下载 3 个模型带来的 5–8s 波动耗时。

---

## 4. 优化后预期耗时与验收标准

### 4.1 预期耗时目标

| 场景 | 当前实测 (`92405569`) | 完成 V2 方案 A–D 后预期 | 完成方案 E 后预期 |
| --- | --- | --- | --- |
| **命中缓存冷启动** | **224.3s（3 分 44 秒）** | **~120–140s（约 2 分 10 秒）** | **~110–130s（约 2 分钟）** |
| **缓存未命中冷构建** | **526.6s（8 分 47 秒）** | **~250–290s（约 4 分 30 秒）** | **~240–280s（约 4 分 15 秒）** |

### 4.2 验收检查项

1. **单测覆盖**：
   - `prepare_env_cache_plan()` 在预装目录仅新增 `__pycache__` 时**绝不**将 `torch`/`torchvision`/`triton` 列入 `added`；新增/升级普通包与带 C 扩展包（如 `sam2`）能完整按 `RECORD` 收入 `added` 并成功还原；
   - `dependencies()` 在 `uv` 批量安装成功时仅执行含 `install.py` 的插件脚本，不重复调用 `cm-cli restore-dependencies`；批量失败时正确回退 `cm-cli`；
   - `install()` 同步阶段不再调用阻塞式 `inventory()`，`boot_fingerprint` 异步就绪前不晋升 `head.json`；
   - `/studio_native/state` 单次请求最多触发一次目录扫描且不回归 `busy` 判真保护。
2. **真实 L4 两轮实测验收**：
   - **第 1 轮（V2 冷构建）**：确认 `native_plugins_deps` 降至 ~130s 以内、后台生成的 `native-env-v2-*.tar.gz` 体积降至 **<300 MiB**、不含 `torch/torchvision/triton`、收尾空窗显著缩短。
   - **第 2 轮（全新 L4 命中 V2 缓存）**：确认 `native_env_cache` 降至 **<15s**、`studio_native_state.py` 导入 **<0.5s**、端到端总耗时进入 **2 分钟级别**、19 项插件 0 导入失败、实际工作流出图验证通过且释放后零残留租约。
