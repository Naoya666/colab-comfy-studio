# 创作模式 GPU 启动与全链路耗时优化方案

## 1. 优化背景与瓶颈拆解

在完成原生 ComfyUI 多轮启动提速（[NATIVE_COMFYUI_STARTUP_OPTIMIZATION.md](NATIVE_COMFYUI_STARTUP_OPTIMIZATION.md)、[NATIVE_COMFYUI_STARTUP_OPTIMIZATION_V2.md](NATIVE_COMFYUI_STARTUP_OPTIMIZATION_V2.md)）后，对照审查创作模式（前端创作页 + 创作 API 统一常驻池 `POOL_CREATION`）的冷启动、切换工作流/LoRA、控制面调度及产物回传全链路，识别出以下核心瓶颈与缺陷：

1. **前端首次冷启被无效 CPU Prep 阻塞 2–5 分钟**：
   - `runtime_worker._env_key()` 拼接了 `_gpu_cache_tag()`，但仅有 CPU 运行时（`_run_prep`）以 `build=True` 打包环境缓存（无 `nvidia-smi`，生成随机 `unidentified-<pid>-<time_ns>` tag），导致 GPU 运行时 **100% 缓存未命中**；
   - `_restore_weights(prep=True)` 对已绑定在模型库的 `source == "drive"` 权重缺少 `if not prep:` 判断，CPU VM 会把已在 Drive 中的多 GB 权重全量下载校验后再删除；
   - `_pack_env()` 沿用目录 `st_mtime_ns` 启发式，易受 `__pycache__` 污染打包预装 `torch`/`torchvision`/`triton`。
2. **`restore_env` 串行两轮 `pip` 且阻塞权重下载（~30–40s）**：
   - `_pip_install()` 串行执行两次 `pip install`（`requirements.txt` + `huggingface_hub`），未启用 `uv` 也未写入 `gpu-constraints.txt` 保护预装 PyTorch/CUDA；
   - 整个 `restore_env` 与 `restore_weights`、串行 `_restore_user_loras` 严格串行。
3. **`restore_weights` 单流下载 + 二次全盘重读算 SHA-256（~40–100+s）**：
   - `DriveCache.fetch()` 使用单连接 `urllib` 边下边只算 MD5，落盘后 `_local_model_hit()` 再次全盘重读 10–20 GB 权重计算 SHA-256；
   - 同机切工作流或复验本地权重缺少 Stamp 缓存，每次重算全盘 SHA-256。
4. **同机切工作流或换用户 LoRA 重建整个 `ComfyUI/` 与 Python 环境（~30–40s）**：
   - `_creation_service()` 将代码版本、自定义节点与权重/LoRA 揉成单一指纹，换一个 LoRA 或切到同 ComfyUI 版本的工作流都会重走 `git checkout` + `pip install` 并删除旧工作流权重。
5. **控制面与后段收尾开销（~3–20s）**：
   - 每次进入 `loading`（含热复用）均串行向 Drive 查询 `owner`/`models`/`runtime-cache`/`loras/h3` 子目录 ID；前端准入串行 `check_remote()`；
   - `build_bootstrap_code()` 对创作任务仍携带原生专用的 `node_persistence.py` 与 `native_runtime.py`（~170 KB base64）；
   - MCP 产物回传每 1 MiB 重建一次 Jupyter WebSocket 连接；
   - `job_history.py` 白名单遗漏 `restore_loras`、`creation_warming` 与 `creation_python`。

---

## 2. 落地优化清单

### 2.1 冷启动提速（P0）
- **跳过纯 Drive 模型的自动 CPU Prep 并修复 `_run_prep` Bug**（[scheduler.py](../../api/app/gpu/scheduler.py)、[runtime_worker.py](../../api/app/gpu/runtime_worker.py)）：
  - 当工作流所有权重均为 `source == "drive"`（模型库绑定）且 `CREATION_SKIP_DRIVE_PREP=1` 时，直接跳过自动 CPU Prep；
  - 保留未缓存 HuggingFace 权重的 CPU Prep 能力，修复 `prep=True` 对 `source == "drive"` 权重仅校验元数据而不下载字节，并不再打包不可复用的 CPU `env` 归档。
- **`uv` 合并安装 + `gpu-constraints.txt` + 自定义节点回退能力**（[runtime_worker.py](../../api/app/gpu/runtime_worker.py)）：
  - `_pip_install()` 写入 `gpu-constraints.txt` 锁定预装 `torch`/`torchvision`/`torchaudio`/`triton`/`nvidia-*`，合并 `huggingface_hub` 与 `ComfyUI/requirements.txt`（及未来可能有的自定义节点依赖）单次 `uv pip install --system` 安装，失败自动回退合并 `pip`，安装后校验受保护包未被篡改；
  - 保留 `REVIEWED_CUSTOM_NODES` 并发 shallow clone 能力，并将 `_pack_env`/`_unpack_env` 升级为基于 `.dist-info/RECORD` 语义差集 + `pigz` + 同挂载点原子回滚的 v2 引擎（由 `CREATION_ENV_CACHE=auto|1|0` 控制，无额外依赖时默认跳过 tar 缓存直接走 `uv`）。
- **全流水线并发启动**（[runtime_worker.py](../../api/app/gpu/runtime_worker.py)）：
  - 在 `CREATION_PARALLEL_BOOTSTRAP=1` 下，完成 ~2–3s 的 ComfyUI checkout 后，后台并发执行 `uv` 依赖安装，同时前台并发拉取主权重与用户 LoRA（HF 兜底仅在缺少 `huggingface_hub` 时等待 `uv` 完成），把 Python 依赖安装从关键路径隐藏。
- **纯 Python 多流 Range Drive 下载 + 有界滑窗单遍双哈希 + 本地 Stamp 缓存**（[runtime_worker.py](../../api/app/gpu/runtime_worker.py)）：
  - 对 ≥32 MiB 的 Drive 权重启用纯 Python 多流分片下载（默认 6 路、16 MiB 分片、进程级 12 流并发信号量、严格 `(streams + 2)` 有界内存滑窗 ≤ ~128 MiB 峰值内存防 T4 OOM），严格校验 `206` + `Content-Range` 与单分片指数退避重试，遇到 HTTP `200` 或非校验错误自动无损回退单流；
  - 多流与单流均在写盘同时单遍计算 **SHA-256 + MD5**，彻底消除下载后的二次全盘重读；
  - 持久化 VM 本地 `.model-hash-cache.json`（按 `(dev, ino, size, mtime_ns, ctime_ns)` 索引）+ C 级 `hashlib.file_digest`，同机复验权重降至毫秒级。

### 2.2 同机切工作流与换用户 LoRA（P1）
- **双层指纹与原地保留 `ComfyUI/`**（[runtime_worker.py](../../api/app/gpu/runtime_worker.py)）：
  - 拆分 `process_fingerprint`（进程热复用）与 `code_env_fingerprint`（仅含 `comfy_commit`、`python`、`custom_nodes` 及非权重依赖签名，落盘 `env-state.json`）；
  - 同 ComfyUI 版本换用户 LoRA 或切工作流时，安全停旧进程后**原地保留 `ComfyUI/` 与 Python 环境，仅重启 ComfyUI 进程**；
  - `models/loras/user/` 始终按本次任务白名单严格清理维持隔离；基础大模型权重利用 Colab 200+ GB 本地盘**优先留存**，仅在剩余空间不足（低于 `needed + 5 GiB`）时按 `.model-lru.json` 自动 LRU 淘汰最久未用旧模型。

### 2.3 控制面、后段产物回传与可观测性（P1）
- **缓存 Drive 子目录 ID 与并发远程预检**（[drive.py](../../api/app/gpu/drive.py)、[admission.py](../../api/app/admission.py)）：
  - 将 `owner`/`models`/`runtime-cache`/`loras`/`loras/h3` 目录 ID 持久化缓存至 `files` 表（`kind="drive_folder"`），消除每次 `loading` 的 3–5 次同步 Drive API 查询；前端准入对多个绑定模型并发执行 `check_remote()`。
- **精简 Bootstrap 载荷与单 WS 4 MiB 产物流式回传**（[mcp.py](../../api/app/gpu/mcp.py)）：
  - 创作任务下发 bootstrap 时不再附带 `node_persistence.py` / `native_runtime.py`，并透传 `CREATION_*` 开关至 Colab 运行时；
  - Colab 隧道任务拉取产物时复用单条 IPv4 优先 WebSocket 连接按 **4 MiB/块** 连续拉取（非隧道 MCP 保持原有 1 MiB 协议兼容），保留断点续传。
- **阶段埋点补齐与内部快轮询**（[scheduler.py](../../api/app/gpu/scheduler.py)、[job_history.py](../../api/app/job_history.py)、[job-stage-rows.ts](../../web/src/lib/job-stage-rows.ts)、[runtime_worker.py](../../api/app/gpu/runtime_worker.py)）：
  - 补全 `creation_python`、`restore_loras`、`creation_warming` 阶段映射与前端展示；
  - 收紧本机 `/system_stats`（1s → 0.3s）与 `/history`（2s → 0.5s）轮询间隔，并输出 Drive 权重下载吞吐（MiB/s、流数）日志。

---

## 3. 独立环境变量开关（支持一键回退）

所有优化项均可通过 `.env` 独立开关并由 `mcp.py` 自动透传至 Colab 运行时：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CREATION_USE_UV` | `1` | 合并基础依赖安装并优先使用 `uv` + `gpu-constraints.txt`（失败自动回退 `pip`） |
| `CREATION_PARALLEL_BOOTSTRAP` | `1` | `uv` 依赖安装、用户 LoRA 与模型权重并发准备 |
| `CREATION_MULTI_STREAM_DRIVE` | `1` | Drive 大权重（≥32 MiB）纯 Python 多流 Range 并发下载与单遍双哈希 |
| `CREATION_DRIVE_STREAMS` | `6` | 单文件并发 Range 流数（1–12，进程总并发上限 12） |
| `CREATION_LOCAL_HASH_CACHE` | `1` | 同 VM 本地 Stamp 哈希缓存（`.model-hash-cache.json`） |
| `CREATION_SKIP_DRIVE_PREP` | `1` | 全 Drive 绑定权重工作流跳过自动 CPU Prep |
| `CREATION_FAST_CONTROL_PLANE` | `1` | Drive 子目录缓存、并发预检、精简 bootstrap、单 WS 4 MiB 产物拉取与快轮询 |
| `CREATION_ENV_CACHE` | `auto` | 创作模式环境归档缓存策略（`auto`=仅自定义节点含额外依赖时启用，`1`=强制，`0`=关闭） |
