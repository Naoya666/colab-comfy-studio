# 原生 ComfyUI Custom Nodes 持久化

## 现有架构与改动

控制端仍在 VPS，Colab CLI 通过现有可信 bootstrap 发送程序；Colab 当前 Python 解释器运行固定 commit 的官方 ComfyUI，`manager_requirements.txt` 安装官方 Manager，使用 `--enable-manager`。Drive 仍由已有 rclone 挂载在 `/content/ai-studio/native/drive`，根目录是当前用户的 `AI-Studio/<owner>`，不是整个个人 Drive。

过去 `custom_nodes` 链接到 Drive，逐个执行 requirements，插件从 FUSE 导入。现在：

- `api/app/gpu/node_persistence.py`：独立标准库模块；源码清单、归档、校验、Manager CLI、依赖、稳定检测和后台同步。
- `native_runtime.py`：挂载后恢复到本地 `ComfyUI/custom_nodes`，恢复依赖，检查启动日志，再建立 Tunnel；安装本地同步扩展和重启前置脚本。
- `mcp.py`：随可信 bootstrap 发送模块及可选回滚 generation。
- `native.py`、`native_routes.py`：通过已有受保护 Tunnel 读取已验证清单，原子保存至现有 VPS `settings` 表，无新增数据库、服务或公网端口。
- `tests/test_node_persistence.py` 和已有原生测试：离线验收与启动回归。

模型仍使用既有 Drive 模型目录；`user/input/output` 保持既有持久化布局。旧 `custom_nodes` Drive 目录会迁移到本地，原目录保留；当归档清单存在时，以清单为准，防止旧目录让已删除插件复活。

## 存储格式与提交协议

Drive `native-comfy/plugins/` 下保存：

```
objects/<sha256>.tar.gz       每个插件的不可变源码对象
history/<generation>.json    完整插件集合及官方 Manager snapshot
pending.json                 已完整上传、等待重启验证的候选集合
head.json                    已通过加载验证的集合
```

插件条目含路径、HTTPS 仓库地址（如果可获得）、commit（Git 插件）、版本（pyproject）、源码 checksum、归档 SHA256、启用状态、文件/目录类型。清单另含备份时间和 generation。非 Git 插件以源码 checksum 标识版本；没有版本声明时 version 为 null。禁用支持 `*.disabled`、`.disabled/`；删除表现为完整集合中缺失的条目。Registry 原生元数据随源码保留，官方 snapshot 另外保存 Registry 版本。

插件从 Colab 本地 SSD 执行。后台每 10 秒检查内容，Manager 队列空闲且同一内容持续稳定至少 30 秒才开始归档；Git 锁存在、文件改变、Manager 忙都会延迟。归档之后再检查一次整个集合，避免提交半写入状态。上传使用直接 rclone copyto，不依赖 FUSE 延迟回写；每次写入后读回验证。顺序是对象 → 不可变历史 → pending → 本地提交标记。

运行中安装/更新的代码可能尚未加载，因此候选对象会及时保存，但不会立即覆盖 head。下一次 ComfyUI 启动完成节点导入后，检查本次启动日志的导入失败标记及依赖错误；只有启动时的源码集合与稳定集合一致，才将 pending 提升为 head。VPS 的 watchdog 和 `/api/gpu/native/plugins` 从现有认证通道拉取已验证 head，整份清单在一次现有 settings 事务中更新，并保留最近 20 次清单历史。

启动默认优先恢复 pending，以免已上传但未来得及重启的安装因 VM 回收而丢失；它必须重新通过依赖和导入验证。候选损坏或依赖失败时明确阻止成功启动，不自动跳过插件，不自动升级。

中断前没有更新 pending 时，旧 pending/head 继续有效；孤立对象可留待以后清理。不能承诺保住在 30 秒稳定窗口内尚未上传的变更。硬销毁、Drive 不可达时也无法保证最后的本地写入。正常主动停止时，现有释放逻辑会等待 Manager、Drive 写入及源码候选同步完成；最终读取清单后再释放。候选已持久化但还需重启验证时，可以正常停止，并在下个实例验证。VPS 掉线不影响 Colab 继续保存候选和 head。

## 依赖与 Manager 官方能力

优先复用[官方 cm-cli 文档](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/docs/en/cm-cli.md)中的 `save-snapshot`、`restore-dependencies`，兼容 PyPI Manager 的 `cm_cli/__main__.py` 和传统 `cm-cli.py`，使用同一个 `sys.executable` 和明确的 `COMFYUI_PATH`、`--user-directory`。不开发插件安装/更新接口；用户继续通过 Manager 操作。

源码恢复选择归档而非每次 `restore-snapshot`：官方快照擅长恢复仓库/Registry 版本，但不能完整保存工作区配置和本地修改，且可能再次访问 Git/Registry。归档保留完整集合和禁用状态，既不调用 update all，也不 checkout 最新分支。官方 snapshot 作为辅助元数据保存在历史中；不恢复其中的 pip 环境或 ComfyUI commit。

| 方式 | 冷启动网络与 Drive 成本 | 适用范围 |
| --- | --- | --- |
| 完整 Git clone | 源码、历史、Git 服务请求；每插件重复 | 手动排障或没有缓存的初次安装 |
| Manager restore-snapshot | 解析仓库/Registry，可能重新下载源码与依赖 | 官方版本清单迁移/人工修复 |
| 本实现归档恢复 | 每个插件一个压缩对象；本地缓存命中无需再下载 | 日常恢复、精确回滚、带本地配置的插件 |

Git 对象只打包当前 commit 的 tree/blob，恢复为浅仓库，以便 Manager 后续更新；不备份历史、Git config、认证头、hook、reflog。无需冷启动再克隆 Git 仓库。无变化插件复用原归档；源码变化只重新上传该插件。校验需要读取源码内容；非常大型插件可增加扫描成本。首次上传增加一次读回校验流量，后续恢复无需上传。

依赖成功标记仅留在当前 VM，指纹包括插件集合、Python、平台、ComfyUI 和受保护 GPU 包版本；同一 VM 多次重启且未变化时跳过重复恢复。新 VM 必须在当前环境安装依赖，不能复制旧 site-packages 或虚拟环境。未持久化 wheel 缓存，因此新 VM 的依赖仍可能需要联网下载。

Manager 会处理 requirements.txt 和 install.py；没有 requirements 的插件会记录 install.py 是否存在、检查 README 是否含安装说明，并在日志指向安装说明。不会自动执行 README 的任意 shell 命令。额外的系统依赖或无法自动推断的手动安装步骤需要用户按插件说明处理；失败不会伪装成成功。

`restore-dependencies` 某些版本会吞掉单个安装脚本的 False 返回值，适配器将其转为非零退出。另检查错误输出、`pip check` 和实际加载。`plugin-errors.json` 记录涉及插件、错误和经敏感字段过滤的详情，本地 `plugin-dependencies.log` 保存完整安装日志（0600，不备份）。VPS 启动任务会收到明确异常。

启动及 Manager 子进程继承 `PIP_CONSTRAINT`/`UV_CONSTRAINT`，锁定当前 torch、torchvision、torchaudio、triton、nvidia-* 包；依赖不兼容时失败，不能静默替换当前 GPU 栈。安装后再次比较版本。版本约束不是恶意代码沙箱；插件自己绕过 pip、显式忽略约束或修改系统仍具有该进程的权限。

## 状态与回滚

使用已有 Studio 登录或 Bearer Token 调用：

```
GET /api/gpu/native/plugins
```

返回 manifest、history、pending_generation、restart_required、durable、error、install_notes。GPU 不在线时可读取 VPS 已保存清单。generation 是 64 位 SHA256，可以在停止当前原生会话后传给现有启动接口：

```json
{
  "account_id": "<已有账号>",
  "spec": {"key": "<已有 GPU 规格>"},
  "plugin_generation": "<history 中的 64 位 generation>"
}
```

调用 `POST /api/gpu/native/start` 仍需原有 Idempotency-Key。回滚恢复整组插件，再按当前 GPU/Python 环境恢复依赖并验证加载。历史不存在或校验失败时直接报错。

本地缓存损坏：重新下载同一个 SHA256 对象。远端缓存损坏：拒绝解包和覆盖本地插件，输出具体插件；选择一个对象完好的历史 generation 重新启动。若所有历史对象都损坏，应从可信来源按明确 commit/Registry 版本人工修复，不会静默改装最新版。

## 安全边界

第三方节点和 install.py 可执行任意代码，应只安装信任的插件。节点与 ComfyUI 同权限，无法保证抵御恶意插件读取运行时凭据；这个方案不声称提供隔离。

复用现有 Studio 会话/Bearer 鉴权、写操作 Origin 检查、内部 Cloudflare Access Service Auth；ComfyUI 与 rclone RC 仍只监听 loopback。部署必须继续保护整个内部 Tunnel 域名，不能绕过 Studio 直连 Manager。

归档验证 SHA256、源码 checksum、路径和成员类型，拒绝路径穿越、链接和特殊文件。源文件排除 `.env*`、私钥、Token/credential 目录、环境、缓存、模型和输出目录，并扫描常见 Token/私钥格式；Git 当前提交内容也扫描。发现可能凭据时停止备份并指明文件，不发送该对象。自定义配置中的未知格式秘密无法可靠自动识别，不能把任意配置文件当安全资源；敏感配置应保存在源码之外，不应手工放入插件目录。插件源文件单文件上限 64 MiB，模型应放到已有模型库。

## 验证与真实 GPU 验收

离线测试使用真实归档和本地 Git、模拟 Drive/Manager 传输；覆盖首次安装并通过验证、全新 VM 恢复、Manager 新增候选恢复、精确回滚、禁用与删除、缓存损坏检测及历史恢复、依赖失败、无变化归档/依赖复用、同步中断、目录未稳定、恶意归档、敏感配置、Git 历史隔离、VPS 原子清单，以及原有 Manager 鉴权与启动回归。

```
pytest tests/test_node_persistence.py tests/test_native_comfy.py -q
```

离线测试不等同于真实 GPU 认证。上线前在已配置的 Colab/Drive 上执行以下八项：

1. Manager 安装可信测试插件，重启并确认 object_info 和节点加载正常，head/VPS 清单出现插件。
2. 停止并销毁实例，用同账号/Drive 新建，确认从本地 custom_nodes 加载插件。
3. 会话中新增插件，等待 durable=true，确认 restart_required=true；重启后确认清单提升。
4. 更新插件并验证，记录两个 generation；用旧 generation 启动，检查 commit/版本一致。
5. Manager 删除插件，等待同步；新建实例确认插件未复活。
6. 破坏测试归档，确认启动阻止且日志报告具体插件；指定完好历史 generation 恢复。
7. 测试插件使用不兼容依赖，确认启动失败并报告插件，torch/CUDA 版本不变。
8. 同 VM 重启三次，确认依赖标记命中、无变化源码对象未重传；新 VM 必须重建依赖，不能复用旧环境。

2026-11-17 自动验证：完整 pytest 为 **697 passed、3 subtests passed**（57.17 秒，1 个既有警告）；持久化及原生回归为 **89 passed**。改动 Python 文件的 Ruff 与 `git diff --check` 通过。未分配真实 GPU，未修改外部 Drive、DNS 或 Cloudflare。
