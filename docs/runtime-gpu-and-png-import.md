# 运行时 GPU 选择与 PNG 工作流导入

本次保留现有工作流、OAuth、数据库表、调度器及审核边界，没有新增模型管理系统或节点编辑器。

## GPU 使用及接口

创作页选择 Colab 后，在运行设置中选择「默认 / 推荐 GPU」或当前账号返回的完整 GPU 规格。支持列表来自现有 `GET /api/gpu/compute/specs?provider=colab&account_id=...`，不内置 T4/L4/A100/G4/H100 的可申请性。资格不代表当前有容量；自动账号模式最终选中账号后仍会重新校验资格。浏览器 localStorage 按工作流版本 ID 保存选择；确认提交弹窗也显示所选规格。

任务与批次的 `execution_policy` 增加可选字段：

```json
{
  "provider": "colab",
  "compute_spec": {
    "variant": "VARIANT_GPU",
    "accelerator": "A100",
    "shape": "SHAPE_STANDARD"
  }
}
```

不传或传 null 时沿用 `manifest.compute_spec`；传入时必须提供完整三元组。支持现有 CLI 的枚举别名，申请前归一化并与账号的 eligible 规格精确比较。申请失败沿用现有安全错误处理与清理流程，绝不改用其他 GPU。原工作流 manifest 不被修改。

工作流增加轻量可选配置，例如：

```json
{
  "compute_spec": {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"},
  "compute_requirements": {
    "min_vram_gb": 24,
    "min_ram_gb": 32,
    "accelerators": ["L4", "A100"]
  },
  "verified_compute_specs": [
    {"variant": "VARIANT_GPU", "accelerator": "L4", "shape": "SHAPE_STANDARD"}
  ]
}
```

`accelerators` 是明确的兼容型号/架构约束，可省略；**不要把所有推荐型号都写为约束**。GB 按十进制容量解释。提供方有容量信息时在申请前检查；没有时在 GPU 内通过实际显存和 RAM 检查，检查在环境和模型恢复前进行。此情况下发现容量不足仍可能产生分配期间的 CU。现有清单未提供要求时不推测要求。未填写 verified 列表的旧工作流将默认规格视为已有验证基线，其他规格给出未验证风险提示；不会更改参数、量化、精度或分辨率。

任务保留 `requested_spec`、`snapshot.runtime_compute_spec`、`compute_warnings`，公开 `runtime_spec_requested`、有证据时的 `runtime_spec_reported`、`gpu_reported` 和 `gpu_diagnostics`。请求、提供方规格与实测名称/显存分别写日志。CLI 未提供分配信息时不再用请求冒充实测。T4/L4/A100/H100 还会核对实际设备名称；G4 等逻辑规格通过提供方规格核对，并保留实际设备名称。

实例复用同时检查账号、空闲/占用状态、完整计算规格和环境兼容签名。单任务及批次使用相同签名算法，GPU 不匹配时申请新实例，不中断其他任务；旧实例继续执行原有空闲截止、硬截止和 watchdog 清理规则。模型缓存路径不变；原有依赖/包版本缓存键额外包含 GPU 名称、计算能力及驱动身份。不能读取设备身份时不共享环境缓存，可能重新构建环境；CPU 预缓存不能代替不同 GPU 架构的环境验证。

实际扣费来自 Google 账号余额流水。参考估算优先取独占实例的实时费率，其次使用 GPU 规格内置费率。未知规格可用 `GPU_UNIT_RATE`（CU/分钟）兜底，留空时界面会显示费率未知，仍可提交任务。选择器会将未获取的显存和费率信息标为未知。

## 图片导入使用及接口

进入「工作流与模型 → 从图片导入 ComfyUI 工作流」。支持文件选择、拖拽及移动端文件/相册选择。PNG 在浏览器本地解析，原图不上传、不持久化、不申请 GPU。结果展示缩略图、文件名、两类 JSON 检测结果、节点数、文件名线索和节点类型，可下载 workflow.json / prompt.json、复制、取消重选。

支持 tEXt、iTXt（压缩和未压缩）、zTXt 与 UTF-8。限制：32 MiB 图片、4 MiB 累计压缩文本及累计解压文本、10000 个数据块/节点、64 层 JSON 线索遍历、1 亿像素；校验 PNG 签名、块边界、CRC、文件头与结束块。解压按流限额，拒绝重复 workflow/prompt 元数据。普通 PNG、A1111/Forge parameters、格式混淆、损坏文件、无效 JSON、转码丢失元数据均有提示。只有一种有效格式时保留该格式，不进行可视化到 API 的猜测转换。

复用 `POST /api/gpu/workflows/import`，增加一种请求形式：

```json
{
  "workflow_key": "image-unique-key",
  "image_extraction": {
    "workflow": {"nodes": [], "links": []},
    "prompt": {"1": {"class_type": "Loader", "inputs": {}}}
  }
}
```

workflow/prompt 可各自缺省。服务端重新检查两种结构，不信任前端校验。保存为原有 `draft` 状态，增加 `image_extraction`（原始 JSON）、`image_import_summary`（文件和节点线索）、`pending_configuration`（缺失配置）。不生成下载来源、SHA-256、模型版本、节点仓库或 object_info；不自动安装、执行任何内容。仅有 workflow 时 `api_graph` 为空。

展开图片草稿可继续下载原始 JSON，并通过 JSON 配置区逐步补充配置：

`PUT /api/gpu/workflows/{id}/configuration`

请求包含 `model_id`、`manifest`、`api_graph`、`object_info`。允许保存未完成配置，更新缺失信息；原始提取结果不变。只允许管理员配置未审核的图片草稿，配置更改撤销审核/冒烟状态。未配齐配置不能 validate；之后继续原有静态校验、许可/依赖审查、固定节点白名单、真实冒烟和发布流程。图片成功提取不等于可运行。导入权限与现有管理员导入权限一致。

## 修改文件及用途

| 文件 | 用途 |
| --- | --- |
| api/app/job_schemas.py | 向后兼容的可选运行时 compute_spec |
| api/app/gpu/specs.py | 规格归一化、要求检查、GPU 兼容签名 |
| api/app/admission.py | 冻结运行规格与默认值、日志及规格对应估算 |
| api/app/gpu/scheduler.py | 精确规格申请、无回退、分配规格核对 |
| api/app/gpu/instances.py | 实例规格隔离、保留实例费率预算 |
| api/app/gpu/colab.py | 从实际 CLI assignment 提取规格，不伪造实测 |
| api/app/gpu/rates.py | 未知 GPU 的明确配置费率，取消 A100 隐式替代 |
| api/app/gpu/repository.py | 统一批次兼容签名、实测设备日志 |
| api/app/gpu/runtime_worker.py | 实测设备/容量检查、GPU 环境缓存隔离 |
| api/app/presenters.py | 公开安全的设备与运行规格信息 |
| api/app/gpu/image_import.py | 图片草稿结构校验、文件节点线索和缺失配置 |
| api/app/gpu/registry.py | 图片提取接入已有草稿，保留原始数据与校验门禁 |
| api/app/main.py | 图片草稿配置 API，复用现有导入 API |
| web/src/App.tsx | 每工作流 GPU 选择记忆、提交及确认弹窗 |
| web/src/pages/CreatePage.tsx | 创作页接入 GPU 选择器 |
| web/src/components/WorkflowGpuSelection.tsx | 动态账号规格列表与风险提示 |
| web/src/lib/v020-types.ts | 可选运行时规格类型 |
| web/src/lib/gpu-label.ts | 实测设备与显存展示 |
| web/src/lib/png-workflow.ts | 浏览器本地 PNG 元数据解析与安全限制 |
| web/src/components/ImageWorkflowImport.tsx | 上传、拖拽、提取预览、下载复制及导入 |
| web/src/components/ImageDraftConfiguration.tsx | 保留原始 JSON、缺失配置提示、逐步保存 |
| web/src/pages/WorkflowManager.tsx | 工作流管理入口及草稿配置区 |
| tests/test_runtime_gpu_selection.py | GPU 覆盖/复用/申请/错误/费率和草稿集成测试 |
| web/tests/png-workflow.test.mjs | PNG 三类文本、部分格式、异常和压缩炸弹测试 |
| scripts/runtime_gpu_png_browser_check.py | 390×844 移动端 PNG 导入与 GPU 选择记忆浏览器验收（GPU 数据为 fixture） |
| docs/runtime-gpu-and-png-import.md | 本开发报告与升级使用说明 |

## 验证与限制

自动化验证命令：

```bash
python -m pytest -q
cd web
npm run build
node --test tests/*.test.mjs
npm run test:contract
```

实际测试结果：后端全量 550 passed，另有 3 个 subtests passed；前端 57 passed；TypeScript/Vite 生产构建和两项契约脚本通过；修改的 Python 文件通过 Ruff 检查。后端只有测试依赖的 Starlette/httpx 弃用提示，无测试失败。

本次没有真实 Colab GPU 或真实 CU 支出测试。GPU 规格查询、L4/A100 请求体、失败、规格隔离等使用本地 mock / httpx MockTransport；不代表这些 GPU 当前可申请或内置工作流已在这些 GPU 上运行成功。

本地 Chromium 浏览器另验证了 390×844 视口：压缩 PNG 本地提取、两份下载按钮、实际 POST 导入成功并返回待配置草稿、无原图上传/无 GPU 任务创建、无页面 JS 错误或横向溢出。GPU 选择 UI 使用 L4/A100 假规格验证动态列表、确认弹窗和刷新后记忆，并未提交真实任务。可先启动 `python scripts/offline_server.py --port 18788`，另一个终端运行 `python scripts/runtime_gpu_png_browser_check.py` 复现（需要 Playwright Chromium）。

初版只支持 PNG；压缩元数据需要支持 DecompressionStream 的现代浏览器。模型和自定义节点只提供文件/类型线索，无法从图片可靠推断仓库、版本和完整依赖；自定义节点来源仍待现有审核流程确认。JSON 配置区用于补充运行配置，不提供节点编辑或自动模型下载。

## 升级

无需新增依赖、数据库迁移或 OAuth 重新授权；新增数据放在既有 JSON 字段中，旧工作流及旧请求继续可用。先备份现有数据库和数据卷；待当前任务完成后，按现有部署方式重建并同时更新 api、worker、watchdog，以避免新旧调度/缓存代码混用。例如使用仓库的 compose 示例：

```bash
docker compose -f docker-compose.example.yml up -d --build api worker watchdog
```

实际部署若使用自定义 compose 文件，请沿用原文件和原数据卷。源码部署执行现有安装步骤、`cd web && npm ci && npm run build`，再重启 API、worker 和 watchdog。保留原有环境变量；如需对 G4/H100 等未知规格显示参考估算，可配置 `GPU_UNIT_RATE`。首次在不同架构 GPU 上运行可能重建环境，但继续复用已有模型缓存。
