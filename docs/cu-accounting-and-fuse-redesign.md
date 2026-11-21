# Colab GPU Studio CU 统计、预估与熔断改造文档

参考基线：`2b95b15`。动手实现前以当前分支实际代码为准。

---

## 1. 背景与改造目标

### 1.1 现状核心痛点

当前项目存在三种真实产生 CU 的途径：**创作模式单次任务（`save`）**、**创作模式长时间/智能策略复用（`smart` / `retain` / `batch` / 创作 API 预热复用）**、**原生 ComfyUI 长时间运行（`system/native-comfy`）**。现有实现围绕「单次任务结束即关机」演化，存在以下结构性问题：

1. **长运行与复用途径大面积漏记**：`gpu_usage` 仅在 `Scheduler._cleanup()` 确认删除租约时写入；智能策略保留、批次接力、原生 ComfyUI 引导完成、watchdog 孤儿回收及 Colab 远端主动回收均不写入 `gpu_usage`、也不采集 `ccu_after`。
2. **两套账本割裂、前后端口径错配**：v0.2 的 `gpu_instance_usage_segments` 从未被 `/api/gpu/usage` 和前端读取且 `actual_units` 恒为 `None`；前端「今日预估」「已用」实际展示全量历史累计；设置页拿历史累计除以每日上限算使用率；明细行的「估算」是事前预测值而汇总行是事后时长估算。
3. **熔断机制反直觉且割裂**：
   - 采用「最坏时长预扣（`max_runtime_minutes × rate`）」卡准入，原生 ComfyUI（默认 720 分钟）与 A100（默认 60 分钟）在默认限额下连开机都无法通过；
   - 复用同一台保留实例时叠加预扣 `budget_reserved` 与 `retention_reserved`，越复用越容易误触发熔断；
   - 只有事前拦截，没有运行中真正的熔断止损；
   - 配置分散在 `.env`（`GPU_UNIT_RATE`、`GPU_MAX_JOB_UNITS`、`GPU_DAILY_UNITS`、`GPU_MONTHLY_UNITS`、`GPU_MAX_RUNTIME_MINUTES`）与账号目录（`account.budget`）两处，必须手填全局单一费率且互相打架。

### 1.2 改造目标与产品原则

本轮改造以**精准、直观、极简**为核心原则：

1. **权威总账（按本地自然日）**：直接以 Google 账号实时余额流水为权威依据，精准展示**今日扣费**（按本地自然日 `00:00` 切日）与**历史扣费**（历史累计 + 按自然日趋势）。自用场景**不做站外消耗剥离**，账号真实余额下降即计入扣费（仅过滤余额上升的充值/续费跳变）。
2. **全途径明细归因**：单次创作任务、智能策略/批次复用（整台实例会话 + 每轮任务拆解 + 空闲待机）、原生 ComfyUI 长时间运行，全部实现开机建账、运行中实时刷新、关机延迟复采封账。
3. **极简两道真熔断**：仅保留 **今日总上限（温和熔断）** 与 **余额底线（强制熔断）** 两道保护。**废除**单次任务 CU 上限（`GPU_MAX_JOB_UNITS`）、最坏时长预扣、账号级强制预算卡口，以及全局运行时长上限截断（运行时长完全由任务或原生会话自身设置决定）。
4. **预估回归纯参考**：不再强迫配置 `GPU_UNIT_RATE`。预估费率自动取 Colab 实时费率或内置 GPU 规格费率，仅用于界面提示，不拦截开机。
5. **前端直接配置**：在前端提供「今日总上限」和「余额底线」的可视化设置与实时熔断状态展示，持久化至数据库，即时生效。

---

## 2. 权威总账与明细归因设计

### 2.1 时间边界：本地自然日

- 新增配置项 `STUDIO_TIMEZONE`（[config.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/config.py)），默认 `Asia/Shanghai`，使用 Python 标准库 `zoneinfo.ZoneInfo` 解析。
- 每次采样时间同时存储 UTC epoch（`sampled_at: float`）与对应的本地自然日键（`local_date: "YYYY-MM-DD"`）。
- **今日扣费** = 当前本地日期（`local_date == today_local_date`）内所有真实扣费流水之和；次日 `00:00:00` 自动归零并解除温和熔断。

### 2.2 第 1 层：账号余额流水表（`gpu_ccu_ledger`）

在数据库新增 `gpu_ccu_ledger` 表（Alembic 迁移 `0007_cu_ledger.py`），取代当前多 owner 会互相覆盖的单文件 `ccu_state.json`。

#### 采样触发时机
1. **周期巡检**：沿用 `worker.py` 每 60 秒轮询一次各启用 Colab 账号的 `/tun/m/ccu-info`（`source="tick"`）。
2. **关键生命周期节点**：实例创建前（`source="alloc_before"`）、实例确认删除/回收后（`source="delete_after"`）。
3. **手动查余额**：用户点击「查询剩余 CU」（`/api/gpu/compute/cu-balance`）成功返回时同步落库（`source="manual"`）。

#### 扣费增量计算规则（按 `account_id` 独立连续计算）
读取该账号最近一条有效余额记录 `prev`：
- **首次采样**（无 `prev`）：记录基线快照，`event_type="baseline"`，`delta_spent=0.0`。
- **余额下降**（`curr < prev.balance`）：`delta_spent = round(prev.balance - curr, 6)`，`event_type="spend"`。
- **余额不变**（`curr == prev.balance`）：仅更新最新快照心跳或按需去重落库，`delta_spent=0.0`（避免无变化时表无限膨胀：可仅在余额变化、实例集合变化、或距上次落库超过 10 分钟时追加新行，同时维护账号当前最新快照状态）。
- **余额上升**（`curr > prev.balance`）：识别为充值/续费/赠送，`event_type="topup"`，`delta_spent=0.0`，不产生负数扣费，并以新余额作为后续基准。
- **采样失败**（网络/超时/未返回有效数值）：不写伪造余额，仅记录账号最新采样错误供前端提示，不触发强制熔断。

#### 核心字段建议（存入 `CommonMixin.data` 及必要索引列）
- 索引列：`owner_id`、`account_id`、`local_date`（`String(10)`）、`state`（`event_type`）。
- JSON `data`：`sampled_at`、`balance`、`prev_balance`、`delta_spent`、`rate_hourly`、`assignments`、`active_instance_ids`、`reconciling_instance_ids`、`source`。

> **历史数据衔接**：保留原有 `gpu_usage` 表不删除。新统计界面中的「历史累计扣费」= `gpu_ccu_ledger` 汇总 + 改造上线前旧 `gpu_usage` 中已有 `units_actual` 之和（按迁移分界时间去重），确保升级后历史账目不断档、不重复。

### 2.3 第 2 层：实例会话账本（统一复用 `gpu_instance_usage_segments`）

无论是单次任务、智能策略复用还是原生 ComfyUI，底层都唯一对应一台 `GpuInstance`。改造后每台真实 Colab 实例对应一条完整的**实例会话账单记录**，覆盖全部生命周期路径：

#### 会话类型（`session_kind`）
- `single_job`：创作模式单次任务（`save`）
- `smart_reuse`：创作模式智能策略 / 保留 / 批次复用 / 创作 API 预热复用
- `native_comfy`：原生 ComfyUI 长时间运行
- `prep`：CPU 缓存准备
- `smoke`：系统或工作流真实冒烟

#### 全路径统一闭环（修复所有漏记入口）
1. **开机建账**：在 `Scheduler._allocation()` 成功拿到远端 `handle`（或创建实例进入 `preparing`）时，立即打开/创建该 `instance_id` 的会话记录（`state="open"`），固化：
   - `started_at`（开机时间）、`spec_key`、`gpu_label`
   - `unit_rate`（写入实例本身和会话记录，不再在 watchdog 回收时丢失）
   - `ccu_before`（开机前余额）、`actual_cu = 0.0`、`estimated_cu = 0.0`
2. **运行中实时入账**：每当 `gpu_ccu_ledger` 对某账号记录到 `delta_spent > 0`，立即按第 2.4 节归属规则累加到对应实例会话的 `actual_cu`，并同步按当前时长 × 有效费率更新 `estimated_cu`。原生 ComfyUI 与长运行智能策略在运行中即可每分钟看到最新消耗。
3. **全路径统一关机收口**：以下四条退出路径必须**全部调用同一个收口函数** `close_instance_session(instance_id, ended_at, reason)`：
   - 正常任务清理 `Scheduler._cleanup()`（含 `save` 关机与取消关机）
   - 后台空闲超时/手动停止 `Scheduler._cleanup_orphan_lease()`
   - Colab 远端主动回收 `InstanceService._close_reclaimed_runtime()`（同时修复 `Scheduler._cleanup()` 在 `RUNTIME_MISSING` 时提前返回跳过收口的问题）
   - 历史孤儿实例对账 `Scheduler._reconcile_deleted_lease_instances()`
4. **关机延迟复采封账（消除 Colab 延迟入账串账）**：
   - 实例确认远端删除后，会话先进入 `state="reconciling"`，设置 `reconcile_until = ended_at + 120`（2 分钟延迟窗口），同时立即触发一次 `delete_after` 余额采样。
   - 在 `reconcile_until` 之前的周期巡检中，若该账号没有新开机器而余额继续下降（Colab 延迟扣款入账），该笔 `delta_spent` 自动补归到这台处于 `reconciling` 的会话上。
   - 超过 `reconcile_until`（或同账号新开了下一台实例时立即提前结算），将状态置为 `state="closed"` 正式封账。

### 2.4 余额扣减归属与每轮任务拆解

#### 账号流水 → 实例会话归属
当某账号产生一笔 `delta_spent > 0`：
1. 找出该账号当前候选实例：所有 `state="open"` 的实例；若无 `open` 实例，则查找处于 `state="reconciling"`（关机 120 秒内）的实例。
2. **仅有 1 台候选实例**（自用最常见情况）：100% 归属该实例（`attribution = "exclusive"`）。
3. **同时有多台候选实例**（例如同时存在 GPU 实例 + CPU `prep` 实例）：按各实例有效费率 `unit_rate` 加权分摊（`attribution = "prorated"`）。
4. **无任何候选实例**：该笔 `delta_spent` 仍正常计入账号的 `today_actual_cu` 与 `history_actual_cu`，但不强行挂到无关实例上（无站外分类复杂度）。

#### 实例会话 → 每轮任务拆解
为解决复用任务缺少耗时起点的问题，在 `Scheduler._load()` 真正开始执行任务时向 `job` 写入 `execution_started_at`，任务离开 `running`/`uploading` 进入清理或交接（含 `handoff_lease` 与 `retained_idle` 提前返回）时写入 `execution_ended_at`，计算每轮任务的 `execution_seconds`。

会话内每轮任务的 CU 归因规则：
- **单次任务会话（仅关联 1 个任务且无空闲保留）**：该任务直接 1:1 继承会话的 `actual_cu`（`job.actual_units = session.actual_cu`），随关机延迟复采一并刷新。
- **智能策略 / 批次 / 创作 API 复用会话（关联 ≥1 个任务且存在空闲保留或多次执行）**：
  - 总运行时间 `T_total = max(ended_at - started_at, sum(execution_seconds))`；
  - 每个关联任务占用时长 `t_i = execution_seconds_i`；
  - 空闲/启动等待时长 `t_idle = max(0, T_total - sum(t_i))`；
  - 按时长比例分摊整台会话的实测 `actual_cu`：
    - `job_i.actual_units = round(session.actual_cu * (t_i / T_total), 6)`
    - `session.idle_cu = round(session.actual_cu - sum(job_i.actual_units), 6)`
  - 这样在前端既能看到整台机器的总实扣，展开后也能清楚看到每一轮生图/视频任务和空闲保留各花了多少 CU，且加总严格相等。
- **原生 ComfyUI 会话**：整场原生运行作为一条 `native_comfy` 会话记录实时展示和封账，同时将最终 `actual_cu` 同步回写到对应的引导任务 `job.actual_units`，消除对账列表里「未采到余额」的空白。

---

## 3. 预估机制简化（取消强迫配置，回归参考提示）

1. **移除 `GPU_UNIT_RATE` 必填门禁**：
   - `admission.py`、`native.py`、`compute_routes.py`、`review_routes.py`、`instances.py` 中不再因 `GPU_UNIT_RATE` 未设置而抛出 `RATE_UNKNOWN` / `BudgetError`。
2. **统一费率解析顺序**：`rates.minute_rate(spec_key, live_rate_hourly=None, fallback_rate=None)`
   1. 若有该实例关联账号的实时 `consumptionRateHourly`（且当前仅该实例在线），优先使用 `live_rate_hourly / 60`；
   2. 否则按实际 GPU 规格查内置费率表 `HOURLY`（`T4: 1.15/h`、`T4 HIGHMEM: 1.20/h`、`L4: 1.64/h`、`A100: 5.37/h`、`NONE/CPU: 0.07/h`）；
   3. 若遇到未知新规格，尝试使用环境变量 `GPU_UNIT_RATE` 兜底；若未配置则返回 `None`（界面显示「估算费率未知」），**绝不阻止开机**。
3. **预估仅做界面参考，不参与熔断**：
   - **创作模式**：沿用历史同工作流/模式的中位数时长（数据源切换为新的已完成任务 `execution_seconds`）× 规格费率，在提交按钮旁给出轻量参考估算。
   - **原生 ComfyUI**：不再计算并展示 `max_runtime_minutes × cfg.unit_rate` 这种误导性大数；改为展示所选 GPU 规格的参考小时费率（如 `L4 约 1.64 CU/小时`）与当前余额约可运行小时数。
   - **CUDA 冒烟 & 原生会话入库**：全部改用 `rates.minute_rate(spec)` 而非全局 `cfg.unit_rate`，确保参考估算贴合真实所选显卡。

---

## 4. 熔断机制重构（两道真保护 + 去除冗余限制）

### 4.1 废除的旧限制

为消除割裂与误拦截，本次改造明确移除以下机制：
- **移除**最坏时长预扣（`budget_units`、`budget_reserved`、`retention_reserved` 均不再作为准入或保留拦截条件）。
- **移除**单次任务 CU 上限（`GPU_MAX_JOB_UNITS` / `max_job_units`）。
- **移除**月度上限硬卡口（`GPU_MONTHLY_UNITS`）及账号目录必填安全预算（`account.budget` 缺失不再报 `GPU account budget is not configured`）。
- **移除**全局运行时长截断：不再使用 `GPU_MAX_RUNTIME_MINUTES` 截断任务提交的 `max_runtime_minutes`（如 [scheduler.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/scheduler.py#L497-L506) 的 `min(requested, self.config.max_runtime_minutes)`），运行时长上限完全由创作任务设置、工作流/批次参数或原生 ComfyUI 会话设置自行决定（仅校验为正整数）。

### 4.2 新的极简熔断配置（持久化单一事实源）

配置统一存放在 `settings` 表（`record_id = f"cu-guard:{owner_id}"`），由前端页面直接读写，服务重启不丢失：

| 字段 | 类型 | 含义 | 默认值 |
| --- | --- | --- | --- |
| `daily_limit_cu` | `float \| null` | **今日总上限（CU）**，跨账号当日实扣总和达到此值触发**温和熔断**；`null` 表示不限制 | 首次初始化时可继承 `GPU_DAILY_UNITS`（或默认 `null`） |
| `min_balance_cu` | `float \| null` | **余额底线（CU）**，账号实时余额降至此值及以下触发**强制熔断**；`null` 表示不限制 | 默认 `null` |

> **多账号下的余额底线语义**：
> - 任一启用账号的有效最新余额 `<= min_balance_cu` 时，对该账号触发**强制熔断**（停止该账号上的在线实例，且自动分配跳过该账号）。
> - 单账号场景下，账号级底线与全局底线完全等价；多账号场景下，既能保护低余额账号不被扣穿，也不会误杀另一个余额充足的账号。

### 4.3 熔断分级与精确行为矩阵

抽出统一领域服务 `CuGuardService`（建议新建 `api/app/gpu/cu_guard.py`），在**准入阶段**（创作单次/批次、Creation API、原生 ComfyUI、冒烟、`InstanceService.allocate`）与**运行中巡检阶段**（`worker.py` 每次完成 `tick_ccu` 后 + `Scheduler.watchdog()`）统一执行：

| 熔断级别 | 触发条件 | 新任务 / 新开机 | 运行中的创作实例（`busy`） | 空闲保留中的创作实例（`retained_idle`） | 运行中的原生 ComfyUI | 恢复方式 |
| --- | --- | --- | --- | --- | --- | --- |
| **正常** (`ok`) | 均未触发 | 正常放行 | 正常执行与保留 | 按策略保留 | 正常运行 | — |
| **温和熔断** (`soft_daily_limit`) | `daily_limit_cu != null` 且 `today_actual_cu >= daily_limit_cu` | **拒绝**，返回错误码 `CU_DAILY_LIMIT_REACHED`（含今日已耗与上限） | **不中断当前任务**；强制覆盖其收尾策略为 `save`（取消 `smart`/`retain` 保留与 `batch` 接力），当前任务完成后立即删机 | **立即结束保留**：标记实例为 `releasing`，由 watchdog 立即删机 | **排空模式**：设置 `native_draining = True`，通过 `begin_write` 拒绝新提示词与上传，复用 `protect_lease` 等待当前队列与 Drive 同步空闲后立即删机 | 本地自然日次日 `00:00` 自动恢复；或前端调高/清空 `daily_limit_cu` 后立即恢复 |
| **强制熔断** (`hard_balance_floor`) | `min_balance_cu != null` 且有效最新余额 `balance <= min_balance_cu`（优先级高于温和熔断） | **拒绝该账号**新任务/新开机，返回错误码 `CU_BALANCE_FLOOR_REACHED`（含当前余额与底线） | **立即强制中止**：向当前任务写入 `cancel_requested = True`、实例写入 `manual_stop_requested = True` 与 `fuse_forced_stop = True`，立即进入 `cleaning_up` 删除远端运行时 | **立即删机**：标记 `manual_stop_requested = True`、`state = "releasing"`，watchdog 立即删机 | **立即强制停机**：标记 `manual_stop_requested = True`、`native_draining = True`、`force_stop = True`，**跳过 `protect_lease` 的队列忙碌保护**，直接执行远端删机并确认 | 账号余额恢复至 `> min_balance_cu`；或前端调低/清空 `min_balance_cu` 后立即恢复 |

#### 关键边界保护
1. **防误杀（采样失败不触发强制熔断）**：仅当账号拥有明确数值 `balance is not None` 且 `balance <= min_balance_cu` 时才触发强制熔断；网络抖动导致的采样失败只告警，绝不误杀正在出图的实例。
2. **Mock 模式完全豁免**：`provider == "mock"` 的任务与实例不写扣费流水、不受今日上限与余额底线拦截。
3. **停止必须确认远端删除**：无论是温和熔断还是强制熔断，实例与租约都严格复用现有删机确认与重试机制（`cleanup_pending` 退避重试），不得在未确认远端删除前伪造 `deleted` 状态。
4. **日志与事件留痕**：触发温和/强制熔断停机时，向 `events` 写入明确事件（如「触发今日 CU 上限温和熔断，当前任务完成后释放实例」/「触发余额底线强制熔断，立即停止实例止损」），方便用户在实例详情和任务日志中回溯原因。

---

## 5. 后端 API 设计

### 5.1 新增与升级接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/gpu/cu/guard` | 获取当前「今日总上限」与「余额底线」配置及实时熔断状态 |
| `PUT` | `/api/gpu/cu/guard` | 管理员保存 `daily_limit_cu`、`min_balance_cu`（均允许 `null` 或 `>= 0` 数值）；保存后立即重算熔断状态并返回 |
| `GET` | `/api/gpu/usage` | 升级为权威汇总接口（保持路径兼容，补充新字段），返回今日实扣、历史累计、按日汇总、账号余额快照、熔断状态、在线实时消耗与最近明细 |
| `GET` | `/api/gpu/cu/sessions` | 分页/按自然日筛选实例会话扣费明细（支持 `date=YYYY-MM-DD`、`kind=all\|single_job\|smart_reuse\|native_comfy`、`limit`），每条会话可附带其下每轮任务拆解与 `idle_cu` |
| `GET` | `/api/gpu/compute/cu-balance` | 保持现有手动刷新余额接口，查询成功后同步写入 `gpu_ccu_ledger` 并刷新熔断判定 |

### 5.2 `GET /api/gpu/usage` 核心返回结构（示意）

```json
{
  "timezone": "Asia/Shanghai",
  "local_date": "2026-11-21",
  "today_actual_cu": 2.345,
  "history_actual_cu": 48.912,
  "active_runtimes": 1,
  "guard": {
    "daily_limit_cu": 10.0,
    "min_balance_cu": 5.0,
    "status": "ok",
    "soft_tripped": false,
    "hard_tripped_accounts": [],
    "message": null
  },
  "accounts": [
    {
      "account_id": "...",
      "alias": "主账号",
      "balance": 86.42,
      "rate_hourly": 1.64,
      "assignments": 1,
      "today_actual_cu": 2.345,
      "history_actual_cu": 48.912,
      "sampled_at": 1795267200.0,
      "below_floor": false,
      "error": null
    }
  ],
  "daily_history": [
    { "date": "2026-11-21", "actual_cu": 2.345, "sessions": 3 },
    { "date": "2026-11-20", "actual_cu": 6.12, "sessions": 5 }
  ],
  "active_sessions": [
    {
      "instance_id": "...",
      "session_kind": "native_comfy",
      "gpu_label": "L4",
      "started_at": 1795263600.0,
      "elapsed_minutes": 60.0,
      "actual_cu": 1.62,
      "estimated_cu": 1.64,
      "draining": false
    }
  ],
  "recent_sessions": [
    {
      "instance_id": "...",
      "session_kind": "smart_reuse",
      "state": "closed",
      "gpu_label": "L4",
      "local_date": "2026-11-21",
      "started_at": 1795250000.0,
      "ended_at": 1795251800.0,
      "duration_minutes": 30.0,
      "actual_cu": 0.81,
      "estimated_cu": 0.82,
      "idle_cu": 0.45,
      "jobs": [
        { "job_id": "...", "workflow_key": "krea2-moss", "state": "completed", "execution_seconds": 420, "actual_cu": 0.19 },
        { "job_id": "...", "workflow_key": "krea2-moss", "state": "completed", "execution_seconds": 370, "actual_cu": 0.17 }
      ]
    }
  ]
}
```

> 为平滑兼容现有调用方，旧字段如 `daily_limit`（映射到 `guard.daily_limit_cu`）、`actual_units`（映射到 `history_actual_cu`）可保留为只读兼容字段，但前端所有展示一律切到明确的新字段。

---

## 6. 前端界面改造方案

### 6.1 新增「CU 熔断保护」设置组件（`CuGuardSettings`）

在 **算力页（`ComputePage.tsx`）** 与 **设置页（`SettingsPage.tsx`）** 均提供入口（复用同一受控组件）：
- **今日总上限（CU）**：数字输入框，留空表示不限制。副标题明确标注：「按本地自然日 00:00 重置；触及后触发**温和熔断**——禁止新任务，运行中任务完成后自动释放，原生 ComfyUI 队列空闲后释放」。
- **余额底线（CU）**：数字输入框，留空表示不限制。副标题明确标注：「实时余额降至此值以下触发**强制熔断**——立即禁止新开机并强制停止在线实例止损」。
- **保存与状态反馈**：调用 `PUT /api/gpu/cu/guard`，保存成功后立即刷新 `usage` 与熔断横幅；未保存修改给出提醒，输入负数或非法值在前端和后端双重拦截。

### 6.2 算力页（`ComputePage.tsx`）改造

1. **CU 用量总览卡片**（修复「今日」与「历史」混淆）：
   - **今日扣费**：展示 `today_actual_cu` + 本地自然日日期（若设置了今日上限，附带 `已用 / 上限` 进度条）。
   - **历史累计扣费**：展示 `history_actual_cu`。
   - **当前剩余 CU**：展示各账号实时余额合计与最近采样时间，保留「立即查询余额」按钮（刷新同时更新流水与熔断状态）。
   - **熔断状态栏**：
     - 正常：绿色/中性标签；
     - 温和熔断中：橙色提示条，说明今日已达上限、可等次日 00:00 恢复或直接调高上限；
     - 强制熔断中：红色警示条，明确指出触及余额底线的账号与当前余额。
2. **消耗明细卡片**（替换旧的残缺「扣费对账」卡片）：
   - **Tab 1：运行会话与任务明细**：
     - 顶部展示当前**运行中会话**（原生 ComfyUI、智能策略、单次任务）的实时已运行分钟数与已扣 CU；
     - 下方列出近期已结束与对账中会话，支持按类型筛选（全部 / 单次 / 智能复用 / 原生 ComfyUI）；
     - 单次任务直接展示单行实扣；智能策略/批次复用支持展开查看每轮任务的耗时、分摊实扣 CU 及空闲待机 CU；原生 ComfyUI 显示整场会话的起止时间、GPU 规格与实扣 CU。
   - **Tab 2：按自然日汇总**：
     - 列出历史每日总扣费与运行会话数，直观核对历史每一天花了多少 CU。

### 6.3 其他页面联动清理

- **设置页（`SettingsPage.tsx`）**：
  - 「用量与额度」卡片改为展示 `今日实扣 / 今日上限` 与真实今日使用率（不再拿历史累计除每日上限）；
  - 直接嵌入 `CuGuardSettings`（今日总上限 & 余额底线）；
  - 账号目录（`AccountDirectory.tsx`）：移除原强制必填的「安全预算（站内估算 CU）」输入框，改为展示每个账号的**实时余额、今日实扣、底线状态**。
- **创作页（`CreatePage.tsx`）**：
  - 修复 `settings?.budget?.daily_units` 字段错配，改为读取 `usage?.guard?.daily_limit_cu` 与 `today_actual_cu`；
  - 处于温和/强制熔断时在提交区域显示明确提示与直达修改限额的链接。
- **原生 ComfyUI 页（`NativeComfyPage.tsx`）**：
  - 启动前展示所选 GPU 规格的参考小时费率与当前余额约可运行小时数；
  - 运行中顶部状态栏展示本次原生会话的**已运行时长 + 实时已扣 CU**，以及若触发温和熔断排空时的倒计时/排空状态提示。

---

## 7. 涉及代码入口清单

| 模块 | 文件路径 | 改造内容 |
| --- | --- | --- |
| 配置 | [api/app/config.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/config.py)、[api/app/gpu/compute.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/compute.py) | 新增 `STUDIO_TIMEZONE`；将 `GPU_UNIT_RATE`、`GPU_MAX_JOB_UNITS`、`GPU_DAILY_UNITS`、`GPU_MONTHLY_UNITS`、`GPU_MAX_RUNTIME_MINUTES` 从硬门禁降级为可选兜底/初始值 |
| 数据层 | [api/app/gpu/repository.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/repository.py)、`migrations/versions/0007_cu_ledger.py` | 新增 `GpuCcuLedgerRow` (`gpu_ccu_ledger`) 及迁移脚本；扩展 `gpu_instance_usage_segments` 读写辅助方法 |
| 余额流水与归因 | [api/app/gpu/ccu.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/ccu.py)、[api/app/worker.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/worker.py) | 持久化采样与增量扣费计算、充值过滤、自然日归档、向活跃/延迟对账会话入账；替换 `ccu_state.json`；巡检后执行运行中熔断检查 |
| 熔断服务 | 新建 `api/app/gpu/cu_guard.py` | 实现 `CuGuardService`：配置读写、今日/余额判定、准入校验、运行中温和/强制熔断执行 |
| 费率与预估 | [api/app/gpu/rates.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/rates.py) | 接入实时费率优先、移除无 `GPU_UNIT_RATE` 抛错、改用新执行时长样本计算参考预估 |
| 准入与调度 | [api/app/admission.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/admission.py)、[api/app/gpu/native.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/native.py)、[api/app/compute_routes.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/compute_routes.py)、[api/app/gpu/scheduler.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/scheduler.py)、[api/app/gpu/instances.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/gpu/instances.py) | 统一接入 `CuGuardService`；移除旧预算预扣与全局时长截断；统一四条关机路径的会话收口与延迟复采；补写任务 `execution_started_at` / `execution_ended_at` 与分摊逻辑；强制熔断跳过 `protect_lease` 忙碌保护 |
| HTTP 路由 | [api/app/main.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/main.py)、[api/app/compute_routes.py](file:///home/dev/workspace/projects/colab-gpu-studio/api/app/compute_routes.py) | 新增 `GET/PUT /api/gpu/cu/guard`、`GET /api/gpu/cu/sessions`；重构 `/api/gpu/usage` 汇总逻辑；`cu-balance` 手动刷新联动入账与熔断 |
| 前端 | [web/src/lib/gpu-api.ts](file:///home/dev/workspace/projects/colab-gpu-studio/web/src/lib/gpu-api.ts)、新建 `web/src/components/CuGuardSettings.tsx`、[web/src/pages/ComputePage.tsx](file:///home/dev/workspace/projects/colab-gpu-studio/web/src/pages/ComputePage.tsx)、[web/src/pages/SettingsPage.tsx](file:///home/dev/workspace/projects/colab-gpu-studio/web/src/pages/SettingsPage.tsx)、[web/src/components/AccountDirectory.tsx](file:///home/dev/workspace/projects/colab-gpu-studio/web/src/components/AccountDirectory.tsx)、[web/src/pages/CreatePage.tsx](file:///home/dev/workspace/projects/colab-gpu-studio/web/src/pages/CreatePage.tsx)、[web/src/pages/NativeComfyPage.tsx](file:///home/dev/workspace/projects/colab-gpu-studio/web/src/pages/NativeComfyPage.tsx) | 新增今日上限与余额底线设置组件；重构算力页用量总览与会话/按日明细；修复设置页与创作页口径；原生页显示实时 CU 消耗 |

---

## 8. 推荐实施顺序与验收清单

### 8.1 分阶段实施顺序

1. **阶段 1（底层账本与迁移）**：新增 `gpu_ccu_ledger` 与迁移 `0007_cu_ledger.py`，实现余额采样落库、自然日切日、充值跳变过滤、以及历史 `gpu_usage` 基线兼容。
2. **阶段 2（实例全路径闭环与任务分摊）**：改造 `gpu_instance_usage_segments` 写入与延迟复采封账，打通正常关机、watchdog 孤儿清理、Colab 远端回收三条退出路径，并实现共享实例下的每轮任务与空闲待机 CU 分摊。
3. **阶段 3（熔断与预估重构）**：落地 `CuGuardService`，替换旧的双重预算预扣、移除 `GPU_UNIT_RATE` / `GPU_MAX_JOB_UNITS` / `account.budget` / 全局时长截断硬卡口，接通运行中温和熔断与强制熔断。
4. **阶段 4（接口与前端）**：提供 `/api/gpu/cu/guard`、`/api/gpu/cu/sessions` 与新版 `/api/gpu/usage`，完成前端配置面板、明细视图与各页面口径修复。

### 8.2 验收用例清单

1. **自然日与余额流水准确性**：
   - 跨本地 `00:00` 采样时，昨日与今日扣费严格分桶，次日 `today_actual_cu` 归零且 `history_actual_cu` 连续累计；
   - 余额突然上升（模拟充值）时记为 `topup`、`delta_spent = 0`，之后继续下降正常计费；
   - 余额采样失败时报错但不写假数据、不误触发强制熔断。
2. **三种途径全闭环**：
   - 创作单次任务（含关机后 60 秒延迟扣费）：实扣准确归属该任务与今日总计；
   - 创作智能策略/批次复用多轮任务 + 空闲超时由 watchdog 关机：整台会话实扣完整记录，展开后各轮任务分摊 CU + `idle_cu` 等于会话总实扣；
   - 原生 ComfyUI 长时间运行（分别验证空闲超时关机、手动停止、Colab 远端 `RUNTIME_MISSING` 回收）：运行中实时可见已耗 CU，结束后完整进历史明细；
   - 未设置 `GPU_UNIT_RATE`、原生设 720 分钟、A100 设 120 分钟均可正常提交开机，不再被旧预算或全局时长截断拦截。
3. **温和熔断（今日总上限）**：
   - `today_actual_cu >= daily_limit_cu` 时新任务与原生启动被拒（`CU_DAILY_LIMIT_REACHED`）；
   - 正在跑图的创作实例跑完当前任务后自动忽略 `smart`/`retain` 直接释放；已空闲保留的实例立即释放；
   - 运行中的原生 ComfyUI 进入 `native_draining`（新写入被拒），等待当前队列空闲后释放；
   - 在前端调高或清空 `daily_limit_cu` 后，熔断状态立即解除并可正常提交。
4. **强制熔断（余额底线）**：
   - 账号余额 `<= min_balance_cu` 时新开机被拒（`CU_BALANCE_FLOOR_REACHED`）；
   - 正在跑图的创作实例立即取消并删除远端运行时；
   - 正在忙碌中的原生 ComfyUI 跳过忙碌保护直接执行远端删机确认；
   - 余额恢复或前端调低/清空 `min_balance_cu` 后恢复可用。
5. **前端持久化与一致性**：
   - 在算力页或设置页修改「今日总上限」与「余额底线」并刷新页面、重启服务后配置保持一致；
   - 算力页「今日扣费」、设置页使用率百分比、按日历史总和与明细加总口径完全一致。
