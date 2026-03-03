# TaskState 完整规格实例（可直接套用）

本文档给出一个面向当前 Lumina 代码的 `Spec Card` 完整样例，目标是作为后续任务状态机相关需求的默认起草模板。

## 1. Feature

规范任务生命周期与快捷指令语义，保证 `create/cancel/retry`、`/cancel`、`/retry` 在单会话内行为稳定且可回归验证。

## 2. Scope（边界）

- 允许修改：
  - `core/protocols/contracts.py`
  - `core/tasks/manager.py`
  - `core/orchestrator/task_shortcuts.py`
  - `tests/test_task_flow_regression.py`
- 禁止修改：
  - `service/pet` 协议字段
  - `core/memory` 读写语义
  - 持久化文件格式（`runtime/tasks/*.json` 的字段兼容性）

## 3. Constraints（约束）

- 不引入新三方依赖。
- 保持 `TaskState` 枚举值兼容：`pending/running/succeeded/failed/cancelled`。
- 保持用户指令兼容：`/cancel` 和 `/retry` 无需显式 `task_id`。

## 4. Functionality

### 4.1 State Model

- `TaskState.PENDING`
- `TaskState.RUNNING`
- `TaskState.SUCCEEDED`
- `TaskState.FAILED`
- `TaskState.CANCELLED`

### 4.2 Pre / Post / Invariants

#### A) `create_task(session_id, user_text)`

- Pre:
  - `session_id` 为非空字符串。
- Post:
  - 返回新建 `TaskRecord`。
  - `state == pending`。
  - 任务被持久化到 `TaskStore`。
- Invariants:
  - `task_id` 全局唯一。

#### B) `set_state(task_id, state, error=None)`

- Pre:
  - `task_id` 对应任务存在。
  - `state` 属于 `TaskState` 枚举。
- Post:
  - 任务状态更新为目标状态。
  - `error` 按入参写入。
  - 更新时间 `updated_at` 前进。
- Invariants:
  - 不改变 `task_id/session_id/user_text/created_at`。

#### C) `cancel_task(task_id)`

- Pre:
  - `task_id` 对应任务存在。
- Post:
  - 若原状态为 `pending` 或 `running`：置为 `cancelled`，返回 `True`。
  - 若原状态为 `succeeded/failed/cancelled`：不变，返回 `False`。
- Invariants:
  - 取消不会清空 `step_results` 历史。

#### D) `retry_task(task_id)`

- Pre:
  - `task_id` 对应任务存在。
- Post:
  - 若原状态为 `failed/cancelled`：
    - 置为 `pending`
    - `error = None`
    - `step_results = []`
    - 返回任务对象
  - 若原状态不在可重试集合：返回原任务对象（状态不变）。
  - 若任务不存在：返回 `None`。
- Invariants:
  - 重试不改变 `task_id/session_id/user_text/created_at`。

#### E) `get_current_task(session_id)`（用于 `/cancel` 目标解析）

- Pre:
  - 给定会话下可能存在多个任务。
- Post:
  - 优先返回最近变更的 `running` 任务。
  - 若无 `running`，返回最近变更的 `pending` 任务。
  - 否则返回 `None`。

#### F) `get_latest_retryable_task(session_id)`（用于 `/retry` 目标解析）

- Pre:
  - 给定会话下可能存在多个任务。
- Post:
  - 返回最近变更的 `failed/cancelled` 任务。
  - 若不存在返回 `None`。

#### G) `execute_task_shortcut(...)`

- Pre:
  - 输入文本是 `"/cancel"` 或 `"/retry"`（大小写与首尾空白可容忍）。
- Post:
  - `/cancel`：取消当前目标任务并返回统一聊天回复。
  - `/retry`：重试最近可重试任务并返回统一聊天回复。
  - `meta.task_command` 与 `meta.task_id` 正确回填。
- Invariants:
  - 非快捷指令不应改变任务状态。

### 4.3 Algorithm / Intent

1. 快捷指令只做“目标解析 + 状态操作 + 文本回执”，不承担执行器逻辑。
2. 目标解析在会话维度执行，禁止跨会话操作任务。
3. 状态更新后立即持久化，确保 `/cancel` 和 `/retry` 回包与落盘一致。

## 5. Modularity（Rely / Guarantee）

### Rely

- `TaskStore.save/load/list_recent` 提供可重复读取的一致结果。
- `ChatAgent.reply_with_task_result(...)` 仅做回复组织，不改任务状态。
- `Orchestrator` 只通过 `TaskManager` 公共接口操作任务。

### Guarantee

- `TaskManager` 负责任务状态与持久化一致性。
- `task_shortcuts` 负责快捷指令解析与任务目标选择，不泄露内部存储细节。
- `service` 层不直接访问 `TaskStore` 或任务内部缓存。

## 6. Concurrency

### Pre-lock

- 同一 `TaskManager` 的同一会话任务操作应串行进入（当前默认由上层流程保证）。

### Post-lock

- 每次状态更新调用结束后，任务对象与持久化文件状态一致。

### Atomicity / Ordering

1. `/cancel`：目标解析完成后，必须先更新状态再生成回执文本。
2. `/retry`：状态回到 `pending` 与清空 `error/step_results` 必须在同一事务性更新中完成。
3. 列表读取与目标选择应基于同一轮“最近变更序”排序视图。

## 7. Patch DAG（演化样例）

以下给出一个“修复状态机回归”的 DAG 示例，便于后续照抄：

- Leaf-1：`TaskManager.cancel_task` 终态保护修复（禁止二次取消）。
- Leaf-2：`TaskManager.retry_task` 非可重试态保持原样。
- Intermediate-1：`get_current_task/get_latest_retryable_task` 目标选择与最近变更序统一。
- Root-1：`execute_task_shortcut` 集成 leaf/intermediate 语义，对外协议字段保持不变。

## 8. Validation（验收）

必须通过：

1. `python -m unittest tests.test_task_flow_regression -v`
2. `python scripts/health_check.py --skip-network`

建议补充（若本次改动涉及状态机逻辑）：

1. `test_task_manager_cancel_and_retry`
2. `test_task_manager_get_current_task_prefers_running`
3. `test_task_shortcut_cancel_current_task`
4. `test_task_shortcut_retry_latest_retryable_task`

## 9. Copy-Paste 模板（空白版）

```text
[Feature]
...

[Scope]
...

[Constraints]
...

[Functionality]
Pre:
- ...
Post:
- success: ...
- fail: ...
Invariants:
- ...
Algorithm/Intent:
- ...

[Modularity: Rely/Guarantee]
Rely:
- ...
Guarantee:
- ...

[Concurrency]
Pre-lock:
- ...
Post-lock:
- ...
Atomicity/Ordering:
- ...

[Patch DAG]
Leaf:
- ...
Intermediate:
- ...
Root:
- ...

[Validation]
- ...
```
