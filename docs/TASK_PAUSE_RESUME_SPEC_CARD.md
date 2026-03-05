# Task Pause/Resume Convergence Spec Card

[目标]
将现有“一次性跑完整任务图”改造为“可暂停-追问-恢复-再收敛”的任务闭环，确保在信息不足时不会继续执行下游步骤，而是向用户追问并在用户补充后继续同一 task 直到完成或显式失败。

[边界]
允许修改目录/文件：
- `core/orchestrator/*`
- `core/tasks/*`
- `core/protocols/*`
- `core/agentic/chat_agent.py`（仅新增等待态回复入口，可选）
- `service/pet/main.py`（仅扩展 meta，不改现有字段语义）
- `tests/*`
- `ARCHITECTURE.md`
- `AGENTS.md`（如需同步规范）

禁止修改目录/文件：
- `core/memory/*`
- `core/tools/*` 公共接口签名
- `runtime/*` 产物（除手工验证临时文件）

[约束]
1. 不引入新第三方依赖。
2. 不修改 `service -> orchestrator` 现有公共入口签名：`handle_user_message(...)` / `record_session_round(...)`。
3. 不修改 `ExecutorAgent.run_task(...)` 签名。
4. 不修改 `ExecutorRunResult` 字段结构（`output_text/tool_events/error/step_results`）。
5. websocket 对外结构保持兼容：仅新增可选 meta 字段，不删除或改名现有字段。
6. 继续遵守单用户模式（`default_user_id`），不引入 `user_id` 参数回传。
7. 运行路径必须复用 `core.paths`，不新增硬编码绝对路径。
8. `/cancel` 与 `/retry` 语义保持不变（仍优先 running，再 pending / 最新可重试）。

[Functionality]
Pre:
- 输入消息通过现有 WS 契约校验（JSON object + content:str）。
- 会话中可能存在一个 `WAITING_USER_INPUT` 任务。
- task graph 可能包含 `depends_on/input_bindings/并发策略`。

Post:
success:
- 新任务执行中若某 step 判定 `need_info`，任务进入 `WAITING_USER_INPUT`，立即停止下游调度。
- 返回用户“明确追问问题 + 需要补充的字段列表”，并保留同一 `task_id`。
- 用户下一轮回复后，系统恢复该 task，继续执行未完成步骤并最终收敛到 succeeded 或再次 waiting。
- 若信息充分，保持现有 planner->executor->critic 流程，结果与旧行为兼容。

fail:
- waiting 上下文损坏或恢复参数非法：任务进入 failed，`error.code=TASK_RESUME_INVALID`，`retryable=true`。
- 多轮追问仍无进展（超过 `max_clarify_rounds`）：任务 failed，`error.code=TASK_NOT_CONVERGED`，`retryable=true`。
- 任何阶段被 `/cancel`：任务 cancel 优先，后续 finalize 不得覆盖。

Invariants:
- 上游 step 为 `failed/cancelled/blocked/waiting_user_input` 时，下游不得执行。
- 同一时刻同一任务最多一个“等待输入点”（`pending_step_id` 唯一）。
- `waiting->running->(waiting|succeeded|failed|cancelled)` 状态迁移必须合法且原子。
- `task_id` 在暂停与恢复过程中不变化。
- service 层不直接访问 TaskManager 内部对象，只读 orchestrator 返回 meta。
- 旧客户端在不识别新 meta 字段时仍可正常工作。

Algorithm/Intent:
- 扩展 `TaskState`：新增 `WAITING_USER_INPUT`。
- 扩展 `TaskRecord` 持久化字段：
  - `task_snapshot: Dict`（用于恢复执行图）
  - `waiting_for_input: Dict | None`
  - `convergence: Dict`（`round_count/replan_count/last_progress_score`）
- 在 runner 中把 `need_info` 视为“可恢复中断”而非普通失败：
  - 记录 `pending_step_id/clarify_question/required_fields/raw_reason`
  - route 到 finalize_waiting（不走 failed finalize）
- orchestrator 收到新消息时的优先顺序：
  - `/cancel` or `/retry`
  - 若存在 waiting task -> `resume_task_flow`
  - 否则 -> `create new task`
- `resume_task_flow`：
  - 把用户补充输入绑定到 waiting step（建议约定 `$user_reply` 或 `waiting_for_input.collected_inputs`）
  - 将 task state: `WAITING_USER_INPUT -> RUNNING`
  - 从 `task_snapshot` 继续调度，不重建计划
- 引入 task-level convergence loop（有界）：
  - `PLAN -> EXECUTE -> EVALUATE -> (ASK_USER | REPLAN | FINISH)`
  - `max_replan_rounds` 与 `max_clarify_rounds` 来自 `config.json`（默认小值，如 2/3）
- meta 扩展（兼容）：
  - `task_waiting_input: bool`
  - `task_waiting_step_id: str`
  - `task_clarify_question: str`
  - `task_required_fields: List[str]`

[Modularity: Rely/Guarantee]
Rely:
- executor 输出仍可归一为“步骤状态/摘要/建议”文本。
- planner/critic 接口稳定。
- TaskStore 原子写（tmp+replace）可靠。

Guarantee:
- orchestrator 对外接口签名不变。
- service 不跨层读取 `task_manager` 内部状态。
- 任务暂停/恢复全部通过 orchestrator 编排完成。
- 新增状态与字段向后兼容旧任务文件（缺字段时默认 `None/空结构`）。

[Concurrency]
Pre-lock:
- 进入 cancel/retry/resume 前必须在 TaskManager 内持有同一把 `RLock`。
- resume 仅允许在 `state=WAITING_USER_INPUT` 时开始。

Post-lock:
- 状态变更与 `waiting_for_input` 清理在同一临界区完成。
- 持久化成功后才释放锁；异常路径也必须释放锁。

Atomicity/Ordering:
- `cancel vs resume` 并发：先成功 CAS 状态者生效；后到达请求返回 rejected。
- finalize 不得覆盖 cancelled（保持现有保护语义）。
- resume 成功后，必须先恢复 snapshot，再允许 select_ready_steps。
- 同一 waiting task 的重复用户消息只允许一次“成功恢复”，其余应回落为普通 chat 或提示正在处理中。

[Patch DAG]
Leaf:
- L1: `core/protocols/contracts.py` 增加 `TaskState.WAITING_USER_INPUT`。
- L2: `core/tasks/record.py` 增加 `task_snapshot/waiting_for_input/convergence` 字段与序列化兼容。
- L3: `core/tasks/manager.py` 增加 `set_waiting_input/get_waiting_task/resume_waiting_task` 原子操作。
- L4: `core/orchestrator/langgraph_task_runner.py` 增加 wait 分支与 resume 入口。
- L5: `core/orchestrator/orchestrator.py` 增加 waiting 优先路由与 resume 执行。
- L6: `service/pet/main.py` 扩展 route_meta 映射（新增可选字段）。
- L7: `tests` 新增 waiting/resume/cancel-race/replan-converge 回归。
- L8: `ARCHITECTURE.md` 更新状态机与时序。

Intermediate:
- I1: 合并 L1+L2+L3，先打通持久化与状态迁移。
- I2: 合并 L4+L5，打通暂停与恢复主链路。
- I3: 合并 L6+L7，打通对外可观测与回归保障。
- I4: 合并 L8，完成文档一致性。

Root:
- R1: 对外保持兼容前提下，实现“信息不足即追问、补充后继续、任务级收敛循环”。

[验收]
必须通过命令：
1. `conda activate agent`
2. `python -m unittest discover -s tests -v`
3. `python scripts/health_check.py --skip-network`

新增必测用例：
- 首轮 step need_info -> task 进入 `WAITING_USER_INPUT`，`S2+` 下游不执行。
- 用户补充后同 `task_id` 恢复执行并成功收敛。
- waiting 状态下 `/cancel` 生效且不被 finalize 覆盖。
- waiting 状态下 `/retry` 行为与 `failed/cancelled` 语义兼容。
- `cancel vs resume` 并发原子性（只能一个成功）。
- websocket 输出兼容：旧字段不变，新字段可选出现。

手工检查点：
- 复现“北京烤鸭推荐”场景：首轮应先追问偏好而非返回“暂时无法找到结果”。
- 回复“预算200/人，在东城区，想安静一点”后，应继续同一 task 并给出门店推荐。
- `runtime/tasks` 对应 task 文件应出现 `waiting_for_input` 与恢复后的 step 状态演进。
