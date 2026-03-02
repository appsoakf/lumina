# Lumina 架构文档（Phase 6）

## 1. 项目定位
Lumina 是一个本地部署的实时 AI 语音助手框架。当前 Phase 6 已实现：
- 多智能体编排（chat/planner/executor/critic）
- 任务生命周期管理（create/query/cancel/retry）
- 场景工作流（travel_workflow）
- 本地记忆系统（Memory OS Lite）
- 现有情感驱动 TTS 流式链路保持不变

入口：`main.py -> service.pet.main.run_pet()`。

## 2. 新增记忆系统（core/memory）
- `models.py`: `MemoryType`、`MemoryRecord`
- `store.py`: SQLite 持久化（`runtime/memory/memory.db`）
- `policy.py`: 记忆写入策略（profile/commitment/episodic）
- `ingestor.py`: 从用户语句抽取偏好与待办候选
- `retriever.py`: 记忆检索（profile/open commitments/relevant）
- `service.py`: 统一网关 `MemoryService`

记忆类型：
- `profile`（长期偏好）
- `commitment`（待办承诺）
- `episodic`（会话片段）
- `procedural`（任务流程经验）
- `artifact`（预留）

## 3. Orchestrator（Phase 6）
`core/orchestrator/lumina_orchestrator.py` 新能力：
1. 每轮请求前：注入记忆上下文到 agent history（个性化与连续性）
2. 每轮请求后：自动沉淀记忆（偏好、待办、会话摘要、流程经验）
3. 增加记忆指令：
   - `记住 xxx`
   - `我的偏好` / `查看记忆`
   - `我的待办` / `查看待办`
   - `完成待办 #<memory_id>`
4. 保留任务指令：
   - `查询任务 <task_id>`
   - `取消任务 <task_id>`
   - `重试任务 <task_id>`

## 4. 端到端流程（当前）
1. WS 收到用户消息
2. orchestrator 先处理任务指令 / 记忆指令
3. 非指令请求：注入记忆上下文后做 chat/task 路由
4. task 模式执行多 agent 链路 + task manager 状态跟踪
5. 结果写回 memory + session + trace
6. 输出走 emotion/translate/tts/audio streaming

## 5. 持久化目录
- `runtime/memory/` 记忆库
- `runtime/tasks/` 任务状态
- `runtime/sessions/` 会话快照
- `runtime/traces/` 事件追踪
- `runtime/notes/` 工具输出

## 6. 协议与状态
- `TaskState`: `pending/running/succeeded/failed/cancelled`
- `OrchestrationResult`: 兼容原接口，新增 `phase=phase6` 等 meta
- 错误码体系仍沿用 `core/error_codes.py`

## 7. 设计边界（简洁版）
- 当前记忆检索为本地 SQLite + 关键词匹配，不引入复杂向量基础设施
- 未引入权限白名单与执行防护（遵循 Phase 4-Lite 约束）
- 后续可增量扩展：向量检索、冲突消解、记忆衰减与归档
