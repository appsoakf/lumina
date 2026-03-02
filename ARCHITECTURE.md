# Lumina 架构文档（Phase 4-Lite）

## 1. 项目定位
Lumina 是一个本地部署的实时 AI 语音助手框架。当前 Phase 4-Lite 已实现：
- 常驻 `chat_agent`
- `planner_agent`、`executor_agent`、`critic_agent`
- `orchestrator` 统一调度
- `travel_workflow` 场景化执行
- `capability registry` 动态路由
- `task manager/store` 任务生命周期管理（创建/查询/取消/重试）
- 现有情感驱动 TTS 流式链路保持不变

入口：`main.py -> service.pet.main.run_pet()`。

## 2. 核心模块职责
- `core/capabilities/registry.py`
  - 维护 agent 能力声明与优先级
  - 运行时按 capability 解析目标 agent（不再硬编码调用对象）

- `core/tasks/models.py`
  - `TaskRecord` 任务状态模型
- `core/tasks/store.py`
  - 任务落盘到 `runtime/tasks`
- `core/tasks/manager.py`
  - 生命周期接口：创建、查询、更新状态、取消、重试、写步骤结果

- `core/orchestrator/lumina_orchestrator.py`
  - Phase 4-Lite 中心编排器
  - 支持任务指令：`查询任务 <task_id>`、`取消任务 <task_id>`、`重试任务 <task_id>`
  - task 执行链：`chat -> planner/workflow -> executor(step-loop) -> critic -> chat`

- `core/workflows/travel_workflow.py`
  - 旅游场景约束解析、缺失信息追问、模板化计划、质量复核

- `service/pet/main.py`
  - WebSocket 接入层
  - 调用 orchestrator + trace/session
  - 文本->分句->翻译->TTS->音频流推送

## 3. 端到端流程（当前）
1. 客户端发送 `{"content":"..."}` 到 `/ws`
2. orchestrator 先检查是否是任务管理指令（查询/取消/重试）
3. 若是普通对话：`chat_agent` 直接回复
4. 若是任务：创建 task_id 并进入执行链
5. 执行过程中的计划和步骤结果写入任务记录
6. 任务完成后状态写回 `SUCCEEDED/FAILED`
7. 最终文本继续走情感解析与 TTS 流式输出

## 4. 关键协议
- WebSocket 输入：`{"content":"用户文本"}`
- WebSocket 输出：
  - `emotion_text`
  - `audio_chunk`
  - `audio_done`
  - `done`
  - `error`（统一错误码结构）

## 5. 可靠性与观测
- 熔断：`translate`、`tts`
- Trace：异步队列写盘（`runtime/traces`）
- Session：会话落盘（`runtime/sessions`）
- Task：生命周期落盘（`runtime/tasks`）

## 6. 配置与错误体系
- 统一配置入口：`core/config.py`
- 统一错误码：`core/error_codes.py`
- 统一错误对象：`core/errors.py`

## 7. 精简约束
Phase 4-Lite 未引入：
- 工具权限白名单
- 执行防护策略（命令/路径限制）

设计目标是先保证扩展性与任务可管理性，后续再增量补安全策略。
