# Lumina 架构文档

## 1. 项目定位
Lumina 是一个本地部署的实时 AI 语音助手框架。当前已实现：
- 多智能体编排（chat/planner/executor/critic）
- 工具基础设施分层（`core/tools`: BaseTool + ToolRegistry + built-in tools + web_search）
- 任务生命周期管理（create/cancel/retry）
- LangGraph 任务编排链路（planner -> langgraph runner -> executor/critic）
- 本地记忆系统（Memory OS Lite，含可选向量检索）
- LLM 调用基础层（`core/llm`: client 工厂 + chat 调用服务 + 翻译引擎）
- Agent 公共能力复用（`BaseLLMAgent` + `JSONParseMixin`）
- 现有情感驱动 TTS 流式链路保持不变

入口：`main.py -> service.pet.main.run_pet()`。

## 2. 记忆系统（core/memory）
- `models.py`: `MemoryType`、`MemoryRecord`（含 `content_hash`）
- `store.py`: SQLite 持久化（`runtime/memory/memory.db`）+ TTL 清理 + 去重检测
- `policy.py`: 记忆写入策略、TTL、去重窗口、清理周期
- `ingestor.py`: 从用户语句抽取偏好与待办候选
- `turn_summarizer.py`: 每轮对话异步提取主题与偏好候选
- `retriever.py`: 关键词检索（类型权重 + 时间衰减）
- `service.py`: 统一网关 `MemoryService`（长期语义记忆 + 短期会话历史）

向量检索组件（可选）：
- `embedding.py`: Embedding provider（OpenAI API）
- `vector_store.py`: Qdrant 封装（upsert/search/delete）
- `indexer.py`: 异步向量入库队列
- `hybrid_retriever.py`: 关键词 + 向量混合召回与重排

记忆类型：
- `profile`（长期偏好）
- `commitment`（待办承诺）
- `episodic`（会话片段）
- `procedural`（任务流程经验）
- `artifact`（预留）

## 3. Orchestrator
`core/orchestrator/orchestrator.py` 新能力：
1. 每轮请求前：注入记忆上下文到 agent history（个性化与连续性）
2. 每轮请求后：自动沉淀记忆（偏好、待办、会话主题摘要、流程经验）
3. 任务快捷指令：
   - `/cancel`
   - `/retry`
4. 对外统一公共接口：
   - `handle_user_message(user_text, session_id)`
   - `record_session_round(session_id, user_text, assistant_reply, metadata)`
   - `close()`
   （`service` 层不直接访问 orchestrator 内部依赖）

## 4. 端到端流程（当前）
1. WS 收到用户消息并做 JSON + 字段契约校验（坏消息返回错误并继续会话）
2. `service` 调用 `orchestrator.handle_user_message(...)`（不显式传递 history）
3. orchestrator 通过 `MemoryService.get_recent_history` + `build_context` 组装上下文
4. orchestrator 处理任务快捷指令；其余请求执行 chat/task 路由
5. task 模式下由 `LangGraphTaskRunner` 依据 `depends_on` / `input_bindings` / `graph_policy` 调度执行，`TaskManager` 持续跟踪状态
6. orchestrator 在回合内沉淀长期记忆；主题/偏好由异步总结器入库，向量检索启用时异步 upsert Qdrant
7. `service` 调用 `orchestrator.record_session_round(...)` 落盘短期会话历史
8. 输出走 emotion/translate/tts/audio streaming + trace

### 4.1 Executor 按需 ReAct（step 内）
- 默认策略：`react_mode=auto`
  - 先做一次 direct pass（不带 tools）。
  - 若输出可交付（`步骤状态: 成功/失败`）则直接返回，不进入工具循环。
  - 若输出为 `需补充信息`（或明显信息不足）则升级到 ReAct。
- ReAct 循环（有上限）：`DECIDE -> ACT(工具调用) -> OBSERVE(回填工具结果) -> DECIDE ...`
- 保护策略：
  - `max_tool_rounds`：限制最大决策轮次；
  - `max_repeated_tool_call`：检测重复工具调用，避免停滞循环。
- 对外兼容：仍返回 `ExecutorRunResult(output_text, tool_events, error)`，不暴露内部推理文本。

### 4.2 Tool 分层（core/tools）
- 统一抽象：`BaseTool` 提供 schema、调用入口、参数错误归一化、重试钩子。
- 注册中心：`ToolRegistry` 负责对外暴露 `list_schemas/call`，兼容 executor function-calling。
- 默认工具集：`get_current_time`、`write_note`、`list_notes`、`web_search`。
- `core/agentic/tools.py` 仅保留历史兼容导出，具体实现迁移至 `core/tools/*`。

## 5. 持久化目录
- 运行目录通过 `core/paths.py` 统一解析：
  - 默认：`<project_root>/runtime`
  - 可选覆盖：环境变量 `LUMINA_RUNTIME_DIR`
- 子目录：
  - `runtime/memory/` 记忆库
  - `runtime/tasks/` 任务状态
  - `runtime/sessions/` 会话快照
  - `runtime/traces/` 事件追踪
  - `runtime/notes/` 工具输出

## 6. 协议与状态
- `TaskState`: `pending/running/succeeded/failed/cancelled`
- `StepState`: `pending/ready/running/succeeded/failed/cancelled/blocked/skipped`
- `OrchestrationResult`: 兼容原接口，包含 `intent/final_reply/executor_result/meta`
- 错误码体系统一由 `core/utils/errors.py` 管理

## 7. 向量检索配置（可选）
配置位于 `service/pet/config.json` 的 `memory_vector` 节点，关键项：
- `enabled`: 是否启用混合检索（默认 `false`）
- `embedding_model` / `embedding_api_url` / `embedding_api_key`
- `qdrant_url` / `qdrant_collection` / `vector_dim`
- `top_k_vector` / `top_k_keyword`
- `write_async` / `queue_size` / `max_retries`

当 `enabled=false` 或向量链路异常时，系统自动回退到关键词检索。

## 8. 设计边界（简洁版）
- 当前为单用户架构：逻辑上固定 `default_user_id`，不做多用户隔离
- 向量检索是增强层：SQLite 仍是主数据源，Qdrant 仅做检索索引
- 未引入权限白名单与执行防护（遵循当前精简约束）
- 后续可增量扩展：超时预算/熔断、索引回填、召回阈值与可观测性

## 9. 数据流示意（示例）
示例用户输入：
`帮我规划一个北京3日游，预算3000元。另外我喜欢博物馆和清淡饮食。`

### 9.1 端到端示意图
```text
Client
  -> WS(/ws)
  -> service.pet.main.websocket_handler
  -> handle_bot_reply
  -> Orchestrator.handle_user_message
      -> MemoryService.get_recent_history (session short-term)
      -> MemoryService.build_context
          -> (keyword) MemoryRetriever
          -> (optional) HybridMemoryRetriever
              -> EmbeddingProvider.embed(query)
              -> QdrantVectorStore.search
              -> LongTermMemoryStore.get_by_ids
      -> ChatAgent.classify_intent => TASK
      -> PlannerAgent 生成计划
      -> LangGraphTaskRunner 执行步骤调度
          -> ExecutorAgent 执行步骤与工具调用
          -> CriticAgent 评审执行结果
      -> ChatAgent.reply_with_task_result 生成最终文本
      -> MemoryService.ingest_turn
          -> MemoryPolicy + MemoryIngestor
          -> LongTermMemoryStore.add (SQLite)
          -> MemoryVectorIndexer.enqueue (async)
              -> EmbeddingProvider.embed(content)
              -> QdrantVectorStore.upsert
  -> Orchestrator.record_session_round
      -> MemoryService.record_session_round
          -> ShortTermMemoryStore.save_round
  -> EmotionEngine + TranslateEngine + TTSEngine
  -> WS audio/text streaming response
```

### 9.2 模块级输入/处理/输出

| 模块 | 输入 | 处理 | 输出 |
|---|---|---|---|
| `websocket_handler` / `handle_bot_reply` | WS JSON：`{"content": "..."} ` | 校验请求契约（JSON object + `content:str`）；坏消息返回 `error` 并继续会话；调用 orchestrator 公共接口，不显式维护 history 列表 | `orchestrated.final_reply` + 流式音频输出 |
| `Orchestrator.handle_user_message` | `user_text`, `session_id` | 判断任务快捷指令或进入 chat/task 路由；从 memory 读取短期 history，注入长期上下文并执行多 agent 协作 | `OrchestrationResult(intent, final_reply, executor_result, meta)` |
| `MemoryService.build_context` | `query` | 拉取 profile/commitment/relevant；启用向量时做 hybrid 检索并回退兜底 | 可注入 history 的 memory context 文本 |
| `MemoryService.get_recent_history` / `record_session_round` | `session_id`, 回合消息 | 读取/写入短期会话历史（`ShortTermMemoryStore` 持久化） | recent history / session snapshot |
| `HybridMemoryRetriever.search`（可选） | `query`, `limit`, `memory_types` | 关键词召回 + 向量召回，按综合分重排（vector/keyword/recency/type） | 相关记忆列表（TopK） |
| `ChatCompletionService` / `BaseLLMAgent` / `JSONParseMixin` | agent 消息与模型原始输出 | `core/llm` 统一 client 初始化与 chat 调用（同步/流式），agent 侧复用调用与 JSON 清洗解析 | 降低各 agent 重复实现和解析偏差 |
| `core/tools` (`BaseTool/ToolRegistry`) | tool schema + tool args + tool context | 统一工具抽象、注册、调用和错误归一化 | 标准化 `ToolResult` + callable tool schemas |
| `PlannerAgent` | 任务请求 + 历史上下文 | 拆解执行步骤，输出 `steps + depends_on + input_bindings + graph_policy` | `PlanResult(goal, steps, graph_policy)` |
| `LangGraphTaskRunner` | `task_id`, `plan`, `history`, `graph_policy` | 在 LangGraph 状态图内完成 ready 选择、步骤执行、失败阻断、取消检测和评审收敛 | `task_snapshot + step_results + first_error` |
| `ExecutorAgent` | 单步任务输入 + 工具上下文 | 按需 ReAct：先直出，必要时进入工具循环并收敛 | `ExecutorRunResult(output_text, tool_events, error)` |
| `CriticAgent` | `user_text`, `plan`, `execution_graph` | 质量评审，给出 `pass/revise` 与建议 | `CriticResult` |
| `ChatAgent.reply_with_task_result` | 用户请求 + 执行总结 + history | 组织最终可读回复，补齐情绪 JSON 格式 | 最终回复文本（首行情绪 JSON） |
| `MemoryService.ingest_turn` | `user_text`, `assistant_reply`, `meta` | commitment/procedural 同步写入；turn_summarizer 异步提取 topic/profile 并入库，失败时同步兜底 | 新增 memory 记录 ID（可多条） |
| `MemoryVectorIndexer`（可选） | `memory_id`, `content`, payload | 异步 embedding + Qdrant upsert，失败重试 | 向量索引点位（不阻塞主链路） |
| `Emotion/Translate/TTS` | 最终回复文本 | 情绪解析 -> 逐句翻译（每次调用返回独立 `TranslateResult`，不使用共享错误状态）-> 逐句 TTS 流式合成 | `emotion_text` + `audio_chunk` + `audio_done` |

### 9.3 本示例下的关键结果
1. 请求被识别为 `TASK`，生成“北京 3 日游”计划并执行。
2. “喜欢博物馆/清淡饮食”被抽取为 `profile`，写入 SQLite。
3. 若启用向量检索，该条 profile 会异步写入 Qdrant，供后续语义检索使用。
4. 下次请求 `build_context` 会优先注入这些偏好，影响规划与回复风格。
