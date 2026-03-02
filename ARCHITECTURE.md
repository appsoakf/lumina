# Lumina 架构文档

## 1. 项目定位
Lumina 是一个本地部署的实时 AI 语音助手框架。当前已实现：
- 多智能体编排（chat/planner/executor/critic）
- 任务生命周期管理（create/query/cancel/retry）
- 通用任务链路（planner/executor/critic）
- 本地记忆系统（Memory OS Lite，含可选向量检索）
- 现有情感驱动 TTS 流式链路保持不变

入口：`main.py -> service.pet.main.run_pet()`。

## 2. 记忆系统（core/memory）
- `models.py`: `MemoryType`、`MemoryRecord`（含 `content_hash`）
- `store.py`: SQLite 持久化（`runtime/memory/memory.db`）+ TTL 清理 + 去重检测
- `policy.py`: 记忆写入策略、TTL、去重窗口、清理周期
- `ingestor.py`: 从用户语句抽取偏好与待办候选
- `retriever.py`: 关键词检索（类型权重 + 时间衰减）
- `service.py`: 统一网关 `MemoryService`

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
6. 若启用向量检索：memory 写入后异步 embedding + upsert Qdrant
7. 记忆过期清理时同步删除 Qdrant 点位
8. 输出走 emotion/translate/tts/audio streaming

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
- 未引入权限白名单与执行防护（遵循 Phase 4-Lite 约束）
- 后续可增量扩展：超时预算/熔断、索引回填、召回阈值与可观测性

## 9. 数据流示意（示例）
示例用户输入：
`帮我规划一个北京3日游，预算3000元。另外记住我喜欢博物馆和清淡饮食。`

### 9.1 端到端示意图
```text
Client
  -> WS(/ws)
  -> service.pet.main.websocket_handler
  -> handle_bot_reply
  -> LuminaOrchestrator.handle_user_message
      -> MemoryService.build_context
          -> (keyword) MemoryRetriever
          -> (optional) HybridMemoryRetriever
              -> EmbeddingProvider.embed(query)
              -> QdrantVectorStore.search
              -> MemoryStore.get_by_ids
      -> ChatAgent.classify_intent => TASK
      -> PlannerAgent 生成计划
      -> ExecutorAgent 执行步骤与工具调用
      -> CriticAgent 评审执行结果
      -> ChatAgent.reply_with_task_result 生成最终文本
      -> MemoryService.ingest_turn
          -> MemoryPolicy + MemoryIngestor
          -> MemoryStore.add (SQLite)
          -> MemoryVectorIndexer.enqueue (async)
              -> EmbeddingProvider.embed(content)
              -> QdrantVectorStore.upsert
  -> EmotionEngine + TranslateEngine + TTSEngine
  -> WS audio/text streaming response
```

### 9.2 模块级输入/处理/输出

| 模块 | 输入 | 处理 | 输出 |
|---|---|---|---|
| `websocket_handler` / `handle_bot_reply` | WS JSON：`{"content": "..."} ` | 解析文本、维护会话 `history/session_id`、调用 orchestrator | `orchestrated.final_reply` + 流式音频输出 |
| `LuminaOrchestrator.handle_user_message` | `user_text`, `history`, `session_id`, `user_id` | 指令判断（任务/记忆命令），记忆上下文注入，chat/task 路由，多 agent 协作 | `OrchestrationResult(intent, final_reply, executor_result, meta)` |
| `MemoryService.build_context` | `query` | 拉取 profile/commitment/relevant；启用向量时做 hybrid 检索并回退兜底 | 可注入 history 的 memory context 文本 |
| `HybridMemoryRetriever.search`（可选） | `query`, `limit`, `memory_types` | 关键词召回 + 向量召回，按综合分重排（vector/keyword/recency/type） | 相关记忆列表（TopK） |
| `PlannerAgent` | 任务请求 + 历史上下文 | 拆解执行步骤，生成 plan | `PlanResult(goal, steps)` |
| `ExecutorAgent` | 单步任务输入 + 工具上下文 | 多轮 function calling，执行工具并汇总 step result | `ExecutorRunResult(output_text, tool_events, error)` |
| `CriticAgent` | `user_text`, `plan`, `execution_graph` | 质量评审，给出 `pass/revise` 与建议 | `CriticResult` |
| `ChatAgent.reply_with_task_result` | 用户请求 + 执行总结 + history | 组织最终可读回复，补齐情绪 JSON 格式 | 最终回复文本（首行情绪 JSON） |
| `MemoryService.ingest_turn` | `user_text`, `assistant_reply`, `meta` | 抽取 profile/commitment/episodic/procedural，去重后写入 SQLite | 新增 memory 记录 ID（可多条） |
| `MemoryVectorIndexer`（可选） | `memory_id`, `content`, payload | 异步 embedding + Qdrant upsert，失败重试 | 向量索引点位（不阻塞主链路） |
| `Emotion/Translate/TTS` | 最终回复文本 | 情绪解析 -> 逐句翻译 -> 逐句 TTS 流式合成 | `emotion_text` + `audio_chunk` + `audio_done` |

### 9.3 本示例下的关键结果
1. 请求被识别为 `TASK`，生成“北京 3 日游”计划并执行。
2. “喜欢博物馆/清淡饮食”被抽取为 `profile`，写入 SQLite。
3. 若启用向量检索，该条 profile 会异步写入 Qdrant，供后续语义检索使用。
4. 下次请求 `build_context` 会优先注入这些偏好，影响规划与回复风格。
