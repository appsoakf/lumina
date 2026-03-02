# Memory System Design

## 1. 实现目标

Lumina Memory 系统面向本地部署个人助手，核心目标：

1. 支持连续对话记忆（偏好、待办、近期历史、任务模板）。
2. 保持单用户架构下的实现简洁与可维护性。
3. 本地优先：SQLite 为主存，Qdrant 作为可选向量索引。
4. 稳定优先：TTL 过期、去重幂等、向量失败自动降级。

## 2. 当前实现概览

当前已实现的模块：

- `models.py`：`MemoryType`、`MemoryRecord`（含 `content_hash`）。
- `policy.py`：写入策略、TTL、去重窗口、清理周期。
- `ingestor.py`：从用户文本提取 profile/commitment 候选。
- `turn_summarizer.py`：每轮对话异步提取主题与偏好候选。
- `store.py`：SQLite 持久化、过期过滤/清理、去重检测、按 ID 回表。
- `retriever.py`：关键词检索（含类型权重与时间衰减排序）。
- `embedding.py`：Embedding Provider（OpenAI API）。
- `vector_store.py`：Qdrant 读写封装（自动降级）。
- `indexer.py`：异步向量入库队列与重试。
- `hybrid_retriever.py`：关键词 + 向量混合召回与重排。
- `service.py`：统一入口，供 orchestrator 调用。

单用户约束（已落地）：

- `default_user_id` 固定，逻辑上不做多用户隔离。
- 保留 `user_id` 参数仅做兼容，不参与检索分区。

## 3. 数据模型

主表：`memories`（SQLite）

- 关键字段：`id`, `memory_type`, `content`, `content_hash`, `ttl_seconds`, `payload`, `created_at`, `updated_at`
- 兼容字段：`user_id`, `session_id`

Qdrant 点位（可选）：

- `point_id = memory_id`
- `vector = embedding(content)`
- `payload`：`memory_type/tags/created_at/ttl_seconds/source/content_hash`

## 4. 数据流

### 4.1 写入链路

1. Orchestrator 调用 `MemoryService.ingest_turn(...)`。
2. `service` 同步写入 commitment/procedural（按策略）。
3. `turn_summarizer` 异步提取本轮 topic/profile 候选。
4. `service` 计算 `content_hash`，按窗口去重后写入 SQLite（异步失败会同步兜底）。
5. 若开启向量检索：提交异步索引队列，后台 embedding 后 upsert Qdrant。
6. 周期触发过期清理，清理 SQLite 后同步删除 Qdrant 点位。

### 4.2 检索链路

1. `build_context(query)` 触发检索。
2. `memory_vector.enabled=false`：走关键词检索。
3. `memory_vector.enabled=true`：走混合检索（关键词 + 向量），失败自动回退关键词。
4. 返回 TopK 后按“用户偏好/未完成事项/相关历史”拼接上下文。

### 4.3 交互方式

- 默认无显式记忆指令，采用“自由对话 + 自动提取”。
- 每轮对话后异步抽取 topic/profile，commitment/procedural 继续按策略写入。
- 记忆读取由 `build_context(query)` 在后续轮次自动注入，无需手工触发。

## 5. 结构图

```text
User Message
    |
    v
Orchestrator
    |
    +--> MemoryService
          |
          +--> MemoryPolicy + MemoryIngestor
          +--> AsyncTurnSummarizer(topic/profile)
          +--> LongTermMemoryStore (SQLite)
          |      |- add / search / list_recent
          |      |- dedupe / ttl filter / purge_expired_ids
          |
          +--> MemoryRetriever (keyword)
          +--> HybridMemoryRetriever (keyword + vector rerank)
                     |
                     +--> EmbeddingProvider
                     +--> QdrantVectorStore
          |
          +--> MemoryVectorIndexer (async upsert queue)
```

## 6. 配置

文件：`service/pet/config.json`

```json
"memory_vector": {
  "enabled": false,
  "provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "embedding_api_url": "https://api.siliconflow.cn/v1",
  "embedding_api_key": "",
  "qdrant_url": "http://127.0.0.1:6333",
  "qdrant_collection": "lumina_memory_vectors",
  "vector_dim": 1536,
  "top_k_vector": 12,
  "top_k_keyword": 12,
  "write_async": true,
  "queue_size": 512,
  "max_retries": 3
}
```

说明：

- `enabled=false` 时维持纯关键词检索。
- 启用后若 embedding/Qdrant 异常，会自动降级到关键词检索。

## 7. TODO（可改进项）

1. 为向量检索链路增加超时预算与熔断，避免拖慢主对话路径。
2. 提供全量回填脚本（SQLite 历史数据重建 Qdrant 索引）。
3. 优化混合召回策略：在召回阶段就做类型过滤，减少截断损失。
4. 增加向量最小相关性阈值（`score_threshold`）降低噪音命中。
5. 增加可观测性指标：`vector_hit_rate/fallback_count/embedding_latency/qdrant_latency`。
6. 将 `MemoryService.close()` 接入服务生命周期，避免退出时队列残留。
