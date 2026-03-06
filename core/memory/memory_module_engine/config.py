"""记忆系统配置"""
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """记忆系统配置"""

    storage_path: str = "./memory_data"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_dim: int = 4096

    # Working memory
    working_memory_size: int = 50
    overflow_process_ratio: float = 0.5
    overflow_cluster_similarity_threshold: float = 0.82
    # working 命中后置信度提升步长（渐进提升，越接近 1 增幅越小）
    working_confidence_boost: float = 0.02
    # working 命中后重要度提升步长（按命中周期触发）
    working_importance_boost: float = 0.02
    # 每命中 N 次，执行一次重要度提升
    working_importance_boost_every_hits: int = 3
    # 命中次数达到阈值后，允许从 working 晋升到 long-term
    working_persist_recall_threshold: int = 2
    # working 晋升 long-term 的重要度下限
    working_persist_importance_threshold: float = 0.50

    # Retrieval scoring
    default_top_k: int = 10
    relevance_weight: float = 0.5
    importance_weight: float = 0.3
    recency_weight: float = 0.2
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    # Access counting
    enable_async_mark_access: bool = True
    access_flush_interval_ms: int = 500
    access_flush_batch_size: int = 256

    # Repeat estimation
    near_repeat_similarity_threshold: float = 0.88

    # Persistence gate
    persist_importance_threshold: float = 0.70

    # Decay / forgetting / compression
    compression_threshold: float = 0.18
    eviction_threshold: float = 0.10
    compressed_retention_days: int = 45
    base_half_life_days: float = 30.0

    # Consolidation
    worker_interval_seconds: int = 1800
    consolidate_batch_size: int = 64
    consolidate_time_budget_ms: int = 250
    consolidate_tick_seconds: int = 3
    consolidate_cursor_reset_hours: int = 24
    low_importance_delete_threshold: float = 0.20
    high_importance_reinforce_threshold: float = 0.80
    consolidate_dedupe_similarity_threshold: float = 0.92
    dedupe_candidate_k: int = 24
    dedupe_max_pairs_per_step: int = 600
    dedupe_threshold: float = 0.92

    # Feature flags
    enable_incremental_consolidate: bool = True

    # LLM extraction/summarization (loaded from root config.yaml -> memory.llm.*)
    llm_enabled: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_timeout_seconds: int = 30
    llm_temperature_extract: float = 0.0
    llm_temperature_summary: float = 0.2
