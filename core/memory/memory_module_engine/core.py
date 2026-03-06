"""Memory system core: FIFO working memory + long-term consolidation."""
from __future__ import annotations

import threading
import time
import uuid
import heapq
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .config import MemoryConfig
from .embedding import EmbeddingProvider
from .long_term import LongTermMemory
from .models import MemoryItem
from .overflow_processor import OverflowProcessor
from .signal_extractor import SignalExtractor
from .utils import (
    DecayEngine,
    ImportanceScorer,
    MemoryCompressor,
    WriteGate,
    clamp,
    normalize_text,
    recency_score,
)
from .working import WorkingMemory


class Memory:
    """Main memory API."""

    def __init__(
        self,
        storage_path: str = "./memory_data",
        embedding_provider: Optional[EmbeddingProvider] = None,
        auto_consolidate: bool = True,
        config_overrides: Optional[dict] = None,
    ):
        if embedding_provider is None:
            raise ValueError(
                "embedding_provider is required. "
                "Example: Memory(embedding_provider=OpenAIEmbedding(api_key='sk-xxx'))"
            )

        self.config = MemoryConfig(storage_path=storage_path)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.embedder = embedding_provider
        self.scorer = ImportanceScorer()
        self.signal_extractor = SignalExtractor(
            llm_enabled=self.config.llm_enabled,
            llm_model=self.config.llm_model,
            llm_api_key=self.config.llm_api_key,
            llm_base_url=self.config.llm_base_url,
            llm_timeout_seconds=self.config.llm_timeout_seconds,
            llm_temperature=self.config.llm_temperature_extract,
        )
        self.write_gate = WriteGate(threshold=self.config.persist_importance_threshold)
        self.working = WorkingMemory(
            max_size=self.config.working_memory_size,
            confidence_boost=self.config.working_confidence_boost,
            importance_boost=self.config.working_importance_boost,
            importance_boost_every_hits=self.config.working_importance_boost_every_hits,
        )
        self.long_term = LongTermMemory(storage_path, vector_dim=self.embedder.get_dimension())
        self.decay_engine = DecayEngine(
            compression_threshold=self.config.compression_threshold,
            eviction_threshold=self.config.eviction_threshold,
            compressed_retention_days=self.config.compressed_retention_days,
            base_half_life_days=self.config.base_half_life_days,
        )
        self.compressor = MemoryCompressor()
        self.overflow_processor = OverflowProcessor(
            similarity_threshold=self.config.overflow_cluster_similarity_threshold,
            llm_enabled=self.config.llm_enabled,
            llm_model=self.config.llm_model,
            llm_api_key=self.config.llm_api_key,
            llm_base_url=self.config.llm_base_url,
            llm_timeout_seconds=self.config.llm_timeout_seconds,
            llm_temperature=self.config.llm_temperature_summary,
        )

        self.auto_consolidate = auto_consolidate
        # 写路径单通道：持久化 / consolidate / mark_access 串行执行。
        self._long_term_write_lock = threading.RLock()
        self._worker_stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memory-core")
        self._pending_access_counts: dict[str, int] = {}
        self._pending_access_lock = threading.RLock()
        self._access_flush_stop_event = threading.Event()
        self._access_flush_wakeup_event = threading.Event()
        self._access_flush_thread: Optional[threading.Thread] = None

        if self.config.enable_async_mark_access:
            self._start_access_flush_worker()
        if self.auto_consolidate:
            # 初始化后自动启动后台 consolidate 线程。
            self._start_worker()

    def add(self, content: str) -> str:
        """Add raw user content into memory system."""
        raw_content = normalize_text(content)
        if not raw_content:
            raise ValueError("content cannot be empty")

        now = time.time()
        extract_future = self._executor.submit(self.signal_extractor.extract, raw_content)
        embedding_future = self._executor.submit(self.embedder.encode, raw_content)

        extracted = extract_future.result()
        embedding = embedding_future.result()
        metadata = extracted.metadata

        near_repeat_score, repeat_count = self._estimate_repeat_signals(embedding)
        metadata.near_repeat_score = near_repeat_score
        metadata.repeat_count = repeat_count
        metadata.created_at = now
        metadata.store = "working"
        metadata.state = "active"

        importance = self.scorer.calculate(raw_content, metadata)
        metadata.confidence = clamp(metadata.confidence + 0.05 * near_repeat_score)

        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=raw_content,
            importance=importance,
            embedding=embedding,
            recall_count=0,
            metadata=metadata,
        )
        self.working.add(item)

        should_immediate_persist = self.write_gate.evaluate(
            metadata=metadata,
            importance=importance,
        )

        if should_immediate_persist:
            self._persist_item(item)
            self.working.remove(item.id)

        self._process_overflow_if_needed()
        return item.id

    def search(self, query: str, top_k: int = 10) -> list[MemoryItem]:
        """
        统一检索入口（working + long-term 混排）。

        关键点：
        1. 读路径不拿长期写锁，避免被 consolidate 写事务阻塞；
        2. 命中 long-term 后默认异步批量更新 recall_count（可回退同步）；
        3. 最终结果仍在 core 里统一打分，保证排序策略单一且可控。
        """
        query_text = normalize_text(query)
        if not query_text:
            return []

        query_embedding = self.embedder.encode(query_text)
        working_candidates = self.working.search(query_embedding, top_k=max(top_k * 2, top_k))

        # 读连接检索候选，不阻塞写通道。
        long_term_candidates = self.long_term.search_candidates(
            query_text=query_text,
            query_embedding=query_embedding,
            limit=max(top_k * 4, 20),
            min_importance=0.0,
        )

        # 缓存查询近似的候选记忆
        ranked_by_id: dict[str, tuple[float, MemoryItem]] = {}

        # 对全部working memory计算最终的查询分数
        for sim, item in working_candidates:
            score = self._compose_search_score(item=item, relevance=sim)
            current = ranked_by_id.get(item.id)
            if current is None or score > current[0]:
                ranked_by_id[item.id] = (score, item)

        # 对全部long term memory计算最终的查询分数
        for candidate in long_term_candidates:
            item = candidate["item"]
            relevance = (
                self.config.vector_weight * candidate.get("vector_score", 0.0)
                + self.config.keyword_weight * candidate.get("keyword_score", 0.0)
            )
            score = self._compose_search_score(item=item, relevance=relevance)
            current = ranked_by_id.get(item.id)
            if current is None or score > current[0]:
                ranked_by_id[item.id] = (score, item)

        final = [
            item for _, item in sorted(ranked_by_id.values(), key=lambda x: x[0], reverse=True)[:top_k]
        ]

        long_term_hit_ids = [item.id for item in final if item.metadata.store == "long_term"]
        if long_term_hit_ids:
            if self.config.enable_async_mark_access:
                self._enqueue_access_counts(long_term_hit_ids)
            else:
                with self._long_term_write_lock:
                    self.long_term.mark_access(long_term_hit_ids)

        # 对本轮命中的 working 记忆做“命中驱动晋升”判断：
        # 满足 recall_count + importance 双阈值后，直接转为长期记忆。
        self._promote_working_hits(final)
        return final

    def consolidate_step(self) -> dict:
        """
        手动触发一次“增量 consolidate 小步”。

        与后台 tick 逻辑一致，便于手动推进和观测处理进度。
        """
        return self._consolidate_long_term_step()

    def consolidate(self) -> dict:
        """手动触发一次长期记忆 consolidate（与后台周期任务同一条链路）。"""
        if self.config.enable_incremental_consolidate:
            return self._consolidate_long_term_step()
        return self._consolidate_long_term_full()

    def get_stats(self) -> dict:
        long_term_memories = self.long_term.get_all(include_archived=True)
        working_items = self.working.get_all()
        return {
            "working_count": len(working_items),
            "long_term_count": len(long_term_memories),
            "avg_importance": (
                sum(item.importance for item in long_term_memories) / len(long_term_memories)
                if long_term_memories
                else 0.0
            ),
        }

    def _compose_search_score(self, item: MemoryItem, relevance: float) -> float:
        """
        计算查询分数，综合考虑语义相关度、重要性和时间近期度
        """
        recency = recency_score(
            timestamp=item.metadata.created_at,
            half_life_days=max(item.metadata.half_life_days, 1.0),
        )
        score = (
            self.config.relevance_weight * clamp(relevance)
            + self.config.importance_weight * clamp(item.importance)
            + self.config.recency_weight * recency
        )
        return round(clamp(score), 6)

    def _estimate_repeat_signals(self, embedding: list[float]) -> tuple[float, int]:
        similarities = self.working.similarity_scores(embedding)
        similarities.extend(self.long_term.find_similar_scores(embedding, limit=5))

        if not similarities:
            return 0.0, 0

        top = heapq.nlargest(min(3, len(similarities)), similarities)
        mean_top = sum(top) / len(top)
        near_repeat = clamp((mean_top - 0.78) / 0.22)
        repeat_count = sum(1 for sim in similarities if sim >= self.config.near_repeat_similarity_threshold)
        return round(near_repeat, 4), int(repeat_count)

    def _persist_item(self, item: MemoryItem):
        item.metadata.store = "long_term"
        # 每当存储进长期记忆时，计算半衰期
        item.metadata.half_life_days = self.decay_engine.compute_half_life(item)
        with self._long_term_write_lock:
            self.long_term.add(item)

    def _promote_working_hits(self, results: list[MemoryItem]):
        """
        将反复命中的 working 记忆晋升到 long-term。

        设计意图：
        1. 让“被用户反复问到/命中”的短期记忆自动沉淀；
        2. 避免仅靠初始写入时的一次打分决定长期保留；
        3. 用双阈值（命中次数 + 重要度）限制误晋升。
        """
        for item in results:
            if item.metadata.store != "working":
                continue
            # 命中次数不足，不晋升。
            if item.recall_count < self.config.working_persist_recall_threshold:
                continue
            # 重要度不足，不晋升。
            if item.importance < self.config.working_persist_importance_threshold:
                continue
            # 满足条件则持久化，并从 working 队列移除，避免双份不一致。
            self._persist_item(item)
            self.working.remove(item.id)

    def _process_overflow_if_needed(self):
        max_size = max(int(self.config.working_memory_size), 1)
        if len(self.working) <= max_size:
            return

        batch_size = max(1, int(max_size * self.config.overflow_process_ratio))
        oldest_batch = self.working.pop_oldest(batch_size)
        if not oldest_batch:
            return

        # 对旧记忆作聚类，并持久化
        clusters = self.overflow_processor.cluster(oldest_batch)
        summary_items = self.overflow_processor.build_summaries(clusters)
        for summary in summary_items:
            summary.embedding = self.embedder.encode(summary.content)
            self._persist_item(summary)

    def _start_access_flush_worker(self):
        """启动访问计数异步落库 worker。"""
        if self._access_flush_thread and self._access_flush_thread.is_alive():
            return
        self._access_flush_thread = threading.Thread(
            target=self._access_flush_worker_loop,
            daemon=True,
            name="memory-access-flush-worker",
        )
        self._access_flush_thread.start()

    def _access_flush_worker_loop(self):
        """
        周期性将命中计数批量写入 long-term，减少 search 请求内同步写事务。
        """
        interval_s = max(int(self.config.access_flush_interval_ms), 20) / 1000.0
        batch_size = max(int(self.config.access_flush_batch_size), 1)

        while not self._access_flush_stop_event.is_set():
            self._access_flush_wakeup_event.wait(timeout=interval_s)
            self._access_flush_wakeup_event.clear()
            try:
                self._flush_pending_access_counts(max_items=batch_size)
            except Exception:
                continue

        # 退出前尽量清空剩余计数，避免命中统计丢失。
        while self._flush_pending_access_counts(max_items=batch_size) > 0:
            continue

    def _enqueue_access_counts(self, memory_ids: list[str]):
        """
        将本次命中的 long-term id 聚合到内存缓冲区。
        """
        local_counts: dict[str, int] = {}
        for memory_id in memory_ids:
            if not memory_id:
                continue
            local_counts[memory_id] = local_counts.get(memory_id, 0) + 1
        if not local_counts:
            return

        pending_size = 0
        with self._pending_access_lock:
            for memory_id, inc in local_counts.items():
                self._pending_access_counts[memory_id] = (
                    self._pending_access_counts.get(memory_id, 0) + inc
                )
            pending_size = len(self._pending_access_counts)

        if pending_size >= max(int(self.config.access_flush_batch_size), 1):
            self._access_flush_wakeup_event.set()

    def _flush_pending_access_counts(self, max_items: Optional[int] = None) -> int:
        """
        将缓冲区计数批量写入 long-term。

        返回本次实际刷新的 memory id 数量。
        """
        with self._pending_access_lock:
            if not self._pending_access_counts:
                return 0

            if max_items is None or len(self._pending_access_counts) <= max_items:
                counts = dict(self._pending_access_counts)
                self._pending_access_counts.clear()
            else:
                counts = {}
                selected_ids = list(self._pending_access_counts.keys())[: max(int(max_items), 1)]
                for memory_id in selected_ids:
                    counts[memory_id] = self._pending_access_counts.pop(memory_id)

        if not counts:
            return 0

        try:
            with self._long_term_write_lock:
                self.long_term.mark_access_counts(counts, commit=True)
            return len(counts)
        except Exception:
            # 写入失败时计数回灌，避免丢失。
            with self._pending_access_lock:
                for memory_id, inc in counts.items():
                    self._pending_access_counts[memory_id] = (
                        self._pending_access_counts.get(memory_id, 0) + int(inc)
                    )
            return 0

    def _start_worker(self):
        """
        启动后台 consolidate worker。

        链路入口：
        1. __init__ 时若 auto_consolidate=True 自动调用；
        2. worker 线程每隔 worker_interval_seconds 调一次 _consolidate_long_term。
        """
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._consolidation_worker_loop,
            daemon=True,
            name="memory-consolidation-worker",
        )
        self._worker_thread.start()

    def _consolidation_worker_loop(self):
        """
        后台调度主循环。

        调度策略：
        1. 开启增量模式时走“高频小步”（consolidate_tick_seconds）；
        2. 关闭增量模式时回退到“低频全量”策略（worker_interval_seconds）；
        3. 单轮失败不会终止线程，防止后台线程意外退出造成长期退化。
        """
        if self.config.enable_incremental_consolidate:
            interval = max(int(self.config.consolidate_tick_seconds), 1)
        else:
            interval = max(int(self.config.worker_interval_seconds), 1)

        while not self._worker_stop_event.wait(interval):
            try:
                if self.config.enable_incremental_consolidate:
                    self._consolidate_long_term_step()
                else:
                    self._consolidate_long_term_full()
            except Exception:
                # Worker failures should not kill user requests.
                continue

    def _consolidate_long_term_step(self) -> dict:
        """
        执行一次预算驱动的增量 consolidate。

        步骤：
        1. 在写锁内调用 long_term.consolidate_step，按 batch/time budget 小步推进；
        2. 统一返回本轮去重/衰减/强化统计。
        """
        with self._long_term_write_lock:
            step = self.long_term.consolidate_step(
                decay_engine=self.decay_engine,
                compressor=self.compressor,
                batch_size=max(int(self.config.consolidate_batch_size), 1),
                time_budget_ms=max(int(self.config.consolidate_time_budget_ms), 1),
                cursor_reset_hours=max(int(self.config.consolidate_cursor_reset_hours), 1),
                low_importance_delete_threshold=float(self.config.low_importance_delete_threshold),
                high_importance_reinforce_threshold=float(
                    self.config.high_importance_reinforce_threshold
                ),
                dedupe_threshold=float(self.config.dedupe_threshold),
                dedupe_candidate_k=max(int(self.config.dedupe_candidate_k), 1),
                dedupe_max_pairs=max(int(self.config.dedupe_max_pairs_per_step), 1),
            )
        return step

    def _consolidate_long_term_full(self) -> dict:
        """
        长期记忆 consolidate 主流程（后台与手动触发共用）：

        1) 去重：合并近似重复记忆，累计 recall_count，保留高质量条目；
        2) 衰减：根据半衰期计算 decayed value，决定压缩/淘汰；
        3) 策略强化/清理：
           - 低重要且少访问的记忆删除；
           - 高重要记忆延长半衰期（强化保留）。

        返回值为本轮统计信息，便于观测和调试。
        """
        with self._long_term_write_lock:
            # 近似重复去重。
            dedupe = self.long_term.dedupe_by_similarity(
                threshold=self.config.consolidate_dedupe_similarity_threshold,
            )
            # 衰减判定，执行压缩/淘汰。
            decay = self.long_term.apply_decay(self.decay_engine, self.compressor)
            # 读取最新长期记忆，执行策略强化/清理。
            items = self.long_term.get_all(include_archived=False)

            reinforced = 0
            removed = 0
            for item in items:
                if (
                    item.importance < self.config.low_importance_delete_threshold
                    and item.recall_count < 2
                    and not item.metadata.explicit_remember
                ):
                    self.long_term.delete(item.id)
                    removed += 1
                    continue

                if item.importance >= self.config.high_importance_reinforce_threshold:
                    # 高价值记忆强化：延长半衰期，降低后续被衰减淘汰概率。
                    item.metadata.half_life_days *= 1.2
                    self.long_term.update_item(item, update_vector=False)
                    reinforced += 1

        stats = {
            "dedupe_merged": dedupe.get("merged", 0),
            "compressed": decay.get("compressed", 0),
            "evicted": decay.get("evicted", 0),
            "reinforced": reinforced,
            "removed_low_importance": removed,
        }
        return stats

    def close(self):
        self._worker_stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

        if self.config.enable_async_mark_access:
            self._access_flush_stop_event.set()
            self._access_flush_wakeup_event.set()
            if self._access_flush_thread and self._access_flush_thread.is_alive():
                self._access_flush_thread.join(timeout=3.0)
            self._flush_pending_access_counts(max_items=None)

        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._long_term_write_lock:
            self.long_term.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
