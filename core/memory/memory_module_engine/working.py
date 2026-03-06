"""FIFO working-memory queue."""
from __future__ import annotations

import threading
from collections import deque
from typing import List

import numpy as np

from .models import MemoryItem
from .utils import cosine_similarity


class WorkingMemory:
    """Thread-safe FIFO queue for short-term conversational memories."""

    def __init__(
        self,
        max_size: int = 50,
        confidence_boost: float = 0.02,
        importance_boost: float = 0.02,
        importance_boost_every_hits: int = 3,
    ):
        self.max_size = max_size
        # 命中反馈参数：用于在 working 阶段轻量更新记忆质量信号。
        self.confidence_boost = max(confidence_boost, 0.0)
        self.importance_boost = max(importance_boost, 0.0)
        self.importance_boost_every_hits = max(int(importance_boost_every_hits), 1)
        self._queue: deque[MemoryItem] = deque()
        self._index: dict[str, MemoryItem] = {}
        self._lock = threading.RLock()

    def add(self, item: MemoryItem) -> str:
        with self._lock:
            self._queue.append(item)
            self._index[item.id] = item
            return item.id

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            item = self._index.pop(memory_id, None)
            if item is None:
                return False
            self._queue = deque(m for m in self._queue if m.id != memory_id)
            return True

    def search(self, query_embedding: List[float], top_k: int) -> List[tuple[float, MemoryItem]]:
        with self._lock:
            # 先按语义相似度计算候选，保持 working 检索足够轻量。
            scored: list[tuple[float, MemoryItem]] = []
            for item in self._queue:
                if not item.embedding:
                    continue
                sim = cosine_similarity(query_embedding, item.embedding)
                scored.append((sim, item))
            scored.sort(key=lambda x: x[0], reverse=True)

            results: list[tuple[float, MemoryItem]] = []
            for sim, item in scored[:top_k]:
                # 命中次数作为“短期有效性”信号，后续可能触发晋升。
                item.recall_count += 1
                # 命中后轻量提升置信度，避免一次命中导致过度漂移。
                item.metadata.confidence = min(
                    1.0,
                    item.metadata.confidence + self.confidence_boost * (1.0 - item.metadata.confidence),
                )
                # 按固定命中周期小幅提升重要度，强调“持续被访问”的价值。
                if item.recall_count % self.importance_boost_every_hits == 0:
                    item.importance = min(1.0, item.importance + self.importance_boost)
                results.append((sim, item))
            return results

    def pop_oldest(self, count: int) -> List[MemoryItem]:
        count = max(int(count), 0)
        popped = []
        with self._lock:
            for _ in range(min(count, len(self._queue))):
                item = self._queue.popleft()
                self._index.pop(item.id, None)
                popped.append(item)
        return popped

    def get_all(self) -> List[MemoryItem]:
        with self._lock:
            return list(self._queue)

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return cosine_similarity(vec1, vec2)

    def similarity_scores(self, query_embedding: List[float]) -> List[float]:
        """
        批量计算 query 与当前 working 记忆的余弦相似度。

        与逐条 cosine_similarity 在语义上等价，仅减少 Python 循环与重复数组构建开销。
        """
        with self._lock:
            embeddings = [item.embedding for item in self._queue if item.embedding]

        if not embeddings:
            return []

        matrix = np.asarray(embeddings, dtype=float)
        query = np.asarray(query_embedding, dtype=float)
        q_norm = float(np.linalg.norm(query))
        if q_norm == 0.0:
            return [0.0] * int(matrix.shape[0])

        row_norms = np.linalg.norm(matrix, axis=1)
        denom = row_norms * q_norm
        denom = np.where(denom == 0.0, 1.0, denom)
        scores = np.dot(matrix, query) / denom
        return [float(v) for v in scores]

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self):
        with self._lock:
            self._queue.clear()
            self._index.clear()
