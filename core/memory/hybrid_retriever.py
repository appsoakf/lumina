from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.config import MemoryVectorConfig
from core.memory.embedding import EmbeddingProvider
from core.memory.models import MemoryType
from core.memory.retriever import MemoryRetriever
from core.memory.store import LongTermMemoryStore
from core.memory.vector_store import QdrantVectorStore


class HybridMemoryRetriever:
    """Hybrid retrieval: keyword recall + vector recall + weighted rerank."""

    TYPE_WEIGHT = {
        MemoryType.PROFILE.value: 1.35,
        MemoryType.COMMITMENT.value: 1.25,
        MemoryType.PROCEDURAL.value: 1.15,
        MemoryType.EPISODIC.value: 1.0,
        MemoryType.ARTIFACT.value: 1.05,
    }

    def __init__(
        self,
        keyword_retriever: MemoryRetriever,
        store: LongTermMemoryStore,
        embedder: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        cfg: MemoryVectorConfig,
    ):
        self.keyword_retriever = keyword_retriever
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store
        self.cfg = cfg

    def search(self, query: str, limit: int = 6, memory_types: Optional[List[str]] = None) -> List[Dict]:
        candidate_limit = max(limit * 2, self.cfg.top_k_keyword)
        keyword_rows = self.keyword_retriever.search_relevant(query=query, limit=candidate_limit)
        keyword_rows = self._filter_memory_types(keyword_rows, memory_types)

        if not self._vector_ready() or not (query or "").strip():
            return keyword_rows[:limit]

        vector = self.embedder.embed(query)
        if not vector:
            return keyword_rows[:limit]

        vector_hits = self.vector_store.search(
            query_vector=vector,
            limit=max(limit * 2, self.cfg.top_k_vector),
            memory_types=memory_types,
        )
        if not vector_hits:
            return keyword_rows[:limit]

        hit_ids = [int(h["memory_id"]) for h in vector_hits if h.get("memory_id") is not None]
        by_id = self.store.get_by_ids(hit_ids)

        merged: Dict[int, Dict] = {}
        for row in keyword_rows:
            memory_id = row.get("memory_id")
            if memory_id is None:
                continue
            merged[int(memory_id)] = row

        for memory_id, row in by_id.items():
            merged[int(memory_id)] = row

        keyword_rank = self._rank_to_score(keyword_rows)
        vector_score = {int(h["memory_id"]): self._normalize_vector_score(h.get("vector_score", 0.0)) for h in vector_hits}

        scored = []
        for memory_id, row in merged.items():
            score = self._hybrid_score(
                row=row,
                keyword_score=keyword_rank.get(memory_id, 0.0),
                vector_score=vector_score.get(memory_id, 0.0),
            )
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _vector_ready(self) -> bool:
        return bool(self.cfg.enabled and self.embedder.is_ready() and self.vector_store.is_ready())

    def _filter_memory_types(self, rows: List[Dict], memory_types: Optional[List[str]]) -> List[Dict]:
        if not memory_types:
            return rows
        allow = set(memory_types)
        return [r for r in rows if str(r.get("memory_type")) in allow]

    def _rank_to_score(self, rows: List[Dict]) -> Dict[int, float]:
        if not rows:
            return {}
        total = max(len(rows), 1)
        scores: Dict[int, float] = {}
        for idx, row in enumerate(rows):
            memory_id = row.get("memory_id")
            if memory_id is None:
                continue
            scores[int(memory_id)] = max((total - idx) / total, 0.0)
        return scores

    def _hybrid_score(self, row: Dict, keyword_score: float, vector_score: float) -> float:
        recency_norm = self._recency_score(row) / 2.5
        type_norm = self._type_score(row) / 1.35
        return (
            0.45 * vector_score
            + 0.25 * keyword_score
            + 0.20 * max(min(recency_norm, 1.0), 0.0)
            + 0.10 * max(min(type_norm, 1.0), 0.0)
        )

    def _type_score(self, row: Dict) -> float:
        return self.TYPE_WEIGHT.get(str(row.get("memory_type", "")), 1.0)

    def _normalize_vector_score(self, score: float) -> float:
        s = float(score)
        if s < 0:
            return max(min((s + 1.0) / 2.0, 1.0), 0.0)
        return max(min(s, 1.0), 0.0)

    def _recency_score(self, row: Dict) -> float:
        created_at = self._parse_dt(row.get("created_at"))
        if created_at is None:
            return 0.0
        age_hours = max((datetime.now(timezone.utc) - created_at).total_seconds() / 3600.0, 0.0)
        return 2.5 / (1.0 + age_hours / 72.0)

    def _parse_dt(self, value: str):
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
