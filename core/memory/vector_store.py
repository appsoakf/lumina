import logging
from typing import Any, Dict, List, Optional

from core.config import MemoryVectorConfig
from core.utils import log_event, log_exception

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - graceful fallback when dependency is missing
    QdrantClient = None
    qmodels = None


class QdrantVectorStore:
    """Qdrant-backed vector index with graceful fallback."""

    def __init__(self, cfg: MemoryVectorConfig):
        self.cfg = cfg
        self.client: Optional[Any] = None
        self._ready = bool(cfg.enabled)

        if not self._ready:
            return

        if QdrantClient is None or qmodels is None:
            log_event(
                logger,
                logging.WARNING,
                "memory.qdrant.unavailable",
                "qdrant-client 未安装，回退到关键词检索",
                component="memory",
                fallback="keyword_only",
            )
            self._ready = False
            return

        try:
            self.client = QdrantClient(url=cfg.qdrant_url, timeout=5.0)
            self._ensure_collection()
        except Exception:
            log_exception(
                logger,
                "memory.qdrant.init.error",
                "Qdrant 初始化失败，回退到关键词检索",
                component="memory",
                fallback="keyword_only",
                qdrant_url=cfg.qdrant_url,
            )
            self.client = None
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready and self.client is not None and qmodels is not None

    def _ensure_collection(self) -> None:
        if not self.is_ready():
            return

        try:
            self.client.get_collection(self.cfg.qdrant_collection)
            return
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.cfg.qdrant_collection,
            vectors_config=qmodels.VectorParams(size=self.cfg.vector_dim, distance=qmodels.Distance.COSINE),
        )

    def upsert(self, memory_id: int, vector: List[float], payload: Dict[str, Any]) -> bool:
        if not self.is_ready() or not vector:
            return False
        if len(vector) != self.cfg.vector_dim:
            log_event(
                logger,
                logging.WARNING,
                "memory.qdrant.vector_dim_mismatch",
                "向量维度不匹配，跳过写入",
                component="memory",
                memory_id=memory_id,
                vector_dim_got=len(vector),
                vector_dim_expect=self.cfg.vector_dim,
            )
            return False

        try:
            point = qmodels.PointStruct(id=int(memory_id), vector=vector, payload=payload or {})
            self.client.upsert(
                collection_name=self.cfg.qdrant_collection,
                points=[point],
                wait=False,
            )
            return True
        except Exception:
            log_exception(
                logger,
                "memory.qdrant.upsert.error",
                "Qdrant upsert 失败",
                component="memory",
                memory_id=memory_id,
            )
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 8,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_ready() or not query_vector:
            return []
        if len(query_vector) != self.cfg.vector_dim:
            return []

        query_filter = None
        if memory_types:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="memory_type",
                        match=qmodels.MatchAny(any=memory_types),
                    )
                ]
            )

        try:
            rows = self.client.search(
                collection_name=self.cfg.qdrant_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            hits: List[Dict[str, Any]] = []
            for row in rows:
                point_id = getattr(row, "id", None)
                if point_id is None:
                    continue
                payload = getattr(row, "payload", {}) or {}
                score = float(getattr(row, "score", 0.0))
                hits.append({"memory_id": int(point_id), "vector_score": score, "payload": payload})
            return hits
        except Exception:
            log_exception(
                logger,
                "memory.qdrant.search.error",
                "Qdrant 检索失败",
                component="memory",
                limit=limit,
            )
            return []

    def delete(self, memory_ids: List[int]) -> None:
        if not self.is_ready() or not memory_ids:
            return
        ids = [int(x) for x in memory_ids]
        try:
            self.client.delete(
                collection_name=self.cfg.qdrant_collection,
                points_selector=qmodels.PointIdsList(points=ids),
                wait=False,
            )
        except Exception:
            log_exception(
                logger,
                "memory.qdrant.delete.error",
                "Qdrant 删除失败",
                component="memory",
                delete_count=len(ids),
            )
