import logging
import re
import time
from hashlib import sha1
from typing import Dict, List, Optional

from core.config import load_app_config
from core.memory.embedding import OpenAIEmbeddingProvider
from core.memory.hybrid_retriever import HybridMemoryRetriever
from core.memory.indexer import MemoryVectorIndexer
from core.memory.ingestor import MemoryIngestor
from core.memory.models import MemoryRecord, MemoryType
from core.memory.policy import MemoryPolicy
from core.memory.retriever import MemoryRetriever
from core.memory.store import MemoryStore
from core.memory.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class MemoryService:
    """Unified memory gateway for orchestrator and agents."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        policy: Optional[MemoryPolicy] = None,
        ingestor: Optional[MemoryIngestor] = None,
        default_user_id: str = "default",
    ):
        self.store = store or MemoryStore()
        self.policy = policy or MemoryPolicy()
        self.ingestor = ingestor or MemoryIngestor()
        self.retriever = MemoryRetriever(self.store)
        self.default_user_id = default_user_id
        self._last_cleanup_ts = 0.0

        self.vector_cfg = load_app_config().memory_vector
        self.embedder = OpenAIEmbeddingProvider(self.vector_cfg)
        self.vector_store = QdrantVectorStore(self.vector_cfg)
        self.hybrid_retriever = HybridMemoryRetriever(
            keyword_retriever=self.retriever,
            store=self.store,
            embedder=self.embedder,
            vector_store=self.vector_store,
            cfg=self.vector_cfg,
        )
        self.vector_indexer = MemoryVectorIndexer(
            embedder=self.embedder,
            vector_store=self.vector_store,
            enabled=self.vector_cfg.enabled and self.vector_cfg.write_async,
            queue_size=self.vector_cfg.queue_size,
            max_retries=self.vector_cfg.max_retries,
        )

    def remember(self, user_id: str = "", session_id: str = "", content: str = "", tags: str = "") -> int:
        self._maybe_cleanup()
        rec = MemoryRecord(
            memory_id=None,
            user_id=self.default_user_id,
            session_id=session_id,
            memory_type=MemoryType.PROFILE,
            content=content,
            tags=tags or "profile,manual",
            ttl_seconds=self.policy.default_ttl_seconds(MemoryType.PROFILE),
            source="manual",
            payload={},
        )
        return self._add_if_not_duplicate(rec)

    def list_profile(self, user_id: str = "", limit: int = 8) -> List[Dict]:
        self._maybe_cleanup()
        return self.retriever.get_profile(limit=limit)

    def list_commitments(self, user_id: str = "", limit: int = 10) -> List[Dict]:
        self._maybe_cleanup()
        return self.retriever.get_open_commitments(limit=limit)

    def close_commitment(self, user_id: str = "", memory_id: int = 0) -> bool:
        self._maybe_cleanup()
        rows = self.store.list_recent(memory_type=MemoryType.COMMITMENT, limit=50)
        target = None
        for r in rows:
            if int(r.get("memory_id", -1)) == int(memory_id):
                target = r
                break
        if not target:
            return False

        payload = target.get("payload") or {}
        payload["status"] = "done"
        self.store.update_payload(memory_id=memory_id, payload=payload)
        return True

    def build_context(self, user_id: str = "", query: str = "") -> str:
        self._maybe_cleanup()
        profile = self.retriever.get_profile(limit=4)
        commitments = self.retriever.get_open_commitments(limit=4)
        if self.vector_cfg.enabled:
            relevant = self.hybrid_retriever.search(
                query=query,
                limit=4,
                memory_types=[
                    MemoryType.EPISODIC.value,
                    MemoryType.PROCEDURAL.value,
                    MemoryType.ARTIFACT.value,
                ],
            )
        else:
            relevant = self.retriever.search_relevant(query=query, limit=4)

        lines = []
        if profile:
            lines.append("用户偏好:")
            for p in profile:
                lines.append(f"- {p.get('content')}")

        if commitments:
            lines.append("未完成事项:")
            for c in commitments:
                due = (c.get("payload") or {}).get("due", "")
                suffix = f" (截止:{due})" if due else ""
                lines.append(f"- #{c.get('memory_id')} {c.get('content')}{suffix}")

        if relevant:
            lines.append("相关历史:")
            for r in self._dedupe_rows(relevant):
                lines.append(f"- {r.get('content')}")

        return "\n".join(lines).strip()

    def ingest_turn(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        assistant_reply: str,
        meta: Optional[Dict] = None,
    ) -> None:
        _ = user_id  # kept for backward-compatible call signature
        self._maybe_cleanup()
        effective_user_id = self.default_user_id
        # profile memory
        if self.policy.should_store_profile(user_text):
            for c in self.ingestor.extract_profile_candidates(user_text):
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=effective_user_id,
                    session_id=session_id,
                    memory_type=MemoryType.PROFILE,
                    content=c["content"],
                    tags=c.get("tags", "profile"),
                    ttl_seconds=self.policy.default_ttl_seconds(MemoryType.PROFILE),
                    source="user_utterance",
                    payload={"meta": meta or {}},
                )
                self._add_if_not_duplicate(rec)

        # commitment memory
        if self.policy.should_store_commitment(user_text):
            for c in self.ingestor.extract_commitment_candidates(user_text):
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=effective_user_id,
                    session_id=session_id,
                    memory_type=MemoryType.COMMITMENT,
                    content=c["content"],
                    tags=c.get("tags", "commitment"),
                    ttl_seconds=self.policy.default_ttl_seconds(MemoryType.COMMITMENT),
                    source="user_utterance",
                    payload=c.get("payload", {}),
                )
                self._add_if_not_duplicate(rec)

        # episodic summary
        if self.policy.should_store_episode(user_text):
            content = f"USER:{user_text} | ASSISTANT:{assistant_reply[:180]}"
            rec = MemoryRecord(
                memory_id=None,
                user_id=effective_user_id,
                session_id=session_id,
                memory_type=MemoryType.EPISODIC,
                content=content,
                tags="episodic,turn",
                ttl_seconds=self.policy.default_ttl_seconds(MemoryType.EPISODIC),
                source="dialog_turn",
                payload={"meta": meta or {}},
            )
            self._add_if_not_duplicate(rec)

        # procedural capture for successful tasks
        if meta and meta.get("task_mode") and meta.get("task_id") and not meta.get("task_error"):
            plan = meta.get("plan")
            if plan:
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=effective_user_id,
                    session_id=session_id,
                    memory_type=MemoryType.PROCEDURAL,
                    content=f"任务模板: {plan.get('goal', '')}",
                    tags="procedural,task_template",
                    source="task_summary",
                    payload={"plan": plan, "workflow": meta.get("workflow", "general")},
                )
                self._add_if_not_duplicate(rec)

    def parse_memory_command(self, text: str) -> Optional[Dict]:
        t = text.strip()

        if t.startswith("记住"):
            value = re.sub(r"^记住[:：]?", "", t).strip()
            return {"action": "remember", "value": value}

        if t.startswith("我的偏好") or t.startswith("查看记忆"):
            return {"action": "list_profile"}

        if t.startswith("我的待办") or t.startswith("查看待办"):
            return {"action": "list_commitments"}

        m = re.search(r"完成待办\s*#?(\d+)", t)
        if m:
            return {"action": "close_commitment", "memory_id": int(m.group(1))}

        return None

    def cleanup_expired(self, user_id: Optional[str] = None) -> int:
        _ = user_id  # kept for backward-compatible call signature
        expired_ids = self.store.purge_expired_ids()
        self._delete_vector_points(expired_ids)
        return len(expired_ids)

    def _maybe_cleanup(self) -> None:
        now = time.time()
        interval = max(self.policy.cleanup_interval_seconds(), 30)
        if now - self._last_cleanup_ts < interval:
            return
        self._last_cleanup_ts = now
        expired_ids = self.store.purge_expired_ids()
        self._delete_vector_points(expired_ids)

    def _normalize(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _hash_content(self, memory_type: MemoryType, text: str) -> str:
        normalized = self._normalize(text)
        return sha1(f"{memory_type.value}:{normalized}".encode("utf-8")).hexdigest()

    def _add_if_not_duplicate(self, rec: MemoryRecord) -> int:
        rec.content_hash = self._hash_content(rec.memory_type, rec.content)
        window = self.policy.dedupe_window_seconds(rec.memory_type)
        existing_id = self.store.find_recent_duplicate_id(
            user_id=None,
            memory_type=rec.memory_type,
            content_hash=rec.content_hash,
            window_seconds=window,
        )
        if existing_id is not None:
            # Idempotent write path: return existing id for stable behavior.
            return existing_id
        memory_id = self.store.add(rec)
        self._index_memory(memory_id, rec)
        return memory_id

    def _index_memory(self, memory_id: int, rec: MemoryRecord) -> None:
        if not self.vector_cfg.enabled:
            return

        payload = {
            "memory_type": rec.memory_type.value,
            "tags": rec.tags,
            "created_at": rec.created_at,
            "ttl_seconds": rec.ttl_seconds,
            "source": rec.source,
            "content_hash": rec.content_hash,
        }

        if self.vector_cfg.write_async and self.vector_indexer.is_enabled():
            self.vector_indexer.enqueue(memory_id=memory_id, content=rec.content, payload=payload)
            return

        vector = self.embedder.embed(rec.content)
        if not vector:
            return
        self.vector_store.upsert(memory_id=memory_id, vector=vector, payload=payload)

    def _delete_vector_points(self, memory_ids: List[int]) -> None:
        if not memory_ids or not self.vector_cfg.enabled:
            return
        self.vector_store.delete(memory_ids)

    def _dedupe_rows(self, rows: List[Dict]) -> List[Dict]:
        seen = set()
        out: List[Dict] = []
        for row in rows:
            key = row.get("memory_id") or f"{row.get('memory_type')}::{row.get('content_hash')}::{row.get('content')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def close(self) -> None:
        try:
            self.vector_indexer.close()
        except Exception as exc:
            logger.warning(f"Memory vector indexer close failed: {exc}")
