import logging
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
from core.memory.short_term_store import ShortTermMemoryStore
from core.memory.store import LongTermMemoryStore
from core.memory.turn_summarizer import AsyncTurnSummarizer, TurnSummary, TurnSummaryExtractor
from core.memory.vector_store import QdrantVectorStore
from core.utils import log_exception

logger = logging.getLogger(__name__)


class MemoryService:
    """Unified memory gateway for orchestrator and agents."""

    def __init__(
        self,
        long_term_store: Optional[LongTermMemoryStore] = None,
        policy: Optional[MemoryPolicy] = None,
        ingestor: Optional[MemoryIngestor] = None,
        short_term_store: Optional[ShortTermMemoryStore] = None,
        turn_summary_extractor: Optional[TurnSummaryExtractor] = None,
        turn_summarizer: Optional[AsyncTurnSummarizer] = None,
        short_history_limit: int = 24,
        default_user_id: str = "default",
    ):
        self.long_term_store = long_term_store or LongTermMemoryStore()
        self.policy = policy or MemoryPolicy()
        self.ingestor = ingestor or MemoryIngestor()
        self.retriever = MemoryRetriever(self.long_term_store)
        self.short_term_store = short_term_store or ShortTermMemoryStore()
        self.short_history_limit = max(int(short_history_limit), 0)
        self.default_user_id = default_user_id
        self._last_cleanup_ts = 0.0

        self.vector_cfg = load_app_config().memory_vector
        self.embedder = OpenAIEmbeddingProvider(self.vector_cfg)
        self.vector_store = QdrantVectorStore(self.vector_cfg)
        self.hybrid_retriever = HybridMemoryRetriever(
            keyword_retriever=self.retriever,
            store=self.long_term_store,
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
        self.turn_summary_extractor = turn_summary_extractor or TurnSummaryExtractor(ingestor=self.ingestor)
        self.turn_summarizer = turn_summarizer or AsyncTurnSummarizer(
            extractor=self.turn_summary_extractor,
            on_summary=self._persist_turn_summary,
            enabled=True,
            queue_size=256,
        )

    def get_recent_history(self, session_id: str, limit_messages: Optional[int] = None) -> List[Dict[str, str]]:
        limit = self.short_history_limit if limit_messages is None else limit_messages
        rows = self.short_term_store.load_history(session_id=session_id, limit_messages=limit)
        history: List[Dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip()
            content = str(row.get("content", ""))
            if not role:
                continue
            history.append({"role": role, "content": content})
        return history

    def record_session_round(
        self,
        session_id: str,
        user_text: str,
        assistant_reply: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, str]]:
        history = self.short_term_store.load_history(session_id=session_id, limit_messages=None)
        if user_text.strip():
            history.append({"role": "user", "content": user_text})
        if assistant_reply.strip():
            history.append({"role": "assistant", "content": assistant_reply})
        self.short_term_store.save_round(session_id=session_id, history=history, metadata=metadata or {})
        return self.get_recent_history(session_id=session_id, limit_messages=None)

    def build_context(self, query: str = "") -> str:
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
        user_text: str,
        assistant_reply: str,
        meta: Optional[Dict] = None,
    ) -> None:
        self._maybe_cleanup()

        turn_item = self._build_turn_summary_item(
            session_id=session_id,
            user_text=user_text,
            assistant_reply=assistant_reply,
            meta=meta,
        )
        if not self.turn_summarizer.enqueue(turn_item):
            summary = self.turn_summary_extractor.summarize(user_text=user_text, assistant_reply=assistant_reply)
            self._persist_turn_summary(summary, turn_item)

        # commitment memory
        if self.policy.should_store_commitment(user_text):
            for c in self.ingestor.extract_commitment_candidates(user_text):
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=self.default_user_id,
                    session_id=session_id,
                    memory_type=MemoryType.COMMITMENT,
                    content=c["content"],
                    tags=c.get("tags", "commitment"),
                    ttl_seconds=self.policy.default_ttl_seconds(MemoryType.COMMITMENT),
                    source="user_utterance",
                    payload=c.get("payload", {}),
                )
                self._add_if_not_duplicate(rec)

        # procedural capture for successful tasks
        if meta and meta.get("task_mode") and meta.get("task_id") and not meta.get("task_error"):
            plan = meta.get("plan")
            if plan:
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=self.default_user_id,
                    session_id=session_id,
                    memory_type=MemoryType.PROCEDURAL,
                    content=f"任务模板: {plan.get('goal', '')}",
                    tags="procedural,task_template",
                    source="task_summary",
                    payload={"plan": plan},
                )
                self._add_if_not_duplicate(rec)

    def _build_turn_summary_item(
        self,
        session_id: str,
        user_text: str,
        assistant_reply: str,
        meta: Optional[Dict],
    ) -> Dict[str, object]:
        return {
            "session_id": session_id,
            "user_text": user_text or "",
            "assistant_reply": assistant_reply or "",
            "meta": meta or {},
        }

    def _persist_turn_summary(self, summary: TurnSummary, item: Dict[str, object]) -> None:
        try:
            self._maybe_cleanup()
            session_id = str(item.get("session_id", "")).strip()
            user_text = str(item.get("user_text", ""))
            assistant_reply = str(item.get("assistant_reply", ""))
            meta_obj = item.get("meta")
            meta = meta_obj if isinstance(meta_obj, dict) else {}

            self._persist_profile_candidates(
                session_id=session_id,
                candidates=summary.profile_candidates,
                meta=meta,
                topic=summary.topic,
            )
            self._persist_topic_summary(
                session_id=session_id,
                user_text=user_text,
                assistant_reply=assistant_reply,
                topic=summary.topic,
                meta=meta,
            )
        except Exception:
            log_exception(
                logger,
                "memory.turn_summary.persist.error",
                "Turn summary 落盘失败，已跳过本次写入",
                component="memory",
            )

    def _persist_profile_candidates(
        self,
        session_id: str,
        candidates: List[str],
        meta: Dict,
        topic: str,
    ) -> None:
        for value in candidates:
            content = str(value).strip()
            if not content:
                continue
            rec = MemoryRecord(
                memory_id=None,
                user_id=self.default_user_id,
                session_id=session_id,
                memory_type=MemoryType.PROFILE,
                content=content,
                tags="profile,preference,auto",
                ttl_seconds=self.policy.default_ttl_seconds(MemoryType.PROFILE),
                source="turn_summary",
                payload={"meta": meta, "topic": topic},
            )
            self._add_if_not_duplicate(rec)

    def _persist_topic_summary(
        self,
        session_id: str,
        user_text: str,
        assistant_reply: str,
        topic: str,
        meta: Dict,
    ) -> None:
        if not self.policy.should_store_episode(user_text):
            return

        topic_text = (topic or "").strip() or "本轮对话"
        content = f"主题:{topic_text} | USER:{user_text[:120]} | ASSISTANT:{assistant_reply[:120]}"
        rec = MemoryRecord(
            memory_id=None,
            user_id=self.default_user_id,
            session_id=session_id,
            memory_type=MemoryType.EPISODIC,
            content=content,
            tags="episodic,topic,turn",
            ttl_seconds=self.policy.default_ttl_seconds(MemoryType.EPISODIC),
            source="turn_summary",
            payload={"meta": meta, "topic": topic_text},
        )
        self._add_if_not_duplicate(rec)

    def cleanup_expired(self) -> int:
        expired_ids = self.long_term_store.purge_expired_ids()
        self._delete_vector_points(expired_ids)
        return len(expired_ids)

    def _maybe_cleanup(self) -> None:
        now = time.time()
        interval = max(self.policy.cleanup_interval_seconds(), 30)
        if now - self._last_cleanup_ts < interval:
            return
        self._last_cleanup_ts = now
        expired_ids = self.long_term_store.purge_expired_ids()
        self._delete_vector_points(expired_ids)

    def _normalize(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _hash_content(self, memory_type: MemoryType, text: str) -> str:
        normalized = self._normalize(text)
        return sha1(f"{memory_type.value}:{normalized}".encode("utf-8")).hexdigest()

    def _add_if_not_duplicate(self, rec: MemoryRecord) -> int:
        rec.content_hash = self._hash_content(rec.memory_type, rec.content)
        window = self.policy.dedupe_window_seconds(rec.memory_type)
        existing_id = self.long_term_store.find_recent_duplicate_id(
            memory_type=rec.memory_type,
            content_hash=rec.content_hash,
            window_seconds=window,
        )
        if existing_id is not None:
            # Idempotent write path: return existing id for stable behavior.
            return existing_id
        memory_id = self.long_term_store.add(rec)
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
            self.turn_summarizer.close()
        except Exception:
            log_exception(
                logger,
                "memory.turn_summary.close.error",
                "Turn summary 关闭失败",
                component="memory",
            )
        try:
            self.vector_indexer.close()
        except Exception:
            log_exception(
                logger,
                "memory.vector_indexer.close.error",
                "向量索引器关闭失败",
                component="memory",
            )
