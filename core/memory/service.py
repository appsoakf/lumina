import json
import logging
import math
import re
import threading
import time
from datetime import datetime
from hashlib import sha1
from typing import Dict, List, Optional

from core.config import MemoryVectorConfig, load_app_config
from core.memory.memory_module_engine import Memory as EngineMemory
from core.memory.memory_module_engine import OpenAIEmbedding
from core.memory.memory_module_engine.embedding import EmbeddingProvider as EngineEmbeddingProvider
from core.memory.memory_module_engine.models import MemoryItem
from core.paths import runtime_memory_dir, runtime_sessions_dir
from core.utils import log_exception

logger = logging.getLogger(__name__)


class DeterministicEmbeddingProvider(EngineEmbeddingProvider):
    """Local deterministic embedding provider for offline-safe memory operations."""

    def __init__(self, dim: int = 384):
        self._dim = max(int(dim), 64)

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        source = (text or "").strip().lower()
        if not source:
            return vec
        for idx, ch in enumerate(source):
            slot = idx % self._dim
            vec[slot] += ((ord(ch) % 97) + 1) / 97.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def get_dimension(self) -> int:
        return self._dim


class MemoryService:
    """Unified memory gateway for orchestrator and agents."""

    # 轻量规则：用于把输入映射为结构化记忆标签。
    PROFILE_PATTERNS = [
        re.compile(r"我喜欢(.+)$"),
        re.compile(r"我不喜欢(.+)$"),
        re.compile(r"我的偏好(?:是)?(.+)$"),
        re.compile(r"我偏好(.+)$"),
        re.compile(r"我习惯(.+)$"),
    ]
    COMMITMENT_PATTERNS = [
        re.compile(r"(?:提醒我|记得|待办[:：]?)(.+)$"),
        re.compile(r"(.+?)(?:截止|在)(\d{1,2}月\d{1,2}日|\d{4}[/-]\d{1,2}[/-]\d{1,2})"),
    ]
    LIST_SPLIT_RE = re.compile(r"[、,，]|和|以及")
    TOPIC_SPLIT_RE = re.compile(r"[。！？!?;\n]")

    DEDUPE_WINDOW_SECONDS = {
        "profile": 90 * 24 * 3600,
        "commitment": 24 * 3600,
        "episodic": 2 * 3600,
        "procedural": 7 * 24 * 3600,
    }

    def __init__(
        self,
        short_history_limit: int = 24,
        default_user_id: str = "default",
    ):
        self.short_history_limit = max(int(short_history_limit), 0)
        self.default_user_id = default_user_id

        self.vector_cfg = load_app_config().memory_vector
        self._dedupe_lock = threading.RLock()
        self._session_lock = threading.RLock()
        self._recent_hashes: Dict[str, float] = {}

        self._session_dir = runtime_sessions_dir()
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._engine = self._build_engine(self.vector_cfg)

    def _build_engine(self, vector_cfg: MemoryVectorConfig) -> EngineMemory:
        storage_dir = runtime_memory_dir() / "memory_module"
        storage_dir.mkdir(parents=True, exist_ok=True)

        embedder = self._build_embedder(vector_cfg)
        # 运行默认关闭 LLM 抽取，避免主链路强依赖外网。
        overrides = {"llm_enabled": False}
        return EngineMemory(
            storage_path=str(storage_dir),
            embedding_provider=embedder,
            auto_consolidate=True,
            config_overrides=overrides,
        )

    def _build_embedder(self, vector_cfg: MemoryVectorConfig) -> EngineEmbeddingProvider:
        api_key = (vector_cfg.embedding_api_key or "").strip()
        use_openai = bool(vector_cfg.enabled and api_key)
        if use_openai:
            try:
                return OpenAIEmbedding(
                    api_key=api_key,
                    model=vector_cfg.embedding_model,
                    dimensions=max(int(vector_cfg.vector_dim), 64),
                    base_url=vector_cfg.embedding_api_url or None,
                    cache_enabled=True,
                    cache_max_entries=4096,
                )
            except Exception:
                log_exception(
                    logger,
                    "memory.embedder.init.error",
                    "OpenAI embedding 初始化失败，回退本地确定性 embedding",
                    component="memory",
                    fallback="deterministic_embedding",
                )
        return DeterministicEmbeddingProvider(dim=max(int(vector_cfg.vector_dim), 384))

    def _safe_session_id(self, session_id: str) -> str:
        raw = str(session_id or "default").strip() or "default"
        return re.sub(r"[^A-Za-z0-9._-]", "_", raw)

    def _session_path(self, session_id: str):
        safe = self._safe_session_id(session_id)
        return self._session_dir / f"{safe}.json"

    def _load_round(self, session_id: str) -> Dict[str, object]:
        path = self._session_path(session_id)
        if not path.exists():
            return {"session_id": self._safe_session_id(session_id), "history": [], "metadata": {}}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history")
            metadata = data.get("metadata")
            return {
                "session_id": data.get("session_id", self._safe_session_id(session_id)),
                "saved_at": data.get("saved_at", ""),
                "history": history if isinstance(history, list) else [],
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        except Exception:
            return {"session_id": self._safe_session_id(session_id), "history": [], "metadata": {}}

    def _save_round(self, session_id: str, history: List[Dict[str, object]], metadata: Dict[str, object]) -> None:
        path = self._session_path(session_id)
        payload = {
            "session_id": self._safe_session_id(session_id),
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "history": history,
            "metadata": metadata,
        }
        temp_path = str(path) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        # 原子替换，避免并发写入导致半文件状态。
        import os

        os.replace(temp_path, path)

    def get_recent_history(self, session_id: str, limit_messages: Optional[int] = None) -> List[Dict[str, str]]:
        limit = self.short_history_limit if limit_messages is None else limit_messages
        with self._session_lock:
            payload = self._load_round(session_id)
        history = payload.get("history")
        if not isinstance(history, list):
            return []

        rows = history[-limit:] if isinstance(limit, int) and limit > 0 else history
        out: List[Dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip()
            content = str(row.get("content", ""))
            if not role:
                continue
            out.append({"role": role, "content": content})
        return out

    def record_session_round(
        self,
        session_id: str,
        user_text: str,
        assistant_reply: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, str]]:
        with self._session_lock:
            payload = self._load_round(session_id)
            history = payload.get("history") if isinstance(payload.get("history"), list) else []
            if (user_text or "").strip():
                history.append({"role": "user", "content": str(user_text)})
            if (assistant_reply or "").strip():
                history.append({"role": "assistant", "content": str(assistant_reply)})
            self._save_round(session_id, history, metadata or {})
        return self.get_recent_history(session_id=session_id, limit_messages=None)

    def ingest_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_reply: str,
        meta: Optional[Dict] = None,
    ) -> None:
        user_text = str(user_text or "").strip()
        assistant_reply = str(assistant_reply or "").strip()
        meta = meta if isinstance(meta, dict) else {}

        # 1) 结构化偏好与待办。
        for profile in self._extract_profile_candidates(user_text):
            self._persist_memory("profile", profile)

        for commitment in self._extract_commitment_candidates(user_text):
            self._persist_memory("commitment", commitment)

        # 2) 对话片段。
        if len(user_text) >= 6:
            topic = self._extract_topic(user_text=user_text, assistant_reply=assistant_reply)
            episodic = f"主题:{topic} | USER:{user_text[:120]} | ASSISTANT:{assistant_reply[:120]}"
            self._persist_memory("episodic", episodic)

        # 3) 任务经验。
        if meta.get("task_mode") and meta.get("task_id") and not meta.get("task_error"):
            plan = meta.get("plan")
            if isinstance(plan, dict):
                goal = str(plan.get("goal", "")).strip()
                if goal:
                    self._persist_memory("procedural", f"任务模板: {goal}")

    def build_context(self, query: str = "") -> str:
        query_text = str(query or "").strip()
        profile_rows = self._prefixed_entries(
            query=(f"{query_text} 偏好 喜欢 习惯 profile".strip()),
            prefix="profile",
            limit=4,
        )
        commitment_rows = self._prefixed_entries(
            query=(f"{query_text} 待办 提醒 截止 commitment".strip()),
            prefix="commitment",
            limit=4,
        )
        relevant_rows = self._search_entries(query=query_text or "最近对话", limit=8)

        covered_ids = {item.id for item in profile_rows}
        covered_ids.update(item.id for item in commitment_rows)
        relevant_rows = [item for item in relevant_rows if item.id not in covered_ids]

        lines: List[str] = []
        if profile_rows:
            lines.append("用户偏好:")
            for item in profile_rows:
                lines.append(f"- {self._strip_prefix(item.content, 'profile')}")

        if commitment_rows:
            lines.append("未完成事项:")
            for item in commitment_rows:
                lines.append(f"- {self._strip_prefix(item.content, 'commitment')}")

        if relevant_rows:
            lines.append("相关历史:")
            for item in relevant_rows[:6]:
                lines.append(f"- {self._strip_prefix(item.content, 'episodic')}")

        return "\n".join(lines).strip()

    def _persist_memory(self, memory_type: str, content: str) -> str:
        normalized = str(content or "").strip()
        if not normalized:
            return ""
        full_text = f"{memory_type}: {normalized}"
        if self._is_recent_duplicate(memory_type, full_text):
            return ""
        try:
            return self._engine.add(full_text)
        except Exception:
            log_exception(
                logger,
                "memory.engine.add.error",
                "memory_module 写入失败，已跳过该条记忆",
                component="memory",
                fallback="skip_memory_write",
            )
            return ""

    def _hash_content(self, memory_type: str, text: str) -> str:
        normalized = " ".join(str(text or "").strip().lower().split())
        return sha1(f"{memory_type}:{normalized}".encode("utf-8")).hexdigest()

    def _is_recent_duplicate(self, memory_type: str, content: str) -> bool:
        content_hash = self._hash_content(memory_type, content)
        key = f"{memory_type}:{content_hash}"
        now = time.time()
        window = int(self.DEDUPE_WINDOW_SECONDS.get(memory_type, 24 * 3600))

        with self._dedupe_lock:
            last = self._recent_hashes.get(key)
            self._recent_hashes[key] = now

            # 控制 map 尺寸并清理过旧 key。
            if len(self._recent_hashes) > 4096:
                cutoff = now - max(window, 3600)
                stale_keys = [k for k, ts in self._recent_hashes.items() if ts < cutoff]
                for stale in stale_keys:
                    self._recent_hashes.pop(stale, None)

        return last is not None and (now - last) <= window

    def _search_entries(self, query: str, limit: int) -> List[MemoryItem]:
        try:
            rows = self._engine.search(query=query, top_k=max(int(limit), 1))
        except Exception:
            log_exception(
                logger,
                "memory.engine.search.error",
                "memory_module 检索失败，返回空结果",
                component="memory",
                fallback="empty_context",
            )
            return []

        out: List[MemoryItem] = []
        seen = set()
        for item in rows:
            if item.id in seen:
                continue
            seen.add(item.id)
            out.append(item)
        return out[: max(int(limit), 1)]

    def _prefixed_entries(self, query: str, prefix: str, limit: int) -> List[MemoryItem]:
        raw = self._search_entries(query=query, limit=max(limit * 3, limit))
        prefix_flag = f"{prefix}:"
        tagged: List[MemoryItem] = []
        for item in raw:
            content = str(item.content or "").strip().lower()
            if content.startswith(prefix_flag):
                tagged.append(item)
            if len(tagged) >= max(int(limit), 1):
                break
        return tagged[: max(int(limit), 1)]

    def _strip_prefix(self, text: str, prefix: str) -> str:
        content = str(text or "").strip()
        prefix_flag = f"{prefix}:"
        if content.lower().startswith(prefix_flag):
            return content[len(prefix_flag) :].strip()
        return content

    def _extract_profile_candidates(self, text: str) -> List[str]:
        source = str(text or "").strip()
        if not source:
            return []

        candidates: List[str] = []
        for pattern in self.PROFILE_PATTERNS:
            matched = pattern.search(source)
            if not matched:
                continue
            raw = matched.group(1).strip(" ，,。！？!?\n\t")
            if not raw:
                continue
            pieces = [p.strip(" ，,。！？!?\n\t") for p in self.LIST_SPLIT_RE.split(raw) if p.strip()]
            candidates.extend(pieces or [raw])

        deduped: List[str] = []
        seen = set()
        for item in candidates:
            norm = " ".join(item.lower().split())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(item)
            if len(deduped) >= 6:
                break
        return deduped

    def _extract_commitment_candidates(self, text: str) -> List[str]:
        source = str(text or "").strip()
        if not source:
            return []

        out: List[str] = []
        for pattern in self.COMMITMENT_PATTERNS:
            matched = pattern.search(source)
            if not matched:
                continue
            if len(matched.groups()) == 1:
                todo = matched.group(1).strip()
                due = ""
            else:
                todo = matched.group(1).strip()
                due = matched.group(2).strip()
            if not todo:
                continue
            out.append(f"{todo}{f' | due:{due}' if due else ''}")

        deduped: List[str] = []
        seen = set()
        for item in out:
            norm = " ".join(item.lower().split())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(item)
        return deduped

    def _extract_topic(self, user_text: str, assistant_reply: str) -> str:
        def first_clause(text: str) -> str:
            src = str(text or "").strip()
            if not src:
                return ""
            parts = [p.strip() for p in self.TOPIC_SPLIT_RE.split(src) if p.strip()]
            return parts[0] if parts else src

        topic = first_clause(user_text)
        for prefix in ["请帮我", "帮我", "请", "我想让你", "我想", "我希望", "我需要", "能不能", "可以"]:
            if topic.startswith(prefix):
                topic = topic[len(prefix) :].strip()
                break

        if not topic:
            topic = first_clause(assistant_reply)

        topic = topic.strip() or "本轮对话"
        if len(topic) > 48:
            topic = topic[:48].rstrip() + "..."
        return topic

    def close(self) -> None:
        try:
            self._engine.close()
        except Exception:
            log_exception(
                logger,
                "memory.engine.close.error",
                "memory_module 关闭失败",
                component="memory",
            )
