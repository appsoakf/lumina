import re
from typing import Dict, List, Optional

from core.memory.ingestor import MemoryIngestor
from core.memory.models import MemoryRecord, MemoryType
from core.memory.policy import MemoryPolicy
from core.memory.retriever import MemoryRetriever
from core.memory.store import MemoryStore


class MemoryService:
    """Unified memory gateway for orchestrator and agents."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        policy: Optional[MemoryPolicy] = None,
        ingestor: Optional[MemoryIngestor] = None,
    ):
        self.store = store or MemoryStore()
        self.policy = policy or MemoryPolicy()
        self.ingestor = ingestor or MemoryIngestor()
        self.retriever = MemoryRetriever(self.store)

    def remember(self, user_id: str, session_id: str, content: str, tags: str = "") -> int:
        rec = MemoryRecord(
            memory_id=None,
            user_id=user_id,
            session_id=session_id,
            memory_type=MemoryType.PROFILE,
            content=content,
            tags=tags or "profile,manual",
            ttl_seconds=self.policy.default_ttl_seconds(MemoryType.PROFILE),
            source="manual",
            payload={},
        )
        return self.store.add(rec)

    def list_profile(self, user_id: str, limit: int = 8) -> List[Dict]:
        return self.retriever.get_profile(user_id=user_id, limit=limit)

    def list_commitments(self, user_id: str, limit: int = 10) -> List[Dict]:
        return self.retriever.get_open_commitments(user_id=user_id, limit=limit)

    def close_commitment(self, user_id: str, memory_id: int) -> bool:
        rows = self.store.list_recent(user_id=user_id, memory_type=MemoryType.COMMITMENT, limit=50)
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

    def build_context(self, user_id: str, query: str) -> str:
        profile = self.retriever.get_profile(user_id=user_id, limit=4)
        commitments = self.retriever.get_open_commitments(user_id=user_id, limit=4)
        relevant = self.retriever.search_relevant(user_id=user_id, query=query, limit=4)

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
            for r in relevant:
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
        # profile memory
        if self.policy.should_store_profile(user_text):
            for c in self.ingestor.extract_profile_candidates(user_text):
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=MemoryType.PROFILE,
                    content=c["content"],
                    tags=c.get("tags", "profile"),
                    ttl_seconds=self.policy.default_ttl_seconds(MemoryType.PROFILE),
                    source="user_utterance",
                    payload={"meta": meta or {}},
                )
                self.store.add(rec)

        # commitment memory
        if self.policy.should_store_commitment(user_text):
            for c in self.ingestor.extract_commitment_candidates(user_text):
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=MemoryType.COMMITMENT,
                    content=c["content"],
                    tags=c.get("tags", "commitment"),
                    ttl_seconds=self.policy.default_ttl_seconds(MemoryType.COMMITMENT),
                    source="user_utterance",
                    payload=c.get("payload", {}),
                )
                self.store.add(rec)

        # episodic summary
        if self.policy.should_store_episode(user_text):
            content = f"USER:{user_text} | ASSISTANT:{assistant_reply[:180]}"
            rec = MemoryRecord(
                memory_id=None,
                user_id=user_id,
                session_id=session_id,
                memory_type=MemoryType.EPISODIC,
                content=content,
                tags="episodic,turn",
                ttl_seconds=self.policy.default_ttl_seconds(MemoryType.EPISODIC),
                source="dialog_turn",
                payload={"meta": meta or {}},
            )
            self.store.add(rec)

        # procedural capture for successful tasks
        if meta and meta.get("task_mode") and meta.get("task_id") and not meta.get("task_error"):
            plan = meta.get("plan")
            if plan:
                rec = MemoryRecord(
                    memory_id=None,
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=MemoryType.PROCEDURAL,
                    content=f"任务模板: {plan.get('goal', '')}",
                    tags="procedural,task_template",
                    source="task_summary",
                    payload={"plan": plan, "workflow": meta.get("workflow", "general")},
                )
                self.store.add(rec)

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
