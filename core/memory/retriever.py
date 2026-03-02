from typing import Dict, List

from core.memory.models import MemoryType
from core.memory.store import MemoryStore


class MemoryRetriever:
    """Simple hybrid retrieval: recent profile/commitment + keyword search."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def get_profile(self, user_id: str, limit: int = 5) -> List[Dict]:
        return self.store.list_recent(user_id=user_id, memory_type=MemoryType.PROFILE, limit=limit)

    def get_open_commitments(self, user_id: str, limit: int = 8) -> List[Dict]:
        rows = self.store.list_recent(user_id=user_id, memory_type=MemoryType.COMMITMENT, limit=30)
        open_rows = [r for r in rows if (r.get("payload") or {}).get("status", "open") == "open"]
        return open_rows[:limit]

    def search_relevant(self, user_id: str, query: str, limit: int = 6) -> List[Dict]:
        if not query.strip():
            return self.store.list_recent(user_id=user_id, memory_type=MemoryType.EPISODIC, limit=limit)
        return self.store.search(user_id=user_id, query=query, limit=limit)
