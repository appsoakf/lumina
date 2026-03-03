import re
from datetime import datetime, timezone
from typing import Dict, List

from core.memory.models import MemoryType
from core.memory.store import LongTermMemoryStore


class MemoryRetriever:
    """Hybrid retrieval with keyword hit, type weights and time decay."""

    TYPE_WEIGHT = {
        MemoryType.PROFILE.value: 1.35,
        MemoryType.COMMITMENT.value: 1.25,
        MemoryType.PROCEDURAL.value: 1.15,
        MemoryType.EPISODIC.value: 1.0,
        MemoryType.ARTIFACT.value: 1.05,
    }

    def __init__(self, store: LongTermMemoryStore):
        self.store = store

    def get_profile(self, limit: int = 5) -> List[Dict]:
        return self.store.list_recent(memory_type=MemoryType.PROFILE, limit=limit)

    def get_open_commitments(self, limit: int = 8) -> List[Dict]:
        rows = self.store.list_recent(memory_type=MemoryType.COMMITMENT, limit=30)
        open_rows = [r for r in rows if (r.get("payload") or {}).get("status", "open") == "open"]
        return open_rows[:limit]

    def search_relevant(self, query: str = "", limit: int = 6) -> List[Dict]:
        query = (query or "").strip()
        if not query:
            return self.store.list_recent(memory_type=MemoryType.EPISODIC, limit=limit)

        # Candidate pool = keyword hits + recent records.
        pool = self.store.search(query=query, limit=max(limit * 5, 20))
        pool.extend(self.store.list_recent(memory_type=None, limit=max(limit * 4, 20)))
        deduped = self._dedupe(pool)

        tokens = self._tokenize(query)
        scored = []
        for row in deduped:
            score = self._rank_score(row, tokens)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _dedupe(self, rows: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        seen = set()
        for row in rows:
            key = row.get("memory_id") or f"{row.get('memory_type')}::{row.get('content_hash')}::{row.get('content')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _rank_score(self, row: Dict, tokens: List[str]) -> float:
        return self._keyword_score(row, tokens) + self._type_score(row) + self._time_decay_score(row)

    def _keyword_score(self, row: Dict, tokens: List[str]) -> float:
        text = f"{row.get('content', '')} {row.get('tags', '')}".lower()
        if not tokens:
            return 0.0
        score = 0.0
        for token in tokens:
            if token in text:
                score += 1.0
                if token in (row.get("content", "").lower()):
                    score += 0.3
        return score

    def _type_score(self, row: Dict) -> float:
        return self.TYPE_WEIGHT.get(str(row.get("memory_type", "")), 1.0)

    def _time_decay_score(self, row: Dict) -> float:
        created_at = self._parse_dt(row.get("created_at"))
        if created_at is None:
            return 0.0

        now = datetime.now(timezone.utc)
        age_hours = max((now - created_at).total_seconds() / 3600.0, 0.0)
        # Soft decay: within 3 days still has visible recency bonus.
        return 2.5 / (1.0 + age_hours / 72.0)

    def _tokenize(self, query: str) -> List[str]:
        q = query.lower().strip()
        parts = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", q)
        tokens = [p for p in parts if len(p) >= 2]
        if not tokens and q:
            tokens = [q]
        return tokens

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
