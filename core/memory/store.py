import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from core.memory.models import MemoryRecord, MemoryType, utc_now


class MemoryStore:
    """SQLite-backed memory store for local-first memory management."""

    def __init__(self, db_path: str = "D:/lumina/runtime/memory/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    confidence REAL DEFAULT 1.0,
                    ttl_seconds INTEGER,
                    source TEXT DEFAULT '',
                    payload TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_type ON memories(user_id, memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_created ON memories(user_id, created_at DESC)")
            conn.commit()
        finally:
            conn.close()

    def add(self, record: MemoryRecord) -> int:
        conn = self._connect()
        try:
            now = utc_now()
            cur = conn.execute(
                """
                INSERT INTO memories(
                    user_id, session_id, memory_type, content, tags,
                    confidence, ttl_seconds, source, payload, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.user_id,
                    record.session_id,
                    record.memory_type.value,
                    record.content,
                    record.tags,
                    record.confidence,
                    record.ttl_seconds,
                    record.source,
                    json.dumps(record.payload or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def list_recent(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 20,
    ) -> List[Dict]:
        conn = self._connect()
        try:
            if memory_type is None:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, memory_type.value, limit),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def search(self, user_id: str, query: str, limit: int = 8, memory_types: Optional[List[MemoryType]] = None) -> List[Dict]:
        conn = self._connect()
        try:
            like = f"%{query}%"
            if memory_types:
                placeholders = ",".join(["?"] * len(memory_types))
                sql = (
                    "SELECT * FROM memories WHERE user_id = ? AND (content LIKE ? OR tags LIKE ?) "
                    f"AND memory_type IN ({placeholders}) ORDER BY id DESC LIMIT ?"
                )
                params = [user_id, like, like] + [m.value for m in memory_types] + [limit]
                rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ? AND (content LIKE ? OR tags LIKE ?)
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, like, like, limit),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def update_payload(self, memory_id: int, payload: Dict) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE memories SET payload = ?, updated_at = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False), utc_now(), memory_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        payload = {}
        try:
            payload = json.loads(row["payload"] or "{}")
        except Exception:
            payload = {}
        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "memory_type": row["memory_type"],
            "content": row["content"],
            "tags": row["tags"],
            "confidence": row["confidence"],
            "ttl_seconds": row["ttl_seconds"],
            "source": row["source"],
            "payload": payload,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
