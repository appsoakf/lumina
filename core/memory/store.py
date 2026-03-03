import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.memory.models import MemoryRecord, MemoryType, utc_now
from core.paths import memory_db_path


class LongTermMemoryStore:
    """SQLite-backed memory store for local-first memory management."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path is not None else memory_db_path()
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
                    content_hash TEXT DEFAULT '',
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
            self._ensure_column(conn, "content_hash", "TEXT DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_type ON memories(user_id, memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_created ON memories(user_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_hash ON memories(user_id, memory_type, content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type_created ON memories(memory_type, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type_hash ON memories(memory_type, content_hash)")
            conn.commit()
        finally:
            conn.close()

    def _ensure_column(self, conn: sqlite3.Connection, name: str, ddl: str) -> None:
        rows = conn.execute("PRAGMA table_info(memories)").fetchall()
        names = {r[1] for r in rows}
        if name not in names:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {ddl}")

    def add(self, record: MemoryRecord) -> int:
        conn = self._connect()
        try:
            now = utc_now()
            cur = conn.execute(
                """
                INSERT INTO memories(
                    user_id, session_id, memory_type, content, content_hash, tags,
                    confidence, ttl_seconds, source, payload, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.user_id,
                    record.session_id,
                    record.memory_type.value,
                    record.content,
                    record.content_hash,
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
        memory_type: Optional[MemoryType] = None,
        limit: int = 20,
    ) -> List[Dict]:
        conn = self._connect()
        try:
            fetch_limit = max(limit * 3, limit)
            if memory_type is None:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (fetch_limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE memory_type = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (memory_type.value, fetch_limit),
                ).fetchall()
            result = [self._row_to_dict(r) for r in rows]
            result = [r for r in result if not self._is_expired(r)]
            return result[:limit]
        finally:
            conn.close()

    def search(
        self,
        query: str = "",
        limit: int = 8,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[Dict]:
        conn = self._connect()
        try:
            fetch_limit = max(limit * 4, limit)
            like = f"%{query}%"
            if memory_types:
                placeholders = ",".join(["?"] * len(memory_types))
                sql = (
                    "SELECT * FROM memories WHERE (content LIKE ? OR tags LIKE ?) "
                    f"AND memory_type IN ({placeholders}) ORDER BY id DESC LIMIT ?"
                )
                params = [like, like] + [m.value for m in memory_types] + [fetch_limit]
                rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE (content LIKE ? OR tags LIKE ?)
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (like, like, fetch_limit),
                ).fetchall()
            result = [self._row_to_dict(r) for r in rows]
            result = [r for r in result if not self._is_expired(r)]
            return result[:limit]
        finally:
            conn.close()

    def exists_recent_duplicate(
        self,
        memory_type: MemoryType,
        content_hash: str,
        window_seconds: int,
    ) -> bool:
        return self.find_recent_duplicate_id(
            memory_type=memory_type,
            content_hash=content_hash,
            window_seconds=window_seconds,
        ) is not None

    def find_recent_duplicate_id(
        self,
        memory_type: MemoryType,
        content_hash: str,
        window_seconds: int,
    ) -> Optional[int]:
        if not content_hash:
            return None

        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, created_at, ttl_seconds, content_hash
                FROM memories
                WHERE memory_type = ? AND content_hash = ?
                ORDER BY id DESC
                LIMIT 30
                """,
                (memory_type.value, content_hash),
            ).fetchall()

            now = datetime.now(timezone.utc)
            for row in rows:
                item = {
                    "memory_id": row["id"],
                    "created_at": row["created_at"],
                    "ttl_seconds": row["ttl_seconds"],
                    "content_hash": row["content_hash"],
                }
                if self._is_expired(item):
                    continue
                created = self._parse_dt(row["created_at"])
                if created is None:
                    continue
                if now - created <= timedelta(seconds=max(window_seconds, 0)):
                    return int(row["id"])
            return None
        finally:
            conn.close()

    def get_by_ids(self, memory_ids: List[int]) -> Dict[int, Dict]:
        if not memory_ids:
            return {}

        ids = [int(i) for i in memory_ids]
        placeholders = ",".join(["?"] * len(ids))
        conn = self._connect()
        try:
            rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(ids)).fetchall()
            out: Dict[int, Dict] = {}
            for row in rows:
                mapped = self._row_to_dict(row)
                if self._is_expired(mapped):
                    continue
                out[int(mapped["memory_id"])] = mapped
            return out
        finally:
            conn.close()

    def purge_expired(self) -> int:
        return len(self.purge_expired_ids())

    def purge_expired_ids(self) -> List[int]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT id, created_at, ttl_seconds FROM memories").fetchall()

            expired_ids: List[int] = []
            for row in rows:
                if self._is_expired(
                    {
                        "created_at": row["created_at"],
                        "ttl_seconds": row["ttl_seconds"],
                    }
                ):
                    expired_ids.append(int(row["id"]))

            if not expired_ids:
                return []

            placeholders = ",".join(["?"] * len(expired_ids))
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", tuple(expired_ids))
            conn.commit()
            return expired_ids
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
            "content_hash": row["content_hash"] or "",
            "tags": row["tags"],
            "confidence": row["confidence"],
            "ttl_seconds": row["ttl_seconds"],
            "source": row["source"],
            "payload": payload,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _parse_dt(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _is_expired(self, record: Dict) -> bool:
        ttl = record.get("ttl_seconds")
        if ttl is None:
            return False
        try:
            ttl_int = int(ttl)
        except Exception:
            return False
        if ttl_int <= 0:
            return True
        created = self._parse_dt(record.get("created_at"))
        if created is None:
            return False
        return datetime.now(timezone.utc) > created + timedelta(seconds=ttl_int)
