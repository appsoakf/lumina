"""Long-term memory store (SQLite + Qdrant + FTS)."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .models import MemoryItem, MemoryMetadata
from .utils import DecayEngine, MemoryCompressor, clamp, normalize_text, tokenize


class LongTermMemory:
    """Persistent long-term memory with hybrid retrieval support."""

    def __init__(self, storage_path: str, vector_dim: int = 384):
        os.makedirs(storage_path, exist_ok=True)

        db_path = os.path.join(storage_path, "memory.db")
        # 写连接：所有写事务（persist / consolidate / mark_access）都走这条连接。
        # 因此设置check_same_thread=False，允许这个连接被不同线程访问
        self._write_db = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self._write_db.row_factory = sqlite3.Row
        self._write_db.execute("PRAGMA journal_mode=WAL")
        self._write_db.execute("PRAGMA busy_timeout=5000")

        # 读连接：检索只读查询走这条连接，尽量避免与写事务互相阻塞。
        # isolation_level=None 使用 autocommit 读，缩短读事务生命周期。
        self._read_db = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._read_db.row_factory = sqlite3.Row
        self._read_db.execute("PRAGMA journal_mode=WAL")
        self._read_db.execute("PRAGMA busy_timeout=5000")
        self._read_db.execute("PRAGMA query_only=1")

        self._write_lock = threading.RLock()
        self._read_lock = threading.RLock()
        # 延迟向量删除集合：用于“DB 事务先提交，再删向量”保证一致性。
        self._pending_vector_deletes: set[str] = set()

        self.vector_store = QdrantClient(path=os.path.join(storage_path, "vectors"))
        self.vector_dim = vector_dim
        self.collection_name = f"long_term_{self.vector_dim}"

        self._init_tables()
        self._init_collection()

    def _init_tables(self):
        db = self._write_db
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                recall_count INTEGER DEFAULT 0,
                metadata TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS consolidation_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._validate_memories_schema()
        self._init_indexes()
        self._init_fts()
        # 初始化时将cursor置为0
        self._set_state_value_locked("cursor_rowid", "0")
        db.commit()

    def _validate_memories_schema(self):
        """
        严格校验 memories 表结构是否与当前版本一致。

        该项目已移除历史迁移/兼容逻辑；若本地仍使用旧库结构，
        直接抛错并提示清理 `memory_data` 后重建。
        """
        expected_columns = {
            "id",
            "content",
            "importance",
            "recall_count",
            "metadata",
            "created_at",
            "updated_at",
        }
        actual_columns = self._table_columns(self._write_db, "memories")
        if actual_columns != expected_columns:
            raise RuntimeError(
                "Unsupported memories schema detected. "
                "Expected columns: "
                f"{sorted(expected_columns)}, got: {sorted(actual_columns)}. "
                "Please reset storage (e.g. remove memory_data/memory.db) and reinitialize."
            )

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def _init_indexes(self):
        db = self._write_db
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)"
        )

    def _init_fts(self):
        db = self._write_db
        db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(id UNINDEXED, content, tokenize='unicode61')
            """
        )
        self._rebuild_fts()

    def _rebuild_fts(self):
        db = self._write_db
        db.execute("DELETE FROM memories_fts")
        db.execute(
            """
            INSERT INTO memories_fts(id, content)
            SELECT id, content
            FROM memories
            """
        )

    def _init_collection(self):
        collections = [c.name for c in self.vector_store.get_collections().collections]
        if self.collection_name not in collections:
            self.vector_store.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            return

        info = self.vector_store.get_collection(self.collection_name)
        existing_dim = None
        try:
            existing_dim = info.config.params.vectors.size
        except Exception:
            existing_dim = self.vector_dim
        if existing_dim != self.vector_dim:
            self.collection_name = f"long_term_{self.vector_dim}_v2"
            collections = [c.name for c in self.vector_store.get_collections().collections]
            if self.collection_name not in collections:
                self.vector_store.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                )

    def add(self, item: MemoryItem, commit: bool = True) -> str:
        now = time.time()
        item.metadata.store = "long_term"
        item.metadata.created_at = item.metadata.created_at or now

        with self._write_lock:
            self._write_db.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, content, importance, recall_count, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.content,
                    float(item.importance),
                    int(item.recall_count),
                    json.dumps(item.metadata.to_dict(), ensure_ascii=False),
                    float(item.metadata.created_at),
                    now,
                ),
            )
            self._upsert_fts(item.id, item.content)
            if item.embedding:
                self._upsert_vector(item.id, item.embedding, item)
            if commit:
                self._write_db.commit()
        return item.id

    def update_item(self, item: MemoryItem, update_vector: bool = False, commit: bool = True):
        now = time.time()
        with self._write_lock:
            self._write_db.execute(
                """
                UPDATE memories
                SET content = ?, importance = ?, recall_count = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    item.content,
                    float(item.importance),
                    int(item.recall_count),
                    json.dumps(item.metadata.to_dict(), ensure_ascii=False),
                    now,
                    item.id,
                ),
            )
            self._upsert_fts(item.id, item.content)
            if update_vector and item.embedding:
                self._upsert_vector(item.id, item.embedding, item)
            if commit:
                self._write_db.commit()

    def _upsert_fts(self, memory_id: str, content: str):
        self._write_db.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        self._write_db.execute(
            "INSERT INTO memories_fts(id, content) VALUES (?, ?)",
            (memory_id, content),
        )

    def _upsert_vector(self, memory_id: str, embedding: list, item: MemoryItem):
        self.vector_store.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "importance": float(item.importance),
                        "state": item.metadata.state,
                    },
                )
            ],
        )

    def delete(self, memory_id: str, commit: bool = True):
        """
        删除长期记忆（SQLite + FTS + 向量）。

        一致性策略：
        - commit=True：立即提交 DB，再同步删除向量；
        - commit=False：仅记录“待删向量”，由外层事务提交后统一删除。
          这样可避免“DB 回滚但向量已删”的不一致。
        """
        should_delete_vector = False
        with self._write_lock:
            self._write_db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._write_db.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            if commit:
                self._write_db.commit()
                should_delete_vector = True
            else:
                self._pending_vector_deletes.add(memory_id)
        if should_delete_vector:
            self.vector_store.delete(collection_name=self.collection_name, points_selector=[memory_id])

    def _get_by_id_with_conn(self, conn: sqlite3.Connection, memory_id: str) -> Optional[MemoryItem]:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def get_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        with self._read_lock:
            return self._get_by_id_with_conn(self._read_db, memory_id)

    def _get_by_ids_with_conn(self, conn: sqlite3.Connection, memory_ids: Iterable[str]) -> List[MemoryItem]:
        memory_ids = list(memory_ids)
        if not memory_ids:
            return []
        placeholders = ", ".join(["?"] * len(memory_ids))
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_by_ids(self, memory_ids: Iterable[str]) -> List[MemoryItem]:
        with self._read_lock:
            return self._get_by_ids_with_conn(self._read_db, memory_ids)

    def _get_all_with_conn(self, conn: sqlite3.Connection, include_archived: bool = False) -> List[MemoryItem]:
        rows = conn.execute("SELECT * FROM memories ORDER BY created_at DESC").fetchall()
        items = [self._row_to_item(row) for row in rows]
        if include_archived:
            return items
        return [item for item in items if item.metadata.state != "archived"]

    def get_all(self, include_archived: bool = False) -> List[MemoryItem]:
        with self._read_lock:
            return self._get_all_with_conn(self._read_db, include_archived=include_archived)

    def search_candidates(
        self,
        query_text: str,
        query_embedding: List[float],
        limit: int,
        min_importance: float = 0.0,
    ) -> List[dict]:
        """
        混合召回候选集（向量语义 + FTS 关键词）。

        返回结构统一为:
        {
            "item": MemoryItem,
            "vector_score": float,
            "keyword_score": float,
        }
        最终排序权重由 core 统一计算，这里只负责“召回 + 打底分数”。
        """
        vector_scores = self._vector_candidates(query_embedding, limit * 3)
        keyword_scores = self._keyword_candidates(query_text, limit * 3)
        candidate_ids = list(set(vector_scores.keys()) | set(keyword_scores.keys()))
        if not candidate_ids:
            return []

        placeholders = ", ".join(["?"] * len(candidate_ids))
        with self._read_lock:
            rows = self._read_db.execute(
                f"""
                SELECT * FROM memories
                WHERE id IN ({placeholders})
                  AND importance >= ?
                """,
                (*candidate_ids, float(min_importance)),
            ).fetchall()

        items = []
        for row in rows:
            item = self._row_to_item(row)
            if item.metadata.state == "archived":
                continue
            items.append(
                {
                    "item": item,
                    "vector_score": vector_scores.get(item.id, 0.0),
                    "keyword_score": keyword_scores.get(item.id, 0.0),
                }
            )
        return items

    def _vector_candidates(self, query_embedding: List[float], limit: int) -> Dict[str, float]:
        try:
            vector_results = self.vector_store.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
            ).points
        except Exception:
            return {}

        scored: dict[str, float] = {}
        for hit in vector_results:
            mem_id = str(hit.id)
            score = clamp((float(hit.score) + 1.0) / 2.0)
            scored[mem_id] = max(scored.get(mem_id, 0.0), score)
        return scored

    def _keyword_candidates(self, query_text: str, limit: int) -> Dict[str, float]:
        tokens = tokenize(query_text)[:12]
        if not tokens:
            return {}
        match_query = " OR ".join(f"{tok}*" if len(tok) > 2 else tok for tok in tokens)

        try:
            with self._read_lock:
                rows = self._read_db.execute(
                    """
                    SELECT id, bm25(memories_fts) AS rank
                    FROM memories_fts
                    WHERE memories_fts MATCH ?
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return {}

        scored = {}
        for row in rows:
            rank = float(row["rank"]) if row["rank"] is not None else 1000.0
            score = clamp(1.0 / (1.0 + abs(rank)))
            scored[row["id"]] = max(scored.get(row["id"], 0.0), score)
        return scored

    def _keyword_candidate_ids_write(self, text: str, limit: int) -> list[str]:
        tokens = tokenize(text)[:12]
        if not tokens:
            return []
        match_query = " OR ".join(f"{tok}*" if len(tok) > 2 else tok for tok in tokens)

        try:
            rows = self._write_db.execute(
                """
                SELECT id, bm25(memories_fts) AS rank
                FROM memories_fts
                WHERE memories_fts MATCH ?
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        ranked = sorted(
            (
                (
                    row["id"],
                    clamp(1.0 / (1.0 + abs(float(row["rank"]) if row["rank"] is not None else 1000.0))),
                )
                for row in rows
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return [memory_id for memory_id, _ in ranked[:limit]]

    def mark_access(self, memory_ids: Iterable[str], commit: bool = True):
        """
        同步访问计数更新入口。

        在检索请求内批量更新 recall_count，保证统计实时可见。
        """
        counts: dict[str, int] = {}
        for memory_id in memory_ids:
            if not memory_id:
                continue
            counts[memory_id] = counts.get(memory_id, 0) + 1
        self._mark_access_counts(counts, commit=commit)

    def mark_access_counts(self, counts: Dict[str, int], commit: bool = True) -> int:
        """
        批量更新访问计数（id -> increment）。

        该接口用于异步聚合后的增量落库，避免在调用端重复展开 memory_id 列表。
        """
        normalized: dict[str, int] = {}
        for memory_id, increment in counts.items():
            if not memory_id:
                continue
            inc = int(increment)
            if inc <= 0:
                continue
            normalized[memory_id] = normalized.get(memory_id, 0) + inc
        return self._mark_access_counts(normalized, commit=commit)

    def _mark_access_counts(self, counts: Dict[str, int], commit: bool = True) -> int:
        """
        将 id->count 的访问增量批量写入 DB。

        更新字段：
        - recall_count: 累加访问次数；
        - metadata: 回写规范化后的 metadata；
        - updated_at: 标记最近访问时间。
        """
        if not counts:
            return 0

        with self._write_lock:
            memory_ids = list(counts.keys())
            placeholders = ", ".join(["?"] * len(memory_ids))
            rows = self._write_db.execute(
                f"SELECT id, recall_count, metadata FROM memories WHERE id IN ({placeholders})",
                tuple(memory_ids),
            ).fetchall()

            now = time.time()
            updates = []
            touched = 0
            for row in rows:
                increment = int(counts.get(row["id"], 0))
                if increment <= 0:
                    continue

                metadata_dict = {}
                if row["metadata"]:
                    try:
                        metadata_dict = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata_dict = {}

                metadata = MemoryMetadata.from_dict(metadata_dict)
                metadata.store = "long_term"
                updates.append(
                    (
                        int(row["recall_count"] or 0) + increment,
                        json.dumps(metadata.to_dict(), ensure_ascii=False),
                        now,
                        row["id"],
                    )
                )
                touched += increment

            if updates:
                self._write_db.executemany(
                    """
                    UPDATE memories
                    SET recall_count = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    updates,
                )
            if commit:
                self._write_db.commit()

        return touched

    def find_similar(self, query_embedding: List[float], limit: int = 5) -> List[tuple[float, MemoryItem]]:
        scored = self._vector_candidates(query_embedding, limit)
        if not scored:
            return []
        items = self.get_by_ids(scored.keys())
        return sorted(
            [(scored.get(item.id, 0.0), item) for item in items],
            key=lambda x: x[0],
            reverse=True,
        )

    def find_similar_scores(self, query_embedding: List[float], limit: int = 5) -> List[float]:
        """
        仅返回向量近邻分数，用于轻量相似度估计场景（例如 add 阶段重复信号估计）。
        """
        scored = self._vector_candidates(query_embedding, limit)
        if not scored:
            return []
        return sorted((float(v) for v in scored.values()), reverse=True)[: max(int(limit), 1)]

    def consolidate_step(
        self,
        decay_engine: DecayEngine,
        compressor: MemoryCompressor,
        batch_size: int = 64,
        time_budget_ms: int = 250,
        cursor_reset_hours: int = 24,
        low_importance_delete_threshold: float = 0.20,
        high_importance_reinforce_threshold: float = 0.80,
        dedupe_threshold: float = 0.92,
        dedupe_candidate_k: int = 24,
        dedupe_max_pairs: int = 600,
    ) -> dict:
        """
        增量 consolidate 主入口（小步执行，预算驱动）。

        核心机制：
        1. 通过 consolidation_state.cursor_rowid 记录断点，支持下次续跑；
        2. 每轮只处理一个 batch，并受 time_budget_ms 约束；
        3. 单个事务内完成“去重 + 衰减/强化 + 状态推进”；
        4. 事务提交后再批量执行向量删除，保证 DB/向量一致性。
        """
        batch_size = max(int(batch_size), 1)
        time_budget_ms = max(int(time_budget_ms), 1)
        cursor_reset_hours = max(int(cursor_reset_hours), 1)
        dedupe_candidate_k = max(int(dedupe_candidate_k), 1)
        dedupe_max_pairs = max(int(dedupe_max_pairs), 1)
        dedupe_threshold = float(dedupe_threshold)
        low_importance_delete_threshold = float(low_importance_delete_threshold)
        high_importance_reinforce_threshold = float(high_importance_reinforce_threshold)

        start_ts = time.monotonic()
        deadline = start_ts + (time_budget_ms / 1000.0)

        with self._write_lock:
            # 查询本次从哪里开始，即cursor rowid值
            state = self._load_cursor_state_locked(reset_hours=cursor_reset_hours)
            cursor_rowid = state["cursor"]
            cursor_reset = state["cursor_reset"]
            cursor_wrapped = False

            # 基于 cursor rowid 顺序小步扫描，避免全表一次性处理。
            rows = self._write_db.execute(
                """
                SELECT rowid, *
                FROM memories
                WHERE rowid > ?
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (cursor_rowid, batch_size),
            ).fetchall()

            if not rows and cursor_rowid > 0:
                # 本轮已扫描到尾部，回绕到开头继续。
                cursor_wrapped = True
                cursor_rowid = 0
                self._set_state_value_locked("cursor_rowid", "0")
                rows = self._write_db.execute(
                    """
                    SELECT rowid, *
                    FROM memories
                    WHERE rowid > ?
                    ORDER BY rowid ASC
                    LIMIT ?
                    """,
                    (cursor_rowid, batch_size),
                ).fetchall()

            # log term memory为空
            if not rows:
                self._set_state_value_locked("cursor_rowid", "0")
                self._write_db.commit()
                return {
                    "processed": 0,
                    "dedupe_pairs": 0,
                    "dedupe_merged": 0,
                    "compressed": 0,
                    "evicted": 0,
                    "reinforced": 0,
                    "removed_low_importance": 0,
                    "cursor_rowid": 0,
                    "cursor_reset": cursor_reset,
                    "cursor_wrapped": cursor_wrapped,
                    "time_budget_hit": False,
                    "elapsed_ms": int((time.monotonic() - start_ts) * 1000),
                }

            source_rows: list[tuple[int, MemoryItem]] = [
                (int(row["rowid"]), self._row_to_item(row)) for row in rows
            ]
            last_rowid = int(rows[-1]["rowid"])

            try:
                # 步骤1：在当前 batch 上做去重（候选模式或回退全扫描模式）。
                dedupe = self._dedupe_sources_step(
                    source_rows=source_rows,
                    threshold=dedupe_threshold,
                    dedupe_candidate_k=dedupe_candidate_k,
                    dedupe_max_pairs=dedupe_max_pairs,
                    deadline=deadline,
                )
                # 步骤2：对当前 batch 执行衰减、压缩、淘汰和强化。
                decay = self._apply_decay_policy_step(
                    source_rows=source_rows,
                    decay_engine=decay_engine,
                    compressor=compressor,
                    low_importance_delete_threshold=low_importance_delete_threshold,
                    high_importance_reinforce_threshold=high_importance_reinforce_threshold,
                    deadline=deadline,
                )
                # 步骤3：提交 cursor 进度并提交事务。
                self._set_state_value_locked("cursor_rowid", str(last_rowid))
                self._write_db.commit()
                # 步骤4：事务成功后统一删向量，避免回滚不一致。
                self._flush_pending_vector_deletes()
            except Exception:
                self._write_db.rollback()
                # 事务失败时清理待删集合，避免脏状态泄漏到下一轮。
                self._clear_pending_vector_deletes()
                raise

        elapsed_ms = int((time.monotonic() - start_ts) * 1000)
        time_budget_hit = elapsed_ms >= time_budget_ms
        return {
            "processed": len(source_rows),
            "dedupe_pairs": dedupe.get("pairs", 0),
            "dedupe_merged": dedupe.get("merged", 0),
            "compressed": decay.get("compressed", 0),
            "evicted": decay.get("evicted", 0),
            "reinforced": decay.get("reinforced", 0),
            "removed_low_importance": decay.get("removed_low_importance", 0),
            "cursor_rowid": last_rowid,
            "cursor_reset": cursor_reset,
            "cursor_wrapped": cursor_wrapped,
            "time_budget_hit": time_budget_hit,
            "elapsed_ms": elapsed_ms,
        }

    def _dedupe_sources_step(
        self,
        source_rows: list[tuple[int, MemoryItem]],
        threshold: float,
        dedupe_candidate_k: int,
        dedupe_max_pairs: int,
        deadline: float,
    ) -> dict:
        """
        增量去重子步骤：仅处理当前 batch 源记忆。

        采用“候选召回 + 精判”：
        - 候选集A：向量近邻；
        - 候选集B：FTS 关键词候选；
        - 精判：_near_duplicate_score。

        资源保护：
        - 受 deadline（时间预算）与 dedupe_max_pairs（配对预算）双重限制。
        """
        merged = 0
        pairs = 0
        removed_ids: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()

        for _, source_seed in source_rows:
            if time.monotonic() >= deadline:
                break
            if pairs >= dedupe_max_pairs:
                break
            if source_seed.id in removed_ids:
                continue

            source_item = self._get_by_id_with_conn(self._write_db, source_seed.id)
            if source_item is None or source_item.metadata.state == "archived":
                continue

            # 候选集A：向量邻近候选（语义近似）。
            candidate_ids = set(
                self._vector_neighbor_ids(source_item.id, limit=dedupe_candidate_k)
            )
            # 候选集B：关键词候选（词面近似）。
            candidate_ids.update(
                self._keyword_candidate_ids_write(source_item.content, limit=dedupe_candidate_k)
            )
            candidate_ids.discard(source_item.id)
            candidates = self._get_by_ids_with_conn(self._write_db, candidate_ids)

            for candidate in candidates:
                if time.monotonic() >= deadline:
                    break
                if pairs >= dedupe_max_pairs:
                    break
                if candidate.id == source_item.id:
                    continue
                if candidate.id in removed_ids:
                    continue
                if candidate.metadata.state == "archived":
                    continue

                pair_key = tuple(sorted((source_item.id, candidate.id)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pairs += 1

                # 先做轻量近重复评分，未过阈值直接跳过。
                score = self._near_duplicate_score(source_item.content, candidate.content)
                if score < threshold:
                    continue

                # 二次读取最新状态，防止在本轮之前已被其它操作修改/删除。
                left_live = self._get_by_id_with_conn(self._write_db, source_item.id)
                right_live = self._get_by_id_with_conn(self._write_db, candidate.id)
                if left_live is None or right_live is None:
                    continue
                if left_live.metadata.state == "archived" or right_live.metadata.state == "archived":
                    continue
                if self._near_duplicate_score(left_live.content, right_live.content) < threshold:
                    continue

                keeper, loser = self._pick_keeper(left_live, right_live)
                keeper.recall_count += loser.recall_count
                keeper.metadata.repeat_count += 1 + loser.metadata.repeat_count
                keeper.importance = round(
                    clamp(max(keeper.importance, loser.importance) + 0.02),
                    4,
                )
                keeper.metadata.confidence = clamp(
                    max(keeper.metadata.confidence, loser.metadata.confidence)
                )

                self.update_item(keeper, update_vector=False, commit=False)
                self.delete(loser.id, commit=False)
                removed_ids.add(loser.id)
                merged += 1

                if source_item.id == loser.id:
                    break
                if source_item.id == keeper.id:
                    source_item = keeper

        return {"merged": merged, "pairs": pairs}

    def _apply_decay_policy_step(
        self,
        source_rows: list[tuple[int, MemoryItem]],
        decay_engine: DecayEngine,
        compressor: MemoryCompressor,
        low_importance_delete_threshold: float,
        high_importance_reinforce_threshold: float,
        deadline: float,
    ) -> dict:
        """
        增量衰减子步骤：只处理本批 source item 对应的最新记录。

        执行顺序：
        1. 计算最新 half_life；
        2. 先判断 evict；
        3. 再判断 compress；
        4. 应用低价值删除与高价值强化策略；
        5. 回写更新。
        """
        compressed = 0
        evicted = 0
        reinforced = 0
        removed_low_importance = 0

        for _, source_seed in source_rows:
            if time.monotonic() >= deadline:
                break

            item = self._get_by_id_with_conn(self._write_db, source_seed.id)
            if item is None or item.metadata.state == "archived":
                continue

            item.metadata.half_life_days = decay_engine.compute_half_life(item)
            if decay_engine.should_evict(item):
                self.delete(item.id, commit=False)
                evicted += 1
                continue

            if decay_engine.should_compress(item):
                compressor.compress(item)
                compressed += 1

            if (
                item.importance < low_importance_delete_threshold
                and item.recall_count < 2
                and not item.metadata.explicit_remember
            ):
                self.delete(item.id, commit=False)
                removed_low_importance += 1
                continue

            if item.importance >= high_importance_reinforce_threshold:
                item.metadata.half_life_days *= 1.2
                reinforced += 1

            self.update_item(item, update_vector=False, commit=False)

        return {
            "compressed": compressed,
            "evicted": evicted,
            "reinforced": reinforced,
            "removed_low_importance": removed_low_importance,
        }

    def _vector_neighbor_ids(self, memory_id: str, limit: int) -> list[str]:
        """
        获取某条记忆的向量近邻 id。

        查询策略：
        - 优先使用“按 point id 查询近邻”；
        - 若当前后端不支持该能力，则先 retrieve 向量，再用向量查询近邻。
        """
        limit = max(int(limit), 1)
        hits = []
        try:
            hits = self.vector_store.query_points(
                collection_name=self.collection_name,
                query=memory_id,
                limit=limit + 1,
            ).points
        except Exception:
            try:
                points = self.vector_store.retrieve(
                    collection_name=self.collection_name,
                    ids=[memory_id],
                    with_vectors=True,
                )
            except Exception:
                return []
            if not points:
                return []
            vector = points[0].vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()), None)
            if not vector:
                return []
            try:
                hits = self.vector_store.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=limit + 1,
                ).points
            except Exception:
                return []

        neighbor_ids: list[str] = []
        for hit in hits:
            candidate_id = str(hit.id)
            if candidate_id == memory_id:
                continue
            neighbor_ids.append(candidate_id)
            if len(neighbor_ids) >= limit:
                break
        return neighbor_ids

    def _load_cursor_state_locked(self, reset_hours: int) -> dict:
        """
        读取并校验 consolidate cursor 状态。

        若 cursor 长时间未更新（超过 reset_hours），自动重置为 0，
        防止异常中断后长期停留在旧断点。
        """
        row = self._write_db.execute(
            "SELECT value, updated_at FROM consolidation_state WHERE key = ?",
            ("cursor_rowid",),
        ).fetchone()
        if row is None:
            self._set_state_value_locked("cursor_rowid", "0")
            return {"cursor": 0, "cursor_reset": False}

        try:
            cursor = max(int(row["value"]), 0)
        except (TypeError, ValueError):
            cursor = 0

        now = time.time()
        updated_at = float(row["updated_at"] or now)
        # reset_hours默认为24(一天)
        if now - updated_at >= reset_hours * 3600:
            cursor = 0
            self._set_state_value_locked("cursor_rowid", "0")
            return {"cursor": 0, "cursor_reset": True}

        return {"cursor": cursor, "cursor_reset": False}

    def _set_state_value_locked(self, key: str, value: str):
        """写入 consolidation_state（UPSERT）。"""
        now = time.time()

        # ON CONFLICT(key)：如果不存在，插入新行；若key存在，则更新
        # excluded.xxx 表示“本次准备插入的值”
        # 单条 SQL 原子完成“插入或更新”，避免先查再写的竞态。
        self._write_db.execute(
            """
            INSERT INTO consolidation_state(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key)
            DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now),
        )

    def _flush_pending_vector_deletes(self):
        """
        提交后批量执行向量删除。

        只在 DB 事务已经成功提交后调用，保证“先 DB 后向量”的一致性顺序。
        """
        pending_ids: list[str] = []
        with self._write_lock:
            if not self._pending_vector_deletes:
                return
            pending_ids = list(self._pending_vector_deletes)
            self._pending_vector_deletes.clear()

        if pending_ids:
            self.vector_store.delete(
                collection_name=self.collection_name,
                points_selector=pending_ids,
            )

    def _clear_pending_vector_deletes(self):
        """回滚/异常时清空延迟删除集合，避免脏删除泄漏到下一事务。"""
        with self._write_lock:
            self._pending_vector_deletes.clear()

    def dedupe_by_similarity(self, threshold: float = 0.92) -> dict:
        """
        consolidate 子步骤 1：按内容近似度去重。

        规则：
        - 两条记忆近似度 >= threshold 视为重复；
        - 通过 _pick_keeper 保留“重要度/置信度/召回更高”的一条；
        - loser 的 recall_count 合并到 keeper，保证访问历史不丢失。
        """
        with self._write_lock:
            items = self._get_all_with_conn(self._write_db, include_archived=False)
            merged = 0
            removed_ids: set[str] = set()

            try:
                for i in range(len(items)):
                    left = items[i]
                    if left.id in removed_ids:
                        continue
                    for j in range(i + 1, len(items)):
                        right = items[j]
                        if right.id in removed_ids:
                            continue
                        score = self._near_duplicate_score(left.content, right.content)
                        if score < threshold:
                            continue

                        keeper, loser = self._pick_keeper(left, right)
                        keeper.recall_count += loser.recall_count
                        keeper.metadata.repeat_count += 1 + loser.metadata.repeat_count
                        keeper.importance = round(
                            clamp(max(keeper.importance, loser.importance) + 0.02),
                            4,
                        )
                        keeper.metadata.confidence = clamp(
                            max(keeper.metadata.confidence, loser.metadata.confidence)
                        )

                        self.update_item(keeper, update_vector=False, commit=False)
                        self.delete(loser.id, commit=False)
                        removed_ids.add(loser.id)
                        merged += 1

                        if left.id == loser.id:
                            break
                self._write_db.commit()
                self._flush_pending_vector_deletes()
            except Exception:
                self._write_db.rollback()
                self._clear_pending_vector_deletes()
                raise

        return {"merged": merged}

    def apply_decay(self, decay_engine: DecayEngine, compressor: MemoryCompressor) -> dict:
        """
        consolidate 子步骤 2：按遗忘曲线执行压缩/淘汰。

        流程：
        1. 先重新计算半衰期（融合重要度、置信度、召回次数等信号）；
        2. should_evict 命中则直接删除；
        3. 否则若 should_compress 命中则压缩内容；
        4. 最后回写更新后的 item。
        """
        with self._write_lock:
            items = self._get_all_with_conn(self._write_db, include_archived=False)
            compressed = 0
            evicted = 0
            try:
                for item in items:
                    item.metadata.half_life_days = decay_engine.compute_half_life(item)
                    if decay_engine.should_evict(item):
                        self.delete(item.id, commit=False)
                        evicted += 1
                        continue
                    if decay_engine.should_compress(item):
                        compressor.compress(item)
                        compressed += 1
                    self.update_item(item, update_vector=False, commit=False)
                self._write_db.commit()
                self._flush_pending_vector_deletes()
            except Exception:
                self._write_db.rollback()
                self._clear_pending_vector_deletes()
                raise
        return {"compressed": compressed, "evicted": evicted}

    def _pick_keeper(self, a: MemoryItem, b: MemoryItem) -> tuple[MemoryItem, MemoryItem]:
        score_a = (
            clamp(a.importance),
            clamp(a.metadata.confidence),
            int(a.recall_count),
        )
        score_b = (
            clamp(b.importance),
            clamp(b.metadata.confidence),
            int(b.recall_count),
        )
        if score_a >= score_b:
            return a, b
        return b, a

    def _near_duplicate_score(self, left: str, right: str) -> float:
        left_ngrams = self._char_ngrams(left)
        right_ngrams = self._char_ngrams(right)
        if not left_ngrams or not right_ngrams:
            return 0.0
        return len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)

    def _char_ngrams(self, text: str, n: int = 2) -> set[str]:
        """
        快速、无模型依赖的“文本近重复检测, 用 Jaccard 计算
        """
        normalized = normalize_text(text).replace(" ", "")
        if not normalized:
            return set()
        if len(normalized) <= n:
            return {normalized}
        return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        metadata_dict: dict = {}
        if row["metadata"]:
            try:
                metadata_dict = json.loads(row["metadata"])
            except json.JSONDecodeError:
                metadata_dict = {}

        metadata = MemoryMetadata.from_dict(metadata_dict)
        metadata.store = "long_term"
        metadata.created_at = float(row["created_at"] if row["created_at"] is not None else metadata.created_at)

        content = str(row["content"])
        if content == "[empty]":
            content = ""

        recall_count = int(row["recall_count"] or 0)

        return MemoryItem(
            id=row["id"],
            content=content,
            importance=float(row["importance"] if row["importance"] is not None else 0.5),
            embedding=None,
            recall_count=recall_count,
            metadata=metadata,
        )

    def close(self):
        try:
            self._flush_pending_vector_deletes()
        except Exception:
            pass

        with self._read_lock:
            self._read_db.close()
        with self._write_lock:
            self._write_db.close()
        self.vector_store.close()
