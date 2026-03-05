import logging
import threading
import time
from queue import Full, Queue
from typing import Dict, Optional

from core.memory.embedding import EmbeddingProvider
from core.memory.vector_store import QdrantVectorStore
from core.utils import log_event

logger = logging.getLogger(__name__)


class MemoryVectorIndexer:
    """Async index writer to keep vector IO off the main conversation path."""

    _SENTINEL = object()

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        enabled: bool,
        queue_size: int = 512,
        max_retries: int = 3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.max_retries = max(max_retries, 0)
        self._closed = False
        self._lock = threading.Lock()
        self._queue: Queue = Queue(maxsize=max(queue_size, 32))
        self._enabled = bool(enabled and embedder and vector_store and embedder.is_ready() and vector_store.is_ready())

        self._worker: Optional[threading.Thread] = None
        if self._enabled:
            self._worker = threading.Thread(target=self._run, name="MemoryVectorIndexer", daemon=True)
            self._worker.start()

    def is_enabled(self) -> bool:
        return self._enabled

    def enqueue(self, memory_id: int, content: str, payload: Dict) -> None:
        if not self._enabled:
            return
        item = {
            "memory_id": int(memory_id),
            "content": content or "",
            "payload": payload or {},
            "attempt": 0,
        }
        try:
            self._queue.put_nowait(item)
        except Full:
            log_event(
                logger,
                logging.WARNING,
                "memory.vector.queue.drop",
                "向量索引队列已满，丢弃写入请求",
                component="memory",
                memory_id=int(memory_id),
                queue_size=self._queue.qsize(),
            )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                self._handle_item(item)
            finally:
                self._queue.task_done()

    def _handle_item(self, item: Dict) -> None:
        memory_id = int(item.get("memory_id", 0))
        content = str(item.get("content", ""))
        payload = item.get("payload", {}) or {}
        attempt = int(item.get("attempt", 0))

        vector = self.embedder.embed(content)
        if not vector:
            return

        ok = self.vector_store.upsert(memory_id=memory_id, vector=vector, payload=payload)
        if ok:
            return

        if attempt >= self.max_retries:
            return

        backoff = 0.2 * (2 ** attempt)
        time.sleep(backoff)
        item["attempt"] = attempt + 1
        try:
            self._queue.put_nowait(item)
        except Full:
            log_event(
                logger,
                logging.WARNING,
                "memory.vector.queue.retry_drop",
                "向量索引重试队列已满，丢弃重试请求",
                component="memory",
                memory_id=memory_id,
                attempt=int(item.get("attempt", 0)),
                queue_size=self._queue.qsize(),
            )

    def flush(self, timeout: Optional[float] = None) -> bool:
        if not self._enabled:
            return True

        done = threading.Event()

        def _wait_join() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=_wait_join, daemon=True).start()
        return done.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if not self._enabled:
            return

        self._queue.put(self._SENTINEL)
        self._queue.join()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
