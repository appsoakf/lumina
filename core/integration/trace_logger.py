import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional


class TraceLogger:
    """Async JSONL trace logger backed by queue + writer thread."""

    _SENTINEL = object()

    def __init__(
        self,
        trace_dir: str = "D:/lumina/runtime/traces",
        session_id: str = "default",
        max_queue_size: int = 1024,
    ):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"trace-{session_id}.jsonl"

        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._closed = False
        self._lock = threading.Lock()

        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"TraceLogger-{session_id}",
            daemon=True,
        )
        self._writer.start()

    def _writer_loop(self) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            while True:
                item = self._queue.get()
                try:
                    if item is self._SENTINEL:
                        return
                    f.write(item)
                    f.flush()
                finally:
                    self._queue.task_done()

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self._queue.put(line)

    def flush(self, timeout: Optional[float] = None) -> bool:
        done = threading.Event()

        def _wait_join() -> None:
            self._queue.join()
            done.set()

        with self._lock:
            if self._closed:
                return True

        waiter = threading.Thread(target=_wait_join, daemon=True)
        waiter.start()
        return done.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._queue.put(self._SENTINEL)
        self._queue.join()
        self._writer.join(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log("trace_logger_error", {"type": str(exc_type), "message": str(exc_val)})
        self.close()
