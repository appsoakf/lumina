import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from typing import Any, Dict, Optional, Union

from core.paths import runtime_traces_dir
from core.utils.logging_helpers import log_event

logger = logging.getLogger(__name__)


class TraceLogger:
    """Async JSONL trace logger backed by queue + writer thread."""

    _SENTINEL = object()

    def __init__(
        self,
        trace_dir: Optional[Union[str, Path]] = None,
        session_id: str = "default",
        max_queue_size: int = 1024,
        flush_every: int = 20,
        drop_on_overflow: bool = True,
    ):
        self.trace_dir = Path(trace_dir) if trace_dir is not None else runtime_traces_dir()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"trace-{session_id}.jsonl"

        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._flush_every = max(int(flush_every), 1)
        self._drop_on_overflow = bool(drop_on_overflow)
        self._dropped_count = 0
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
            pending = 0
            while True:
                item = self._queue.get()
                try:
                    if item is self._SENTINEL:
                        if pending > 0:
                            f.flush()
                        return
                    f.write(item)
                    pending += 1
                    if pending >= self._flush_every:
                        f.flush()
                        pending = 0
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
        if self._drop_on_overflow:
            try:
                self._queue.put_nowait(line)
            except Full:
                with self._lock:
                    self._dropped_count += 1
        else:
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
            dropped_count = self._dropped_count

        if dropped_count > 0:
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "trace_logger_drop",
                "payload": {
                    "dropped": dropped_count,
                    "reason": "queue_full",
                },
            }
            line = json.dumps(row, ensure_ascii=False) + "\n"
            try:
                self._queue.put_nowait(line)
            except Full:
                log_event(
                    logger,
                    logging.WARNING,
                    "trace.logger.drop_summary_lost",
                    "Trace logger 丢弃摘要写入失败",
                    component="trace",
                    reason="queue_full",
                )

        self._queue.put(self._SENTINEL)
        self._queue.join()
        self._writer.join(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log("trace_logger_error", {"type": str(exc_type), "message": str(exc_val)})
        self.close()
