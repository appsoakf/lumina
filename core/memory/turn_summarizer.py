import logging
import re
import threading
from dataclasses import dataclass, field
from queue import Full, Queue
from typing import Callable, Dict, List, Optional

from core.memory.ingestor import MemoryIngestor
from core.utils import log_event, log_exception

logger = logging.getLogger(__name__)


@dataclass
class TurnSummary:
    topic: str = ""
    profile_candidates: List[str] = field(default_factory=list)


class TurnSummaryExtractor:
    """Extract topic + profile hints from one dialog turn."""

    TOPIC_SPLIT_RE = re.compile(r"[。！？!?;\n]")
    PROFILE_HINT_RE = re.compile(r"(?:我喜欢|我不喜欢|我的偏好(?:是)?|我偏好|我习惯)([^。！？\n]{1,40})")
    LIST_SPLIT_RE = re.compile(r"[、,，]|和|以及")

    def __init__(self, ingestor: Optional[MemoryIngestor] = None):
        self.ingestor = ingestor or MemoryIngestor()

    def summarize(self, user_text: str, assistant_reply: str) -> TurnSummary:
        topic = self._extract_topic(user_text=user_text, assistant_reply=assistant_reply)
        profiles = self._extract_profile_candidates(user_text=user_text)
        return TurnSummary(topic=topic, profile_candidates=profiles)

    def _extract_topic(self, user_text: str, assistant_reply: str) -> str:
        first = self._first_clause(user_text)
        first = self._strip_prefix(first)
        if not first:
            first = self._first_clause(assistant_reply)
        topic = first.strip()
        if len(topic) > 48:
            topic = topic[:48].rstrip() + "..."
        return topic

    def _extract_profile_candidates(self, user_text: str) -> List[str]:
        candidates: List[str] = []

        for item in self.ingestor.extract_profile_candidates(user_text):
            content = str(item.get("content", "")).strip()
            candidates.extend(self._split_preferences(content))

        for match in self.PROFILE_HINT_RE.finditer(user_text or ""):
            candidates.extend(self._split_preferences(match.group(1)))

        deduped: List[str] = []
        seen = set()
        for value in candidates:
            norm = self._normalize(value)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(value.strip())
            if len(deduped) >= 6:
                break
        return deduped

    def _first_clause(self, text: str) -> str:
        source = (text or "").strip()
        if not source:
            return ""
        parts = [x.strip() for x in self.TOPIC_SPLIT_RE.split(source) if x.strip()]
        return parts[0] if parts else source

    def _strip_prefix(self, text: str) -> str:
        out = (text or "").strip()
        prefixes = [
            "请帮我",
            "帮我",
            "请",
            "我想让你",
            "我想",
            "我希望",
            "我需要",
            "能不能",
            "可以",
        ]
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix) :].strip()
                break
        return out

    def _split_preferences(self, text: str) -> List[str]:
        source = (text or "").strip(" ，,。！？!?\n\t")
        if not source:
            return []
        parts = [p.strip(" ，,。！？!?\n\t") for p in self.LIST_SPLIT_RE.split(source) if p.strip()]
        return parts or [source]

    def _normalize(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())


class AsyncTurnSummarizer:
    """Async turn summarizer to keep memory extraction off the hot path."""

    _SENTINEL = object()

    def __init__(
        self,
        extractor: Optional[TurnSummaryExtractor],
        on_summary: Callable[[TurnSummary, Dict[str, object]], None],
        enabled: bool = True,
        queue_size: int = 256,
    ):
        self.extractor = extractor or TurnSummaryExtractor()
        self.on_summary = on_summary
        self._enabled = bool(enabled and on_summary is not None)
        self._queue: Queue = Queue(maxsize=max(int(queue_size), 32))
        self._closed = False
        self._lock = threading.Lock()

        self._worker: Optional[threading.Thread] = None
        if self._enabled:
            self._worker = threading.Thread(target=self._run, name="AsyncTurnSummarizer", daemon=True)
            self._worker.start()

    def is_enabled(self) -> bool:
        return self._enabled

    def enqueue(self, item: Dict[str, object]) -> bool:
        if not self._enabled:
            return False
        try:
            self._queue.put_nowait(dict(item))
            return True
        except Full:
            log_event(
                logger,
                logging.WARNING,
                "memory.turn_summary.queue_full",
                "Turn summary 队列已满，回退同步提取",
                component="memory",
                queue_size=self._queue.qsize(),
            )
            return False

    def summarize_now(self, item: Dict[str, object]) -> TurnSummary:
        return self.extractor.summarize(
            user_text=str(item.get("user_text", "")),
            assistant_reply=str(item.get("assistant_reply", "")),
        )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                summary = self.summarize_now(item)
                self.on_summary(summary, item)
            except Exception:
                log_exception(
                    logger,
                    "memory.turn_summary.worker.error",
                    "Turn summary 异步处理失败，跳过当前条目",
                    component="memory",
                )
            finally:
                self._queue.task_done()

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
