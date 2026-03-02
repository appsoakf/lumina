import threading
from dataclasses import dataclass, field
from queue import Queue

@dataclass
class AudioChunk:
    audio_bytes: bytes  # 裸 PCM 数据


@dataclass
class SentenceSlot:
    index: int
    chinese_text: str
    japanese_text: str | None = None
    chunk_queue: Queue = field(default_factory=Queue)
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None


class EmotionContext:
    def __init__(self):
        self.event = threading.Event()
        self.ref_audio_path: str | None = None
        self.prompt_text: str | None = None
        self.emotion: str | None = None
        self.intensity: str | None = None


class OrderedSentenceMap:
    def __init__(self):
        self._slots: list[SentenceSlot] = []
        self._lock = threading.Lock()
        self._new_slot_event = threading.Event()
        self._all_registered = False

    def register(self, index: int, text: str) -> SentenceSlot:
        slot = SentenceSlot(index=index, chinese_text=text)
        with self._lock:
            self._slots.append(slot)
            self._new_slot_event.set()
        return slot

    def mark_all_registered(self):
        with self._lock:
            self._all_registered = True
            self._new_slot_event.set()

    def iter_slots_in_order(self):
        """按注册顺序 yield slot，阻塞等待新 slot 或完成标记。"""
        idx = 0
        while True:
            with self._lock:
                if idx < len(self._slots):
                    slot = self._slots[idx]
                    idx += 1
                    yield slot
                    continue
                if self._all_registered:
                    return
            self._new_slot_event.clear()
            self._new_slot_event.wait(timeout=1.0)
