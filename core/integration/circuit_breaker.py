import time
from collections import defaultdict


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = defaultdict(int)
        self.open_since = {}

    def is_open(self, key: str) -> bool:
        opened = self.open_since.get(key)
        if not opened:
            return False
        if time.time() - opened > self.recovery_timeout:
            self.close(key)
            return False
        return True

    def record(self, key: str, ok: bool) -> None:
        if ok:
            self.close(key)
            return
        self.failures[key] += 1
        if self.failures[key] >= self.failure_threshold:
            self.open_since[key] = time.time()

    def close(self, key: str) -> None:
        self.failures[key] = 0
        self.open_since.pop(key, None)
