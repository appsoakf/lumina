from core.memory.models import MemoryType


class MemoryPolicy:
    """Minimal memory write policy to avoid noisy storage."""

    def should_store_profile(self, text: str) -> bool:
        triggers = ["我喜欢", "我不喜欢", "我的偏好", "我习惯", "记住"]
        return any(t in text for t in triggers)

    def should_store_commitment(self, text: str) -> bool:
        triggers = ["待办", "提醒", "截止", "明天", "下周", "任务"]
        return any(t in text for t in triggers)

    def should_store_episode(self, text: str) -> bool:
        return len(text.strip()) >= 6

    def default_ttl_seconds(self, memory_type: MemoryType):
        if memory_type == MemoryType.EPISODIC:
            return 7 * 24 * 3600
        if memory_type == MemoryType.COMMITMENT:
            return 30 * 24 * 3600
        return None
