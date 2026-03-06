"""工具函数和轻量策略组件。"""
from __future__ import annotations

import math
import re
import time
import numpy as np

from .models import MemoryItem, MemoryMetadata


def cosine_similarity(vec1: list, vec2: list) -> float:
    """计算余弦相似度"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0:
        return 0.0
    return float(np.dot(v1, v2) / denominator)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_text(text: str) -> str:
    """
    不会删除标点，也不会做分词
    去首尾空白，并把连续空白（空格/换行/Tab）压成一个空格
    TODO: 考虑用LLM作摘要
    """
    text = (text or "").lower().strip()
    return re.sub(r"\s+", " ", text)


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [tok for tok in re.split(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", normalized) if tok]


def recency_score(timestamp: float, half_life_days: float = 30.0) -> float:
    """
    计算时间近期评分，越近值越高；越旧的记忆值越低
    """
    half_life_days = max(half_life_days, 0.1)
    age_days = max((time.time() - timestamp) / 86400, 0.0)
    return math.exp(-math.log(2) * age_days / half_life_days)


class ImportanceScorer:
    """计算记忆重要性。"""

    def calculate(self, content: str, metadata: MemoryMetadata = None) -> float:
        """
        计算重要性 (0-1)
        融合 LLM 信号 + 近似重复信号 + 文本长度
        """
        metadata = metadata or MemoryMetadata()
        length_score = min(len(content) / 500, 1.0)
        explicit = 1.0 if metadata.explicit_remember else 0.0
        future_use = 1.0 if metadata.future_use else 0.0
        emotion = clamp(metadata.emotion_intensity)
        repeat = clamp(metadata.near_repeat_score)
        repeat_count = min(max(metadata.repeat_count, 0), 10) / 10.0
        urgency = clamp(metadata.temporal_urgency)
        density = clamp(metadata.information_density)
        llm_hint = clamp(metadata.llm_importance_hint)

        score = 0.10
        score += 0.10 * length_score
        score += 0.20 * explicit
        score += 0.15 * future_use
        score += 0.10 * emotion
        score += 0.10 * repeat
        score += 0.05 * repeat_count
        score += 0.10 * urgency
        score += 0.10 * density
        score += 0.15 * llm_hint

        if metadata.user_rating is not None:
            score = score * 0.7 + clamp(metadata.user_rating) * 0.3

        return round(clamp(score), 3)


class WriteGate:
    """Simple persistence gate driven by final importance score."""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def evaluate(
        self,
        metadata: MemoryMetadata | None = None,
        importance: float = 0.5,
    ) -> bool:
        metadata = metadata or MemoryMetadata()
        imp = clamp(importance)
        explicit = bool(metadata.explicit_remember)
        return bool(explicit or imp >= self.threshold)


class MemoryCompressor:
    """Generate compact summaries for low-value records."""

    def __init__(self, max_chars: int = 180):
        self.max_chars = max_chars

    def compress(self, item: MemoryItem) -> MemoryItem:
        compact = normalize_text(item.content)
        if len(compact) > self.max_chars:
            compact = compact[: self.max_chars].rstrip() + "..."

        item.content = compact
        item.metadata.state = "compressed"
        return item


class DecayEngine:
    """
    计算记忆的半衰期。

    两个时机：
    1. working memory 固化为长期记忆时；
    2. long-term consolidate 时重算半衰期，用于压缩或淘汰判断。
    """

    def __init__(
        self,
        compression_threshold: float = 0.18,
        eviction_threshold: float = 0.10,
        compressed_retention_days: int = 45,
        base_half_life_days: float = 30.0,
    ):
        self.compression_threshold = compression_threshold
        self.eviction_threshold = eviction_threshold
        self.compressed_retention_days = compressed_retention_days
        self.base_half_life_days = base_half_life_days

    def compute_half_life(self, item: MemoryItem) -> float:
        """
        重要度越高、置信度越高、被召回越多、重复越多，半衰期越长，遗忘越慢。
        """
        return max(
            self.base_half_life_days
            * (
                1.0
                + 1.5 * clamp(item.importance)
                + 0.8 * clamp(item.metadata.confidence)
                + 0.2 * math.log1p(max(item.recall_count, 0))
                + 0.2 * min(max(item.metadata.repeat_count, 0), 10) / 10.0
            ),
            3.0,
        )

    def decayed_value(self, item: MemoryItem, now: float | None = None) -> float:
        now = now or time.time()
        half_life = max(item.metadata.half_life_days or self.compute_half_life(item), 0.1)
        age_days = max((now - item.metadata.created_at) / 86400, 0.0)
        decay = math.exp(-math.log(2) * age_days / half_life)
        return clamp(clamp(item.importance) * decay * (0.7 + 0.3 * clamp(item.metadata.confidence)))

    def should_compress(self, item: MemoryItem, now: float | None = None) -> bool:
        if item.metadata.state != "active":
            return False
        return self.decayed_value(item, now) < self.compression_threshold

    def should_evict(self, item: MemoryItem, now: float | None = None) -> bool:
        now = now or time.time()
        if item.metadata.state == "archived":
            return True
        if item.metadata.state != "compressed":
            return self.decayed_value(item, now) < self.eviction_threshold
        age_days = max((now - item.metadata.created_at) / 86400, 0.0)
        return (
            age_days > self.compressed_retention_days
            and self.decayed_value(item, now) < (self.eviction_threshold + 0.03)
        )
