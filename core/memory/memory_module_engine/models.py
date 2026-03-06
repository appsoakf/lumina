"""Memory and user-model data structures."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class MemoryMetadata:
    """Metadata bucket for factors affecting retention and importance."""

    # Lifecycle fields moved from MemoryItem.
    created_at: float = field(default_factory=time.time)
    store: str = "working"  # working | long_term
    state: str = "active"  # active | compressed | archived | uncertain
    half_life_days: float = 30.0

    # Importance-related signals.
    explicit_remember: bool = False
    future_use: bool = False
    emotion_intensity: float = 0.0
    near_repeat_score: float = 0.0
    repeat_count: int = 0
    temporal_urgency: float = 0.0
    information_density: float = 0.5
    llm_importance_hint: float = 0.5
    confidence: float = 0.5
    user_rating: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "MemoryMetadata":
        if not isinstance(data, dict):
            return cls()
        created_at = float(data.get("created_at", time.time()))
        return cls(
            created_at=created_at,
            store=str(data.get("store", "working")),
            state=str(data.get("state", "active")),
            half_life_days=float(data.get("half_life_days", 30.0)),
            explicit_remember=bool(data.get("explicit_remember", False)),
            future_use=bool(data.get("future_use", False)),
            emotion_intensity=float(data.get("emotion_intensity", 0.0)),
            near_repeat_score=float(data.get("near_repeat_score", 0.0)),
            repeat_count=int(data.get("repeat_count", 0)),
            temporal_urgency=float(data.get("temporal_urgency", 0.0)),
            information_density=float(data.get("information_density", 0.5)),
            llm_importance_hint=float(data.get("llm_importance_hint", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            user_rating=data.get("user_rating"),
        )


@dataclass
class MemoryItem:
    """Minimal unified memory record."""

    id: str
    content: str
    importance: float
    embedding: Optional[list] = None
    recall_count: int = 0
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
