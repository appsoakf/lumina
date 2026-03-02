from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class MemoryType(str, Enum):
    PROFILE = "profile"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    COMMITMENT = "commitment"
    ARTIFACT = "artifact"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    memory_id: Optional[int]
    user_id: str
    session_id: str
    memory_type: MemoryType
    content: str
    content_hash: str = ""
    tags: str = ""
    confidence: float = 1.0
    ttl_seconds: Optional[int] = None
    source: str = ""
    payload: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "confidence": self.confidence,
            "ttl_seconds": self.ttl_seconds,
            "source": self.source,
            "payload": self.payload or {},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
