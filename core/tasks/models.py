from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.protocols import TaskState


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    user_text: str
    state: TaskState = TaskState.PENDING
    plan: Optional[Dict[str, Any]] = None
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "user_text": self.user_text,
            "state": self.state.value,
            "plan": self.plan,
            "step_results": self.step_results,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRecord":
        return cls(
            task_id=data["task_id"],
            session_id=data.get("session_id", ""),
            user_text=data.get("user_text", ""),
            state=TaskState(data.get("state", TaskState.PENDING.value)),
            plan=data.get("plan"),
            step_results=data.get("step_results", []),
            error=data.get("error"),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )
