from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RoutingIntent(str, Enum):
    CHAT = "chat"
    TASK = "task"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PlanItem:
    step_id: str
    title: str
    instruction: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "instruction": self.instruction,
        }


@dataclass
class PlanResult:
    goal: str
    steps: List[PlanItem] = field(default_factory=list)
    raw_text: str = ""
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "raw_text": self.raw_text,
            "error": self.error,
        }


@dataclass
class CriticResult:
    quality: str = "pass"
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    summary: str = ""
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality": self.quality,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class ExecutorRunResult:
    output_text: str
    tool_events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    step_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_text": self.output_text,
            "tool_events": self.tool_events,
            "error": self.error,
            "step_results": self.step_results,
        }


@dataclass
class OrchestrationResult:
    intent: RoutingIntent
    final_reply: str
    executor_result: Optional[ExecutorRunResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "final_reply": self.final_reply,
            "executor_result": self.executor_result.to_dict() if self.executor_result else None,
            "meta": self.meta,
        }
