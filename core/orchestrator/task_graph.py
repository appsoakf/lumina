from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.protocols import PlanResult, TaskState


@dataclass
class TaskNode:
    step_id: str
    title: str
    instruction: str
    state: TaskState = TaskState.PENDING
    output_text: str = ""
    tool_events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "instruction": self.instruction,
            "state": self.state.value,
            "output_text": self.output_text,
            "tool_events": self.tool_events,
            "error": self.error,
        }


class TaskGraph:
    """DAG-lite task graph: Phase 2 uses sequential dependencies by default."""

    def __init__(self, goal: str, nodes: List[TaskNode]):
        self.goal = goal
        self.nodes = nodes

    @classmethod
    def from_plan(cls, plan: PlanResult) -> "TaskGraph":
        nodes = [
            TaskNode(step_id=s.step_id, title=s.title, instruction=s.instruction)
            for s in plan.steps
        ]
        return cls(goal=plan.goal, nodes=nodes)

    def pending_nodes(self) -> List[TaskNode]:
        return [n for n in self.nodes if n.state == TaskState.PENDING]

    def mark_running(self, step_id: str) -> None:
        node = self._get_node(step_id)
        node.state = TaskState.RUNNING

    def mark_done(self, step_id: str, output_text: str, tool_events: List[Dict[str, Any]]) -> None:
        node = self._get_node(step_id)
        node.state = TaskState.SUCCEEDED
        node.output_text = output_text
        node.tool_events = tool_events
        node.error = None

    def mark_failed(self, step_id: str, output_text: str, tool_events: List[Dict[str, Any]], error: Dict[str, Any]) -> None:
        node = self._get_node(step_id)
        node.state = TaskState.FAILED
        node.output_text = output_text
        node.tool_events = tool_events
        node.error = error

    def completed_context(self) -> str:
        lines = []
        for n in self.nodes:
            if n.state in {TaskState.SUCCEEDED, TaskState.FAILED}:
                status = "成功" if n.state == TaskState.SUCCEEDED else "失败"
                lines.append(f"[{n.step_id}:{status}] {n.title}: {n.output_text}")
        return "\n".join(lines).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    def _get_node(self, step_id: str) -> TaskNode:
        for n in self.nodes:
            if n.step_id == step_id:
                return n
        raise ValueError(f"Unknown step_id: {step_id}")
