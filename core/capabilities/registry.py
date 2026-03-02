from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CapabilitySpec:
    agent_name: str
    capabilities: List[str] = field(default_factory=list)
    priority: int = 100


class CapabilityRegistry:
    """Minimal dynamic routing registry for agents."""

    def __init__(self):
        self._specs: Dict[str, CapabilitySpec] = {}

    def register(self, spec: CapabilitySpec) -> None:
        self._specs[spec.agent_name] = spec

    def resolve_agent(self, capability: str) -> str:
        matched = [s for s in self._specs.values() if capability in s.capabilities]
        if not matched:
            raise ValueError(f"No agent capability registered for: {capability}")
        matched.sort(key=lambda s: s.priority)
        return matched[0].agent_name

    def list_specs(self) -> List[CapabilitySpec]:
        return list(self._specs.values())


def build_default_registry() -> CapabilityRegistry:
    r = CapabilityRegistry()
    r.register(CapabilitySpec(agent_name="chat_agent", capabilities=["chat", "response_compose"], priority=10))
    r.register(CapabilitySpec(agent_name="planner_agent", capabilities=["task_planning", "travel_planning"], priority=20))
    r.register(CapabilitySpec(agent_name="executor_agent", capabilities=["task_execution", "tool_execution"], priority=30))
    r.register(CapabilitySpec(agent_name="critic_agent", capabilities=["task_review"], priority=40))
    return r
