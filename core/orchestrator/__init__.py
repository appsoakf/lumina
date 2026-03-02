from .task_graph import TaskGraph, TaskNode

__all__ = ["Orchestrator", "TaskGraph", "TaskNode"]


def __getattr__(name: str):
    if name == "Orchestrator":
        from .orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
