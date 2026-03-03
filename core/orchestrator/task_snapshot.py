import json
from typing import Any, Dict, List


def completed_context(task_snapshot: Dict[str, Any]) -> str:
    lines: List[str] = []
    for node in task_snapshot.get("nodes") or []:
        state = str(node.get("state") or "")
        if state not in {"succeeded", "failed", "cancelled", "blocked"}:
            continue
        status = {
            "succeeded": "成功",
            "failed": "失败",
            "cancelled": "取消",
            "blocked": "阻塞",
        }.get(state, state)
        lines.append(f"[{node.get('step_id')}:{status}] {node.get('title')}: {node.get('output_text')}")
    return "\n".join(lines).strip()


def resolve_step_inputs(task_snapshot: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    nodes = task_snapshot.get("nodes") or []
    index = {str(node.get("step_id")): node for node in nodes}
    node = index.get(step_id)
    if node is None:
        raise ValueError(f"Unknown step_id: {step_id}")

    resolved: Dict[str, Any] = {}
    for binding in node.get("input_bindings") or []:
        if not isinstance(binding, dict):
            continue
        source = str(binding.get("from") or "").strip()
        target = str(binding.get("to") or "").strip()
        if not source or not target:
            continue

        if source.startswith("$const:"):
            raw = source[len("$const:") :]
            try:
                resolved[target] = json.loads(raw)
            except Exception:
                resolved[target] = raw
            continue

        source_step, sep, field = source.partition(".")
        if not sep:
            field = "output_text"
        upstream = index.get(source_step)
        if upstream is None:
            continue
        resolved[target] = upstream.get(field)
    return resolved


def step_result_from_node(node: Dict[str, Any]) -> Dict[str, Any]:
    state = str(node.get("state") or "")
    return {
        "step_id": str(node.get("step_id") or ""),
        "title": str(node.get("title") or ""),
        "depends_on": list(node.get("depends_on") or []),
        "input_bindings": list(node.get("input_bindings") or []),
        "state": state,
        "output_text": str(node.get("output_text") or ""),
        "error": node.get("error"),
    }
