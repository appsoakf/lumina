import json
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolContext:
    session_id: str = "default"


@dataclass
class ToolResult:
    ok: bool
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_model_text(self) -> str:
        payload: Dict[str, Any] = {"ok": self.ok, "content": self.content}
        if self.meta:
            payload["meta"] = dict(self.meta)
        return json.dumps(payload, ensure_ascii=False)

