import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict


@dataclass
class ToolContext:
    user_id: str = "anonymous"
    session_id: str = "default"


@dataclass
class ToolResult:
    ok: bool
    content: str

    def to_model_text(self) -> str:
        payload = {"ok": self.ok, "content": self.content}
        return json.dumps(payload, ensure_ascii=False)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[..., ToolResult]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, schema: Dict[str, Any], fn: Callable[..., ToolResult]) -> None:
        self._tools[name] = fn
        self._schemas[name] = schema

    def list_schemas(self) -> list[Dict[str, Any]]:
        return [self._schemas[name] for name in self._tools.keys()]

    def call(self, name: str, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        fn = self._tools.get(name)
        if fn is None:
            return ToolResult(ok=False, content=f"Unknown tool: {name}")
        try:
            return fn(ctx=ctx, **args)
        except TypeError as exc:
            return ToolResult(ok=False, content=f"Invalid args for {name}: {exc}")
        except Exception as exc:
            return ToolResult(ok=False, content=f"Tool {name} failed: {exc}")


NOTES_DIR = Path("D:/lumina/runtime/notes")
NOTES_DIR.mkdir(parents=True, exist_ok=True)


def _tool_get_current_time(*, ctx: ToolContext, timezone: str = "UTC") -> ToolResult:
    now = datetime.utcnow().isoformat() + "Z"
    return ToolResult(ok=True, content=f"UTC time={now}, requested_timezone={timezone}")


def _tool_write_note(*, ctx: ToolContext, filename: str, content: str) -> ToolResult:
    safe_name = Path(filename).name
    if not safe_name.endswith(".md"):
        safe_name = f"{safe_name}.md"
    path = NOTES_DIR / safe_name
    entry = f"\n## {datetime.utcnow().isoformat()}Z | user={ctx.user_id} session={ctx.session_id}\n{content}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
    return ToolResult(ok=True, content=f"note_written:{path}")


def _tool_list_notes(*, ctx: ToolContext) -> ToolResult:
    files = sorted([p.name for p in NOTES_DIR.glob("*.md")])
    return ToolResult(ok=True, content=json.dumps({"notes": files}, ensure_ascii=False))


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        name="get_current_time",
        schema={
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current time for user-facing responses or scheduling tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Requested timezone name, e.g. UTC or Asia/Shanghai.",
                        }
                    },
                    "required": [],
                },
            },
        },
        fn=_tool_get_current_time,
    )

    registry.register(
        name="write_note",
        schema={
            "type": "function",
            "function": {
                "name": "write_note",
                "description": "Write user-requested notes into markdown files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        fn=_tool_write_note,
    )

    registry.register(
        name="list_notes",
        schema={
            "type": "function",
            "function": {
                "name": "list_notes",
                "description": "List note files created by write_note tool.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        fn=_tool_list_notes,
    )

    return registry
