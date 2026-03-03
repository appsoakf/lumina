import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import runtime_notes_dir
from core.tools.base import BaseTool
from core.tools.models import ToolContext, ToolResult


class WriteNoteTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="write_note",
            description="Write user-requested notes into markdown files.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        )

    def run(self, *, ctx: ToolContext, filename: str, content: str, **kwargs: Any) -> ToolResult:
        _ = kwargs
        safe_name = Path(filename).name
        if not safe_name.endswith(".md"):
            safe_name = f"{safe_name}.md"
        notes_dir = runtime_notes_dir()
        notes_dir.mkdir(parents=True, exist_ok=True)
        path = notes_dir / safe_name
        entry = f"\n## {datetime.utcnow().isoformat()}Z | session={ctx.session_id}\n{content}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
        return ToolResult(ok=True, content=f"note_written:{path}")


class ListNotesTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="list_notes",
            description="List note files created by write_note tool.",
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

    def run(self, *, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        _ = ctx, kwargs
        notes_dir = runtime_notes_dir()
        notes_dir.mkdir(parents=True, exist_ok=True)
        files = sorted([p.name for p in notes_dir.glob("*.md")])
        return ToolResult(ok=True, content=json.dumps({"notes": files}, ensure_ascii=False))

