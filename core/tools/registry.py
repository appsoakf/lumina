from typing import Any, Dict, List

from core.tools.base import BaseTool
from core.tools.models import ToolContext, ToolResult


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def list_schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(str(name or "").strip())
        if tool is None:
            return ToolResult(ok=False, content=f"Unknown tool: {name}")
        return tool.invoke(args=args, ctx=ctx)

