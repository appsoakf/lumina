from core.tools.builtin_notes import ListNotesTool, WriteNoteTool
from core.tools.builtin_time import GetCurrentTimeTool
from core.tools.registry import ToolRegistry
from core.tools.web_search import WebSearchTool


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GetCurrentTimeTool())
    registry.register(WriteNoteTool())
    registry.register(ListNotesTool())
    registry.register(WebSearchTool())
    return registry

