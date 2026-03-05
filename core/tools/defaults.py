from core.config import load_app_config
from core.tools.builtin_notes import ListNotesTool, WriteNoteTool
from core.tools.builtin_time import GetCurrentTimeTool
from core.tools.file_io import ReadFileTool, ReadPdfTool, WriteMarkdownTool
from core.tools.registry import ToolRegistry
from core.tools.web_search import WebSearchTool


def build_default_registry() -> ToolRegistry:
    app_cfg = load_app_config()
    registry = ToolRegistry()
    registry.register(GetCurrentTimeTool())
    registry.register(WriteNoteTool())
    registry.register(ListNotesTool())
    if app_cfg.tools.file_io.enabled:
        registry.register(ReadFileTool(config=app_cfg.tools.file_io))
        registry.register(ReadPdfTool(config=app_cfg.tools.file_io))
        registry.register(WriteMarkdownTool(config=app_cfg.tools.file_io))
    registry.register(WebSearchTool(config=app_cfg.tools.web_search))
    return registry
