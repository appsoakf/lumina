"""
Compatibility bridge for historical imports.

Tool implementations now live under `core.tools`.
"""

from core.tools import BaseTool, ToolContext, ToolRegistry, ToolResult, WebSearchTool, build_default_registry

__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "WebSearchTool",
    "build_default_registry",
]

