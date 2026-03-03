from datetime import datetime
from typing import Any

from core.tools.base import BaseTool
from core.tools.models import ToolContext, ToolResult


class GetCurrentTimeTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="get_current_time",
            description="Get current time for user-facing responses or scheduling tasks.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Requested timezone name, e.g. UTC or Asia/Shanghai.",
                    }
                },
                "required": [],
            },
        )

    def run(self, *, ctx: ToolContext, timezone: str = "UTC", **kwargs: Any) -> ToolResult:
        _ = ctx, kwargs
        now = datetime.utcnow().isoformat() + "Z"
        return ToolResult(ok=True, content=f"UTC time={now}, requested_timezone={timezone}")

