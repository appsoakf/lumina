import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from core.tools.models import ToolContext, ToolResult


class BaseTool(ABC):
    """Base contract for all callable tools."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        max_retries: int = 0,
        retry_backoff_sec: float = 0.2,
    ):
        self.name = str(name).strip()
        self.description = str(description).strip()
        self.parameters_schema = dict(parameters_schema or {})
        self.max_retries = max(int(max_retries), 0)
        self.retry_backoff_sec = max(float(retry_backoff_sec), 0.0)

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def invoke(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        safe_args = args if isinstance(args, dict) else {}
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                result = self.run(ctx=ctx, **safe_args)
                if isinstance(result, ToolResult):
                    return result
                return self.error_result(
                    code="TOOL_RUNTIME_ERROR",
                    message=f"Tool {self.name} returned invalid result type: {type(result).__name__}",
                    retryable=False,
                )
            except TypeError as exc:
                return self.error_result(
                    code="TOOL_BAD_ARGS",
                    message=f"Invalid args for {self.name}: {exc}",
                    retryable=False,
                )
            except Exception as exc:
                retryable = self.is_retryable_exception(exc)
                if retryable and attempt < attempts - 1:
                    delay = self.retry_backoff_sec * (2 ** attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                return self.error_result(
                    code="TOOL_RUNTIME_ERROR",
                    message=f"Tool {self.name} failed: {exc}",
                    retryable=retryable,
                )

        return self.error_result(
            code="TOOL_RUNTIME_ERROR",
            message=f"Tool {self.name} failed unexpectedly",
            retryable=True,
        )

    def ok_result(self, payload: Dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=json.dumps(payload, ensure_ascii=False))

    def error_result(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        details: Dict[str, Any] | None = None,
    ) -> ToolResult:
        payload: Dict[str, Any] = {
            "error_code": str(code),
            "message": str(message),
            "retryable": bool(retryable),
        }
        if details:
            payload["details"] = dict(details)
        return ToolResult(ok=False, content=json.dumps(payload, ensure_ascii=False))

    def clamp_int(self, value: Any, *, default: int, min_value: int, max_value: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = int(default)
        return max(min(parsed, max_value), min_value)

    def is_retryable_exception(self, exc: Exception) -> bool:
        _ = exc
        return False

    @abstractmethod
    def run(self, *, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

