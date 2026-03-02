from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"

    LLM_API_ERROR = "LLM_API_ERROR"
    LLM_STREAM_ERROR = "LLM_STREAM_ERROR"

    TRANSLATE_API_ERROR = "TRANSLATE_API_ERROR"
    TRANSLATE_EMPTY_RESULT = "TRANSLATE_EMPTY_RESULT"

    TTS_CONNECTION_ERROR = "TTS_CONNECTION_ERROR"
    TTS_API_ERROR = "TTS_API_ERROR"
    TTS_STREAM_ERROR = "TTS_STREAM_ERROR"

    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    WEBSOCKET_ERROR = "WEBSOCKET_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    def to_payload(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


def error_payload(
    code: ErrorCode,
    message: str,
    retryable: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "code": code.value,
        "message": message,
        "retryable": retryable,
        "details": details or {},
    }
