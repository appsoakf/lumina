from typing import Any, Dict, Optional

from core.error_codes import ErrorCode


class LuminaError(Exception):
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
