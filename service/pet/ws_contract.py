import json
from typing import Optional, Tuple

from core.utils.errors import AppError, ErrorCode


def parse_user_text(raw_message: str) -> Tuple[Optional[str], Optional[AppError]]:
    """
    Parse and validate websocket request payload.

    Contract:
    - input must be a JSON object
    - `content` must be a string when provided
    - returns trimmed content; empty content is allowed and treated as no-op
    """
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        return None, AppError(
            ErrorCode.WEBSOCKET_ERROR,
            "Invalid websocket JSON payload",
            retryable=False,
            details={"reason": str(exc)},
        )

    if not isinstance(payload, dict):
        return None, AppError(
            ErrorCode.WEBSOCKET_ERROR,
            "Websocket payload must be a JSON object",
            retryable=False,
        )

    content = payload.get("content", "")
    if content is None:
        return "", None
    if not isinstance(content, str):
        return None, AppError(
            ErrorCode.WEBSOCKET_ERROR,
            "Field `content` must be string",
            retryable=False,
            details={"field": "content"},
        )

    return content.strip(), None
