from __future__ import annotations

import logging
import time
from hashlib import sha1
from typing import Any, Dict

_CONTEXT_KEYS = ("session_id", "round", "task_id", "step_id")


def elapsed_ms(start: float) -> int:
    return max(int((time.perf_counter() - start) * 1000), 0)


def summarize_text(
    text: str,
    *,
    preview_chars: int = 120,
    redact: bool = True,
) -> Dict[str, Any]:
    payload = str(text or "")
    preview = payload[: max(int(preview_chars), 0)]
    if len(payload) > len(preview):
        preview = preview.rstrip() + "..."
    result: Dict[str, Any] = {
        "text_len": len(payload),
        "text_preview": preview,
    }
    if redact:
        result["text_sha1"] = sha1(payload.encode("utf-8")).hexdigest()
        result["text_preview"] = ""
    return result


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    event_fields = dict(fields or {})
    extra = {
        "event": str(event or "log.message"),
        "event_fields": event_fields,
    }
    for key in _CONTEXT_KEYS:
        if key in event_fields:
            extra[key] = event_fields[key]
    logger.log(
        level,
        message,
        extra=extra,
    )


def log_exception(
    logger: logging.Logger,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    event_fields = dict(fields or {})
    extra = {
        "event": str(event or "log.error"),
        "event_fields": event_fields,
    }
    for key in _CONTEXT_KEYS:
        if key in event_fields:
            extra[key] = event_fields[key]
    logger.exception(
        message,
        extra=extra,
    )
