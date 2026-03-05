from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator

_DEFAULT_CONTEXT: Dict[str, Any] = {
    "session_id": "-",
    "round": "-",
    "task_id": "-",
    "step_id": "-",
}

_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar(
    "lumina_log_context",
    default=dict(_DEFAULT_CONTEXT),
)


def get_log_context() -> Dict[str, Any]:
    return dict(_LOG_CONTEXT.get())


def clear_log_context() -> None:
    _LOG_CONTEXT.set(dict(_DEFAULT_CONTEXT))


def set_log_context(**kwargs: Any) -> None:
    current = get_log_context()
    for key, value in kwargs.items():
        if value is None:
            current[key] = "-"
            continue
        text = str(value).strip()
        current[key] = text if text else "-"
    _LOG_CONTEXT.set(current)


@contextmanager
def bind_log_context(**kwargs: Any) -> Iterator[None]:
    previous = get_log_context()
    updated = dict(previous)
    for key, value in kwargs.items():
        if value is None:
            updated[key] = "-"
            continue
        text = str(value).strip()
        updated[key] = text if text else "-"
    _LOG_CONTEXT.set(updated)
    try:
        yield
    finally:
        _LOG_CONTEXT.set(previous)
