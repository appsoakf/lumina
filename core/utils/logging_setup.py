from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from core.config import LoggingConfig
from core.paths import runtime_root
from core.utils.log_context import get_log_context

_LOGGING_CONFIGURED = False

_RESERVED_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


def _resolve_level(level_text: str) -> int:
    name = str(level_text or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def _resolve_log_dir(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        return runtime_root() / "logs"
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return runtime_root() / path


class LogContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = get_log_context()
        record.session_id = str(getattr(record, "session_id", ctx.get("session_id", "-")) or "-")
        record.round = str(getattr(record, "round", ctx.get("round", "-")) or "-")
        record.task_id = str(getattr(record, "task_id", ctx.get("task_id", "-")) or "-")
        record.step_id = str(getattr(record, "step_id", ctx.get("step_id", "-")) or "-")
        record.event = str(getattr(record, "event", "log.message") or "log.message")

        event_fields = getattr(record, "event_fields", {}) or {}
        if not isinstance(event_fields, dict):
            event_fields = {}
        record.event_fields = dict(event_fields)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "log.message"),
            "msg": record.getMessage(),
            "logger": record.name,
            "component": record.name.split(".", 1)[0],
            "session_id": getattr(record, "session_id", "-"),
            "round": getattr(record, "round", "-"),
            "task_id": getattr(record, "task_id", "-"),
            "step_id": getattr(record, "step_id", "-"),
            "file": record.pathname,
            "line": record.lineno,
            "func": record.funcName,
        }

        fields = dict(getattr(record, "event_fields", {}) or {})
        if fields:
            payload.update(fields)

        if record.exc_info:
            exception_type = ""
            exception_message = ""
            if len(record.exc_info) == 3 and record.exc_info[1] is not None:
                exception_type = type(record.exc_info[1]).__name__
                exception_message = str(record.exc_info[1])
            payload["exception_type"] = exception_type or "Exception"
            payload["exception_message"] = exception_message
            payload["traceback"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED_FIELDS:
                continue
            if key in {
                "event",
                "event_fields",
                "session_id",
                "round",
                "task_id",
                "step_id",
            }:
                continue
            if key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False)


def _human_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt=(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s %(message)s "
            "| event=%(event)s session=%(session_id)s round=%(round)s task=%(task_id)s step=%(step_id)s"
        ),
        datefmt="%H:%M:%S",
    )


def setup_logging(config: LoggingConfig, *, force: bool = False) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    root.setLevel(_resolve_level(config.level))
    context_filter = LogContextFilter()

    log_dir = _resolve_log_dir(config.log_dir)
    if config.enable_file or config.enable_event_file:
        log_dir.mkdir(parents=True, exist_ok=True)

    if config.enable_console:
        console_handler = logging.StreamHandler()
        if config.format == "json":
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(_human_formatter())
        console_handler.addFilter(context_filter)
        root.addHandler(console_handler)

    if config.enable_file:
        file_handler = logging.FileHandler(log_dir / config.log_file_name, encoding="utf-8")
        if config.format == "json":
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(_human_formatter())
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

    if config.enable_event_file:
        event_handler = logging.FileHandler(log_dir / config.event_file_name, encoding="utf-8")
        event_handler.setFormatter(JsonFormatter())
        event_handler.addFilter(context_filter)
        root.addHandler(event_handler)

    logging.captureWarnings(True)
    _LOGGING_CONFIGURED = True
