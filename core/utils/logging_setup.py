from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from core.config import LoggingConfig
from core.paths import runtime_root
from core.utils.log_context import get_log_context

_LOGGING_CONFIGURED = False

_CONSOLE_PERF_EVENTS = {
    "service.start",
    "ws.session.start",
    "ws.session.end",
    "ws.round.start",
    "orchestrator.intent.done",
    "task.plan.done",
    "task.step.run.done",
    "executor.step.done",
    "task.review.done",
    "orchestrator.task.run.done",
    "orchestrator.chat.reply.done",
    "orchestrator.handle.done",
    "pipeline.tts.total.done",
    "ws.round.summary",
    "ws.round.end",
}

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


class ConsoleEventFilter(logging.Filter):
    def __init__(self, allowed_info_events: Iterable[str]):
        super().__init__()
        self._allowed_info_events = {str(v) for v in allowed_info_events}

    def filter(self, record: logging.LogRecord) -> bool:
        if int(getattr(record, "levelno", logging.INFO)) >= logging.WARNING:
            return True
        event = str(getattr(record, "event", "log.message") or "log.message")
        return event in self._allowed_info_events


class ConsoleFlowFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level = str(getattr(record, "levelname", "INFO") or "INFO")
        event = str(getattr(record, "event", "log.message") or "log.message")
        fields = dict(getattr(record, "event_fields", {}) or {})
        text = self._render_event_line(record=record, event=event, fields=fields)
        return f"{ts} {text}"

    def _render_event_line(
        self,
        *,
        record: logging.LogRecord,
        event: str,
        fields: Dict[str, Any],
    ) -> str:
        level = str(getattr(record, "levelname", "INFO") or "INFO").upper()
        round_id = str(getattr(record, "round", "-") or "-")
        step_id = str(fields.get("step_id") or getattr(record, "step_id", "-") or "-")
        duration_ms = self._as_int(fields.get("duration_ms"), default=-1)
        message = str(record.getMessage() or "").strip()

        if event == "service.start":
            server_ip = str(fields.get("server_ip") or "-")
            server_port = str(fields.get("server_port") or "-")
            enable_translation = bool(fields.get("enable_translation"))
            enable_tts = bool(fields.get("enable_tts"))
            translation_text = "开" if enable_translation else "关"
            tts_text = "开" if enable_tts else "关"
            return f"[系统] 服务启动：{server_ip}:{server_port}（翻译:{translation_text}，TTS:{tts_text}）"

        if event == "ws.session.start":
            return "[会话] 新会话已连接"

        if event == "ws.session.end":
            rounds = self._as_int(fields.get("rounds"), default=-1)
            reason = str(fields.get("reason") or "").strip()
            if rounds >= 0 and reason:
                return f"[会话] 会话结束：共 {rounds} 轮，原因：{reason}"
            if rounds >= 0:
                return f"[会话] 会话结束：共 {rounds} 轮"
            return "[会话] 会话结束"

        if event == "ws.round.start":
            return f"[请求] 开始处理第 {round_id} 轮用户消息"

        if event == "orchestrator.intent.done":
            intent = self._intent_text(fields.get("intent"))
            return f"[意图识别] 结果：{intent}，耗时 {self._ms_text(duration_ms)}"

        if event == "task.plan.done":
            resume_mode = bool(fields.get("resume_mode"))
            mode_text = "恢复" if resume_mode else "新建"
            step_count = self._as_int(fields.get("step_count"), default=-1)
            max_parallelism = self._as_int(fields.get("max_parallelism"), default=-1)
            fail_fast_text = "开" if bool(fields.get("fail_fast")) else "关"
            if step_count >= 0 and max_parallelism >= 0:
                return (
                    f"[任务规划] {mode_text}计划：{step_count} 个步骤，并行度 {max_parallelism}，"
                    f"fail_fast:{fail_fast_text}，耗时 {self._ms_text(duration_ms)}"
                )
            return f"[任务规划] {mode_text}计划完成，耗时 {self._ms_text(duration_ms)}"

        if event == "task.step.run.done":
            ok = bool(fields.get("ok"))
            status = "成功" if ok else "失败"
            return f"[编排步骤] {step_id} {status}，耗时 {self._ms_text(duration_ms)}"

        if event == "executor.step.done":
            ok = bool(fields.get("ok"))
            status = "成功" if ok else "失败"
            rounds = self._as_int(fields.get("rounds"), default=-1)
            llm_calls = self._as_int(fields.get("llm_calls"), default=-1)
            llm_ms = self._safe_ms(fields.get("llm_ms"))
            tool_calls = self._as_int(fields.get("tool_calls"), default=-1)
            tool_ms = self._safe_ms(fields.get("tool_ms"))
            rounds_text = str(rounds) if rounds >= 0 else "-"
            llm_calls_text = str(llm_calls) if llm_calls >= 0 else "-"
            tool_calls_text = str(tool_calls) if tool_calls >= 0 else "-"
            return (
                f"[步骤细分] {step_id} {status} | 轮次:{rounds_text} | "
                f"LLM:{llm_calls_text}次/{llm_ms} | 工具:{tool_calls_text}次/{tool_ms} | "
                f"总耗时 {self._ms_text(duration_ms)}"
            )

        if event == "task.review.done":
            quality = str(fields.get("quality") or "").strip().lower()
            quality_text = {
                "pass": "通过",
                "revise": "需修正",
                "fail": "失败",
            }.get(quality, quality or "-")
            suggestion_count = self._as_int(fields.get("suggestion_count"), default=-1)
            if suggestion_count >= 0:
                return f"[任务评审] 评审{quality_text}，建议 {suggestion_count} 条，耗时 {self._ms_text(duration_ms)}"
            return f"[任务评审] 评审{quality_text}，耗时 {self._ms_text(duration_ms)}"

        if event == "orchestrator.task.run.done":
            waiting_input = bool(fields.get("task_waiting_input"))
            has_error = bool(fields.get("task_error"))
            if has_error:
                status = "失败"
            elif waiting_input:
                status = "等待补充信息"
            else:
                status = "完成"
            step_count = self._as_int(fields.get("step_count"), default=-1)
            if step_count >= 0:
                return f"[编排执行] 任务{status}：{step_count} 个步骤，耗时 {self._ms_text(duration_ms)}"
            return f"[编排执行] 任务{status}，耗时 {self._ms_text(duration_ms)}"

        if event == "orchestrator.chat.reply.done":
            intent = self._intent_text(fields.get("intent"))
            return f"[对话生成] {intent}回复已生成，耗时 {self._ms_text(duration_ms)}"

        if event == "orchestrator.handle.done":
            intent = self._intent_text(fields.get("intent"))
            return f"[编排总计] 本轮{intent}链路耗时 {self._ms_text(duration_ms)}"

        if event == "pipeline.tts.total.done":
            sentence_count = self._as_int(fields.get("sentence_count"), default=-1)
            if sentence_count >= 0:
                return f"[语音处理] 语音阶段完成：{sentence_count} 段文本，耗时 {self._ms_text(duration_ms)}"
            return f"[语音处理] 语音阶段完成，耗时 {self._ms_text(duration_ms)}"

        if event == "ws.round.summary":
            intent = self._intent_text(fields.get("intent"))
            intent_ms = self._safe_ms(fields.get("intent_ms"))
            task_run_ms = self._safe_ms(fields.get("task_run_ms"))
            chat_llm_ms = self._safe_ms(fields.get("chat_llm_ms"))
            tts_ms = self._safe_ms(fields.get("tts_ms"))
            total_ms = self._safe_ms(fields.get("round_total_ms"))
            return (
                f"[本轮汇总] 意图:{intent} | 意图识别:{intent_ms} | 编排执行:{task_run_ms} | "
                f"对话LLM:{chat_llm_ms} | 语音:{tts_ms} | 总耗时:{total_ms}"
            )

        if event == "ws.round.end":
            return f"[请求] 第 {round_id} 轮处理结束，总耗时 {self._ms_text(duration_ms)}"

        # warning/error 或未映射事件使用统一兜底格式。
        if level == "WARNING":
            return f"[警告] {event}：{message or '发生告警'}"
        if level in {"ERROR", "CRITICAL"}:
            return f"[错误] {event}：{message or '发生错误'}"
        return f"[日志] {event}：{message or '-'}"

    def _as_int(self, value: Any, *, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _ms_text(self, value: int) -> str:
        if value < 0:
            return "-"
        return f"{value}ms"

    def _safe_ms(self, value: Any) -> str:
        parsed = self._as_int(value, default=-1)
        return self._ms_text(parsed)

    def _intent_text(self, raw: Any) -> str:
        intent = str(raw or "").strip().lower()
        if intent == "chat":
            return "闲聊"
        if intent == "task":
            return "任务"
        return intent or "-"


def _human_file_formatter() -> logging.Formatter:
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
            console_handler.setFormatter(ConsoleFlowFormatter())
        console_handler.addFilter(context_filter)
        console_handler.addFilter(ConsoleEventFilter(_CONSOLE_PERF_EVENTS))
        root.addHandler(console_handler)

    if config.enable_file:
        file_handler = logging.FileHandler(log_dir / config.log_file_name, encoding="utf-8")
        if config.format == "json":
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(_human_file_formatter())
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

    if config.enable_event_file:
        event_handler = logging.FileHandler(log_dir / config.event_file_name, encoding="utf-8")
        event_handler.setFormatter(JsonFormatter())
        event_handler.addFilter(context_filter)
        root.addHandler(event_handler)

    logging.captureWarnings(True)
    _LOGGING_CONFIGURED = True
