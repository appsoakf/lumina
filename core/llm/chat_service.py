import logging
import time
from typing import Any, Dict, List, Optional

from core.config import load_app_config
from core.llm.client import create_openai_client
from core.utils import elapsed_ms, log_event, log_exception
from core.utils.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)


class ChatCompletionService:
    """Shared chat-completions wrapper with sync and stream helpers."""

    def __init__(
        self,
        *,
        model: str,
        api_url: str,
        api_key: str,
        default_temperature: float = 0.5,
        missing_key_message: Optional[str] = None,
        missing_key_field: str = "api_key",
    ):
        if not api_key:
            raise AppError(
                ErrorCode.CONFIG_MISSING,
                missing_key_message or "Missing LLM API key.",
                details={"field": missing_key_field},
            )
        self.client = create_openai_client(api_key=api_key, base_url=api_url)
        self.model = model
        self.default_temperature = float(default_temperature)

    @classmethod
    def from_chat_config(
        cls,
        *,
        default_temperature: float = 0.5,
        missing_key_message: str = "Missing LLM API key for chat completion service.",
        missing_key_field: str = "chat_api_key",
    ) -> "ChatCompletionService":
        cfg = load_app_config().llm
        return cls(
            model=cfg.chat_model,
            api_url=cfg.chat_api_url,
            api_key=cfg.chat_api_key,
            default_temperature=default_temperature,
            missing_key_message=missing_key_message,
            missing_key_field=missing_key_field,
        )

    @classmethod
    def from_translate_config(
        cls,
        *,
        default_temperature: float = 0.0,
        missing_key_message: str = "Missing translate API key. Set LUMINA_API_KEY or translate_api_key in config.",
        missing_key_field: str = "translate_api_key",
    ) -> "ChatCompletionService":
        cfg = load_app_config().llm
        return cls(
            model=cfg.translate_model,
            api_url=cfg.translate_api_url,
            api_key=cfg.translate_api_key,
            default_temperature=default_temperature,
            missing_key_message=missing_key_message,
            missing_key_field=missing_key_field,
        )

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
        started = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.default_temperature if temperature is None else float(temperature),
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        try:
            response = self.client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            fields: Dict[str, Any] = {
                "component": "llm",
                "model": self.model,
                "duration_ms": elapsed_ms(started),
                "stream": False,
            }
            if usage is not None:
                fields["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
                fields["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
                fields["total_tokens"] = int(getattr(usage, "total_tokens", 0) or 0)
            log_event(
                logger,
                logging.INFO,
                "llm.invoke.done",
                "LLM 同步调用完成",
                **fields,
            )
            return response
        except Exception:
            log_exception(
                logger,
                "llm.invoke.error",
                "LLM 同步调用失败",
                component="llm",
                model=self.model,
                duration_ms=elapsed_ms(started),
                stream=False,
                error_code=ErrorCode.LLM_API_ERROR.value,
                retryable=True,
            )
            raise

    def invoke_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
        started = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.default_temperature if temperature is None else float(temperature),
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        try:
            response = self.client.chat.completions.create(**kwargs)
            log_event(
                logger,
                logging.INFO,
                "llm.stream.open",
                "LLM 流式调用建立成功",
                component="llm",
                model=self.model,
                duration_ms=elapsed_ms(started),
                stream=True,
            )
            return response
        except Exception:
            log_exception(
                logger,
                "llm.stream.error",
                "LLM 流式调用失败",
                component="llm",
                model=self.model,
                duration_ms=elapsed_ms(started),
                stream=True,
                error_code=ErrorCode.LLM_STREAM_ERROR.value,
                retryable=True,
            )
            raise
