from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.config import load_app_config
from core.utils.errors import AppError, ErrorCode


class BaseLLMAgent:
    """Shared OpenAI client bootstrap and chat invocation for agent implementations."""

    def __init__(
        self,
        *,
        missing_key_message: str,
        missing_key_field: str,
        default_temperature: float = 0.5,
    ):
        llm_cfg = load_app_config().llm
        if not llm_cfg.chat_api_key:
            raise AppError(
                ErrorCode.CONFIG_MISSING,
                missing_key_message,
                details={"field": missing_key_field},
            )

        self.client = OpenAI(api_key=llm_cfg.chat_api_key, base_url=llm_cfg.chat_api_url)
        self.model = llm_cfg.chat_model
        self.default_temperature = float(default_temperature)

    def invoke_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
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
        return self.client.chat.completions.create(**kwargs)
