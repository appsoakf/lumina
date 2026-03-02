from typing import Any, Dict, List, Optional

from core.llm.chat_service import ChatCompletionService


class BaseLLMAgent:
    """Shared OpenAI client bootstrap and chat invocation for agent implementations."""

    def __init__(
        self,
        *,
        missing_key_message: str,
        missing_key_field: str,
        default_temperature: float = 0.5,
    ):
        self.default_temperature = float(default_temperature)
        self.llm = ChatCompletionService.from_chat_config(
            default_temperature=self.default_temperature,
            missing_key_message=missing_key_message,
            missing_key_field=missing_key_field,
        )

    def invoke_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
        return self.llm.invoke(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )

    def invoke_chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
        return self.llm.invoke_stream(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
