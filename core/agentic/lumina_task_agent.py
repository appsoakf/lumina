import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from core.agentic.tools import ToolContext, build_default_registry
from core.config import load_app_config
from core.error_codes import ErrorCode
from core.errors import LuminaError

logger = logging.getLogger(__name__)


class LuminaTaskAgent:
    """Lightweight tool-calling agent for Lumina chat+task execution."""

    def __init__(self, max_tool_rounds: int = 3):
        llm_cfg = load_app_config().llm
        if not llm_cfg.chat_api_key:
            raise LuminaError(
                ErrorCode.CONFIG_MISSING,
                "Missing LLM API key for task agent",
                details={"field": "chat_api_key"},
            )
        self.client = OpenAI(api_key=llm_cfg.chat_api_key, base_url=llm_cfg.chat_api_url)
        self.model = llm_cfg.chat_model
        self.chat_prompt = llm_cfg.chat_prompt
        self.max_tool_rounds = max_tool_rounds
        self.registry = build_default_registry()

    def _system_prompt(self) -> str:
        return (
            self.chat_prompt
            + "\n\n你可以在需要时调用工具来完成任务。"
            + "如果用户明确要求执行任务（记录、查询、整理），优先尝试工具。"
            + "最终回复仍必须遵循原格式：第一行情绪JSON，第二行开始正文。"
        )

    def run(self, user_text: str, history: List[Dict[str, str]], user_id: str = "anonymous") -> Dict[str, Any]:
        tool_events: List[Dict[str, Any]] = []
        ctx = ToolContext(user_id=user_id, session_id="default")

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        try:
            for _ in range(self.max_tool_rounds):
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.registry.list_schemas(),
                    tool_choice="auto",
                    stream=False,
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)

                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [tc.model_dump() for tc in tool_calls],
                        }
                    )

                    for tc in tool_calls:
                        tool_name = tc.function.name
                        try:
                            tool_args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            tool_args = {}
                        result = self.registry.call(tool_name, tool_args, ctx)
                        tool_events.append(
                            {
                                "tool": tool_name,
                                "args": tool_args,
                                "ok": result.ok,
                                "result": result.content,
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.to_model_text(),
                            }
                        )
                    continue

                final_content = (msg.content or "").strip()
                if final_content:
                    return {"reply": final_content, "tool_events": tool_events}

            logger.warning("Tool rounds reached max; returning fallback response")
            return {
                "reply": '{"emotion": "平静", "intensity": 1}\n我已经尝试执行任务，但未获得稳定结果。请换一种表达再试一次。',
                "tool_events": tool_events,
            }
        except Exception as exc:
            logger.error(f"Task agent run failed: {exc}")
            return {
                "reply": '{"emotion": "平静", "intensity": 1}\n当前任务执行失败，请稍后重试。',
                "tool_events": tool_events,
                "error": {
                    "code": ErrorCode.TOOL_EXECUTION_ERROR.value,
                    "message": str(exc),
                    "retryable": True,
                },
            }
