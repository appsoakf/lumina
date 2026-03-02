import json
import logging
from typing import Dict, List

from openai import OpenAI

from core.agentic.tools import ToolContext, build_default_registry
from core.config import load_app_config
from core.error_codes import ErrorCode
from core.errors import LuminaError, error_payload
from core.protocols import ExecutorRunResult

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """Task executor agent: function-calling + tool execution."""

    def __init__(self, max_tool_rounds: int = 4):
        llm_cfg = load_app_config().llm
        if not llm_cfg.chat_api_key:
            raise LuminaError(
                ErrorCode.CONFIG_MISSING,
                "Missing LLM API key for executor agent",
                details={"field": "chat_api_key"},
            )

        self.client = OpenAI(api_key=llm_cfg.chat_api_key, base_url=llm_cfg.chat_api_url)
        self.model = llm_cfg.chat_model
        self.max_tool_rounds = max_tool_rounds
        self.registry = build_default_registry()

    def _system_prompt(self) -> str:
        return (
            "你是 executor_agent，负责通过工具完成用户任务。"
            "你可以调用工具，但不要输出情绪JSON。"
            "你的最终输出应是任务结果摘要，清晰、可执行。"
        )

    def run_task(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        user_id: str,
        session_id: str,
    ) -> ExecutorRunResult:
        tool_events = []
        ctx = ToolContext(user_id=user_id, session_id=session_id)

        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_text})

        try:
            for _ in range(self.max_tool_rounds):
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.registry.list_schemas(),
                    tool_choice="auto",
                    stream=False,
                    temperature=0.2,
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
                        event = {
                            "tool": tool_name,
                            "args": tool_args,
                            "ok": result.ok,
                            "result": result.content,
                        }
                        tool_events.append(event)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.to_model_text(),
                            }
                        )
                    continue

                text = (msg.content or "").strip()
                if text:
                    return ExecutorRunResult(output_text=text, tool_events=tool_events)

            return ExecutorRunResult(
                output_text="任务执行未收敛，请用户补充更具体要求。",
                tool_events=tool_events,
                error=error_payload(
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                    message="Executor tool rounds exceeded",
                    retryable=True,
                ),
            )
        except Exception as exc:
            logger.error(f"Executor run failed: {exc}")
            return ExecutorRunResult(
                output_text="任务执行失败。",
                tool_events=tool_events,
                error=error_payload(
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                    message=str(exc),
                    retryable=True,
                ),
            )
