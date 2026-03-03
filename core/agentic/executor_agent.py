import json
import logging
from typing import Any, Dict, List, Optional

from core.tools import ToolContext, build_default_registry
from core.agentic.base import BaseLLMAgent
from core.utils.errors import ErrorCode, error_payload
from core.protocols import ExecutorRunResult

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseLLMAgent):
    """Task executor agent: function-calling + tool execution."""

    REACT_MODE_AUTO = "auto"
    REACT_MODE_ALWAYS = "always"
    REACT_MODE_NEVER = "never"

    STEP_STATUS_SUCCESS = "成功"
    STEP_STATUS_FAILED = "失败"
    STEP_STATUS_NEED_INFO = "需补充信息"
    STEP_STATUS_UNKNOWN = "unknown"

    def __init__(
        self,
        max_tool_rounds: int = 4,
        react_mode: str = REACT_MODE_AUTO,
        max_repeated_tool_call: int = 2,
    ):
        super().__init__(
            missing_key_message="Missing LLM API key for executor agent",
            missing_key_field="chat_api_key",
            default_temperature=0.2,
        )
        self.max_tool_rounds = max(int(max_tool_rounds), 1)
        self.react_mode = self._normalize_react_mode(react_mode)
        self.max_repeated_tool_call = max(int(max_repeated_tool_call), 1)
        self.registry = build_default_registry()

    def _system_prompt(self) -> str:
        return """你是 executor_agent。
你的唯一职责：执行“当前步骤”的任务指令并产出该步骤结果，供下游步骤或评审使用。
你不重新规划任务、不改写总目标、不输出任务计划 JSON、不和用户闲聊。

【输入理解】
你通常会收到以下输入片段：
- 总目标
- 当前步骤（step_id + title）
- 步骤指令（instruction）
- 结构化输入绑定（input_bindings 解析结果）
- 已完成上下文（上游步骤结果）

【执行规则（必须严格遵守）】
1. 只执行当前步骤，不跨步骤完成未来工作。
2. 优先使用结构化输入绑定与已完成上下文，保持与任务图依赖一致。
3. 信息不足时可以调用工具补齐事实；若仍不足，明确写出缺失项与影响。
4. 只能基于输入和工具真实返回结果作答，禁止编造事实或工具结果。
5. 不暴露内部推理过程，仅输出结果与必要依据。

【工具调用规则】
1. 若现有信息足够，避免不必要的工具调用。
2. 工具参数要最小化且精确，避免宽泛查询。
3. 工具失败时，明确失败原因、是否可重试、建议补充信息。

【输出规则（必须严格遵守）】
1. 输出纯文本，不要输出 JSON，不要使用 markdown 代码块。
2. 输出必须包含以下 6 个字段标题，顺序固定：
步骤状态:
结果摘要:
关键依据:
产出详情:
限制与风险:
下一步建议:
3. 若某项无内容，写“无”。

【字段要求】
1. 步骤状态
- 仅允许：成功 / 失败 / 需补充信息。

2. 结果摘要
- 1-3 句，直接说明本步骤是否完成以及核心产出。

3. 关键依据
- 列出支撑结论的依据，可来自输入绑定、已完成上下文、工具返回。

4. 产出详情
- 给出可执行结果，如清单、参数、决策、草案内容或操作结果。

5. 限制与风险
- 说明边界条件、未覆盖范围、不确定性。

6. 下一步建议
- 给出对后续步骤可直接使用的建议；若无需建议写“无”。

【约束】
1. 不输出情绪字段或情绪 JSON（如 emotion/mood/arousal）。
2. 不输出与当前步骤无关的寒暄、角色扮演、口号化内容。
3. 不得声称已完成未实际完成的动作。
4. 语言简洁、明确、可执行，默认中文。

【输出示例】
步骤状态: 成功
结果摘要: 已完成首页与登录接口可用性检查，核心链路可用。
关键依据:
- 工具 http_check 返回首页状态码 200
- 工具 api_probe 返回 /login 状态码 200
产出详情:
- 首页延迟: 230ms
- 登录接口: 正常
- 建议监控项: 错误率、P95 延迟
限制与风险:
- 未覆盖高并发压测场景
下一步建议:
- 增加并发压测并记录 P95/P99
"""

    def _normalize_react_mode(self, mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in {
            self.REACT_MODE_AUTO,
            self.REACT_MODE_ALWAYS,
            self.REACT_MODE_NEVER,
        }:
            return normalized
        logger.warning("Unknown react_mode=%s, fallback to auto", mode)
        return self.REACT_MODE_AUTO

    def _build_messages(self, user_text: str, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _extract_step_status(self, text: str) -> str:
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("步骤状态"):
                continue
            _, _, rhs = line.replace("：", ":", 1).partition(":")
            state_text = rhs.strip()
            if self.STEP_STATUS_NEED_INFO in state_text:
                return self.STEP_STATUS_NEED_INFO
            if self.STEP_STATUS_SUCCESS in state_text:
                return self.STEP_STATUS_SUCCESS
            if self.STEP_STATUS_FAILED in state_text:
                return self.STEP_STATUS_FAILED
            return self.STEP_STATUS_UNKNOWN
        return self.STEP_STATUS_UNKNOWN

    def _looks_information_missing(self, text: str) -> bool:
        payload = str(text or "")
        signals = (
            "需补充信息",
            "信息不足",
            "无法完成",
            "无法判断",
            "缺少",
            "缺失",
            "请补充",
        )
        return any(signal in payload for signal in signals)

    def _is_deliverable_output(self, text: str) -> bool:
        status = self._extract_step_status(text)
        return status in {self.STEP_STATUS_SUCCESS, self.STEP_STATUS_FAILED}

    def _should_upgrade_to_react(self, direct_text: str) -> bool:
        status = self._extract_step_status(direct_text)
        if status == self.STEP_STATUS_NEED_INFO:
            return True
        if status == self.STEP_STATUS_UNKNOWN and self._looks_information_missing(direct_text):
            return True
        return False

    def _react_upgrade_instruction(self) -> str:
        return (
            "上一轮结论显示当前信息不足。"
            "请在必要时调用工具补齐事实，再给出可交付的步骤结果。"
        )

    def _invoke_direct_pass(self, messages: List[Dict[str, Any]]) -> str:
        resp = self.invoke_chat(messages=messages, temperature=0.2)
        msg = resp.choices[0].message
        return (msg.content or "").strip()

    def _tool_call_name(self, tool_call: Any) -> str:
        function = getattr(tool_call, "function", None)
        return str(getattr(function, "name", "") or "").strip()

    def _tool_call_arguments_text(self, tool_call: Any) -> str:
        function = getattr(tool_call, "function", None)
        return str(getattr(function, "arguments", "") or "")

    def _parse_tool_args(self, tool_call: Any) -> Dict[str, Any]:
        raw = self._tool_call_arguments_text(tool_call)
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict):
            return data
        return {}

    def _serialize_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        dump_fn = getattr(tool_call, "model_dump", None)
        if callable(dump_fn):
            dumped = dump_fn()
            if isinstance(dumped, dict):
                return dumped
        return {
            "id": str(getattr(tool_call, "id", "") or ""),
            "type": "function",
            "function": {
                "name": self._tool_call_name(tool_call),
                "arguments": self._tool_call_arguments_text(tool_call),
            },
        }

    def _tool_call_signature(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        encoded_args = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
        return f"{tool_name}|{encoded_args}"

    def _run_react_loop(
        self,
        *,
        messages: List[Dict[str, Any]],
        ctx: ToolContext,
        tool_events: List[Dict[str, Any]],
    ) -> ExecutorRunResult:
        repeated_call_counter: Dict[str, int] = {}
        tool_schemas = self.registry.list_schemas()

        # ReAct state machine:
        # DECIDE(model) -> ACT(call tools) -> OBSERVE(tool outputs) -> DECIDE ...
        for _ in range(self.max_tool_rounds):
            resp = self.invoke_chat(
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            tool_calls = list(getattr(msg, "tool_calls", None) or [])

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [self._serialize_tool_call(tc) for tc in tool_calls],
                    }
                )
                for tc in tool_calls:
                    tool_name = self._tool_call_name(tc)
                    tool_args = self._parse_tool_args(tc)
                    signature = self._tool_call_signature(tool_name, tool_args)
                    repeated_call_counter[signature] = repeated_call_counter.get(signature, 0) + 1
                    if repeated_call_counter[signature] > self.max_repeated_tool_call:
                        return ExecutorRunResult(
                            output_text="任务执行陷入重复工具调用，请用户补充更具体约束。",
                            tool_events=tool_events,
                            error=error_payload(
                                code=ErrorCode.TOOL_EXECUTION_ERROR,
                                message="Executor tool call loop detected",
                                retryable=True,
                            ),
                        )

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
                            "tool_call_id": str(getattr(tc, "id", "") or ""),
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

    def run_task(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        session_id: str,
    ) -> ExecutorRunResult:
        tool_events: List[Dict[str, Any]] = []
        ctx = ToolContext(session_id=session_id)
        messages = self._build_messages(user_text=user_text, history=history)

        try:
            if self.react_mode != self.REACT_MODE_ALWAYS:
                direct_text = self._invoke_direct_pass(messages=messages)
                if direct_text:
                    if self.react_mode == self.REACT_MODE_NEVER:
                        return ExecutorRunResult(output_text=direct_text, tool_events=tool_events)
                    if self._is_deliverable_output(direct_text):
                        return ExecutorRunResult(output_text=direct_text, tool_events=tool_events)
                    if not self._should_upgrade_to_react(direct_text):
                        return ExecutorRunResult(output_text=direct_text, tool_events=tool_events)
                    messages.append({"role": "assistant", "content": direct_text})
                    messages.append({"role": "user", "content": self._react_upgrade_instruction()})
                elif self.react_mode == self.REACT_MODE_NEVER:
                    return ExecutorRunResult(
                        output_text="任务执行失败。",
                        tool_events=tool_events,
                        error=error_payload(
                            code=ErrorCode.TOOL_EXECUTION_ERROR,
                            message="Executor direct pass returned empty response",
                            retryable=True,
                        ),
                    )

            return self._run_react_loop(messages=messages, ctx=ctx, tool_events=tool_events)
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
