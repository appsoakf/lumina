import json
import logging
from typing import Any, Dict, List, Optional

from core.agentic.base import BaseLLMAgent
from core.agentic.json_mixin import JSONParseMixin
from core.protocols import ExecutorRunResult
from core.tools import ToolContext, build_default_registry
from core.utils import log_exception
from core.utils.errors import ErrorCode, error_payload

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseLLMAgent, JSONParseMixin):
    """Task executor agent: unified ReAct loop with function-calling."""

    STATUS_SUCCESS = "成功"
    STATUS_FAILED = "失败"
    STATUS_NEED_INFO = "需补充信息"

    def __init__(self, max_tool_rounds: int = 4, max_repeated_tool_call: int = 2):
        super().__init__(
            missing_key_message="Missing LLM API key for executor agent",
            missing_key_field="chat_api_key",
            default_temperature=0.2,
        )
        self.max_tool_rounds = max(int(max_tool_rounds), 1)
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
3. 信息不足时可自行决定是否调用工具补齐事实；信息足够时直接交付结果。
4. 只能基于输入和工具真实返回结果作答，禁止编造事实或工具结果。
5. 不暴露内部推理过程，仅输出结果与必要依据。

【工具调用规则】
1. 若现有信息足够，避免不必要的工具调用。
2. 工具参数要最小化且精确，避免宽泛查询。
3. 工具失败时，明确失败原因、是否可重试、建议补充信息。

【最终输出规则（必须严格遵守）】
1. 仅输出一个 JSON 对象，不输出其他文字，不使用 markdown 代码块。
2. JSON 顶层字段固定为：
{
  "status": "success|failed|need_info",
  "summary": "string",
  "evidence": ["string"],
  "details": ["string"],
  "risks": ["string"],
  "next_steps": ["string"]
}
3. 若某字段无内容，列表字段输出空数组，字符串字段输出空字符串。
4. 若工具调用后信息仍不足，status 设为 need_info，并在 summary/next_steps 明确缺失项。
5. 输出语言默认中文。

【约束】
1. 不输出情绪字段或情绪 JSON（如 emotion/mood/arousal）。
2. 不输出与当前步骤无关的寒暄、角色扮演、口号化内容。
3. 不得声称已完成未实际完成的动作。
"""

    def _build_messages(self, user_text: str, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_text})
        return messages

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

    def _normalize_status(self, raw_status: Any) -> str:
        text = str(raw_status or "").strip().lower()
        if text in {"success", "succeeded", "ok", "完成", "成功"}:
            return self.STATUS_SUCCESS
        if text in {"failed", "failure", "error", "失败"}:
            return self.STATUS_FAILED
        if text in {"need_info", "need-more-info", "missing_info", "需补充信息", "信息不足"}:
            return self.STATUS_NEED_INFO
        return self.STATUS_NEED_INFO

    def _to_string_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            return items
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []

    def _looks_like_missing_information(
        self,
        *,
        summary: str,
        details: List[str],
        next_steps: List[str],
    ) -> bool:
        text = "\n".join([summary, *details, *next_steps]).strip()
        if not text:
            return False

        negative_hints = {
            "无需补充",
            "不需要补充",
            "信息充足",
            "信息完整",
        }
        if any(hint in text for hint in negative_hints):
            return False

        positive_hints = {
            "信息不足",
            "信息不完整",
            "缺少",
            "缺乏",
            "未提供",
            "需补充",
            "需要补充",
            "请补充",
            "补充信息",
            "询问用户",
            "无法继续",
            "无法判断",
        }
        return any(hint in text for hint in positive_hints)

    def _extract_first_line(self, text: str) -> str:
        for line in str(text or "").splitlines():
            candidate = line.strip()
            if candidate:
                return candidate
        return ""

    def _parse_final_payload(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = self.parse_json_object(text, allow_brace_extract=True)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _fallback_render_text(self, raw_text: str) -> str:
        first_line = self._extract_first_line(raw_text)
        summary = first_line or "模型未返回结构化结果，已保留原始内容。"
        detail = str(raw_text or "").strip() or "无"
        return (
            f"步骤状态: {self.STATUS_NEED_INFO}\n"
            f"结果摘要: {summary}\n"
            "关键依据:\n无\n"
            f"产出详情:\n{detail}\n"
            "限制与风险:\n无\n"
            "下一步建议:\n- 请补充更具体约束或允许调用更多工具"
        )

    def _render_final_text(self, payload: Dict[str, Any], raw_text: str) -> str:
        status = self._normalize_status(payload.get("status"))
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            summary = self._extract_first_line(raw_text) or "无"

        evidence = self._to_string_list(payload.get("evidence"))
        details = self._to_string_list(payload.get("details"))
        risks = self._to_string_list(payload.get("risks"))
        next_steps = self._to_string_list(payload.get("next_steps"))

        # 兜底纠偏：有些模型会把“缺信息”场景误标成 success。
        # 当摘要/细节/下一步建议明显指向“需补充信息”时，统一归一到 need_info。
        if status == self.STATUS_SUCCESS and self._looks_like_missing_information(
            summary=summary,
            details=details,
            next_steps=next_steps,
        ):
            status = self.STATUS_NEED_INFO

        evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "无"
        details_text = "\n".join(f"- {item}" for item in details) if details else "无"
        risks_text = "\n".join(f"- {item}" for item in risks) if risks else "无"
        next_text = "\n".join(f"- {item}" for item in next_steps) if next_steps else "无"

        return (
            f"步骤状态: {status}\n"
            f"结果摘要: {summary}\n"
            f"关键依据:\n{evidence_text}\n"
            f"产出详情:\n{details_text}\n"
            f"限制与风险:\n{risks_text}\n"
            f"下一步建议:\n{next_text}"
        )

    def _normalize_final_output(self, raw_text: str) -> str:
        payload = self._parse_final_payload(raw_text)
        if payload is None:
            return self._fallback_render_text(raw_text)
        return self._render_final_text(payload, raw_text=raw_text)

    def _run_react_loop(
        self,
        *,
        messages: List[Dict[str, Any]],
        ctx: ToolContext,
        tool_events: List[Dict[str, Any]],
    ) -> ExecutorRunResult:
        # 记录“工具名+参数”签名出现次数，避免模型陷入重复调用死循环。
        repeated_call_counter: Dict[str, int] = {}
        tool_schemas = self.registry.list_schemas()

        # ReAct 状态机：
        # DECIDE(模型决策) -> ACT(执行工具) -> OBSERVE(回填工具结果) -> DECIDE ...
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
                # 先把 assistant 的 tool_calls 消息写入上下文，再逐个执行工具并写回 observation。
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

            # 没有 tool_calls 时，若模型给出文本则作为最终可交付结果并统一规范化。
            text = (msg.content or "").strip()
            if text:
                normalized_text = self._normalize_final_output(text)
                return ExecutorRunResult(output_text=normalized_text, tool_events=tool_events)

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
            # 单通道入口：始终通过统一 ReAct 循环收敛，不再分 direct-pass 与升级路径。
            return self._run_react_loop(messages=messages, ctx=ctx, tool_events=tool_events)
        except Exception as exc:
            log_exception(
                logger,
                "executor.run.error",
                "Executor 执行失败",
                component="agent",
                error_code=ErrorCode.TOOL_EXECUTION_ERROR.value,
                retryable=True,
            )
            return ExecutorRunResult(
                output_text="任务执行失败。",
                tool_events=tool_events,
                error=error_payload(
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                    message=str(exc),
                    retryable=True,
                ),
            )
