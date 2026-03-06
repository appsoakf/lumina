import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agentic.executor_agent import ExecutorAgent
from core.tools import ToolResult
from core.utils.errors import ErrorCode


def _executor_json(
    status: str,
    summary: str = "摘要",
    *,
    evidence=None,
    details=None,
    risks=None,
    next_steps=None,
) -> str:
    payload = {
        "status": status,
        "summary": summary,
        "evidence": list(evidence or []),
        "details": list(details or []),
        "risks": list(risks or []),
        "next_steps": list(next_steps or []),
    }
    return json.dumps(payload, ensure_ascii=False)


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, tool_call_id: str, name: str, arguments: str):
        self.id = tool_call_id
        self.function = _FakeToolFunction(name=name, arguments=arguments)

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _FakeMessage:
    def __init__(self, *, content: str = "", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeResponse:
    def __init__(self, message: _FakeMessage):
        self.choices = [_FakeChoice(message)]


class _FakeToolRegistry:
    def __init__(self, tool_names=None):
        self.calls = []
        self._tool_names = list(tool_names or ["get_current_time"])

    def list_schemas(self):
        schemas = []
        for tool_name in self._tool_names:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool_name),
                        "description": "fake tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            )
        return schemas

    def call(self, name, args, ctx):
        self.calls.append(
            {
                "name": name,
                "args": args,
                "session_id": ctx.session_id,
            }
        )
        return ToolResult(ok=True, content=f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}")


class _ExecutorAgentHarness(ExecutorAgent):
    def __init__(
        self,
        *,
        scripted_messages,
        tool_names=None,
        max_tool_rounds=4,
        max_repeated_tool_call=2,
    ):
        self.max_tool_rounds = max_tool_rounds
        self.max_repeated_tool_call = max_repeated_tool_call
        self.registry = _FakeToolRegistry(tool_names=tool_names)
        self._scripted_messages = list(scripted_messages)
        self.invocations = []

    def _system_prompt(self) -> str:
        return "executor prompt"

    def invoke_chat(self, messages, **kwargs):
        self.invocations.append(
            {
                "messages": list(messages),
                **kwargs,
            }
        )
        if not self._scripted_messages:
            raise AssertionError("No scripted message left")
        return _FakeResponse(self._scripted_messages.pop(0))


class ExecutorAgentPromptTests(unittest.TestCase):
    def test_system_prompt_uses_enhanced_contract(self):
        # _system_prompt 不依赖实例状态，绕过 __init__ 避免环境变量依赖。
        prompt = ExecutorAgent._system_prompt(ExecutorAgent.__new__(ExecutorAgent))

        self.assertIn("你的唯一职责", prompt)
        self.assertIn("【最终输出规则（必须严格遵守）】", prompt)
        self.assertIn("仅输出一个 JSON 对象", prompt)
        self.assertIn("\"status\": \"success|failed|need_info\"", prompt)
        self.assertIn("【工具调用规则】", prompt)
        self.assertIn("涉及文件读取或写入时，必须先调用对应文件工具", prompt)
        self.assertIn("不输出情绪字段或情绪 JSON", prompt)


class ExecutorAgentReactLoopTests(unittest.TestCase):
    def test_react_loop_returns_direct_output_without_tool_calls(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[_FakeMessage(content=_executor_json("success", "已完成"))],
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 成功", result.output_text)
        self.assertIn("结果摘要: 已完成", result.output_text)
        self.assertEqual(result.tool_events, [])
        self.assertEqual(len(agent.invocations), 1)
        self.assertIsNotNone(agent.invocations[0].get("tools"))

    def test_react_loop_calls_tool_then_returns_output(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(
                    content="先查时间",
                    tool_calls=[_FakeToolCall("c1", "get_current_time", '{"timezone":"UTC"}')],
                ),
                _FakeMessage(content=_executor_json("success", "工具调用后完成")),
            ],
            max_tool_rounds=3,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 成功", result.output_text)
        self.assertIn("结果摘要: 工具调用后完成", result.output_text)
        self.assertEqual(len(result.tool_events), 1)
        self.assertEqual(result.tool_events[0]["tool"], "get_current_time")
        self.assertEqual(len(agent.registry.calls), 1)
        self.assertEqual(len(agent.invocations), 2)

    def test_react_loop_returns_error_when_rounds_exceeded(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(tool_calls=[_FakeToolCall("c1", "get_current_time", '{"timezone":"UTC"}')]),
                _FakeMessage(tool_calls=[_FakeToolCall("c2", "get_current_time", '{"timezone":"UTC"}')]),
            ],
            max_tool_rounds=2,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.get("code"), ErrorCode.TOOL_EXECUTION_ERROR.value)
        self.assertIn("exceeded", str(result.error.get("message", "")))
        self.assertEqual(len(result.tool_events), 2)
        self.assertEqual(len(agent.invocations), 2)

    def test_react_loop_detects_repeated_tool_call_stall(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(tool_calls=[_FakeToolCall("c1", "get_current_time", '{"timezone":"UTC"}')]),
                _FakeMessage(tool_calls=[_FakeToolCall("c2", "get_current_time", '{"timezone":"UTC"}')]),
            ],
            max_tool_rounds=4,
            max_repeated_tool_call=1,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.get("code"), ErrorCode.TOOL_EXECUTION_ERROR.value)
        self.assertIn("loop detected", str(result.error.get("message", "")))
        self.assertEqual(len(result.tool_events), 1)

    def test_react_loop_fallbacks_to_normalized_text_when_model_not_json(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[_FakeMessage(content="我已经做完了，建议下一步继续。")],
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 需补充信息", result.output_text)
        self.assertIn("产出详情:", result.output_text)

    def test_react_loop_coerces_success_to_need_info_when_payload_marks_missing_info(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(
                    content=_executor_json(
                        "success",
                        "用户未提供预算和位置偏好，当前信息不足以继续。",
                        details=["缺少价格区间和就餐区域"],
                        next_steps=["询问用户预算范围和所在位置"],
                    )
                )
            ],
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 需补充信息", result.output_text)
        self.assertIn("结果摘要: 用户未提供预算和位置偏好", result.output_text)

    def test_react_loop_requires_file_tool_evidence_before_marking_success(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[_FakeMessage(content=_executor_json("success", "已写入文件"))],
            tool_names=["write_markdown", "read_file", "read_pdf"],
        )

        result = agent.run_task(
            user_text="请将内容写入 D:\\test_file.md",
            history=[],
            session_id="s1",
        )

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.get("code"), ErrorCode.TOOL_EXECUTION_ERROR.value)
        self.assertIn("未获取到有效的工具执行记录", result.output_text)
        self.assertEqual(len(result.tool_events), 0)
        self.assertEqual(
            agent.invocations[0].get("tool_choice"),
            {"type": "function", "function": {"name": "write_markdown"}},
        )

    def test_react_loop_forces_read_file_before_final_answer(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(
                    tool_calls=[
                        _FakeToolCall("c1", "read_file", '{"path":"D:\\\\test_file.md"}'),
                    ]
                ),
                _FakeMessage(content=_executor_json("success", "已读取文件内容")),
            ],
            tool_names=["read_file", "read_pdf"],
        )

        result = agent.run_task(
            user_text="请读取 D:\\test_file.md 的内容",
            history=[],
            session_id="s1",
        )

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 成功", result.output_text)
        self.assertEqual(len(result.tool_events), 1)
        self.assertEqual(result.tool_events[0]["tool"], "read_file")
        self.assertEqual(
            agent.invocations[0].get("tool_choice"),
            {"type": "function", "function": {"name": "read_file"}},
        )

    def test_react_loop_returns_error_when_required_file_tool_missing(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[],
            tool_names=["get_current_time"],
        )

        result = agent.run_task(
            user_text="请将内容写入 D:\\test_file.md",
            history=[],
            session_id="s1",
        )

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.get("code"), ErrorCode.TOOL_EXECUTION_ERROR.value)
        self.assertIn("缺少可用工具", result.output_text)


if __name__ == "__main__":
    unittest.main()
