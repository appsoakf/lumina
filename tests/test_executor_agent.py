import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agentic.executor_agent import ExecutorAgent
from core.agentic.tools import ToolResult
from core.utils.errors import ErrorCode


def _step_output(status: str) -> str:
    return (
        f"步骤状态: {status}\n"
        "结果摘要: 摘要\n"
        "关键依据: 无\n"
        "产出详情: 无\n"
        "限制与风险: 无\n"
        "下一步建议: 无"
    )


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
    def __init__(self):
        self.calls = []

    def list_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "fake tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

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
        react_mode=ExecutorAgent.REACT_MODE_AUTO,
        max_tool_rounds=4,
        max_repeated_tool_call=2,
    ):
        self.max_tool_rounds = max_tool_rounds
        self.react_mode = self._normalize_react_mode(react_mode)
        self.max_repeated_tool_call = max_repeated_tool_call
        self.registry = _FakeToolRegistry()
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
        self.assertIn("【输出规则（必须严格遵守）】", prompt)
        self.assertIn("【字段要求】", prompt)
        self.assertIn("步骤状态:", prompt)
        self.assertIn("【工具调用规则】", prompt)
        self.assertIn("不输出情绪字段或情绪 JSON", prompt)
        self.assertIn("【输出示例】", prompt)


class ExecutorAgentReactModeTests(unittest.TestCase):
    def test_auto_mode_returns_direct_output_when_deliverable(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[_FakeMessage(content=_step_output("成功"))],
            react_mode=ExecutorAgent.REACT_MODE_AUTO,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 成功", result.output_text)
        self.assertEqual(result.tool_events, [])
        self.assertEqual(len(agent.invocations), 1)
        self.assertIsNone(agent.invocations[0].get("tools"))

    def test_auto_mode_upgrades_to_react_when_need_more_info(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(content=_step_output("需补充信息")),
                _FakeMessage(
                    content="先查时间",
                    tool_calls=[_FakeToolCall("c1", "get_current_time", '{"timezone":"UTC"}')],
                ),
                _FakeMessage(content=_step_output("成功")),
            ],
            react_mode=ExecutorAgent.REACT_MODE_AUTO,
            max_tool_rounds=3,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 成功", result.output_text)
        self.assertEqual(len(result.tool_events), 1)
        self.assertEqual(result.tool_events[0]["tool"], "get_current_time")
        self.assertEqual(len(agent.registry.calls), 1)
        self.assertEqual(len(agent.invocations), 3)
        self.assertIsNone(agent.invocations[0].get("tools"))
        self.assertIsNotNone(agent.invocations[1].get("tools"))

    def test_never_mode_keeps_direct_answer_without_tool_calls(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[_FakeMessage(content=_step_output("需补充信息"))],
            react_mode=ExecutorAgent.REACT_MODE_NEVER,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNone(result.error)
        self.assertIn("步骤状态: 需补充信息", result.output_text)
        self.assertEqual(result.tool_events, [])
        self.assertEqual(len(agent.invocations), 1)
        self.assertIsNone(agent.invocations[0].get("tools"))

    def test_always_mode_returns_error_when_rounds_exceeded(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(tool_calls=[_FakeToolCall("c1", "get_current_time", '{"timezone":"UTC"}')]),
                _FakeMessage(tool_calls=[_FakeToolCall("c2", "get_current_time", '{"timezone":"UTC"}')]),
            ],
            react_mode=ExecutorAgent.REACT_MODE_ALWAYS,
            max_tool_rounds=2,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.get("code"), ErrorCode.TOOL_EXECUTION_ERROR.value)
        self.assertIn("exceeded", str(result.error.get("message", "")))
        self.assertEqual(len(result.tool_events), 2)
        self.assertEqual(len(agent.invocations), 2)
        self.assertIsNotNone(agent.invocations[0].get("tools"))

    def test_react_loop_detects_repeated_tool_call_stall(self):
        agent = _ExecutorAgentHarness(
            scripted_messages=[
                _FakeMessage(tool_calls=[_FakeToolCall("c1", "get_current_time", '{"timezone":"UTC"}')]),
                _FakeMessage(tool_calls=[_FakeToolCall("c2", "get_current_time", '{"timezone":"UTC"}')]),
            ],
            react_mode=ExecutorAgent.REACT_MODE_ALWAYS,
            max_tool_rounds=4,
            max_repeated_tool_call=1,
        )

        result = agent.run_task(user_text="test", history=[], session_id="s1")

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.get("code"), ErrorCode.TOOL_EXECUTION_ERROR.value)
        self.assertIn("loop detected", str(result.error.get("message", "")))
        self.assertEqual(len(result.tool_events), 1)


if __name__ == "__main__":
    unittest.main()
