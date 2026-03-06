import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agentic.planner_agent import PlannerAgent


class _PlannerAgentStub(PlannerAgent):
    def __init__(self, model_output: str, *, max_steps: int = 6):
        self.max_steps = max_steps
        self._model_output = model_output
        self.captured_system_prompt = ""

    def _invoke(self, messages):
        self.captured_system_prompt = str(messages[0].get("content", ""))
        return self._model_output


class PlannerAgentTests(unittest.TestCase):
    def test_plan_task_uses_enhanced_prompt_contract(self):
        payload = (
            '{"goal":"g","graph_policy":{"max_parallelism":1,"fail_fast":true},'
            '"steps":[{"title":"a","instruction":"do a"},{"title":"b","instruction":"do b","depends_on":["S1"]}]}'
        )
        agent = _PlannerAgentStub(payload)
        result = agent.plan_task(user_text="demo", history=[])

        self.assertEqual(result.goal, "g")
        self.assertEqual(len(result.steps), 2)
        self.assertIn("你的唯一职责", agent.captured_system_prompt)
        self.assertIn("【输出规则（必须严格遵守）】", agent.captured_system_prompt)
        self.assertIn("【输出示例】", agent.captured_system_prompt)
        self.assertIn("步骤数必须为 2-5", agent.captured_system_prompt)
        self.assertIn("max_parallelism: 正整数，建议 1-2", agent.captured_system_prompt)
        self.assertIn("自然语序，不等于依赖关系", agent.captured_system_prompt)
        self.assertIn("可独立完成的子任务（如机票搜索与酒店搜索）", agent.captured_system_prompt)
        self.assertIn("仅在当前步骤确实要消费上游字段时填写 input_bindings", agent.captured_system_prompt)
        self.assertIn("优先并行规划", agent.captured_system_prompt)

    def test_plan_task_fallback_when_output_is_not_json(self):
        agent = _PlannerAgentStub("not json")
        result = agent.plan_task(user_text="demo", history=[])

        self.assertEqual(result.goal, "demo")
        self.assertEqual(len(result.steps), 2)
        self.assertIsNotNone(result.error)
        self.assertIn("Planner failed", str(result.error.get("message", "")))

    def test_plan_task_limits_steps_to_five_when_configured(self):
        payload = (
            '{"goal":"g","graph_policy":{"max_parallelism":1,"fail_fast":true},"steps":['
            '{"title":"a","instruction":"do a"},'
            '{"title":"b","instruction":"do b","depends_on":["S1"]},'
            '{"title":"c","instruction":"do c","depends_on":["S2"]},'
            '{"title":"d","instruction":"do d","depends_on":["S3"]},'
            '{"title":"e","instruction":"do e","depends_on":["S4"]},'
            '{"title":"f","instruction":"do f","depends_on":["S5"]}'
            "]} "
        )
        agent = _PlannerAgentStub(payload, max_steps=5)
        result = agent.plan_task(user_text="demo", history=[])

        self.assertEqual(len(result.steps), 5)
        self.assertEqual([step.step_id for step in result.steps], ["S1", "S2", "S3", "S4", "S5"])


if __name__ == "__main__":
    unittest.main()
