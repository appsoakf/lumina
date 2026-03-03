import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agentic.planner_agent import PlannerAgent


class _PlannerAgentStub(PlannerAgent):
    def __init__(self, model_output: str):
        self.max_steps = 6
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

    def test_plan_task_fallback_when_output_is_not_json(self):
        agent = _PlannerAgentStub("not json")
        result = agent.plan_task(user_text="demo", history=[])

        self.assertEqual(result.goal, "demo")
        self.assertEqual(len(result.steps), 3)
        self.assertIsNotNone(result.error)
        self.assertIn("Planner failed", str(result.error.get("message", "")))


if __name__ == "__main__":
    unittest.main()
