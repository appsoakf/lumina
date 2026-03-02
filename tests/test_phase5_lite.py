import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capabilities.registry import build_default_registry
from core.integration.trace_logger import TraceLogger
from core.protocols import TaskState
from core.tasks.manager import TaskManager
from core.tasks.store import TaskStore
from core.workflows.travel_workflow import TravelWorkflow


class Phase5LiteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="lumina-test-"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_capability_registry_default_resolution(self):
        reg = build_default_registry()
        self.assertEqual(reg.resolve_agent("chat"), "chat_agent")
        self.assertEqual(reg.resolve_agent("task_execution"), "executor_agent")
        self.assertEqual(reg.resolve_agent("task_review"), "critic_agent")

    def test_task_manager_create_and_persist(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        t = manager.create_task(session_id="s1", user_text="test task")
        self.assertEqual(t.state, TaskState.PENDING)

        manager.set_plan(t.task_id, {"goal": "x"})
        manager.append_step_result(t.task_id, {"step_id": "S1", "state": "succeeded"})
        manager.set_state(t.task_id, TaskState.SUCCEEDED)

        loaded = store.load(t.task_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.state, TaskState.SUCCEEDED)
        self.assertEqual(loaded.plan.get("goal"), "x")
        self.assertEqual(len(loaded.step_results), 1)

    def test_task_manager_cancel_and_retry(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        t = manager.create_task(session_id="s1", user_text="cancel me")
        manager.set_state(t.task_id, TaskState.RUNNING)
        self.assertTrue(manager.cancel_task(t.task_id))

        cancelled = manager.get_task(t.task_id)
        self.assertEqual(cancelled.state, TaskState.CANCELLED)

        retried = manager.retry_task(t.task_id)
        self.assertIsNotNone(retried)
        self.assertEqual(retried.state, TaskState.PENDING)
        self.assertEqual(retried.step_results, [])

    def test_travel_workflow_parse_constraints(self):
        wf = TravelWorkflow()
        text = "请帮我规划一个去北京4天旅游计划，预算5000元，2人，喜欢美食和夜景"
        c = wf.parse_constraints(text)

        self.assertEqual(c.destination, "北京")
        self.assertEqual(c.days, 4)
        self.assertEqual(c.budget_cny, 5000)
        self.assertIn("美食", c.preferences)
        self.assertIn("夜景", c.preferences)

    def test_travel_workflow_missing_and_clarification(self):
        wf = TravelWorkflow()
        c = wf.parse_constraints("请给我做个旅游计划")
        missing = wf.missing_required_fields(c)
        self.assertIn("destination", missing)
        self.assertIn("days", missing)

        q = wf.build_clarification_request(c, missing)
        self.assertIn("目的地", q)
        self.assertIn("出行天数", q)

    def test_travel_workflow_plan_template(self):
        wf = TravelWorkflow()
        c = wf.parse_constraints("我想去北京3天旅游")
        plan = wf.build_plan("我想去北京3天旅游", c)

        self.assertGreaterEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].step_id, "S1")

    def test_trace_logger_async_write(self):
        trace_dir = self.temp_dir / "traces"
        logger = TraceLogger(trace_dir=str(trace_dir), session_id="ut1")
        logger.log("round_start", {"round": 1})
        logger.log("round_end", {"round": 1, "cost_sec": 1.2})
        logger.close()

        trace_file = trace_dir / "trace-ut1.jsonl"
        self.assertTrue(trace_file.exists())
        content = trace_file.read_text(encoding="utf-8")
        self.assertIn("round_start", content)
        self.assertIn("round_end", content)


if __name__ == "__main__":
    unittest.main()
