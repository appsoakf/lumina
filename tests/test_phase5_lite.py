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
