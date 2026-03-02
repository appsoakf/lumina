import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capabilities.registry import build_default_registry
from core.orchestrator.task_shortcuts import execute_task_shortcut, parse_task_shortcut
from core.utils.trace_logger import TraceLogger
from core.protocols import TaskState
from core.tasks.manager import TaskManager
from core.tasks.store import TaskStore


class _FakeChatAgent:
    def reply_with_task_result(self, user_text, executor_output, history):
        _ = user_text, history
        return executor_output


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

    def test_task_shortcut_parse(self):
        self.assertEqual(parse_task_shortcut("/cancel"), "cancel")
        self.assertEqual(parse_task_shortcut(" /RETRY "), "retry")
        self.assertIsNone(parse_task_shortcut("/unknown"))

    def test_task_manager_get_current_task_prefers_running(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        running = manager.create_task(session_id="s1", user_text="running task")
        manager.set_state(running.task_id, TaskState.RUNNING)

        manager.create_task(session_id="s1", user_text="new pending task")

        current = manager.get_current_task("s1")
        self.assertIsNotNone(current)
        self.assertEqual(current.task_id, running.task_id)
        self.assertEqual(current.state, TaskState.RUNNING)

    def test_task_shortcut_cancel_current_task(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        chat_agent = _FakeChatAgent()

        task = manager.create_task(session_id="s1", user_text="to cancel")
        manager.set_state(task.task_id, TaskState.RUNNING)

        result = execute_task_shortcut(
            user_text="/cancel",
            session_id="s1",
            history=[],
            task_manager=manager,
            chat_agent=chat_agent,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.meta.get("task_command"), "cancel")
        self.assertEqual(result.meta.get("task_id"), task.task_id)
        self.assertEqual(manager.get_task(task.task_id).state, TaskState.CANCELLED)

    def test_task_shortcut_retry_latest_retryable_task(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        chat_agent = _FakeChatAgent()

        first = manager.create_task(session_id="s1", user_text="first")
        manager.set_state(first.task_id, TaskState.FAILED, error={"code": "X"})

        second = manager.create_task(session_id="s1", user_text="second")
        manager.set_state(second.task_id, TaskState.CANCELLED)

        result = execute_task_shortcut(
            user_text="/retry",
            session_id="s1",
            history=[],
            task_manager=manager,
            chat_agent=chat_agent,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.meta.get("task_command"), "retry")
        self.assertEqual(result.meta.get("task_id"), second.task_id)
        retried = manager.get_task(second.task_id)
        self.assertEqual(retried.state, TaskState.PENDING)
        self.assertEqual(retried.step_results, [])


if __name__ == "__main__":
    unittest.main()
