import shutil
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capabilities.registry import build_default_registry
from core.orchestrator.langgraph_task_runner import LangGraphTaskRunner
from core.orchestrator.task_shortcuts import execute_task_shortcut, parse_task_shortcut
from core.utils.trace_logger import TraceLogger
from core.protocols import CriticResult, ExecutorRunResult, PlanItem, PlanResult, TaskState
from core.tasks.manager import TaskManager
from core.tasks.store import TaskStore


class _FakeChatAgent:
    def reply_with_task_result(self, user_text, executor_output, history):
        _ = user_text, history
        return executor_output


class _FakeExecutorAgent:
    def __init__(self, failed_steps=None):
        self.failed_steps = set(failed_steps or [])
        self.calls = []

    def run_task(self, user_text, history, session_id):
        _ = history, session_id
        marker = "当前步骤:"
        step_id = "UNKNOWN"
        if marker in user_text:
            step_id = user_text.split(marker, 1)[1].strip().split(" ", 1)[0]
        self.calls.append(step_id)
        if step_id in self.failed_steps:
            return ExecutorRunResult(
                output_text=f"{step_id} failed",
                tool_events=[],
                error={"code": "STEP_FAILED", "message": step_id, "retryable": True},
            )
        return ExecutorRunResult(output_text=f"{step_id} ok", tool_events=[])


class _DelayedExecutorAgent(_FakeExecutorAgent):
    def __init__(self, failed_steps=None, delays=None):
        super().__init__(failed_steps=failed_steps)
        self.delays = dict(delays or {})

    def run_task(self, user_text, history, session_id):
        marker = "当前步骤:"
        step_id = "UNKNOWN"
        if marker in user_text:
            step_id = user_text.split(marker, 1)[1].strip().split(" ", 1)[0]
        delay = float(self.delays.get(step_id, 0.0))
        if delay > 0:
            time.sleep(delay)
        return super().run_task(user_text, history, session_id)


class _FakePlannerAgent:
    def __init__(self, plan_result: PlanResult):
        self.plan_result = plan_result

    def plan_task(self, user_text, history):
        _ = user_text, history
        return self.plan_result


class _FakeCriticAgent:
    def review_task(self, user_text, plan_result, execution_graph):
        _ = user_text, plan_result, execution_graph
        return CriticResult(quality="pass", summary="ok")


class TaskFlowRegressionTests(unittest.TestCase):
    def setUp(self):
        runtime_base = PROJECT_ROOT / "runtime"
        runtime_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = runtime_base / f"lumina-test-{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

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
        self.assertTrue(manager.set_state(t.task_id, TaskState.RUNNING))
        self.assertTrue(manager.cancel_task(t.task_id))

        cancelled = manager.get_task(t.task_id)
        self.assertEqual(cancelled.state, TaskState.CANCELLED)

        retried = manager.retry_task(t.task_id)
        self.assertIsNotNone(retried)
        self.assertEqual(retried.state, TaskState.PENDING)
        self.assertEqual(retried.step_results, [])

    def test_task_manager_rejects_invalid_transition(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        t = manager.create_task(session_id="s1", user_text="guard transitions")
        self.assertTrue(manager.set_state(t.task_id, TaskState.RUNNING))
        self.assertTrue(manager.cancel_task(t.task_id))

        # Once cancelled, it should not be overwritten by finalize-style success.
        self.assertFalse(manager.set_state(t.task_id, TaskState.SUCCEEDED))
        self.assertEqual(manager.get_task(t.task_id).state, TaskState.CANCELLED)

    def test_task_manager_atomic_shortcut_operations(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        running = manager.create_task(session_id="s1", user_text="running")
        self.assertTrue(manager.set_state(running.task_id, TaskState.RUNNING))
        pending = manager.create_task(session_id="s1", user_text="pending")
        self.assertEqual(pending.state, TaskState.PENDING)

        cancel_target, cancelled = manager.cancel_current_task("s1")
        self.assertTrue(cancelled)
        self.assertIsNotNone(cancel_target)
        self.assertEqual(cancel_target.task_id, running.task_id)

        first = manager.create_task(session_id="s1", user_text="first")
        self.assertTrue(manager.set_state(first.task_id, TaskState.FAILED, error={"code": "X"}))
        second = manager.create_task(session_id="s1", user_text="second")
        self.assertTrue(manager.set_state(second.task_id, TaskState.CANCELLED))

        retry_target, retried = manager.retry_latest_task("s1")
        self.assertTrue(retried)
        self.assertIsNotNone(retry_target)
        self.assertEqual(retry_target.task_id, second.task_id)
        self.assertEqual(manager.get_task(second.task_id).state, TaskState.PENDING)

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

    def test_langgraph_runner_blocks_downstream_steps(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        task = manager.create_task(session_id="s1", user_text="demo")
        manager.set_state(task.task_id, TaskState.RUNNING)

        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="s1", instruction="do s1"),
                PlanItem(step_id="S2", title="s2", instruction="do s2", depends_on=["S1"]),
                PlanItem(step_id="S3", title="s3", instruction="do s3", depends_on=["S1"]),
                PlanItem(step_id="S4", title="s4", instruction="do s4", depends_on=["S2", "S3"]),
            ],
            graph_policy={"max_parallelism": 2, "fail_fast": False},
        )
        runner = LangGraphTaskRunner(
            task_manager=manager,
            build_step_input=lambda **kwargs: f"当前步骤: {kwargs['step_id']}",
        )
        result = runner.run(
            user_text="demo",
            history=[],
            session_id="s1",
            task_id=task.task_id,
            planner_agent=_FakePlannerAgent(plan),
            executor_agent=_FakeExecutorAgent(failed_steps={"S2"}),
            critic_agent=_FakeCriticAgent(),
        )

        state_map = {item["step_id"]: item["state"] for item in result.task_snapshot["nodes"]}
        self.assertEqual(state_map["S2"], "failed")
        self.assertEqual(state_map["S3"], "succeeded")
        self.assertEqual(state_map["S4"], "blocked")

    def test_langgraph_runner_non_fail_fast_keeps_running_ready_nodes(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        task = manager.create_task(session_id="s1", user_text="demo")
        manager.set_state(task.task_id, TaskState.RUNNING)

        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="s1", instruction="do s1"),
                PlanItem(step_id="S2", title="s2", instruction="do s2", depends_on=["S1"]),
                PlanItem(step_id="S3", title="s3", instruction="do s3", depends_on=["S1"]),
            ],
            graph_policy={"max_parallelism": 1, "fail_fast": False},
        )
        runner = LangGraphTaskRunner(
            task_manager=manager,
            build_step_input=lambda **kwargs: f"当前步骤: {kwargs['step_id']}",
        )
        executor_agent = _FakeExecutorAgent(failed_steps={"S2"})

        result = runner.run(
            user_text="demo",
            history=[],
            session_id="s1",
            task_id=task.task_id,
            planner_agent=_FakePlannerAgent(plan),
            executor_agent=executor_agent,
            critic_agent=_FakeCriticAgent(),
        )

        state_map = {item["step_id"]: item["state"] for item in result.task_snapshot["nodes"]}
        self.assertEqual(executor_agent.calls, ["S1", "S2", "S3"])
        self.assertEqual(state_map["S1"], "succeeded")
        self.assertEqual(state_map["S2"], "failed")
        self.assertEqual(state_map["S3"], "succeeded")
        self.assertIsNotNone(result.first_error)
        self.assertEqual(len(result.step_results), 3)

    def test_langgraph_runner_executes_ready_batch_in_parallel(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        task = manager.create_task(session_id="s1", user_text="demo")
        manager.set_state(task.task_id, TaskState.RUNNING)

        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="s1", instruction="do s1"),
                PlanItem(step_id="S2", title="s2", instruction="do s2", depends_on=["S1"]),
                PlanItem(step_id="S3", title="s3", instruction="do s3", depends_on=["S1"]),
            ],
            graph_policy={"max_parallelism": 2, "fail_fast": False},
        )
        runner = LangGraphTaskRunner(
            task_manager=manager,
            build_step_input=lambda **kwargs: f"当前步骤: {kwargs['step_id']}",
        )
        executor_agent = _DelayedExecutorAgent(delays={"S2": 0.25, "S3": 0.25})

        started = time.perf_counter()
        result = runner.run(
            user_text="demo",
            history=[],
            session_id="s1",
            task_id=task.task_id,
            planner_agent=_FakePlannerAgent(plan),
            executor_agent=executor_agent,
            critic_agent=_FakeCriticAgent(),
        )
        elapsed = time.perf_counter() - started

        self.assertEqual([item["step_id"] for item in result.step_results], ["S1", "S2", "S3"])
        self.assertLess(elapsed, 0.45)

    def test_langgraph_runner_keeps_independent_steps_parallel_without_auto_serialization(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        task = manager.create_task(session_id="s1", user_text="demo")
        manager.set_state(task.task_id, TaskState.RUNNING)

        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="s1", instruction="do s1"),
                PlanItem(step_id="S2", title="s2", instruction="do s2"),
                PlanItem(step_id="S3", title="s3", instruction="do s3", depends_on=["S1", "S2"]),
            ],
            graph_policy={"max_parallelism": 2, "fail_fast": False},
        )
        runner = LangGraphTaskRunner(
            task_manager=manager,
            build_step_input=lambda **kwargs: f"当前步骤: {kwargs['step_id']}",
        )
        executor_agent = _DelayedExecutorAgent(delays={"S1": 0.25, "S2": 0.25})

        started = time.perf_counter()
        result = runner.run(
            user_text="demo",
            history=[],
            session_id="s1",
            task_id=task.task_id,
            planner_agent=_FakePlannerAgent(plan),
            executor_agent=executor_agent,
            critic_agent=_FakeCriticAgent(),
        )
        elapsed = time.perf_counter() - started

        self.assertEqual([item["step_id"] for item in result.step_results], ["S1", "S2", "S3"])
        # S1/S2 should run in the same batch; if auto-serialized, elapsed would be notably longer.
        self.assertLess(elapsed, 0.55)

    def test_task_manager_race_cancel_vs_finalize_is_consistent(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        for i in range(30):
            task = manager.create_task(session_id="race", user_text=f"race-{i}")
            self.assertTrue(manager.set_state(task.task_id, TaskState.RUNNING))

            barrier = threading.Barrier(3)
            results = {"cancel_ok": None, "succeed_ok": None}

            def _cancel():
                barrier.wait()
                results["cancel_ok"] = manager.cancel_task(task.task_id)

            def _finalize_success():
                barrier.wait()
                results["succeed_ok"] = manager.set_state(task.task_id, TaskState.SUCCEEDED)

            t1 = threading.Thread(target=_cancel)
            t2 = threading.Thread(target=_finalize_success)
            t1.start()
            t2.start()
            barrier.wait()
            t1.join()
            t2.join()

            final = manager.get_task(task.task_id)
            self.assertIsNotNone(final)
            self.assertIn(final.state, {TaskState.CANCELLED, TaskState.SUCCEEDED})

            # Exactly one transition wins the race.
            self.assertNotEqual(bool(results["cancel_ok"]), bool(results["succeed_ok"]))
            if final.state == TaskState.CANCELLED:
                self.assertTrue(results["cancel_ok"])
                self.assertFalse(results["succeed_ok"])
            else:
                self.assertTrue(results["succeed_ok"])
                self.assertFalse(results["cancel_ok"])

    def test_task_manager_cancel_current_task_is_atomic_under_concurrency(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        task = manager.create_task(session_id="s1", user_text="cancel concurrently")
        self.assertTrue(manager.set_state(task.task_id, TaskState.RUNNING))

        barrier = threading.Barrier(3)
        outputs = []

        def _cancel_once():
            barrier.wait()
            selected, ok = manager.cancel_current_task("s1")
            outputs.append((selected.task_id if selected else None, ok))

        t1 = threading.Thread(target=_cancel_once)
        t2 = threading.Thread(target=_cancel_once)
        t1.start()
        t2.start()
        barrier.wait()
        t1.join()
        t2.join()

        self.assertEqual(len(outputs), 2)
        self.assertEqual(sum(1 for _, ok in outputs if ok), 1)
        self.assertEqual(sum(1 for task_id, _ in outputs if task_id == task.task_id), 1)
        self.assertEqual(sum(1 for task_id, _ in outputs if task_id is None), 1)
        self.assertEqual(manager.get_task(task.task_id).state, TaskState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
