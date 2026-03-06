import shutil
import sys
import time
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capabilities.registry import build_default_registry
from core.orchestrator.langgraph_task_runner import LangGraphTaskRunner
from core.utils.trace_logger import TraceLogger
from core.protocols import CriticResult, ExecutorRunResult, PlanItem, PlanResult, TaskState
from core.tasks.manager import TaskManager
from core.tasks.store import TaskStore


class _FakeExecutorAgent:
    def __init__(self, failed_steps=None, need_info_steps=None):
        self.failed_steps = set(failed_steps or [])
        self.need_info_steps = set(need_info_steps or [])
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
        if step_id in self.need_info_steps:
            return ExecutorRunResult(
                output_text=(
                    "步骤状态: 需补充信息\n"
                    "结果摘要: 用户未提供足够约束，无法继续该步骤。\n"
                    "关键依据:\n无\n"
                    "产出详情:\n- 当前缺少必要输入。\n"
                    "限制与风险:\n- 下游步骤输入将为空。\n"
                    "下一步建议:\n- 询问用户补充预算与位置偏好"
                ),
                tool_events=[],
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

    def test_task_manager_reset_for_replan_from_failed(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        t = manager.create_task(session_id="s1", user_text="needs reset")
        self.assertTrue(manager.set_state(t.task_id, TaskState.RUNNING))
        self.assertTrue(manager.set_state(t.task_id, TaskState.FAILED, error={"code": "X"}))
        self.assertTrue(manager.reset_task_for_replan(t.task_id))
        reset = manager.get_task(t.task_id)
        self.assertEqual(reset.state, TaskState.PENDING)
        self.assertEqual(reset.step_results, [])
        self.assertIsNone(reset.error)

    def test_task_manager_rejects_invalid_transition(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        t = manager.create_task(session_id="s1", user_text="guard transitions")
        self.assertTrue(manager.set_state(t.task_id, TaskState.RUNNING))
        self.assertTrue(manager.set_state(t.task_id, TaskState.FAILED, error={"code": "X"}))

        # Once failed, it should not be overwritten by finalize-style success.
        self.assertFalse(manager.set_state(t.task_id, TaskState.SUCCEEDED))
        self.assertEqual(manager.get_task(t.task_id).state, TaskState.FAILED)

    def test_task_manager_waiting_resume_operations(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)

        task = manager.create_task(session_id="s1", user_text="waiting")
        self.assertTrue(manager.set_state(task.task_id, TaskState.RUNNING))
        self.assertTrue(
            manager.set_waiting_input(
                task.task_id,
                waiting_for_input={"pending_step_id": "S1", "clarify_question": "请补充预算"},
                task_snapshot={"nodes": []},
                error={"code": "TASK_NEED_USER_INPUT", "message": "need input", "retryable": True},
            )
        )

        waiting_task = manager.get_waiting_task("s1")
        self.assertIsNotNone(waiting_task)
        self.assertEqual(waiting_task.task_id, task.task_id)

        resumed_task, resumed_ok, payload = manager.resume_waiting_task(task.task_id, "预算200")
        self.assertTrue(resumed_ok)
        self.assertIsNotNone(resumed_task)
        self.assertIsNotNone(payload)
        self.assertEqual(resumed_task.state, TaskState.RUNNING)

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

    def test_langgraph_runner_need_info_blocks_downstream_steps_under_fail_fast(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        task = manager.create_task(session_id="s1", user_text="demo")
        manager.set_state(task.task_id, TaskState.RUNNING)

        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="collect", instruction="collect input"),
                PlanItem(step_id="S2", title="search", instruction="search", depends_on=["S1"]),
            ],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        runner = LangGraphTaskRunner(
            task_manager=manager,
            build_step_input=lambda **kwargs: f"当前步骤: {kwargs['step_id']}",
        )
        executor_agent = _FakeExecutorAgent(need_info_steps={"S1"})

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
        self.assertEqual(executor_agent.calls, ["S1"])
        self.assertEqual(state_map["S1"], "waiting_user_input")
        self.assertEqual(state_map["S2"], "pending")
        self.assertIsNone(result.first_error)
        self.assertIsNotNone(result.waiting_for_input)
        self.assertEqual(result.waiting_for_input.get("pending_step_id"), "S1")
        self.assertEqual(manager.get_task(task.task_id).state, TaskState.WAITING_USER_INPUT)

    def test_langgraph_runner_can_resume_waiting_task_with_same_task_id(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        task = manager.create_task(session_id="s1", user_text="demo")
        manager.set_state(task.task_id, TaskState.RUNNING)

        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="collect", instruction="collect input"),
                PlanItem(step_id="S2", title="search", instruction="search", depends_on=["S1"]),
            ],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        runner = LangGraphTaskRunner(
            task_manager=manager,
            build_step_input=lambda **kwargs: f"当前步骤: {kwargs['step_id']}",
        )

        first = runner.run(
            user_text="demo",
            history=[],
            session_id="s1",
            task_id=task.task_id,
            planner_agent=_FakePlannerAgent(plan),
            executor_agent=_FakeExecutorAgent(need_info_steps={"S1"}),
            critic_agent=_FakeCriticAgent(),
        )
        self.assertIsNotNone(first.waiting_for_input)
        persisted = manager.get_task(task.task_id)
        self.assertEqual(persisted.state, TaskState.WAITING_USER_INPUT)

        resumed_task, resumed_ok, waiting_payload = manager.resume_waiting_task(task.task_id, "预算200，东城区")
        self.assertTrue(resumed_ok)
        self.assertIsNotNone(resumed_task)
        self.assertIsNotNone(waiting_payload)
        self.assertEqual(resumed_task.state, TaskState.RUNNING)

        second_executor = _FakeExecutorAgent()
        second = runner.run(
            user_text="demo",
            history=[],
            session_id="s1",
            task_id=task.task_id,
            planner_agent=_FakePlannerAgent(plan),
            executor_agent=second_executor,
            critic_agent=_FakeCriticAgent(),
            resume_plan_result=plan,
            resume_snapshot=manager.get_task(task.task_id).task_snapshot,
            resume_waiting_payload=waiting_payload,
            resume_user_reply="预算200，东城区",
        )

        state_map = {item["step_id"]: item["state"] for item in second.task_snapshot["nodes"]}
        self.assertEqual(state_map["S1"], "succeeded")
        self.assertEqual(state_map["S2"], "succeeded")
        self.assertIsNone(second.waiting_for_input)
        self.assertIsNone(second.first_error)
        self.assertEqual(second_executor.calls, ["S1", "S2"])
        self.assertEqual(manager.get_task(task.task_id).state, TaskState.SUCCEEDED)

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

    def test_langgraph_runner_caps_max_parallelism_to_two(self):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        task = manager.create_task(session_id="s1", user_text="demo")
        manager.set_state(task.task_id, TaskState.RUNNING)

        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="s1", instruction="do s1"),
                PlanItem(step_id="S2", title="s2", instruction="do s2"),
                PlanItem(step_id="S3", title="s3", instruction="do s3"),
            ],
            graph_policy={"max_parallelism": 5, "fail_fast": False},
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
            executor_agent=_FakeExecutorAgent(),
            critic_agent=_FakeCriticAgent(),
        )

        self.assertEqual(result.task_snapshot["policy"].get("max_parallelism"), 2)

if __name__ == "__main__":
    unittest.main()
