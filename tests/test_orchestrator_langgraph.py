import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import Orchestrator
from core.protocols import CriticResult, ExecutorRunResult, PlanItem, PlanResult, RoutingIntent, TaskState
from core.tasks.manager import TaskManager
from core.tasks.store import TaskStore


def _extract_step_id(user_text: str) -> str:
    marker = "当前步骤:"
    if marker not in user_text:
        return "UNKNOWN"
    return user_text.split(marker, 1)[1].strip().split(" ", 1)[0]


class _MemoryStub:
    def __init__(self):
        self.ingested = []
        self.rounds = []
        self.history = []
        self.closed = 0

    def get_recent_history(self, session_id):
        _ = session_id
        return list(self.history)

    def build_context(self, query):
        _ = query
        return ""

    def ingest_turn(self, session_id, user_text, assistant_reply, meta):
        self.ingested.append(
            {
                "session_id": session_id,
                "user_text": user_text,
                "assistant_reply": assistant_reply,
                "meta": meta,
            }
        )

    def record_session_round(self, session_id, user_text, assistant_reply, metadata=None):
        self.rounds.append(
            {
                "session_id": session_id,
                "user_text": user_text,
                "assistant_reply": assistant_reply,
                "metadata": metadata or {},
            }
        )

    def close(self):
        self.closed += 1


class _TaskChatAgent:
    def classify_intent(self, user_text, history):
        _ = user_text, history
        return RoutingIntent.TASK

    def reply_chat(self, user_text, history):
        _ = user_text, history
        return '{"emotion":"平静","intensity":1}\nchat'

    def reply_with_task_result(self, user_text, executor_output, history):
        _ = user_text, history
        return '{"emotion":"平静","intensity":1}\n' + executor_output


class _PlannerStub:
    def __init__(self, plan_result: PlanResult):
        self.plan_result = plan_result

    def plan_task(self, user_text, history):
        _ = user_text, history
        return self.plan_result


class _CriticStub:
    def review_task(self, user_text, plan_result, execution_graph):
        _ = user_text, plan_result, execution_graph
        return CriticResult(quality="pass", summary="ok")


class _ExecutorStub:
    def __init__(self, failed_steps=None):
        self.failed_steps = set(failed_steps or [])
        self.calls = []

    def run_task(self, user_text, history, session_id):
        _ = history, session_id
        step_id = _extract_step_id(user_text)
        self.calls.append(step_id)
        if step_id in self.failed_steps:
            return ExecutorRunResult(
                output_text=f"{step_id} failed",
                tool_events=[],
                error={"code": "STEP_FAILED", "message": step_id, "retryable": True},
            )
        return ExecutorRunResult(output_text=f"{step_id} ok", tool_events=[])


class _NeedInfoThenSuccessExecutor:
    def __init__(self):
        self.calls = []
        self.need_info_emitted = False

    def run_task(self, user_text, history, session_id):
        _ = history, session_id
        step_id = _extract_step_id(user_text)
        self.calls.append(step_id)
        if step_id == "S1" and not self.need_info_emitted:
            self.need_info_emitted = True
            return ExecutorRunResult(
                output_text=(
                    "步骤状态: 需补充信息\n"
                    "结果摘要: 缺少用户预算和区域偏好。\n"
                    "关键依据:\n无\n"
                    "产出详情:\n- 缺少关键约束。\n"
                    "限制与风险:\n- 无法继续筛选。\n"
                    "下一步建议:\n- 请补充预算和所在区域"
                ),
                tool_events=[],
            )
        return ExecutorRunResult(output_text=f"{step_id} ok", tool_events=[])


class _FailThenSuccessExecutor:
    def __init__(self):
        self.failed_once = False
        self.calls = []

    def run_task(self, user_text, history, session_id):
        _ = history, session_id
        step_id = _extract_step_id(user_text)
        self.calls.append(step_id)
        if not self.failed_once:
            self.failed_once = True
            return ExecutorRunResult(
                output_text=f"{step_id} failed once",
                tool_events=[],
                error={"code": "STEP_FAILED", "message": "temporary", "retryable": True},
            )
        return ExecutorRunResult(output_text=f"{step_id} ok", tool_events=[])


class _AlwaysNeedInfoExecutor:
    def __init__(self):
        self.calls = []

    def run_task(self, user_text, history, session_id):
        _ = history, session_id
        step_id = _extract_step_id(user_text)
        self.calls.append(step_id)
        return ExecutorRunResult(
            output_text=(
                "步骤状态: 需补充信息\n"
                "结果摘要: 仍缺少关键偏好信息。\n"
                "关键依据:\n无\n"
                "产出详情:\n- 需要更多约束。\n"
                "限制与风险:\n- 无法继续。\n"
                "下一步建议:\n- 请补充预算和区域"
            ),
            tool_events=[],
        )


class OrchestratorLangGraphTests(unittest.TestCase):
    def setUp(self):
        runtime_base = PROJECT_ROOT / "runtime"
        runtime_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = runtime_base / f"lumina-test-{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_orchestrator(self, plan_result: PlanResult, executor_agent):
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        memory = _MemoryStub()
        orchestrator = Orchestrator(
            chat_agent=_TaskChatAgent(),
            planner_agent=_PlannerStub(plan_result),
            executor_agent=executor_agent,
            critic_agent=_CriticStub(),
            task_manager=manager,
            memory_service=memory,
        )
        return orchestrator, manager

    def test_task_path_succeeds_with_langgraph_runner(self):
        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="s1", instruction="do s1"),
                PlanItem(step_id="S2", title="s2", instruction="do s2", depends_on=["S1"]),
            ],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        orchestrator, manager = self._build_orchestrator(plan, _ExecutorStub())

        result = orchestrator.handle_user_message(
            user_text="请执行任务",
            session_id="s1",
        )

        self.assertEqual(result.intent, RoutingIntent.TASK)
        self.assertIsNotNone(result.executor_result)
        task_id = str(result.meta.get("task_id"))
        task = manager.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.state, TaskState.SUCCEEDED)
        self.assertEqual(len(result.executor_result.step_results), 2)

    def test_slash_command_text_is_treated_as_plain_input(self):
        plan = PlanResult(
            goal="demo",
            steps=[PlanItem(step_id="S1", title="s1", instruction="do s1")],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        orchestrator, manager = self._build_orchestrator(plan, _ExecutorStub())

        result = orchestrator.handle_user_message(
            user_text="/noop",
            session_id="s1",
        )

        self.assertEqual(result.intent, RoutingIntent.TASK)
        self.assertIsNone(result.meta.get("task_command"))
        task_id = str(result.meta.get("task_id"))
        task = manager.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.state, TaskState.SUCCEEDED)

    def test_plan_warning_does_not_force_task_failed(self):
        plan = PlanResult(
            goal="demo",
            steps=[PlanItem(step_id="S1", title="s1", instruction="do s1")],
            error={"code": "INTERNAL_ERROR", "message": "planner fallback", "retryable": True},
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        orchestrator, manager = self._build_orchestrator(plan, _ExecutorStub())

        result = orchestrator.handle_user_message(
            user_text="请执行任务",
            session_id="s1",
        )

        task_id = str(result.meta.get("task_id"))
        task = manager.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.state, TaskState.SUCCEEDED)
        self.assertIsNone(result.executor_result.error)

    def test_orchestrator_close_delegates_memory_close(self):
        plan = PlanResult(
            goal="demo",
            steps=[PlanItem(step_id="S1", title="s1", instruction="do s1")],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        memory = _MemoryStub()
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        orchestrator = Orchestrator(
            chat_agent=_TaskChatAgent(),
            planner_agent=_PlannerStub(plan),
            executor_agent=_ExecutorStub(),
            critic_agent=_CriticStub(),
            task_manager=manager,
            memory_service=memory,
        )

        orchestrator.close()
        self.assertEqual(memory.closed, 1)

    def test_waiting_task_resume_flow_uses_same_task_id(self):
        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="collect", instruction="collect input"),
                PlanItem(step_id="S2", title="search", instruction="search", depends_on=["S1"]),
            ],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        executor = _NeedInfoThenSuccessExecutor()
        orchestrator, manager = self._build_orchestrator(plan, executor)

        first = orchestrator.handle_user_message(
            user_text="请推荐北京烤鸭",
            session_id="s1",
        )
        task_id = str(first.meta.get("task_id"))
        self.assertTrue(first.meta.get("task_waiting_input"))
        self.assertEqual(int(first.meta.get("task_round_count") or 0), 1)
        self.assertEqual(manager.get_task(task_id).state, TaskState.WAITING_USER_INPUT)
        self.assertEqual(first.meta.get("task_waiting_step_id"), "S1")

        second = orchestrator.handle_user_message(
            user_text="预算200，东城区，环境安静",
            session_id="s1",
        )
        self.assertEqual(str(second.meta.get("task_id")), task_id)
        self.assertFalse(bool(second.meta.get("task_waiting_input")))
        self.assertEqual(int(second.meta.get("task_round_count") or 0), 2)
        self.assertEqual(manager.get_task(task_id).state, TaskState.SUCCEEDED)
        self.assertIn("S1", executor.calls)
        self.assertIn("S2", executor.calls)

    def test_orchestrator_replans_retryable_failure_and_converges(self):
        plan = PlanResult(
            goal="demo",
            steps=[PlanItem(step_id="S1", title="only", instruction="do once")],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        executor = _FailThenSuccessExecutor()
        orchestrator, manager = self._build_orchestrator(plan, executor)

        result = orchestrator.handle_user_message(
            user_text="请执行任务",
            session_id="s1",
        )

        task_id = str(result.meta.get("task_id"))
        self.assertEqual(manager.get_task(task_id).state, TaskState.SUCCEEDED)
        self.assertFalse(bool(result.meta.get("task_waiting_input")))
        self.assertEqual(int(result.meta.get("task_replan_count") or 0), 1)
        self.assertGreaterEqual(len(executor.calls), 2)

    def test_orchestrator_marks_not_converged_when_clarify_rounds_exceeded(self):
        plan = PlanResult(
            goal="demo",
            steps=[PlanItem(step_id="S1", title="collect", instruction="collect input")],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        executor = _AlwaysNeedInfoExecutor()
        orchestrator, manager = self._build_orchestrator(plan, executor)
        orchestrator._max_clarify_rounds = 1

        result = orchestrator.handle_user_message(
            user_text="请执行任务",
            session_id="s1",
        )

        task_id = str(result.meta.get("task_id"))
        task = manager.get_task(task_id)
        self.assertEqual(task.state, TaskState.FAILED)
        self.assertFalse(bool(result.meta.get("task_waiting_input")))
        self.assertIsNotNone(result.executor_result.error)
        self.assertEqual(result.executor_result.error.get("code"), "TASK_NOT_CONVERGED")


if __name__ == "__main__":
    unittest.main()
