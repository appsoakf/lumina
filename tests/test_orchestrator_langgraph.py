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


class _CancelOnFirstExecutor:
    def __init__(self, manager: TaskManager):
        self._manager = manager
        self.calls = []

    def run_task(self, user_text, history, session_id):
        _ = history
        step_id = _extract_step_id(user_text)
        self.calls.append(step_id)
        if step_id == "S1":
            task = self._manager.get_current_task(session_id=session_id)
            if task is not None:
                self._manager.cancel_task(task.task_id)
        return ExecutorRunResult(output_text=f"{step_id} ok", tool_events=[])


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

    def test_cancelled_task_is_preserved(self):
        plan = PlanResult(
            goal="demo",
            steps=[
                PlanItem(step_id="S1", title="s1", instruction="do s1"),
                PlanItem(step_id="S2", title="s2", instruction="do s2", depends_on=["S1"]),
            ],
            graph_policy={"max_parallelism": 1, "fail_fast": True},
        )
        store = TaskStore(base_dir=str(self.temp_dir / "tasks"))
        manager = TaskManager(store=store)
        memory = _MemoryStub()
        executor = _CancelOnFirstExecutor(manager)
        orchestrator = Orchestrator(
            chat_agent=_TaskChatAgent(),
            planner_agent=_PlannerStub(plan),
            executor_agent=executor,
            critic_agent=_CriticStub(),
            task_manager=manager,
            memory_service=memory,
        )

        result = orchestrator.handle_user_message(
            user_text="请执行任务",
            session_id="s1",
        )

        task_id = str(result.meta.get("task_id"))
        task = manager.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.state, TaskState.CANCELLED)
        self.assertIsNotNone(result.executor_result.error)
        self.assertEqual(result.executor_result.error.get("code"), "TASK_CANCELLED")
        node_states = {n["step_id"]: n["state"] for n in result.meta.get("task_graph", {}).get("nodes", [])}
        self.assertEqual(node_states.get("S2"), "cancelled")

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


if __name__ == "__main__":
    unittest.main()
