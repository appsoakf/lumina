import logging
from typing import Dict, List, Optional

from core.agentic.chat_agent import ChatAgent
from core.agentic.critic_agent import CriticAgent
from core.agentic.executor_agent import ExecutorAgent
from core.agentic.planner_agent import PlannerAgent
from core.capabilities import CapabilityRegistry, build_default_registry
from core.memory import MemoryService
from core.orchestrator.task_graph import TaskGraph
from core.orchestrator.task_shortcuts import execute_task_shortcut
from core.protocols import CriticResult, ExecutorRunResult, OrchestrationResult, RoutingIntent, TaskState
from core.tasks import TaskManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrator: dynamic routing + task lifecycle + memory system."""

    def __init__(
        self,
        chat_agent: Optional[ChatAgent] = None,
        planner_agent: Optional[PlannerAgent] = None,
        executor_agent: Optional[ExecutorAgent] = None,
        critic_agent: Optional[CriticAgent] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        task_manager: Optional[TaskManager] = None,
        memory_service: Optional[MemoryService] = None,
    ):
        self._agents = {
            "chat_agent": chat_agent or ChatAgent(),
            "planner_agent": planner_agent or PlannerAgent(),
            "executor_agent": executor_agent or ExecutorAgent(),
            "critic_agent": critic_agent or CriticAgent(),
        }
        self.capabilities = capability_registry or build_default_registry()
        self._task_manager = task_manager or TaskManager()
        self._memory = memory_service or MemoryService()

    def _resolve_agent(self, capability: str):
        agent_name = self.capabilities.resolve_agent(capability)
        agent = self._agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent not initialized: {agent_name}")
        return agent_name, agent

    def _augment_history_with_memory(self, history: List[Dict[str, str]], user_id: str, query: str) -> List[Dict[str, str]]:
        context = self._memory.build_context(user_id=user_id, query=query)
        if not context:
            return list(history)
        mem_msg = {
            "role": "system",
            "content": (
                "以下是用户记忆上下文，可用于提升个性化和连续性；"
                "如与当前用户明确指令冲突，以当前指令为准。\n" + context
            ),
        }
        return [mem_msg] + list(history[-12:])

    def _record_memory(self, session_id: str, user_id: str, user_text: str, final_reply: str, meta: Dict) -> None:
        try:
            self._memory.ingest_turn(
                session_id=session_id,
                user_id=user_id,
                user_text=user_text,
                assistant_reply=final_reply,
                meta=meta,
            )
        except Exception as exc:
            logger.warning(f"Memory ingest skipped due to error: {exc}")

    def record_session_round(
        self,
        session_id: str,
        user_text: str,
        assistant_reply: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self._memory.record_session_round(
            session_id=session_id,
            user_text=user_text,
            assistant_reply=assistant_reply,
            metadata=metadata,
        )

    def _build_step_task_input(self, user_text: str, graph: TaskGraph, step_id: str) -> str:
        node = next(n for n in graph.nodes if n.step_id == step_id)
        completed = graph.completed_context()
        return (
            f"总目标: {user_text}\n"
            f"当前步骤: {node.step_id} {node.title}\n"
            f"步骤指令: {node.instruction}\n"
            f"已完成上下文:\n{completed if completed else '无'}\n"
            "请完成当前步骤并输出结果摘要。"
        )

    def _run_plan_steps(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        user_id: str,
        session_id: str,
        task_id: str,
        graph: TaskGraph,
    ) -> Dict[str, object]:
        _, executor_agent = self._resolve_agent("task_execution")

        all_tool_events = []
        first_error = None
        step_results = []

        for node in graph.pending_nodes():
            task = self._task_manager.get_task(task_id)
            if task and task.state == TaskState.CANCELLED:
                cancel_error = {"code": "TASK_CANCELLED", "message": "Task cancelled by user", "retryable": True}
                first_error = first_error or cancel_error
                break

            graph.mark_running(node.step_id)
            step_input = self._build_step_task_input(user_text=user_text, graph=graph, step_id=node.step_id)
            run_result = executor_agent.run_task(
                user_text=step_input,
                history=history,
                user_id=user_id,
                session_id=session_id,
            )
            all_tool_events.extend(run_result.tool_events)

            step_result = {
                "step_id": node.step_id,
                "title": node.title,
                "state": "failed" if run_result.error else "succeeded",
                "output_text": run_result.output_text,
                "error": run_result.error,
            }
            step_results.append(step_result)
            self._task_manager.append_step_result(task_id, step_result)

            if run_result.error:
                graph.mark_failed(
                    step_id=node.step_id,
                    output_text=run_result.output_text,
                    tool_events=run_result.tool_events,
                    error=run_result.error,
                )
                if first_error is None:
                    first_error = run_result.error
            else:
                graph.mark_done(
                    step_id=node.step_id,
                    output_text=run_result.output_text,
                    tool_events=run_result.tool_events,
                )

        return {
            "all_tool_events": all_tool_events,
            "first_error": first_error,
            "step_results": step_results,
        }

    def _compose_general_executor_output(self, graph: TaskGraph, critic: CriticResult) -> str:
        lines = [f"任务目标: {graph.goal}", "", "执行结果:"]
        for node in graph.nodes:
            state_text = {
                "succeeded": "成功",
                "failed": "失败",
                "running": "执行中",
                "pending": "待执行",
            }.get(node.state.value, node.state.value)
            lines.append(f"- [{node.step_id}] {node.title} ({state_text})")
            if node.output_text:
                lines.append(f"  结果: {node.output_text}")
            if node.error:
                lines.append(f"  错误: {node.error.get('code')} | {node.error.get('message')}")

        if critic.summary:
            lines.extend(["", f"评审结论: {critic.summary}"])
        if critic.quality == "revise" and critic.suggestions:
            lines.append("改进建议:")
            for s in critic.suggestions:
                lines.append(f"- {s}")
        return "\n".join(lines)

    def _handle_task_shortcut(
        self,
        user_text: str,
        session_id: str,
        history: List[Dict[str, str]],
    ) -> Optional[OrchestrationResult]:
        _, chat_agent = self._resolve_agent("chat")
        return execute_task_shortcut(
            user_text=user_text,
            session_id=session_id,
            history=history,
            task_manager=self._task_manager,
            chat_agent=chat_agent,
        )

    def handle_user_message(
        self,
        user_text: str,
        session_id: str,
        user_id: str,
    ) -> OrchestrationResult:
        history = self._memory.get_recent_history(session_id=session_id)

        # task shortcuts, e.g. /cancel and /retry
        shortcut_result = self._handle_task_shortcut(
            user_text=user_text,
            session_id=session_id,
            history=history,
        )
        if shortcut_result is not None:
            self._record_memory(session_id, user_id, user_text, shortcut_result.final_reply, shortcut_result.meta)
            return shortcut_result

        enriched_history = self._augment_history_with_memory(history=history, user_id=user_id, query=user_text)

        # 意图识别
        _, chat_agent = self._resolve_agent("chat")
        intent = chat_agent.classify_intent(user_text=user_text, history=enriched_history)

        if intent == RoutingIntent.CHAT:
            final_reply = chat_agent.reply_chat(user_text=user_text, history=enriched_history)
            meta = {"agent_chain": ["chat_agent"], "task_mode": False}
            self._record_memory(session_id, user_id, user_text, final_reply, meta)
            return OrchestrationResult(
                intent=RoutingIntent.CHAT,
                final_reply=final_reply,
                executor_result=None,
                meta=meta,
            )

        # task mode
        task = self._task_manager.create_task(session_id=session_id, user_text=user_text)
        task_id = task.task_id
        self._task_manager.set_state(task_id, TaskState.RUNNING)

        _, planner_agent = self._resolve_agent("task_planning")
        _, critic_agent = self._resolve_agent("task_review")

        plan_result = planner_agent.plan_task(user_text=user_text, history=enriched_history)

        self._task_manager.set_plan(task_id, plan_result.to_dict())

        graph = TaskGraph.from_plan(plan_result)
        run_info = self._run_plan_steps(
            user_text=user_text,
            history=enriched_history,
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            graph=graph,
        )

        critic_result = critic_agent.review_task(
            user_text=user_text,
            plan_result=plan_result,
            execution_graph=graph.to_dict(),
        )

        executor_output = self._compose_general_executor_output(graph=graph, critic=critic_result)

        executor_result = ExecutorRunResult(
            output_text=executor_output,
            tool_events=run_info["all_tool_events"],
            error=run_info["first_error"],
            step_results=run_info["step_results"],
        )

        current_task = self._task_manager.get_task(task_id)
        if not current_task or current_task.state != TaskState.CANCELLED:
            if run_info["first_error"]:
                self._task_manager.set_state(task_id, TaskState.FAILED, error=run_info["first_error"])
            else:
                self._task_manager.set_state(task_id, TaskState.SUCCEEDED)

        final_reply = chat_agent.reply_with_task_result(
            user_text=user_text,
            executor_output=executor_result.output_text,
            history=enriched_history,
        )

        meta = {
            "task_id": task_id,
            "agent_chain": ["chat_agent", "planner_agent", "executor_agent", "critic_agent", "chat_agent"],
            "task_mode": True,
            "plan": plan_result.to_dict(),
            "task_graph": graph.to_dict(),
            "critic": critic_result.to_dict(),
            "task_error": bool(run_info["first_error"]),
        }

        self._record_memory(session_id, user_id, user_text, final_reply, meta)
        return OrchestrationResult(
            intent=RoutingIntent.TASK,
            final_reply=final_reply,
            executor_result=executor_result,
            meta=meta,
        )
