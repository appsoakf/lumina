import logging
import re
from typing import Dict, List, Optional, Tuple

from core.agentic.chat_agent import ChatAgent
from core.agentic.critic_agent import CriticAgent
from core.agentic.executor_agent import ExecutorAgent
from core.agentic.planner_agent import PlannerAgent
from core.capabilities import CapabilityRegistry, build_default_registry
from core.memory import MemoryService
from core.orchestrator.task_graph import TaskGraph
from core.protocols import CriticResult, ExecutorRunResult, OrchestrationResult, RoutingIntent, TaskState
from core.tasks import TaskManager
from core.workflows import TravelWorkflow

logger = logging.getLogger(__name__)


class LuminaOrchestrator:
    """Phase 6 orchestrator: dynamic routing + task lifecycle + memory system."""

    TASK_ID_RE = re.compile(r"(t-[0-9]{14}-[a-f0-9]{8})", re.IGNORECASE)

    def __init__(
        self,
        chat_agent: Optional[ChatAgent] = None,
        planner_agent: Optional[PlannerAgent] = None,
        executor_agent: Optional[ExecutorAgent] = None,
        critic_agent: Optional[CriticAgent] = None,
        travel_workflow: Optional[TravelWorkflow] = None,
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
        self.travel_workflow = travel_workflow or TravelWorkflow()
        self.capabilities = capability_registry or build_default_registry()
        self.task_manager = task_manager or TaskManager()
        self.memory = memory_service or MemoryService()

    def _resolve_agent(self, capability: str):
        agent_name = self.capabilities.resolve_agent(capability)
        agent = self._agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent not initialized: {agent_name}")
        return agent_name, agent

    def _augment_history_with_memory(self, history: List[Dict[str, str]], user_id: str, query: str) -> List[Dict[str, str]]:
        context = self.memory.build_context(user_id=user_id, query=query)
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
            self.memory.ingest_turn(
                session_id=session_id,
                user_id=user_id,
                user_text=user_text,
                assistant_reply=final_reply,
                meta=meta,
            )
        except Exception as exc:
            logger.warning(f"Memory ingest skipped due to error: {exc}")

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
            task = self.task_manager.get_task(task_id)
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
            self.task_manager.append_step_result(task_id, step_result)

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

    def _extract_task_command(self, user_text: str) -> Optional[Tuple[str, str]]:
        text = user_text.strip()
        task_id_match = self.TASK_ID_RE.search(text)
        task_id = task_id_match.group(1) if task_id_match else ""

        if text.startswith("查询任务") or text.lower().startswith("task_status"):
            return ("query", task_id)
        if text.startswith("取消任务") or text.lower().startswith("task_cancel"):
            return ("cancel", task_id)
        if text.startswith("重试任务") or text.lower().startswith("task_retry"):
            return ("retry", task_id)
        return None

    def _format_task_status(self, task_id: str) -> str:
        if not task_id:
            return "未识别到任务ID，请提供形如 t-YYYYMMDDHHMMSS-xxxxxxxx 的任务号。"
        task = self.task_manager.get_task(task_id)
        if not task:
            return f"未找到任务 {task_id}。"

        lines = [f"任务ID: {task.task_id}", f"状态: {task.state.value}", f"更新时间: {task.updated_at}"]
        if task.error:
            lines.append(f"错误: {task.error.get('code')} | {task.error.get('message')}")
        if task.step_results:
            lines.append("步骤摘要:")
            for s in task.step_results[-5:]:
                lines.append(f"- [{s.get('step_id')}] {s.get('state')} | {s.get('title')}")
        return "\n".join(lines)

    def _handle_task_command(
        self,
        user_text: str,
        history: List[Dict[str, str]],
    ) -> Optional[OrchestrationResult]:
        cmd = self._extract_task_command(user_text)
        if not cmd:
            return None

        action, task_id = cmd
        _, chat_agent = self._resolve_agent("chat")

        if action == "query":
            status_text = self._format_task_status(task_id)
        elif action == "cancel":
            ok = self.task_manager.cancel_task(task_id) if task_id else False
            status_text = f"任务 {task_id} 已取消。" if ok else f"取消失败，任务 {task_id or '(缺失ID)'} 不可取消或不存在。"
        elif action == "retry":
            task = self.task_manager.retry_task(task_id) if task_id else None
            status_text = (
                f"任务 {task_id} 已重置为 pending，可再次执行。" if task else f"重试失败，任务 {task_id or '(缺失ID)'} 不存在。"
            )
        else:
            status_text = "未知任务指令。"

        final_reply = chat_agent.reply_with_task_result(
            user_text=user_text,
            executor_output=status_text,
            history=history,
        )

        return OrchestrationResult(
            intent=RoutingIntent.CHAT,
            final_reply=final_reply,
            executor_result=None,
            meta={
                "phase": "phase6",
                "task_command": action,
                "task_id": task_id,
                "task_mode": False,
                "agent_chain": ["chat_agent"],
            },
        )

    def _handle_memory_command(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        session_id: str,
        user_id: str,
    ) -> Optional[OrchestrationResult]:
        cmd = self.memory.parse_memory_command(user_text)
        if not cmd:
            return None

        _, chat_agent = self._resolve_agent("chat")
        action = cmd.get("action", "")

        if action == "remember":
            value = (cmd.get("value") or "").strip()
            if not value:
                result_text = "请在“记住”后提供要保存的信息。"
            else:
                memory_id = self.memory.remember(user_id=user_id, session_id=session_id, content=value)
                result_text = f"已记住（#{memory_id}）：{value}"
        elif action == "list_profile":
            rows = self.memory.list_profile(user_id=user_id, limit=8)
            if not rows:
                result_text = "当前还没有已记录的偏好记忆。"
            else:
                lines = ["你的偏好记忆："]
                for r in rows:
                    lines.append(f"- #{r.get('memory_id')} {r.get('content')}")
                result_text = "\n".join(lines)
        elif action == "list_commitments":
            rows = self.memory.list_commitments(user_id=user_id, limit=8)
            if not rows:
                result_text = "当前没有未完成待办。"
            else:
                lines = ["你的未完成待办："]
                for r in rows:
                    due = (r.get("payload") or {}).get("due", "")
                    suffix = f" (截止:{due})" if due else ""
                    lines.append(f"- #{r.get('memory_id')} {r.get('content')}{suffix}")
                result_text = "\n".join(lines)
        elif action == "close_commitment":
            memory_id = int(cmd.get("memory_id"))
            ok = self.memory.close_commitment(user_id=user_id, memory_id=memory_id)
            result_text = f"待办 #{memory_id} 已标记完成。" if ok else f"未找到待办 #{memory_id}。"
        else:
            result_text = "未知记忆指令。"

        final_reply = chat_agent.reply_with_task_result(
            user_text=user_text,
            executor_output=result_text,
            history=history,
        )

        return OrchestrationResult(
            intent=RoutingIntent.CHAT,
            final_reply=final_reply,
            executor_result=None,
            meta={
                "phase": "phase6",
                "memory_command": action,
                "task_mode": False,
                "agent_chain": ["chat_agent", "memory_service", "chat_agent"],
            },
        )

    def handle_user_message(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        session_id: str,
        user_id: str,
    ) -> OrchestrationResult:
        # task lifecycle commands
        command_result = self._handle_task_command(user_text=user_text, history=history)
        if command_result is not None:
            self._record_memory(session_id, user_id, user_text, command_result.final_reply, command_result.meta)
            return command_result

        # memory commands
        memory_result = self._handle_memory_command(
            user_text=user_text,
            history=history,
            session_id=session_id,
            user_id=user_id,
        )
        if memory_result is not None:
            self._record_memory(session_id, user_id, user_text, memory_result.final_reply, memory_result.meta)
            return memory_result

        enriched_history = self._augment_history_with_memory(history=history, user_id=user_id, query=user_text)

        _, chat_agent = self._resolve_agent("chat")
        intent = chat_agent.classify_intent(user_text=user_text, history=enriched_history)

        if intent == RoutingIntent.CHAT:
            final_reply = chat_agent.reply_chat(user_text=user_text, history=enriched_history)
            meta = {"agent_chain": ["chat_agent"], "task_mode": False, "phase": "phase6"}
            self._record_memory(session_id, user_id, user_text, final_reply, meta)
            return OrchestrationResult(
                intent=RoutingIntent.CHAT,
                final_reply=final_reply,
                executor_result=None,
                meta=meta,
            )

        # task mode (phase6)
        task = self.task_manager.create_task(session_id=session_id, user_text=user_text)
        task_id = task.task_id
        self.task_manager.set_state(task_id, TaskState.RUNNING)

        _, planner_agent = self._resolve_agent("task_planning")
        _, critic_agent = self._resolve_agent("task_review")

        workflow_mode = "general"
        travel_constraints = None

        if self.travel_workflow.is_match(user_text):
            workflow_mode = "travel"
            travel_constraints = self.travel_workflow.parse_constraints(user_text)
            missing = self.travel_workflow.missing_required_fields(travel_constraints)
            if missing:
                clarification = self.travel_workflow.build_clarification_request(travel_constraints, missing)
                self.task_manager.set_state(task_id, TaskState.PENDING)
                self.task_manager.set_plan(task_id, {"workflow": "travel", "constraints": travel_constraints.to_dict()})

                executor_result = ExecutorRunResult(
                    output_text=clarification,
                    tool_events=[],
                    error=None,
                    step_results=[],
                )
                final_reply = chat_agent.reply_with_task_result(
                    user_text=user_text,
                    executor_output=clarification,
                    history=enriched_history,
                )
                meta = {
                    "phase": "phase6",
                    "task_id": task_id,
                    "workflow": "travel",
                    "need_clarification": True,
                    "missing_fields": missing,
                    "constraints": travel_constraints.to_dict(),
                    "agent_chain": ["chat_agent", "workflow_guard", "chat_agent"],
                    "task_mode": True,
                    "task_error": False,
                }
                self._record_memory(session_id, user_id, user_text, final_reply, meta)
                return OrchestrationResult(
                    intent=RoutingIntent.TASK,
                    final_reply=final_reply,
                    executor_result=executor_result,
                    meta=meta,
                )
            plan_result = self.travel_workflow.build_plan(user_text=user_text, constraints=travel_constraints)
        else:
            plan_result = planner_agent.plan_task(user_text=user_text, history=enriched_history)

        self.task_manager.set_plan(task_id, plan_result.to_dict())

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

        if workflow_mode == "travel" and travel_constraints is not None:
            workflow_review = self.travel_workflow.review(
                constraints=travel_constraints,
                graph_dict=graph.to_dict(),
                critic=critic_result,
            )
            executor_output = self.travel_workflow.compose_executor_output(
                constraints=travel_constraints,
                graph_dict=graph.to_dict(),
                critic=critic_result,
                workflow_review=workflow_review,
            )
        else:
            workflow_review = None
            executor_output = self._compose_general_executor_output(graph=graph, critic=critic_result)

        executor_result = ExecutorRunResult(
            output_text=executor_output,
            tool_events=run_info["all_tool_events"],
            error=run_info["first_error"],
            step_results=run_info["step_results"],
        )

        current_task = self.task_manager.get_task(task_id)
        if not current_task or current_task.state != TaskState.CANCELLED:
            if run_info["first_error"]:
                self.task_manager.set_state(task_id, TaskState.FAILED, error=run_info["first_error"])
            else:
                self.task_manager.set_state(task_id, TaskState.SUCCEEDED)

        final_reply = chat_agent.reply_with_task_result(
            user_text=user_text,
            executor_output=executor_result.output_text,
            history=enriched_history,
        )

        meta = {
            "phase": "phase6",
            "task_id": task_id,
            "agent_chain": ["chat_agent", "planner_agent", "executor_agent", "critic_agent", "chat_agent"],
            "task_mode": True,
            "workflow": workflow_mode,
            "plan": plan_result.to_dict(),
            "task_graph": graph.to_dict(),
            "critic": critic_result.to_dict(),
            "task_error": bool(run_info["first_error"]),
        }
        if workflow_mode == "travel" and travel_constraints is not None:
            meta["constraints"] = travel_constraints.to_dict()
            meta["workflow_review"] = workflow_review

        self._record_memory(session_id, user_id, user_text, final_reply, meta)
        return OrchestrationResult(
            intent=RoutingIntent.TASK,
            final_reply=final_reply,
            executor_result=executor_result,
            meta=meta,
        )
