import json
import logging
from typing import Dict, List, Optional

from core.agentic.chat_agent import ChatAgent
from core.agentic.critic_agent import CriticAgent
from core.agentic.executor_agent import ExecutorAgent
from core.agentic.planner_agent import PlannerAgent
from core.capabilities import CapabilityRegistry, build_default_registry
from core.memory import MemoryService
from core.orchestrator.langgraph_task_runner import LangGraphTaskRunner
from core.orchestrator.task_snapshot import completed_context, resolve_step_inputs
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
        self._task_runner = LangGraphTaskRunner(
            task_manager=self._task_manager,
            build_step_input=self._build_step_task_input,
        )

    def _resolve_agent(self, capability: str):
        agent_name = self.capabilities.resolve_agent(capability)
        agent = self._agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent not initialized: {agent_name}")
        return agent_name, agent

    def _augment_history_with_memory(self, history: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
        context = self._memory.build_context(query=query)
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

    def _record_memory(self, session_id: str, user_text: str, final_reply: str, meta: Dict) -> None:
        try:
            self._memory.ingest_turn(
                session_id=session_id,
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

    def _build_step_task_input(self, user_text: str, task_snapshot: Dict[str, object], step_id: str) -> str:
        # 找 step_id 对应节点
        node = next(
            (n for n in (task_snapshot.get("nodes") or []) if str(n.get("step_id")) == step_id),
            None,
        )
        if node is None:
            raise ValueError(f"Unknown step_id: {step_id}")
        
        # 把步骤（成功/失败/取消/阻塞）的结果拼成文本，给当前步骤提供历史上下文
        completed = completed_context(task_snapshot)
        bound_inputs = resolve_step_inputs(task_snapshot, step_id=step_id)
        binding_text = json.dumps(bound_inputs, ensure_ascii=False, indent=2) if bound_inputs else "无"
        return (
            f"总目标: {user_text}\n"
            f"当前步骤: {node.get('step_id')} {node.get('title', '')}\n"
            f"步骤指令: {node.get('instruction', '')}\n"
            f"结构化输入绑定:\n{binding_text}\n"
            f"已完成上下文:\n{completed if completed else '无'}\n"
            "请完成当前步骤并输出结果摘要。"
        )

    def _run_task_mode(
        self,
        *,
        user_text: str,
        history: List[Dict[str, str]],
        session_id: str,
        task_id: str,
    ):
        _, planner_agent = self._resolve_agent("task_planning")
        _, executor_agent = self._resolve_agent("task_execution")
        _, critic_agent = self._resolve_agent("task_review")
        return self._task_runner.run(
            user_text=user_text,
            history=history,
            session_id=session_id,
            task_id=task_id,
            planner_agent=planner_agent,
            executor_agent=executor_agent,
            critic_agent=critic_agent,
        )

    def _extract_step_summary(self, output_text: object) -> str:
        text = str(output_text or "").strip()
        if not text:
            return ""

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                summary = str(payload.get("summary") or "").strip()
                if summary:
                    return summary
        except Exception:
            pass

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("结果摘要"):
                _, _, rhs = line.replace("：", ":", 1).partition(":")
                summary = rhs.strip()
                if summary:
                    return summary
            if line.startswith("步骤状态"):
                continue
            return line
        return ""

    def _compose_general_executor_output(self, task_snapshot: Dict[str, object], critic: CriticResult) -> str:
        lines = [f"任务目标: {task_snapshot.get('goal', '')}", "", "执行进展:"]
        for node in task_snapshot.get("nodes") or []:
            state_value = str(node.get("state") or "")
            state_text = {
                "succeeded": "成功",
                "failed": "失败",
                "running": "执行中",
                "ready": "就绪",
                "pending": "待执行",
                "blocked": "阻塞",
                "cancelled": "已取消",
                "skipped": "跳过",
            }.get(state_value, state_value)

            step_title = str(node.get("title") or "").strip() or str(node.get("step_id") or "步骤")
            lines.append(f"- {step_title}：{state_text}")

            summary = self._extract_step_summary(node.get("output_text"))
            if summary:
                lines.append(f"  说明: {summary}")

            if node.get("error"):
                err = node.get("error") or {}
                message = str(err.get("message") or "").strip() or "该步骤执行未完成。"
                lines.append(f"  问题: {message}")

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
    ) -> OrchestrationResult:
        history = self._memory.get_recent_history(session_id=session_id)

        # task shortcuts, e.g. /cancel and /retry
        shortcut_result = self._handle_task_shortcut(
            user_text=user_text,
            session_id=session_id,
            history=history,
        )
        if shortcut_result is not None:
            self._record_memory(session_id, user_text, shortcut_result.final_reply, shortcut_result.meta)
            return shortcut_result

        enriched_history = self._augment_history_with_memory(history=history, query=user_text)

        # 意图识别
        _, chat_agent = self._resolve_agent("chat")
        intent = chat_agent.classify_intent(user_text=user_text, history=enriched_history)

        if intent == RoutingIntent.CHAT:
            final_reply = chat_agent.reply_chat(user_text=user_text, history=enriched_history)
            meta = {"agent_chain": ["chat_agent"], "task_mode": False}
            self._record_memory(session_id, user_text, final_reply, meta)
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

        task_run = self._run_task_mode(
            user_text=user_text,
            history=enriched_history,
            session_id=session_id,
            task_id=task_id,
        )

        executor_output = self._compose_general_executor_output(
            task_snapshot=task_run.task_snapshot,
            critic=task_run.critic_result,
        )

        executor_result = ExecutorRunResult(
            output_text=executor_output,
            tool_events=task_run.all_tool_events,
            error=task_run.first_error,
            step_results=task_run.step_results,
        )

        final_reply = chat_agent.reply_with_task_result(
            user_text=user_text,
            executor_output=executor_result.output_text,
            history=enriched_history,
        )

        meta = {
            "task_id": task_id,
            "agent_chain": ["chat_agent", "planner_agent", "executor_agent", "critic_agent", "chat_agent"],
            "task_mode": True,
            "plan": task_run.plan_result.to_dict(),
            "task_graph": task_run.task_snapshot,
            "critic": task_run.critic_result.to_dict(),
            "task_error": bool(task_run.first_error),
        }

        self._record_memory(session_id, user_text, final_reply, meta)
        return OrchestrationResult(
            intent=RoutingIntent.TASK,
            final_reply=final_reply,
            executor_result=executor_result,
            meta=meta,
        )

    def close(self) -> None:
        close_fn = getattr(self._memory, "close", None)
        if callable(close_fn):
            close_fn()
