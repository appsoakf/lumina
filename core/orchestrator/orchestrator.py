import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.agentic.chat_agent import ChatAgent
from core.agentic.critic_agent import CriticAgent
from core.agentic.executor_agent import ExecutorAgent
from core.agentic.planner_agent import PlannerAgent
from core.capabilities import CapabilityRegistry, build_default_registry
from core.config import load_app_config
from core.memory import MemoryService
from core.orchestrator.langgraph_task_runner import LangGraphTaskRunner
from core.orchestrator.task_snapshot import completed_context, resolve_step_inputs
from core.protocols import (
    CriticResult,
    ExecutorRunResult,
    OrchestrationResult,
    PlanItem,
    PlanResult,
    RoutingIntent,
    TaskState,
)
from core.tasks import TaskManager
from core.utils import bind_log_context, elapsed_ms, log_event, log_exception

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator: 统一编排入口（路由 + 任务生命周期 + 记忆读写）。

    这层的核心职责是把“用户一轮输入”收敛成可交付结果，并保证：
    1) service 只依赖公开接口，不感知内部 agent/task/memory 细节；
    2) task 模式下支持可恢复执行（waiting_user_input -> resume）；
    3) 出错时有界重规划，不无限循环；
    4) 每轮对话可沉淀可检索记忆。
    """

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
        # 统一管理各能力代理，后续通过 capability -> agent_name 动态解析。
        self._agents = {
            "chat_agent": chat_agent or ChatAgent(),
            "planner_agent": planner_agent or PlannerAgent(),
            "executor_agent": executor_agent or ExecutorAgent(),
            "critic_agent": critic_agent or CriticAgent(),
        }
        # capabilities 决定“某项能力由哪个 agent 承担”，避免硬编码调用关系。
        self.capabilities = capability_registry or build_default_registry()
        # task manager 负责持久化任务状态机；memory service 负责短/长期记忆。
        self._task_manager = task_manager or TaskManager()
        self._memory = memory_service or MemoryService()
        # task runner 封装 LangGraph 调度细节；orchestrator 仅按公共 run 接口调用。
        self._task_runner = LangGraphTaskRunner(
            task_manager=self._task_manager,
            build_step_input=self._build_step_task_input,
        )
        # 收敛护栏：限制自动 replan/clarify 次数，避免任务链路失控。
        task_flow_cfg = load_app_config().task_flow
        self._max_replan_rounds = max(int(task_flow_cfg.max_replan_rounds), 0)
        self._max_clarify_rounds = max(int(task_flow_cfg.max_clarify_rounds), 1)

    def _resolve_agent(self, capability: str):
        # 通过能力名解析具体 agent；该映射由 CapabilityRegistry 统一维护。
        agent_name = self.capabilities.resolve_agent(capability)
        agent = self._agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent not initialized: {agent_name}")
        return agent_name, agent

    def _augment_history_with_memory(self, history: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
        # 查询长期记忆并前置到 system message，作为“软上下文注入”。
        # 注意：明确声明“当前用户指令优先”，防止旧记忆覆盖本轮意图。
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
        # 记忆写入采用 fail-soft：失败只记录日志，不中断主响应链路。
        try:
            self._memory.ingest_turn(
                session_id=session_id,
                user_text=user_text,
                assistant_reply=final_reply,
                meta=meta,
            )
        except Exception:
            log_exception(
                logger,
                "orchestrator.memory.ingest.error",
                "记忆写入失败，本轮已跳过写入",
                component="orchestrator",
                fallback="skip_memory_ingest",
            )

    def record_session_round(
        self,
        session_id: str,
        user_text: str,
        assistant_reply: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        # 对外公开的短期会话落盘入口（供 service 在一轮结束后调用）。
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
        
        # 把步骤（成功/失败/阻塞）的结果拼成文本，给当前步骤提供历史上下文
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
        resume_plan_result: Optional[PlanResult] = None,
        resume_snapshot: Optional[Dict[str, Any]] = None,
        resume_waiting_payload: Optional[Dict[str, Any]] = None,
        resume_user_reply: str = "",
    ):
        # task 模式统一委托给 LangGraphTaskRunner。
        # 这里仅做 agent 解析与参数透传，不掺杂图调度细节。
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
            resume_plan_result=resume_plan_result,
            resume_snapshot=resume_snapshot,
            resume_waiting_payload=resume_waiting_payload,
            resume_user_reply=resume_user_reply,
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

    def _plan_result_from_dict(self, payload: Dict[str, Any], user_text: str) -> PlanResult:
        # waiting 任务恢复时，plan/task_snapshot 来自持久化 JSON；
        # 这里负责把 dict 重新归一为 PlanResult 数据结构。
        raw_steps = payload.get("steps") or []
        steps: List[PlanItem] = []
        for idx, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                continue
            step_id = str(item.get("step_id") or f"S{idx}").strip() or f"S{idx}"
            title = str(item.get("title") or f"步骤{idx}").strip() or f"步骤{idx}"
            instruction = str(item.get("instruction") or title).strip() or title
            depends_on = [str(dep).strip() for dep in (item.get("depends_on") or []) if str(dep).strip()]
            bindings: List[Dict[str, Any]] = []
            for binding in item.get("input_bindings") or []:
                if not isinstance(binding, dict):
                    continue
                source = str(binding.get("from") or "").strip()
                target = str(binding.get("to") or "").strip()
                if not source or not target:
                    continue
                bindings.append({"from": source, "to": target})
            steps.append(
                PlanItem(
                    step_id=step_id,
                    title=title,
                    instruction=instruction,
                    depends_on=depends_on,
                    input_bindings=bindings,
                )
            )
        return PlanResult(
            goal=str(payload.get("goal") or user_text).strip() or user_text,
            steps=steps,
            raw_text=str(payload.get("raw_text") or ""),
            error=payload.get("error") if isinstance(payload.get("error"), dict) else None,
            graph_policy=payload.get("graph_policy") if isinstance(payload.get("graph_policy"), dict) else {},
        )

    def _compose_waiting_executor_output(self, task_snapshot: Dict[str, Any], waiting_for_input: Dict[str, Any]) -> str:
        # 当任务进入 waiting_user_input 时，生成给 chat_agent 的中间可读描述。
        # chat_agent 会基于该文本组织最终对用户的追问回复。
        question = str(waiting_for_input.get("clarify_question") or "").strip()
        required_fields = [str(v).strip() for v in (waiting_for_input.get("required_fields") or []) if str(v).strip()]
        summary = str(waiting_for_input.get("summary") or "当前任务需要补充信息。").strip()

        lines = [
            f"任务目标: {task_snapshot.get('goal', '')}",
            "",
            "当前状态: 等待补充信息",
            f"说明: {summary}",
        ]
        if question:
            lines.append(f"请补充: {question}")
        if required_fields:
            lines.append(f"建议补充字段: {', '.join(required_fields)}")
        return "\n".join(lines)

    def _task_not_converged_error(self, *, reason: str) -> Dict[str, Any]:
        # 统一“未收敛”错误结构，便于前端和上层链路识别重试语义。
        return {
            "code": "TASK_NOT_CONVERGED",
            "message": reason,
            "retryable": True,
        }

    def _should_replan(self, task_run: Any) -> bool:
        # replan 触发条件：
        # 1) 非 waiting（等待用户补充时不自动重规划）；
        # 2) 存在首错；
        # 3) 错误标记为可重试。
        if task_run.waiting_for_input:
            return False
        if task_run.first_error:
            error = task_run.first_error or {}
            retryable = bool(error.get("retryable", True))
            return retryable
        return False

    def _compose_replan_user_text(self, *, original_user_text: str, task_run: Any, round_index: int) -> str:
        # 将“原始目标 + 上轮失败信息 + critic建议”组合成下一轮 planner 输入，
        # 让 replan 具备上下文连续性，而不是盲目重试同一方案。
        lines = [
            original_user_text,
            "",
            f"[重试轮次 {round_index}] 请根据上一轮执行结果修正计划并继续完成任务。",
        ]
        if task_run.first_error:
            err = task_run.first_error
            lines.append(f"上轮错误: {err.get('code', '')} {err.get('message', '')}".strip())

        critic = task_run.critic_result
        if critic and critic.suggestions:
            lines.append("评审建议:")
            for item in critic.suggestions[:4]:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")

        return "\n".join(lines).strip()

    def _mark_not_converged_if_needed(
        self,
        *,
        task_id: str,
        task_run: Any,
        round_count: int,
    ) -> Any:
        # waiting_user_input 超过阈值后，主动转 failed(TASK_NOT_CONVERGED)。
        # 这样可以避免任务长期停在等待态造成“假活跃”。
        if not task_run.waiting_for_input:
            return task_run
        if round_count < self._max_clarify_rounds:
            return task_run

        reason = f"Clarification rounds exceeded max_clarify_rounds={self._max_clarify_rounds}"
        error = self._task_not_converged_error(reason=reason)
        for node in task_run.task_snapshot.get("nodes") or []:
            if str(node.get("state")) == "waiting_user_input":
                node["state"] = "failed"
                node["error"] = error
        task_run.waiting_for_input = None
        task_run.first_error = error
        task_run.critic_result = CriticResult(quality="revise", summary="任务多轮追问后仍未收敛。")
        self._task_manager.set_task_snapshot(task_id, task_run.task_snapshot)
        self._task_manager.set_state(task_id, TaskState.FAILED, error=error)
        return task_run

    def _run_with_convergence_loop(
        self,
        *,
        user_text: str,
        history: List[Dict[str, str]],
        session_id: str,
        task_id: str,
        resume_plan_result: Optional[PlanResult] = None,
        resume_snapshot: Optional[Dict[str, Any]] = None,
        resume_waiting_payload: Optional[Dict[str, Any]] = None,
        resume_user_reply: str = "",
    ) -> tuple[Any, int]:
        """
        任务收敛主循环：
        1) 先跑一轮 task graph；
        2) 若命中可重试错误则触发有界 replan；
        3) 否则返回当前结果。

        返回值：(task_run, replan_used)
        """
        replan_used = 0
        current_user_text = user_text
        use_resume = resume_plan_result is not None and resume_snapshot is not None
        current_plan = resume_plan_result
        current_snapshot = resume_snapshot
        current_waiting_payload = resume_waiting_payload
        current_resume_reply = resume_user_reply

        while True:
            # 单轮执行：可能是新任务，也可能是 waiting 恢复任务。
            task_run = self._run_task_mode(
                user_text=current_user_text,
                history=history,
                session_id=session_id,
                task_id=task_id,
                resume_plan_result=current_plan if use_resume else None,
                resume_snapshot=current_snapshot if use_resume else None,
                resume_waiting_payload=current_waiting_payload if use_resume else None,
                resume_user_reply=current_resume_reply if use_resume else "",
            )
            use_resume = False
            current_plan = None
            current_snapshot = None
            current_waiting_payload = None
            current_resume_reply = ""

            # 不需要 replan 时，直接结束循环并返回当前结果。
            if not self._should_replan(task_run):
                return task_run, replan_used
            if replan_used >= self._max_replan_rounds:
                # 超出 replan 预算：标记 failed 并返回，防止无限自旋。
                reason = f"Replan rounds exceeded max_replan_rounds={self._max_replan_rounds}"
                error = self._task_not_converged_error(reason=reason)
                task_run.first_error = error
                self._task_manager.set_state(task_id, TaskState.FAILED, error=error)
                return task_run, replan_used

            # 进入下一轮自动重试：先把任务状态回退到 RUNNING，再更新 planner 输入。
            replan_used += 1
            self._task_manager.reset_task_for_replan(task_id)
            self._task_manager.set_state(task_id, TaskState.RUNNING)
            current_user_text = self._compose_replan_user_text(
                original_user_text=user_text,
                task_run=task_run,
                round_index=replan_used,
            )

    def _compose_general_executor_output(self, task_snapshot: Dict[str, object], critic: CriticResult) -> str:
        # 将 task graph 快照归一为可读执行摘要，供 chat_agent 二次组织自然语言回复。
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
                "skipped": "跳过",
                "waiting_user_input": "等待补充信息",
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

    def _build_task_orchestration_result(
        self,
        *,
        user_text: str,
        history: List[Dict[str, str]],
        task_id: str,
        task_run: Any,
        round_count: int,
        replan_count: int,
    ) -> OrchestrationResult:
        # 1) 先把 task_run 转成 executor_result（统一兼容字段结构）；
        # 2) 再交给 chat_agent 生成人类可读最终回复；
        # 3) 最后组装完整 meta，供 service/日志/前端消费。
        _, chat_agent = self._resolve_agent("chat")

        if task_run.waiting_for_input:
            executor_output = self._compose_waiting_executor_output(
                task_snapshot=task_run.task_snapshot,
                waiting_for_input=task_run.waiting_for_input,
            )
        else:
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

        chat_started = time.perf_counter()
        final_reply = chat_agent.reply_with_task_result(
            user_text=user_text,
            executor_output=executor_result.output_text,
            history=history,
        )
        chat_llm_ms = elapsed_ms(chat_started)
        log_event(
            logger,
            logging.INFO,
            "orchestrator.chat.reply.done",
            "任务结果回复生成完成",
            component="orchestrator",
            task_id=task_id,
            intent=RoutingIntent.TASK.value,
            duration_ms=chat_llm_ms,
        )

        waiting_payload = task_run.waiting_for_input or {}
        meta = {
            "task_id": task_id,
            "agent_chain": ["chat_agent", "planner_agent", "executor_agent", "critic_agent", "chat_agent"],
            "task_mode": True,
            "plan": task_run.plan_result.to_dict(),
            "task_graph": task_run.task_snapshot,
            "critic": task_run.critic_result.to_dict(),
            "task_error": bool(task_run.first_error),
            "task_waiting_input": bool(task_run.waiting_for_input),
            "task_waiting_step_id": waiting_payload.get("pending_step_id"),
            "task_clarify_question": waiting_payload.get("clarify_question"),
            "task_required_fields": list(waiting_payload.get("required_fields") or []),
            "task_round_count": int(round_count),
            "task_replan_count": int(replan_count),
            "perf": {
                "chat_llm_ms": chat_llm_ms,
            },
        }

        return OrchestrationResult(
            intent=RoutingIntent.TASK,
            final_reply=final_reply,
            executor_result=executor_result,
            meta=meta,
        )

    def handle_user_message(
        self,
        user_text: str,
        session_id: str,
    ) -> OrchestrationResult:
        """
        单轮编排主入口（service 层唯一需要调用的方法）：
        A. 读取短期历史；
        B. 注入长期记忆上下文；
        C. 优先恢复 waiting 任务；
        D. 新请求做 chat/task 路由；
        E. task 模式下执行有界收敛循环并落盘状态。
        """
        started = time.perf_counter()
        intent_name = "-"
        flow_mode = "unknown"
        task_id_for_log = "-"
        intent_ms = -1
        task_run_ms = -1
        chat_llm_ms = -1

        def _to_int(value: Any, default: int = -1) -> int:
            try:
                return int(value)
            except Exception:
                return default

        def _attach_perf(meta: Dict[str, Any]) -> None:
            nonlocal chat_llm_ms
            perf_payload = dict(meta.get("perf") or {})
            candidate_chat_llm = _to_int(perf_payload.get("chat_llm_ms"), default=chat_llm_ms)
            if candidate_chat_llm >= 0:
                chat_llm_ms = candidate_chat_llm
            perf_payload.update(
                {
                    "intent_ms": intent_ms,
                    "task_run_ms": task_run_ms,
                    "chat_llm_ms": chat_llm_ms,
                    "orchestrator_ms": elapsed_ms(started),
                }
            )
            meta["perf"] = perf_payload

        try:
            # A) 读取短期对话历史（上一轮 user/assistant 消息）。
            history = self._memory.get_recent_history(session_id=session_id)

            # B) 将长期记忆注入到历史头部，提升跨轮连续性。
            enriched_history = self._augment_history_with_memory(history=history, query=user_text)

            # C) waiting 任务恢复：若存在等待补充信息的任务，本轮输入优先作为补充继续执行。
            #    该路径沿用原 task_id，不会新建任务，保证任务链路连续可追踪。
            waiting_task = self._task_manager.get_waiting_task(session_id=session_id)
            if waiting_task is not None:
                resumed_task, resumed, waiting_payload = self._task_manager.resume_waiting_task(
                    waiting_task.task_id,
                    user_reply=user_text,
                )
                if resumed and resumed_task is not None:
                    flow_mode = "task_resume"
                    intent_name = RoutingIntent.TASK.value
                    task_id_for_log = resumed_task.task_id
                    with bind_log_context(task_id=resumed_task.task_id):
                        plan_payload = resumed_task.plan if isinstance(resumed_task.plan, dict) else {}
                        plan_result = self._plan_result_from_dict(plan_payload, user_text=resumed_task.user_text)
                        snapshot = resumed_task.task_snapshot if isinstance(resumed_task.task_snapshot, dict) else {}
                        prev_round_count = int((resumed_task.convergence or {}).get("round_count", 0))
                        prev_replan_count = int((resumed_task.convergence or {}).get("replan_count", 0))
                        round_count = prev_round_count + 1

                        task_started = time.perf_counter()
                        task_run, replan_used = self._run_with_convergence_loop(
                            user_text=resumed_task.user_text,
                            history=enriched_history,
                            session_id=session_id,
                            task_id=resumed_task.task_id,
                            resume_plan_result=plan_result,
                            resume_snapshot=snapshot,
                            resume_waiting_payload=waiting_payload,
                            resume_user_reply=user_text,
                        )
                        task_run_ms = elapsed_ms(task_started)
                        task_run = self._mark_not_converged_if_needed(
                            task_id=resumed_task.task_id,
                            task_run=task_run,
                            round_count=round_count,
                        )
                        total_replan_count = prev_replan_count + int(replan_used)
                        self._task_manager.update_convergence(
                            resumed_task.task_id,
                            {
                                "round_count": round_count,
                                "replan_count": total_replan_count,
                                "last_progress_score": 1 if not task_run.waiting_for_input else 0,
                            },
                        )
                        log_event(
                            logger,
                            logging.INFO,
                            "orchestrator.task.run.done",
                            "任务编排执行完成",
                            component="orchestrator",
                            task_id=resumed_task.task_id,
                            duration_ms=task_run_ms,
                            round_count=round_count,
                            replan_count=total_replan_count,
                            step_count=len(task_run.task_snapshot.get("nodes") or []),
                            task_waiting_input=bool(task_run.waiting_for_input),
                            task_error=bool(task_run.first_error),
                        )
                        result = self._build_task_orchestration_result(
                            user_text=user_text,
                            history=enriched_history,
                            task_id=resumed_task.task_id,
                            task_run=task_run,
                            round_count=round_count,
                            replan_count=total_replan_count,
                        )
                    _attach_perf(result.meta)
                    self._record_memory(session_id, user_text, result.final_reply, result.meta)
                    return result

            # D) 常规路由：先判定 CHAT 还是 TASK。
            _, chat_agent = self._resolve_agent("chat")
            intent_started = time.perf_counter()
            intent = chat_agent.classify_intent(user_text=user_text, history=enriched_history)
            intent_ms = elapsed_ms(intent_started)
            intent_name = intent.value
            log_event(
                logger,
                logging.INFO,
                "orchestrator.intent.done",
                "意图识别完成",
                component="orchestrator",
                intent=intent_name,
                duration_ms=intent_ms,
            )

            if intent == RoutingIntent.CHAT:
                # CHAT：直接由 chat_agent 响应，不进入任务状态机。
                flow_mode = "chat"
                chat_started = time.perf_counter()
                final_reply = chat_agent.reply_chat(user_text=user_text, history=enriched_history)
                chat_llm_ms = elapsed_ms(chat_started)
                log_event(
                    logger,
                    logging.INFO,
                    "orchestrator.chat.reply.done",
                    "对话回复生成完成",
                    component="orchestrator",
                    intent=RoutingIntent.CHAT.value,
                    duration_ms=chat_llm_ms,
                )
                meta = {"agent_chain": ["chat_agent"], "task_mode": False}
                _attach_perf(meta)
                self._record_memory(session_id, user_text, final_reply, meta)
                return OrchestrationResult(
                    intent=RoutingIntent.CHAT,
                    final_reply=final_reply,
                    executor_result=None,
                    meta=meta,
                )

            # E) TASK：创建任务并进入收敛循环（包含可重试 replan）。
            flow_mode = "task_new"
            task = self._task_manager.create_task(session_id=session_id, user_text=user_text)
            task_id = task.task_id
            task_id_for_log = task_id
            self._task_manager.set_state(task_id, TaskState.RUNNING)
            round_count = 1
            with bind_log_context(task_id=task_id):
                task_started = time.perf_counter()
                task_run, replan_used = self._run_with_convergence_loop(
                    user_text=user_text,
                    history=enriched_history,
                    session_id=session_id,
                    task_id=task_id,
                )
                task_run_ms = elapsed_ms(task_started)
                task_run = self._mark_not_converged_if_needed(
                    task_id=task_id,
                    task_run=task_run,
                    round_count=round_count,
                )
                total_replan_count = int(replan_used)
                self._task_manager.update_convergence(
                    task_id,
                    {
                        "round_count": round_count,
                        "replan_count": total_replan_count,
                        "last_progress_score": 1 if not task_run.waiting_for_input else 0,
                    },
                )
                log_event(
                    logger,
                    logging.INFO,
                    "orchestrator.task.run.done",
                    "任务编排执行完成",
                    component="orchestrator",
                    task_id=task_id,
                    duration_ms=task_run_ms,
                    round_count=round_count,
                    replan_count=total_replan_count,
                    step_count=len(task_run.task_snapshot.get("nodes") or []),
                    task_waiting_input=bool(task_run.waiting_for_input),
                    task_error=bool(task_run.first_error),
                )
                result = self._build_task_orchestration_result(
                    user_text=user_text,
                    history=enriched_history,
                    task_id=task_id,
                    task_run=task_run,
                    round_count=round_count,
                    replan_count=total_replan_count,
                )
            _attach_perf(result.meta)
            self._record_memory(session_id, user_text, result.final_reply, result.meta)
            return result
        finally:
            log_event(
                logger,
                logging.INFO,
                "orchestrator.handle.done",
                "编排入口处理完成",
                component="orchestrator",
                intent=intent_name,
                task_id=task_id_for_log,
                flow=flow_mode,
                duration_ms=elapsed_ms(started),
                intent_ms=intent_ms,
                task_run_ms=task_run_ms,
                chat_llm_ms=chat_llm_ms,
            )

    def close(self) -> None:
        # 释放底层资源（例如 memory 异步线程/连接）。
        close_fn = getattr(self._memory, "close", None)
        if callable(close_fn):
            close_fn()
