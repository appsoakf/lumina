import logging
import time
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from core.orchestrator.task_snapshot import step_result_from_node
from core.protocols import CriticResult, ExecutorRunResult, PlanResult, TaskState
from core.utils import elapsed_ms, log_event, log_exception


STEP_STATE_PENDING = "pending"
STEP_STATE_READY = "ready"
STEP_STATE_RUNNING = "running"
STEP_STATE_SUCCEEDED = "succeeded"
STEP_STATE_FAILED = "failed"
STEP_STATE_CANCELLED = "cancelled"
STEP_STATE_BLOCKED = "blocked"
STEP_STATE_SKIPPED = "skipped"
STEP_STATE_WAITING_USER_INPUT = "waiting_user_input"

TERMINAL_STATES = {
    STEP_STATE_SUCCEEDED,
    STEP_STATE_FAILED,
    STEP_STATE_CANCELLED,
    STEP_STATE_BLOCKED,
    STEP_STATE_SKIPPED,
}
UPSTREAM_FAILED_STATES = {
    STEP_STATE_FAILED,
    STEP_STATE_CANCELLED,
    STEP_STATE_BLOCKED,
}

ACTION_RUN_STEPS = "run_ready_steps"
ACTION_SELECT_STEPS = "select_ready_steps"
ACTION_REVIEW = "review_task"
ACTION_FINALIZE = "finalize_task"

logger = logging.getLogger(__name__)


@dataclass
class TaskFlowRunResult:
    plan_result: PlanResult
    critic_result: CriticResult
    task_snapshot: Dict[str, Any]
    all_tool_events: List[Dict[str, Any]]
    first_error: Optional[Dict[str, Any]]
    step_results: List[Dict[str, Any]]
    waiting_for_input: Optional[Dict[str, Any]] = None


class TaskFlowState(TypedDict, total=False):
    user_text: str
    history: List[Dict[str, str]]
    session_id: str
    task_id: str
    planner_agent: Any
    executor_agent: Any
    critic_agent: Any
    plan_result: PlanResult
    critic_result: CriticResult
    task_snapshot: Dict[str, Any] # 任务执行到了什么状态
    all_tool_events: List[Dict[str, Any]]
    first_error: Optional[Dict[str, Any]]
    ready_batch: List[str]
    executed_steps: List[str]
    next_action: str
    waiting_for_input: Optional[Dict[str, Any]]
    resume_mode: bool
    resume_plan_result: Optional[PlanResult]
    resume_snapshot: Optional[Dict[str, Any]]
    resume_waiting_payload: Optional[Dict[str, Any]]
    resume_user_reply: str


class LangGraphTaskRunner:
    def __init__(
        self,
        *,
        task_manager: Any,
        build_step_input: Callable[..., str],
    ):
        self._task_manager = task_manager
        self._build_step_input = build_step_input
        self._graph = self._build_graph()

    def run(
        self,
        *,
        user_text: str,
        history: List[Dict[str, str]],
        session_id: str,
        task_id: str,
        planner_agent: Any,
        executor_agent: Any,
        critic_agent: Any,
        resume_plan_result: Optional[PlanResult] = None,
        resume_snapshot: Optional[Dict[str, Any]] = None,
        resume_waiting_payload: Optional[Dict[str, Any]] = None,
        resume_user_reply: str = "",
    ) -> TaskFlowRunResult:
        initial: TaskFlowState = {
            "user_text": user_text,
            "history": history,
            "session_id": session_id,
            "task_id": task_id,
            "planner_agent": planner_agent,
            "executor_agent": executor_agent,
            "critic_agent": critic_agent,
            "all_tool_events": [],
            "first_error": None,
            "ready_batch": [],
            "executed_steps": [],
            "next_action": ACTION_SELECT_STEPS,
            "waiting_for_input": None,
            "resume_mode": resume_plan_result is not None and resume_snapshot is not None,
            "resume_plan_result": resume_plan_result,
            "resume_snapshot": resume_snapshot,
            "resume_waiting_payload": resume_waiting_payload,
            "resume_user_reply": resume_user_reply,
        }

        state = self._graph.invoke(initial)
        critic_result = state.get("critic_result")
        if critic_result is None:
            if state.get("waiting_for_input"):
                critic_result = CriticResult(quality="pass", summary="任务等待用户补充信息。")
            else:
                critic_result = CriticResult(quality="pass", summary="")
        return TaskFlowRunResult(
            plan_result=state["plan_result"],
            critic_result=critic_result,
            task_snapshot=state["task_snapshot"],
            all_tool_events=list(state["all_tool_events"]),
            first_error=state.get("first_error"),
            step_results=self._project_step_results(
                task_snapshot=state["task_snapshot"],
                executed_steps=state.get("executed_steps") or [],
            ),
            waiting_for_input=state.get("waiting_for_input"),
        )

    def _build_graph(self):
        graph = StateGraph(TaskFlowState)
        graph.add_node("plan_task", self._plan_task)
        graph.add_node("select_ready_steps", self._select_ready_steps)
        graph.add_node("run_ready_steps", self._run_ready_steps)
        graph.add_node("review_task", self._review_task)
        graph.add_node("finalize_task", self._finalize_task)

        graph.add_edge(START, "plan_task")
        graph.add_edge("plan_task", "select_ready_steps")
        graph.add_conditional_edges(
            "select_ready_steps",
            self._route_by_next_action,
            {
                ACTION_RUN_STEPS: ACTION_RUN_STEPS,
                ACTION_REVIEW: ACTION_REVIEW,
                ACTION_FINALIZE: ACTION_FINALIZE,
            },
        )
        graph.add_conditional_edges(
            "run_ready_steps",
            self._route_by_next_action,
            {
                ACTION_SELECT_STEPS: ACTION_SELECT_STEPS,
                ACTION_REVIEW: ACTION_REVIEW,
                ACTION_FINALIZE: ACTION_FINALIZE,
            },
        )
        graph.add_edge("review_task", "finalize_task")
        graph.add_edge("finalize_task", END)
        return graph.compile()

    def _plan_task(self, state: TaskFlowState) -> TaskFlowState:
        if state.get("resume_mode"):
            plan_result = state.get("resume_plan_result")
            snapshot = deepcopy(state.get("resume_snapshot") or {})
            waiting_payload = dict(state.get("resume_waiting_payload") or {})
            resume_reply = str(state.get("resume_user_reply") or "").strip()

            if plan_result is None or not isinstance(snapshot, dict) or not snapshot.get("nodes"):
                plan_result = plan_result or PlanResult(goal=state["user_text"], steps=[], raw_text="resume_invalid")
                snapshot = {
                    "goal": plan_result.goal,
                    "policy": self._normalize_policy(plan_result.graph_policy),
                    "nodes": [],
                    "topological_order": [],
                }
                first_error = {
                    "code": "TASK_RESUME_INVALID",
                    "message": "Task resume payload is invalid",
                    "retryable": True,
                }
            else:
                try:
                    self._prepare_resume_snapshot(
                        snapshot=snapshot,
                        waiting_payload=waiting_payload,
                        user_reply=resume_reply,
                    )
                    first_error = None
                except Exception as exc:
                    first_error = {
                        "code": "TASK_RESUME_INVALID",
                        "message": str(exc),
                        "retryable": True,
                    }
        else:
            planner_agent = state["planner_agent"]
            plan_result = planner_agent.plan_task(
                user_text=state["user_text"],
                history=state["history"],
            )

            try:
                steps = self._build_steps(plan_result)
                topological_order = self._build_topological_order(steps)
                policy = self._normalize_policy(plan_result.graph_policy)
                snapshot = {
                    "goal": plan_result.goal,
                    "policy": policy,
                    "nodes": steps,
                    "topological_order": topological_order,
                }
                first_error = state.get("first_error")
            except Exception as exc:
                snapshot = {
                    "goal": plan_result.goal,
                    "policy": self._normalize_policy(plan_result.graph_policy),
                    "nodes": [],
                    "topological_order": [],
                }
                first_error = state.get("first_error") or {
                    "code": "TASK_PLAN_INVALID",
                    "message": str(exc),
                    "retryable": True,
                }

            self._task_manager.set_plan(state["task_id"], plan_result.to_dict())

        state["plan_result"] = plan_result
        state["task_snapshot"] = snapshot
        state["first_error"] = first_error
        state["ready_batch"] = []
        state["next_action"] = ACTION_SELECT_STEPS
        state["waiting_for_input"] = None
        return state

    def _select_ready_steps(self, state: TaskFlowState) -> TaskFlowState:
        # 读取当前任务图快照和任务ID：
        # - snapshot 保存了 nodes/policy/topological_order 等运行时状态
        # - task_id 用于查询是否被用户取消（/cancel）
        snapshot = state["task_snapshot"]
        task_id = state["task_id"]

        # 分支1：任务已被取消
        # 一旦检测到取消，就不再继续选择可执行步骤，而是：
        # 1) 构造统一取消错误
        # 2) 将尚未执行的 pending/ready 步骤全部标记为 cancelled
        # 3) 仅在首错为空时写入 first_error（保持“第一条错误”语义）
        # 4) 清空 ready_batch，路由到 review_task 做结果收敛
        if self._is_task_cancelled(task_id):
            return self._cancel_and_route_review(state=state, snapshot=snapshot)

        # 已进入等待用户输入态时，本轮直接收敛到 finalize，避免继续调度。
        if self._has_waiting_input(snapshot):
            state["ready_batch"] = []
            state["next_action"] = ACTION_FINALIZE
            return state

        # 分支2：任务未取消，尝试刷新并挑选 ready 节点
        # _refresh_ready_nodes 会根据依赖状态把节点从 pending 转成 ready 或 blocked
        self._refresh_ready_nodes(snapshot)

        # _ready_nodes 会按拓扑顺序返回可执行节点，并受 max_parallelism 限制
        # 返回结果就是本轮要交给 _run_ready_steps 执行的批次
        ready_nodes = self._ready_nodes(snapshot, limit=self._max_parallelism(snapshot))
        if ready_nodes:
            # 有可执行节点：写入 ready_batch，并路由到执行节点
            state["ready_batch"] = [str(node.get("step_id")) for node in ready_nodes]
            state["next_action"] = ACTION_RUN_STEPS
            return state

        # 分支3：当前无 ready 节点
        # 先补一轮 blocked 标记，避免依赖失败的节点一直停在 pending/ready
        self._mark_blocked_nodes(snapshot)

        # 如果全图已经结束（所有节点都处于终态），直接进入 review 收尾
        if self._is_finished(snapshot):
            self._route_to_review(state)
            return state

        # 分支4：既没有 ready 节点，又没有结束，说明图进入停滞状态
        # 常见原因：依赖关系不满足且没有可推进节点。
        # 处理策略：
        # 1) 写入 TASK_GRAPH_STALLED 作为首错（如果首错尚未存在）
        # 2) 将剩余 pending/ready 节点标记为 blocked
        # 3) 路由到 review，避免调度循环空转
        stalled_error = {
            "code": "TASK_GRAPH_STALLED",
            "message": "Task graph has no executable ready nodes",
            "retryable": True,
        }
        self._mark_remaining_blocked(snapshot, stalled_error)
        self._set_first_error_once(state, stalled_error)
        self._route_to_review(state)
        return state

    def _run_ready_steps(self, state: TaskFlowState) -> TaskFlowState:
        # 读取本轮执行所需的核心上下文：
        # - snapshot: 当前任务图快照（包含所有节点状态与依赖）
        # - task_id: 用于取消检查与任务结果落盘
        # - executor_agent: 真正执行单步任务的代理
        # - fail_fast: 是否在出现首个步骤错误后阻断剩余未执行步骤
        snapshot = state["task_snapshot"]
        task_id = state["task_id"]
        executor_agent = state["executor_agent"]
        fail_fast = self._fail_fast(snapshot)

        # 入口取消检查：
        # 如果任务在进入执行节点前已经被取消，直接把剩余步骤标记为 cancelled 并路由到 review。
        if self._is_task_cancelled(task_id):
            return self._cancel_and_route_review(state=state, snapshot=snapshot)

        # 将本轮 ready 批次转换为待执行作业列表 jobs。
        # 每个作业保存 (step_id, step_input)：
        # - step_id: 用于后续回写结果
        # - step_input: 给 executor 的最终输入文本（包含上下文/绑定/步骤指令）
        ready_batch = list(state.get("ready_batch") or [])
        jobs: List[Tuple[str, str]] = []
        for step_id in ready_batch:
            node = self._get_node(snapshot, step_id)

            # 在提交并发执行前，把节点状态置为 running，并清空旧错误。
            # 这样在观测层可以即时看到该步骤已被调度。
            node["state"] = STEP_STATE_RUNNING
            node["error"] = None

            # 为每个步骤构造独立的执行输入，确保模型执行时拿到正确的步骤上下文。
            step_input = self._build_step_input(
                user_text=state["user_text"],
                task_snapshot=snapshot,
                step_id=step_id,
            )
            jobs.append((step_id, step_input))

        # 并发执行当前批次：
        # _run_step_batch 内部使用线程池并行调用 executor_agent.run_task。
        # 返回 step_id -> ExecutorRunResult 的映射，便于按原批次顺序回写。
        run_results = self._run_step_batch(
            jobs=jobs,
            executor_agent=executor_agent,
            history=state["history"],
            session_id=state["session_id"],
        )

        # 记录 fail_fast 触发错误：
        # - 只记录第一个触发 fail_fast 的错误
        # - 本轮已启动的并发步骤仍会回收结果并落盘
        # - 阻断发生在本轮结束后（阻断下一轮）
        fail_fast_error: Optional[Dict[str, Any]] = None
        waiting_payload: Optional[Dict[str, Any]] = None
        for step_id in ready_batch:
            node = self._get_node(snapshot, step_id)
            run_result = run_results.get(step_id)
            if run_result is None:
                # 理论上不会缺失；若缺失则兜底成失败结果，避免状态悬空。
                run_result = self._failed_run_result("Missing step execution result")

            # 汇总工具事件到全局事件流，便于上层追踪。
            state["all_tool_events"].extend(list(run_result.tool_events))
            if run_result.error:
                # 步骤执行失败：写失败状态、输出、错误明细，并尝试写入首错。
                node["state"] = STEP_STATE_FAILED
                node["output_text"] = run_result.output_text
                node["tool_events"] = list(run_result.tool_events)
                node["error"] = run_result.error
                self._set_first_error_once(state, run_result.error)
                if fail_fast and fail_fast_error is None:
                    fail_fast_error = run_result.error
            elif self._step_requires_user_input(run_result.output_text):
                # 需补充信息不视为失败，而是挂起当前任务等待用户补充。
                node_waiting_payload = self._build_waiting_payload(
                    step_id=step_id,
                    output_text=run_result.output_text,
                )
                node["state"] = STEP_STATE_WAITING_USER_INPUT
                node["output_text"] = run_result.output_text
                node["tool_events"] = list(run_result.tool_events)
                node["error"] = None
                if waiting_payload is None:
                    waiting_payload = node_waiting_payload
            else:
                # 步骤执行成功：写成功状态与输出。
                node["state"] = STEP_STATE_SUCCEEDED
                node["output_text"] = run_result.output_text
                node["tool_events"] = list(run_result.tool_events)
                node["error"] = None

            # 记录执行顺序，并把当前步骤结果持久化到 TaskManager。
            # 注意：这里写入的是步骤快照投影，保持与外部可观测结构一致。
            state["executed_steps"].append(step_id)
            self._task_manager.append_step_result(task_id, step_result_from_node(node))

        # 默认继续回到 select_ready_steps，挑选下一轮 ready 批次。
        next_action = ACTION_SELECT_STEPS
        if self._is_task_cancelled(task_id):
            # 出现“执行中取消”时，批次后收敛：
            # - 将剩余 pending/ready 标记 cancelled
            # - 记录首错（若为空）
            # - 路由到 review 收尾
            cancel_error = self._cancel_error()
            self._mark_remaining_cancelled(snapshot, cancel_error)
            self._set_first_error_once(state, cancel_error)
            next_action = ACTION_REVIEW
        elif waiting_payload is not None and state.get("first_error") is None:
            state["waiting_for_input"] = waiting_payload
            next_action = ACTION_FINALIZE
        elif fail_fast_error is not None:
            # fail_fast 命中后，阻断剩余未执行步骤，直接进入 review。
            self._mark_remaining_blocked(snapshot, fail_fast_error)
            next_action = ACTION_REVIEW

        # 再做一轮依赖阻断传播，确保下游状态与上游失败一致。
        self._mark_blocked_nodes(snapshot)
        if self._is_finished(snapshot):
            # 全图终态后进入 review，避免无意义循环。
            next_action = ACTION_REVIEW

        # 清空本轮批次并写入下一步路由动作。
        state["ready_batch"] = []
        state["next_action"] = next_action
        return state

    def _review_task(self, state: TaskFlowState) -> TaskFlowState:
        critic_agent = state["critic_agent"]
        critic_result = critic_agent.review_task(
            user_text=state["user_text"],
            plan_result=state["plan_result"],
            execution_graph=state["task_snapshot"],
        )
        state["critic_result"] = critic_result
        return state

    def _finalize_task(self, state: TaskFlowState) -> TaskFlowState:
        task_id = state["task_id"]
        snapshot = state["task_snapshot"]
        self._task_manager.set_task_snapshot(task_id, snapshot)

        task = self._task_manager.get_task(task_id)
        if task and task.state == TaskState.CANCELLED:
            return state

        waiting_for_input = state.get("waiting_for_input")
        if waiting_for_input:
            waiting_error = {
                "code": "TASK_NEED_USER_INPUT",
                "message": str(waiting_for_input.get("summary") or "Task requires additional user input"),
                "retryable": True,
            }
            self._task_manager.set_waiting_input(
                task_id,
                waiting_for_input=waiting_for_input,
                task_snapshot=snapshot,
                error=waiting_error,
            )
            return state

        if state.get("first_error"):
            self._task_manager.set_state(task_id, TaskState.FAILED, error=state["first_error"])
        else:
            self._task_manager.set_state(task_id, TaskState.SUCCEEDED)
        return state

    def _route_by_next_action(self, state: TaskFlowState) -> str:
        action = str(state.get("next_action") or ACTION_REVIEW)
        if action in {ACTION_RUN_STEPS, ACTION_SELECT_STEPS, ACTION_REVIEW, ACTION_FINALIZE}:
            return action
        return ACTION_REVIEW

    def _cancel_error(self) -> Dict[str, Any]:
        return {
            "code": "TASK_CANCELLED",
            "message": "Task cancelled by user",
            "retryable": True,
        }

    def _set_first_error_once(self, state: TaskFlowState, error: Optional[Dict[str, Any]]) -> None:
        if error is None:
            return
        if state.get("first_error") is None:
            state["first_error"] = error

    def _route_to_review(self, state: TaskFlowState) -> None:
        state["ready_batch"] = []
        state["next_action"] = ACTION_REVIEW

    def _cancel_and_route_review(self, *, state: TaskFlowState, snapshot: Dict[str, Any]) -> TaskFlowState:
        cancel_error = self._cancel_error()
        self._mark_remaining_cancelled(snapshot, cancel_error)
        self._set_first_error_once(state, cancel_error)
        self._route_to_review(state)
        return state

    def _build_steps(self, plan_result: PlanResult) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(plan_result.steps, start=1):
            raw_step_id = str(item.step_id or f"S{index}").strip() or f"S{index}"
            if raw_step_id in seen_ids:
                raise ValueError(f"Duplicated step_id: {raw_step_id}")
            seen_ids.add(raw_step_id)

            depends_on = [str(dep).strip() for dep in (item.depends_on or []) if str(dep).strip()]

            bindings: List[Dict[str, str]] = []
            for binding in item.input_bindings or []:
                if not isinstance(binding, dict):
                    continue
                source = str(binding.get("from") or "").strip()
                target = str(binding.get("to") or "").strip()
                if not source or not target:
                    continue
                bindings.append({"from": source, "to": target})

            steps.append(
                {
                    "step_id": raw_step_id,
                    "title": item.title,
                    "instruction": item.instruction,
                    "depends_on": depends_on,
                    "input_bindings": bindings,
                    "state": STEP_STATE_PENDING,
                    "output_text": "",
                    "tool_events": [],
                    "error": None,
                }
            )
        return steps

    def _build_topological_order(self, steps: List[Dict[str, Any]]) -> List[str]:
        indegree: Dict[str, int] = {str(step["step_id"]): 0 for step in steps}
        outgoing: Dict[str, List[str]] = {str(step["step_id"]): [] for step in steps}

        for step in steps:
            step_id = str(step["step_id"])
            visited: set[str] = set()
            for dep in list(step.get("depends_on") or []):
                dep_step = str(dep)
                if dep_step not in indegree:
                    raise ValueError(f"Unknown dependency: {dep_step} -> {step_id}")
                if dep_step == step_id:
                    raise ValueError(f"Self dependency is not allowed: {step_id}")
                if dep_step in visited:
                    continue
                visited.add(dep_step)
                indegree[step_id] += 1
                outgoing[dep_step].append(step_id)

        queue = [str(step["step_id"]) for step in steps if indegree[str(step["step_id"])] == 0]
        order: List[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for nxt in outgoing[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(steps):
            raise ValueError("Cycle detected in task graph")
        return order

    def _max_parallelism(self, snapshot: Dict[str, Any]) -> int:
        policy = dict(snapshot.get("policy") or {})
        raw = policy.get("max_parallelism", 1)
        try:
            value = int(raw)
        except Exception:
            value = 1
        return max(value, 1)

    def _fail_fast(self, snapshot: Dict[str, Any]) -> bool:
        policy = dict(snapshot.get("policy") or {})
        return bool(policy.get("fail_fast", True))

    def _normalize_policy(self, policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = dict(policy or {})
        max_parallelism = payload.get("max_parallelism", 1)
        fail_fast = payload.get("fail_fast", True)
        try:
            max_parallelism = int(max_parallelism)
        except Exception:
            max_parallelism = 1
        return {
            "max_parallelism": max(max_parallelism, 1),
            "fail_fast": bool(fail_fast),
        }

    def _ready_nodes(self, snapshot: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        nodes = [node for node in snapshot["nodes"] if node.get("state") == STEP_STATE_READY]
        rank = {step_id: idx for idx, step_id in enumerate(snapshot.get("topological_order") or [])}
        nodes.sort(key=lambda node: rank.get(str(node.get("step_id")), 10**9))
        if limit is None:
            return nodes
        return nodes[: max(int(limit), 0)]

    def _mark_remaining_cancelled(self, snapshot: Dict[str, Any], error: Optional[Dict[str, Any]]) -> None:
        for node in snapshot["nodes"]:
            if node.get("state") in {STEP_STATE_PENDING, STEP_STATE_READY, STEP_STATE_WAITING_USER_INPUT}:
                node["state"] = STEP_STATE_CANCELLED
                node["error"] = error

    def _mark_remaining_blocked(self, snapshot: Dict[str, Any], error: Optional[Dict[str, Any]]) -> None:
        for node in snapshot["nodes"]:
            if node.get("state") in {STEP_STATE_PENDING, STEP_STATE_READY}:
                node["state"] = STEP_STATE_BLOCKED
                node["error"] = error or node.get("error")

    def _mark_blocked_nodes(self, snapshot: Dict[str, Any]) -> None:
        index = self._node_index(snapshot)
        for node in snapshot["nodes"]:
            if node.get("state") not in {STEP_STATE_PENDING, STEP_STATE_READY}:
                continue
            if self._deps_failed(node, index):
                node["state"] = STEP_STATE_BLOCKED
                if node.get("error") is None:
                    node["error"] = {
                        "code": "UPSTREAM_FAILED",
                        "message": "Blocked due to failed dependency",
                        "retryable": False,
                    }

    def _refresh_ready_nodes(self, snapshot: Dict[str, Any]) -> None:
        """
        对于所有的节点，检查它的依赖节点，只要有一个处于 FAILED/CANCELLED/BLOCKED，就把它标记为 BLOCKED；
        如果全部依赖都成功了，就把它标记为 READY。
        相当于该节点入度为0了，可以开始调度
        """
        index = self._node_index(snapshot)
        for node in snapshot["nodes"]:
            if node.get("state") != STEP_STATE_PENDING:
                continue
            if self._deps_failed(node, index):
                node["state"] = STEP_STATE_BLOCKED
                if node.get("error") is None:
                    node["error"] = {
                        "code": "UPSTREAM_FAILED",
                        "message": "Blocked due to failed dependency",
                        "retryable": False,
                    }
                continue
            if self._deps_succeeded(node, index):
                node["state"] = STEP_STATE_READY

    def _deps_succeeded(self, node: Dict[str, Any], index: Dict[str, Dict[str, Any]]) -> bool:
        for dep in list(node.get("depends_on") or []):
            dep_node = index[str(dep)]
            if dep_node.get("state") != STEP_STATE_SUCCEEDED:
                return False
        return True

    def _deps_failed(self, node: Dict[str, Any], index: Dict[str, Dict[str, Any]]) -> bool:
        for dep in list(node.get("depends_on") or []):
            dep_node = index[str(dep)]
            if dep_node.get("state") in UPSTREAM_FAILED_STATES:
                return True
        return False

    def _is_finished(self, snapshot: Dict[str, Any]) -> bool:
        for node in snapshot["nodes"]:
            if node.get("state") not in TERMINAL_STATES:
                return False
        return True

    def _node_index(self, snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {str(node.get("step_id")): node for node in snapshot["nodes"]}

    def _get_node(self, snapshot: Dict[str, Any], step_id: str) -> Dict[str, Any]:
        for node in snapshot["nodes"]:
            if str(node.get("step_id")) == step_id:
                return node
        raise ValueError(f"Unknown step_id: {step_id}")

    def _is_task_cancelled(self, task_id: str) -> bool:
        task = self._task_manager.get_task(task_id)
        return bool(task and task.state == TaskState.CANCELLED)

    def _run_step_batch(
        self,
        *,
        jobs: List[Tuple[str, str]],
        executor_agent: Any,
        history: List[Dict[str, str]],
        session_id: str,
    ) -> Dict[str, ExecutorRunResult]:
        """
        并发执行一批可运行 step，并按 step_id 收集执行结果。

        参数说明：
        - jobs: [(step_id, step_input)] 形式的任务列表。
        - executor_agent: 负责真正执行 step 的 agent，需提供 run_task(...) 方法。
        - history/session_id: 传给 agent 的上下文参数。

        返回值：
        - Dict[step_id, ExecutorRunResult]。
        - 即使某个 step 抛异常，也会被转换为失败结果写入返回字典，
          保证上层调度始终拿到统一结构进行状态流转。
        """
        if not jobs:
            # 空批次直接返回，避免创建线程池带来无意义开销。
            return {}

        # 当前策略：一个 step 对应一个 worker，优先降低batch内等待时间。至少一个 worker
        max_workers = max(len(jobs), 1)
        results: Dict[str, ExecutorRunResult] = {}

        def _invoke_step(step_input: str) -> Tuple[ExecutorRunResult, int]:
            started = time.perf_counter()
            result = executor_agent.run_task(
                user_text=step_input,
                history=history,
                session_id=session_id,
            )
            return result, elapsed_ms(started)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # future -> step_id 的映射用于后续反查：
            # as_completed 返回的是 future，需要通过该映射定位所属节点。
            futures = {
                pool.submit(
                    _invoke_step,
                    step_input,
                ): step_id
                for step_id, step_input in jobs
            }

            # as_completed 按“完成先后”返回 future，而不是按提交顺序。
            # 这样可以让快任务先被处理并尽早写入结果，提升整体吞吐。
            for future in as_completed(futures):
                step_id = futures[future]
                try:
                    # 正常路径：收集执行器返回的结构化结果。
                    run_result, duration_ms = future.result()
                    results[step_id] = run_result
                    log_event(
                        logger,
                        logging.INFO,
                        "task.step.run.done",
                        f"步骤执行结束：{step_id}",
                        component="orchestrator",
                        step_id=step_id,
                        duration_ms=duration_ms,
                        ok=not bool(run_result.error),
                    )
                except Exception as exc:
                    # 异常路径：将异常统一包装为失败结果，避免异常向上传播
                    # 打断整批收集流程，保证其余已完成任务仍可被记录。
                    log_exception(
                        logger,
                        "task.step.run.error",
                        f"步骤执行异常：{step_id}",
                        component="orchestrator",
                        step_id=step_id,
                        error_code="TOOL_EXECUTION_ERROR",
                        retryable=True,
                    )
                    results[step_id] = self._failed_run_result(str(exc))
        return results

    def _failed_run_result(self, message: str) -> ExecutorRunResult:
        return ExecutorRunResult(
            output_text="任务执行失败。",
            tool_events=[],
            error={
                "code": "TOOL_EXECUTION_ERROR",
                "message": str(message),
                "retryable": True,
            },
        )

    def _step_requires_user_input(self, output_text: Any) -> bool:
        text = str(output_text or "")
        if not text:
            return False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("步骤状态"):
                continue
            _, _, rhs = line.replace("：", ":", 1).partition(":")
            status_text = rhs.strip().lower()
            return (
                "需补充信息" in status_text
                or "need_info" in status_text
                or "信息不足" in status_text
            )
        return False

    def _extract_summary_line(self, output_text: Any) -> str:
        text = str(output_text or "")
        if not text:
            return ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("结果摘要"):
                _, _, rhs = line.replace("：", ":", 1).partition(":")
                summary = rhs.strip()
                if summary:
                    return summary
        return ""

    def _extract_next_steps(self, output_text: Any) -> List[str]:
        text = str(output_text or "")
        if not text:
            return []
        lines = [line.rstrip() for line in text.splitlines()]
        capture = False
        next_steps: List[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("下一步建议"):
                capture = True
                continue
            if not capture:
                continue
            if line.startswith("关键依据") or line.startswith("产出详情") or line.startswith("限制与风险"):
                break
            if line.startswith("-"):
                item = line[1:].strip()
                if item:
                    next_steps.append(item)
            else:
                next_steps.append(line)
        return next_steps

    def _extract_required_fields(self, output_text: Any) -> List[str]:
        text = str(output_text or "")
        candidates = {
            "预算": "budget",
            "价格": "budget",
            "位置": "location",
            "区域": "location",
            "口味": "taste",
            "环境": "environment",
            "人数": "party_size",
            "时间": "time",
        }
        fields: List[str] = []
        for phrase, field in candidates.items():
            if phrase in text and field not in fields:
                fields.append(field)
        return fields

    def _build_waiting_payload(self, *, step_id: str, output_text: Any) -> Dict[str, Any]:
        summary = self._extract_summary_line(output_text)
        next_steps = self._extract_next_steps(output_text)
        clarify_question = next_steps[0] if next_steps else "请补充继续执行该步骤所需的信息。"
        return {
            "pending_step_id": step_id,
            "summary": summary or "信息不足，任务等待用户补充输入。",
            "clarify_question": clarify_question,
            "required_fields": self._extract_required_fields(output_text),
            "raw_reason": str(output_text or ""),
            "resume_count": 0,
        }

    def _prepare_resume_snapshot(
        self,
        *,
        snapshot: Dict[str, Any],
        waiting_payload: Dict[str, Any],
        user_reply: str,
    ) -> None:
        pending_step_id = str(waiting_payload.get("pending_step_id") or "").strip()
        if not pending_step_id:
            raise ValueError("Missing pending_step_id in waiting payload")
        node = self._get_node(snapshot, pending_step_id)
        node["state"] = STEP_STATE_PENDING
        node["error"] = None
        if user_reply:
            self._inject_user_reply_binding(node=node, user_reply=user_reply)

    def _inject_user_reply_binding(self, *, node: Dict[str, Any], user_reply: str) -> None:
        bindings = list(node.get("input_bindings") or [])
        kept: List[Dict[str, str]] = []
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            source = str(binding.get("from") or "")
            target = str(binding.get("to") or "")
            if target == "user_reply" and source.startswith("$const:"):
                continue
            kept.append({"from": source, "to": target})

        encoded = json.dumps(str(user_reply or ""), ensure_ascii=False)
        kept.append({"from": f"$const:{encoded}", "to": "user_reply"})
        node["input_bindings"] = kept

    def _has_waiting_input(self, snapshot: Dict[str, Any]) -> bool:
        for node in snapshot.get("nodes") or []:
            if node.get("state") == STEP_STATE_WAITING_USER_INPUT:
                return True
        return False

    def _project_step_results(self, task_snapshot: Dict[str, Any], executed_steps: List[str]) -> List[Dict[str, Any]]:
        index = {str(node.get("step_id")): node for node in task_snapshot.get("nodes") or []}
        results: List[Dict[str, Any]] = []
        for step_id in executed_steps:
            node = index.get(step_id)
            if node is None:
                continue
            results.append(step_result_from_node(node))
        return results
