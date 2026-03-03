from typing import TYPE_CHECKING, Dict, List, Optional

from core.protocols import OrchestrationResult, RoutingIntent
from core.tasks import TaskManager, TaskRecord

if TYPE_CHECKING:
    from core.agentic.chat_agent import ChatAgent

_SHORTCUT_ACTIONS = {
    "/cancel": "cancel",
    "/retry": "retry",
}


def parse_task_shortcut(text: str) -> Optional[str]:
    normalized = (text or "").strip().lower()
    return _SHORTCUT_ACTIONS.get(normalized)


def resolve_cancel_target(task_manager: TaskManager, session_id: str) -> Optional[TaskRecord]:
    return task_manager.get_current_task(session_id=session_id)


def resolve_retry_target(task_manager: TaskManager, session_id: str) -> Optional[TaskRecord]:
    return task_manager.get_latest_retryable_task(session_id=session_id)


def execute_task_shortcut(
    *,
    user_text: str,
    session_id: str,
    history: List[Dict[str, str]],
    task_manager: TaskManager,
    chat_agent: "ChatAgent",
) -> Optional[OrchestrationResult]:
    action = parse_task_shortcut(user_text)
    if action is None:
        return None

    task_id = ""
    if action == "cancel":
        task, ok = task_manager.cancel_current_task(session_id=session_id)
        if task is None:
            result_text = "当前没有可取消的任务。"
        else:
            task_id = task.task_id
            result_text = f"已取消当前任务（{task.task_id}）。" if ok else f"当前任务 {task.task_id} 不可取消。"
    elif action == "retry":
        task, ok = task_manager.retry_latest_task(session_id=session_id)
        if task is None:
            result_text = "当前没有可重试的任务。"
        else:
            task_id = task.task_id
            result_text = f"已重试最近任务（{task.task_id}）。" if ok else f"任务 {task.task_id} 重试失败。"
    else:
        result_text = "未知任务快捷指令。"

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
            "task_shortcut": action,
            "task_command": action,
            "task_id": task_id,
            "task_mode": False,
            "agent_chain": ["chat_agent"],
        },
    )
