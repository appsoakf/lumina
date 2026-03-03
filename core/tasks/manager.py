import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from core.protocols import TaskState
from core.tasks.record import TaskRecord
from core.tasks.store import TaskStore

logger = logging.getLogger(__name__)


class TaskManager:
    """Minimal task lifecycle manager (create/get/update/cancel/retry)."""

    _ALLOWED_TRANSITIONS = {
        TaskState.PENDING: {TaskState.RUNNING, TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED},
        TaskState.RUNNING: {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED},
        TaskState.SUCCEEDED: set(),
        TaskState.FAILED: {TaskState.PENDING},
        TaskState.CANCELLED: {TaskState.PENDING},
    }

    def __init__(self, store: Optional[TaskStore] = None):
        self.store = store or TaskStore()
        self._cache: Dict[str, TaskRecord] = {}
        self._mutation_seq = 0
        self._mutation_order: Dict[str, int] = {}
        self._lock = threading.RLock()

    def _mark_mutated(self, task_id: str) -> None:
        self._mutation_seq += 1
        self._mutation_order[task_id] = self._mutation_seq

    def _new_task_id(self) -> str:
        return f"t-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"

    def create_task(self, session_id: str, user_text: str) -> TaskRecord:
        with self._lock:
            task = TaskRecord(task_id=self._new_task_id(), session_id=session_id, user_text=user_text)
            self._cache[task.task_id] = task
            self._mark_mutated(task.task_id)
            self.store.save(task)
            return task

    def _to_epoch_ns(self, value: str) -> int:
        text = str(value or "").strip()
        if not text:
            return 0
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        except Exception:
            return 0

    def _task_mtime_ns(self, task_id: str) -> int:
        path = self.store.base_dir / f"{task_id}.json"
        try:
            return int(path.stat().st_mtime_ns)
        except Exception:
            return 0

    def _can_transition(self, current: TaskState, target: TaskState) -> bool:
        if current == target:
            return True
        allowed = self._ALLOWED_TRANSITIONS.get(current, set())
        return target in allowed

    def _get_task_locked(self, task_id: str) -> Optional[TaskRecord]:
        if task_id in self._cache:
            return self._cache[task_id]
        loaded = self.store.load(task_id)
        if loaded:
            self._cache[task_id] = loaded
        return loaded

    def _list_session_tasks_locked(self, session_id: str, limit: int = 50) -> List[TaskRecord]:
        rows = self.store.list_recent(limit=max(int(limit), 1))
        tasks: List[TaskRecord] = []
        for row in rows:
            if str(row.get("session_id", "")) != session_id:
                continue
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            task = self._get_task_locked(task_id)
            if task is not None:
                tasks.append(task)
        tasks.sort(
            key=lambda t: (
                self._mutation_order.get(t.task_id, 0),
                self._to_epoch_ns(t.updated_at),
                self._to_epoch_ns(t.created_at),
                self._task_mtime_ns(t.task_id),
            ),
            reverse=True,
        )
        return tasks

    def _list_session_tasks(self, session_id: str, limit: int = 50) -> List[TaskRecord]:
        with self._lock:
            return self._list_session_tasks_locked(session_id=session_id, limit=limit)

    def _select_current_task_locked(self, session_id: str) -> Optional[TaskRecord]:
        tasks = self._list_session_tasks_locked(session_id=session_id, limit=50)
        for state in (TaskState.RUNNING, TaskState.PENDING):
            for task in tasks:
                if task.state == state:
                    return task
        return None

    def _select_latest_retryable_task_locked(self, session_id: str) -> Optional[TaskRecord]:
        tasks = self._list_session_tasks_locked(session_id=session_id, limit=50)
        for task in tasks:
            if task.state in {TaskState.FAILED, TaskState.CANCELLED}:
                return task
        return None

    def _cancel_task_locked(self, task: TaskRecord) -> bool:
        if task.state == TaskState.CANCELLED:
            return False
        if not self._can_transition(task.state, TaskState.CANCELLED):
            return False
        task.state = TaskState.CANCELLED
        task.touch()
        self._mark_mutated(task.task_id)
        self.store.save(task)
        return True

    def _retry_task_locked(self, task: TaskRecord) -> bool:
        if not self._can_transition(task.state, TaskState.PENDING):
            return False
        task.state = TaskState.PENDING
        task.error = None
        task.step_results = []
        task.touch()
        self._mark_mutated(task.task_id)
        self.store.save(task)
        return True

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._get_task_locked(task_id)

    def set_plan(self, task_id: str, plan: Dict) -> None:
        """
        在planner_agent输出plan之后，调用此方法将plan保存到task中。
        最后，还会更新task的最后修改时间以及维护task在manager的最近变更顺序
        """
        with self._lock:
            task = self._get_task_locked(task_id)
            if not task:
                return
            task.plan = plan
            task.touch()
            self._mark_mutated(task_id)
            self.store.save(task)


    
    def set_state(self, task_id: str, state: TaskState, error: Dict = None) -> bool:
        """
        设置 task的state和error信息。
            * 开始处理一个task时，state为RUNNING, error为None
            * task执行过程出错时，state为FAILED, error为error信息
            * task被取消时，state为CANCELLED, error为None
        最后，还会更新task的最后修改时间以及维护task在manager的最近变更顺序
        """
        with self._lock:
            task = self._get_task_locked(task_id)
            if not task:
                return False
            if not self._can_transition(task.state, state):
                logger.warning(
                    "Rejected task state transition: task_id=%s from=%s to=%s",
                    task_id,
                    task.state.value,
                    state.value,
                )
                return False
            task.state = state
            task.error = error
            task.touch()
            self._mark_mutated(task_id)
            self.store.save(task)
            return True

    def append_step_result(self, task_id: str, step_result: Dict) -> None:
        with self._lock:
            task = self._get_task_locked(task_id)
            if not task:
                return
            task.step_results.append(step_result)
            task.touch()
            self._mark_mutated(task_id)
            self.store.save(task)

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._get_task_locked(task_id)
            if not task:
                return False
            return self._cancel_task_locked(task)

    def cancel_current_task(self, session_id: str) -> Tuple[Optional[TaskRecord], bool]:
        with self._lock:
            task = self._select_current_task_locked(session_id=session_id)
            if task is None:
                return None, False
            ok = self._cancel_task_locked(task)
            return task, ok

    def get_current_task(self, session_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._select_current_task_locked(session_id=session_id)

    def get_latest_retryable_task(self, session_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._select_latest_retryable_task_locked(session_id=session_id)

    def retry_latest_task(self, session_id: str) -> Tuple[Optional[TaskRecord], bool]:
        with self._lock:
            task = self._select_latest_retryable_task_locked(session_id=session_id)
            if task is None:
                return None, False
            ok = self._retry_task_locked(task)
            return task, ok

    def retry_task(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            task = self._get_task_locked(task_id)
            if not task:
                return None
            if task.state not in {TaskState.FAILED, TaskState.CANCELLED}:
                return task
            self._retry_task_locked(task)
            return task
