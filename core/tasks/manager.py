from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from core.protocols import TaskState
from core.tasks.models import TaskRecord
from core.tasks.store import TaskStore


class TaskManager:
    """Minimal task lifecycle manager (create/get/update/cancel/retry)."""

    def __init__(self, store: Optional[TaskStore] = None):
        self.store = store or TaskStore()
        self._cache: Dict[str, TaskRecord] = {}

    def _new_task_id(self) -> str:
        return f"t-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"

    def create_task(self, session_id: str, user_text: str) -> TaskRecord:
        task = TaskRecord(task_id=self._new_task_id(), session_id=session_id, user_text=user_text)
        self._cache[task.task_id] = task
        self.store.save(task)
        return task

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        if task_id in self._cache:
            return self._cache[task_id]
        loaded = self.store.load(task_id)
        if loaded:
            self._cache[task_id] = loaded
        return loaded

    def set_plan(self, task_id: str, plan: Dict) -> None:
        task = self.get_task(task_id)
        if not task:
            return
        task.plan = plan
        task.touch()
        self.store.save(task)

    def set_state(self, task_id: str, state: TaskState, error: Dict = None) -> None:
        task = self.get_task(task_id)
        if not task:
            return
        task.state = state
        task.error = error
        task.touch()
        self.store.save(task)

    def append_step_result(self, task_id: str, step_result: Dict) -> None:
        task = self.get_task(task_id)
        if not task:
            return
        task.step_results.append(step_result)
        task.touch()
        self.store.save(task)

    def cancel_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        if task.state in {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED}:
            return False
        task.state = TaskState.CANCELLED
        task.touch()
        self.store.save(task)
        return True

    def retry_task(self, task_id: str) -> Optional[TaskRecord]:
        task = self.get_task(task_id)
        if not task:
            return None
        if task.state not in {TaskState.FAILED, TaskState.CANCELLED}:
            return task

        task.state = TaskState.PENDING
        task.error = None
        task.step_results = []
        task.touch()
        self.store.save(task)
        return task
