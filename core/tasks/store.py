import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from core.tasks.models import TaskRecord


class TaskStore:
    def __init__(self, base_dir: str = "D:/lumina/runtime/tasks"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path:
        return self.base_dir / f"{task_id}.json"

    def save(self, task: TaskRecord) -> str:
        path = self._task_path(task.task_id)
        temp = str(path) + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(temp, path)
        return str(path)

    def load(self, task_id: str) -> Optional[TaskRecord]:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TaskRecord.from_dict(data)

    def list_recent(self, limit: int = 20) -> List[Dict]:
        rows = []
        for p in self.base_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rows.append(data)
            except Exception:
                continue
        rows.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return rows[:limit]
