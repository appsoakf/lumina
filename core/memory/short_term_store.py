import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.paths import runtime_sessions_dir


class ShortTermMemoryStore:
    def __init__(self, session_dir: Optional[Union[str, Path]] = None):
        self.session_dir = Path(session_dir) if session_dir is not None else runtime_sessions_dir()
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.json"

    def load_round(self, session_id: str) -> Dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            return {"session_id": session_id, "history": [], "metadata": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history")
            if not isinstance(history, list):
                history = []
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            return {
                "session_id": data.get("session_id", session_id),
                "saved_at": data.get("saved_at", ""),
                "history": history,
                "metadata": metadata,
            }
        except Exception:
            return {"session_id": session_id, "history": [], "metadata": {}}

    def load_history(self, session_id: str, limit_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        payload = self.load_round(session_id)
        history = payload.get("history", [])
        if not isinstance(history, list):
            return []
        if isinstance(limit_messages, int) and limit_messages > 0:
            return history[-limit_messages:]
        return history

    def save_round(self, session_id: str, history: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        path = self._session_path(session_id)
        payload = {
            "session_id": session_id,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "history": history,
            "metadata": metadata,
        }
        temp = str(path) + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temp, path)
        return str(path)
