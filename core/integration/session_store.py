import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class SessionStore:
    def __init__(self, session_dir: str = "D:/lumina/runtime/sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save_round(self, session_id: str, history: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        path = self.session_dir / f"{session_id}.json"
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
