import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_base(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def project_root() -> Path:
    return ROOT_DIR


def runtime_root() -> Path:
    raw = str(os.environ.get("LUMINA_RUNTIME_DIR", "")).strip()
    if raw:
        return _resolve_base(raw)
    return ROOT_DIR / "runtime"


def runtime_sessions_dir() -> Path:
    return runtime_root() / "sessions"


def runtime_traces_dir() -> Path:
    return runtime_root() / "traces"


def runtime_tasks_dir() -> Path:
    return runtime_root() / "tasks"


def runtime_notes_dir() -> Path:
    return runtime_root() / "notes"


def runtime_memory_dir() -> Path:
    return runtime_root() / "memory"


def memory_db_path() -> Path:
    return runtime_memory_dir() / "memory.db"


def backups_root() -> Path:
    raw = str(os.environ.get("LUMINA_BACKUP_DIR", "")).strip()
    if raw:
        return _resolve_base(raw)
    return ROOT_DIR / "backups"
