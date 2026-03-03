import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import backups_root, runtime_root


DEFAULT_RUNTIME = runtime_root()
DEFAULT_BACKUP_DIR = backups_root()


def backup_runtime(runtime_dir: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    archive_base = backup_dir / f"lumina-runtime-{timestamp}"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(runtime_dir))
    return Path(archive_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup Lumina runtime directory")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    args = parser.parse_args()

    runtime_dir = Path(args.runtime_dir)
    backup_dir = Path(args.backup_dir)

    if not runtime_dir.exists():
        print(f"runtime directory not found: {runtime_dir}")
        return 1

    archive = backup_runtime(runtime_dir=runtime_dir, backup_dir=backup_dir)
    print(f"backup_created={archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
