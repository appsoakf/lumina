import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import runtime_root


DEFAULT_RUNTIME = runtime_root()
TARGET_SUBDIRS = ["sessions", "traces", "tasks", "notes"]


def cleanup(runtime_dir: Path, keep_days: int, dry_run: bool = True) -> dict:
    now = time.time()
    threshold = now - (keep_days * 86400)

    deleted = []
    skipped = []

    for sub in TARGET_SUBDIRS:
        subdir = runtime_dir / sub
        if not subdir.exists():
            continue

        for p in subdir.rglob("*"):
            if not p.is_file():
                continue
            mtime = p.stat().st_mtime
            if mtime < threshold:
                if dry_run:
                    skipped.append(str(p))
                else:
                    p.unlink(missing_ok=True)
                    deleted.append(str(p))

    return {
        "keep_days": keep_days,
        "dry_run": dry_run,
        "deleted_count": len(deleted),
        "candidates_count": len(skipped) if dry_run else len(deleted),
        "deleted": deleted,
        "candidates": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup old Lumina runtime files")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--keep-days", type=int, default=7)
    parser.add_argument("--apply", action="store_true", help="Apply deletion (default is dry-run)")
    args = parser.parse_args()

    result = cleanup(
        runtime_dir=Path(args.runtime_dir),
        keep_days=args.keep_days,
        dry_run=(not args.apply),
    )

    print(
        f"dry_run={result['dry_run']} keep_days={result['keep_days']} "
        f"candidates={result['candidates_count']} deleted={result['deleted_count']}"
    )

    # Print first few files for quick inspection
    items = result["candidates"] if result["dry_run"] else result["deleted"]
    for p in items[:20]:
        print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
