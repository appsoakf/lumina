import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import project_root


def _resolve_runtime_dir(raw: str) -> Path:
    candidate = Path(str(raw or "").strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root() / candidate).resolve()


def prepare_fresh_runtime(runtime_dir: Path, project_root_dir: Optional[Path] = None) -> Path:
    root = (project_root_dir or project_root()).resolve()
    target = runtime_dir.resolve()

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"runtime directory must be inside project root: root={root} target={target}"
        ) from exc

    if target.exists():
        shutil.rmtree(target, ignore_errors=False)
    target.mkdir(parents=True, exist_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PET service for E2E tests with a fresh runtime directory."
    )
    parser.add_argument(
        "--runtime-dir",
        default="runtime/e2e/current",
        help="Runtime directory used during this E2E run (relative to project root by default).",
    )
    args = parser.parse_args()

    runtime_dir = _resolve_runtime_dir(args.runtime_dir)
    fresh_runtime = prepare_fresh_runtime(runtime_dir)
    os.environ["LUMINA_RUNTIME_DIR"] = str(fresh_runtime)

    print(f"[e2e] using fresh runtime: {fresh_runtime}")

    # Import after runtime env is ready, so all runtime-dependent components
    # read the E2E runtime root on module initialization.
    from service.pet.main import run_pet

    run_pet()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
