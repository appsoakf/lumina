import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import runtime_tasks_dir, runtime_traces_dir


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def summarize(trace_dir: Path, task_dir: Path) -> dict:
    event_counter = Counter()
    error_code_counter = Counter()
    tool_counter = Counter()

    round_durations = []
    trace_files = list(trace_dir.glob("trace-*.jsonl")) if trace_dir.exists() else []

    for tf in trace_files:
        for row in _iter_jsonl(tf):
            event = row.get("event", "unknown")
            event_counter[event] += 1
            payload = row.get("payload", {}) or {}

            if event in {"error", "executor_error", "translate_error", "tts_error", "worker_error"}:
                code = payload.get("code", "UNKNOWN")
                error_code_counter[code] += 1

            if event == "tool_event":
                tool_name = payload.get("tool", "unknown")
                tool_counter[tool_name] += 1

            if event == "round_end":
                cost = payload.get("cost_sec")
                if isinstance(cost, (int, float)):
                    round_durations.append(float(cost))

    task_files = list(task_dir.glob("*.json")) if task_dir.exists() else []
    task_state_counter = Counter()
    for tf in task_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            task_state_counter[data.get("state", "unknown")] += 1
        except Exception:
            continue

    avg_round_sec = round(sum(round_durations) / len(round_durations), 3) if round_durations else 0.0

    return {
        "trace_files": len(trace_files),
        "task_files": len(task_files),
        "event_counts": dict(event_counter),
        "error_code_counts": dict(error_code_counter),
        "tool_call_counts": dict(tool_counter),
        "task_state_counts": dict(task_state_counter),
        "round_count": len(round_durations),
        "avg_round_sec": avg_round_sec,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Lumina local metrics from traces/tasks")
    parser.add_argument("--trace-dir", default=str(runtime_traces_dir()))
    parser.add_argument("--task-dir", default=str(runtime_tasks_dir()))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = summarize(Path(args.trace_dir), Path(args.task_dir))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"trace_files={result['trace_files']} task_files={result['task_files']}")
        print(f"round_count={result['round_count']} avg_round_sec={result['avg_round_sec']}")
        print(f"task_state_counts={result['task_state_counts']}")
        print(f"error_code_counts={result['error_code_counts']}")
        print(f"tool_call_counts={result['tool_call_counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
