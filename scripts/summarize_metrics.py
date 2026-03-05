import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import runtime_root, runtime_tasks_dir, runtime_traces_dir


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


def _latency_stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }

    ordered = sorted(float(v) for v in samples)

    def _pick(percentile: float) -> float:
        if not ordered:
            return 0.0
        index = int(round((len(ordered) - 1) * percentile))
        index = max(min(index, len(ordered) - 1), 0)
        return round(ordered[index], 3)

    avg_ms = round(sum(ordered) / len(ordered), 3)
    return {
        "count": len(ordered),
        "avg_ms": avg_ms,
        "p50_ms": _pick(0.50),
        "p95_ms": _pick(0.95),
        "p99_ms": _pick(0.99),
        "max_ms": round(ordered[-1], 3),
    }


def summarize(trace_dir: Path, task_dir: Path, event_log_dir: Path) -> dict:
    event_counter = Counter()
    error_code_counter = Counter()
    tool_counter = Counter()

    round_durations_ms: List[float] = []
    llm_durations_ms: List[float] = []
    tool_durations_ms: List[float] = []
    tts_durations_ms: List[float] = []
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
                    round_durations_ms.append(float(cost) * 1000.0)

    event_log_files = list(event_log_dir.glob("*.jsonl")) if event_log_dir.exists() else []
    for ef in event_log_files:
        for row in _iter_jsonl(ef):
            event = str(row.get("event", "")).strip()
            if event:
                event_counter[event] += 1

            error_code = str(row.get("error_code", "")).strip()
            if error_code:
                error_code_counter[error_code] += 1

            tool_name = str(row.get("tool", "")).strip()
            if tool_name and event.startswith("tool.call"):
                tool_counter[tool_name] += 1

            duration_ms = row.get("duration_ms")
            if not isinstance(duration_ms, (int, float)):
                continue

            value = float(duration_ms)
            if event == "ws.round.end":
                round_durations_ms.append(value)
            elif event == "llm.invoke.done":
                llm_durations_ms.append(value)
            elif event.startswith("tool.call"):
                tool_durations_ms.append(value)
            elif event == "tts.stream.end":
                tts_durations_ms.append(value)

    task_files = list(task_dir.glob("*.json")) if task_dir.exists() else []
    task_state_counter = Counter()
    for tf in task_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            task_state_counter[data.get("state", "unknown")] += 1
        except Exception:
            continue

    round_latency = _latency_stats(round_durations_ms)
    avg_round_sec = round(round_latency["avg_ms"] / 1000.0, 3) if round_latency["count"] > 0 else 0.0

    return {
        "trace_files": len(trace_files),
        "event_log_files": len(event_log_files),
        "task_files": len(task_files),
        "event_counts": dict(event_counter),
        "error_code_counts": dict(error_code_counter),
        "tool_call_counts": dict(tool_counter),
        "task_state_counts": dict(task_state_counter),
        "round_count": round_latency["count"],
        "avg_round_sec": avg_round_sec,
        "latency_ms": {
            "round": round_latency,
            "llm_invoke": _latency_stats(llm_durations_ms),
            "tool_call": _latency_stats(tool_durations_ms),
            "tts_stream": _latency_stats(tts_durations_ms),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Lumina local metrics from traces/tasks")
    parser.add_argument("--trace-dir", default=str(runtime_traces_dir()))
    parser.add_argument("--task-dir", default=str(runtime_tasks_dir()))
    parser.add_argument("--event-log-dir", default=str(runtime_root() / "logs"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = summarize(Path(args.trace_dir), Path(args.task_dir), Path(args.event_log_dir))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            "trace_files="
            f"{result['trace_files']} event_log_files={result['event_log_files']} task_files={result['task_files']}"
        )
        print(f"round_count={result['round_count']} avg_round_sec={result['avg_round_sec']}")
        print(f"latency_ms={result['latency_ms']}")
        print(f"task_state_counts={result['task_state_counts']}")
        print(f"error_code_counts={result['error_code_counts']}")
        print(f"tool_call_counts={result['tool_call_counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
