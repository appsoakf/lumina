import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_app_config
from core.paths import runtime_notes_dir, runtime_root, runtime_sessions_dir, runtime_tasks_dir, runtime_traces_dir


RUNTIME_DIRS = [
    runtime_root(),
    runtime_sessions_dir(),
    runtime_traces_dir(),
    runtime_tasks_dir(),
    runtime_notes_dir(),
]


def _check_runtime_dirs() -> List[Dict]:
    results = []
    for d in RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        ok = d.exists() and d.is_dir()
        writable = os.access(str(d), os.W_OK)
        results.append({"name": f"dir:{d}", "ok": ok and writable, "detail": "writable" if writable else "not_writable"})
    return results


def _check_tcp(url: str, timeout_sec: float = 2.0) -> dict:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return {"name": f"tcp:{url}", "ok": False, "detail": "invalid_url"}
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    sock = socket.socket()
    sock.settimeout(timeout_sec)
    try:
        sock.connect((host, port))
        return {"name": f"tcp:{host}:{port}", "ok": True, "detail": "reachable"}
    except Exception as exc:
        return {"name": f"tcp:{host}:{port}", "ok": False, "detail": str(exc)}
    finally:
        sock.close()


def _check_http(url: str, timeout_sec: float = 4.0) -> dict:
    try:
        resp = requests.get(url, timeout=timeout_sec)
        return {"name": f"http:{url}", "ok": True, "detail": f"status={resp.status_code}"}
    except Exception as exc:
        return {"name": f"http:{url}", "ok": False, "detail": str(exc)}


def run_health_check(skip_network: bool = False) -> Dict:
    checks: List[Dict] = []

    # config load
    try:
        cfg = load_app_config()
        checks.append({"name": "config:load_app_config", "ok": True, "detail": "loaded"})
    except Exception as exc:
        checks.append({"name": "config:load_app_config", "ok": False, "detail": str(exc)})
        cfg = None

    # runtime dirs
    checks.extend(_check_runtime_dirs())

    # network checks
    if (not skip_network) and cfg is not None:
        checks.append(_check_tcp(cfg.llm.chat_api_url))
        checks.append(_check_tcp(cfg.llm.translate_api_url))
        checks.append(_check_tcp(cfg.tts.gpt_sovits_url))

        checks.append(_check_http(cfg.llm.chat_api_url))
        checks.append(_check_http(cfg.tts.gpt_sovits_url))

    overall_ok = all(c["ok"] for c in checks)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_ok": overall_ok,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Lumina Phase 5-Lite health check")
    parser.add_argument("--skip-network", action="store_true", help="Skip remote endpoint checks")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    result = run_health_check(skip_network=args.skip_network)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"overall_ok={result['overall_ok']}")
        for c in result["checks"]:
            status = "OK" if c["ok"] else "FAIL"
            print(f"[{status}] {c['name']} -> {c['detail']}")

    return 0 if result["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
