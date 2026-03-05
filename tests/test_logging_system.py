import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core.config import LoggingConfig, load_app_config
from core.utils import bind_log_context, log_event
from core.utils.logging_setup import setup_logging
from summarize_metrics import summarize


class LoggingSystemTests(unittest.TestCase):
    def tearDown(self):
        setup_logging(load_app_config().logging, force=True)

    def test_setup_logging_writes_contextual_event_json(self):
        with tempfile.TemporaryDirectory(prefix="lumina-log-test-") as tmp:
            log_dir = Path(tmp)
            config = LoggingConfig(
                level="INFO",
                format="json",
                log_dir=str(log_dir),
                log_file_name="human.log",
                event_file_name="events.jsonl",
                enable_console=False,
                enable_file=False,
                enable_event_file=True,
                slow_threshold_ms=1000,
                redact_user_text=True,
                user_text_preview_chars=120,
            )
            setup_logging(config, force=True)
            test_logger = logging.getLogger("tests.logging")

            with bind_log_context(session_id="s-1", round=2, task_id="t-1", step_id="S1"):
                log_event(
                    test_logger,
                    logging.INFO,
                    "test.event",
                    "logging test message",
                    duration_ms=12,
                    component="test",
                )

            for handler in logging.getLogger().handlers:
                handler.flush()

            event_file = log_dir / "events.jsonl"
            self.assertTrue(event_file.exists())
            rows = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(rows), 1)
            row = rows[-1]
            self.assertEqual(row.get("event"), "test.event")
            self.assertEqual(row.get("session_id"), "s-1")
            self.assertEqual(row.get("task_id"), "t-1")
            self.assertEqual(row.get("step_id"), "S1")
            self.assertEqual(row.get("duration_ms"), 12)

            # restore handlers before leaving temp dir to release file handles on Windows
            setup_logging(load_app_config().logging, force=True)

    def test_summarize_metrics_reads_event_latency(self):
        with tempfile.TemporaryDirectory(prefix="lumina-metrics-test-") as tmp:
            base = Path(tmp)
            trace_dir = base / "traces"
            task_dir = base / "tasks"
            event_dir = base / "logs"
            trace_dir.mkdir(parents=True, exist_ok=True)
            task_dir.mkdir(parents=True, exist_ok=True)
            event_dir.mkdir(parents=True, exist_ok=True)

            (trace_dir / "trace-demo.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "round_end", "payload": {"cost_sec": 1.5}}),
                        json.dumps({"event": "tts_error", "payload": {"code": "TTS_CONNECTION_ERROR"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (event_dir / "events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "ws.round.end", "duration_ms": 1400}),
                        json.dumps({"event": "llm.invoke.done", "duration_ms": 320}),
                        json.dumps({"event": "tool.call.done", "duration_ms": 88, "tool": "web_search"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = summarize(trace_dir=trace_dir, task_dir=task_dir, event_log_dir=event_dir)
            self.assertEqual(result["trace_files"], 1)
            self.assertEqual(result["event_log_files"], 1)
            self.assertGreaterEqual(result["round_count"], 2)
            self.assertGreater(result["latency_ms"]["round"]["p50_ms"], 0)
            self.assertGreater(result["latency_ms"]["llm_invoke"]["avg_ms"], 0)
            self.assertIn("TTS_CONNECTION_ERROR", result["error_code_counts"])


if __name__ == "__main__":
    unittest.main()
