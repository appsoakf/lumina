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
from core.utils.logging_setup import ConsoleEventFilter, ConsoleFlowFormatter, setup_logging
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

    def test_console_event_filter_keeps_perf_info_and_all_warnings(self):
        perf_filter = ConsoleEventFilter({"ws.round.start", "ws.round.end"})

        allowed_info = logging.makeLogRecord(
            {
                "levelno": logging.INFO,
                "levelname": "INFO",
                "event": "ws.round.start",
            }
        )
        blocked_info = logging.makeLogRecord(
            {
                "levelno": logging.INFO,
                "levelname": "INFO",
                "event": "llm.invoke.done",
            }
        )
        allowed_error = logging.makeLogRecord(
            {
                "levelno": logging.ERROR,
                "levelname": "ERROR",
                "event": "llm.invoke.done",
            }
        )

        self.assertTrue(perf_filter.filter(allowed_info))
        self.assertFalse(perf_filter.filter(blocked_info))
        self.assertTrue(perf_filter.filter(allowed_error))

    def test_console_flow_formatter_renders_user_friendly_summary(self):
        formatter = ConsoleFlowFormatter()
        record = logging.makeLogRecord(
            {
                "levelno": logging.INFO,
                "levelname": "INFO",
                "event": "ws.round.summary",
                "session_id": "s-1",
                "round": "2",
                "task_id": "t-1",
                "step_id": "-",
                "event_fields": {
                    "intent": "task",
                    "intent_ms": 160,
                    "task_run_ms": 1020,
                    "chat_llm_ms": 870,
                    "tts_ms": 300,
                    "round_total_ms": 4200,
                },
            }
        )

        line = formatter.format(record)
        self.assertIn("[本轮汇总]", line)
        self.assertIn("意图:任务", line)
        self.assertIn("意图识别:160ms", line)
        self.assertIn("编排执行:1020ms", line)
        self.assertIn("对话LLM:870ms", line)
        self.assertIn("总耗时:4200ms", line)
        self.assertNotIn("session=", line)
        self.assertNotIn("round=", line)
        self.assertNotIn("task=", line)

    def test_console_flow_formatter_renders_task_breakdown_events(self):
        formatter = ConsoleFlowFormatter()

        plan_record = logging.makeLogRecord(
            {
                "levelno": logging.INFO,
                "levelname": "INFO",
                "event": "task.plan.done",
                "round": "1",
                "event_fields": {
                    "resume_mode": False,
                    "step_count": 3,
                    "max_parallelism": 2,
                    "fail_fast": True,
                    "duration_ms": 1200,
                },
            }
        )
        plan_line = formatter.format(plan_record)
        self.assertIn("[任务规划]", plan_line)
        self.assertIn("3 个步骤", plan_line)
        self.assertIn("并行度 2", plan_line)

        step_record = logging.makeLogRecord(
            {
                "levelno": logging.INFO,
                "levelname": "INFO",
                "event": "executor.step.done",
                "step_id": "S1",
                "event_fields": {
                    "ok": True,
                    "rounds": 2,
                    "llm_calls": 2,
                    "llm_ms": 1800,
                    "tool_calls": 1,
                    "tool_ms": 220,
                    "duration_ms": 2200,
                },
            }
        )
        step_line = formatter.format(step_record)
        self.assertIn("[步骤细分]", step_line)
        self.assertIn("S1 成功", step_line)
        self.assertIn("LLM:2次/1800ms", step_line)
        self.assertIn("工具:1次/220ms", step_line)

        review_record = logging.makeLogRecord(
            {
                "levelno": logging.INFO,
                "levelname": "INFO",
                "event": "task.review.done",
                "event_fields": {
                    "quality": "pass",
                    "suggestion_count": 1,
                    "duration_ms": 640,
                },
            }
        )
        review_line = formatter.format(review_record)
        self.assertIn("[任务评审]", review_line)
        self.assertIn("评审通过", review_line)
        self.assertIn("建议 1 条", review_line)


if __name__ == "__main__":
    unittest.main()
