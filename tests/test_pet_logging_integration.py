import json
import logging
import sys
import tempfile
import unittest
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import LoggingConfig, load_app_config
from core.llm.main import TranslateResult
from core.utils.logging_setup import setup_logging
from service.pet.main import TraceLogger, handle_bot_reply
import service.pet.main as pet_main


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    def send(self, payload: str):
        self.messages.append(payload)


class _FakeOrchestrator:
    def __init__(self):
        self.round_records = []

    def handle_user_message(self, user_text: str, session_id: str):
        _ = user_text, session_id
        return SimpleNamespace(
            final_reply='{"emotion":"平静","intensity":1}\n你好。',
            intent=SimpleNamespace(value="chat"),
            meta={"agent_chain": ["chat_agent"], "task_mode": False},
            executor_result=None,
        )

    def record_session_round(self, session_id, user_text, assistant_reply, metadata):
        self.round_records.append(
            {
                "session_id": session_id,
                "user_text": user_text,
                "assistant_reply": assistant_reply,
                "metadata": metadata,
            }
        )

    def close(self):
        return None


class _FakeTranslator:
    def translate_with_status(self, text: str) -> TranslateResult:
        return TranslateResult(text=text, error=None)


class _FakeTTSEngine:
    def synthesize_streaming(self, request):
        _ = request
        return {
            "success": True,
            "audio_stream": [b"\x01\x02\x03"],
            "error_code": None,
            "retryable": False,
        }


class _FakeEmotionEngine:
    def parse_leading_json(self, raw: str):
        _ = raw
        return "平静", "你好。", "1"

    def get_ref_audio_intensity(self, emotion: str, intensity: str) -> str:
        _ = emotion, intensity
        return "ref_audios/calm.wav"

    def get_prompt_text_intensity(self, emotion: str, intensity: str) -> str:
        _ = emotion, intensity
        return "テスト用プロンプト"


class _ImmediateExecutor:
    class _Future:
        def result(self):
            return None

    def submit(self, fn, *args, **kwargs):
        run_ctx = copy_context()
        run_ctx.run(fn, *args, **kwargs)
        return self._Future()

    def shutdown(self, *args, **kwargs):
        _ = args, kwargs
        return None


class PetLoggingIntegrationTests(unittest.TestCase):
    def tearDown(self):
        setup_logging(load_app_config().logging, force=True)

    def test_handle_bot_reply_emits_structured_events(self):
        with tempfile.TemporaryDirectory(prefix="lumina-pet-log-it-") as tmp:
            temp_root = Path(tmp)
            temp_events_dir = temp_root / "logs"
            temp_traces_dir = temp_root / "traces"
            temp_events_dir.mkdir(parents=True, exist_ok=True)
            temp_traces_dir.mkdir(parents=True, exist_ok=True)

            cfg = LoggingConfig(
                level="INFO",
                format="json",
                log_dir=str(temp_events_dir),
                log_file_name="lumina.log",
                event_file_name="events.jsonl",
                enable_console=False,
                enable_file=False,
                enable_event_file=True,
                slow_threshold_ms=1000,
                redact_user_text=True,
                user_text_preview_chars=120,
            )
            setup_logging(cfg, force=True)

            fake_ws = _FakeWebSocket()
            fake_trace = TraceLogger(trace_dir=temp_traces_dir, session_id="it-session")
            fake_orchestrator = _FakeOrchestrator()
            try:
                with patch.object(pet_main, "orchestrator", fake_orchestrator), \
                    patch.object(pet_main, "ENABLE_TRANSLATION", True), \
                    patch.object(pet_main, "ENABLE_TTS", True), \
                    patch.object(pet_main, "translator", _FakeTranslator()), \
                    patch.object(pet_main, "tts", _FakeTTSEngine()), \
                    patch.object(pet_main, "emotion_engine", _FakeEmotionEngine()), \
                    patch.object(pet_main, "executor", _ImmediateExecutor()):
                    handle_bot_reply(
                        ws=fake_ws,
                        user_text="你好，帮我测试日志链路",
                        session_id="it-session",
                        trace=fake_trace,
                        round_num=1,
                    )

                for handler in logging.getLogger().handlers:
                    handler.flush()

                event_file = temp_events_dir / "events.jsonl"
                self.assertTrue(event_file.exists())
                rows = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertGreater(len(rows), 0)

                event_names = {row.get("event") for row in rows}
                self.assertIn("ws.round.start", event_names)
                self.assertIn("orchestrator.route.done", event_names)
                self.assertIn("pipeline.translate.done", event_names)
                self.assertIn("pipeline.tts.stream.done", event_names)
                self.assertIn("ws.round.end", event_names)

                round_end_rows = [row for row in rows if row.get("event") == "ws.round.end"]
                self.assertTrue(round_end_rows)
                latest = round_end_rows[-1]
                self.assertEqual(latest.get("session_id"), "it-session")
                self.assertEqual(latest.get("round"), "1")
                self.assertGreaterEqual(int(latest.get("duration_ms", 0)), 0)

                ws_types = [json.loads(item).get("type") for item in fake_ws.messages]
                self.assertIn("emotion_text", ws_types)
                self.assertIn("audio_chunk", ws_types)
                self.assertIn("audio_done", ws_types)
                self.assertIn("done", ws_types)
            finally:
                fake_trace.close()
                # release file handles before TemporaryDirectory cleanup
                setup_logging(load_app_config().logging, force=True)


if __name__ == "__main__":
    unittest.main()
