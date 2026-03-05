import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm.main import TranslateResult
from core.config import _build_service_config
from service.pet.pipeline import EmotionContext, SentenceSlot
import service.pet.main as pet_main


class _FakeTrace:
    def __init__(self):
        self.events = []

    def log(self, event: str, payload: dict):
        self.events.append((event, payload))


class _FailOnCallTranslator:
    def translate_with_status(self, text: str):
        _ = text
        raise AssertionError("translator should not be called when translation is disabled")


class _StaticTranslator:
    def __init__(self, translated_text: str):
        self.translated_text = translated_text
        self.calls = 0

    def translate_with_status(self, text: str):
        _ = text
        self.calls += 1
        return TranslateResult(text=self.translated_text, error=None)


class _FailOnCallTTS:
    def synthesize_streaming(self, request):
        _ = request
        raise AssertionError("tts should not be called when tts is disabled")


class PetFeatureSwitchTests(unittest.TestCase):
    def test_service_switch_defaults_to_off(self):
        cfg = _build_service_config(
            {
                "pet_name": "pet",
                "username": "user",
                "server_address": "0.0.0.0",
                "server_port": 8080,
            }
        )
        self.assertFalse(cfg.enable_translation)
        self.assertFalse(cfg.enable_tts)

    def test_service_switch_parses_boolean_values(self):
        cfg = _build_service_config(
            {
                "pet_name": "pet",
                "username": "user",
                "server_address": "127.0.0.1",
                "server_port": 9000,
                "enable_translation": "true",
                "enable_tts": 1,
            }
        )
        self.assertTrue(cfg.enable_translation)
        self.assertTrue(cfg.enable_tts)

    def test_sentence_worker_skips_translation_when_switch_off(self):
        slot = SentenceSlot(index=0, chinese_text="你好。")
        emotion_ctx = EmotionContext()
        trace = _FakeTrace()

        with patch.object(pet_main, "ENABLE_TRANSLATION", False), \
            patch.object(pet_main, "ENABLE_TTS", False), \
            patch.object(pet_main, "translator", _FailOnCallTranslator()), \
            patch.object(pet_main, "tts", _FailOnCallTTS()):
            pet_main.sentence_worker(
                slot=slot,
                emotion_ctx=emotion_ctx,
                trace=trace,
                session_id="test-session",
                round_num=1,
            )

        self.assertEqual(slot.japanese_text, "你好。")
        self.assertIsNone(slot.error)
        self.assertTrue(slot.done.is_set())
        self.assertIsNone(slot.chunk_queue.get(timeout=0.2))
        self.assertFalse(any(event == "translate_error" for event, _ in trace.events))

    def test_sentence_worker_skips_tts_when_switch_off(self):
        slot = SentenceSlot(index=0, chinese_text="你好。")
        emotion_ctx = EmotionContext()
        trace = _FakeTrace()
        fake_translator = _StaticTranslator("こんにちは。")

        with patch.object(pet_main, "ENABLE_TRANSLATION", True), \
            patch.object(pet_main, "ENABLE_TTS", False), \
            patch.object(pet_main, "translator", fake_translator), \
            patch.object(pet_main, "tts", _FailOnCallTTS()):
            pet_main.sentence_worker(
                slot=slot,
                emotion_ctx=emotion_ctx,
                trace=trace,
                session_id="test-session",
                round_num=1,
            )

        self.assertEqual(fake_translator.calls, 1)
        self.assertEqual(slot.japanese_text, "こんにちは。")
        self.assertIsNone(slot.error)
        self.assertTrue(slot.done.is_set())
        self.assertIsNone(slot.chunk_queue.get(timeout=0.2))
        self.assertFalse(any(event == "tts_error" for event, _ in trace.events))


if __name__ == "__main__":
    unittest.main()
