import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm.main import TranslateEngine
from core.utils.errors import ErrorCode


class _ChatLLMStub:
    def __init__(self, content=None, exc=None):
        self.content = content
        self.exc = exc

    def invoke(self, messages, temperature):
        _ = messages, temperature
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


def _build_engine(chat_llm):
    engine = TranslateEngine.__new__(TranslateEngine)
    engine.chat_llm = chat_llm
    engine.translate_prompt = "translate prompt"
    return engine


class TranslateEngineTests(unittest.TestCase):
    def test_translate_with_status_success(self):
        engine = _build_engine(_ChatLLMStub(content="こんにちは"))
        result = engine.translate_with_status("你好")

        self.assertTrue(result.ok)
        self.assertIsNone(result.error)
        self.assertEqual(result.text, "こんにちは")

    def test_translate_with_status_empty_result_returns_error(self):
        engine = _build_engine(_ChatLLMStub(content=""))
        result = engine.translate_with_status("你好")

        self.assertFalse(result.ok)
        self.assertEqual(result.text, "")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, ErrorCode.TRANSLATE_EMPTY_RESULT)

    def test_translate_with_status_exception_returns_error(self):
        engine = _build_engine(_ChatLLMStub(exc=RuntimeError("network down")))
        result = engine.translate_with_status("你好")

        self.assertFalse(result.ok)
        self.assertEqual(result.text, "")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, ErrorCode.TRANSLATE_API_ERROR)

    def test_translate_keeps_compatibility(self):
        engine = _build_engine(_ChatLLMStub(content="おはよう"))
        self.assertEqual(engine.translate("早上好"), "おはよう")


if __name__ == "__main__":
    unittest.main()
