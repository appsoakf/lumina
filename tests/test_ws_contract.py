import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils.errors import ErrorCode
from service.pet.ws_contract import parse_user_text


class WebSocketContractTests(unittest.TestCase):
    def test_parse_user_text_success(self):
        text, err = parse_user_text('{"content":"  hello  "}')
        self.assertEqual(text, "hello")
        self.assertIsNone(err)

    def test_parse_user_text_accepts_missing_content_as_empty(self):
        text, err = parse_user_text("{}")
        self.assertEqual(text, "")
        self.assertIsNone(err)

    def test_parse_user_text_rejects_invalid_json(self):
        text, err = parse_user_text("{")
        self.assertIsNone(text)
        self.assertIsNotNone(err)
        self.assertEqual(err.code, ErrorCode.WEBSOCKET_ERROR)

    def test_parse_user_text_rejects_non_object_payload(self):
        text, err = parse_user_text("[]")
        self.assertIsNone(text)
        self.assertIsNotNone(err)
        self.assertEqual(err.code, ErrorCode.WEBSOCKET_ERROR)

    def test_parse_user_text_rejects_non_string_content(self):
        text, err = parse_user_text('{"content":123}')
        self.assertIsNone(text)
        self.assertIsNotNone(err)
        self.assertEqual(err.code, ErrorCode.WEBSOCKET_ERROR)


if __name__ == "__main__":
    unittest.main()
