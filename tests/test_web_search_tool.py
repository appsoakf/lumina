import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.tools import ToolContext, WebSearchTool, build_default_registry


class _FakeResponse:
    def __init__(self, payload, *, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.content = b"{}"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"http status {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, *, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        return self.response


class WebSearchToolTests(unittest.TestCase):
    def test_build_default_registry_contains_web_search_schema(self):
        registry = build_default_registry()
        names = [item["function"]["name"] for item in registry.list_schemas()]
        self.assertIn("web_search", names)

    def test_web_search_duckduckgo_success(self):
        payload = {
            "Heading": "Python",
            "AbstractText": "Python is a programming language.",
            "AbstractURL": "https://www.python.org",
            "AbstractSource": "Wikipedia",
            "RelatedTopics": [
                {
                    "Text": "Python docs - Official docs",
                    "FirstURL": "https://docs.python.org/3/",
                },
                {
                    "Name": "nested",
                    "Topics": [
                        {
                            "Text": "PEP index",
                            "FirstURL": "https://peps.python.org/",
                        }
                    ],
                },
            ],
        }
        fake_session = _FakeSession(response=_FakeResponse(payload))
        with patch.dict(os.environ, {"LUMINA_WEB_SEARCH_PROVIDER": "duckduckgo"}, clear=False):
            tool = WebSearchTool(session=fake_session)
            result = tool.run(ctx=ToolContext(session_id="s1"), text="python", top_k=2)

        self.assertTrue(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["query"], "python")
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["results"][0]["id"], "R1")
        self.assertIn("python", fake_session.calls[0]["params"]["q"].lower())

    def test_web_search_invalid_input(self):
        tool = WebSearchTool(session=_FakeSession(response=_FakeResponse({})))
        result = tool.run(ctx=ToolContext(session_id="s1"), text="  ")
        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_BAD_INPUT")

    def test_web_search_timeout_error(self):
        timeout_session = _FakeSession(exc=requests.Timeout("timeout"))
        with patch.dict(os.environ, {"LUMINA_WEB_SEARCH_PROVIDER": "duckduckgo"}, clear=False):
            tool = WebSearchTool(session=timeout_session)
            result = tool.run(ctx=ToolContext(session_id="s1"), text="python")

        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_TIMEOUT")
        self.assertTrue(body["retryable"])

    def test_web_search_searxng_success(self):
        payload = {
            "results": [
                {
                    "title": "Python",
                    "url": "https://www.python.org",
                    "content": "Official site.",
                    "engine": "searxng",
                    "publishedDate": "2025-01-01",
                }
            ]
        }
        fake_session = _FakeSession(response=_FakeResponse(payload))
        with patch.dict(
            os.environ,
            {
                "LUMINA_WEB_SEARCH_PROVIDER": "searxng",
                "LUMINA_WEB_SEARCH_ENDPOINT": "http://127.0.0.1:8080/search",
            },
            clear=False,
        ):
            tool = WebSearchTool(session=fake_session)
            result = tool.run(ctx=ToolContext(session_id="s1"), text="python", top_k=1, language="en")

        self.assertTrue(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["provider"], "searxng")
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["results"][0]["published_at"], "2025-01-01")


if __name__ == "__main__":
    unittest.main()

