import json
import sys
import unittest
from pathlib import Path

import requests
from duckduckgo_search.exceptions import TimeoutException as DDGTimeoutException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DuckDuckGoConfig, SerpApiConfig, WebSearchConfig
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


class _FakeDDGS:
    def __init__(self, *, rows=None, exc=None):
        self.rows = rows or []
        self.exc = exc
        self.calls = []

    def text(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return list(self.rows)


class WebSearchToolTests(unittest.TestCase):
    def _cfg(
        self,
        *,
        provider: str = "duckduckgo",
        fallback_provider: str = "serpapi",
        serpapi_api_key: str = "test-key",
    ) -> WebSearchConfig:
        return WebSearchConfig(
            provider=provider,
            fallback_provider=fallback_provider,
            timeout_sec=8.0,
            max_top_k=5,
            duckduckgo=DuckDuckGoConfig(
                region="wt-wt",
                safesearch="moderate",
                backend="auto",
                timelimit="",
            ),
            serpapi=SerpApiConfig(
                endpoint="https://serpapi.com/search.json",
                api_key=serpapi_api_key,
                engine="google",
                gl="us",
                hl="en",
                tbm="nws",
            ),
        )

    def test_build_default_registry_contains_web_search_schema(self):
        registry = build_default_registry()
        names = [item["function"]["name"] for item in registry.list_schemas()]
        self.assertIn("web_search", names)

    def test_web_search_duckduckgo_success(self):
        fake_ddgs = _FakeDDGS(
            rows=[
                {
                    "title": "Python Official",
                    "href": "https://www.python.org/",
                    "body": "Python language home page.",
                    "date": "2026-03-04",
                },
                {
                    "title": "Python Docs",
                    "href": "https://docs.python.org/3/",
                    "body": "Official docs.",
                },
            ]
        )
        fake_session = _FakeSession(response=_FakeResponse({}))
        tool = WebSearchTool(
            config=self._cfg(provider="duckduckgo", fallback_provider="none"),
            session=fake_session,
            ddgs_factory=lambda: fake_ddgs,
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="python", top_k=2)

        self.assertTrue(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["query"], "python")
        self.assertEqual(body["provider"], "duckduckgo")
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["results"][0]["id"], "R1")
        self.assertEqual(len(fake_ddgs.calls), 1)
        self.assertEqual(len(fake_session.calls), 0)

    def test_web_search_invalid_input(self):
        tool = WebSearchTool(
            config=self._cfg(),
            session=_FakeSession(response=_FakeResponse({})),
            ddgs_factory=lambda: _FakeDDGS(rows=[]),
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="  ")
        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_BAD_INPUT")

    def test_web_search_duckduckgo_timeout_error_without_fallback(self):
        fake_ddgs = _FakeDDGS(exc=DDGTimeoutException("ddg timeout"))
        tool = WebSearchTool(
            config=self._cfg(provider="duckduckgo", fallback_provider="none"),
            session=_FakeSession(response=_FakeResponse({})),
            ddgs_factory=lambda: fake_ddgs,
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="python")

        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_TIMEOUT")
        self.assertTrue(body["retryable"])

    def test_web_search_fallback_to_serpapi_on_duckduckgo_empty(self):
        fake_ddgs = _FakeDDGS(rows=[])
        payload = {
            "news_results": [
                {
                    "title": "OpenAI launches new model",
                    "link": "https://example.com/news/openai",
                    "snippet": "Model update summary.",
                    "source": "Example News",
                    "date": "2026-03-04",
                }
            ]
        }
        fake_session = _FakeSession(response=_FakeResponse(payload))
        tool = WebSearchTool(
            config=self._cfg(provider="duckduckgo", fallback_provider="serpapi", serpapi_api_key="test-key"),
            session=fake_session,
            ddgs_factory=lambda: fake_ddgs,
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="openai", top_k=1, language="en")

        self.assertTrue(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["provider"], "serpapi")
        self.assertEqual(body["count"], 1)
        self.assertTrue(body.get("fallback_used"))
        self.assertEqual(body["results"][0]["published_at"], "2026-03-04")
        self.assertEqual(fake_session.calls[0]["params"]["api_key"], "test-key")

    def test_web_search_upstream_error_when_serpapi_missing_key(self):
        fake_ddgs = _FakeDDGS(rows=[])
        tool = WebSearchTool(
            config=self._cfg(provider="duckduckgo", fallback_provider="serpapi", serpapi_api_key=""),
            session=_FakeSession(response=_FakeResponse({})),
            ddgs_factory=lambda: fake_ddgs,
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="openai")

        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_UPSTREAM_ERROR")


if __name__ == "__main__":
    unittest.main()
