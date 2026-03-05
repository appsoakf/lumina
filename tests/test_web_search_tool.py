import json
import sys
import unittest
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import UapiSearchConfig, WebSearchConfig
from core.tools import ToolContext, WebSearchTool, build_default_registry


class _FakeResponse:
    def __init__(self, payload, *, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.content = b"{}" if payload is not None else b""

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

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        return self.response


class WebSearchToolTests(unittest.TestCase):
    def _cfg(
        self,
        *,
        endpoint: str = "https://uapis.cn/api/v1/search/aggregate",
        api_key: str = "test-key",
        timeout_sec: float = 8.0,
        max_top_k: int = 5,
        default_sort: str = "relevance",
        default_fetch_full: bool = False,
    ) -> WebSearchConfig:
        return WebSearchConfig(
            timeout_sec=timeout_sec,
            max_top_k=max_top_k,
            uapis=UapiSearchConfig(
                endpoint=endpoint,
                api_key=api_key,
                default_sort=default_sort,
                default_fetch_full=default_fetch_full,
            ),
        )

    def test_build_default_registry_contains_web_search_schema(self):
        registry = build_default_registry()
        names = [item["function"]["name"] for item in registry.list_schemas()]
        self.assertIn("web_search", names)

    def test_web_search_uapis_success(self):
        payload = {
            "query": "python",
            "total_results": 2,
            "results": [
                {
                    "title": "Python Official",
                    "url": "https://www.python.org/",
                    "snippet": "Python language home page.",
                    "source": "uapi-search",
                    "publish_time": "2026-03-05T00:00:00Z",
                    "score": 0.91,
                },
                {
                    "title": "Python Docs",
                    "url": "https://docs.python.org/3/",
                    "snippet": "Official docs.",
                    "source": "uapi-search",
                },
            ],
            "sources": [{"name": "uapi-search", "status": "success", "result_count": 2}],
            "process_time_ms": 425,
            "cached": False,
        }
        fake_session = _FakeSession(response=_FakeResponse(payload))
        tool = WebSearchTool(config=self._cfg(api_key="test-key"), session=fake_session)
        result = tool.run(ctx=ToolContext(session_id="s1"), text="python", top_k=2)

        self.assertTrue(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["query"], "python")
        self.assertEqual(body["provider"], "uapis")
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["results"][0]["id"], "R1")
        self.assertAlmostEqual(body["results"][0]["score"], 0.91)
        self.assertEqual(body["results"][0]["published_at"], "2026-03-05T00:00:00Z")

        self.assertEqual(len(fake_session.calls), 1)
        call = fake_session.calls[0]
        self.assertEqual(call["url"], "https://uapis.cn/api/v1/search/aggregate")
        self.assertEqual(call["json"]["query"], "python")
        self.assertEqual(call["json"]["limit"], 2)
        self.assertEqual(call["json"]["sort"], "relevance")
        self.assertEqual(call["json"]["timeout_ms"], 8000)
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call["headers"]["X-API-Key"], "test-key")

    def test_web_search_invalid_input(self):
        tool = WebSearchTool(
            config=self._cfg(),
            session=_FakeSession(response=_FakeResponse({})),
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="  ")
        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_BAD_INPUT")

    def test_web_search_timeout_returns_retryable_error(self):
        tool = WebSearchTool(
            config=self._cfg(),
            session=_FakeSession(exc=requests.Timeout("uapis timeout")),
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="python")

        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_TIMEOUT")
        self.assertTrue(body["retryable"])

    def test_web_search_maps_recency_and_filters(self):
        payload = {
            "query": "openai",
            "results": [
                {
                    "title": "OpenAI News",
                    "url": "https://example.com/openai",
                    "snippet": "Latest updates.",
                    "source": "uapi-search",
                }
            ],
        }
        fake_session = _FakeSession(response=_FakeResponse(payload))
        tool = WebSearchTool(config=self._cfg(default_sort="relevance"), session=fake_session)

        result = tool.run(
            ctx=ToolContext(session_id="s1"),
            text="openai",
            top_k=1,
            recency_days=7,
            site="openai.com",
            filetype="pdf",
            sort="date",
            fetch_full=True,
            timeout_ms=500,
        )

        self.assertTrue(result.ok)
        call_json = fake_session.calls[0]["json"]
        self.assertEqual(call_json["query"], "openai")
        self.assertEqual(call_json["limit"], 1)
        self.assertEqual(call_json["site"], "openai.com")
        self.assertEqual(call_json["filetype"], "pdf")
        self.assertEqual(call_json["time_range"], "week")
        self.assertEqual(call_json["sort"], "date")
        self.assertTrue(call_json["fetch_full"])
        self.assertEqual(call_json["timeout_ms"], 1000)

    def test_web_search_upstream_http_error(self):
        payload = {"code": "UNAUTHORIZED", "message": "无效的访问令牌"}
        tool = WebSearchTool(
            config=self._cfg(),
            session=_FakeSession(response=_FakeResponse(payload, status_code=401)),
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="openai")

        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_UPSTREAM_ERROR")
        self.assertIn("UNAUTHORIZED", body["message"])

    def test_web_search_empty_result(self):
        tool = WebSearchTool(
            config=self._cfg(),
            session=_FakeSession(response=_FakeResponse({"query": "openai", "results": []})),
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="openai")

        self.assertFalse(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["error_code"], "WEB_SEARCH_EMPTY")

    def test_web_search_respects_top_k_limit(self):
        payload = {
            "query": "python",
            "results": [
                {"title": "A", "url": "https://a.com", "snippet": "a"},
                {"title": "B", "url": "https://b.com", "snippet": "b"},
                {"title": "C", "url": "https://c.com", "snippet": "c"},
            ],
        }
        tool = WebSearchTool(
            config=self._cfg(max_top_k=2),
            session=_FakeSession(response=_FakeResponse(payload)),
        )
        result = tool.run(ctx=ToolContext(session_id="s1"), text="python", top_k=5)

        self.assertTrue(result.ok)
        body = json.loads(result.content)
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["results"]), 2)


if __name__ == "__main__":
    unittest.main()
