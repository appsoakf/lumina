import json
import os
from typing import Any, Dict, List, Optional

import requests

from core.tools.base import BaseTool
from core.tools.models import ToolContext, ToolResult


class WebSearchTool(BaseTool):
    DEFAULT_PROVIDER = "duckduckgo"
    DEFAULT_DDG_ENDPOINT = "https://api.duckduckgo.com/"
    DEFAULT_SEARXNG_ENDPOINT = "http://127.0.0.1:8080/search"

    def __init__(self, *, session: Optional[requests.Session] = None):
        timeout_sec = self._read_float_env("LUMINA_WEB_SEARCH_TIMEOUT_SEC", 8.0)
        self.max_top_k = self._read_int_env("LUMINA_WEB_SEARCH_MAX_TOP_K", 5, min_value=1)
        self.provider = str(os.environ.get("LUMINA_WEB_SEARCH_PROVIDER", self.DEFAULT_PROVIDER)).strip().lower()
        self.endpoint = str(os.environ.get("LUMINA_WEB_SEARCH_ENDPOINT", "")).strip()
        self.api_key = str(os.environ.get("LUMINA_WEB_SEARCH_API_KEY", "")).strip()
        self.timeout_sec = max(timeout_sec, 1.0)
        self.session = session or requests.Session()
        super().__init__(
            name="web_search",
            description="Search the web and return top relevant snippets.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Search query text."},
                    "top_k": {"type": "integer", "description": "Number of results, range 1-5.", "minimum": 1, "maximum": 5},
                    "language": {"type": "string", "description": "Preferred language code, e.g. zh-CN or en."},
                    "recency_days": {"type": "integer", "description": "Optional freshness hint in days."},
                    "site": {"type": "string", "description": "Optional site filter, e.g. docs.python.org."},
                },
                "required": ["text"],
            },
            max_retries=1,
            retry_backoff_sec=0.25,
        )

    def is_retryable_exception(self, exc: Exception) -> bool:
        return isinstance(exc, (requests.Timeout, requests.ConnectionError, requests.HTTPError))

    def run(
        self,
        *,
        ctx: ToolContext,
        text: str,
        top_k: int = 3,
        language: str = "zh-CN",
        recency_days: Optional[int] = None,
        site: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        _ = ctx, kwargs
        query = str(text or "").strip()
        if not query:
            return self.error_result(
                code="WEB_SEARCH_BAD_INPUT",
                message="Field `text` is required for web_search",
                retryable=False,
            )

        limit = self.clamp_int(top_k, default=3, min_value=1, max_value=self.max_top_k)
        normalized_query = self._compose_query(query=query, site=site, recency_days=recency_days)

        try:
            if self.provider == "searxng":
                rows = self._search_searxng(query=normalized_query, language=language, top_k=limit)
            else:
                rows = self._search_duckduckgo(query=normalized_query, top_k=limit)
        except requests.Timeout:
            return self.error_result(
                code="WEB_SEARCH_TIMEOUT",
                message="web_search request timeout",
                retryable=True,
            )
        except requests.RequestException as exc:
            return self.error_result(
                code="WEB_SEARCH_UPSTREAM_ERROR",
                message=f"web_search request failed: {exc}",
                retryable=True,
            )

        if not rows:
            return self.error_result(
                code="WEB_SEARCH_EMPTY",
                message="No relevant web results found",
                retryable=False,
            )

        payload = {
            "query": query,
            "provider": self.provider,
            "count": len(rows),
            "results": rows,
        }
        return self.ok_result(payload)

    def _search_duckduckgo(self, *, query: str, top_k: int) -> List[Dict[str, Any]]:
        endpoint = self.endpoint or self.DEFAULT_DDG_ENDPOINT
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
        }
        resp = self.session.get(endpoint, params=params, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        rows: List[Dict[str, Any]] = []

        abstract_text = str(data.get("AbstractText") or "").strip()
        abstract_url = str(data.get("AbstractURL") or "").strip()
        heading = str(data.get("Heading") or "").strip()
        if abstract_text and abstract_url:
            rows.append(
                self._normalize_result(
                    index=len(rows) + 1,
                    title=heading or query,
                    url=abstract_url,
                    snippet=abstract_text,
                    source=str(data.get("AbstractSource") or "duckduckgo"),
                )
            )

        for item in data.get("RelatedTopics") or []:
            self._collect_ddg_topic(item, rows)
            if len(rows) >= top_k:
                break

        return rows[:top_k]

    def _collect_ddg_topic(self, item: Any, rows: List[Dict[str, Any]]) -> None:
        if isinstance(item, dict) and isinstance(item.get("Topics"), list):
            for child in item["Topics"]:
                self._collect_ddg_topic(child, rows)
            return

        if not isinstance(item, dict):
            return
        title = str(item.get("Text") or "").strip()
        url = str(item.get("FirstURL") or "").strip()
        if not title or not url:
            return
        rows.append(
            self._normalize_result(
                index=len(rows) + 1,
                title=title.split(" - ", 1)[0].strip() or title,
                url=url,
                snippet=title,
                source="duckduckgo",
            )
        )

    def _search_searxng(self, *, query: str, language: str, top_k: int) -> List[Dict[str, Any]]:
        endpoint = self.endpoint or self.DEFAULT_SEARXNG_ENDPOINT
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        params = {
            "q": query,
            "format": "json",
            "language": language,
        }
        resp = self.session.get(endpoint, params=params, headers=headers, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}

        rows: List[Dict[str, Any]] = []
        for row in data.get("results") or []:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            snippet = str(row.get("content") or "").strip()
            if not title or not url:
                continue
            rows.append(
                self._normalize_result(
                    index=len(rows) + 1,
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=str(row.get("engine") or "searxng"),
                    published_at=str(row.get("publishedDate") or ""),
                )
            )
            if len(rows) >= top_k:
                break
        return rows

    def _normalize_result(
        self,
        *,
        index: int,
        title: str,
        url: str,
        snippet: str,
        source: str,
        published_at: str = "",
    ) -> Dict[str, Any]:
        return {
            "id": f"R{index}",
            "title": str(title).strip(),
            "url": str(url).strip(),
            "snippet": str(snippet).strip(),
            "source": str(source).strip() or "web",
            "published_at": str(published_at).strip() or None,
        }

    def _compose_query(self, *, query: str, site: Optional[str], recency_days: Optional[int]) -> str:
        output = str(query or "").strip()
        domain = str(site or "").strip()
        if domain:
            output = f"site:{domain} {output}".strip()
        if recency_days is not None:
            try:
                days = int(recency_days)
            except Exception:
                days = 0
            if days > 0:
                output = f"{output} within last {days} days"
        return output

    def _read_int_env(self, key: str, default: int, *, min_value: int = 0) -> int:
        raw = str(os.environ.get(key, "")).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except Exception:
            return default
        return max(value, min_value)

    def _read_float_env(self, key: str, default: float) -> float:
        raw = str(os.environ.get(key, "")).strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default

