import json
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, TimeoutException as DDGTimeoutException

from core.config import DuckDuckGoConfig, SerpApiConfig, WebSearchConfig, load_app_config
from core.tools.base import BaseTool
from core.tools.models import ToolContext, ToolResult


class WebSearchTool(BaseTool):
    def __init__(
        self,
        *,
        config: Optional[WebSearchConfig] = None,
        session: Optional[requests.Session] = None,
        ddgs_factory: Optional[Callable[[], Any]] = None,
    ):
        cfg = config or load_app_config().tools.web_search
        self.primary_provider = str(cfg.provider).strip().lower()
        self.fallback_provider = str(cfg.fallback_provider).strip().lower()
        self.max_top_k = max(int(cfg.max_top_k), 1)
        self.timeout_sec = max(float(cfg.timeout_sec), 1.0)
        self.duckduckgo_cfg = cfg.duckduckgo
        self.serpapi_cfg = cfg.serpapi
        self.session = session or requests.Session()
        self.ddgs_factory = ddgs_factory or DDGS
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
        return isinstance(
            exc,
            (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
                DDGTimeoutException,
            ),
        )

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
        normalized_query = self._compose_query(query=query, site=site)
        attempts: List[Dict[str, Any]] = []

        for provider in self._provider_order():
            rows, failure = self._run_provider(
                provider=provider,
                query=normalized_query,
                language=language,
                top_k=limit,
                recency_days=recency_days,
            )
            if failure is not None:
                attempts.append(failure)
                continue
            if rows:
                payload: Dict[str, Any] = {
                    "query": query,
                    "provider": provider,
                    "count": len(rows),
                    "results": rows,
                }
                if provider != self.primary_provider:
                    payload["fallback_used"] = True
                if attempts:
                    payload["attempt_warnings"] = attempts
                return self.ok_result(payload)
            attempts.append(
                {
                    "provider": provider,
                    "status": "empty",
                    "message": "No relevant results",
                }
            )

        return self._final_failure(attempts)

    def _provider_order(self) -> List[str]:
        order = [self.primary_provider]
        fallback = self.fallback_provider
        if fallback and fallback != "none" and fallback != self.primary_provider:
            order.append(fallback)
        return order

    def _run_provider(
        self,
        *,
        provider: str,
        query: str,
        language: str,
        top_k: int,
        recency_days: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        try:
            if provider == "duckduckgo":
                return self._search_duckduckgo(query=query, top_k=top_k, recency_days=recency_days), None
            if provider == "serpapi":
                return self._search_serpapi(
                    query=query,
                    language=language,
                    top_k=top_k,
                    recency_days=recency_days,
                ), None
            return [], {
                "provider": provider,
                "status": "error",
                "message": "Unsupported provider",
            }
        except (requests.Timeout, DDGTimeoutException) as exc:
            return [], {
                "provider": provider,
                "status": "timeout",
                "message": str(exc) or "request timeout",
            }
        except (requests.RequestException, DuckDuckGoSearchException, RuntimeError, ValueError) as exc:
            return [], {
                "provider": provider,
                "status": "error",
                "message": str(exc),
            }

    def _search_duckduckgo(self, *, query: str, top_k: int, recency_days: Optional[int]) -> List[Dict[str, Any]]:
        timelimit = self._resolve_ddg_timelimit(self.duckduckgo_cfg, recency_days)
        raw_rows = self._ddgs_text(
            keywords=query,
            region=self.duckduckgo_cfg.region,
            safesearch=self.duckduckgo_cfg.safesearch,
            timelimit=timelimit,
            backend=self.duckduckgo_cfg.backend,
            max_results=top_k,
        )
        rows: List[Dict[str, Any]] = []
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("heading") or "").strip()
            url = str(row.get("href") or row.get("url") or "").strip()
            snippet = str(row.get("body") or row.get("snippet") or row.get("description") or "").strip()
            if not title or not url:
                continue
            rows.append(
                self._normalize_result(
                    index=len(rows) + 1,
                    title=title,
                    url=url,
                    snippet=snippet,
                    source="duckduckgo",
                    published_at=str(row.get("date") or row.get("published") or ""),
                )
            )
            if len(rows) >= top_k:
                break
        return rows

    def _ddgs_text(
        self,
        *,
        keywords: str,
        region: str,
        safesearch: str,
        timelimit: Optional[str],
        backend: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        original_showwarning = warnings.showwarning
        original_filters = list(warnings.filters)

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            is_runtime = isinstance(category, type) and issubclass(category, RuntimeWarning)
            if is_runtime and "has been renamed to `ddgs`" in text:
                return
            return original_showwarning(message, category, filename, lineno, file=file, line=line)

        try:
            warnings.showwarning = _showwarning
            client = self.ddgs_factory()
        finally:
            warnings.showwarning = original_showwarning
            warnings.filters[:] = original_filters

        kwargs = {
            "keywords": keywords,
            "region": region or None,
            "safesearch": safesearch,
            "timelimit": timelimit or None,
            "backend": backend,
            "max_results": max_results,
        }
        if hasattr(client, "__enter__") and hasattr(client, "__exit__"):
            with client as ddgs:
                rows = ddgs.text(**kwargs)
        else:
            rows = client.text(**kwargs)
        return list(rows or [])

    def _search_serpapi(
        self,
        *,
        query: str,
        language: str,
        top_k: int,
        recency_days: Optional[int],
    ) -> List[Dict[str, Any]]:
        cfg = self.serpapi_cfg
        if not cfg.api_key:
            raise RuntimeError("serpapi api_key is required for fallback search")

        params: Dict[str, Any] = {
            "q": query,
            "api_key": cfg.api_key,
            "engine": cfg.engine,
            "num": top_k,
        }
        if cfg.gl:
            params["gl"] = cfg.gl
        hl = self._normalize_language(language=language, fallback=cfg.hl)
        if hl:
            params["hl"] = hl
        if cfg.tbm:
            params["tbm"] = cfg.tbm
        tbs = self._resolve_serpapi_tbs(recency_days)
        if tbs:
            params["tbs"] = tbs

        resp = self.session.get(cfg.endpoint, params=params, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            raise RuntimeError("serpapi response must be a JSON object")
        if data.get("error"):
            raise RuntimeError(str(data.get("error")))

        rows: List[Dict[str, Any]] = []
        rows.extend(self._collect_serpapi_rows(data.get("news_results") or [], top_k=top_k, start_index=len(rows) + 1))
        if len(rows) < top_k:
            rows.extend(
                self._collect_serpapi_rows(
                    data.get("organic_results") or [],
                    top_k=top_k - len(rows),
                    start_index=len(rows) + 1,
                )
            )
        return rows[:top_k]

    def _collect_serpapi_rows(self, raw_rows: List[Any], *, top_k: int, start_index: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            url = str(raw.get("link") or "").strip()
            snippet = str(raw.get("snippet") or "").strip()
            source = str(raw.get("source") or "serpapi").strip()
            published_at = str(raw.get("date") or raw.get("published") or "").strip()
            if not title or not url:
                continue
            rows.append(
                self._normalize_result(
                    index=start_index + len(rows),
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=source,
                    published_at=published_at,
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

    def _compose_query(self, *, query: str, site: Optional[str]) -> str:
        output = str(query or "").strip()
        domain = str(site or "").strip()
        if domain:
            output = f"site:{domain} {output}".strip()
        return output

    def _normalize_language(self, *, language: str, fallback: str) -> str:
        text = str(language or "").strip()
        if not text:
            return str(fallback or "").strip().lower()
        return text.split("-", 1)[0].strip().lower()

    def _resolve_ddg_timelimit(self, cfg: DuckDuckGoConfig, recency_days: Optional[int]) -> Optional[str]:
        if recency_days is None:
            return cfg.timelimit or None
        try:
            days = int(recency_days)
        except Exception:
            return cfg.timelimit or None
        if days <= 0:
            return cfg.timelimit or None
        if days <= 1:
            return "d"
        if days <= 7:
            return "w"
        if days <= 31:
            return "m"
        return "y"

    def _resolve_serpapi_tbs(self, recency_days: Optional[int]) -> str:
        if recency_days is None:
            return ""
        try:
            days = int(recency_days)
        except Exception:
            return ""
        if days <= 0:
            return ""
        if days <= 1:
            return "qdr:d"
        if days <= 7:
            return "qdr:w"
        if days <= 31:
            return "qdr:m"
        return "qdr:y"

    def _final_failure(self, attempts: List[Dict[str, Any]]) -> ToolResult:
        if any(item.get("status") == "timeout" for item in attempts):
            return self.error_result(
                code="WEB_SEARCH_TIMEOUT",
                message="web_search request timeout",
                retryable=True,
                details={"attempts": attempts},
            )
        if any(item.get("status") == "error" for item in attempts):
            return self.error_result(
                code="WEB_SEARCH_UPSTREAM_ERROR",
                message="web_search request failed",
                retryable=True,
                details={"attempts": attempts},
            )
        return self.error_result(
            code="WEB_SEARCH_EMPTY",
            message="No relevant web results found",
            retryable=False,
            details={"attempts": attempts},
        )
