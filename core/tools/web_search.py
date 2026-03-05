import logging
import time
from typing import Any, Dict, List, Optional

import requests

from core.config import UapiSearchConfig, WebSearchConfig, load_app_config
from core.tools.base import BaseTool
from core.tools.models import ToolContext, ToolResult
from core.utils import elapsed_ms, log_event


logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    def __init__(
        self,
        *,
        config: Optional[WebSearchConfig] = None,
        session: Optional[requests.Session] = None,
    ):
        cfg = config or load_app_config().tools.web_search
        self.max_top_k = max(int(cfg.max_top_k), 1)
        self.timeout_sec = max(float(cfg.timeout_sec), 1.0)
        self.uapis_cfg = cfg.uapis
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
                    "filetype": {"type": "string", "description": "Optional file type filter, e.g. pdf/docx."},
                    "sort": {"type": "string", "description": "Sort mode: relevance or date."},
                    "fetch_full": {"type": "boolean", "description": "Whether to fetch full page content."},
                    "timeout_ms": {"type": "integer", "description": "Optional per-request timeout in ms."},
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
        filetype: Optional[str] = None,
        sort: Optional[str] = None,
        fetch_full: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        _ = kwargs
        query = str(text or "").strip()
        if not query:
            return self.error_result(
                code="WEB_SEARCH_BAD_INPUT",
                message="Field `text` is required for web_search",
                retryable=False,
            )

        limit = self.clamp_int(top_k, default=3, min_value=1, max_value=self.max_top_k)
        payload = self._build_request_payload(
            query=query,
            limit=limit,
            site=site,
            filetype=filetype,
            recency_days=recency_days,
            sort=sort,
            fetch_full=fetch_full,
            timeout_ms=timeout_ms,
        )

        started = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            "web_search.request",
            (
                "web_search 接收请求 "
                f"query={query} top_k={limit} language={str(language or '').strip() or '-'} "
                f"site={str(site or '').strip() or '-'} filetype={str(filetype or '').strip() or '-'}"
            ),
            component="tool",
            session_id=str(getattr(ctx, "session_id", "") or "-"),
            query=query,
            top_k=limit,
            language=str(language or "").strip() or "-",
            recency_days=recency_days if recency_days is not None else -1,
            site=str(site or "").strip() or "-",
            filetype=str(filetype or "").strip() or "-",
        )

        try:
            data = self._search_uapis(payload=payload)
        except requests.Timeout as exc:
            log_event(
                logger,
                logging.WARNING,
                "web_search.response.error",
                "web_search 返回失败 error_code=WEB_SEARCH_TIMEOUT",
                component="tool",
                session_id=str(getattr(ctx, "session_id", "") or "-"),
                error_code="WEB_SEARCH_TIMEOUT",
                retryable=True,
                duration_ms=elapsed_ms(started),
                error_message=str(exc),
            )
            return self.error_result(
                code="WEB_SEARCH_TIMEOUT",
                message="web_search request timeout",
                retryable=True,
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            log_event(
                logger,
                logging.WARNING,
                "web_search.response.error",
                "web_search 返回失败 error_code=WEB_SEARCH_UPSTREAM_ERROR",
                component="tool",
                session_id=str(getattr(ctx, "session_id", "") or "-"),
                error_code="WEB_SEARCH_UPSTREAM_ERROR",
                retryable=True,
                duration_ms=elapsed_ms(started),
                error_message=str(exc),
            )
            return self.error_result(
                code="WEB_SEARCH_UPSTREAM_ERROR",
                message=f"web_search request failed: {exc}",
                retryable=True,
            )

        rows = self._collect_rows(data=data, top_k=limit)
        if not rows:
            log_event(
                logger,
                logging.INFO,
                "web_search.response.empty",
                "web_search 无结果",
                component="tool",
                session_id=str(getattr(ctx, "session_id", "") or "-"),
                duration_ms=elapsed_ms(started),
            )
            return self.error_result(
                code="WEB_SEARCH_EMPTY",
                message="No relevant web results found",
                retryable=False,
            )

        total_results = data.get("total_results")
        if not isinstance(total_results, int):
            total_results = len(rows)
        sources = data.get("sources")
        process_time_ms = data.get("process_time_ms")
        cached = data.get("cached")

        result_payload: Dict[str, Any] = {
            "query": str(data.get("query") or query),
            "provider": "uapis",
            "count": len(rows),
            "total_results": total_results,
            "results": rows,
        }
        if isinstance(sources, list):
            result_payload["sources"] = sources
        if isinstance(process_time_ms, int):
            result_payload["process_time_ms"] = process_time_ms
        if isinstance(cached, bool):
            result_payload["cached"] = cached

        titles_preview = " | ".join(
            str(item.get("title") or "").strip()[:60]
            for item in rows[:3]
            if str(item.get("title") or "").strip()
        )
        log_event(
            logger,
            logging.INFO,
            "web_search.response.success",
            f"web_search provider 成功：uapis count={len(rows)} preview={titles_preview}",
            component="tool",
            session_id=str(getattr(ctx, "session_id", "") or "-"),
            provider="uapis",
            count=len(rows),
            duration_ms=elapsed_ms(started),
        )
        return self.ok_result(result_payload)

    def _search_uapis(self, *, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._build_headers(self.uapis_cfg)
        resp = self.session.post(
            self.uapis_cfg.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )

        if resp.status_code >= 400:
            raise RuntimeError(self._extract_error_message(resp))

        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            raise RuntimeError("uapis response must be a JSON object")

        # 即使是 200，也可能返回业务层错误体。
        if data.get("code") and data.get("message") and not isinstance(data.get("results"), list):
            raise RuntimeError(f"{data.get('code')}: {data.get('message')}")
        return data

    def _build_headers(self, cfg: UapiSearchConfig) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if cfg.api_key:
            # 兼容不同网关鉴权方式。
            headers["Authorization"] = f"Bearer {cfg.api_key}"
            headers["X-API-Key"] = cfg.api_key
        return headers

    def _extract_error_message(self, resp: requests.Response) -> str:
        try:
            payload = resp.json()
        except Exception:
            payload = {}

        if isinstance(payload, dict):
            code = str(payload.get("code") or "").strip()
            message = str(payload.get("message") or "").strip()
            if code and message:
                return f"{code}: {message}"
            if message:
                return message

        text = str(getattr(resp, "text", "") or "").strip()
        if text:
            return f"HTTP {resp.status_code}: {text[:240]}"
        return f"HTTP {resp.status_code}"

    def _build_request_payload(
        self,
        *,
        query: str,
        limit: int,
        site: Optional[str],
        filetype: Optional[str],
        recency_days: Optional[int],
        sort: Optional[str],
        fetch_full: Optional[bool],
        timeout_ms: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "limit": max(1, int(limit))}

        site_text = str(site or "").strip()
        if site_text:
            payload["site"] = site_text

        filetype_text = str(filetype or "").strip().lower()
        if filetype_text:
            payload["filetype"] = filetype_text

        time_range = self._resolve_time_range(recency_days)
        if time_range:
            payload["time_range"] = time_range

        sort_value = str(sort or "").strip().lower()
        if sort_value not in {"relevance", "date"}:
            sort_value = self.uapis_cfg.default_sort
        payload["sort"] = sort_value

        fetch_full_value = self.uapis_cfg.default_fetch_full if fetch_full is None else bool(fetch_full)
        if fetch_full_value:
            payload["fetch_full"] = True

        payload["timeout_ms"] = self._resolve_timeout_ms(timeout_ms)
        return payload

    def _resolve_timeout_ms(self, timeout_ms: Optional[int]) -> int:
        if timeout_ms is None:
            value = int(self.timeout_sec * 1000)
        else:
            try:
                value = int(timeout_ms)
            except Exception:
                value = int(self.timeout_sec * 1000)
        return max(1000, min(value, 30000))

    def _resolve_time_range(self, recency_days: Optional[int]) -> Optional[str]:
        if recency_days is None:
            return None
        try:
            days = int(recency_days)
        except Exception:
            return None
        if days <= 0:
            return None
        if days <= 1:
            return "day"
        if days <= 7:
            return "week"
        if days <= 31:
            return "month"
        return "year"

    def _collect_rows(self, *, data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        raw_rows = data.get("results")
        if not isinstance(raw_rows, list):
            return []

        rows: List[Dict[str, Any]] = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            url = str(raw.get("url") or "").strip()
            snippet = str(raw.get("snippet") or "").strip()
            source = str(raw.get("source") or raw.get("domain") or "uapis").strip()
            published_at = str(raw.get("publish_time") or "").strip()
            if not title or not url:
                continue
            item = self._normalize_result(
                index=len(rows) + 1,
                title=title,
                url=url,
                snippet=snippet,
                source=source,
                published_at=published_at,
            )
            score = raw.get("score")
            if isinstance(score, (int, float)):
                item["score"] = float(score)
            rows.append(item)
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
            "source": str(source).strip() or "uapis",
            "published_at": str(published_at).strip() or None,
        }
