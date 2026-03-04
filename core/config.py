import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from core.utils.errors import AppError, ErrorCode


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_CONFIG_PATH = ROOT_DIR / "config.json"


@dataclass
class LLMConfig:
    chat_model: str
    chat_api_url: str
    chat_api_key: str
    translate_model: str
    translate_api_url: str
    translate_api_key: str
    chat_prompt: str
    translate_prompt: str


@dataclass
class TTSConfig:
    gpt_sovits_url: str
    ref_path: str
    prompt_text: str
    prompt_lang: str


@dataclass
class ServiceConfig:
    pet_name: str
    username: str
    server_address: str
    server_port: int


@dataclass
class MemoryVectorConfig:
    enabled: bool
    provider: str
    embedding_model: str
    embedding_api_url: str
    embedding_api_key: str
    qdrant_url: str
    qdrant_collection: str
    vector_dim: int
    top_k_vector: int
    top_k_keyword: int
    write_async: bool
    queue_size: int
    max_retries: int


@dataclass
class WebSearchConfig:
    provider: str
    fallback_provider: str
    timeout_sec: float
    max_top_k: int
    duckduckgo: "DuckDuckGoConfig"
    serpapi: "SerpApiConfig"


@dataclass
class DuckDuckGoConfig:
    region: str
    safesearch: str
    backend: str
    timelimit: str


@dataclass
class SerpApiConfig:
    endpoint: str
    api_key: str
    engine: str
    gl: str
    hl: str
    tbm: str


@dataclass
class ToolsConfig:
    web_search: WebSearchConfig


@dataclass
class AppConfig:
    llm: LLMConfig
    tts: TTSConfig
    service: ServiceConfig
    memory_vector: MemoryVectorConfig
    tools: ToolsConfig


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Config file not found",
            details={"path": str(path)},
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Config root must be a JSON object",
            details={"path": str(path)},
        )
    return payload


def _require_section(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    section = raw.get(key)
    if not isinstance(section, dict):
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing required config section",
            details={"section": key, "path": str(ROOT_CONFIG_PATH)},
        )
    return section


def _env_or(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _to_int(value: Any, path: str) -> int:
    try:
        return int(value)
    except Exception:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid integer config value",
            details={"field": path, "value": value},
        )


def _to_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise AppError(
        ErrorCode.CONFIG_INVALID,
        "Invalid boolean config value",
        details={"field": path, "value": value},
    )


def _to_float(value: Any, path: str) -> float:
    try:
        return float(value)
    except Exception:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid float config value",
            details={"field": path, "value": value},
        )


def _build_llm_config(raw: Dict[str, Any]) -> LLMConfig:
    cfg = LLMConfig(
        chat_model=str(raw.get("chat_model", "")).strip(),
        chat_api_url=str(raw.get("chat_api_url", "")).strip(),
        chat_api_key=_env_or("LUMINA_API_KEY", str(raw.get("chat_api_key", "")).strip()),
        translate_model=str(raw.get("translate_model", "")).strip(),
        translate_api_url=str(raw.get("translate_api_url", "")).strip(),
        translate_api_key=_env_or("LUMINA_API_KEY", str(raw.get("translate_api_key", "")).strip()),
        chat_prompt=str(raw.get("chat_prompt", "")),
        translate_prompt=str(raw.get("translate_prompt", "")),
    )

    required = {
        "llm.chat_model": cfg.chat_model,
        "llm.chat_api_url": cfg.chat_api_url,
        "llm.translate_model": cfg.translate_model,
        "llm.translate_api_url": cfg.translate_api_url,
        "llm.chat_prompt": cfg.chat_prompt,
        "llm.translate_prompt": cfg.translate_prompt,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing required LLM config fields",
            details={"fields": missing},
        )
    return cfg


def _build_tts_config(raw: Dict[str, Any]) -> TTSConfig:
    cfg = TTSConfig(
        gpt_sovits_url=str(raw.get("gpt_sovits_url", "")).strip(),
        ref_path=str(raw.get("ref_path", "")).strip(),
        prompt_text=str(raw.get("prompt_text", "")).strip(),
        prompt_lang=str(raw.get("prompt_lang", "ja")).strip() or "ja",
    )
    required = {
        "tts.gpt_sovits_url": cfg.gpt_sovits_url,
        "tts.ref_path": cfg.ref_path,
        "tts.prompt_text": cfg.prompt_text,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing required TTS config fields",
            details={"fields": missing},
        )
    return cfg


def _build_service_config(raw: Dict[str, Any]) -> ServiceConfig:
    cfg = ServiceConfig(
        pet_name=str(raw.get("pet_name", "")).strip(),
        username=str(raw.get("username", "")).strip(),
        server_address=str(raw.get("server_address", "0.0.0.0")).strip(),
        server_port=_to_int(raw.get("server_port", 8080), "service.server_port"),
    )
    required = {
        "service.pet_name": cfg.pet_name,
        "service.username": cfg.username,
        "service.server_address": cfg.server_address,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing required service config fields",
            details={"fields": missing},
        )
    return cfg


def _build_memory_vector_config(raw: Dict[str, Any], llm_raw: Dict[str, Any]) -> MemoryVectorConfig:
    mv = dict(raw or {})

    enabled_raw = os.environ.get("LUMINA_MEMORY_VECTOR_ENABLED", mv.get("enabled", False))
    enabled = _to_bool(enabled_raw, "memory_vector.enabled")

    cfg = MemoryVectorConfig(
        enabled=enabled,
        provider=str(os.environ.get("LUMINA_MEMORY_VECTOR_PROVIDER", mv.get("provider", "openai"))).strip() or "openai",
        embedding_model=str(
            os.environ.get("LUMINA_EMBEDDING_MODEL", mv.get("embedding_model", "text-embedding-3-small"))
        ).strip(),
        embedding_api_url=str(
            os.environ.get("LUMINA_EMBEDDING_API_URL", mv.get("embedding_api_url", llm_raw.get("chat_api_url", "")))
        ).strip(),
        embedding_api_key=str(
            os.environ.get(
                "LUMINA_EMBEDDING_API_KEY",
                os.environ.get("LUMINA_API_KEY", mv.get("embedding_api_key", llm_raw.get("chat_api_key", ""))),
            )
        ).strip(),
        qdrant_url=str(os.environ.get("LUMINA_QDRANT_URL", mv.get("qdrant_url", "http://127.0.0.1:6333"))).strip(),
        qdrant_collection=str(
            os.environ.get("LUMINA_QDRANT_COLLECTION", mv.get("qdrant_collection", "lumina_memory_vectors"))
        ).strip(),
        vector_dim=_to_int(
            os.environ.get("LUMINA_VECTOR_DIM", mv.get("vector_dim", 1536)),
            "memory_vector.vector_dim",
        ),
        top_k_vector=_to_int(
            os.environ.get("LUMINA_VECTOR_TOP_K", mv.get("top_k_vector", 12)),
            "memory_vector.top_k_vector",
        ),
        top_k_keyword=_to_int(
            os.environ.get("LUMINA_KEYWORD_TOP_K", mv.get("top_k_keyword", 12)),
            "memory_vector.top_k_keyword",
        ),
        write_async=_to_bool(
            os.environ.get("LUMINA_VECTOR_WRITE_ASYNC", mv.get("write_async", True)),
            "memory_vector.write_async",
        ),
        queue_size=_to_int(
            os.environ.get("LUMINA_VECTOR_QUEUE_SIZE", mv.get("queue_size", 512)),
            "memory_vector.queue_size",
        ),
        max_retries=_to_int(
            os.environ.get("LUMINA_VECTOR_MAX_RETRIES", mv.get("max_retries", 3)),
            "memory_vector.max_retries",
        ),
    )

    if cfg.enabled:
        required = {
            "memory_vector.embedding_model": cfg.embedding_model,
            "memory_vector.embedding_api_url": cfg.embedding_api_url,
            "memory_vector.embedding_api_key": cfg.embedding_api_key,
            "memory_vector.qdrant_url": cfg.qdrant_url,
            "memory_vector.qdrant_collection": cfg.qdrant_collection,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise AppError(
                ErrorCode.CONFIG_MISSING,
                "Missing required memory vector config fields",
                details={"fields": missing},
            )

    return cfg


def _build_duckduckgo_config(raw: Dict[str, Any]) -> DuckDuckGoConfig:
    region = str(raw.get("region", "")).strip()
    safesearch = str(raw.get("safesearch", "")).strip().lower()
    backend = str(raw.get("backend", "")).strip().lower()
    timelimit = str(raw.get("timelimit", "")).strip().lower()

    if not region:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing duckduckgo region",
            details={"field": "tools.web_search.duckduckgo.region"},
        )
    if safesearch not in {"strict", "moderate", "off"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid duckduckgo safesearch",
            details={
                "field": "tools.web_search.duckduckgo.safesearch",
                "value": safesearch,
                "allowed": ["strict", "moderate", "off"],
            },
        )
    if backend not in {"auto", "html", "lite"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid duckduckgo backend",
            details={
                "field": "tools.web_search.duckduckgo.backend",
                "value": backend,
                "allowed": ["auto", "html", "lite"],
            },
        )
    if timelimit and timelimit not in {"d", "w", "m", "y"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid duckduckgo timelimit",
            details={
                "field": "tools.web_search.duckduckgo.timelimit",
                "value": timelimit,
                "allowed": ["", "d", "w", "m", "y"],
            },
        )

    return DuckDuckGoConfig(
        region=region,
        safesearch=safesearch,
        backend=backend,
        timelimit=timelimit,
    )


def _build_serpapi_config(raw: Dict[str, Any]) -> SerpApiConfig:
    endpoint = str(raw.get("endpoint", "")).strip()
    api_key = str(raw.get("api_key", "")).strip()
    engine = str(raw.get("engine", "")).strip()
    gl = str(raw.get("gl", "")).strip()
    hl = str(raw.get("hl", "")).strip()
    tbm = str(raw.get("tbm", "")).strip()

    if not endpoint:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing serpapi endpoint",
            details={"field": "tools.web_search.serpapi.endpoint"},
        )
    if not engine:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing serpapi engine",
            details={"field": "tools.web_search.serpapi.engine"},
        )

    return SerpApiConfig(
        endpoint=endpoint,
        api_key=api_key,
        engine=engine,
        gl=gl,
        hl=hl,
        tbm=tbm,
    )


def _build_web_search_config(raw: Dict[str, Any]) -> WebSearchConfig:
    provider = str(raw.get("provider", "")).strip().lower()
    fallback_provider = str(raw.get("fallback_provider", "")).strip().lower()
    timeout_raw = raw.get("timeout_sec")
    max_top_k_raw = raw.get("max_top_k")

    if provider not in {"duckduckgo", "serpapi"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Unsupported web_search provider",
            details={
                "field": "tools.web_search.provider",
                "value": provider,
                "allowed": ["duckduckgo", "serpapi"],
            },
        )
    if fallback_provider not in {"none", "duckduckgo", "serpapi"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Unsupported web_search fallback provider",
            details={
                "field": "tools.web_search.fallback_provider",
                "value": fallback_provider,
                "allowed": ["none", "duckduckgo", "serpapi"],
            },
        )
    if fallback_provider == provider:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "web_search fallback provider must be different from primary provider",
            details={
                "field": "tools.web_search.fallback_provider",
                "value": fallback_provider,
            },
        )
    if timeout_raw is None:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing tools.web_search timeout_sec",
            details={"field": "tools.web_search.timeout_sec"},
        )
    if max_top_k_raw is None:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing tools.web_search max_top_k",
            details={"field": "tools.web_search.max_top_k"},
        )

    duckduckgo_raw = raw.get("duckduckgo")
    serpapi_raw = raw.get("serpapi")
    if not isinstance(duckduckgo_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing tools.web_search.duckduckgo config section",
            details={"field": "tools.web_search.duckduckgo"},
        )
    if not isinstance(serpapi_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing tools.web_search.serpapi config section",
            details={"field": "tools.web_search.serpapi"},
        )

    timeout_sec = _to_float(timeout_raw, "tools.web_search.timeout_sec")
    max_top_k = _to_int(max_top_k_raw, "tools.web_search.max_top_k")

    if timeout_sec <= 0:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "tools.web_search.timeout_sec must be > 0",
            details={"field": "tools.web_search.timeout_sec", "value": timeout_sec},
        )
    if max_top_k < 1:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "tools.web_search.max_top_k must be >= 1",
            details={"field": "tools.web_search.max_top_k", "value": max_top_k},
        )

    return WebSearchConfig(
        provider=provider,
        fallback_provider=fallback_provider,
        timeout_sec=timeout_sec,
        max_top_k=max_top_k,
        duckduckgo=_build_duckduckgo_config(duckduckgo_raw),
        serpapi=_build_serpapi_config(serpapi_raw),
    )


def _build_tools_config(raw: Dict[str, Any]) -> ToolsConfig:
    web_search_raw = raw.get("web_search")
    if not isinstance(web_search_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing required tools.web_search config section",
            details={"field": "tools.web_search"},
        )
    return ToolsConfig(web_search=_build_web_search_config(web_search_raw))


@lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    raw = _load_json(ROOT_CONFIG_PATH)

    llm_raw = _require_section(raw, "llm")
    tts_raw = _require_section(raw, "tts")
    service_raw = _require_section(raw, "service")
    tools_raw = _require_section(raw, "tools")
    memory_raw = raw.get("memory_vector") or {}
    if not isinstance(memory_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "memory_vector must be an object",
            details={"field": "memory_vector"},
        )

    return AppConfig(
        llm=_build_llm_config(llm_raw),
        tts=_build_tts_config(tts_raw),
        service=_build_service_config(service_raw),
        memory_vector=_build_memory_vector_config(memory_raw, llm_raw),
        tools=_build_tools_config(tools_raw),
    )
