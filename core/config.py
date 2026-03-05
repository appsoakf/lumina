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
    enable_translation: bool
    enable_tts: bool


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
    timeout_sec: float
    max_top_k: int
    uapis: "UapiSearchConfig"


@dataclass
class UapiSearchConfig:
    endpoint: str
    api_key: str
    default_sort: str
    default_fetch_full: bool


@dataclass
class ToolsConfig:
    web_search: WebSearchConfig


@dataclass
class LoggingConfig:
    level: str
    format: str
    log_dir: str
    log_file_name: str
    event_file_name: str
    enable_console: bool
    enable_file: bool
    enable_event_file: bool
    slow_threshold_ms: int
    redact_user_text: bool
    user_text_preview_chars: int


@dataclass
class TaskFlowConfig:
    max_replan_rounds: int
    max_clarify_rounds: int


@dataclass
class AppConfig:
    llm: LLMConfig
    tts: TTSConfig
    service: ServiceConfig
    memory_vector: MemoryVectorConfig
    tools: ToolsConfig
    logging: LoggingConfig
    task_flow: TaskFlowConfig


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
        enable_translation=_to_bool(raw.get("enable_translation", False), "service.enable_translation"),
        enable_tts=_to_bool(raw.get("enable_tts", False), "service.enable_tts"),
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


def _build_uapis_config(raw: Dict[str, Any]) -> UapiSearchConfig:
    endpoint = str(raw.get("endpoint", "")).strip()
    api_key = str(raw.get("api_key", "")).strip()
    default_sort = str(raw.get("default_sort", "relevance")).strip().lower() or "relevance"
    default_fetch_full = _to_bool(raw.get("default_fetch_full", False), "tools.web_search.uapis.default_fetch_full")

    if not endpoint:
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing uapis endpoint",
            details={"field": "tools.web_search.uapis.endpoint"},
        )
    if default_sort not in {"relevance", "date"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid uapis default_sort",
            details={
                "field": "tools.web_search.uapis.default_sort",
                "value": default_sort,
                "allowed": ["relevance", "date"],
            },
        )

    return UapiSearchConfig(
        endpoint=endpoint,
        api_key=api_key,
        default_sort=default_sort,
        default_fetch_full=default_fetch_full,
    )


def _build_web_search_config(raw: Dict[str, Any]) -> WebSearchConfig:
    timeout_raw = raw.get("timeout_sec")
    max_top_k_raw = raw.get("max_top_k")

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

    uapis_raw = raw.get("uapis")
    if not isinstance(uapis_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_MISSING,
            "Missing tools.web_search.uapis config section",
            details={"field": "tools.web_search.uapis"},
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
        timeout_sec=timeout_sec,
        max_top_k=max_top_k,
        uapis=_build_uapis_config(uapis_raw),
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


def _build_logging_config(raw: Dict[str, Any]) -> LoggingConfig:
    payload = dict(raw or {})

    level = str(payload.get("level", "INFO")).strip().upper() or "INFO"
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid logging level",
            details={
                "field": "logging.level",
                "value": level,
                "allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            },
        )

    fmt = str(payload.get("format", "both")).strip().lower() or "both"
    if fmt not in {"human", "json", "both"}:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "Invalid logging format mode",
            details={
                "field": "logging.format",
                "value": fmt,
                "allowed": ["human", "json", "both"],
            },
        )

    log_dir = str(payload.get("log_dir", "logs")).strip() or "logs"
    log_file_name = str(payload.get("log_file_name", "lumina.log")).strip() or "lumina.log"
    event_file_name = str(payload.get("event_file_name", "events.jsonl")).strip() or "events.jsonl"

    enable_console = _to_bool(payload.get("enable_console", True), "logging.enable_console")
    enable_file = _to_bool(payload.get("enable_file", True), "logging.enable_file")
    enable_event_file = _to_bool(payload.get("enable_event_file", True), "logging.enable_event_file")

    slow_threshold_ms = _to_int(payload.get("slow_threshold_ms", 1000), "logging.slow_threshold_ms")
    if slow_threshold_ms < 0:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "logging.slow_threshold_ms must be >= 0",
            details={"field": "logging.slow_threshold_ms", "value": slow_threshold_ms},
        )

    redact_user_text = _to_bool(payload.get("redact_user_text", True), "logging.redact_user_text")
    user_text_preview_chars = _to_int(
        payload.get("user_text_preview_chars", 120),
        "logging.user_text_preview_chars",
    )
    if user_text_preview_chars < 0:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "logging.user_text_preview_chars must be >= 0",
            details={"field": "logging.user_text_preview_chars", "value": user_text_preview_chars},
        )

    return LoggingConfig(
        level=level,
        format=fmt,
        log_dir=log_dir,
        log_file_name=log_file_name,
        event_file_name=event_file_name,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_event_file=enable_event_file,
        slow_threshold_ms=slow_threshold_ms,
        redact_user_text=redact_user_text,
        user_text_preview_chars=user_text_preview_chars,
    )


def _build_task_flow_config(raw: Dict[str, Any]) -> TaskFlowConfig:
    payload = dict(raw or {})
    max_replan_rounds = _to_int(payload.get("max_replan_rounds", 2), "task_flow.max_replan_rounds")
    max_clarify_rounds = _to_int(payload.get("max_clarify_rounds", 3), "task_flow.max_clarify_rounds")
    if max_replan_rounds < 0:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "task_flow.max_replan_rounds must be >= 0",
            details={"field": "task_flow.max_replan_rounds", "value": max_replan_rounds},
        )
    if max_clarify_rounds < 1:
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "task_flow.max_clarify_rounds must be >= 1",
            details={"field": "task_flow.max_clarify_rounds", "value": max_clarify_rounds},
        )
    return TaskFlowConfig(
        max_replan_rounds=max_replan_rounds,
        max_clarify_rounds=max_clarify_rounds,
    )


@lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    raw = _load_json(ROOT_CONFIG_PATH)

    llm_raw = _require_section(raw, "llm")
    tts_raw = _require_section(raw, "tts")
    service_raw = _require_section(raw, "service")
    tools_raw = _require_section(raw, "tools")
    logging_raw = raw.get("logging") or {}
    if not isinstance(logging_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "logging must be an object",
            details={"field": "logging"},
        )
    memory_raw = raw.get("memory_vector") or {}
    if not isinstance(memory_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "memory_vector must be an object",
            details={"field": "memory_vector"},
        )
    task_flow_raw = raw.get("task_flow") or {}
    if not isinstance(task_flow_raw, dict):
        raise AppError(
            ErrorCode.CONFIG_INVALID,
            "task_flow must be an object",
            details={"field": "task_flow"},
        )

    return AppConfig(
        llm=_build_llm_config(llm_raw),
        tts=_build_tts_config(tts_raw),
        service=_build_service_config(service_raw),
        memory_vector=_build_memory_vector_config(memory_raw, llm_raw),
        tools=_build_tools_config(tools_raw),
        logging=_build_logging_config(logging_raw),
        task_flow=_build_task_flow_config(task_flow_raw),
    )
