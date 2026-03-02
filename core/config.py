import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.error_codes import ErrorCode
from core.errors import LuminaError


ROOT_DIR = Path(__file__).resolve().parents[1]


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
class AppConfig:
    llm: LLMConfig
    tts: TTSConfig
    service: ServiceConfig
    memory_vector: MemoryVectorConfig


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise LuminaError(
            ErrorCode.CONFIG_MISSING,
            "Config file not found",
            details={"path": str(path)},
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _env_or(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _to_int(value, path: str) -> int:
    try:
        return int(value)
    except Exception:
        raise LuminaError(
            ErrorCode.CONFIG_INVALID,
            "Invalid integer config value",
            details={"field": path, "value": value},
        )


def _to_bool(value, path: str) -> bool:
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
    raise LuminaError(
        ErrorCode.CONFIG_INVALID,
        "Invalid boolean config value",
        details={"field": path, "value": value},
    )


def _build_llm_config(raw: dict) -> LLMConfig:
    cfg = LLMConfig(
        chat_model=raw.get("chat_model", ""),
        chat_api_url=raw.get("chat_api_url", ""),
        chat_api_key=_env_or("LUMINA_API_KEY", raw.get("chat_api_key", "")),
        translate_model=raw.get("translate_model", ""),
        translate_api_url=raw.get("translate_api_url", ""),
        translate_api_key=_env_or("LUMINA_API_KEY", raw.get("translate_api_key", "")),
        chat_prompt=raw.get("chat_prompt", ""),
        translate_prompt=raw.get("translate_prompt", ""),
    )

    required = {
        "chat_model": cfg.chat_model,
        "chat_api_url": cfg.chat_api_url,
        "translate_model": cfg.translate_model,
        "translate_api_url": cfg.translate_api_url,
        "chat_prompt": cfg.chat_prompt,
        "translate_prompt": cfg.translate_prompt,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise LuminaError(
            ErrorCode.CONFIG_MISSING,
            "Missing required LLM config fields",
            details={"fields": missing},
        )
    return cfg


def _build_tts_config(raw: dict) -> TTSConfig:
    cfg = TTSConfig(
        gpt_sovits_url=raw.get("GPT-SoVITS_url", ""),
        ref_path=raw.get("ref_path", ""),
        prompt_text=raw.get("prompt_text", ""),
        prompt_lang=raw.get("prompt_lang", "ja"),
    )
    required = {
        "GPT-SoVITS_url": cfg.gpt_sovits_url,
        "ref_path": cfg.ref_path,
        "prompt_text": cfg.prompt_text,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise LuminaError(
            ErrorCode.CONFIG_MISSING,
            "Missing required TTS config fields",
            details={"fields": missing},
        )
    return cfg


def _build_service_config(raw: dict) -> ServiceConfig:
    char_info = raw.get("character_info", {})
    cfg = ServiceConfig(
        pet_name=char_info.get("pet_name", ""),
        username=char_info.get("username", ""),
        server_address=raw.get("server_address", "0.0.0.0"),
        server_port=_to_int(raw.get("server_port", 8080), "server_port"),
    )
    required = {
        "character_info.pet_name": cfg.pet_name,
        "character_info.username": cfg.username,
        "server_address": cfg.server_address,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise LuminaError(
            ErrorCode.CONFIG_MISSING,
            "Missing required service config fields",
            details={"fields": missing},
        )
    return cfg


def _build_memory_vector_config(service_raw: dict, llm_raw: dict) -> MemoryVectorConfig:
    mv = service_raw.get("memory_vector", {}) or {}

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
        vector_dim=_to_int(os.environ.get("LUMINA_VECTOR_DIM", mv.get("vector_dim", 1536)), "memory_vector.vector_dim"),
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
            raise LuminaError(
                ErrorCode.CONFIG_MISSING,
                "Missing required memory vector config fields",
                details={"fields": missing},
            )

    return cfg


@lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    llm_raw = _load_json(ROOT_DIR / "core" / "llm" / "config.json")
    tts_raw = _load_json(ROOT_DIR / "core" / "tts" / "config.json")
    service_raw = _load_json(ROOT_DIR / "service" / "pet" / "config.json")

    return AppConfig(
        llm=_build_llm_config(llm_raw),
        tts=_build_tts_config(tts_raw),
        service=_build_service_config(service_raw),
        memory_vector=_build_memory_vector_config(service_raw, llm_raw),
    )
