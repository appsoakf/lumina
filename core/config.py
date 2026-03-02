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
class AppConfig:
    llm: LLMConfig
    tts: TTSConfig
    service: ServiceConfig


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


@lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    llm_raw = _load_json(ROOT_DIR / "core" / "llm" / "config.json")
    tts_raw = _load_json(ROOT_DIR / "core" / "tts" / "config.json")
    service_raw = _load_json(ROOT_DIR / "service" / "pet" / "config.json")

    return AppConfig(
        llm=_build_llm_config(llm_raw),
        tts=_build_tts_config(tts_raw),
        service=_build_service_config(service_raw),
    )
