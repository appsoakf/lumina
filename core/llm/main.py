import logging
from typing import Generator, Optional

from openai import OpenAI

from core.config import load_app_config
from core.error_codes import ErrorCode
from core.errors import LuminaError

logger = logging.getLogger(__name__)


class ChatEngine:
    def __init__(self):
        cfg = load_app_config().llm
        if not cfg.chat_api_key:
            raise LuminaError(
                ErrorCode.CONFIG_MISSING,
                "Missing LLM API key. Set LUMINA_API_KEY or chat_api_key in config.",
                details={"field": "chat_api_key"},
            )
        self.client = OpenAI(api_key=cfg.chat_api_key, base_url=cfg.chat_api_url)
        self.prompt = cfg.chat_prompt
        self.model = cfg.chat_model
        self.last_error: Optional[LuminaError] = None

    def generate_stream(self, msg: str, history: list[dict]) -> Generator[str, None, None]:
        self.last_error = None
        try:
            history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": self.prompt}]
            messages.extend(history)

            for chunk in self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            ):
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            self.last_error = LuminaError(
                ErrorCode.LLM_STREAM_ERROR,
                f"LLM stream failed: {exc}",
                retryable=True,
            )
            logger.error(self.last_error.message)
            yield "LLM生成出错，请稍后重试。"


class TranslateEngine:
    def __init__(self):
        cfg = load_app_config().llm
        if not cfg.translate_api_key:
            raise LuminaError(
                ErrorCode.CONFIG_MISSING,
                "Missing translate API key. Set LUMINA_API_KEY or translate_api_key in config.",
                details={"field": "translate_api_key"},
            )
        self.client = OpenAI(api_key=cfg.translate_api_key, base_url=cfg.translate_api_url)
        self.model = cfg.translate_model
        self.translate_prompt = cfg.translate_prompt
        self.last_error: Optional[LuminaError] = None

    def translate(self, text: str) -> str:
        self.last_error = None
        try:
            messages = [
                {"role": "system", "content": self.translate_prompt + "/no_think"},
                {"role": "user", "content": text},
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            result = (completion.choices[0].message.content or "").strip()
            if not result:
                self.last_error = LuminaError(
                    ErrorCode.TRANSLATE_EMPTY_RESULT,
                    "Translate returned empty content",
                    retryable=True,
                )
                logger.warning(self.last_error.message)
            return result
        except Exception as exc:
            self.last_error = LuminaError(
                ErrorCode.TRANSLATE_API_ERROR,
                f"Translate failed: {exc}",
                retryable=True,
            )
            logger.error(self.last_error.message)
            return ""
