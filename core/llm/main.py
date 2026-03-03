import logging
from dataclasses import dataclass
from typing import Optional

from core.config import load_app_config
from core.llm.chat_service import ChatCompletionService
from core.utils.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslateResult:
    """Single-call translation outcome; no shared mutable error state."""

    text: str
    error: Optional[AppError] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class TranslateEngine:
    def __init__(self):
        cfg = load_app_config().llm
        self.chat_llm = ChatCompletionService.from_translate_config(
            default_temperature=0.0,
            missing_key_message="Missing translate API key. Set LUMINA_API_KEY or translate_api_key in config.",
            missing_key_field="translate_api_key",
        )
        self.translate_prompt = cfg.translate_prompt

    def translate(self, text: str) -> str:
        return self.translate_with_status(text).text

    def translate_with_status(self, text: str) -> TranslateResult:
        try:
            messages = [
                {"role": "system", "content": self.translate_prompt + "/no_think"},
                {"role": "user", "content": text},
            ]
            completion = self.chat_llm.invoke(
                messages=messages,
                temperature=0.0,
            )
            result = (completion.choices[0].message.content or "").strip()
            if not result:
                err = AppError(
                    ErrorCode.TRANSLATE_EMPTY_RESULT,
                    "Translate returned empty content",
                    retryable=True,
                )
                logger.warning(err.message)
                return TranslateResult(text="", error=err)
            return TranslateResult(text=result)
        except Exception as exc:
            err = AppError(
                ErrorCode.TRANSLATE_API_ERROR,
                f"Translate failed: {exc}",
                retryable=True,
            )
            logger.error(err.message)
            return TranslateResult(text="", error=err)
