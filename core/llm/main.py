import logging
from typing import Optional

from openai import OpenAI

from core.config import load_app_config
from core.utils.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)


class TranslateEngine:
    def __init__(self):
        cfg = load_app_config().llm
        if not cfg.translate_api_key:
            raise AppError(
                ErrorCode.CONFIG_MISSING,
                "Missing translate API key. Set LUMINA_API_KEY or translate_api_key in config.",
                details={"field": "translate_api_key"},
            )
        self.client = OpenAI(api_key=cfg.translate_api_key, base_url=cfg.translate_api_url)
        self.model = cfg.translate_model
        self.translate_prompt = cfg.translate_prompt
        self.last_error: Optional[AppError] = None

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
                self.last_error = AppError(
                    ErrorCode.TRANSLATE_EMPTY_RESULT,
                    "Translate returned empty content",
                    retryable=True,
                )
                logger.warning(self.last_error.message)
            return result
        except Exception as exc:
            self.last_error = AppError(
                ErrorCode.TRANSLATE_API_ERROR,
                f"Translate failed: {exc}",
                retryable=True,
            )
            logger.error(self.last_error.message)
            return ""
