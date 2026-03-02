import logging
import threading
from typing import Any, Dict, Optional

import requests as sync_requests
from pydantic import BaseModel

from core.config import load_app_config
from core.utils.errors import ErrorCode

logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    text: str
    text_lang: str = "ja"
    ref_audio_path: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_lang: Optional[str] = None
    temperature: float = 0.6
    top_k: int = 5
    top_p: float = 1.0
    speed_factor: float = 1.0
    batch_size: int = 1
    parallel_infer: bool = False
    media_type: str = "wav"


class TTSEngine:
    def __init__(self):
        cfg = load_app_config().tts
        self.base_url = cfg.gpt_sovits_url
        self.default_ref_path = cfg.ref_path
        self.default_prompt_text = cfg.prompt_text
        self.default_prompt_lang = cfg.prompt_lang
        self._thread_local = threading.local()

    def _get_sync_session(self) -> sync_requests.Session:
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = sync_requests.Session()
        return self._thread_local.session

    def _build_payload(self, request: TTSRequest) -> dict:
        return {
            "text": request.text,
            "text_lang": request.text_lang,
            "ref_audio_path": request.ref_audio_path or self.default_ref_path,
            "prompt_text": request.prompt_text or self.default_prompt_text,
            "prompt_lang": request.prompt_lang or self.default_prompt_lang,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "speed_factor": request.speed_factor,
            "batch_size": request.batch_size,
            "text_split_method": "cut5",
            "media_type": request.media_type,
            "parallel_infer": False,
        }

    def _failure(self, code: ErrorCode, message: str, retryable: bool = True) -> Dict[str, Any]:
        return {
            "success": False,
            "error": message,
            "error_code": code.value,
            "retryable": retryable,
        }

    def synthesize_streaming(self, request: TTSRequest) -> Dict[str, Any]:
        payload = self._build_payload(request)
        payload["streaming_mode"] = True

        session = self._get_sync_session()
        try:
            resp = session.post(
                f"{self.base_url}/tts",
                json=payload,
                stream=True,
                timeout=180,
            )
            if resp.status_code != 200:
                resp.close()
                return self._failure(
                    ErrorCode.TTS_API_ERROR,
                    f"TTS API error ({resp.status_code})",
                    retryable=True,
                )
        except Exception as exc:
            logger.error(f"TTS connection error: {exc}")
            return self._failure(
                ErrorCode.TTS_CONNECTION_ERROR,
                f"TTS connection failed: {exc}",
                retryable=True,
            )

        def audio_stream():
            try:
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            except Exception as exc:
                logger.error(f"TTS stream iteration error: {exc}")
            finally:
                resp.close()

        return {
            "success": True,
            "audio_stream": audio_stream(),
            "error_code": None,
            "retryable": False,
        }
