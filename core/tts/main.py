import httpx
import os
import json
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

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

class TTSEngine:
    def __init__(self, base_url: str = None):

        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.base_url = config["GPT-SoVITS_url"]
        self.default_ref_path = config["ref_path"]
        self.default_prompt_text = config["prompt_text"]
        self.default_prompt_lang = config["prompt_lang"]
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=180.0)
        return self._client

    def _build_payload(self, request: TTSRequest) -> dict:
        """构建 TTS API 请求参数。"""
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
            "media_type": "wav",
            "parallel_infer": True,
        }

    async def synthesize(self, request: TTSRequest, request_id: str = None) -> Dict[str, Any]:
        """非流式合成 TTS，返回完整音频字节。"""
        req_id = request_id or "unknown"
        payload = self._build_payload(request)
        payload["streaming_mode"] = False

        client = await self._get_client()
        try:
            resp = await client.post(f"{self.base_url}/tts", json=payload)

            if resp.status_code != 200:
                error = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                logger.error(f"[{req_id}] TTS error: API {resp.status_code} - {error}")
                return {"success": False, "error": f"API 错误 ({resp.status_code}): {error}"}

            return {"success": True, "audio_bytes": resp.content}

        except httpx.RequestError as e:
            logger.error(f"[{req_id}] TTS error: connection failed - {e}")
            return {"success": False, "error": f"连接失败: {str(e)}"}
        except Exception as e:
            logger.error(f"[{req_id}] TTS error: {e}")
            return {"success": False, "error": f"合成异常: {str(e)}"}

    async def synthesize_streaming(self, request: TTSRequest, request_id: str = None):
        """流式合成 TTS，立即建连并验证状态码，返回 async generator 逐块 yield 音频字节。"""
        req_id = request_id or "unknown"
        payload = self._build_payload(request)
        payload["streaming_mode"] = True

        client = await self._get_client()

        try:
            resp_ctx = client.stream("POST", f"{self.base_url}/tts", json=payload)
            resp = await resp_ctx.__aenter__()
            if resp.status_code != 200:
                await resp_ctx.__aexit__(None, None, None)
                return {"success": False, "error": f"API 错误 ({resp.status_code})"}
        except Exception as e:
            logger.error(f"[{req_id}] TTS streaming connection error: {e}")
            return {"success": False, "error": str(e)}

        async def audio_stream():
            try:
                async for chunk in resp.aiter_bytes(chunk_size=None):
                    if chunk:
                        yield chunk
            except Exception as e:
                logger.error(f"[{req_id}] TTS streaming iteration error: {e}")
            finally:
                await resp_ctx.__aexit__(None, None, None)

        return {"success": True, "audio_stream": audio_stream()}