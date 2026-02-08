import httpx
import os
import json
from typing import Optional, Dict, Any, AsyncGenerator
from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    text_lang: str = "zh"
    ref_audio_path: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_lang: Optional[str] = None
    temperature: float = 0.6
    top_k: int = 5
    top_p: float = 1.0
    speed_factor: float = 1.0
    batch_size: int = 1
    streaming: bool = False

class TTSEngine:
    def __init__(self, base_url: str = None):

        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.base_url = config["GPT-SoVITS_url"]
        self.default_ref_path = config["ref_path"]
        self.default_prompt_text = config["prompt_text"]
        self.default_prompt_lang = config["prompt_lang"]

    async def synthesize(self, request: TTSRequest) -> Dict[str, Any]:
        """
        异步合成 TTS，返回音频字节或流式生成器。
        :return: {"success": bool, "audio_bytes": bytes or AsyncGenerator, "error": str}
        """
        payload = {
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
            "streaming_mode": request.streaming,
            "parallel_infer": True,
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                resp = await client.post(f"{self.base_url}/tts", json=payload)
                if resp.status_code != 200:
                    error = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    return {"success": False, "error": f"API 错误 ({resp.status_code}): {error}"}

                if request.streaming:
                    async def audio_stream() -> AsyncGenerator[bytes, None]:
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                yield chunk
                    return {"success": True, "audio_stream": audio_stream()}
                else:
                    return {"success": True, "audio_bytes": resp.content}

            except httpx.RequestError as e:
                return {"success": False, "error": f"连接失败: {str(e)}"}
            except Exception as e:
                return {"success": False, "error": f"合成异常: {str(e)}"}