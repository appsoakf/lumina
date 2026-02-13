import json
import logging

logger = logging.getLogger(__name__)


class EmotionEngine:
    """情感 → TTS 参考音频映射。"""

    def __init__(self):
        self.emotions_map = {
            "内疚": {"ref_path": "ref_audios/guilt.wav", "prompt_text": "わ、私……何か不適切なことを、言ったかしら"},
            "温柔": {"ref_path": "ref_audios/soft.wav", "prompt_text": "新年、明けましておめでとう、先生"},
            "平静": {"ref_path": "ref_audios/calm.wav", "prompt_text": "全て順調よ。でも、安心できる段階ではないわ"},
            "抱歉": {"ref_path": "ref_audios/sorry.wav", "prompt_text": "こ、これはわざとじゃないの……！ごめんなさい"},
            "害羞": {"ref_path": "ref_audios/shy.wav", "prompt_text": "ッ……！な、なにを……私の「お世話」……!"}
        }
        self.default_emotion = "平静"

    def parse_response(self, raw: str) -> tuple[str, str]:
        """从 LLM 的 JSON 输出中提取 (emotion, text)。
        解析失败时返回 (default_emotion, raw)，保证不中断流程。"""
        try:
            data = json.loads(raw)
            emotion = data.get("emotion", self.default_emotion)
            text = data.get("text", raw)
            return emotion, text
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Emotion parse failed, using default. raw={raw[:80]}")
            return self.default_emotion, raw

    def get_ref_audio(self, emotion: str) -> str:
        """emotion → ref_audio_path，未知情感回退到默认。"""
        info = self.emotions_map.get(emotion, self.emotions_map[self.default_emotion])
        return info["ref_path"]

    def get_prompt_text(self, emotion: str) -> str:
        """emotion → prompt_text，未知情感回退到默认。"""
        info = self.emotions_map.get(emotion, self.emotions_map[self.default_emotion])
        return info["prompt_text"]
