import json
import logging

from core.utils import log_event

logger = logging.getLogger(__name__)


class EmotionEngine:
    """情感 → TTS 参考音频映射。"""

    def __init__(self):
        self.emotions_map = {
            "愤怒_1": {"ref_path": "ref_audios/anger.wav", "prompt_text": "ギリギリまで抵抗してみるから邪魔はしないでくれ"},
            "愤怒_2": {"ref_path": "ref_audios/anger.wav", "prompt_text": "ギリギリまで抵抗してみるから邪魔はしないでくれ"},
            "愤怒_3": {"ref_path": "ref_audios/anger.wav", "prompt_text": "ギリギリまで抵抗してみるから邪魔はしないでくれ"},
            "平静_1": {"ref_path": "ref_audios/calm.wav", "prompt_text": "悪くないな。眠れないのなら一緒にどうだ?"},
            "平静_2": {"ref_path": "ref_audios/calm.wav", "prompt_text": "悪くないな。眠れないのなら一緒にどうだ?"},
            "平静_3": {"ref_path": "ref_audios/calm.wav", "prompt_text": "悪くないな。眠れないのなら一緒にどうだ?"},
            "自信_1": {"ref_path": "ref_audios/confident.wav", "prompt_text": "遠慮するな! 私もろとも攻撃してくれ!"},
            "自信_2": {"ref_path": "ref_audios/confident.wav", "prompt_text": "遠慮するな! 私もろとも攻撃してくれ!"},
            "自信_3": {"ref_path": "ref_audios/confident.wav", "prompt_text": "遠慮するな! 私もろとも攻撃してくれ!"},
            "失落_1": {"ref_path": "ref_audios/depressed.wav", "prompt_text": "力と耐久力には自信があるのだが不器用で"},
            "失落_2": {"ref_path": "ref_audios/depressed.wav", "prompt_text": "力と耐久力には自信があるのだが不器用で"},
            "失落_3": {"ref_path": "ref_audios/depressed.wav", "prompt_text": "力と耐久力には自信があるのだが不器用で"},
            "怀疑_1": {"ref_path": "ref_audios/doubt.wav", "prompt_text": "いいだろう、私の一世一代の大勝負"},
            "怀疑_2": {"ref_path": "ref_audios/doubt.wav", "prompt_text": "いいだろう、私の一世一代の大勝負"},
            "怀疑_3": {"ref_path": "ref_audios/doubt.wav", "prompt_text": "いいだろう、私の一世一代の大勝負"},
            "兴奋_1": {"ref_path": "ref_audios/excited_1.wav", "prompt_text": "鎧がないこの状態で、もし重い一撃をくらったら……"},
            "兴奋_2": {"ref_path": "ref_audios/excited_2.wav", "prompt_text": "私の水着姿で変な妄想をするな"},
            "兴奋_3": {"ref_path": "ref_audios/excited_3.wav", "prompt_text": "やはりパーティーは最高だ"},
            "开心_1": {"ref_path": "ref_audios/happy_1.wav", "prompt_text": "何でも言ってくれ。夜中に子供たちにプレゼントを届ける。"},
            "开心_2": {"ref_path": "ref_audios/happy_2.wav", "prompt_text": "一歩たりとも引くつもりはない今は軽装だが私はクルセイダー"},
            "开心_3": {"ref_path": "ref_audios/happy_1.wav", "prompt_text": "何でも言ってくれ。夜中に子供たちにプレゼントを届ける。"},
            "严肃_1": {"ref_path": "ref_audios/serious.wav", "prompt_text": "己の使命を全うしよう昔から魔王にエロい目に合わされるのは"},
            "严肃_2": {"ref_path": "ref_audios/serious.wav", "prompt_text": "己の使命を全うしよう昔から魔王にエロい目に合わされるのは"},
            "严肃_3": {"ref_path": "ref_audios/serious.wav", "prompt_text": "己の使命を全うしよう昔から魔王にエロい目に合わされるのは"},
            "害羞_1": {"ref_path": "ref_audios/shame.wav", "prompt_text": "あまりジロジロ見られると恥ずかしいんだが"},
            "害羞_2": {"ref_path": "ref_audios/shame.wav", "prompt_text": "あまりジロジロ見られると恥ずかしいんだが"},
            "害羞_3": {"ref_path": "ref_audios/shame.wav", "prompt_text": "あまりジロジロ見られると恥ずかしいんだが"}
        }
        self.default_emotion = "平静"
        self.default_intensity = "1"

        
    def parse_leading_json(self, raw: str) -> tuple[str, str, str]:
        """从文本开头提取句首 JSON，返回 (emotion, text, intensity)。
        格式: {"emotion": "开心", "intensity": 2}\n正常文本
        解析失败时返回 (default_emotion, raw, default_intensity)。"""
        try:
            # 找到第一个 } 的位置
            brace_end = raw.find('}')
            if brace_end == -1:
                log_event(
                    logger,
                    logging.WARNING,
                    "emotion.parse.header_missing",
                    "情绪JSON解析失败：未找到结束花括号",
                    component="emotion",
                    text_preview=str(raw or "")[:80],
                    text_len=len(str(raw or "")),
                )
                return self.default_emotion, raw, self.default_intensity
            json_str = raw[:brace_end + 1]
            data = json.loads(json_str)
            emotion = str(data.get("emotion", self.default_emotion))
            intensity = str(data.get("intensity", self.default_intensity))
            # } 之后的内容为正文，去掉开头的换行/空白
            text = raw[brace_end + 1:].lstrip('\n')
            if not text.strip():
                text = ""  # fallback: 如果没有正文，返回空字符串
            return emotion, text, intensity
        except (json.JSONDecodeError, AttributeError) as e:
            log_event(
                logger,
                logging.WARNING,
                "emotion.parse.json_error",
                "情绪JSON解析失败：JSON格式不合法",
                component="emotion",
                error_message=str(e),
                text_preview=str(raw or "")[:80],
                text_len=len(str(raw or "")),
            )
            return self.default_emotion, raw, self.default_intensity

    def get_ref_audio_intensity(self, emotion: str, intensity: str) -> str:
        """emotion + intensity → ref_audio_path，未知情感回退到默认。"""
        info = self.emotions_map.get(emotion + "_" + intensity, 
                                     self.emotions_map[self.default_emotion + "_" + self.default_intensity])
        return info["ref_path"]

    def get_prompt_text_intensity(self, emotion: str, intensity: str) -> str:
        """emotion + intensity → prompt_text，未知情感回退到默认。"""
        info = self.emotions_map.get(emotion + "_" + intensity, 
                                     self.emotions_map[self.default_emotion + "_" + self.default_intensity])
        return info["prompt_text"]
