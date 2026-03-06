"""LLM-based signal extraction from raw user input."""
from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent
from typing import Optional

from .models import MemoryMetadata
from .utils import clamp


@dataclass
class ExtractedSignals:
    metadata: MemoryMetadata


class SignalExtractor:
    """Use LLM as the single judge for metadata extraction."""

    def __init__(
        self,
        llm_enabled: bool = False,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = "",
        llm_base_url: str = "",
        llm_timeout_seconds: int = 30,
        llm_temperature: float = 0.0,
    ):
        self.llm_enabled = llm_enabled
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_timeout_seconds = llm_timeout_seconds
        self.llm_temperature = llm_temperature

    def extract(self, content: str) -> ExtractedSignals:
        llm_data = self._extract_with_llm(content)
        if llm_data:
            metadata = MemoryMetadata(
                explicit_remember=bool(llm_data.get("explicit_remember", False)),
                future_use=bool(llm_data.get("future_use", False)),
                emotion_intensity=clamp(float(llm_data.get("emotion_intensity", 0.0))),
                near_repeat_score=0.0,  # Filled later in core via similarity lookup.
                repeat_count=0,
                temporal_urgency=clamp(float(llm_data.get("temporal_urgency", 0.0))),
                information_density=clamp(float(llm_data.get("information_density", 0.5))),
                llm_importance_hint=clamp(float(llm_data.get("importance_hint", 0.5))),
                confidence=clamp(float(llm_data.get("confidence", 0.6))),
                user_rating=self._safe_optional_float(llm_data.get("user_rating")),
            )
            return ExtractedSignals(metadata=metadata)

        # Fallback: neutral low-confidence to avoid over-persisting noisy inputs.
        return ExtractedSignals(
            metadata=MemoryMetadata(
                explicit_remember=False,
                future_use=False,
                emotion_intensity=0.0,
                near_repeat_score=0.0,
                repeat_count=0,
                temporal_urgency=0.0,
                information_density=0.4,
                llm_importance_hint=0.45,
                confidence=0.35,
                user_rating=None,
            )
        )

    def _extract_with_llm(self, content: str) -> Optional[dict]:
        if not self.llm_enabled:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None

        if not self.llm_api_key:
            return None

        prompt = dedent(
            f"""
            请提取以下用户输入的记忆评分信号，并按约定 JSON Schema 输出。
            要求：
            1. 所有分数字段都在 0-1 之间。
            2. 如果无法判断 user_rating，返回 null。

            示例：
            输入：请记住，下周三提醒我提交项目周报，我现在有点焦虑。
            输出示例：{{"explicit_remember":true,"future_use":true,"emotion_intensity":0.72,"temporal_urgency":0.66,"information_density":0.74,"importance_hint":0.83,"confidence":0.88,"user_rating":null}}

            待分析输入：{content}
            """
        ).strip()

        try:
            client = OpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url or None,
                timeout=self.llm_timeout_seconds,
            )
            resp = client.chat.completions.create(
                model=self.llm_model,
                temperature=self.llm_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": dedent(
                            """
                            你是一个记忆信号提取器。
                            你的任务是把用户输入转成结构化记忆评分因子。
                            必须严格按照给定 JSON Schema 输出，不要输出额外文本。
                            """
                        ).strip(),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "memory_signal_extraction",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "explicit_remember": {"type": "boolean"},
                                "future_use": {"type": "boolean"},
                                "emotion_intensity": {"type": "number", "minimum": 0, "maximum": 1},
                                "temporal_urgency": {"type": "number", "minimum": 0, "maximum": 1},
                                "information_density": {"type": "number", "minimum": 0, "maximum": 1},
                                "importance_hint": {"type": "number", "minimum": 0, "maximum": 1},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "user_rating": {
                                    "anyOf": [
                                        {"type": "number", "minimum": 0, "maximum": 1},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "required": [
                                "explicit_remember",
                                "future_use",
                                "emotion_intensity",
                                "temporal_urgency",
                                "information_density",
                                "importance_hint",
                                "confidence",
                                "user_rating",
                            ],
                        },
                    },
                },
            )
            text = resp.choices[0].message.content or ""
            return json.loads(text)
        except Exception:
            return None

    def _safe_optional_float(self, value) -> Optional[float]:
        if value is None:
            return None
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return None
