"""Overflow processing for old working memories."""
from __future__ import annotations

import json
import math
import time
import uuid
from textwrap import dedent
from typing import List

from .models import MemoryItem, MemoryMetadata
from .utils import clamp, cosine_similarity, normalize_text


class OverflowProcessor:
    """Cluster and summarize FIFO overflow memories."""

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        max_cluster_size: int = 6,
        llm_enabled: bool = False,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = "",
        llm_base_url: str = "",
        llm_timeout_seconds: int = 30,
        llm_temperature: float = 0.2,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size
        self.llm_enabled = llm_enabled
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_timeout_seconds = llm_timeout_seconds
        self.llm_temperature = llm_temperature

    def cluster(self, items: List[MemoryItem]) -> List[List[MemoryItem]]:
        clusters: List[List[MemoryItem]] = []
        for item in items:
            placed = False
            for cluster in clusters:
                if self._is_close(item, cluster):
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])
        return clusters

    def build_summaries(self, clusters: List[List[MemoryItem]]) -> List[MemoryItem]:
        summaries = []
        now = time.time()
        for cluster in clusters:
            if not cluster:
                continue

            # overflow 汇总不是简单平均：
            # 被更多次命中的 working 记忆在摘要中占更高权重。
            summary_text = self._try_llm_summary(cluster) or self._summarize_cluster(cluster)
            avg_importance = self._weighted_average(
                cluster,
                value_getter=lambda item: item.importance,
            )
            avg_confidence = self._weighted_average(
                cluster,
                value_getter=lambda item: item.metadata.confidence,
            )
            avg_emotion = self._weighted_average(
                cluster,
                value_getter=lambda item: item.metadata.emotion_intensity,
            )

            metadata = MemoryMetadata(
                created_at=now,
                store="long_term",
                state="active",
                half_life_days=30.0,
                explicit_remember=False,
                future_use=any(item.metadata.future_use for item in cluster),
                emotion_intensity=clamp(avg_emotion),
                near_repeat_score=1.0,
                repeat_count=len(cluster),
                temporal_urgency=clamp(
                    self._weighted_average(
                        cluster,
                        value_getter=lambda item: item.metadata.temporal_urgency,
                    )
                ),
                information_density=0.9,
                llm_importance_hint=clamp(min(avg_importance + 0.05, 1.0)),
                confidence=clamp(min(avg_confidence + 0.05, 1.0)),
                user_rating=None,
            )

            summary_item = MemoryItem(
                id=str(uuid.uuid4()),
                content=summary_text,
                importance=round(clamp(min(avg_importance + 0.08, 1.0)), 4),
                embedding=None,
                recall_count=0,
                metadata=metadata,
            )
            summaries.append(summary_item)
        return summaries

    def _weighted_average(self, cluster: List[MemoryItem], value_getter) -> float:
        # recall_count 越高，说明该条在工作记忆阶段被反复命中，汇总时赋予更高权重。
        # 采用 log1p 是为了“增长但不爆炸”：高命中条目更重要，但不会无限放大。
        weighted_sum = 0.0
        weight_total = 0.0
        for item in cluster:
            weight = 1.0 + math.log1p(max(item.recall_count, 0))
            weighted_sum += weight * float(value_getter(item))
            weight_total += weight
        if weight_total <= 0:
            return 0.0
        return weighted_sum / weight_total

    def _is_close(self, item: MemoryItem, cluster: List[MemoryItem]) -> bool:
        if len(cluster) >= self.max_cluster_size:
            return False
        sims = []
        for other in cluster:
            if item.embedding is None or other.embedding is None:
                sims.append(self._text_overlap(item, other))
            else:
                sims.append(cosine_similarity(item.embedding, other.embedding))
        return (sum(sims) / max(len(sims), 1)) >= self.similarity_threshold

    def _text_overlap(self, a: MemoryItem, b: MemoryItem) -> float:
        left = set(normalize_text(a.content).split())
        right = set(normalize_text(b.content).split())
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _summarize_cluster(self, cluster: List[MemoryItem]) -> str:
        lines = []
        for idx, item in enumerate(cluster, start=1):
            text = normalize_text(item.content)
            short = text[:70] + ("..." if len(text) > 70 else "")
            lines.append(f"{idx}. {short}")
        return " | ".join(lines)

    def _try_llm_summary(self, cluster: List[MemoryItem]) -> str | None:
        if not self.llm_enabled:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None

        if not self.llm_api_key:
            return None

        snippets = []
        for idx, item in enumerate(cluster, start=1):
            snippets.append(f"{idx}. {item.content.strip()}")
        joined_snippets = "\n".join(snippets)
        prompt = dedent(
            f"""
            请将下面多条短记忆聚类后压缩为一条高密度中文摘要，并严格按 JSON Schema 输出。
            要求：
            1. summary 控制在 220 字以内。
            2. 保留核心事实与时间线，不要重复。
            3. 语气客观，不做额外推测。

            示例：
            输入：1. 用户下周要交周报。2. 用户希望周三提醒。3. 用户对延期感到焦虑。
            输出示例：{{"summary":"用户下周需提交周报，希望在周三收到提醒，并对延期存在焦虑情绪。"}}

            待处理输入：
            {joined_snippets}
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
                            你是记忆聚类摘要器。
                            你必须严格按给定 JSON Schema 输出，
                            不要输出解释文本或 markdown。
                            """
                        ).strip(),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "overflow_cluster_summary",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "summary": {"type": "string", "minLength": 1, "maxLength": 220}
                            },
                            "required": ["summary"],
                        },
                    },
                },
            )
            content = resp.choices[0].message.content or ""
            parsed = json.loads(content)
            return parsed.get("summary")
        except Exception:
            return None
