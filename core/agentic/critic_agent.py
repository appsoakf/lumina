import json
import logging
from typing import Dict, List

from core.agentic.base import BaseLLMAgent
from core.agentic.json_mixin import JSONParseMixin
from core.utils import log_exception
from core.utils.errors import ErrorCode, error_payload
from core.protocols import CriticResult, PlanResult

logger = logging.getLogger(__name__)


class CriticAgent(BaseLLMAgent, JSONParseMixin):
    """Critic agent: reviews multi-step execution quality and returns corrections."""

    def __init__(self):
        super().__init__(
            missing_key_message="Missing LLM API key for critic agent",
            missing_key_field="chat_api_key",
            default_temperature=0.1,
        )

    def _invoke(self, messages: List[Dict[str, str]]) -> str:
        completion = self.invoke_chat(messages, temperature=0.1)
        return (completion.choices[0].message.content or "").strip()

    def _extract_json(self, text: str) -> Dict:
        return self.parse_json_object(text, allow_brace_extract=False)

    def review_task(self, user_text: str, plan_result: PlanResult, execution_graph: Dict) -> CriticResult:
        system_prompt = (
            "你是 critic_agent，负责审查任务执行质量。"
            "请只输出 JSON，格式："
            "{\"quality\":\"pass|revise\",\"issues\":[\"...\"],\"suggestions\":[\"...\"],\"summary\":\"...\"}"
            "如果结果可直接交付，quality=pass；否则给 revise 和具体改进建议。"
        )
        payload = {
            "user_request": user_text,
            "plan": plan_result.to_dict(),
            "execution_graph": execution_graph,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            raw = self._invoke(messages)
            data = self._extract_json(raw)
            quality = str(data.get("quality") or "pass").lower()
            issues = [str(x) for x in (data.get("issues") or [])]
            suggestions = [str(x) for x in (data.get("suggestions") or [])]
            summary = str(data.get("summary") or "")

            if quality not in {"pass", "revise"}:
                quality = "pass"

            return CriticResult(
                quality=quality,
                issues=issues,
                suggestions=suggestions,
                summary=summary,
            )
        except Exception as exc:
            log_exception(
                logger,
                "critic.review.error",
                "Critic 执行失败，返回默认评审结果",
                component="agent",
                fallback="pass",
            )
            return CriticResult(
                quality="pass",
                issues=[],
                suggestions=[],
                summary="评审代理暂不可用，已返回当前执行结果。",
                error=error_payload(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Critic failed: {exc}",
                    retryable=True,
                ),
            )
