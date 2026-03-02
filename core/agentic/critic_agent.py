import json
import logging
import re
from typing import Dict, List

from openai import OpenAI

from core.config import load_app_config
from core.error_codes import ErrorCode
from core.errors import LuminaError, error_payload
from core.protocols import CriticResult, PlanResult

logger = logging.getLogger(__name__)


class CriticAgent:
    """Critic agent: reviews multi-step execution quality and returns corrections."""

    def __init__(self):
        llm_cfg = load_app_config().llm
        if not llm_cfg.chat_api_key:
            raise LuminaError(
                ErrorCode.CONFIG_MISSING,
                "Missing LLM API key for critic agent",
                details={"field": "chat_api_key"},
            )
        self.client = OpenAI(api_key=llm_cfg.chat_api_key, base_url=llm_cfg.chat_api_url)
        self.model = llm_cfg.chat_model

    def _invoke(self, messages: List[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=0.1,
        )
        return (completion.choices[0].message.content or "").strip()

    def _extract_json(self, text: str) -> Dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
        return json.loads(text)

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
            logger.warning(f"Critic fallback applied: {exc}")
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
