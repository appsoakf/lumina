import json
import logging
import re
from typing import Dict, List

from openai import OpenAI

from core.config import load_app_config
from core.error_codes import ErrorCode
from core.errors import LuminaError, error_payload
from core.protocols import PlanItem, PlanResult

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Planner agent: decomposes user request into executable steps."""

    def __init__(self, max_steps: int = 6):
        llm_cfg = load_app_config().llm
        if not llm_cfg.chat_api_key:
            raise LuminaError(
                ErrorCode.CONFIG_MISSING,
                "Missing LLM API key for planner agent",
                details={"field": "chat_api_key"},
            )
        self.client = OpenAI(api_key=llm_cfg.chat_api_key, base_url=llm_cfg.chat_api_url)
        self.model = llm_cfg.chat_model
        self.max_steps = max_steps

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

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _fallback_plan(self, user_text: str) -> PlanResult:
        steps = [
            PlanItem(step_id="S1", title="明确目标与约束", instruction=f"提炼用户任务目标与限制条件：{user_text}"),
            PlanItem(step_id="S2", title="生成可执行方案", instruction="给出结构化方案（步骤、时间、资源、风险）。"),
            PlanItem(step_id="S3", title="输出最终清单", instruction="整理为用户可直接执行的清单与建议。"),
        ]
        return PlanResult(goal=user_text, steps=steps, raw_text="fallback_plan")

    def plan_task(self, user_text: str, history: List[Dict[str, str]]) -> PlanResult:
        system_prompt = (
            "你是 planner_agent，负责把用户需求拆成执行步骤。"
            "输出 JSON，格式如下："
            "{\"goal\":\"...\",\"steps\":[{\"title\":\"...\",\"instruction\":\"...\"}]}"
            "步骤数 2-6，instruction 要具体可执行。"
            "不要输出额外解释文字。"
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_text})

        try:
            raw = self._invoke(messages)
            payload = self._extract_json(raw)
            goal = str(payload.get("goal") or user_text).strip()
            raw_steps = payload.get("steps") or []

            steps: List[PlanItem] = []
            for i, item in enumerate(raw_steps[: self.max_steps], start=1):
                title = str(item.get("title") or f"步骤{i}").strip()
                instruction = str(item.get("instruction") or title).strip()
                steps.append(PlanItem(step_id=f"S{i}", title=title, instruction=instruction))

            if not steps:
                fallback = self._fallback_plan(user_text)
                fallback.error = error_payload(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="Planner returned empty steps, fallback applied",
                    retryable=True,
                )
                return fallback

            return PlanResult(goal=goal, steps=steps, raw_text=raw)
        except Exception as e:
            logger.warning(f"Planner fallback applied: {e}")
            fallback = self._fallback_plan(user_text)
            fallback.error = error_payload(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Planner failed: {e}",
                retryable=True,
            )
            return fallback
