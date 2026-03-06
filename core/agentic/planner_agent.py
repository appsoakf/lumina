import logging
from typing import Dict, List

from core.agentic.base import BaseLLMAgent
from core.agentic.json_mixin import JSONParseMixin
from core.utils import log_exception
from core.utils.errors import ErrorCode, error_payload
from core.protocols import PlanItem, PlanResult

logger = logging.getLogger(__name__)


class PlannerAgent(BaseLLMAgent, JSONParseMixin):
    """Planner agent: decomposes user request into executable steps."""

    def __init__(self, max_steps: int = 5):
        super().__init__(
            missing_key_message="Missing LLM API key for planner agent",
            missing_key_field="chat_api_key",
            default_temperature=0.1,
        )
        self.max_steps = max_steps

    def _invoke(self, messages: List[Dict[str, str]]) -> str:
        completion = self.invoke_chat(messages, temperature=0.1)
        return (completion.choices[0].message.content or "").strip()

    def _extract_json(self, text: str) -> Dict:
        return self.parse_json_object(text, allow_brace_extract=True)

    def _fallback_plan(self, user_text: str) -> PlanResult:
        steps = [
            PlanItem(
                step_id="S1",
                title="明确目标与关键约束",
                instruction=f"提炼用户任务目标、必要约束与可执行范围：{user_text}",
                depends_on=[],
            ),
            PlanItem(
                step_id="S2",
                title="交付可执行结果",
                instruction="基于约束直接给出可执行方案与最终清单，包含关键风险与下一步建议。",
                depends_on=["S1"],
            ),
        ]
        return PlanResult(
            goal=user_text,
            steps=steps,
            raw_text="fallback_plan",
            graph_policy={"max_parallelism": 2, "fail_fast": True},
        )

    def plan_task(self, user_text: str, history: List[Dict[str, str]]) -> PlanResult:
        system_prompt = """你是 planner_agent。
你的唯一职责：把用户需求拆解为可执行的任务步骤，供 executor_agent 按步骤执行。
你不执行任务、不评审任务、不和用户闲聊，只输出任务计划 JSON。

【输出规则（必须严格遵守）】
1. 只输出一个 JSON 对象。
2. 不要输出任何 JSON 以外的文字。
3. 不要使用 markdown 代码块，不要加注释。

【JSON 顶层结构】
{
  "goal": "string",
  "graph_policy": {
    "max_parallelism": 2,
    "fail_fast": true
  },
  "steps": [
    {
      "title": "string",
      "instruction": "string",
      "depends_on": ["S1"],
      "input_bindings": [
        {"from": "S1.output_text", "to": "context"}
      ]
    }
  ]
}

【字段要求】
1. goal
- 简洁复述用户最终目标，不要改写用户核心意图。

2. graph_policy
- max_parallelism: 正整数，建议 1-2。
- fail_fast: 布尔值。默认 true；当步骤相互独立且允许部分完成时可设为 false。

3. steps
- 步骤数必须为 2-5。
- 每个步骤必须包含 title 和 instruction。
- title: 简洁、可读，描述该步骤目标。
- instruction: 必须具体可执行，写清动作与产出。
- depends_on: 仅允许引用之前步骤（如 S1、S2），不能引用未来步骤、不能自依赖、不能形成环。
- input_bindings: 可为空；用于把上游结果映射到当前步骤输入。
  - from 支持：
    - "Sx.output_text"（上游步骤输出）
    - "Sx.error"（上游错误信息）
    - "$const:<json_or_text>"（常量）
  - to 为当前步骤输入变量名。

【规划质量要求】
1. 步骤要原子化：每步只做一件主要事情。
2. 步骤要可验证：执行后能判断完成/失败。
3. 依赖要最小化：仅在确有数据或顺序依赖时设置 depends_on。
4. 如果用户信息不足，第一步应先“澄清假设/提取约束”，后续步骤基于该结果继续。
5. 禁止把“搜索 -> 筛选 -> 输出”拆成多个纯文本改写步骤；若能一次产出结果，必须合并为同一步。
6. 优先给出短链路计划，默认 2 步，最多 5 步。

【输出示例】
{
  "goal": "整理北京三日游方案并给出预算建议",
  "graph_policy": {
    "max_parallelism": 2,
    "fail_fast": false
  },
  "steps": [
    {
      "title": "提取需求约束",
      "instruction": "整理用户已给出的预算、偏好与时间约束，输出结构化约束清单。",
      "depends_on": [],
      "input_bindings": []
    },
    {
      "title": "生成候选行程",
      "instruction": "基于约束清单生成 2 套三日行程候选，每套包含每日安排与交通建议。",
      "depends_on": ["S1"],
      "input_bindings": [
        {"from": "S1.output_text", "to": "constraints"}
      ]
    },
    {
      "title": "形成最终方案",
      "instruction": "对候选行程进行预算核算与取舍，输出推荐方案、预算明细和执行清单。",
      "depends_on": ["S2"],
      "input_bindings": [
        {"from": "S2.output_text", "to": "itinerary_candidates"}
      ]
    }
  ]
}
"""
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_text})

        try:
            raw = self._invoke(messages)
            payload = self._extract_json(raw)
            goal = str(payload.get("goal") or user_text).strip()
            raw_steps = payload.get("steps") or []
            graph_policy = payload.get("graph_policy") if isinstance(payload.get("graph_policy"), dict) else {}

            steps: List[PlanItem] = []
            for i, item in enumerate(raw_steps[: self.max_steps], start=1):
                title = str(item.get("title") or f"步骤{i}").strip()
                instruction = str(item.get("instruction") or title).strip()
                step_id = f"S{i}"
                depends_raw = item.get("depends_on")
                depends_on = []
                if isinstance(depends_raw, list):
                    depends_on = [str(v).strip() for v in depends_raw if str(v).strip()]
                allowed_dep_ids = {f"S{j}" for j in range(1, i)}
                depends_on = [dep for dep in depends_on if dep in allowed_dep_ids]

                bindings_raw = item.get("input_bindings")
                input_bindings = []
                if isinstance(bindings_raw, list):
                    for binding in bindings_raw:
                        if not isinstance(binding, dict):
                            continue
                        source = str(binding.get("from") or "").strip()
                        target = str(binding.get("to") or "").strip()
                        if not source or not target:
                            continue
                        input_bindings.append({"from": source, "to": target})

                steps.append(
                    PlanItem(
                        step_id=step_id,
                        title=title,
                        instruction=instruction,
                        depends_on=depends_on,
                        input_bindings=input_bindings,
                    )
                )

            if not steps:
                fallback = self._fallback_plan(user_text)
                fallback.error = error_payload(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="Planner returned empty steps, fallback applied",
                    retryable=True,
                )
                return fallback

            return PlanResult(goal=goal, steps=steps, raw_text=raw, graph_policy=graph_policy)
        except Exception as e:
            log_exception(
                logger,
                "planner.plan.error",
                "Planner 执行失败，使用兜底计划",
                component="agent",
                fallback="default_plan",
            )
            fallback = self._fallback_plan(user_text)
            fallback.error = error_payload(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Planner failed: {e}",
                retryable=True,
            )
            return fallback
