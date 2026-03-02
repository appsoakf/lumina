import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.protocols import CriticResult, PlanItem, PlanResult


@dataclass
class TravelConstraints:
    destination: str = ""
    days: Optional[int] = None
    budget_cny: Optional[int] = None
    start_date: str = ""
    travelers: str = ""
    preferences: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "destination": self.destination,
            "days": self.days,
            "budget_cny": self.budget_cny,
            "start_date": self.start_date,
            "travelers": self.travelers,
            "preferences": self.preferences,
        }


class TravelWorkflow:
    """Phase 3 domain workflow: travel planning with constraints and quality checks."""

    DEST_KEYWORDS = ["去", "到", "旅游", "旅行", "行程", "攻略", "北京", "上海", "广州", "深圳"]

    def is_match(self, user_text: str) -> bool:
        text = user_text.strip()
        return any(k in text for k in self.DEST_KEYWORDS)

    def parse_constraints(self, user_text: str) -> TravelConstraints:
        text = user_text.strip()
        c = TravelConstraints()

        # destination
        m_dest = re.search(r"(?:去|到)([\u4e00-\u9fa5A-Za-z]{2,10})", text)
        if m_dest:
            c.destination = m_dest.group(1)
        elif "北京" in text:
            c.destination = "北京"

        # days
        m_days = re.search(r"(\d{1,2})\s*(?:天|日)", text)
        if m_days:
            c.days = int(m_days.group(1))

        # budget
        m_budget = re.search(r"(?:预算|花费|费用|控制在)?\s*(\d{3,6})\s*(?:元|块|人民币)", text)
        if m_budget:
            c.budget_cny = int(m_budget.group(1))

        # start date (simple)
        m_date = re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}月\d{1,2}日)", text)
        if m_date:
            c.start_date = m_date.group(1)

        # travelers
        m_people = re.search(r"(\d+)\s*(?:人|位)", text)
        if m_people:
            c.travelers = f"{m_people.group(1)}人"
        elif "情侣" in text:
            c.travelers = "情侣"
        elif "家庭" in text:
            c.travelers = "家庭"

        # preferences
        pref_tokens = []
        for token in ["美食", "拍照", "亲子", "历史", "博物馆", "轻松", "特种兵", "省钱", "高端", "夜景"]:
            if token in text:
                pref_tokens.append(token)
        c.preferences = "、".join(pref_tokens)

        return c

    def missing_required_fields(self, constraints: TravelConstraints) -> List[str]:
        missing = []
        if not constraints.destination:
            missing.append("destination")
        if constraints.days is None:
            missing.append("days")
        return missing

    def build_clarification_request(self, constraints: TravelConstraints, missing: List[str]) -> str:
        parts = []
        if "destination" in missing:
            parts.append("目的地")
        if "days" in missing:
            parts.append("出行天数")

        known = []
        if constraints.budget_cny:
            known.append(f"预算约{constraints.budget_cny}元")
        if constraints.start_date:
            known.append(f"出发时间{constraints.start_date}")
        if constraints.travelers:
            known.append(f"出行人数/类型{constraints.travelers}")

        known_text = f"已识别信息：{'; '.join(known)}。" if known else ""

        return (
            "为了给你生成可执行的详细旅游计划，我还需要补充信息："
            f"{', '.join(parts)}。"
            f"{known_text}"
            "请直接回复这些信息，我会继续生成完整行程。"
        )

    def build_plan(self, user_text: str, constraints: TravelConstraints) -> PlanResult:
        days_text = f"{constraints.days}天" if constraints.days else "待确认天数"
        budget_text = f"预算{constraints.budget_cny}元" if constraints.budget_cny else "预算待确认"
        dest_text = constraints.destination or "目的地待确认"

        steps = [
            PlanItem(
                step_id="S1",
                title="行程框架设计",
                instruction=f"围绕{dest_text}设计{days_text}的旅行主线，输出每日主题安排。",
            ),
            PlanItem(
                step_id="S2",
                title="交通与住宿建议",
                instruction=f"根据{days_text}给出交通方式与住宿分区建议，并说明理由。",
            ),
            PlanItem(
                step_id="S3",
                title="预算与费用分配",
                instruction=f"结合{budget_text}给出分项预算（交通/住宿/餐饮/门票/机动）。",
            ),
            PlanItem(
                step_id="S4",
                title="执行清单与风险提示",
                instruction="整理出出发前清单、每日注意事项、天气与高峰期风险规避建议。",
            ),
        ]

        return PlanResult(goal=user_text, steps=steps, raw_text="travel_workflow_template")

    def review(self, constraints: TravelConstraints, graph_dict: Dict[str, Any], critic: CriticResult) -> Dict[str, Any]:
        issues: List[str] = []
        suggestions: List[str] = []

        nodes = graph_dict.get("nodes", [])
        succeeded = [n for n in nodes if n.get("state") == "succeeded"]
        failed = [n for n in nodes if n.get("state") == "failed"]

        if len(succeeded) < 2:
            issues.append("有效步骤结果过少，行程细节不足。")
            suggestions.append("补充每日行程、交通与预算细节。")

        if failed:
            issues.append(f"存在失败步骤 {len(failed)} 个，结果完整性受影响。")
            suggestions.append("重试失败步骤，或补充用户约束后再次执行。")

        if constraints.days and constraints.days > 10:
            suggestions.append("天数较长，建议按前中后段分层规划，提高可执行性。")

        quality = "revise" if issues else "pass"

        # merge critic output
        all_issues = issues + critic.issues
        all_suggestions = suggestions + critic.suggestions
        if critic.quality == "revise" and "模型评审认为需要修订。" not in all_issues:
            all_issues.append("模型评审认为需要修订。")

        summary = critic.summary or ("结果可直接交付。" if quality == "pass" else "建议先修订后交付。")

        return {
            "quality": "revise" if all_issues else "pass",
            "issues": all_issues,
            "suggestions": all_suggestions,
            "summary": summary,
        }

    def compose_executor_output(
        self,
        constraints: TravelConstraints,
        graph_dict: Dict[str, Any],
        critic: CriticResult,
        workflow_review: Dict[str, Any],
    ) -> str:
        lines = ["旅游任务执行报告", ""]
        lines.append("约束信息:")
        lines.append(f"- 目的地: {constraints.destination or '待确认'}")
        lines.append(f"- 天数: {constraints.days if constraints.days is not None else '待确认'}")
        lines.append(f"- 预算: {constraints.budget_cny if constraints.budget_cny is not None else '待确认'}")
        lines.append(f"- 出发时间: {constraints.start_date or '待确认'}")
        lines.append(f"- 出行对象: {constraints.travelers or '待确认'}")
        lines.append(f"- 偏好: {constraints.preferences or '未指定'}")
        lines.append("")

        lines.append("步骤执行结果:")
        for n in graph_dict.get("nodes", []):
            state = n.get("state", "unknown")
            state_text = {
                "succeeded": "成功",
                "failed": "失败",
                "running": "执行中",
                "pending": "待执行",
            }.get(state, state)
            lines.append(f"- [{n.get('step_id')}] {n.get('title')} ({state_text})")
            if n.get("output_text"):
                lines.append(f"  结果: {n.get('output_text')}")
            if n.get("error"):
                lines.append(f"  错误: {n.get('error', {}).get('code')} | {n.get('error', {}).get('message')}")

        lines.append("")
        lines.append("评审结论:")
        lines.append(f"- critic: {critic.quality} | {critic.summary or '无'}")
        lines.append(f"- workflow: {workflow_review.get('quality')} | {workflow_review.get('summary')}")

        issues = workflow_review.get("issues") or []
        if issues:
            lines.append("主要问题:")
            for issue in issues:
                lines.append(f"- {issue}")

        suggestions = workflow_review.get("suggestions") or []
        if suggestions:
            lines.append("改进建议:")
            for s in suggestions:
                lines.append(f"- {s}")

        return "\n".join(lines)
