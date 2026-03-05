import json
import logging
import re
from typing import Dict, List

from core.agentic.base import BaseLLMAgent
from core.config import load_app_config
from core.protocols import RoutingIntent
from core.utils import log_exception

logger = logging.getLogger(__name__)


class ChatAgent(BaseLLMAgent):
    """Resident chat agent: user communication + final response composition."""

    TASK_KEYWORDS = {
        "规划", "计划", "安排", "整理", "总结", "生成", "写", "记录", "查询", "查", "执行", "帮我",
        "提醒", "导出", "保存", "创建", "文件", "清单", "攻略", "行程", "预算",
    }

    def __init__(self):
        super().__init__(
            missing_key_message="Missing LLM API key for chat agent",
            missing_key_field="chat_api_key",
            default_temperature=0.5,
        )
        llm_cfg = load_app_config().llm
        self.chat_prompt = llm_cfg.chat_prompt

    def _invoke(self, messages: List[Dict[str, str]], temperature: float = 0.5) -> str:
        completion = self.invoke_chat(messages, temperature=temperature)
        return (completion.choices[0].message.content or "").strip()

    def _ensure_emotion_format(self, text: str) -> str:
        text = text.strip()
        if not text:
            return '{"emotion": "平静", "intensity": 1}\n我在这里，主人。'

        lines = text.splitlines()
        first_line = lines[0].strip() if lines else ""

        try:
            payload = json.loads(first_line)
            if isinstance(payload, dict) and payload.get("emotion") and payload.get("intensity") is not None:
                return text
        except Exception:
            pass

        body = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        return '{"emotion": "平静", "intensity": 1}\n' + body

    def _keyword_intent(self, user_text: str) -> RoutingIntent:
        if any(keyword in user_text for keyword in self.TASK_KEYWORDS):
            return RoutingIntent.TASK
        return RoutingIntent.CHAT

    def classify_intent(self, user_text: str, history: List[Dict[str, str]]) -> RoutingIntent:
        keyword_intent = self._keyword_intent(user_text)

        router_prompt = (
            "你是路由器，只判断当前用户消息是否需要调用任务执行代理。"
            "如果只是闲聊、情感回应，输出 chat；"
            "如果需要规划、整理、查询、记录、生成内容、执行任务，输出 task。"
            "注意只输出一个词：chat 或 task。"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": router_prompt}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_text})

        try:
            decision = self._invoke(messages, temperature=0.0).lower().strip()
            if "task" in decision:
                return RoutingIntent.TASK
            if "chat" in decision:
                return RoutingIntent.CHAT
            return keyword_intent
        except Exception:
            log_exception(
                logger,
                "chat.intent.classify.error",
                "意图识别失败，降级为关键词规则",
                component="agent",
                fallback="keyword_rule",
            )
            return keyword_intent

    def reply_chat(self, user_text: str, history: List[Dict[str, str]]) -> str:
        system_prompt = (
            self.chat_prompt
            + "\n\n你是常驻 chat_agent，负责和用户交流并提供情绪价值。"
            + "你的最终输出必须保持：第一行情绪JSON，第二行开始为正文。"
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_text})
        reply = self._invoke(messages, temperature=0.7)
        return self._ensure_emotion_format(reply)

    def reply_with_task_result(
        self,
        user_text: str,
        executor_output: str,
        history: List[Dict[str, str]],
    ) -> str:
        system_prompt = (
            self.chat_prompt
            + "\n\n你是常驻 chat_agent。"
            + "现在你将根据 executor_agent 的任务执行结果，给用户一个自然、温暖、明确的最终回复。"
            + "最终输出必须保持：第一行情绪JSON，第二行开始为正文。"
        )
        payload = {
            "user_request": user_text,
            "executor_output": executor_output,
        }
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        reply = self._invoke(messages, temperature=0.6)
        return self._ensure_emotion_format(reply)
