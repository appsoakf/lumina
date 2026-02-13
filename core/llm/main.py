import json
import os
import logging
from openai import OpenAI
from typing import Generator

logger = logging.getLogger(__name__)
class LLMEngine:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.client = OpenAI(api_key=config["api_key"], base_url=config["api_url"])
        self.emotion_enabled = config.get("emotion_enabled", "False") == "True"
        if self.emotion_enabled:
            self.prompt = config["emotion_prompt"]
        else:
            self.prompt = config["prompt"]
        self.model = config["model"]
        self.translate_prompt = config.get("translate_prompt", "将以下中文翻译成日文，只输出日文，不要任何解释：")
        # 与用户对话的消息列表
        self.conversation_history = []

    def _build_messages(self, *, mode: str, stream: bool, user_text: str = None) -> list[dict]:
        if mode == "chat":
            system_content = self.prompt + ("/no_think" if stream else "")
            messages = [{"role": "system", "content": system_content}]
            messages.extend(self.conversation_history)
            return messages
        elif mode == "translate":
            system_content = self.translate_prompt + ("/no_think" if stream else "")
            return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_text},
            ]

    def _call_api(self, messages: list[dict], *, stream: bool):
        return self.client.chat.completions.create(
            model=self.model, messages=messages, stream=stream
        )

    def generate_by_api(self, msg) -> str:
        try:
            self.conversation_history.append({"role": "user", "content": msg})
            messages = self._build_messages(mode="chat", stream=False)
            completion = self._call_api(messages, stream=False)
            self.conversation_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"本地llm模块generate出错，错误详情：{e}"

    def generate_by_api_stream(self, msg: str, request_id: str = None) -> Generator:
        try:
            self.conversation_history.append({"role": "user", "content": msg})
            messages = self._build_messages(mode="chat", stream=True)
            for chunk in self._call_api(messages, stream=True):
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            req_id = request_id or "unknown"
            logger.error(f"[{req_id}] LLM stream error: {e}")
            yield f"LLM生成出错：{e}"

    def append_history(self, role: str, msg: str):
        self.conversation_history.append({"role": role, "content": msg})

    def translate_stream(self, text: str, request_id: str = None) -> Generator:
        """将中文流式翻译为日文，yields chunks。不影响对话历史。"""
        try:
            messages = self._build_messages(mode="translate", stream=True, user_text=text)
            for chunk in self._call_api(messages, stream=True):
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"[{request_id or 'unknown'}] Translation stream error: {e}")

    def translate(self, text: str) -> str:
        """将中文翻译为日文，不影响对话历史。"""
        try:
            messages = self._build_messages(mode="translate", stream=False, user_text=text)
            completion = self._call_api(messages, stream=False)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ""
