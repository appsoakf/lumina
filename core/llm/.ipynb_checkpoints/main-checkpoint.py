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
        self.prompt = config["prompt"]
        self.model = config["model"]
        self.translate_prompt = config.get("translate_prompt", "将以下中文翻译成日文，只输出日文，不要任何解释：")
        # 与用户对话的消息列表
        self.conversation_history = []
    
    def generate_by_api(self, msg) -> str:
        try:
            self.conversation_history.append({"role":"user", "content": msg})
            messages = [{"role": "system", "content": self.prompt}]

            # 将当前的prompt + 之前所有的对话作为llm的输入
            messages.extend(self.conversation_history)
            completion = self.client.chat.completions.create(model=self.model, messages=messages)
            self.conversation_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"本地llm模块generate出错，错误详情：{e}"
        
    def generate_by_api_stream(self, msg: str, request_id: str = None) -> Generator:
        try:
            self.conversation_history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": self.prompt}]
            messages.extend(self.conversation_history)

            stream_response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )

            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            req_id = request_id or "unknown"
            logger.error(f"[{req_id}] LLM stream error: {e}")
            yield f"LLM生成出错：{e}"


    def append_history(self, role: str, msg: str):
        self.conversation_history.append({"role": role, "content": msg})

    def translate(self, text: str) -> str:
        """将中文翻译为日文，不影响对话历史。"""
        try:
            messages = [
                {"role": "system", "content": self.translate_prompt},
                {"role": "user", "content": text}
            ]
            completion = self.client.chat.completions.create(model=self.model, messages=messages)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ""
