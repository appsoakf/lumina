import json
import os
from openai import OpenAI
from typing import Generator
class LLMEngine:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.client = OpenAI(api_key=config["api_key"], base_url=config["api_url"])
        self.prompt = config["prompt"]
        self.model = config["model"]
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
        
    def generate_by_api_stream(self, msg) -> Generator:
        try:
            self.conversation_history.append({"role":"user", "content": msg})
            messages = [{"role": "system", "content": self.prompt}]
            # 将当前的prompt + 之前所有的对话作为llm的输入
            messages.extend(self.conversation_history)
            stream_response = self.client.chat.completions.create(model=self.model, messages=messages, stream=True)

            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    delta = chunk.choices[0].delta.content
                    yield delta

        except Exception as e:
            yield f"本地llm模块generate_stream出错，错误详情：{e}"


    def append_history(self, role: str, msg: str):
        self.conversation_history.append({"role":role, "content": msg})
    # def generate_by_localllm(self, prompt):
