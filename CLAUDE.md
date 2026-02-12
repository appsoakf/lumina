# Lumina

AI 虚拟角色对话系统，支持文本对话和语音合成。

## 项目结构

```
lumina/
├── main.py                 # 入口文件
├── core/                   # 核心模块
│   ├── llm/               # 大语言模型引擎
│   │   ├── main.py        # LLMEngine 类
│   │   └── config.json    # LLM 配置（API地址、模型、prompt）
│   └── tts/               # 语音合成引擎
│       ├── main.py        # TTSEngine 类
│       └── config.json    # TTS 配置（GPT-SoVITS地址、参考音频）
├── service/               # 服务层
│   └── webchat/           # Web 聊天服务
│       ├── main.py        # FastAPI 应用
│       ├── config.json    # 服务配置（端口、角色信息）
│       └── static/        # 前端资源
│           ├── index.html
│           ├── css/
│           └── js/app.js
└── data/
    └── assets/image/      # 角色头像等资源
```

## 核心模块

### LLMEngine (`core/llm/main.py`)

文本生成引擎，使用 OpenAI 兼容 API。

```python
llm = LLMEngine()
# 非流式
response = llm.generate_by_api("你好")
# 流式
for chunk in llm.generate_by_api_stream("你好"):
    print(chunk, end="")
# 翻译（独立调用，不影响对话历史）
ja_text = llm.translate("你好")
```

### TTSEngine (`core/tts/main.py`)

语音合成引擎，调用 GPT-SoVITS API。

```python
tts = TTSEngine()
req = TTSRequest(text="你好", streaming=False)
result = await tts.synthesize(req)
# result: {"success": True, "audio_bytes": bytes}
```

## 数据流

```
用户输入 → WebSocket → LLM流式生成(中文) → 实时文本显示
                                ↓ (流结束后)
                         LLM翻译(中→日) → TTS合成 → 音频播放
```

1. 前端通过 WebSocket 发送用户消息
2. 后端调用 LLM 流式生成纯中文回复，实时推送文本到前端
3. 流结束后，若 TTS 开启，调用 `llm.translate()` 将中文翻译为日文
4. 将日文发送给 TTS 合成音频，返回前端播放

## 运行

```bash
# 前置：启动 GPT-SoVITS 服务（默认 127.0.0.1:9880）
python main.py
# 访问 http://localhost:6006
```

## 配置

### LLM (`core/llm/config.json`)
- `api_url`: OpenAI 兼容 API 地址
- `api_key`: API 密钥
- `model`: 模型名称
- `prompt`: 聊天系统提示词（只输出中文）
- `translate_prompt`: 翻译提示词（中文→日文，供 TTS 使用）

### TTS (`core/tts/config.json`)
- `GPT-SoVITS_url`: TTS 服务地址
- `ref_path`: 参考音频路径
- `prompt_text`: 参考音频文本
- `prompt_lang`: 参考音频语言

### WebChat (`service/webchat/config.json`)
- `server_port`: 服务端口
- `character_info`: 角色名称配置

## 依赖

核心依赖：
- `fastapi`, `uvicorn`: Web 服务
- `openai`: LLM API 客户端
- `httpx`: 异步 HTTP 客户端
- `pydantic`: 数据验证
