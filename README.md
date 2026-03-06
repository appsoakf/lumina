# Lumina

Lumina 是一个基于 WebSocket 的宠物对话与任务执行服务。  
它把多 Agent 编排（Chat/Planner/Executor/Critic）、工具调用、记忆模块、可选翻译与可选 TTS 串成一条实时交互链路，适合做「可聊天、可执行任务、可持续记忆」的角色化助手。

## Lumina 能做什么

- 实时对话：WebSocket 输入一句话，返回情绪 + 文本（可选返回音频流）。
- 自动路由：自动判断当前请求是闲聊（chat）还是任务（task）。
- 任务执行：task 模式下通过 planner/executor/critic 协同完成多步骤任务。
- 工具增强：内置 `web_search`、`read_file/read_pdf/write_markdown`
- 记忆增强：保留会话历史并沉淀长期记忆，用于后续上下文增强。
- 可恢复任务：任务进入 `waiting_user_input` 后，用户补充信息即可续跑。

## 快速上手部署

### 1. 环境准备

- Python 3.10+
- 推荐使用 Conda（项目默认环境名：`agent`）

```powershell
conda activate agent
pip install -r requirements.txt
```

如果你没有 `agent` 环境，可先创建：

```powershell
conda create -n agent python=3.10 -y
conda activate agent
pip install -r requirements.txt
```

### 2. 配置 `config.json`

至少确认以下配置可用：

- `llm.chat_model` / `llm.chat_api_url`
- `llm.translate_model` / `llm.translate_api_url`
- `service.server_address` / `service.server_port`
- `service.enable_translation` / `service.enable_tts`

推荐把 API Key 放环境变量（优先级高于配置文件）：

```powershell
$env:LUMINA_API_KEY="your_api_key"
```

可选运行目录覆盖：

```powershell
$env:LUMINA_RUNTIME_DIR="D:\\lumina\\runtime"
$env:LUMINA_BACKUP_DIR="D:\\lumina\\backups"
```

### 3. 启动服务

```powershell
python main.py
```

默认监听：`ws://0.0.0.0:8080/ws`

### 4. 最小连通验证

在浏览器控制台（F12）执行：

```javascript
const ws = new WebSocket("ws://127.0.0.1:8080/ws");
ws.onopen = () => ws.send(JSON.stringify({ content: "你好，介绍一下你自己" }));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

## WebSocket 协议速览

客户端发送：

```json
{"content":"用户输入文本"}
```

服务端常见消息类型：

- `emotion_text`：文本回复与情绪信息
- `audio_chunk`：Base64 音频分片（开启 TTS 时可能出现）
- `audio_done`：音频阶段结束
- `done`：本轮结束
- `error`：错误信息（包含 `code/message/retryable/details`）

## 目录结构（核心）

```text
main.py                 # 入口，启动 PET websocket 服务
core/
  agentic/              # chat/planner/executor/critic agents
  orchestrator/         # 路由与任务编排主入口
  tools/                # 工具注册与工具实现
  memory/               # 记忆服务与 memory module engine
  tasks/                # 任务状态机与持久化
  llm/                  # LLM client 与调用封装
  paths.py              # runtime 路径统一解析
service/pet/            # websocket handler 与 pipeline
scripts/                # 健康检查、E2E、清理与指标脚本
tests/                  # 单元与回归测试
runtime/                # 会话/任务/trace/notes/memory 运行产物
```

## 常用命令

```powershell
# 健康检查（推荐）
python scripts/health_check.py --skip-network

# 全量单测
python -m unittest discover -s tests -v

# 关键回归
python -m unittest tests.test_task_flow_regression tests.test_orchestrator_langgraph tests.test_ws_contract tests.test_translate_engine -v

# E2E 清洁启动（重建 runtime/e2e/current）
python scripts/run_pet_e2e.py
```

## 运行与排障建议

- 首次部署建议先关闭 `enable_translation` 和 `enable_tts`，先打通文本链路。
- 若任务多轮追问未收敛，检查 `task_flow.max_clarify_rounds` 与任务输入质量。
- 工具调用相关问题优先看日志与 trace：
  - `logs/`（结构化事件）
  - `runtime/traces/`（会话级 JSONL 轨迹）
