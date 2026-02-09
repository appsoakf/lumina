# 文本与语音同步流式输出

## 动机

原有 webchat 服务的文本和语音是**串行**的：LLM 流式输出全部文本完成后，才将完整文本发给 TTS 合成语音，导致用户看完所有文字后才能听到声音。目标是让语音随文本一起实时输出——边看文字边听语音。

### 原有流程的三个瓶颈

1. **串行处理**：LLM 输出完毕 → TTS 开始合成，两个阶段无重叠
2. **整段文本一次性合成**：将完整回复作为一个整体发给 TTS，合成耗时与文本长度成正比
3. **单个音频 blob 传输**：生成一个完整的 WAV 文件后一次性发送给前端

## 方案设计

### 核心思路：按句分片 + 并发 TTS + 前端音频队列

```
LLM 流式 chunk → 句子累积器 → 完整句子 → asyncio.create_task(TTS)
                                              ↓
                                    asyncio.Queue (有序)
                                              ↓
                                    audio_dispatcher 协程 → WebSocket audio_chunk
                                              ↓
                                    前端 audioQueue → 依次播放
```

### 新增 WebSocket 消息类型

| 类型 | 方向 | 内容 | 用途 |
|------|------|------|------|
| `audio_chunk` | Server→Client | `{type, data: base64, index: int}` | 单句音频片段 |
| `audio_done` | Server→Client | `{type}` | 所有音频片段发送完毕 |

## 实现细节

### 后端 `service/webchat/main.py`

**1. `split_sentences(text)` 工具函数**

- 用正则 `(?<=[。！？!?.])` 按句末标点切分
- 不按逗号切分，避免片段过短导致语音不自然
- 标点保留在句子末尾

**2. `synthesize_sentence(sentence)` 异步辅助函数**

- 对单个句子调用 `tts.synthesize(TTSRequest(text=..., streaming=False))`
- 返回 base64 编码的音频字符串，失败返回 None
- 使用非流式模式，避免 httpx 上下文管理器提前关闭的问题

**3. 重写 `handle_bot_reply_stream()`**

- `sentence_buffer` 累积 LLM chunk，检测到句子边界时立即创建 TTS 异步任务
- `asyncio.Queue` + `audio_dispatcher` 协程并发运行，按顺序 `await` TTS 任务并发送 `audio_chunk`
- LLM 流结束后 flush 剩余 buffer 作为最后一个句子
- 发送 `audio_done` 信号表示所有音频片段已发完
- TTS 任务在 LLM 流式过程中就开始执行，第一个句子的音频可能在 LLM 还在生成后续文本时就已发送给前端
- TTS 关闭时不创建 dispatcher，不发起任何 TTS 请求

### 前端 `service/webchat/static/js/app.js`

**1. 音频队列状态**

- `audioQueue = []` — 按序存放 base64 音频数据
- `isPlayingAudio = false` — 是否正在播放

**2. `playNextAudioChunk()` 函数**

- 从队列取出第一个音频，创建 `new Audio(data:audio/wav;base64,...)` 播放
- `audio.onended` 回调中递归调用自身，播放下一个
- 错误时跳过当前片段继续播放下一个

**3. WebSocket 消息处理**

- `audio_chunk`：push 到 `audioQueue`，若未在播放则调用 `playNextAudioChunk()`
- `audio_done`：无需特殊处理，队列自然排空
- `stream_start`：重置 `audioQueue` 和 `isPlayingAudio`，防止上一轮残留数据

### 未修改的文件

- `core/tts/main.py` — 直接复用现有非流式 `synthesize()` 方法
- `core/llm/main.py` — 同步 Generator 接口不变
- `static/index.html`、各 `config.json` — 无需改动

## 达到的效果

| 场景 | 改进前 | 改进后 |
|------|--------|--------|
| 多句回复 | 全部文本输出完毕后等待 TTS 合成整段音频，再一次性播放 | 第一句文本输出后即开始合成并播放，后续句子边生成边播放 |
| 单句回复 | 行为相同但多了一次完整合成的等待 | 产生 1 个 audio_chunk，文本和语音几乎同时出现 |
| 无标点长文本 | 与改进前行为一致 | LLM 流结束后 flush 为一个整句，退化为原有行为 |
| TTS 服务不可用 | 文本正常，音频报错 | 文本正常显示，音频静默跳过，无报错弹窗 |
| TTS 关闭 | 不发起 TTS 请求 | 不创建 dispatcher，不发起任何 TTS 请求 |
| 连续发送消息 | 可能出现音频残留 | `stream_start` 重置音频队列，无残留 |

### 性能提升

- **首句音频延迟**：从「等待全部文本 + 全段 TTS 合成」降低为「等待第一句文本 + 单句 TTS 合成」
- **并发利用**：多个句子的 TTS 合成可并行执行，总合成时间接近最长单句而非所有句子之和
- **用户感知**：从"先看完文字再听声音"变为"边看文字边听声音"
