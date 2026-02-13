import logging
import os
import asyncio
import re
import uvicorn
import base64
import uuid
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
from core.llm.main import LLMEngine
from core.tts.main import TTSEngine, TTSRequest
from core.emotion.main import EmotionEngine

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 禁用 uvicorn access log
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 对话轮数计数器
conversation_round = 0

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
username = config["character_info"]["username"]
mate_name = config["character_info"]["mate_name"]
server_ip = config["server_address"]
server_port = config["server_port"]
live2d_port = config["live2d_port"]
mmd_port = config["mmd_port"]
vrm_port = config["vrm_port"]


app = FastAPI()

@app.on_event("startup")
async def warmup_tts():
    """服务启动时预热 TTS，避免首次请求冷启动延迟。"""
    logger.info("[WARMUP] Sending TTS warmup request...")
    try:
        req = TTSRequest(text="你好")
        result = await tts.synthesize(req, request_id="warmup")
        if result.get("success"):
            logger.info("[WARMUP] TTS warmup completed successfully")
        else:
            logger.warning(f"[WARMUP] TTS warmup failed: {result.get('error')}")
    except Exception as e:
        logger.warning(f"[WARMUP] TTS warmup exception: {e}")

static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/data/assets/image", StaticFiles(directory="data/assets/image"))

llm = LLMEngine()
tts = TTSEngine()
emotion_engine = EmotionEngine()

web_chat_history = []

class ConnectionManager:
    def __init__(self):
        self.connection: WebSocket = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connection = websocket
        await websocket.send_json({"type": "history", "data": web_chat_history})

    def disconnect(self):
        self.connection = None

    async def send(self, message: dict):
        if self.connection:
            await self.connection.send_json(message)

manager = ConnectionManager()


@app.get("/")
async def get():
    index_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    print("[get html]")
    return FileResponse(index_path)


@app.get("/api/config")
async def get_config():
    return {
        "server_ip": server_ip,
        "live2d_port": live2d_port,
        "mmd_port": mmd_port,
        "vrm_port": vrm_port,
        "username": username,
        "mate_name": mate_name
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action == "send":
                user_text = data.get("text", "").strip()
                if not user_text:
                    continue
                tts_enabled = data.get("tts_enabled", True)  # 新增：默认开启
                tts_mode = data.get("tts_mode", 1)
                new_msg = {"sender": username, "text": user_text}
                web_chat_history.append(new_msg)
                await manager.send({"type": "message", "data": new_msg})
                await manager.send({"type": "processing"})
                asyncio.create_task(handle_bot_reply(user_text, tts_enabled, tts_mode))
            elif action == "clear":
                web_chat_history.clear()
                await manager.send({"type": "clear"})
    except WebSocketDisconnect:
        manager.disconnect()


_SENTINEL = object()

async def iterate_in_thread(sync_iterable):
    """将同步可迭代对象的 next() 调用放到线程池执行，避免阻塞事件循环。"""
    loop = asyncio.get_running_loop()
    it = iter(sync_iterable)

    def _next():
        try:
            return next(it)
        except StopIteration:
            return _SENTINEL

    while True:
        value = await loop.run_in_executor(None, _next)
        if value is _SENTINEL:
            break
        yield value



SENTENCE_DELIMITERS = re.compile(r'(?<=[。！？；\n])')

def split_sentences(text: str) -> tuple[list[str], str]:
    """按中文句末标点切分，返回 (完整句子列表, 剩余buffer)。"""
    parts = SENTENCE_DELIMITERS.split(text)
    if len(parts) <= 1:
        return [], text
    sentences = [p for p in parts[:-1] if p.strip()]
    return sentences, parts[-1]


async def _stream_llm(user_text: str, request_id: str, round_num: int,
                      start_time: float, on_chunk=None, silent=False):
    """LLM 流式生成阶段，返回 full_reply。
    on_chunk: 可选回调，v1 模式用于实时分句入队。
    silent: True 时不推送 stream_chunk/stream_done 到前端（v2+emotion 用）。"""
    full_reply = ""

    async for chunk in iterate_in_thread(
        llm.generate_by_api_stream(user_text, request_id=request_id)
    ):
        if not chunk:
            continue
        full_reply += chunk
        if not silent:
            await manager.send({"type": "stream_chunk", "data": chunk})
        if on_chunk:
            await on_chunk(chunk)

    if not silent:
        await manager.send({"type": "stream_done"})

    web_chat_history.append({"sender": mate_name, "text": full_reply})
    llm.append_history("assistant", full_reply)

    return full_reply


async def _translate_and_enqueue(full_reply: str, request_id: str, round_num: int,
                                 tts_queue: asyncio.Queue):
    """流式翻译中文→日文，分句后入队，最后发送哨兵。"""
    ja_buffer = ""
    async for ja_chunk in iterate_in_thread(
        llm.translate_stream(full_reply, request_id=request_id)
    ):
        if not ja_chunk:
            continue
        ja_buffer += ja_chunk
        sentences, ja_buffer = split_sentences(ja_buffer)
        for s in sentences:
            await tts_queue.put(s)

    if ja_buffer.strip():
        await tts_queue.put(ja_buffer)

    await tts_queue.put(None)  # 哨兵



async def _tts_consumer(queue: asyncio.Queue, request_id: str, mode: int,
                        llm_done_time: float, round_num: int,
                        start_time: float,
                        ref_audio_path: str = None, prompt_text: str = None):
    """统一 TTS consumer。
    mode=1: LLM 边生成边分句(中文) → 每句独立翻译(中→日) → 流式 TTS → 按 RIFF 头切分发送音频 chunk
            特点：LLM 生成与翻译+TTS 并行，首段音频延迟最低，但每句需独立翻译调用
    mode=2: LLM 生成完毕后，整段中文流式翻译(中→日) → 日文分句入队 → 流式 TTS → 按 RIFF 头切分发送
            特点：仅 1 次翻译调用，翻译与 TTS 通过 Queue 解耦并行，首段音频需等 LLM 生成完毕
    """
    loop = asyncio.get_running_loop()
    index = 0
    first_audio_sent = False

    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        try:
            tts_start = time.time()

            if mode == 1:
                # v1: 从 queue 取出中文句子 → 翻译为日文 → 流式 TTS → 按 RIFF 头切分 → 逐段发送音频 chunk
                translate_start = time.time()
                ja_text = await loop.run_in_executor(None, llm.translate, sentence)
                if not ja_text:
                    continue
                translate_end = time.time()

                tts_req = TTSRequest(text=ja_text)
                result = await tts.synthesize_streaming(tts_req, request_id=request_id)
                if result.get("success"):
                    audio_stream = result.get("audio_stream")
                    wav_buf = bytearray()
                    async for audio_chunk in audio_stream:
                        if not audio_chunk:
                            continue
                        wav_buf.extend(audio_chunk)
                        # GPT-SoVITS 流式返回多段拼接 WAV，按 RIFF 头切分
                        while True:
                            riff_pos = wav_buf.find(b'RIFF', 4) if len(wav_buf) > 4 else -1
                            if riff_pos == -1:
                                break
                            complete_wav = bytes(wav_buf[:riff_pos])
                            wav_buf = wav_buf[riff_pos:]
                            audio_b64 = base64.b64encode(complete_wav).decode('utf-8')
                            await manager.send({"type": "audio_chunk", "data": audio_b64, "index": index})
                            if not first_audio_sent:
                                now = time.time()
                                first_sentence_ready = tts_start - start_time
                                translate_latency = translate_end - translate_start
                                first_tts_latency = now - translate_end
                                request_to_first_audio = now - start_time
                                logger.info(
                                    f"[Round {round_num}][{request_id}] First audio sent"
                                    f" | request_to_first_audio={request_to_first_audio:.3f}s"
                                    f" (first_sentence_ready={first_sentence_ready:.3f}s"
                                    f" + translate={translate_latency:.3f}s"
                                    f" + tts={first_tts_latency:.3f}s)"
                                )
                                first_audio_sent = True
                    # flush remaining buffer as the last WAV segment
                    if wav_buf:
                        audio_b64 = base64.b64encode(bytes(wav_buf)).decode('utf-8')
                        await manager.send({"type": "audio_chunk", "data": audio_b64, "index": index})
                        if not first_audio_sent:
                            now = time.time()
                            first_sentence_ready = tts_start - start_time
                            translate_latency = translate_end - translate_start
                            first_tts_latency = now - translate_end
                            request_to_first_audio = now - start_time
                            logger.info(
                                f"[Round {round_num}][{request_id}] First audio sent"
                                f" | request_to_first_audio={request_to_first_audio:.3f}s"
                                f" (first_sentence_ready={first_sentence_ready:.3f}s"
                                f" + translate={translate_latency:.3f}s"
                                f" + tts={first_tts_latency:.3f}s)"
                            )
                            first_audio_sent = True
                    index += 1
                else:
                    logger.error(f"[Round {round_num}][{request_id}] TTS streaming failed: {result.get('error')}")

            elif mode == 2:
                # v2: 从 queue 取出日文句子 → 流式 TTS → 按 RIFF 头切分 → 逐段发送音频 chunk
                tts_req = TTSRequest(text=sentence, ref_audio_path=ref_audio_path, prompt_text=prompt_text)
                result = await tts.synthesize_streaming(tts_req, request_id=request_id)
                if result.get("success"):
                    audio_stream = result.get("audio_stream")
                    wav_buf = bytearray()
                    async for audio_chunk in audio_stream:
                        if not audio_chunk:
                            continue
                        wav_buf.extend(audio_chunk)
                        # GPT-SoVITS 流式返回多段拼接 WAV，按 RIFF 头切分
                        while True:
                            riff_pos = wav_buf.find(b'RIFF', 4) if len(wav_buf) > 4 else -1
                            if riff_pos == -1:
                                break
                            complete_wav = bytes(wav_buf[:riff_pos])
                            wav_buf = wav_buf[riff_pos:]
                            audio_b64 = base64.b64encode(complete_wav).decode('utf-8')
                            await manager.send({"type": "audio_chunk", "data": audio_b64, "index": index})
                            if not first_audio_sent:
                                now = time.time()
                                llm_latency = llm_done_time - start_time
                                translate_to_first_sentence = tts_start - llm_done_time
                                first_tts_latency = now - tts_start
                                request_to_first_audio = now - start_time
                                logger.info(
                                    f"[Round {round_num}][{request_id}] First audio sent"
                                    f" | request_to_first_audio={request_to_first_audio:.3f}s"
                                    f" (llm={llm_latency:.3f}s"
                                    f" + translate_to_first_sentence={translate_to_first_sentence:.3f}s"
                                    f" + tts={first_tts_latency:.3f}s)"
                                )
                                first_audio_sent = True
                    # flush remaining buffer as the last WAV segment
                    if wav_buf:
                        audio_b64 = base64.b64encode(bytes(wav_buf)).decode('utf-8')
                        await manager.send({"type": "audio_chunk", "data": audio_b64, "index": index})
                        if not first_audio_sent:
                            now = time.time()
                            llm_latency = llm_done_time - start_time
                            translate_to_first_sentence = tts_start - llm_done_time
                            first_tts_latency = now - tts_start
                            request_to_first_audio = now - start_time
                            logger.info(
                                f"[Round {round_num}][{request_id}] First audio sent"
                                f" | request_to_first_audio={request_to_first_audio:.3f}s"
                                f" (llm={llm_latency:.3f}s"
                                f" + translate_to_first_sentence={translate_to_first_sentence:.3f}s"
                                f" + tts={first_tts_latency:.3f}s)"
                            )
                            first_audio_sent = True
                    index += 1
                else:
                    logger.error(f"[Round {round_num}][{request_id}] TTS streaming failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"[Round {round_num}][{request_id}] TTS consumer error on sentence {index}: {e}")



async def handle_bot_reply(user_text: str, tts_enabled: bool, tts_mode: int):
    """处理机器人回复的统一入口。
    tts_mode=1: LLM 边生成边分句，每句独立翻译+流式TTS（LLM 与 TTS 并行）
    tts_mode=2: LLM 生成完毕后，整段流式翻译+流式TTS（翻译与 TTS 通过 Queue 并行）
    """
    global conversation_round
    conversation_round += 1
    round_num = conversation_round

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[Round {round_num}][{request_id}] Start processing | mode=v{tts_mode} | tts_enabled={tts_enabled}")
    await manager.send({"type": "stream_start", "sender": mate_name})

    consumer_task: asyncio.Task | None = None

    try:
        # --- v1: LLM 生成与翻译+TTS 并行 ---
        # 流程: LLM 流式生成中文 → 实时按标点分句 → 每句入队 → consumer 逐句翻译(中→日) → 流式 TTS
        if tts_mode == 1 and tts_enabled and not llm.emotion_enabled:
            # LLM 生成期间实时分句入队（中文句子），consumer 同步消费
            tts_queue = asyncio.Queue()
            consumer_task = asyncio.create_task(
                _tts_consumer(tts_queue, request_id, mode=1,
                              llm_done_time=start_time, round_num=round_num,
                              start_time=start_time)
            )
            buffer = ""

            async def _on_chunk(chunk):
                nonlocal buffer
                buffer += chunk
                sentences, buffer = split_sentences(buffer)
                for s in sentences:
                    await tts_queue.put(s)

            full_reply = await _stream_llm(
                user_text, request_id, round_num, start_time, on_chunk=_on_chunk
            )

            # flush 剩余 buffer
            if buffer.strip():
                await tts_queue.put(buffer)
            await tts_queue.put(None)
            await consumer_task
            await manager.send({"type": "audio_done"})

        else:
            # --- v2 或 TTS 关闭: 先完成 LLM 生成，再处理 TTS ---
            # 流程: LLM 流式生成中文(完整) → 整段流式翻译(中→日) → 日文分句入队 → consumer 流式 TTS
            # v2 + emotion_enabled: 静默收集 LLM JSON 输出，解析后一次性发送纯文本
            use_emotion = llm.emotion_enabled
            full_reply = await _stream_llm(
                user_text, request_id, round_num, start_time,
                silent=use_emotion
            )
            llm_done_time = time.time()

            # 情感解析（仅 v2 + emotion_enabled）
            ref_audio_path = None
            prompt_text = None
            if use_emotion:
                emo, text_only = emotion_engine.parse_response(full_reply)
                ref_audio_path = emotion_engine.get_ref_audio(emo)
                prompt_text = emotion_engine.get_prompt_text(emo)
                logger.info(f"[Round {round_num}][{request_id}] Emotion: {emo} → {ref_audio_path}")
                # 更新历史和 LLM 记录为纯文本
                web_chat_history[-1]["text"] = text_only
                # 一次性发送纯文本到前端
                await manager.send({"type": "stream_chunk", "data": text_only})
                await manager.send({"type": "stream_done"})
                full_reply = text_only

            if tts_enabled and full_reply.strip():
                # v2: 整段翻译与流式 TTS 通过 Queue 解耦并行
                tts_queue = asyncio.Queue()
                consumer_task = asyncio.create_task(
                    _tts_consumer(tts_queue, request_id, mode=2,
                                  llm_done_time=llm_done_time,
                                  round_num=round_num,
                                  start_time=start_time,
                                  ref_audio_path=ref_audio_path,
                                  prompt_text=prompt_text)
                )
                await _translate_and_enqueue(full_reply, request_id, round_num, tts_queue)
                await consumer_task
                await manager.send({"type": "audio_done"})

    except Exception as e:
        logger.error(f"[Round {round_num}][{request_id}] stream error: {e}")
        await manager.send({"type": "stream_error", "text": f"\n（生成出错：{str(e)}）"})
        await manager.send({"type": "stream_done"})
        if consumer_task is not None:
            consumer_task.cancel()

    finally:
        await manager.send({"type": "done"})


def run_webchat():
    uvicorn.run(app, host="0.0.0.0", port=int(server_port))
