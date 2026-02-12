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

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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
        req = TTSRequest(text="你好", streaming=False)
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
                if tts_mode == 3:
                    asyncio.create_task(handle_bot_reply_stream_v3(user_text, tts_enabled))
                elif tts_mode == 2:
                    asyncio.create_task(handle_bot_reply_stream_v2(user_text, tts_enabled))
                else:
                    asyncio.create_task(handle_bot_reply_stream(user_text, tts_enabled))
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


async def synthesize_sentence(sentence: str, request_id: str = None) -> str | None:
    """对单个句子调用 TTS，返回 base64 音频字符串，失败返回 None。"""
    try:
        tts_req = TTSRequest(text=sentence, streaming=False)
        result = await tts.synthesize(tts_req, request_id=request_id)
        if result.get("success"):
            audio_data = result.get("audio_bytes")
            if audio_data:
                return base64.b64encode(audio_data).decode('utf-8')
        else:
            logger.error(f"TTS synthesis failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"TTS exception: {e}")
    return None


SENTENCE_DELIMITERS = re.compile(r'(?<=[。！？；\n])')

def split_sentences(text: str) -> tuple[list[str], str]:
    """按中文句末标点切分，返回 (完整句子列表, 剩余buffer)。"""
    parts = SENTENCE_DELIMITERS.split(text)
    if len(parts) <= 1:
        return [], text
    sentences = [p for p in parts[:-1] if p.strip()]
    return sentences, parts[-1]


async def _tts_consumer(queue: asyncio.Queue, request_id: str):
    """从队列顺序取句子，翻译→TTS合成→发送音频。收到 None 哨兵时退出。"""
    loop = asyncio.get_running_loop()
    index = 0
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        try:
            logger.info(f"[{request_id}] Translating sentence {index}: {sentence[:30]}...")
            ja_text = await loop.run_in_executor(None, llm.translate, sentence)
            if ja_text:
                logger.info(f"[{request_id}] Sentence {index} translated: {ja_text[:30]}...")
                audio_base64 = await synthesize_sentence(ja_text, request_id=request_id)
                if audio_base64:
                    await manager.send({"type": "audio_chunk", "data": audio_base64, "index": index})
                    index += 1
        except Exception as e:
            logger.error(f"[{request_id}] TTS consumer error on sentence {index}: {e}")


async def handle_bot_reply_stream(user_text: str, tts_enabled: bool = True):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    full_reply = ""  # 完整中文回复
    buffer = ""      # 未切分的文本缓冲

    await manager.send({"type": "stream_start", "sender": mate_name})

    # 如果 TTS 开启，启动消费者协程
    tts_queue: asyncio.Queue | None = None
    consumer_task: asyncio.Task | None = None
    if tts_enabled:
        tts_queue = asyncio.Queue()
        consumer_task = asyncio.create_task(_tts_consumer(tts_queue, request_id))

    try:
        # 流式获取 LLM 中文回复
        async for chunk in iterate_in_thread(llm.generate_by_api_stream(user_text, request_id=request_id)):
            if not chunk:
                continue
            full_reply += chunk
            await manager.send({"type": "stream_chunk", "data": chunk})

            # 分句：将完整句子送入 TTS 队列
            if tts_queue is not None:
                buffer += chunk
                sentences, buffer = split_sentences(buffer)
                for s in sentences:
                    await tts_queue.put(s)

        await manager.send({"type": "stream_done"})
        llm_latency = time.time() - start_time
        logger.info(f"[{request_id}] LLM completed | latency={llm_latency:.3f}s | text={full_reply[:50]}...")

        # 保存聊天历史
        web_chat_history.append({"sender": mate_name, "text": full_reply})
        llm.append_history("assistant", full_reply)

        # flush 剩余 buffer 作为最后一句
        if tts_queue is not None:
            if buffer.strip():
                await tts_queue.put(buffer)
            await tts_queue.put(None)  # 哨兵，通知消费者退出
            await consumer_task
            await manager.send({"type": "audio_done"})

    except Exception as e:
        logger.error(f"[{request_id}] stream error: {e}")
        await manager.send({"type": "stream_error", "text": f"\n（生成出错：{str(e)}）"})
        await manager.send({"type": "stream_done"})
        if consumer_task is not None:
            consumer_task.cancel()

    finally:
        total_latency = time.time() - start_time
        logger.info(f"[{request_id}] Request completed | total_latency={total_latency:.3f}s | tts_enabled={tts_enabled}")
        await manager.send({"type": "done"})


async def handle_bot_reply_stream_v2(user_text: str, tts_enabled: bool = True):
    """方案二：整段流式翻译 + 日文分句 TTS。"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    full_reply = ""

    await manager.send({"type": "stream_start", "sender": mate_name})

    try:
        # Phase 1: LLM 流式生成中文
        async for chunk in iterate_in_thread(
            llm.generate_by_api_stream(user_text, request_id=request_id)
        ):
            if not chunk:
                continue
            full_reply += chunk
            await manager.send({"type": "stream_chunk", "data": chunk})

        await manager.send({"type": "stream_done"})
        llm_latency = time.time() - start_time
        logger.info(f"[{request_id}] LLM completed | latency={llm_latency:.3f}s")

        web_chat_history.append({"sender": mate_name, "text": full_reply})
        llm.append_history("assistant", full_reply)

        # Phase 2: 整段中文 → 流式翻译日文 → 按句切分 → 逐句 TTS
        if tts_enabled and full_reply.strip():
            ja_buffer = ""
            audio_index = 0

            async for ja_chunk in iterate_in_thread(
                llm.translate_stream(full_reply, request_id=request_id)
            ):
                if not ja_chunk:
                    continue
                ja_buffer += ja_chunk
                sentences, ja_buffer = split_sentences(ja_buffer)
                for s in sentences:
                    logger.info(f"[{request_id}] JA sentence {audio_index}: {s[:30]}...")
                    audio_base64 = await synthesize_sentence(s, request_id=request_id)
                    if audio_base64:
                        await manager.send({
                            "type": "audio_chunk", "data": audio_base64,
                            "index": audio_index
                        })
                        audio_index += 1

            # flush 剩余 buffer
            if ja_buffer.strip():
                audio_base64 = await synthesize_sentence(ja_buffer, request_id=request_id)
                if audio_base64:
                    await manager.send({
                        "type": "audio_chunk", "data": audio_base64,
                        "index": audio_index
                    })

            await manager.send({"type": "audio_done"})

    except Exception as e:
        logger.error(f"[{request_id}] stream error: {e}")
        await manager.send({"type": "stream_error", "text": f"\n（生成出错：{str(e)}）"})
        await manager.send({"type": "stream_done"})

    finally:
        total_latency = time.time() - start_time
        logger.info(f"[{request_id}] Request completed | total={total_latency:.3f}s")
        await manager.send({"type": "done"})


async def _tts_consumer_v3(queue: asyncio.Queue, request_id: str):
    """从队列取日文句子 → TTS合成 → 发送音频。收到 None 哨兵时退出。"""
    index = 0
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        try:
            logger.info(f"[{request_id}] TTS v3 sentence {index}: {sentence[:30]}...")
            audio_base64 = await synthesize_sentence(sentence, request_id=request_id)
            if audio_base64:
                await manager.send({
                    "type": "audio_chunk", "data": audio_base64,
                    "index": index
                })
                index += 1
        except Exception as e:
            logger.error(f"[{request_id}] TTS v3 consumer error on sentence {index}: {e}")


async def handle_bot_reply_stream_v3(user_text: str, tts_enabled: bool = True):
    """方案三：整段流式翻译 + Queue 解耦 + 日文分句 TTS 并行。"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    full_reply = ""

    await manager.send({"type": "stream_start", "sender": mate_name})

    tts_queue: asyncio.Queue | None = None
    consumer_task: asyncio.Task | None = None

    try:
        # Phase 1: LLM 流式生成中文（与 v2 相同）
        async for chunk in iterate_in_thread(
            llm.generate_by_api_stream(user_text, request_id=request_id)
        ):
            if not chunk:
                continue
            full_reply += chunk
            await manager.send({"type": "stream_chunk", "data": chunk})

        await manager.send({"type": "stream_done"})
        llm_latency = time.time() - start_time
        logger.info(f"[{request_id}] LLM completed | latency={llm_latency:.3f}s")

        web_chat_history.append({"sender": mate_name, "text": full_reply})
        llm.append_history("assistant", full_reply)

        # Phase 2: 整段中文 → 流式翻译日文 → 按句入队 → consumer 并行 TTS
        if tts_enabled and full_reply.strip():
            tts_queue = asyncio.Queue()
            consumer_task = asyncio.create_task(
                _tts_consumer_v3(tts_queue, request_id)
            )

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

            # flush 剩余 buffer
            if ja_buffer.strip():
                await tts_queue.put(ja_buffer)

            await tts_queue.put(None)  # 哨兵
            await consumer_task
            await manager.send({"type": "audio_done"})

    except Exception as e:
        logger.error(f"[{request_id}] stream error: {e}")
        await manager.send({"type": "stream_error", "text": f"\n（生成出错：{str(e)}）"})
        await manager.send({"type": "stream_done"})
        if consumer_task is not None:
            consumer_task.cancel()

    finally:
        total_latency = time.time() - start_time
        logger.info(f"[{request_id}] Request completed | total={total_latency:.3f}s")
        await manager.send({"type": "done"})


def run_webchat():
    uvicorn.run(app, host="0.0.0.0", port=int(server_port))
