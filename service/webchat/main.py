import logging
import os
import re
import asyncio
import uvicorn
import base64
import uuid
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
                new_msg = {"sender": username, "text": user_text}
                web_chat_history.append(new_msg)
                await manager.send({"type": "message", "data": new_msg})
                await manager.send({"type": "processing"})
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


def split_sentences(text: str) -> list[str]:
    """按句末标点切分文本，标点保留在句子末尾。"""
    parts = re.split(r'(?<=[。！？!?.])', text)
    sentences = [p for p in parts if p.strip()]
    return sentences


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


async def handle_bot_reply_stream(user_text: str, tts_enabled: bool = True):
    request_id = str(uuid.uuid4())[:8]
    full_reply = ""

    await manager.send({"type": "stream_start", "sender": mate_name})

    audio_queue: asyncio.Queue[str | None] = asyncio.Queue()
    sentence_buffer = ""

    async def audio_dispatcher():
        """按句子顺序，发送完整音频到前端。"""
        idx = 0
        while True:
            sentence = await audio_queue.get()
            if sentence is None:
                break
            try:
                audio_base64 = await synthesize_sentence(sentence, request_id=request_id)
                if audio_base64:
                    await manager.send({"type": "audio_chunk", "data": audio_base64, "index": idx})
                    idx += 1
            except Exception as e:
                logger.error(f"[{request_id}] audio_dispatcher error: {e}")
        await manager.send({"type": "audio_done"})

    dispatcher_task = asyncio.create_task(audio_dispatcher()) if tts_enabled else None

    try:
        async for chunk in iterate_in_thread(llm.generate_by_api_stream(user_text, request_id=request_id)):
            if not chunk:
                continue
            full_reply += chunk
            await manager.send({"type": "stream_chunk", "data": chunk})

            if tts_enabled:
                sentence_buffer += chunk
                sentences = split_sentences(sentence_buffer)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        await audio_queue.put(s)
                    sentence_buffer = sentences[-1]

        await manager.send({"type": "stream_done"})

        if tts_enabled and sentence_buffer.strip():
            await audio_queue.put(sentence_buffer)

        web_chat_history.append({"sender": mate_name, "text": full_reply})

    except Exception as e:
        logger.error(f"[{request_id}] stream error: {e}")
        await manager.send({"type": "stream_error", "text": f"\n（生成出错：{str(e)}）"})
        await manager.send({"type": "stream_done"})

    finally:
        llm.append_history("assistant", full_reply)
        if tts_enabled and dispatcher_task:
            await audio_queue.put(None)
            await dispatcher_task
        await manager.send({"type": "done"})


def run_webchat():
    uvicorn.run(app, host="0.0.0.0", port=int(server_port))
