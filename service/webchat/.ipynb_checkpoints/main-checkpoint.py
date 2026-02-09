import logging
import os
import re
import asyncio
import uvicorn
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
from core.llm.main import LLMEngine
from core.tts.main import TTSEngine, TTSRequest

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
username = config["character_info"]["username"]
matename = config["character_info"]["mate_name"]
serverIp = config["server_address"]
serverPort = config["server_port"]
live2dPort = config["live2d_port"]
mmdPort = config["mmd_port"]
vrmPort =config["vrm_port"]


app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/data/assets/image", StaticFiles(directory="data/assets/image"))

# logging.getLogger('lumina').setLevel(logging.ERROR)

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
        "server_ip": serverIp,
        "live2d_port": live2dPort,
        "mmd_port": mmdPort,
        "vrm_port": vrmPort,
        "username": username,
        "mate_name": matename
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

# async def handle_bot_reply(user_text: str):   
#     bot_reply = llm.generate_by_api(user_text).replace("#", "").replace("*", "")
#     new_msg = {"sender": matename, "text": bot_reply}
#     web_chat_history.append(new_msg)
#     await manager.send({"type": "message", "data": new_msg})
#     await manager.send({"type": "done"})

def split_sentences(text: str) -> list[str]:
    """按句末标点切分文本，标点保留在句子末尾。"""
    parts = re.split(r'(?<=[。！？!?.])', text)
    sentences = [p for p in parts if p.strip()]
    return sentences


async def synthesize_sentence(sentence: str) -> str | None:
    """对单个句子调用 TTS，返回 base64 音频字符串，失败返回 None。"""
    try:
        tts_req = TTSRequest(text=sentence, streaming=False)
        result = await tts.synthesize(tts_req)
        if result.get("success"):
            audio_data = result.get("audio_bytes")
            if audio_data:
                return base64.b64encode(audio_data).decode('utf-8')
        else:
            print(f"TTS processing failed for sentence: {result.get('error')}")
    except Exception as e:
        print(f"TTS exception for sentence: {e}")
    return None


async def handle_bot_reply_stream(user_text: str, tts_enabled: bool = True):
    full_reply = ""
    # 通知前端：开始一个新的 bot 消息（空内容）
    await manager.send({
        "type": "stream_start",
        "sender": matename
    })

    # TTS 并发相关
    audio_queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()
    sentence_buffer = ""
    audio_index = 0

    async def audio_dispatcher():
        """从队列中按顺序取出 TTS 任务，await 结果后发送给前端。"""
        idx = 0
        while True:
            task = await audio_queue.get()
            if task is None:  # 哨兵值，表示没有更多音频
                break
            try:
                audio_base64 = await task
                if audio_base64:
                    await manager.send({
                        "type": "audio_chunk",
                        "data": audio_base64,
                        "index": idx
                    })
            except Exception as e:
                print(f"Audio dispatcher error: {e}")
            idx += 1
        await manager.send({"type": "audio_done"})

    # 启动 audio_dispatcher 协程（仅在 TTS 开启时）
    dispatcher_task = None
    if tts_enabled:
        dispatcher_task = asyncio.create_task(audio_dispatcher())

    try:
        for chunk in llm.generate_by_api_stream(user_text):
            if not chunk:
                continue
            full_reply += chunk
            # 实时推送文本增量
            await manager.send({
                "type": "stream_chunk",
                "data": chunk
            })

            # TTS：按句子边界切分并提交
            if tts_enabled:
                sentence_buffer += chunk
                sentences = split_sentences(sentence_buffer)
                if len(sentences) > 1:
                    # 最后一个元素可能是不完整的句子，保留在 buffer 中
                    for s in sentences[:-1]:
                        task = asyncio.create_task(synthesize_sentence(s))
                        await audio_queue.put(task)
                        audio_index += 1
                    sentence_buffer = sentences[-1]

        # 流结束，推送完成信号
        await manager.send({"type": "stream_done"})

        # flush 剩余 buffer
        if tts_enabled and sentence_buffer.strip():
            task = asyncio.create_task(synthesize_sentence(sentence_buffer))
            await audio_queue.put(task)

        # 存入完整历史（用于重连）
        web_chat_history.append({"sender": matename, "text": full_reply})

    except Exception as e:
        await manager.send({
            "type": "stream_error",
            "text": f"\n（生成出错：{str(e)}）"
        })
        await manager.send({"type": "stream_done"})

    finally:
        llm.append_history("assistant", full_reply)

        if tts_enabled and dispatcher_task:
            # 发送哨兵值通知 dispatcher 结束
            await audio_queue.put(None)
            await dispatcher_task

        # 无论成功失败，都通知前端可以继续输入
        await manager.send({"type": "done"})


def run_webchat():
    uvicorn.run(app, host="0.0.0.0", port=int(serverPort))
