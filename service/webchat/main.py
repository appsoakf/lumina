import logging
import os
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

async def handle_bot_reply_stream(user_text: str, tts_enabled: bool = True):
    full_reply = ""
    # 通知前端：开始一个新的 bot 消息（空内容）
    await manager.send({
        "type": "stream_start",
        "sender": matename
    })

    try:
        for chunk in llm.generate_by_api_stream(user_text):
            if not chunk:
                continue
            full_reply += chunk
            # 实时推送增量
            await manager.send({
                "type": "stream_chunk",
                "data": chunk
            })

        # 流结束，推送完成信号
        await manager.send({"type": "stream_done"})

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
        # 修改：仅在启用TTS时生成音频
        if full_reply and tts_enabled:
            try:
                tts_req = TTSRequest(text=full_reply)
                result = await tts.synthesize(tts_req)
                if result.get("success"):
                    audio_data = result.get("audio_bytes")
                    if audio_data:
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        await manager.send({"type": "audio", "data": audio_base64})
                else:
                    print(f"TTS processing failed: {result.get('error')}")
            except Exception as e:
                print(f"TTS exception: {e}")

        # 无论成功失败，都通知前端可以继续输入
        await manager.send({"type": "done"})


def run_webchat():
    uvicorn.run(app, host="0.0.0.0", port=int(serverPort))
