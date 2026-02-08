import logging
import os
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
async def get():
    return get_html_template()


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
                new_msg = {"sender": username, "text": user_text}
                web_chat_history.append(new_msg)
                await manager.send({"type": "message", "data": new_msg})
                await manager.send({"type": "processing"})
                asyncio.create_task(handle_bot_reply_stream(user_text))
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

async def handle_bot_reply_stream(user_text: str):
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
        if full_reply:
            tts_req = TTSRequest(
                text = full_reply
            )
            await tts.synthesize(tts_req)

        # 无论成功失败，都通知前端可以继续输入
        await manager.send({"type": "done"})


def get_html_template():
    html = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>对话 - lumina Web版</title>
        <link rel="icon" href="/data/assets/image/logo.png" type="image/png">
        <style>
        
        /* 原有样式保持不变，建议添加以下几行支持流式光标效果 */
        .bot-message .message-content .text-streaming::after {
            content: "|";
            animation: blink 1s step-end infinite;
        }
        @keyframes blink {
            50% { opacity: 0; }
        }
        body{font-family:'Arial',sans-serif;margin:0;padding:0;background-color:#f5f7fa;color:#333;
            background-image: url('/data/assets/image/bg.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            position: relative;
        }
        body::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(245, 247, 250, 0.25);
            z-index: -1;
        }
        .header{background-color:rgba(88, 126, 244, 0.25);color:white;padding:15px 20px;box-shadow:0 2px 5px rgba(0,0,0,0.1);position:relative;}
        .header h1{margin:0;font-size:24px;font-weight:600;display:inline-flex;align-items:center;}
        .header-logo{width:32px;height:32px;margin-right:10px;vertical-align:middle;}
        .header-info{margin-top:10px;font-size:14px;}
        .header-info a{color:#fff;text-decoration:none;margin-right:10px;background-color:rgba(255,255,255,0.15);padding:5px 10px;border-radius:4px;transition:background-color 0.3s;}
        .header-info a:hover{background-color:rgba(255,255,255,0.25);}
        .header-info button{background-color:rgba(255,255,255,0.25);border:none;color:white;padding:5px 10px;border-radius:4px;cursor:pointer;font-size:12px;margin-right:10px;transition:background-color 0.3s;}
        .header-info button:hover{background-color:rgba(255,255,255,0.3);}
        .chat-container{height:calc(100vh - 180px);overflow-y:auto;padding:15px;background-color:rgba(255, 255, 255, 0.25);margin:10px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
        .message{margin-bottom:15px;padding:10px 15px;border-radius:18px;max-width:80%;word-wrap:break-word;line-height:1.4;display:flex;align-items:flex-start;}
        .user-message{background-color:rgba(88, 126, 244, 0.25);color:white;margin-left:auto;border-bottom-right-radius:4px;max-width:33%;}
        .bot-message{background-color:rgba(240, 242, 245, 0.25);margin-right:auto;border-bottom-left-radius:4px;max-width:33%;}
        .message-avatar{width:36px;height:36px;border-radius:50%;margin-right:10px;flex-shrink:0;}
        .user-message .message-avatar{margin-left:10px;margin-right:0;order:2;}
        .message-content{flex-grow:1;}
        .message-sender{font-weight:bold;margin-bottom:5px;font-size:14px;}
        .input-area{position:fixed;bottom:0;left:0;right:0;background-color:rgba(255, 255, 255, 0.25);padding:15px;box-shadow:0 -2px 5px rgba(0,0,0,0.1);display:flex;align-items:center;}
        #msgInput{flex:1;padding:12px 15px;border:1px solid rgba(221, 221, 221, 0.5);border-radius:24px;outline:none;font-size:16px;margin-right:10px;background-color:rgba(255, 255, 255, 0.5);}
        #msgInput:focus{border-color:#587EF4;}
        .send-btn{background-color:rgba(88, 126, 244, 0.25);color:white;border:none;padding:12px 20px;border-radius:24px;cursor:pointer;font-weight:bold;transition:background-color 0.3s;}
        .send-btn:hover{background-color:rgba(71, 104, 201, 0.3);}
        .new-chat-btn{background-color:rgba(88, 126, 244, 0.25);color:white;border:none;padding:12px 15px;border-radius:50%;cursor:pointer;font-weight:bold;transition:background-color 0.3s;margin-right:10px;width:45px;height:45px;display:flex;align-items:center;justify-content:center;font-size:20px;}
        .new-chat-btn:hover{background-color:rgba(71, 104, 201, 0.3);}
        .mate-avatar{width:40px;height:40px;border-radius:50%;margin-right:10px;vertical-align:middle;}
        .header-content{display:flex;align-items:center;}
        .ai-disclaimer {
            position: fixed;
            bottom: 70px;
            left: 15px;
            font-size: 12px;
            color: #666;
            z-index: 10;
        }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <img class="mate-avatar" src="/data/assets/image/{mate_name}.png" alt="{mate_name}的头像">
                <div>
                    <h1>{mate_name}</h1>
                    <div class="header-info">
                        <a href="http://{server_ip}:{live2d_port}" target="_blank">Live2D角色</a>
                        <a href="http://{server_ip}:{mmd_port}" target="_blank">MMD 3D角色</a>
                        <a href="http://{server_ip}:{mmd_port}/vmd" target="_blank">MMD 3D动作</a>
                        <a href="http://{server_ip}:{vrm_port}" target="_blank">VRM 3D角色</a>
                    </div>
                </div>
            </div>
        </div>
        <div class="chat-container" id="chat"></div>
        <div class="input-area">
            <button class="new-chat-btn" onclick="clearChat()" title="开启新对话">+</button>
            <input id="msgInput" placeholder="和{mate_name}聊天...">
            <button class="send-btn" onclick="sendMsg()">发送</button>
        </div>
        <div class="ai-disclaimer">内容由AI生成,请仔细甄别</div>
        <script>
            let ws = new WebSocket("ws://" + location.host + "/ws");
            let userName = "{username}";
            let mateName = "{mate_name}";
            let isProcessing = false;
            let currentBotMessageElement = null;

            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    const chat = document.getElementById("chat");

                    if (data.type === "history") {
                        if (Array.isArray(data.data)) {
                            data.data.forEach(msg => addMessage(msg.sender, msg.text));
                        }
                    } else if (data.type === "message") {
                        addMessage(data.data.sender, data.data.text);
                    } else if (data.type === "clear") {
                        chat.innerHTML = "";
                    } else if (data.type === "processing") {
                        isProcessing = true;
                        disableInput();
                    } else if (data.type === "stream_start") {
                        currentBotMessageElement = addMessage(data.sender, "");
                        const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                        if(textDiv) textDiv.classList.add("text-streaming");
                    } else if (data.type === "stream_chunk") {
                        if (currentBotMessageElement && data.data) {
                            const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                            if(textDiv) textDiv.innerHTML += (data.data || "").replace(/\\n/g, "<br>");
                        }
                    } else if (data.type === "stream_done") {
                        if (currentBotMessageElement) {
                            const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                            if(textDiv) textDiv.classList.remove("text-streaming");
                            currentBotMessageElement = null;
                        }
                    } else if (data.type === "stream_error") {
                        const errorText = (data.text || "未知错误").replace(/\\n/g, "<br>");
                        if (currentBotMessageElement) {
                            const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                            if(textDiv) {
                                textDiv.classList.remove("text-streaming");
                                textDiv.innerHTML += errorText;
                            }
                            currentBotMessageElement = null;
                        } else {
                            addMessage(mateName, errorText);
                        }
                    } else if (data.type === "done") {
                        isProcessing = false;
                        enableInput();
                    }
                    if (chat) chat.scrollTop = chat.scrollHeight;
                } catch (e) {
                    console.error("Error processing message:", e);
                    // 尝试恢复
                    isProcessing = false;
                    enableInput();
                }
            };

            function sendMsg() {
                if (isProcessing) return;
                const text = document.getElementById("msgInput").value.trim();
                if (!text) { 
                    alert("请输入内容"); 
                    return; 
                }
                ws.send(JSON.stringify({action: "send", text: text}));
                document.getElementById("msgInput").value = "";
            }

            function clearChat() {
                if (confirm("确定要清空网页聊天记录吗？")) {
                    ws.send(JSON.stringify({action: "clear"}));
                }
            }

            function addMessage(who, text) {
                const messageDiv = document.createElement("div");
                messageDiv.className = who === userName ? "message user-message" : "message bot-message";
                const avatarImg = document.createElement("img");
                avatarImg.className = "message-avatar";
                avatarImg.alt = who + "的头像";
                avatarImg.src = who === userName 
                    ? "/data/assets/image/" + userName + ".png" 
                    : "/data/assets/image/" + mateName + ".png";
                const contentDiv = document.createElement("div");
                contentDiv.className = "message-content";
                const senderDiv = document.createElement("div");
                senderDiv.className = "message-sender";
                senderDiv.textContent = who;
                const textDiv = document.createElement("div");
                textDiv.innerHTML = (text || "").replace(/\\n/g, "<br>");
                contentDiv.appendChild(senderDiv);
                contentDiv.appendChild(textDiv);
                messageDiv.appendChild(avatarImg);
                messageDiv.appendChild(contentDiv);
                document.getElementById("chat").appendChild(messageDiv);
                return messageDiv;
            }

            function disableInput() {
                document.getElementById("msgInput").disabled = true;
                document.querySelector(".send-btn").disabled = true;
                document.getElementById("msgInput").placeholder = "正在思考，请稍等...";
            }

            function enableInput() {
                document.getElementById("msgInput").disabled = false;
                document.querySelector(".send-btn").disabled = false;
                document.getElementById("msgInput").placeholder = "和{mate_name}聊天...";
            }

            document.getElementById("msgInput").addEventListener("keydown", e => {
                if(e.key === "Enter" && !e.shiftKey) { 
                    e.preventDefault(); 
                    sendMsg(); 
                }
            });
        </script>
    </body>
    </html>'''
    return html.replace("{server_ip}", str(serverIp)) \
               .replace("{live2d_port}", str(live2dPort)) \
               .replace("{mmd_port}", str(mmdPort)) \
               .replace("{vrm_port}", str(vrmPort)) \
               .replace("{username}", str(username)) \
               .replace("{mate_name}", str(matename))


def run_webchat():
    uvicorn.run(app, host="0.0.0.0", port=int(serverPort))
