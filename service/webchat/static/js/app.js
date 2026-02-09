let config = null;
let ws = null;
let isProcessing = false;
let currentBotMessageElement = null;
let ttsEnabled = localStorage.getItem('ttsEnabled') !== 'false'; // 默认开启
let audioQueue = [];
let isPlayingAudio = false;

async function initApp() {
    try {
        const response = await fetch('/api/config');
        config = await response.json();
        updateUIWithConfig();
        initWebSocket();
        setupEventListeners();
        setupTTSToggle();
    } catch (e) {
        console.error("Failed to load config:", e);
    }
}

function updateUIWithConfig() {
    document.getElementById('mate-name-title').textContent = config.mate_name;
    document.getElementById('header-avatar').src = `/data/assets/image/${config.mate_name}.png`;
    document.getElementById('header-avatar').alt = `${config.mate_name}的头像`;

    const headerLinks = document.getElementById('header-links');
    const ttsButton = document.getElementById('tts-toggle');
    const existingLinks = `
        <a href="http://${config.server_ip}:${config.live2d_port}" target="_blank">Live2D角色</a>
        <a href="http://${config.server_ip}:${config.mmd_port}" target="_blank">MMD 3D角色</a>
        <a href="http://${config.server_ip}:${config.mmd_port}/vmd" target="_blank">MMD 3D动作</a>
        <a href="http://${config.server_ip}:${config.vrm_port}" target="_blank">VRM 3D角色</a>
    `;
    headerLinks.innerHTML = existingLinks;
    // 重新添加TTS按钮到最前面
    headerLinks.insertAdjacentHTML('afterbegin', '<button id="tts-toggle" class="tts-toggle" title="切换语音"><span id="tts-icon">🔊</span> <span id="tts-text">语音开启</span></button>');
    // 重新初始化TTS toggle（因为DOM已重建）
    setupTTSToggle();

    document.getElementById('msgInput').placeholder = `和${config.mate_name}聊天...`;
}

function initWebSocket() {
    ws = new WebSocket("ws://" + location.host + "/ws");

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
                audioQueue = [];
                isPlayingAudio = false;
                currentBotMessageElement = addMessage(data.sender, "");
                const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                if(textDiv) textDiv.classList.add("text-streaming");
            } else if (data.type === "stream_chunk") {
                if (currentBotMessageElement && data.data) {
                    const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                    if(textDiv) {
                        textDiv.innerHTML += (data.data || "").replace(/\n/g, "<br>");
                        scrollToBottom();
                    }
                }
            } else if (data.type === "stream_done") {
                if (currentBotMessageElement) {
                    const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                    if(textDiv) textDiv.classList.remove("text-streaming");
                    currentBotMessageElement = null;
                }
            } else if (data.type === "stream_error") {
                const errorText = (data.text || "未知错误").replace(/\n/g, "<br>");
                if (currentBotMessageElement) {
                    const textDiv = currentBotMessageElement.querySelector(".message-content div:last-of-type");
                    if(textDiv) {
                        textDiv.classList.remove("text-streaming");
                        textDiv.innerHTML += errorText;
                    }
                    currentBotMessageElement = null;
                } else {
                    addMessage(config.mate_name, errorText);
                }
            } else if (data.type === "audio_chunk") {
                audioQueue.push(data.data);
                if (!isPlayingAudio) {
                    playNextAudioChunk();
                }
            } else if (data.type === "audio_done") {
                // 所有音频片段已发送，队列会自然排空
            } else if (data.type === "done") {
                isProcessing = false;
                enableInput();
            }
        } catch (e) {
            console.error("WebSocket message error:", e);
        }
    };

    ws.onerror = function(error) {
        console.error("WebSocket error:", error);
    };

    ws.onclose = function() {
        console.log("WebSocket closed");
    };
}

function setupEventListeners() {
    document.getElementById("msgInput").addEventListener("keydown", e => {
        if(e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMsg();
        }
    });
}

function setupTTSToggle() {
    const toggleBtn = document.getElementById('tts-toggle');
    if (!toggleBtn) return;

    const icon = document.getElementById('tts-icon');
    const text = document.getElementById('tts-text');

    // 初始化UI状态
    updateTTSUI();

    // 移除旧的事件监听器（如果存在）
    const newToggleBtn = toggleBtn.cloneNode(true);
    toggleBtn.parentNode.replaceChild(newToggleBtn, toggleBtn);

    // 添加点击事件
    newToggleBtn.addEventListener('click', () => {
        ttsEnabled = !ttsEnabled;
        localStorage.setItem('ttsEnabled', ttsEnabled);
        updateTTSUI();
    });

    function updateTTSUI() {
        const currentIcon = document.getElementById('tts-icon');
        const currentText = document.getElementById('tts-text');
        const currentBtn = document.getElementById('tts-toggle');

        if (!currentIcon || !currentText || !currentBtn) return;

        if (ttsEnabled) {
            currentIcon.textContent = '🔊';
            currentText.textContent = '语音开启';
            currentBtn.classList.remove('disabled');
        } else {
            currentIcon.textContent = '🔇';
            currentText.textContent = '语音关闭';
            currentBtn.classList.add('disabled');
        }
    }
}

function playNextAudioChunk() {
    if (audioQueue.length === 0) {
        isPlayingAudio = false;
        return;
    }
    isPlayingAudio = true;
    const audioData = audioQueue.shift();
    try {
        const audio = new Audio("data:audio/wav;base64," + audioData);
        audio.onended = function() {
            playNextAudioChunk();
        };
        audio.onerror = function(e) {
            console.error("Audio playback error:", e);
            playNextAudioChunk();
        };
        audio.play().catch(e => {
            console.error("Audio play error:", e);
            playNextAudioChunk();
        });
    } catch (e) {
        console.error("Audio processing error:", e);
        playNextAudioChunk();
    }
}

function sendMsg() {
    if (isProcessing) return;
    const input = document.getElementById("msgInput");
    const text = input.value.trim();
    if (!text) return;
    ws.send(JSON.stringify({
        action: "send",
        text: text,
        tts_enabled: ttsEnabled
    }));
    input.value = "";
}

function clearChat() {
    if (confirm("确定要清空网页聊天记录吗？")) {
        ws.send(JSON.stringify({action: "clear"}));
    }
}

function addMessage(who, text) {
    const messageDiv = document.createElement("div");
    messageDiv.className = who === config.username ? "message user-message" : "message bot-message";
    const avatarImg = document.createElement("img");
    avatarImg.className = "message-avatar";
    avatarImg.alt = who + "的头像";
    avatarImg.src = who === config.username
        ? "/data/assets/image/" + config.username + ".png"
        : "/data/assets/image/" + config.mate_name + ".png";
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    const senderDiv = document.createElement("div");
    senderDiv.className = "message-sender";
    senderDiv.textContent = who;
    const textDiv = document.createElement("div");
    textDiv.innerHTML = (text || "").replace(/\n/g, "<br>");
    contentDiv.appendChild(senderDiv);
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(avatarImg);
    messageDiv.appendChild(contentDiv);
    document.getElementById("chat").appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function scrollToBottom() {
    const chat = document.getElementById("chat");
    chat.scrollTop = chat.scrollHeight;
}

function disableInput() {
    document.getElementById("msgInput").disabled = true;
    document.querySelector(".send-btn").disabled = true;
    document.getElementById("msgInput").placeholder = "正在思考，请稍等...";
}

function enableInput() {
    document.getElementById("msgInput").disabled = false;
    document.querySelector(".send-btn").disabled = false;
    document.getElementById("msgInput").placeholder = `和${config.mate_name}聊天...`;
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
