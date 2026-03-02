import base64
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from flask import Flask
from flask_sock import Sock

from core.config import load_app_config
from core.emotion.main import EmotionEngine
from core.utils import TraceLogger
from core.utils.errors import ErrorCode, error_payload
from core.llm.main import TranslateEngine
from core.orchestrator import Orchestrator
from core.tts.main import TTSEngine, TTSRequest
from service.pet.pipeline import AudioChunk, EmotionContext, OrderedSentenceMap, SentenceSlot

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# --- config ---
app_config = load_app_config()
username = app_config.service.username
server_ip = app_config.service.server_address
server_port = app_config.service.server_port

# --- engines ---
orchestrator = Orchestrator()
translator = TranslateEngine()
tts = TTSEngine()
emotion_engine = EmotionEngine()

# --- Flask app ---
app = Flask(__name__)
sock = Sock(app)

# --- thread pool ---
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

SENTENCE_DELIMITERS = re.compile(r'(?<=[。！？；\n])')


def split_sentences(text: str) -> tuple[list[str], str]:
    parts = SENTENCE_DELIMITERS.split(text)
    if len(parts) <= 1:
        return [], text
    sentences = [p for p in parts[:-1] if p.strip()]
    return sentences, parts[-1]


def ws_send(ws, msg: dict):
    ws.send(json.dumps(msg, ensure_ascii=False))


def ws_send_error(ws, code: ErrorCode, message: str, retryable: bool = False, details: dict = None):
    payload = error_payload(code=code, message=message, retryable=retryable, details=details)
    ws_send(ws, {"type": "error", **payload})


def sentence_worker(slot: SentenceSlot, emotion_ctx: EmotionContext, trace: TraceLogger):
    """Translate + wait emotion + TTS stream into sentence queue."""
    try:
        start = time.time()

        # 1) translate
        ja_text = translator.translate(slot.chinese_text)
        ok = bool(ja_text)
        if translator.last_error is not None:
            trace.log("translate_error", translator.last_error.to_payload())
        if not ok:
            ja_text = slot.chinese_text

        slot.japanese_text = ja_text
        logger.info(f"Slot {slot.index} translated in {time.time() - start:.2f}s")

        # 2) wait emotion
        emotion_ctx.event.wait(timeout=3)

        # 3) TTS streaming
        tts_req = TTSRequest(
            text=ja_text,
            ref_audio_path=emotion_ctx.ref_audio_path,
            prompt_text=emotion_ctx.prompt_text,
            media_type="raw",
        )
        result = tts.synthesize_streaming(tts_req)

        if result.get("success"):
            for chunk in result["audio_stream"]:
                slot.chunk_queue.put(AudioChunk(audio_bytes=chunk))
        else:
            slot.error = result.get("error", "TTS failed")
            trace.log(
                "tts_error",
                {
                    "code": result.get("error_code", ErrorCode.TTS_API_ERROR.value),
                    "message": slot.error,
                    "retryable": result.get("retryable", True),
                },
            )
            logger.error(f"TTS failed slot {slot.index}: {slot.error}")

        slot.chunk_queue.put(None)
        slot.done.set()
    except Exception as exc:
        logger.error(f"Worker error slot {slot.index}: {exc}")
        slot.error = str(exc)
        trace.log(
            "worker_error",
            {
                "code": ErrorCode.PIPELINE_ERROR.value,
                "message": str(exc),
                "retryable": True,
                "slot": slot.index,
            },
        )
        slot.chunk_queue.put(None)
        slot.done.set()


def consume_and_send(ws, ordered_map: OrderedSentenceMap):
    for slot in ordered_map.iter_slots_in_order():
        while True:
            item = slot.chunk_queue.get()
            if item is None:
                break
            audio_b64 = base64.b64encode(item.audio_bytes).decode("utf-8")
            ws_send(ws, {"type": "audio_chunk", "data": audio_b64})
    ws_send(ws, {"type": "audio_done"})


def handle_bot_reply(ws, user_text: str, session_id: str, trace: TraceLogger, round_num: int):
    started = time.time()

    trace.log("round_start", {"round": round_num, "user_text": user_text})

    emotion_ctx = EmotionContext()
    ordered_map = OrderedSentenceMap()
    sentence_index = 0

    tool_events = []
    try:
        orchestrated = orchestrator.handle_user_message(
            user_text=user_text,
            session_id=session_id,
            user_id=username,
        )
        full_reply = orchestrated.final_reply

        trace.log(
            "orchestration_route",
            {
                "intent": orchestrated.intent.value,
                "meta": orchestrated.meta,
            },
        )

        if orchestrated.executor_result is not None:
            tool_events = orchestrated.executor_result.tool_events
            for event in tool_events:
                trace.log("tool_event", event)
            if orchestrated.executor_result.error:
                trace.log("executor_error", orchestrated.executor_result.error)

        # parse emotion header and text
        emotion, text, intensity = emotion_engine.parse_leading_json(full_reply)
        emotion_ctx.emotion = emotion
        emotion_ctx.intensity = intensity
        emotion_ctx.ref_audio_path = emotion_engine.get_ref_audio_intensity(emotion, intensity)
        emotion_ctx.prompt_text = emotion_engine.get_prompt_text_intensity(emotion, intensity)
        emotion_ctx.event.set()

        cn_text = text if text.strip() else full_reply
        trace.log(
            "emotion_selected",
            {
                "emotion": emotion,
                "intensity": intensity,
                "text_preview": cn_text[:120],
            },
        )

        # sentence split -> parallel workers
        text_buffer = cn_text
        sentences, text_buffer = split_sentences(text_buffer)
        for s in sentences:
            slot = ordered_map.register(sentence_index, s)
            executor.submit(sentence_worker, slot, emotion_ctx, trace)
            sentence_index += 1

        if text_buffer.strip():
            slot = ordered_map.register(sentence_index, text_buffer)
            executor.submit(sentence_worker, slot, emotion_ctx, trace)
            sentence_index += 1

        ordered_map.mark_all_registered()

        ws_send(
            ws,
            {
                "type": "emotion_text",
                "emotion": emotion_ctx.emotion,
                "intensity": emotion_ctx.intensity,
                "text": cn_text,
            },
        )

        consume_and_send(ws, ordered_map)

        orchestrator.record_session_round(
            session_id=session_id,
            user_text=user_text,
            assistant_reply=full_reply,
            metadata={
                "round": round_num,
                "duration_ms": int((time.time() - started) * 1000),
                "tool_events": len(tool_events),
                "route_intent": orchestrated.intent.value,
                "task_id": orchestrated.meta.get("task_id"),
                "phase": orchestrated.meta.get("phase"),
            },
        )

    except Exception as exc:
        trace.log(
            "error",
            {
                "code": ErrorCode.PIPELINE_ERROR.value,
                "message": str(exc),
                "retryable": True,
                "round": round_num,
            },
        )
        logger.error(f"[Round {round_num}] error: {exc}")
        ws_send_error(
            ws,
            code=ErrorCode.PIPELINE_ERROR,
            message="Pipeline execution failed",
            retryable=True,
            details={"reason": str(exc), "round": round_num},
        )
    finally:
        ws_send(ws, {"type": "done"})
        trace.log("round_end", {"round": round_num, "cost_sec": round(time.time() - started, 2)})


@sock.route('/ws')
def websocket_handler(ws):
    logger.info("WebSocket connected")
    session_id = f"ws-{uuid.uuid4().hex[:10]}"
    trace = TraceLogger(session_id=session_id)
    trace.log("session_start", {"session_id": session_id})
    round_count = 0

    disconnect_reason = "client_closed"
    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break
            data = json.loads(raw)
            user_text = data.get("content", "").strip()
            if not user_text:
                continue
            round_count += 1
            handle_bot_reply(ws, user_text, session_id, trace, round_count)
    except Exception as exc:
        disconnect_reason = str(exc)
        logger.info(f"WebSocket disconnected: {exc}")
        trace.log(
            "session_disconnect",
            {
                "code": ErrorCode.WEBSOCKET_ERROR.value,
                "message": disconnect_reason,
                "retryable": True,
            },
        )
    finally:
        trace.log("session_end", {"reason": disconnect_reason, "rounds": round_count})
        trace.close()


def run_pet():
    logger.info(f"Pet service starting on {server_ip}:{server_port}")
    app.run(host=server_ip, port=server_port, threaded=True)
