import base64
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from flask import Flask
from flask_sock import Sock

from core.config import load_app_config
from core.emotion.main import EmotionEngine
from core.utils import (
    TraceLogger,
    bind_log_context,
    clear_log_context,
    elapsed_ms,
    log_event,
    log_exception,
    set_log_context,
    summarize_text,
)
from core.utils.logging_setup import setup_logging
from core.utils.errors import ErrorCode, error_payload
from core.llm.main import TranslateEngine
from core.orchestrator import Orchestrator
from core.tts.main import TTSEngine, TTSRequest
from service.pet.pipeline import AudioChunk, EmotionContext, OrderedSentenceMap, SentenceSlot
from service.pet.ws_contract import parse_user_text

# --- config ---
app_config = load_app_config()
server_ip = app_config.service.server_address
server_port = app_config.service.server_port
ENABLE_TRANSLATION = app_config.service.enable_translation
ENABLE_TTS = app_config.service.enable_tts

# --- logging ---
setup_logging(app_config.logging)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# --- engines ---
orchestrator = Orchestrator()
translator: Optional[TranslateEngine] = TranslateEngine() if ENABLE_TRANSLATION else None
tts: Optional[TTSEngine] = TTSEngine() if ENABLE_TTS else None
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


def sentence_worker(
    slot: SentenceSlot,
    emotion_ctx: EmotionContext,
    trace: TraceLogger,
    session_id: str,
    round_num: int,
):
    """Translate + wait emotion + TTS stream into sentence queue (gated by config)."""
    set_log_context(
        session_id=session_id,
        round=round_num,
        step_id=f"sentence_slot_{slot.index}",
    )
    try:
        # 1) translate（可通过 service.enable_translation 开关）
        ja_text = slot.chinese_text
        if ENABLE_TRANSLATION and translator is not None:
            translate_start = time.perf_counter()
            translated = translator.translate_with_status(slot.chinese_text)
            if translated.error is not None:
                trace.log("translate_error", translated.error.to_payload())
            if translated.ok:
                ja_text = translated.text
            translate_duration = elapsed_ms(translate_start)
            log_event(
                logger,
                logging.INFO,
                "pipeline.translate.done",
                f"句子翻译完成（slot={slot.index}）",
                slot_index=slot.index,
                duration_ms=translate_duration,
                fallback_used=not translated.ok,
            )
        else:
            log_event(
                logger,
                logging.INFO,
                "pipeline.translate.skipped",
                f"翻译已关闭，跳过翻译（slot={slot.index}）",
                slot_index=slot.index,
            )

        slot.japanese_text = ja_text

        # 2) wait emotion + 3) TTS streaming（可通过 service.enable_tts 开关）
        if ENABLE_TTS and tts is not None:
            wait_start = time.perf_counter()
            emotion_ready = emotion_ctx.event.wait(timeout=3)
            log_event(
                logger,
                logging.INFO,
                "pipeline.emotion.wait",
                f"情绪上下文等待结束（slot={slot.index}）",
                slot_index=slot.index,
                duration_ms=elapsed_ms(wait_start),
                emotion_ready=emotion_ready,
            )

            tts_start = time.perf_counter()
            tts_req = TTSRequest(
                text=ja_text,
                ref_audio_path=emotion_ctx.ref_audio_path,
                prompt_text=emotion_ctx.prompt_text,
                media_type="raw",
            )
            result = tts.synthesize_streaming(tts_req)

            if result.get("success"):
                chunk_count = 0
                byte_total = 0
                for chunk in result["audio_stream"]:
                    slot.chunk_queue.put(AudioChunk(audio_bytes=chunk))
                    chunk_count += 1
                    byte_total += len(chunk)
                log_event(
                    logger,
                    logging.INFO,
                    "pipeline.tts.stream.done",
                    f"TTS 音频流推送完成（slot={slot.index}）",
                    slot_index=slot.index,
                    duration_ms=elapsed_ms(tts_start),
                    chunk_count=chunk_count,
                    bytes_total=byte_total,
                )
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
                log_event(
                    logger,
                    logging.ERROR,
                    "pipeline.tts.stream.error",
                    f"TTS 生成失败（slot={slot.index}）",
                    slot_index=slot.index,
                    duration_ms=elapsed_ms(tts_start),
                    error_code=result.get("error_code", ErrorCode.TTS_API_ERROR.value),
                    retryable=bool(result.get("retryable", True)),
                    error_message=slot.error,
                )
        else:
            log_event(
                logger,
                logging.INFO,
                "pipeline.tts.stream.skipped",
                f"TTS 已关闭，跳过音频合成（slot={slot.index}）",
                slot_index=slot.index,
            )

        slot.chunk_queue.put(None)
        slot.done.set()
    except Exception as exc:
        log_exception(
            logger,
            "pipeline.worker.error",
            f"句子工作线程异常（slot={slot.index}）",
            slot_index=slot.index,
            error_code=ErrorCode.PIPELINE_ERROR.value,
            retryable=True,
        )
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
    finally:
        clear_log_context()


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
    started = time.perf_counter()
    def _as_int(payload: dict, key: str, default: int = -1) -> int:
        try:
            return int(payload.get(key))
        except Exception:
            return default

    summary = summarize_text(
        user_text,
        preview_chars=app_config.logging.user_text_preview_chars,
        redact=app_config.logging.redact_user_text,
    )
    with bind_log_context(session_id=session_id, round=round_num):
        trace.log("round_start", {"round": round_num, **summary})
        log_event(
            logger,
            logging.INFO,
            "ws.round.start",
            f"开始处理第 {round_num} 轮请求",
            round=round_num,
            **summary,
        )

        emotion_ctx = EmotionContext()
        ordered_map = OrderedSentenceMap()
        sentence_index = 0

        tool_events = []
        route_intent: Optional[str] = None
        route_task_id: Optional[str] = None
        route_meta: dict = {}
        route_perf: dict = {}
        route_duration_ms = -1
        tts_total_ms = -1
        try:
            route_start = time.perf_counter()
            orchestrated = orchestrator.handle_user_message(
                user_text=user_text,
                session_id=session_id,
            )
            route_duration_ms = elapsed_ms(route_start)
            full_reply = orchestrated.final_reply
            route_intent = orchestrated.intent.value
            route_task_id = str(orchestrated.meta.get("task_id") or "").strip() or None
            route_perf = dict(orchestrated.meta.get("perf") or {})

            route_meta = {
                "task_mode": bool(orchestrated.meta.get("task_mode")),
                "task_id": route_task_id,
                "agent_chain": list(orchestrated.meta.get("agent_chain") or []),
                "task_error": bool(orchestrated.meta.get("task_error")),
                "task_waiting_input": bool(orchestrated.meta.get("task_waiting_input")),
                "task_waiting_step_id": str(orchestrated.meta.get("task_waiting_step_id") or "").strip() or None,
                "task_clarify_question": str(orchestrated.meta.get("task_clarify_question") or "").strip() or None,
                "task_required_fields": list(orchestrated.meta.get("task_required_fields") or []),
                "task_round_count": int(orchestrated.meta.get("task_round_count") or 0),
                "task_replan_count": int(orchestrated.meta.get("task_replan_count") or 0),
            }
            trace.log(
                "orchestration_route",
                {
                    "intent": route_intent,
                    "meta": route_meta,
                },
            )
            log_event(
                logger,
                logging.INFO,
                "orchestrator.route.done",
                "编排路由完成",
                duration_ms=route_duration_ms,
                intent=route_intent,
                task_id=route_task_id or "-",
                task_mode=route_meta["task_mode"],
                task_error=route_meta["task_error"],
                task_waiting_input=route_meta["task_waiting_input"],
                task_round_count=route_meta["task_round_count"],
                task_replan_count=route_meta["task_replan_count"],
            )

            if orchestrated.executor_result is not None:
                tool_events = orchestrated.executor_result.tool_events
                for event in tool_events:
                    trace.log("tool_event", event)
                if orchestrated.executor_result.error:
                    trace.log("executor_error", orchestrated.executor_result.error)
                    log_event(
                        logger,
                        logging.WARNING,
                        "executor.result.error",
                        "执行器返回错误结果",
                        error_code=orchestrated.executor_result.error.get("code"),
                        retryable=bool(orchestrated.executor_result.error.get("retryable")),
                    )

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
            log_event(
                logger,
                logging.INFO,
                "pipeline.emotion.selected",
                "情绪解析完成",
                emotion=emotion,
                intensity=intensity,
                text_len=len(cn_text),
            )

            # sentence split -> parallel workers
            tts_total_start = time.perf_counter()
            text_buffer = cn_text
            sentences, text_buffer = split_sentences(text_buffer)
            for s in sentences:
                slot = ordered_map.register(sentence_index, s)
                executor.submit(
                    sentence_worker,
                    slot,
                    emotion_ctx,
                    trace,
                    session_id,
                    round_num,
                )
                sentence_index += 1

            if text_buffer.strip():
                slot = ordered_map.register(sentence_index, text_buffer)
                executor.submit(
                    sentence_worker,
                    slot,
                    emotion_ctx,
                    trace,
                    session_id,
                    round_num,
                )
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

            audio_start = time.perf_counter()
            consume_and_send(ws, ordered_map)
            log_event(
                logger,
                logging.INFO,
                "ws.audio.send.done",
                "音频分片发送完成",
                duration_ms=elapsed_ms(audio_start),
                sentence_count=sentence_index,
            )
            tts_total_ms = elapsed_ms(tts_total_start)
            log_event(
                logger,
                logging.INFO,
                "pipeline.tts.total.done",
                "语音阶段处理完成",
                duration_ms=tts_total_ms,
                sentence_count=sentence_index,
                enable_tts=bool(ENABLE_TTS and tts is not None),
            )

            orchestrator.record_session_round(
                session_id=session_id,
                user_text=user_text,
                assistant_reply=full_reply,
                metadata={
                    "round": round_num,
                    "duration_ms": elapsed_ms(started),
                    "tool_events": len(tool_events),
                    "route_intent": route_intent,
                    "task_id": route_task_id,
                    "task_waiting_input": bool(route_meta.get("task_waiting_input")) if isinstance(route_meta, dict) else False,
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
            log_exception(
                logger,
                "ws.round.error",
                f"第 {round_num} 轮处理失败",
                error_code=ErrorCode.PIPELINE_ERROR.value,
                retryable=True,
            )
            ws_send_error(
                ws,
                code=ErrorCode.PIPELINE_ERROR,
                message="Pipeline execution failed",
                retryable=True,
                details={"reason": str(exc), "round": round_num},
            )
        finally:
            ws_send(ws, {"type": "done"})
            round_duration_ms = elapsed_ms(started)
            round_cost_sec = round(round_duration_ms / 1000, 2)
            trace.log("round_end", {"round": round_num, "cost_sec": round_cost_sec})
            log_event(
                logger,
                logging.INFO,
                "ws.round.summary",
                f"第 {round_num} 轮性能摘要",
                intent=route_intent or "-",
                task_id=route_task_id or "-",
                duration_ms=round_duration_ms,
                route_ms=route_duration_ms,
                intent_ms=_as_int(route_perf, "intent_ms"),
                task_run_ms=_as_int(route_perf, "task_run_ms"),
                chat_llm_ms=_as_int(route_perf, "chat_llm_ms"),
                orchestrator_ms=_as_int(route_perf, "orchestrator_ms"),
                tts_ms=tts_total_ms,
                round_total_ms=round_duration_ms,
            )
            log_event(
                logger,
                logging.INFO,
                "ws.round.end",
                f"第 {round_num} 轮处理结束",
                duration_ms=round_duration_ms,
                tool_event_count=len(tool_events),
                route_intent=route_intent or "-",
                task_id=route_task_id or "-",
            )


@sock.route('/ws')
def websocket_handler(ws):
    session_id = f"ws-{uuid.uuid4().hex[:10]}"
    with bind_log_context(session_id=session_id):
        trace = TraceLogger(session_id=session_id)
        trace.log("session_start", {"session_id": session_id})
        log_event(
            logger,
            logging.INFO,
            "ws.session.start",
            "WebSocket 会话已连接",
            session_id=session_id,
        )
        round_count = 0

        disconnect_reason = "client_closed"
        try:
            while True:
                raw = ws.receive()
                if raw is None:
                    break
                user_text, request_error = parse_user_text(raw)
                if request_error is not None:
                    trace.log("invalid_request", request_error.to_payload())
                    log_event(
                        logger,
                        logging.WARNING,
                        "ws.request.invalid",
                        "收到非法 WebSocket 消息",
                        error_code=request_error.code.value,
                        retryable=request_error.retryable,
                        details=request_error.details,
                    )
                    ws_send_error(
                        ws,
                        code=request_error.code,
                        message=request_error.message,
                        retryable=request_error.retryable,
                        details=request_error.details,
                    )
                    continue
                if not user_text:
                    continue
                round_count += 1
                handle_bot_reply(ws, user_text, session_id, trace, round_count)
        except Exception as exc:
            disconnect_reason = str(exc)
            log_exception(
                logger,
                "ws.session.disconnect.error",
                "WebSocket 会话异常断开",
                error_code=ErrorCode.WEBSOCKET_ERROR.value,
                retryable=True,
            )
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
            log_event(
                logger,
                logging.INFO,
                "ws.session.end",
                "WebSocket 会话结束",
                rounds=round_count,
                reason=disconnect_reason,
            )


def shutdown_runtime():
    try:
        orchestrator.close()
    except Exception:
        log_exception(
            logger,
            "runtime.shutdown.orchestrator.error",
            "Orchestrator 关闭失败",
        )
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        log_exception(
            logger,
            "runtime.shutdown.executor.error",
            "线程池关闭失败",
        )


def run_pet():
    log_event(
        logger,
        logging.INFO,
        "service.start",
        f"Pet 服务启动监听 {server_ip}:{server_port}",
        server_ip=server_ip,
        server_port=server_port,
        enable_translation=ENABLE_TRANSLATION,
        enable_tts=ENABLE_TTS,
    )
    try:
        app.run(host=server_ip, port=server_port, threaded=True)
    finally:
        shutdown_runtime()
