# Lumina 日志规范

## 1. 目标
- 语义清晰：日志消息可直接阅读。
- 异常可定位：包含代码位置、异常类型、traceback。
- 性能可测试：关键链路输出耗时字段（`duration_ms`）。

## 2. 配置
日志配置位于 `config.json` 的 `logging` 节点：
- `level`: 日志级别（`DEBUG/INFO/WARNING/ERROR/CRITICAL`）
- `format`: `human/json/both`
- `log_dir`: 相对路径默认挂到 `runtime/`
- `log_file_name`: 文本日志文件名
- `event_file_name`: 结构化 JSONL 文件名（默认 `events.jsonl`）
- `enable_console/enable_file/enable_event_file`
- `slow_threshold_ms`
- `redact_user_text/user_text_preview_chars`

## 3. 输出位置
- 结构化事件日志：`runtime/logs/events.jsonl`
- 文本日志：`runtime/logs/lumina.log`
- 兼容追踪日志：`runtime/traces/trace-*.jsonl`

## 4. 结构化事件字段
通用字段：
- `ts`
- `level`
- `event`
- `msg`
- `logger`
- `component`
- `session_id`
- `round`
- `task_id`
- `step_id`
- `file`
- `line`
- `func`

可选字段：
- `duration_ms`
- `error_code`
- `retryable`
- `exception_type`
- `exception_message`
- `traceback`
- 业务字段（如 `tool`, `model`, `chunk_count`, `bytes_total`）

## 5. 关键事件
- WS 链路：`ws.session.start/end`, `ws.round.start/end`, `ws.round.error`
- 编排链路：`orchestrator.route.done`, `task.step.run.done/error`
- LLM 链路：`llm.invoke.done`, `llm.invoke.error`, `llm.stream.open/error`
- 工具链路：`tool.call.done/retry/error/bad_args`
- TTS 链路：`tts.request.done/error`, `tts.stream.end/error`
- Memory 链路：`memory.*`（queue/full/init/search/upsert 等）

## 6. 指标汇总
命令：
```bash
python scripts/summarize_metrics.py --json
```

可选参数：
```bash
python scripts/summarize_metrics.py --event-log-dir runtime/logs --json
```

输出新增：
- `event_log_files`
- `latency_ms.round/llm_invoke/tool_call/tts_stream`
  - `count`, `avg_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `max_ms`

## 7. 使用约束
- 新增异常捕获时，优先使用结构化异常日志接口。
- fallback 分支必须留痕，不允许静默吞错。
- 不在日志中输出 API key 等敏感信息。
