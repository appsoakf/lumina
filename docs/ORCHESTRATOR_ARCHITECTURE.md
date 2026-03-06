# Orchestrator 编排系统架构图

本文档聚焦 `service -> orchestrator -> agent/task/tool/memory` 的编排链路，帮助快速理解“用户输入后，系统如何持续推进直到收敛或可恢复退出”。

## 1. 总览组件图

```mermaid
flowchart LR
    Client[Client]
    WS[service/pet/main.py<br/>websocket_handler]
    Pipe[service/pet/main.py<br/>handle_bot_reply]
    Orch[core/orchestrator/orchestrator.py<br/>Orchestrator.handle_user_message]

    Chat[core/agentic/chat_agent.py<br/>ChatAgent]
    Planner[core/agentic/planner_agent.py<br/>PlannerAgent]
    Runner[core/orchestrator/langgraph_task_runner.py<br/>LangGraphTaskRunner]
    Executor[core/agentic/executor_agent.py<br/>ExecutorAgent ReAct]
    Critic[core/agentic/critic_agent.py<br/>CriticAgent]

    Registry[core/tools/registry.py<br/>ToolRegistry]
    Tools[core/tools/*<br/>time/notes/file_io/web_search]
    TaskMgr[core/tasks/manager.py<br/>TaskManager]
    Memory[core/memory/service.py<br/>MemoryService]

    Out[Emotion/Translate/TTS<br/>stream output]

    Client --> WS --> Pipe --> Orch
    Orch --> Memory
    Orch --> Chat
    Orch --> TaskMgr
    Orch --> Planner
    Orch --> Runner
    Runner --> Executor
    Runner --> Critic
    Executor --> Registry --> Tools
    Pipe --> Out --> Client
```

## 2. 单轮请求时序图

```mermaid
sequenceDiagram
    participant C as Client
    participant S as service.pet.main
    participant O as Orchestrator
    participant M as MemoryService
    participant TM as TaskManager
    participant R as LangGraphTaskRunner
    participant A as Agents(chat/planner/executor/critic)
    participant T as ToolRegistry/Tools

    C->>S: WS message {content}
    S->>O: handle_user_message(user_text, session_id)
    O->>M: get_recent_history + build_context

    alt waiting_task exists
        O->>TM: resume_waiting_task(task_id, user_reply)
        O->>R: run(resume_plan/snapshot/payload)
        R->>A: planner/executor/critic
        A->>T: function calling (optional)
        T-->>A: tool result
        R-->>O: task snapshot + first_error + waiting
    else classify intent
        O->>A: chat_agent.classify_intent
        alt CHAT
            O->>A: chat_agent.reply_chat
        else TASK
            O->>TM: create_task + set RUNNING
            O->>R: run(new task)
            R->>A: planner -> executor -> critic
            A->>T: tools (optional)
            T-->>A: tool result
            R-->>O: task run result
        end
    end

    O->>M: ingest_turn (long-term memory)
    O-->>S: OrchestrationResult
    S->>M: record_session_round (short-term history)
    S-->>C: emotion_text / audio_chunk / done
```

## 3. Task 模式内部 DAG 调度图

```mermaid
flowchart TD
    P[plan_task]
    S[select_ready_steps]
    R[run_ready_steps]
    V[review_task]
    F[finalize_task]

    P --> S
    S -->|has ready batch| R
    S -->|graph finished or stalled| V
    S -->|waiting_user_input detected| F
    R -->|next round| S
    R -->|fail_fast/finished| V
    R -->|waiting_user_input| F
    V --> F
```

步骤级状态（节点 state）：
- `pending -> ready -> running -> succeeded`
- 异常分支：`failed / blocked`
- 缺信息分支：`waiting_user_input`（等待用户补充后恢复同一 `task_id`）

## 4. 收敛循环（Orchestrator 层）

```mermaid
flowchart TD
    A[run_task_mode once]
    B{waiting_for_input?}
    C{first_error retryable?}
    D{replan_used < max_replan_rounds?}
    E[compose replan user text<br/>reset_task_for_replan + set RUNNING]
    F[return task_run]
    G[mark TASK_NOT_CONVERGED<br/>set FAILED]

    A --> B
    B -->|yes| F
    B -->|no| C
    C -->|no| F
    C -->|yes| D
    D -->|yes| E --> A
    D -->|no| G --> F
```

说明：
- 该循环是“有界持续推进”，不是无限自旋。
- 另外还有 `max_clarify_rounds`：waiting 追问轮次超限会转 `TASK_NOT_CONVERGED`。

## 5. 代码映射（快速定位）

| 职责 | 关键入口 |
|---|---|
| WS 接入与回合处理 | `service/pet/main.py` -> `websocket_handler` / `handle_bot_reply` |
| 编排主入口 | `core/orchestrator/orchestrator.py` -> `handle_user_message` |
| 任务图调度 | `core/orchestrator/langgraph_task_runner.py` -> `run` |
| 状态持久化与快捷指令 | `core/tasks/manager.py` |
| 工具调用协议 | `core/tools/base.py` / `core/tools/registry.py` |
| 记忆上下文与沉淀 | `core/memory/service.py` |
