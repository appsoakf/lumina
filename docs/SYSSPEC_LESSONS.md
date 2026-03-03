# SYSSPEC 论文学习笔记：用形式化方法规范 AI 编程

本文档基于 `specfs.pdf`（FAST 2026，"Sharpen the Spec, Cut the Code"）提炼，目标是把论文中的方法转为本仓库可执行的工程规范。

## 1. 论文核心结论（提炼）

1. 仅用自然语言给 LLM 下达复杂系统开发任务，歧义会迅速放大。
2. 将“提示词”升级为“结构化规格（specification）”后，代码生成正确性和可演化性明显提高。
3. 规格应至少覆盖三类语义：
   - 功能语义（Functionality）
   - 模块组合语义（Modularity）
   - 并发语义（Concurrency）
4. 演化应作用在规格层，而不是直接改实现层；通过 DAG patch 管理依赖传播。
5. LLM 不可靠是常态，必须配套“生成-校验-反馈重试”闭环。

论文中的关键证据（用于校准直觉）：
- Ext4 20 年演化中，约 82.4% 提交是 bug fix + maintenance；新功能提交约 5.1%。
- 仅靠“更大上下文 + 更多代码”并不足够；结构化规格比 oracle-style baseline 更稳定。
- 消融结果表明：功能规格 + 模块规格能解决大部分非并发模块；线程安全模块还需要并发规格 + validator。

## 2. 可迁移的方法论

### 2.1 规格三分法（必须同时存在）

1. 功能规格（Hoare 风格）
- Pre-condition：输入和前置状态
- Post-condition：输出/状态变化/错误分支
- Invariants：跨函数保持为真的约束
- Algorithm/Intent：关键实现策略（尤其性能和复杂分支）

2. 模块规格（Rely/Guarantee）
- Rely：依赖模块向我保证什么
- Guarantee：我对外保证什么
- 要求：依赖关系可组合、可替换、可局部重生成

3. 并发规格（独立于功能规格）
- 不把并发细节混在功能描述里
- 先生成/实现顺序逻辑，再进行并发约束加固（锁、原子性、顺序）

### 2.2 演化机制：DAG 规格补丁

- Leaf：自包含变更（通常局部模块）
- Intermediate：消费子节点 guarantee，逐层集成
- Root：对外 guarantee 语义保持兼容（可替换旧实现）

工程意义：
- 改动影响可见化
- 避免“只改一点点，实际破坏全局”
- 支持增量重生成/重构，而非全量重写

### 2.3 生成闭环：两阶段 + 反馈重试

- Phase 1：功能正确（顺序语义）
- Phase 2：并发正确（加锁/原子性/顺序）
- Validator：
  - 规格一致性检查（SpecEval）
  - 传统回归测试（单测/集成测试）
- 失败必须回写具体反馈，不接受“重新试一次”式盲重试

## 3. 面向 Lumina 的落地规范

结合当前仓库架构（`service -> orchestrator -> agentic/llm/memory/tasks`），建议把每次需求都转成“规格卡（Spec Card）”。

### 3.1 Spec Card 模板（建议直接复制使用）

```text
[Feature]
一句话描述目标与边界。

[Functionality]
Pre:
- ...
Post:
- success: ...
- fail: ...
Invariants:
- ...
Algorithm/Intent:
- ...

[Modularity: Rely/Guarantee]
Rely:
- 依赖模块A保证 ...
- 依赖模块B保证 ...
Guarantee:
- 本模块对外保证 ...
- 错误码/状态语义 ...

[Concurrency]
Pre-lock state:
- ...
Post-lock state:
- ...
Atomicity/ordering:
- ...

[Patch DAG]
Leaf:
- ...
Intermediate:
- ...
Root:
- ...（外部语义保持兼容）

[Validation]
- 单测: ...
- 回归: ...
- 手工场景: ...
```

### 3.2 本仓库优先固化的“全局不变量”

1. 任务状态机不变量（`core/tasks`）
- 不允许非法迁移（如 `succeeded -> running`）
- `/cancel` 优先取消 running，其次 pending
- `/retry` 仅允许 failed/cancelled，且重试目标是最近可重试任务

2. 接口边界不变量（`service` 与 `orchestrator`）
- `service` 只调用 orchestrator 公共接口
- 禁止 `service` 直接触达 orchestrator 内部依赖

3. 记忆一致性不变量（`core/memory`）
- TTL、去重窗口语义稳定
- 失败降级路径可预期（向量链路异常时回退关键词检索）

4. 追踪与可观测性不变量
- 关键路径有 trace
- 错误分支不丢上下文

### 3.3 并发规格在本项目中的写法建议

对异步/并发路径（WS 会话、任务执行、记忆异步入库）单列并发规格，不与业务语义混写：

- 会话级互斥对象是什么（session_id 锁、队列或事件循环约束）
- 哪些操作必须原子（任务状态迁移、重试目标选择）
- 哪些顺序必须保证（先落盘再回包/先取消再回执）
- 失败后资源释放条件（锁、句柄、后台任务）

## 4. 推荐开发流程（以后默认执行）

1. 先写 Spec Card，再动代码。
2. 先实现/验证功能语义，再加并发语义。
3. 变更跨模块时先画 Patch DAG，按 DAG 顺序实现。
4. 每轮改动执行最小验证：
   - `python -m unittest tests.test_task_flow_regression -v`
   - `python scripts/health_check.py --skip-network`
5. 失败时记录“哪条规格被违反”，再修复；避免无依据重试。

## 5. 对 AI 编程实践的启示

1. 从“prompt engineering”升级到“spec engineering”。
2. 规格的价值不只在正确性，也在演化成本控制。
3. 并发语义必须显式化，否则模型会在复杂分支中遗漏锁/顺序细节。
4. 模块化不是只拆文件，而是要有可校验的 Rely/Guarantee 合约。
5. LLM 最好用于“受约束生成”，而不是“自由发挥生成”。

## 6. 使用建议（团队协作）

- 新需求评审时，先评审 Spec Card，再评审代码。
- Code Review 增加一项：实现是否逐条满足规格。
- 缺陷复盘时，同时修规格与实现；优先找“规格遗漏/歧义”。
- 对高风险改动（状态机、协议、并发）强制要求 Patch DAG 与回归证据。

---

简述：这篇论文最重要的启发不是“让 LLM 写更多代码”，而是“先把系统意图形式化，再让 LLM 在约束内实现并持续演化”。
