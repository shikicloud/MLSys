---
title: "Agent Lightning：用强化学习训练任何 AI Agent"
category: agentic-rl
tags: [agent-lightning, microsoft, training-agent-disaggregation, lightningrl, opentelemetry, langchain, autogen, openai-agents-sdk, paper-review]
created: 2026-06-02
updated: 2026-06-02
status: mature
paper: arXiv:2508.03680
code: https://github.com/microsoft/agent-lightning
---

# Agent Lightning：用强化学习训练任何 AI Agent

> [!info] 论文元数据
> - **论文**：[arXiv:2508.03680](https://arxiv.org/abs/2508.03680) — *Agent Lightning: Train ANY AI Agents with Reinforcement Learning*，2025-08-05
> - **代码**：[microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) — Apache-2.0, v0.3.0（2025 年 12 月），17.3k stars
> - **作者**：Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, Yuqing Yang
> - **机构**：Microsoft Research
> - **联系**：agent-lightning@microsoft.com

> [!important] 在 agentic-RL 栈中的位置
> Agent Lightning 是**第一个公开发表**的（2025 年 8 月）将 agent 执行与 RL 训练完全解耦的框架。它比 [[prorl-agent|ProRL Agent]]（2026 年 3 月）早 7 个月，比 [[polar|Polar]]（2026 年 5 月）早 9 个月。它开创的架构模式 ——*agent 跑在 client；LLM 在 server 上通过 OpenAI 兼容 API 暴露；trajectory 以 transition 形式回传*—— 成为后续 NVIDIA 两篇论文延续的模板。任何调研 agentic-RL 基础设施的人都应该先读这篇。

---

## 摘要（2 分钟读完）

**它是什么。** Agent Lightning 是微软研究院的框架，用于**几乎零代码改动**地对任何基于 LLM 的 agent 做 RL 训练。它支持基于 LangChain、OpenAI Agents SDK、AutoGen 或从零开始用 Python 写的 agent —— 框架把你的 agent 包装在一个 "Lightning Client" 里，通过 OpenTelemetry 捕获每次 LLM 调用，把得到的 `(input, output, reward)` transition 发送到与 RL 训练器（默认：veRL）共置的 "Lightning Server"，并更新你 agent 所调用的 endpoint 背后的 policy LLM。

**核心 idea。** **把 agent 执行 cast 成一个 MDP，其中 "一个 action = 一次 LLM 调用产生的 token 序列"**，然后把这些 action 加上它们的上下文收集成一个扁平的 transition 列表。三个子部件支撑这个 idea：

1. **统一数据接口**：每个 agent 的轨迹归约成 `{(input^x,k_i, output^x,k_i, reward^x,k_i)}`，无论 agent 是用 LangChain / AutoGen / OpenAI SDK / 自定义实现。Tool call 等非 LLM 组件更新状态但不是 training transition。
2. **LightningRL 分层 credit assignment**：episode 级别 return → per-action value（当前实现 = 都等于最终 return；future-work 支持学习 value head）→ per-token advantage。绕开了其他所有多轮 RL 框架使用的"sequence 拼接 + masking"做法。
3. **Training-Agent Disaggregation（TA 解耦）**：Lightning Server（管 RL，暴露训练中模型的 OpenAI 兼容 API）↔ Lightning Client（跑 agent，捕获 OpenTelemetry trace，回传 transition）。每个 task 拿到独立的 server 端 endpoint URL。

去掉 MDP 抽象就退回每框架定制；去掉 LightningRL 就退回 context 膨胀和 masking；去掉 TA 解耦就退回紧耦合无法扩展。

**头号实验结果。** Llama-3.2-3B-Instruct base + GRPO，跨三种 agent 框架和三种任务，全部展示稳定的 RL 提升曲线（论文 Fig. 5-7）：

| 任务 | 框架 | 数据集 | reward 曲线 |
| ---- | --- | ----- | ----------: |
| Text-to-SQL | LangChain | Spider | 0.0 → 0.65（训练）/ 0.15 → 0.55（测试）|
| 开放域 QA（RAG）| OpenAI Agents SDK | MuSiQue | 0.05 → 0.40（训练）/ 0.0 → 0.22（测试）|
| 数学 QA（工具使用）| AutoGen | Calc-X | 0.05 → 0.85（训练）/ 0.05 → 0.78（测试）|

重点不是绝对数值（3B 模型小） —— 重点是 **同一个框架 + 同一个训练器优化了用三种不同框架写的 agent，没有任何 per-framework 工程**。

**为什么重要。**

- **首次展示了 trainer-agnostic + agent-agnostic 的真正解耦。** Lightning Server 只跟 trainer 说话；Lightning Client 只跟 agent 说话。任何一边都能换掉不影响另一边。这是 [[polar|Polar]] 后来用 URL-swap 推到极致的架构模式。
- **OpenTelemetry 是它差异化的资产。** 已经用 OpenTelemetry 给 agent 做 observability 的生产团队（Langfuse、Phoenix、AgentOps）"免费"获得 RL 训练 —— trace pipeline 已经存在了。
- **多 agent 选择性优化 work。** Text-to-SQL 实验用 3 个 agent（writer, checker, re-writer）但只训 2 个；LightningRL 自然处理 per-agent transition 的 credit assignment，**不需要 masking**。
- **2027 年预测。** 这种 MDP-with-LLM-action 抽象会成为 agentic RL 论文的标准形式化模型 —— 已经被 ProRL Agent 和 Polar 以不同表述采纳。Agent Lightning 的具体贡献（OpenTelemetry 原生 + 框架多元支持 + 选择性优化）会留下来，因为它跟 harness-proxy 路线**互补而非竞争**。

---

# 详细内容（深入阅读从这里开始）

## 背景：Agent Lightning 之前的 agentic-RL 基础设施为什么不行

2025 年中期，所有多轮 agent-RL 框架都有同样的形态：**trainer + agent 在同一进程**，序列拼接、masking、喂给 PPO/GRPO。论文显式列出了失败模式（§5.2）：

| 失败模式 | 出了什么问题 |
| ------- | ----------- |
| **上下文长度爆炸** | 多轮拼接产生远超 LLM 上下文窗口的序列，尤其是带 MCP tool 交换和多 agent 通信的场景 |
| **位置编码破坏** | RoPE 假设 token 连续性；masking 方法注入的不连续性破坏假设 |
| **框架特定 masking** | 每个 agent 框架 × trainer 组合都需要定制 masking；不能泛化 |
| **紧耦合** | Agent 代码住在 trainer 进程内；换 trainer 就要重新实现 agent loop |
| **要求 Ray 专业能力** | 大多数 RL 框架（veRL、ROLL、OpenRLHF）假设用户能跟 Ray 集成，对应用开发者来说门槛过高 |

Agent Lightning 定位对比的先前框架：

| 框架 | 耦合模式 | 多轮处理 |
| ---- | ------- | ------- |
| **OpenRLHF** ([Hu et al., 2024](https://arxiv.org/abs/2405.11143)) | Agent loop 嵌入 trainer | Sequence 拼接 + masking |
| **TRL** ([von Werra et al., 2020](https://github.com/huggingface/trl)) | 同上 | 同上 |
| **veRL** ([Sheng et al., 2024](https://arxiv.org/abs/2409.19256)) | 同上 | 同上 |
| **AReaL** ([Fu et al., 2025](https://arxiv.org/abs/2505.24298)) | 同上 | 同上 |
| **ROLL** ([Wang et al., 2025a](https://arxiv.org/abs/2506.06122)) | 同上 | 同上 |
| **Search-R1** ([Jin et al., 2025](https://arxiv.org/abs/2503.09516)) | 应用特定 | 检索 token loss masking（[[search-r1]]）|
| **rLLM** ([Tan et al., 2025](https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31)) | 应用特定 | Per-task masking |
| **DeepSWE** ([Luo et al., 2025](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33)) | SWE 特定 | SWE 特定 |
| **Agent Lightning（本工作）** | **通过 TA 解耦** | **Per-transition；无 masking** |

论文对贡献的原话表述（§1）：

> "Agent Lightning is the first framework to achieve a full decoupling between agents and RL training. This decoupling enables Agent Lightning to be seamlessly applied to ANY AI agent regardless of implementation approach, with almost ZERO code modifications."

## 三个核心组件详解

框架有三个承重部件。其它都是 plumbing。

### 组件 1 — 统一数据接口 + MDP 形式化

第一步是数学动作：把 agent 执行 cast 成一个 POMDP。完整的 tuple 是 $\mathcal{M} = (\mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{T}, \mathcal{R})$：

- **$\mathcal{S}$** —— 状态空间；一个状态 = agent 执行上下文中所有 *语义变量* 的快照
- **$\mathcal{O}$** —— 观察空间；policy LLM 实际看到的 input（只有对它可见的语义变量）
- **$\mathcal{A}$** —— 动作空间；**一个 action = 一次 LLM 调用生成的整个 token 序列**（不是一个 token）
- **$\mathcal{T}(s'|s,a)$** —— 转移动力学，通常未知
- **$\mathcal{R}(s,a)$** —— 标量 reward 函数

任务 $x$ 的第 $k$ 次执行在 timestep $t$ 的状态包含 $V$ 个语义变量：
$$\text{state}_t(x,k) = \{\text{variable}_i^{x,k,t}\}_{i=1}^V$$

每次组件调用（LLM 或 tool）是一个 `call`：
$$\text{call}_i^{x,k} = (\text{meta}_i^{x,k}, \text{input}_i^{x,k}, \text{output}_i^{x,k}), \quad \text{output}_i^{x,k} = C_i(\text{input}_i^{x,k})$$

其中 $C_i \in \mathcal{M} \cup \mathcal{T}$ 要么是 LLM 要么是 tool。关键点：**input 和 output 本身就是特定 timestep 的语义变量**：
$$\text{input}_i^{x,k} \in \text{state}_{t_1}(x,k), \quad \text{output}_i^{x,k} \in \text{state}_{t_2}(x,k)$$

RL 数据提取时，只有 LLM 调用变成 transition（Eq. 6）：
$$\text{execution}^{RL}(x,k) = \{(\text{input}_i^{x,k}, \text{output}_i^{x,k}, r_i^{x,k})\}_{i=1}^T$$

Tool 调用修改状态但不产生梯度 —— 因果上重要但不可训练。

**示例：RAG agent。** 论文 Fig. 2 通过状态更新追踪一个 4 步 RAG agent：

![统一数据接口：4 步 RAG agent 的状态演化和轨迹提取（论文 Fig. 2）](CN/wiki/agentic-rl/agent-lightning-figs/unified-data-interface-rag.png)


```
state_0: {UserInput=U, Query=NA, Passages=NA, Answer=NA}
state_1: Q = LLM(U)             → {U, Q, NA, NA}
state_2: P = Search(Q)          → {U, Q, P,  NA}
state_3: A = LLM(U, P)          → {U, Q, P,  A }
```

提取出的轨迹**只包含两次 LLM 调用**，reward 在最终 answer 上：

```
trajectory = [
  (input=U,    output=Q, reward=NA),
  (input=(U,P), output=A, reward=R)
]
```

Search 工具对状态的贡献被保留（P 成为第二次 LLM 调用 input 的一部分），但 search 调用本身不出现作为训练样本。**这是文献里 "agentic RL 里什么可训练 vs 什么是环境" 最干净的形式化**。

> [!quote] 这种抽象的架构红利
> "The unified data interface captures all state changes, including those caused by non-LLM components, enabling flexible and highly customizable optimization methods. ... It accommodates arbitrary and complex agent interaction logic **without requiring explicit parsing of the entire execution DAG**, thereby greatly simplifying RL-based optimization for diverse agent workflows."

短语 *"without requiring explicit parsing of the entire execution DAG"* 是承重的 claim：框架**不需要知道**你 agent 的控制流、依赖关系或子 agent 结构。它只收集 LLM 调用及其 input。

### 组件 2 — LightningRL 分层 credit assignment

标准 token 级 RL loss（Eq. 8）：
$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{X}, \text{output} \sim \pi_\theta(\cdot|x)} \left[ \sum_{j=1}^N \log \pi_\theta(y_j | x, y_{<j}) \cdot A_j \right]$$

多轮 agentic RL 的挑战：每个 *action* 是一个多 token LLM 响应，但 reward 通常只在轨迹末尾到来。LightningRL 是把它们连起来的 credit-assignment 胶水。

**两步 credit assignment**（§3.3.2）：

1. **第一步 —— Episode return → per-action value。** Credit assignment 模块把 episode 级 return $R$ 分配到 episode 内每个 action $a_t$。

   *当前实现*：identity 分配 —— 每个 action 拿同一个值，等于最终 return。
   
   *论文承认的局限*：这是占位符。论文说 future work 会引入一个学习的 high-level value function 做 per-action advantage 估计。

2. **第二步 —— Per-action value → per-token token 级 loss。** 在每个 action 内部（= 一次 LLM 响应），用标准 PPO / GRPO / REINFORCE++ token 级 loss，把该 action 分到的 value 当 advantage。**这是单轮 RL 机器原样使用**。

**为什么这个重要：它完全绕开了 masking 问题。** 论文 Fig. 3 对比三种做法：

![LightningRL vs Single-call GRPO 和之前的多轮 GRPO（论文 Fig. 3）](CN/wiki/agentic-rl/agent-lightning-figs/lightningrl-vs-grpo.png)


- **(a) Single-call GRPO**：同任务 response 分组估 advantage —— 对单轮 work 但对多轮不 work。
- **(b) 之前的多轮 GRPO**：整条轨迹拼接，非 LLM token 被 mask 掉。破坏 RoPE，让 kernel 复杂，框架特定。
- **(c) LightningRL**：轨迹**分解成独立 transition**；同任务 transition 分组估 advantage。**不需要 masking**。原生兼容单轮 RL 基础设施。

论文原话（§3.3.2）：

> "Third, compared to the masking strategy, LightningRL offers a more robust and scalable implementation. Masking-based approaches not only require tight coupling between training and agent execution logic, but also disrupt the continuity of tokens in LLMs, which is assumed in the widely used position encoding approaches, such as Rotary Positional Embeddings (RoPE)."

**多 agent 扩展。** 因为每个 transition 都是独立的，并带有来源 agent 的标签，你可以**只把目标子集 agent 的 transition 加入梯度更新**来选择性优化多 agent 系统。Text-to-SQL 实验就是这么做的：3 个 agent（SQL writer、checker、re-writer），训 2 个，冻 1 个 —— 完全不需要 masking。

### 组件 3 — Training-Agent Disaggregation（TA 解耦）

系统架构有两个进程，通过 HTTP 通信（论文 Fig. 4）：

![Training-Agent Disaggregation 架构（论文 Fig. 4）](CN/wiki/agentic-rl/agent-lightning-figs/training-agent-disaggregation.png)

**Lightning Server**（与 RL 框架共置在 GPU 服务器上）：
- 编排 RL 训练循环
- 管理 task、batch、通信
- **暴露训练中模型的 OpenAI 兼容 API**（Pydantic 类型化）
- Inventory-style task dispatch：维护可用 task 的清单；client ready 时分配给它
- **每个 task 拿到独立的 OpenAI 兼容 endpoint URL**，跟 task 一起传给 client

**Lightning Client**（不需要 GPU；可以跑在任何地方）：
- 两个子组件：
  - 通信模块：跟 server 的数据上传/下载
  - Runtime：执行 agent 并收集 trace
- 通过 task 特定的 API endpoint 连到 server
- 把带 reward 的 trace 回传

**事件驱动的控制流**（论文 Appendix B 的时序图）：

```
1. Client 把 dataset 上传到 Server
2. Server 启动 RL framework，加载初始 model
3. for each batch in dataset:
     for each task in batch:
       Server → Client：(task, task 特定的 API URL)
       Client 启动 agent 执行
       for each LLM call in agent run:
         Agent → API URL → Server（转发到 vLLM/SGLang）→ Agent
       Client 收集 trace + reward → Server
     Server → RL Framework：trajectory batch
     RL Framework 更新权重 → Server 发布新 model
```

互相独立是关键属性：

> [!quote] Trainer 和 agent 的互相独立（论文 §3.4.1）
> "This design renders the training framework (e.g., VeRL) **agent-agnostic**; its sole concern is the optimization of the LLM and management of hardware resources. Conversely, the agent, operating on the client side, is **trainer-agnostic**, allowing it to function independently of the training framework's implementation details."

**Agent 执行的 Data Parallelism** 是两层策略：

- **节点内并行**：一个 Client 进程在单机上 spawn 多个 worker 子进程，每个跑一个 agent 实例。
- **节点间并行**：多个 Client 进程在不同机器上，各自有自己的 worker。

这让 rollout 吞吐能扩展到 agent 执行 + tool I/O 能处理的极限，跟训练集群规模独立。

### 辅助机制（可以略过）

> [!note]- Automatic Intermediate Rewarding（AIR）—— 解决 agent 的稀疏 reward
>
> 长 horizon agent 任务面临稀疏 reward（只有最终任务成功或失败）。AIR 是 Agent Lightning 的机制，**把系统监控信号转成 intermediate reward**，不需要人工标注。
>
> 可用的具体信号：
> - Tool 调用返回状态（成功 / 失败 / 超时）
> - 代码执行 exit code
> - 编译错误
> - 网络响应码
>
> Client 把这些信号映射成 per-transition reward 分量，跟最终 outcome reward 相加。开发者按任务类型自定义映射。
>
> 论文承认这是"对稀疏 reward 问题的训练侧 workaround" —— 更干净的方案（基于 verifier 的 PRM）被定位为 future work。

> [!note]- 不改代码捕获数据 —— 两条路径
>
> Client 提供的两种 instrumentation 方式：
>
> 1. **OpenTelemetry + AgentOps**（推荐）。Client 用 OpenTelemetry 的 tracing 能力自动 instrument agent 代码，捕获执行 trace、LLM 调用、环境交互。**兼容生产 observability stack**（Langfuse、Phoenix、Datadog APM、AWS X-Ray 等）—— 任何已经用 OpenTelemetry 做 agent observability 的团队**免费**获得 RL 数据收集。
> 2. **OpenAI 兼容 API endpoint 内嵌 tracing**（fallback）。对无法用 OpenTelemetry instrument 的 agent，Client 在 API endpoint 里提供一个基础捕获机制。轻量；不依赖外部组件就能 work。
>
> OpenTelemetry 路径是更有意思的设计选择 —— 它把 Agent Lightning 定位成跟既有 agent observability 生态集成，而这个生态跟 RL 训练基础设施是独立成熟起来的。

> [!note]- 错误处理与鲁棒性
>
> Client 为 RL 特有的失败模式实现了完整错误处理（比纯推理失败更频繁，因为有 exploration）：
>
> - Agent 代码 crash —— 检测、记录、task 重试或重分给其它 worker
> - 网络中断 —— Tool 调用带 timeout 和 retry
> - Long-hanging tool 调用 —— 每个 task 有 wall-clock budget
> - 无效输出 —— 记录用于 debug，transition 丢弃
>
> 失败的 task 不打断整体训练；设计假设失败常见，通过隔离 per-task 状态在工程上绕过。

> [!note]- 用于扩展的 Environment 和 Reward Services
>
> 环境和 reward 函数代价不同。Client 支持两种部署模式：
>
> 1. **In-worker** —— 轻量环境（如计算器、小 SQL DB）在 agent 同 worker 里跑。
> 2. **Pooled service** —— 重型环境（手机模拟器、复杂 reward 计算）作为共享服务托管，所有 worker 连过来。
>
> Pooled service 摊销初始化成本（如把 10GB reward model 加载一次 vs per-worker）并减少内存压力。论文提到这可以扩展到"更复杂的 serverless 架构"但当前用简单池化设计。

## 头号实验证据

**实验设置。** Llama-3.2-3B-Instruct base model。RL 算法没有显式说是 PPO 还是 GRPO，但论文引用了两者并在 LightningRL 框架内讨论 GRPO advantage normalization。三个任务，三个不同 agent 框架：

| 任务 | 框架 | 数据集 | 工具 | Agent 总数 | 训练 agent 数 |
| ---- | --- | ----- | --- | ---------: | ----------: |
| Text-to-SQL | LangChain | Spider | SQL 执行器 | 3 | **2** |
| 开放域 QA | OpenAI Agents SDK | MuSiQue | Wikipedia 检索器 | 1 | 1 |
| 数学 QA | AutoGen | Calc-X | 计算器 | 1 | 1 |

**主要结果**（从 Fig. 5-7 提取）：

| 任务 | 训练 reward（起 → 终）| 测试 reward（起 → 终）|
| ---- | --------------------: | --------------------: |
| **Text-to-SQL（Spider）** | 0.0 → 0.65 | 0.15 → 0.55 |
| **MuSiQue（RAG）** | 0.05 → 0.40 | 0.0 → 0.22 |
| **Calc-X（数学 QA）** | 0.05 → 0.85 | 0.05 → 0.78 |

> [!success] 实验真正展示的是什么
> 数字本身不大（3B 模型，学习友好的任务）。贡献在结构：**同一个 RL pipeline 训练了三个用三种不同框架写的 agent，没有任何 per-framework 工程**。这是头号信息。

**关键"ablation"藏在多 agent 设置里。** Text-to-SQL pipeline 有 3 个 agent 但只优化 2 个 —— 第三个保持不变。LightningRL 处理这种情况只需要**简单地不在梯度更新里包含第三个 agent 的 transition**。用 masking 方法的话，这需要为每种多 agent 拓扑精心构造 mask；这里则是 free 的。

> [!example]- 全部实验结果 —— 完整细节
>
> **Spider 上的 Text-to-SQL**（§4.1）
>
> Agent 用 LangChain 实现 3 个子 agent：SQL writer、SQL checker、re-writer。SQL writer 生成 query；如果执行失败，checker 决定是 rewrite 还是直接回答；re-writer 修正 query 或基于检索信息回答。**只训练 writer 和 re-writer**；checker 用同一个 LLM 但冻住。Reward = 最终答案是否正确。
>
> Llama-3.2-3B-Instruct 起步接近零 reward（语法错误占主导），在 400 步内稳定爬升到 ~0.65 训练 / ~0.55 测试。测试曲线在 step 200 左右 plateau（早期收敛）。
>
> **OpenAI Agents SDK 上的 MuSiQue（RAG）**（§4.2）
>
> Wikipedia 上的多 hop QA（2100 万文档，BGE embedding，cosine 检索）。Agent 生成搜索 query，根据检索 passage 决定是 refine 还是回答。Reward = $0.9 \cdot R_{\text{correctness}} + 0.1 \cdot R_{\text{format}}$，其中 format reward 要求 `<think>...</think>`、`<query>...</query>`、`<answer>...</answer>` 结构。
>
> 测试 reward 在 200 步内从 ~0 爬到 ~0.22。MuSiQue 确实很难（multi-hop、大检索语料、自由形式 query），3B 模型 undersized；绝对数字不是重点。
>
> **AutoGen 上的 Calc-X（数学 QA）**（§4.3）
>
> 需要调用计算器的数学题。单 LLM agent 决定何时调用计算器并整合结果。Reward = 答案正确性。
>
> 最强结果：训练 ~0.85，测试 ~0.78，450 步内。工具使用 + 确定性验证是最简单的 RL setting，曲线反映了这点。

## 优势与局限

**优势。**

- **MDP 形式化是文献里最干净的。** 把一次 LLM 调用 cast 成一个 action，状态用语义变量表示，让数学跟标准单轮 RL 方法自然吻合。
- **OpenTelemetry 原生**是真差异化 —— 生产 agent 团队可以复用自己的 observability stack。
- **Trainer-agnostic + agent-agnostic** 是属性而不是口号；架构通过 OpenAI API 边界真正强制了这点。
- **多 agent 选择性优化**从抽象中自然得到，不需要 bespoke masking。

**局限。**

> [!warning] Credit assignment 故事不完整
> 论文对此诚实（§3.3.2）：当前 LightningRL 实现给 episode 里每个 action 同一个 value = 最终 return。对中间动作重要性差异大的长 horizon agent，这把 exploration 跟 execution 混淆，**邀请了类似 credit-misattribution 的 reward hacking** —— 跟 [[polar|Polar]] 的 `per_request` 模式遇到的同样失败模式。学习 value head 被点名为 future work。

- **只在 3B 上测过。** Llama-3.2-3B-Instruct 小。扩展到 7B / 14B / 70B 没展示；模型规模 scaling 行为未知。
- **任务选择对 RL 友好。** Spider / MuSiQue / Calc-X 都有自动正确性检查。没有 SWE-Bench、GAIA、OSWorld。"任何 AI agent" 的 claim 技术上成立但落在难度谱的简单端。
- **"几乎零代码改动" 需要把 agent function 包装在 `Client` 里。** 实际 overhead 很小（~5 行，见下面 Listing 2）但非零。完全不能改代码的 agent（如闭源二进制如 Claude Code）**没法训** —— 这是 [[polar|Polar]] 后来用 URL-swap 填补的 gap。
- **没有跟其它 agent-RL 框架的 head-to-head 对比。** 论文在 related work 部分对比 SkyRL / VeRL-Tool / rLLM / GEM 但没在同任务上跑它们。
- **多 LLM 多 agent setting 只简短讨论**（§3.2.2 提到 MARL 扩展）；实验是单 LLM 多角色。

## 这意味着什么

Agent Lightning 是**正确命名了问题的论文**。在它之前，每个多轮 RL 框架都是"在 trainer 内重建 agent" 的特例；在它之后，解耦的 client-server 模式成为明显的目标。ProRL Agent（2026 年 3 月）采用同样的架构原则，把 rollout 侧打磨成生产级服务。Polar（2026 年 5 月）进一步把 Python API 集成边界替换为 HTTP API proxy 边界，移除了最后的代码改动要求。

2027 年的三个预测：

1. **Agent Lightning 的 MDP-with-LLM-call-as-action 成为标准形式化。** ProRL Agent 和 Polar 已经用本质上同样的抽象（不同表面呈现）。数学太干净不会被丢弃。
2. **OpenTelemetry 集成成为入场券。** 生产 agent 团队不会接受不说他们已经在用的 observability 协议的 RL 框架。Polar / ProRL Agent 很可能加 OpenTelemetry 支持；veRL / NeMo-RL 会需要适配器。
3. **LightningRL credit-assignment 占位符成为瓶颈。** Identity 分配对短轨迹 work 但在长 horizon SWE-Bench 风格任务上失败。谁把鲁棒的 per-action value head + token 级 GAE 落到 LightningRL 的 transition 格式之上，谁就发出下一篇明显的 paper。

Agent Lightning **不**解决的：

- **闭源 / 黑箱 agent 训练**：需要 Python API 包装。Polar 的 URL-swap 是答案。
- **跨重 tokenization 的严格 token 一致性**：trace 捕获用 text 级 wire 格式，所以训练的 LLM 的 tokenizer 必须跟 server 侧推理引擎产生的匹配。ProRL Agent 的 token-in/token-out wire protocol 解决这点；Polar 泛化了它。
- **生产规模实证验证**：3B + Calc-X 是研究规模。Polar 在 Qwen3.5-4B/8B/14B 上的 SWE-Bench-Verified 数字是这个领域需要的生产规模 demonstration。

## 源代码与复现

```bash
# 安装
pip install agentlightning

# 或者 nightly 从 Test PyPI
pip install --index-url https://test.pypi.org/simple/ agentlightning
```

**最小端到端示例**（论文 Appendix A，Listing 2）：

```python
# train.py
from agent import agent_function
from environments.game_server import GameServer
from agent_lightning import Client, Resource, Task

game_server = GameServer()

# Task 对应 .jsonl 里的数据；Resource 包含 LLM endpoint
def agent_run(resource: Resource, task: Task):
    game_url = game_server.create_game(task.game_seed)
    answer = agent_function(resource.model_api, game_url)
    reward = game_url.score(answer, task.ground_truth)
    return reward

client = Client(os.environ["AgentLightningServerUrl"])
client.upload_data("data/train.jsonl", test_file=None)
client.train(agent_run, nworkers=-1)
```

原 agent 代码（`agent.py`）不动；新文件（`train.py`）把 `agent_function` 包装到 Client 接口。

**Agent Lightning 项目的 folder 结构**（Listing 1）：

```
agent/
├── data/train.jsonl              # Task 数据集
├── environments/game_server.py   # Tool / env 服务器
└── agent.py                      # 原 agent（不改）
```

| 文件路径 | 角色 |
| ------- | ---- |
| `agentlightning/client.py` | `Client` 类 —— 用户接触的 wrapper |
| `agentlightning/server.py` | `Server` 类 —— 编排 RL framework |
| `agentlightning/runtime.py` | Agent 执行 + trace 捕获 |
| `agentlightning/algorithms/` | LightningRL 实现 + GRPO/PPO 适配 |
| `dashboard/` | 监控 UI（TypeScript，实时 trace 检查）|
| `examples/` | 参考集成：LangChain、AutoGen、OpenAI SDK |

**社区验证的外部集成**：

- **Youtu-Agent** —— 通过修改 branch 验证到 128-GPU RL 训练
- **DeepWerewolf** —— 用 AgentScope + Agent Lightning 做的多 agent 狼人杀游戏
- **vLLM 博客（2025 年 10 月）** —— *"No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL"* 讨论 Agent Lightning 跟 vLLM 的集成，解决 tokenizer mismatch 失败模式

## 相关阅读

- [[prorl-agent]] —— ProRL Agent：NVIDIA 2026 年 3 月的框架，基于 Agent Lightning 的解耦原则，加上打磨过的生产级 rollout 服务、rootless Singularity sandbox 和 Slurm 原生部署。同一架构家族。
- [[polar]] —— Polar：2026 年 5 月 ProRL Agent 的后继，通过 LLM API proxy + URL swap 把 Agent Lightning 的"几乎零代码改动"推到**"真正零代码改动"**。能训 Agent Lightning 够不着的闭源 agent（Codex、Claude Code）。
- [[search-r1]] —— Search-R1：2025 年早期标杆 agentic-RL 论文。应用特定（检索用 tool use），用 sequence 拼接 + retrieved-token loss masking —— 正是 Agent Lightning 反对的模式。
- [[nemo-gym]] —— NVIDIA 的 RL 环境 catalog；跟 Agent Lightning 互补而非竞争（环境层 vs rollout-driver 层）。
- [[agentic-rl-foundations]] —— Agentic RL 入门 hub；在更广 landscape 里定位 Agent Lightning。
- [[on-policy-distillation]] —— 替代范式：dense teacher KL 完全替代 RL。互斥方向。
- [[grpo]]、[[ppo-for-llm]] —— LightningRL per-action loss 插入的单轮 RL 算法。
- [[rl-training-frameworks]] —— 上下文：Agent Lightning 在 veRL / OpenRLHF / TRL / ROLL 里的位置。

## 参考文献

- Xufang Luo et al. *Agent Lightning: Train ANY AI Agents with Reinforcement Learning.* arXiv:2508.03680, August 2025. https://arxiv.org/abs/2508.03680
- microsoft/agent-lightning. https://github.com/microsoft/agent-lightning
- 文档. https://microsoft.github.io/agent-lightning/
- OpenTelemetry. https://opentelemetry.io/
- AgentOps. https://www.agentops.ai/
- vLLM 博客：*No More Retokenization Drift.* https://vllm.ai/blog/2025-10-22-agent-lightning-vllm（2025 年 10 月 22 日）
