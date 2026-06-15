---
title: "ThunderAgent：简单、快速、Program-Aware 的 Agentic 推理系统"
category: llm-serving-for-agents
tags: [thunderagent, agent-serving, program-abstraction, kv-cache, scheduling, vllm, sglang, openhands, swe-bench, icml-2026, paper-review]
created: 2026-06-15
updated: 2026-06-15
status: mature
paper: arXiv:2602.13692
code: https://github.com/ThunderAgent-org/ThunderAgent
---

# ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System

> [!info] 论文元信息
> - **论文**：[arXiv:2602.13692](https://arxiv.org/abs/2602.13692) —— *ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System*，v1 2026-02-14，v2 2026-03-10
> - **会议**：**ICML 2026 Spotlight（top 2.2%）**
> - **代码**：[ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent)（MIT 协议，支持 vLLM + SGLang 后端，OpenAI 兼容 API）
> - **作者**：Hao Kang\*¹（Georgia Tech）、Ziyang Li\*²（Individual Researcher）、Xinyu Yang\*³（CMU）、Weili Xu\*⁴（UIUC）、Yinfang Chen⁴（UIUC）、Junxiong Wang⁵（Together AI）、Beidi Chen³（CMU）、Tushar Krishna¹（Georgia Tech）、Chenfeng Xu⁵（Together AI）、Simran Arora⁵（Together AI）。\* 表共同一作。通讯作者 `hkang342@gatech.edu`。
> - **机构**：Georgia Tech、CMU、UIUC、Together AI

> [!important] ThunderAgent 在 agent-serving 栈中的位置
> ThunderAgent 在 multi-turn agent serving 的版图里**同时既是竞争者也是基石**：
>
> - **vs [[continuum|Continuum]]**：同期竞争者，解同样的 KV cache 管理问题但路径完全不同。Continuum 预测工具时长再用 TTL pin KV；ThunderAgent 把 workflow 抽象成 **agentic program**，周期性监测 thrashing，反应式 Pause/Restore 程序。ThunderAgent **在所有 6 个 benchmark 上都赢 Continuum**（SWEAgent、OpenHands、ToolOrchestra、ScienceAgent，跨 GLM-4.6 + Qwen3 模型），在 Continuum 的 TTL 预测失败的随机工具场景下经常领先 2-3× 倍率。
> - **vs [[mori|MORI]]**：**ThunderAgent 是 MORI 跑在其上的系统层** —— MORI 的 ~3,300 行 Python 实现于 ThunderAgent + 500 行于 SGLang HiCache。MORI 的两层 idleness 调度跟 ThunderAgent 的 program 抽象正交。
> - **vs [[infercept|InferCept]]**：把 InferCept 的"interception 是一等公民事件"论点从 request 级提升到 *workflow* 级。InferCept 用 Discard/Preserve/Swap 在 per-request 管 KV；ThunderAgent 把整个 multi-turn workflow 抽象成 program $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$，按 program 粒度调度。
>
> 如果只读一篇论文理解当下 agent serving 系统的现状，这是 top 3 之一（另两篇是 Continuum 和 MORI）。

---

## 摘要（2 分钟读完）

**它是什么。** ThunderAgent（ICML 2026 Spotlight）是一个 **program-aware 的 agentic 推理与 rollout 库**，坐落在 agent 客户端跟推理后端（vLLM、SGLang）之间，**按 workflow 粒度而非按 request 粒度**调度 KV cache 与工具资源。它暴露 OpenAI 兼容 API，唯一新增的参数是 `program_id`，剩下的事系统全权接管，跟踪每个 multi-turn workflow 当作一等公民"agentic program"的整个生命周期。

**唯一核心思想。** **把 agentic workflow 当作 LLM Program 处理。** 一个 program $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ 把现有系统分开追踪的四样东西统一起来：

1. **$c$** = 当前 context 长度（KV cache 内存占用）
2. **$\mathcal{T}$** = workflow 依赖的工具环境集合（Docker sandbox、网络端口）
3. **$\mathcal{L}$** = 后端 GPU 节点 placement
4. **$\tau, s$** = 执行阶段（$\tau \in \{R, A\}$ = Reasoning / Acting）+ 调度状态（$s \in \{$Active, Paused, Terminated$\}$）

三个子件让这个抽象承重：

1. **Program-aware scheduler** —— 固定周期 $\Delta t = 5$ s 的 thrashing 监视器检测 `C_total < ∑ c_p` 时触发 **Pause**（按 $S_{pause}(P) = 1/c_P + \mathbb{1}(\tau = A)$ 优先驱逐 context 最短的 Acting program）和 **Restore**（按 $S_{restore}(P) = 1/c_P + \mathbb{1}(\tau = R)$ 优先恢复 context 最短的 Reasoning program）。shortest-first 驱逐策略**可证明最优**（Lemma 4.1：recompute 代价是 context 长度的二次方 $c_i^2$）。
2. **跨 backend 的全局 program-aware waiting queue** —— 替代 per-node KV-aware 路由（vLLM/SGLang 这样路由导致 90 分钟内跨节点内存不平衡 51%）。Paused KV 节点无关，所以任何有空闲的 backend 都能 restore 它。
3. **Program-aware 工具资源管理** —— **hook-based 垃圾回收**在 `Terminated` 状态触发立即拆 sandbox 和释放端口（保持磁盘用量近常数 vs 基线线性增长），**异步环境准备**在高优先级 Restoring program 还在 LLM 阶段时就先把它的 Docker 容器 / 包依赖装好（隐藏 29-47 秒的初始化延迟）。

去掉 program 抽象，调度器没法区分"这个 request 在 workflow 中途、马上回来"vs"这是全新 request" —— 退化成 vLLM/SGLang 行为。去掉全局队列就 51% 跨节点不平衡。去掉 hook-based GC，磁盘 ~250 个 workflow 后就溢出。去掉异步环境准备，每个 workflow 启动都要付 47 s 设置时间在关键路径上。

**头条结果。** 8×H100 上 6 个 benchmark（GLM-4.6 355B MoE、Qwen3-235B、Qwen3-8B）：

| Workload | vs vLLM | vs Continuum |
| -------- | ------: | -----------: |
| SWEAgent / GLM-4.6 | **2.65×** | 1.52× |
| OpenHands(code) / GLM-4.6 | **3.58×** | 1.08× |
| ToolOrchestra(HLE) / Qwen3-8B | **1.48×** | 0.65× ← Continuum 在这里**输给** vLLM |
| SWEAgent / Qwen3-235B | **3.02×** | 1.44× |
| OpenHands(code) / Qwen3-235B | **2.43×** | 1.22× |
| ScienceAgent / GLM-4.6 | **1.24×** | 1.06× |

外加：
- **RL rollout：在分布式 GPU 节点上比 prior SOTA 提升 1.79–3.92×**。
- **磁盘内存：节省多达 4.2×**（近常数 vs 线性增长）。
- **KV cache 命中率**：可预测工具的 workload（a, b, d, e）ThunderAgent 维持 ~100%；Continuum 在高并发下跌到 ~60%。

**为什么重要。**

- **第一个把整个 multi-turn workflow 作为调度单位的系统。** vLLM/SGLang/InferCept 都在 request 粒度调度；Continuum 通过 TTL 部分聚合但仍 pin 在一个节点上。ThunderAgent 的 $P$ tuple 是文献里"program 即调度原语"最干净的表述。
- **闭式二次代价结果（Lemma 4.1）一锤定音解决 eviction 问题。** Recompute 代价 $\propto c_i^2$ 意味**永远驱逐 context 最短的 program**。这是个很小但持久的定理，后续每个 agent-serving 调度器都该用上。
- **后续工作的系统层基础。** MORI 的两层 idleness 调度器直接建在 ThunderAgent 上。之后的 agent-serving 系统要么建在 ThunderAgent 的 program 抽象上，要么自己重新发明等价物。
- **在随机工具执行 workload 上决定性击败 Continuum。** Continuum 的 TTL 预测模型在工具延迟重尾时失败（ToolOrchestra panel c —— Continuum 0.65× vs vLLM，即*更慢*）。ThunderAgent 的时间衰减权重 $f(t) = 2^{-t}$ 处理确定性和随机两种场景。
- **2027 预期。** "Program ID" 会成为 vLLM `Request` struct 里的一等字段。Per-program 调度 API 成为标配。Continuum vs ThunderAgent 的辩论（预测式 TTL vs 反应式 thrashing 监视）成为教科书 trade-off 讨论。

---

# 深入（drill-down 从这里开始）

## 背景：现有"request-aware"系统的三种失败模式

现有栈 —— vLLM/SGLang 做推理、Kubernetes/Docker 管工具 —— 按 request 粒度调度。每次 LLM 调用独立、每次工具调用独立。论文 Figure 1 和 Figure 2 量化了这种架构的三个失败。

![ThunderAgent 跟前代 agent 推理系统在并行 workflow 数增加下的对比。(a) Throughput degradation。(b) KV cache thrashing（E2E latency 跟 KV 命中率）。(c) SWE-Agent、OpenHands、ToolOrchestra 上的 speedup。（论文 Fig. 1）](CN/wiki/llm-serving-for-agents/thunderagent-figs/fig1-motivation.png)

### 失败 1：KV cache thrashing —— **7.14× E2E latency 暴涨**

Request-aware 引擎在工具执行间隔里驱逐 KV cache 让新到来的请求腾位置。工具返回后，整个历史必须**重新 prefill**（因为每轮都把上下文 append 到完整历史里）。论文测出在高并发下这种 thrashing 单独导致 **7.14× E2E latency 增加** —— 视觉上 Figure 1b：vLLM 的 KV 命中率从 ~80% 崩到 ~30%（并行 workflow 24 → 96），E2E latency 翻三倍。

这就是 [[continuum|Continuum]] 用 TTL pinning 解决的、[[infercept|InferCept]] 用 Preserve 动作解决的同一个问题 —— 但那些方案假设你能 *预测* 工具时长（Continuum）或 interception 是 per-request 的（InferCept）。真实的 agentic workflow 比如 ToolOrchestra 交错调用很多不可预测时长的工具（网络 API 从毫秒到分钟）。

### 失败 2：跨节点内存不平衡 —— **51% 峰值不平衡**

vLLM 的 KV-aware router 和 SGLang 的 prefix-aware router **都把同前缀的请求贪心路由到同一个 DP 节点**最大化 cache 复用。对 prompt 多样的 chatbot 是合理的；对 agent 是病态的，因为 **每个 workflow 都从相同的 system prompt + tool definitions 开始**。

论文测量（Figure 2a）：90 分钟的 OpenHands RL rollout，2 个 8×H100 节点上，Node 0 跟 Node 1 之间的内存不平衡超过 20% 持续 **37+ 分钟**，峰值达 **51%**。半个 GPU 内存空闲、另一半在 thrash。

### 失败 3：工具生命周期无感 —— **47s 环境准备时间、磁盘线性增长**

工具编排器（Kubernetes、Docker）不知道 *agentic workflow* 什么时候算完成 —— 只知道单次 `docker run` 什么时候返回。结果：

- **磁盘用量随处理过的 workflow 数线性增长**（Figure 2b）：finished workflow 留下的没用的 Docker 镜像和 stopped sandbox 永远不被回收。~250 个 workflow 后磁盘就超 2 TB 容量。
- **环境准备时间从 29.9 s 涨到 47.2 s**（并行 workflow 24 → 96，Figure 2c）。当这成为每个 workflow 启动的关键路径时，吞吐塌方。

## Agentic program 抽象

§4.1 把 **agentic program** $P$ 定义为 tuple：

$$P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle \quad (1)$$

| 符号 | 含义 |
| ---- | ---- |
| $ID$ | 全局唯一标识；客户端在 OpenAI 兼容 API 里以 `program_id` 形式传 |
| $c$ | 当前 context token 数（= KV cache 内存占用） |
| $\mathcal{T}$ | 工具环境集合（Docker sandbox、网络端口、数据库连接） |
| $\mathcal{L}$ | 后端（GPU 节点）placement，给空间局部性用 |
| $\tau$ | 执行阶段：**R**easoning（LLM decoding）或 **A**cting（等工具） |
| $s$ | 调度状态：Active、Paused、Terminated |

两个**原语操作**做状态转移：

- **Restore$(P)$**：$\langle ID, c, \mathcal{T}, \emptyset, \tau, \text{Paused}\rangle \to \langle ID, c, \mathcal{T}, \mathcal{L}', \tau, \text{Active}\rangle$ —— 接纳到有容量的 backend $\mathcal{L}'$。
- **Pause$(P)$**：$\langle ID, c, \mathcal{T}, \mathcal{L}, \tau, \text{Active}\rangle \to \langle ID, c, \mathcal{T}, \emptyset, \tau, \text{Paused}\rangle$ —— 释放 KV cache；工具环境留磁盘上不动，恢复时快。

这个抽象是整套设计的概念枢纽。一旦有了 program-level 的状态向量，cost model（下节）就能被建模成可解的优化问题。

## Cost 模型

§4.2 用 **Space-Time Product（STP）** 作主要 cost 度量：

$$\text{Cost}_x = \int_0^{t_x} M_x(t) \, dt \quad (2)$$

由于 KV 内存 $M_x(t)$ 直接就是 token 数，cost 简化成 **token 数对时间的积分**。Agentic 推理总 cost 分解成五项：

$$\text{Cost}_{\text{total}} \approx \text{Cost}_{\text{decode}} + \text{Cost}_{\text{prefill}} + \text{Cost}_{\text{recompute}} + \text{Cost}_{\text{unused}} + \text{Cost}_{\text{caching}} \quad (3)$$

| 项 | 有效？ | 来源 |
| -- | :----: | ---- |
| $\text{Cost}_{\text{decode}}$ | ✓ | 主动 token 生成 |
| $\text{Cost}_{\text{prefill}}$ | ✓ | 首轮或新上下文 prefill |
| $\text{Cost}_{\text{recompute}}$ | ✗ | KV thrashing → 恢复时重 prefill |
| $\text{Cost}_{\text{unused}}$ | ✗ | DP 节点间内存不平衡 |
| $\text{Cost}_{\text{caching}}$ | ✗ | 长工具调用期间 KV 被 pin |

调度器目标：**最小化 $\text{Cost}_{\text{recompute}} + \text{Cost}_{\text{unused}} + \text{Cost}_{\text{caching}}$**。ThunderAgent 三个组件各打一项。

> [!important] Lemma 4.1（Recompute 的二次代价）
> 对 context 长度 $c_i$ 的 program $P_i$：
>
> $$\text{Cost}_{\text{recompute}} = \int_0^{t_{\text{recompute}}} c_i(t) \, dt \propto c_i^2 \quad (8)$$
>
> 直觉：recompute 时间线性于 $c$，那期间内存占用也是 $c$ → 积分是 $c^2$。
>
> **后果**：要驱逐 program 释放 $\Delta C$ 内存时，最优子集 $S$ 应最小化 $\sum_{i \in S} c_i^2$，约束 $\sum_{i \in S} c_i \geq \Delta C$。**贪心答案：先驱逐 context 最短的 program。** 这是 ThunderAgent shortest-first 驱逐策略的形式化依据。

## 三个组件详解

![ThunderAgent 架构概览：全局 waiting queue 喂 N 个推理后端；周期性 thrashing monitor 用 program-aware 元数据触发跨后端 Pause/Restore。（论文 Fig. 3）](CN/wiki/llm-serving-for-agents/thunderagent-figs/fig3-architecture.png)

### 组件 1 —— Program-aware 调度器带周期 thrashing 监视器（§4.3.1）

跟 Continuum 在 workflow 到达时一次性 TTL 预测不同，ThunderAgent **每 $\Delta t = 5$ s 周期性运行一次 monitor**，对每个 backend $\mathcal{L}$ 评估 thrashing 条件：

$$C_{\text{total}} < \sum_{p \in \mathcal{L}} c_p \quad (6)$$

context 长度之和超过 backend 的 KV 容量时，thrashing 在即。带 high/low watermark $\lambda_{\max}, \lambda_{\min}$（默认都 = 1.0）的 hysteresis 窗口，调度器：

- 按 $S_{\text{pause}}(P) = 1/c_P + \mathbb{1}(\tau = A)$ **Pause** context 最短的 Acting program，直到 $\sum c_p < \lambda_{\max} \cdot C_{\text{total}}$。
- 按 $S_{\text{restore}}(P) = 1/c_P + \mathbb{1}(\tau = R)$ **Restore** context 最短的 Reasoning program，当 $\sum c_p < \lambda_{\min} \cdot C_{\text{total}}$。

指示函数编码了一个关键优先级：**Pause 时先驱 Acting 后驱 Reasoning**（Acting program 反正没在产 token，pause 只损失 caching）、**Restore 时先复 Reasoning 后复 Acting**（Reasoning program 立刻产 decode token，最大化吞吐）。

一个微妙但重要的精化：thrashing 检查里不用二元 pause/keep，而对 Acting program 的 token 权重应用**时间衰减函数** $f(t) = 2^{-t}$：

$$C_{\text{total}} < \sum_{p \in \mathcal{L}, \tau = R} c_p + \sum_{q \in \mathcal{L}, \tau = A} c_q \times f(t_q) \quad (7)$$

一个 program Acting 越久（长 tool call），它的有效权重越低，调度器越倾向驱逐它。论文证明（Appendix E.1）**指数衰减在工具执行时间满足 memoryless property 时是最优的** —— 即剩余时间独立于已经过的时间。这是 Continuum TTL 过期机制的形式化对偶，但是反应式而非预测式的。

### 组件 2 —— 全局 program-aware waiting queue（§4.3.2）

跨节点内存不平衡（上面失败 2）的根源：paused program 蹲在 *per-node* waiting queue 里，所以有空闲内存的节点没法接过载节点队列里的活。ThunderAgent 的修复：**所有 DP 副本共享一个全局队列**。

让这安全的洞察：**一个 program 被 paused 后，KV cache 就没了 —— 所以 restore 时它跟节点无关**。把 paused program 路由到不同节点不会损失任何 cache 收益。这跟 vLLM 的 KV-aware router（贴节点维持 cache locality）相反但是对的，因为 *cache locality 已经被 pause 破坏了*。

形式化界：$\text{Cost}_{\text{unused}} < c_{\min} \cdot \Delta t$ per node，其中 $c_{\min}$ 是 paused program 里最小的 token 长度。所以内存不平衡被最小 paused program 的 footprint × 监视间隔 上界 —— 很小。

### 组件 3 —— 工具资源管理（§4.4）

两个机制配合：

**Hook-based 垃圾回收。** 生命周期 hook 把工具资源拆除耦合到 program 调度状态。$s = \text{Terminated}$ 时收集器立即拆 sandbox、网络 socket、计算 slot。Figure 2b 的结果：active 磁盘用量保持近常数（~0.5 TB）跟处理过多少 workflow 无关；而 request-aware baseline 线性增长，~300 个 workflow 就超过 2 TB 容量。

**异步环境准备。** 工具环境设置的慢路径（Docker 镜像 pull + 装依赖 29.9-47.2 s）跟 LLM reasoning 重叠起来。调度器盯全局等待队列；当一个高优先级 program（高 $S_{\text{restore}}$）接近 restore 阈值时，系统**在 GPU 内存还没分配前就异步启动它的工具环境**。LLM 准备好 decode 时，环境也已经准备好。

这本质上是 **prefetching** 应用到 agent 基础设施 —— 跟 [[speccache|SpecCache]] 对 action observation 做的 LLM prefetching 正交。

## 标志性证据

![ThunderAgent vs vLLM 跟 Continuum，跨 6 个 workload × 3 个模型 × 2-4 个数据集。ThunderAgent 在 serving 吞吐上 1.24-3.58× 领先；Continuum 在随机工具 workload（c, f）上*输给* vLLM。（论文 Fig. 4）](CN/wiki/llm-serving-for-agents/thunderagent-figs/fig4-serving-results.png)

6 个 panel，都 8×H100 节点（除了 ToolOrchestra 在 RTX 5090）：

| Panel | Workload | 模型 | vLLM（baseline） | Continuum | ThunderAgent | TA vs Continuum |
| ----- | -------- | ---- | --------------- | --------- | ------------ | --------------- |
| (a) | SWEAgent | GLM-4.6 | 1× | 1.52× | **2.65×** | 1.74× 快 |
| (b) | OpenHands(code) | GLM-4.6 | 1× | 1.08× | **3.58×** | 3.31× 快 |
| (c) | ToolOrchestra (HLE) | Qwen3-8B | 1× | **0.65×** ← Continuum 输给 vLLM | **1.48×** | 2.28× 快 |
| (d) | SWEAgent | Qwen3-235B | 1× | 1.44× | **3.02×** | 2.10× 快 |
| (e) | OpenHands(code) | Qwen3-235B | 1× | 1.22× | **2.43×** | 1.99× 快 |
| (f) | ScienceAgent | GLM-4.6 | 1× | 1.06× | **1.24×** | 1.17× 快 |

**Panel (c) 是关键证据。** ToolOrchestra on HLE（Humanity's Last Exam）的工具时长 *高度随机* —— 网络 API 延迟跨数量级。**Continuum 的 TTL 预测在这种随机性下崩溃**，系统过度 pin KV，导致比 vanilla vLLM 还差。ThunderAgent 的反应式 thrashing 监视处理确定性和随机两种工具模式。

**RL rollout。** 同 workflow 在 2× 8×H100 节点做 RL 数据收集：**1.79–3.92× 改善** 比 prior SOTA（论文发表时领先的分布式 RL rollout 组合 vLLM + SGLang Gateway）。

> [!example]- KV cache 命中率分解（论文 Fig. 5）
> ThunderAgent 在可预测工具调用时长的 workload（a, b, d, e）上维持 **~100% KV 命中率**，最高到 192 并发 workflow。随机 workload（c, f）上 ThunderAgent 动态用 hit rate 换 less idle caching —— 仍然击败 vLLM 跟 Continuum。Continuum 的命中率在高并发下从 >90% 跌到 ~60%，因为过度 pin 的 KV 把新接纳饿死。

## 局限与开放问题

**论文承认（或没承认）的局限。**

- **单集群。** $\mathcal{L}$ 假设所有 backend 都能从一个调度器接触到。地理分布部署（参见 [[prfaas|PrFaaS]]）需要在上面套一层。
- **没讨论 SLO/优先级。** 所有 program 一视同仁；没有"这个 program 是用户面对的，优先级高"的概念。真实生产需要。
- **时间衰减常数** $f(t) = 2^{-t}$ 是硬编码；论文在 memoryless 工具时间下证明了它是最优的，但没测非 memoryless 分布下的敏感性。
- **Δt = 5 s 监视间隔**是固定配置 —— 没有对突发到达模式的自适应机制。
- **Hook-based GC 假设工具行为良好**，尊重 SIGTERM/cleanup hook。fork 出脱节进程或在 sandbox 外泄漏资源的工具不在覆盖范围。
- **没 program 间公平性。** Context 快速增长的 program 可以饿死小 program（shortest-first 驱逐是反公平方向贪心的）。
- **没跟 InferCept 对比。** InferCept 的 Discard/Preserve/Swap 谱系没被显式引用 —— InferCept 只作为背景出现。
- **Workload 覆盖范围**：SWE-Agent（编程）、OpenHands（编程 + 科学）、ToolOrchestra（路由）、ScienceAgent（科学）。没 GUI agent、没多模态 agent、没 human-in-the-loop chatbot benchmark。

## 优势

- **Program 抽象是文献里最干净的。** $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ 经久不衰 —— 会出现在教科书里。
- **二次代价 Lemma 一锤定音解决驱逐。** 不再争"LRU 还是按大小驱"—— 答案是按 *大小平方*，贪心结果是 *最短优先*。
- **6 个 benchmark 全赢，包括随机的。** Continuum 在 panel (c) 和 (f) 的失败对预测式 TTL 思路是致命的；ThunderAgent 的反应式做法处理两种场景。
- **OpenAI 兼容 API，只多一个参数。** 集成简单 —— 客户端在请求里加 `program_id`，别的不变。
- **MIT 协议开源，vLLM + SGLang 后端。** 没厂商锁定。
- **MORI 的系统层基础。** MORI 做的事都基于 ThunderAgent 的原语，所以未来两层 / 多层 offloading 研究都建在这个栈上。

## 实现要点

- **架构**：调度器进程蹲在一个或多个 vLLM/SGLang 后端前面。通过 HTTP（OpenAI 兼容）+ 内部 RPC（Pause/Restore）通信。
- **周期 monitor**：线程每 $\Delta t = 5$ s 扫一次 backend 状态。
- **后端**：vLLM 跟 SGLang 开箱即用。工具编排后端可配置（Kubernetes、Docker Compose、自定义）。
- **代码量**：小（一个调度器 + 少量 wrapper）—— MORI 在上面加 ~3,300 行，给个 ThunderAgent 自身 LoC 的参考。
- **超参数**：$\Delta t = 5$ s、$f(t) = 2^{-t}$、$\lambda_{\max} = \lambda_{\min} = 1.0$（默认无 hysteresis）。
- **测过的模型**：GLM-4.6（355B MoE、FP8、TP=8）、Qwen3-235B（FP8、TP=8）、Qwen3-8B（FP16）。
- **测过的硬件**：8×H100 节点（大模型）、单 RTX 5090（ToolOrchestra 小模型设置）。

## 这意味着什么

**对 agent serving 研究。** ThunderAgent 是未来 agent-serving 研究将建立的 **系统层基础**。谱系如下：

- **第 1 代（2023-2024）**：vLLM（PagedAttention）→ SGLang（RadixAttention）—— request 级调度、agent 无感。
- **第 2 代（2024）**：[[infercept|InferCept]]（per-request Discard/Preserve/Swap）—— 第一个 agent-aware 但仍 request 粒度。
- **第 3a 代（2025-11）**：[[continuum|Continuum]]（per-program 预测式 TTL）—— program-aware 但是预测式。
- **第 3b 代（2026-02）**：**ThunderAgent（per-program 反应式 thrashing 监视）**—— program-aware 且反应式。**在随机 workload 上击败 3a 代。**
- **第 4 代（2026-05）**：[[mori|MORI]]（在 ThunderAgent 上的两层 GPU+CPU）—— 用层级化内存扩展 3b。

Continuum vs ThunderAgent 的分歧是 **预测式 vs 反应式** trade-off。预测式（TTL）在工具时长建模准且稳定时赢；反应式（thrashing 监视）在它们随机且对抗时赢。生产系统大概率两个都想要，由 workload 分类器门控。

**对生产部署。** ThunderAgent 是当前 agent-serving 研究系统里最 production-ready 的：OpenAI 兼容 API、MIT 协议、vLLM + SGLang 支持、内置异步环境管理。自然部署路径：

1. 把 ThunderAgent 塞到现有 vLLM/SGLang 集群前面。
2. 改客户端代码传 `program_id`（每请求一行）。
3. 配置工具编排器后端（Kubernetes/Docker）。
4. 观察 1.5-3.6× 吞吐改善。

多租户部署需要公平性或 SLO 的话，需自定义调度扩展（论文没涉及）。

**2027 预期。** "Agentic program" 数据结构会作为 vLLM 上游的一等概念出现。Continuum vs ThunderAgent 辩论会成为 agent serving 教科书里"预测式 vs 反应式"trade-off 的标准讨论。

## 相关阅读

- [[continuum]] —— Continuum：同期竞争者；预测式 TTL 路线 vs ThunderAgent 的反应式监视。
- [[mori]] —— MORI：建在 **ThunderAgent 之上**的两层 GPU+CPU offloader（~3,300 行 Python）。
- [[infercept]] —— InferCept：per-request KV 管理用 Discard/Preserve/Swap；ThunderAgent 扩展到 per-program。
- [[speccache]] —— 正交的对 *extra-LLM 环境 observation* 的 speculative cache；跟 ThunderAgent 可组合。
- [[agent-serving-challenges]] —— agent serving 跟 chatbot 不同的更广背景。
- [[multi-turn-optimization]] —— 多轮 KV 复用、prefix caching、session 管理。
- [[agentic-ai-workload-characteristics]] —— 启发所有这些调度器设计的 workload 测量。

## 源码

[github.com/ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent) —— MIT 协议、vLLM + SGLang 后端、OpenAI 兼容 API。

```bash
git clone https://github.com/ThunderAgent-org/ThunderAgent.git
cd ThunderAgent
pip install -e .
```

客户端集成就一行代码 —— 给 OpenAI 兼容请求加 `program_id`，剩下交给 ThunderAgent。

## 论文引用

完整引用、BibTeX、标志数字见 [[thunderagent-citation|sources/papers/thunderagent/citation.md]]。
