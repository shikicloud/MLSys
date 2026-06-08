---
title: "InferCept：面向增强型 LLM 推理的高效中断支持"
category: llm-serving-for-agents
tags: [infercept, kv-cache, interception, augmented-llm, vllm, paged-attention, icml-2024, agent-serving, paper-review]
created: 2026-06-08
updated: 2026-06-08
status: mature
paper: arXiv:2402.01869
code: https://github.com/WukLab/InferCept
---

# InferCept: Efficient Intercept Support for Augmented Large Language Model Inference

> [!info] 论文元信息
> - **论文**：[arXiv:2402.01869](https://arxiv.org/abs/2402.01869) —— *InferCept: Efficient Intercept Support for Augmented Large Language Model Inference*，ICML 2024 (PMLR 235)；v1 2024-02-02，v2 2024-05-30
> - **代码**：[WukLab/InferCept](https://github.com/WukLab/InferCept) —— UCSD WukLab 团队发布的 vLLM fork
> - **作者**：Reyna Abhyankar\* (UCSD)、Zijian He\* (UCSD)、Vikranth Srivatsa (UCSD)、Hao Zhang (UCSD)、Yiying Zhang (UCSD)
> - **机构**：UC San Diego（全部作者）
> - **实现**：vLLM fork，基于 PagedAttention 做 KV 内存管理

> [!important] InferCept 在 agent-serving 演进谱系中的位置
> InferCept 是**第一个面向"被中断的增强型 LLM"专门设计的推理框架** —— [[continuum|Continuum]] 和 [[mori|MORI]] 都把它当作 baseline 明确引用的直接前驱。它确立了核心论点：**一次工具调用 / 人类响应是 *interception*（中断），不是 request 的结束**；这段中断期间的 KV cache 是 *temporarily unused context*（暂时未用的上下文），值得作为一类一等资源来管理。Continuum 与 MORI 后来打的两个失败模式 —— **per-turn 排队延迟** 和 **长尾工具延迟下的无界保留** —— 都源于 InferCept 仅基于 reload cost 的决策模型留下的缺口。

---

## 摘要（2 分钟读完）

**它是什么。** InferCept（ICML 2024，UCSD WukLab）是一个基于 vLLM fork 的 LLM 推理框架，专门面向 **增强型 LLM 工作负载** —— LLM 被外部实体打断（工具调用、聊天人类、图像生成调用、TTS、虚拟环境）。它把未解决的问题命名为"中断期间 KV cache 怎么处理"，并提出 **min-waste interception**（最小浪费中断）：根据哪种动作能最小化 GPU 内存浪费，逐 request、逐 interception 在 `Discard` / `Preserve` / `Swap` 之间动态选择。

**唯一核心思想。** **用闭式方程量化每种 KV-cache 选项的 GPU 内存浪费，然后逐 request、逐 interception 选出最小那个。** 三个子部件支撑它：

1. **浪费方程** —— `WasteDiscard`、`WastePreserve`、`WasteSwap`、`WasteChunkD`（分块重算）的闭式公式，参数化于中断时长、上下文长度、批大小，以及每 token KV 内存 $M$。
2. **每种选项的工程加速以缩小其浪费项** —— *swap 流水线 + 分块*（按层粒度把 CPU↔GPU PCIe 与 forward kernel 重叠）消除 96% 的 `Swap` 浪费；*重算分块*（把一次 recompute 切成 S 大小的块，S 是 GPU 饱和点）把 `Discard` 浪费砍掉约一半；*预算化 swap-out* 避免 PCIe 链路被打满。
3. **Min-waste 调度** —— 每次迭代，按 `min(WastePreserve, WasteChunkD)` 降序对被中断 request 排序，再依次分给 swap（若 swap 预算未耗尽）、preserve 或 discard。对于持续时间高方差的中断（chatbot 人类响应、图像生成），使用 `t_now − t_call` 动态估计器，达到 oracle 性能的 93%。

去掉浪费方程，InferCept 就退化成 ad-hoc 启发式；去掉 swap 流水线，`Swap` 的 26% 浪费会拖垮吞吐；去掉 min-waste 调度器，系统就退回 vLLM 那套"interception = request 结束"。

**头条结果。** A100 GPU 上跨 **GPT-J-6B、Vicuna-13B、Llama-3-70B**，六种 augmentation 类型（math、QA、VE、chatbot、image、TTS）混合负载评测：

| 配置 | Baseline | InferCept | 提升 |
| ---- | -------- | --------- | ---- |
| **6B / 1 GPU** | vLLM (Discard) | **1.6× 更高 request arrival rate**（同 normalized latency） | normalized latency 降低 1.9×–5.7× |
| **13B / 1 GPU** | vLLM | 1.25× 更高 request rates | 3.1× 更高负载（同最小 TTFT） |
| **13B / 2 GPUs (TP)** | vLLM | 1.8× 更高 request arrival rate | **normalized latency 降低 1.6×–10×** |
| **70B / 4 GPUs (TP)** | vLLM | 同 RPS 下 2× 更高 request arrival rate | **normalized latency 降低 1.3×–12×** |
| 对比 SOTA 吞吐 | vLLM | **吞吐提升 1.6×–2×**；完成请求数翻倍 | 即摘要的标志数字 |

GPU 内存浪费：vLLM 约 25%、完整 InferCept **0.69%** —— 消除超过 60% 的重算浪费和 96% 的 swap 浪费。

**为什么重要。**

- **第一个吃螃蟹解决增强型 LLM 问题的系统。** 在 InferCept 之前，所有推理系统（vLLM、Orca、DeepSpeed-Inference、TensorRT-LLM、FastServe）都把 interception 当作 request 结束 —— discard 上下文，等工具返回时再重新发起。InferCept 命名了这个问题（"min-waste interception"）和三种动作（`Discard` / `Preserve` / `Swap`），后续工作（Continuum、MORI、Autellix、Pie）都沿用这套词表。
- **闭式浪费模型是真正经久不衰的贡献。** 即便后来 [[continuum|Continuum]] 补上了 InferCept 没考虑的 per-turn queueing delay 项，那四个 `Waste*` 方程仍是 canonical 的建模框架，在 MORI 的两层 idleness 推导和 Autellix 的 program-aware 调度里都以扩展形式重现。
- **重算分块是真正的新点。** 把一次 recompute 切成 S 大小的块（S 是 GPU query-token 饱和点）让 recompute 去填充 decode 留下的空闲核 —— 这是一个干净的协同优化，后来 Sarathi-Serve 的 chunked prefill 把它推广了。
- **遗留的局限播下了之后两年的研究种子。** (1) 模型只用 *reload cost* —— 没有 per-turn 排队代价 —— Continuum 用 TTL 补上。(2) 没有保留时长上界 —— 一个挂死的工具调用能把 KV 锁住不放 —— Continuum 的 TTL 过期机制补上。(3) 只有单层（GPU）—— 没有 CPU / 磁盘空闲层 —— MORI 用两层 offloader 补上。

---

# 深入（drill-down 从这里开始）

## 背景：增强型 LLM 推理为什么不一样

**增强型 LLM** 指的是在生成过程中调用外部实体的 LLM —— 计算器、知识库 QA 工具、虚拟环境（ALFWorld）、聊天人类、图像生成模型（Stable Diffusion）、TTS 模型（Bark）。每次调用都制造一次 **interception**：LLM decoding 暂停，等待外部返回，然后必须恢复。论文 §2 测了六种 augmentation 的中断属性（论文 Table 1，此处复述）：

| Augmentation | 中断时长（秒）—— 均值, 方差 | 每 req 中断次数 | 上下文长度（token） |
| ------------ | --------------------------- | --------------- | ------------------- |
| **Math（计算器，GSM8K）** | 9e-5, 6e-5 | 3.75, 1.3 | 1422, 738 |
| **QA（多跳，Wikipedia）** | 0.69, 0.17 | 2.52, 1.73 | 1846, 428 |
| **Virtual Env（ALFWorld）** | 0.09, 0.014 | 28.18, 15.2 | 2185, 115 |
| **Chatbot（ShareGPT）** | 28.6, 15.6 | 4.45, 1.96 | 753, 703 |
| **Image（ChatGPT + SD）** | 20.03, 7.8 | 6.91, 3.93 | 1247, 792 |
| **TTS（ChatGPT + Bark）** | 17.24, 7.6 | 6.91, 3.93 | 1251, 792 |

从这张表能看出三个驱动整个设计的结构性观察：

1. **中断时长双峰化。** 自动化短任务（Math 亚 ms、QA 亚秒、VE ~90 ms）vs. 人类/模型驱动的长任务（Chatbot 28.6 s、Image 20 s、TTS 17 s）—— 没有一种通用策略。
2. **长任务的方差也很大。** Chatbot 方差 15.6 s vs. 均值 28.6 s —— 同类型下一次中断可能 13 s 也可能 44 s。静态画像无法预测。
3. **上下文都很大。** 连 VE 都有 2185 token；Image/TTS 1247–1251。在 28 s 人类响应窗口里把这些 KV 都留在 GPU 上是巨大浪费；丢掉再重算又是巨额 prefill 账单。没有"KV 很小所以怎么搞都行"的捷径。

现有推理系统（vLLM、Orca、DeepSpeed-Inference、FastServe、TensorRT-LLM）一律把 interception 当作 request 的 **结束** 并丢弃 KV。中断结束后入一个新 request、从零重新 prefill。论文测了这个 `Discard` 策略，发现**混合负载下重算占用 37–40% 的总 forward 时间** —— 一半 GPU flops 都在重做几秒前还存在的 prefill。

幼稚补丁 `Preserve`（中断期间把 KV 钉在 GPU 上）则把整段中断时长内的 GPU 内存都浪费掉。再幼稚补丁 `Swap`（中断时把 KV 移到 CPU，恢复时再换回 GPU）避免了 GPU 内存浪费，但前台计算会被 PCIe 带宽阻塞，并且单个被中断上下文散落在很多 PagedAttention block 里造成大量小 kernel launch。

InferCept 的贡献是把这三种动作之间的选择**形式化、参数化、按 request 按 interception 做出来**。

## 四个浪费方程

浪费模型量化"GPU 内存 × 时间"在中断期间的损失。记 $C_i^j$ = request $i$ 在中断 $j$ 时的上下文 token 数；$M$ = 每 token KV 内存；$T_{fwd}(C)$ = 上下文 $C$ 的 forward 延迟；$T_{INT}^j$ = 中断 $j$ 的时长；$T_{swap}(C)$ = $C$ token 的 swap 延迟；$C_{other}$ = 所有其他并发 running request 的上下文长度之和；$C_{batch}$ = 整批上下文总和。

**Discard 浪费** —— 重算代价 + 其他 request 被阻塞：

$$\text{WasteDiscard}_i^j = T_{fwd}(C_i^j) \times C_i^j \times M + T_{fwd}(C_i^j) \times C_{other} \times M \quad (1)$$

两项含义：(a) 重算被 discard 的 request 时其上下文占用的内存，(b) 其他 running request 必须保留的内存（在等待这一次性 recompute 完成期间）—— 这就是 *iteration-time blowup* 项。

**Preserve 浪费** —— 整个中断期间空挂的 GPU 内存：

$$\text{WastePreserve}_i^j = T_{INT}^j \times C_i \times M \quad (2)$$

线性于中断时长。短中断（Math、QA）可以接受；长中断（Chatbot、Image、TTS）灾难。

**Swap 浪费** —— 两个方向的 swap 都会阻塞 compute：

$$\text{WasteSwap}_i^j = 2 \times T_{swap}(C_i^j) \times C_{batch} \times M \quad (3)$$

因子 2 = swap-out + swap-in。代价以**其他** request 在 swap kernel 占用 GPU 期间被空挂的内存来计量。

**分块 discard 浪费** —— 把 recompute 切成 $n$ 次迭代，每次块大小 $C_i^j/n$：

$$\text{WasteChunkD}_i^j = \frac{T_{fwd}(C_i^j) \times C_i^j \times M}{2} + n \times T_{fwd}\!\left(\frac{C_i^j}{n}\right) \times C_{other} \times M \quad (4)$$

第一项是一次性重算成本的 *一半*（因为分块重算与 decode 交错，利用了原本闲置的 GPU 核）；右边项把单一巨型 $T_{fwd}(C_i^j)$ 换成 $n$ 个较小的 $T_{fwd}(C_i^j/n)$ —— 当 GPU query-token 饱和点 $S$ 已经达到时，$n \times T_{fwd}(C/n) \leq T_{fwd}(C)$ 通常成立，所以净是赚的。

Min-waste 调度规则（eq. 5）：

$$\text{Waste}_i^j = \min\!\left(\text{WastePreserve}_i^j, \text{WasteChunkD}_i^j\right)$$

Swap 被处理成 *预算化* 的第三选项 —— 优先分给 `min(Preserve, ChunkD)` 浪费最高的那些 request，直到 swap 预算（受 PCIe 带宽和空闲 CPU 内存约束）耗尽。

> [!warning] 浪费模型遗漏了什么 —— 也正是为什么 Continuum / MORI 存在的理由
> 公式 1 的 $C_{other}$ 项捕获了 *并发* 迭代时间放大，**但没有捕获**：被 preserve 或 swap 的 request 在中断结束后被放回队列时，要排在新接纳的工作后面所付出的 *排队代价*。Eq. 2 和 3 都没把这种 *等待* 计进去。[[continuum|Continuum]] 的 cost-benefit 模型加上 `OutOfOrderCost` 项就是为修这个。
>
> Eq. 2 也对 $T_{INT}$ 没有上界。如果一个 `pytest` 工具调用挂了 60 s，`WastePreserve` 线性增长 —— 模型在 *下一次* 决策时正确地建议"切到 discard" —— 但 *已经* pin 在那的 KV 从工具开始那刻起就没有任何 eviction 机制。[[continuum|Continuum]] 的 TTL 过期机制就是 InferCept 静态决策所缺的那道边界。

## 三大组件详解

### 组件 1 —— Swap 流水线与分块（§4.1）

幼稚 `Swap`（strawman）在中断发生时同步发起 CUDA memcpy kernel，整个 $T_{swap}(C)$ 时间窗里阻塞 forward。论文测出这种 `Swap` *本身* 就浪费 26% 的 GPU 资源，并且总负载时间里有超过 25% 在等 swap kernel。

**Swap 流水线** 把每个 model layer 的 swap 当作一个独立的流水线 stage。当 layer $i+2$ 的 swap kernel 启动时，layer $i+1$ 的 swap 在搬数据，layer $i$ 的上下文已经被释放并参与到了正常 forward 计算。本质上这是**按层粒度对 memcpy 和 matmul 做流水线** —— 等效于 ZeRO-Offload 的 overlap 但粒度落在每个 token block。

**Swap 分块** 进一步把单次 swap-out 或 swap-in 切到多个 model-forwarding 迭代里。论文计算一个 *swap limit* $N_i = T_{fwd}^{-1}(B_i)$，其中 $B_i$ 是 batch 大小，$T_{fwd}^{-1}$ 反演每迭代延迟 —— 即一次 forward 延迟内可以"免费"swap 多少 token。

**Swap-in / swap-out 预算化** 是一个有约束的最优化。每次迭代选择 swap 出/进多少 token，使得：

1. swap-in + swap-out ≤ $N_i$（不超过可隐藏 swap 预算）
2. swap-out 内存 ≤ 空闲 CPU + swap-in（CPU 容量约束）
3. swap-in + 新 token 内存 ≤ swap-out + 空闲 GPU（GPU 容量约束）

调度时解一个小 LP。最终结果：消除 `Swap` 方程 96% 的浪费。

### 组件 2 —— 重算分块（§4.2）

Decoding request 每 request 只需要 GPU 核处理 *一个 query token*；recompute 需要核处理 *整个上下文长度* 的一个 request。这两者在 GPU 的计算/内存利用谱系上正好处于两端。关键洞察：**一批 decoding request 通常在跑出 GPU 内存之前根本填不满 GPU 核** —— 有空闲算力可以让 recompute 来填，并且不超内存预算。

论文把 **GPU 饱和点** $S$ 定义成 query-token 数的临界值，超过它则迭代时间单调增长。$S$ 按模型架构离线 profiling 得到。重算的 chunk size 就是 $S - \text{running\_group\_size}$ —— 每次迭代正好的多余 query-token 容量。

幼稚 `Discard` 一口气在一次迭代里重算整个上下文，付 $T_{fwd}(C_i^j)$ 一次但把所有其他 request 都阻塞那一次迭代。分块重算付 $n \times T_{fwd}(C_i^j / n)$ 跨 $n$ 次迭代，但 *每次迭代都更短*，所以对其他 request 的阻塞被最小化 —— Eq. 4 量化了净赢。

这在概念上邻近 Sarathi 的 *chunked prefill*（也是 2023），但有两个区别：(a) Sarathi 分块的是新 request 的 prefill 以均衡每迭代延迟；InferCept 分块的是 *被中断 request* 的 *重算*；(b) InferCept 的 chunk size 由 running batch 的剩余容量动态决定，不是静态配置。

### 组件 3 —— 跨 request 动作决策与调度（§4.3）

每个调度步，InferCept 维护**三个队列**：

1. **Running queue** —— 当前正在 decode 的 request。
2. **Swap queue** —— 已恢复但中断期间被 swap 出去的 request。按 **原始到达时间** 排序（中断前后 FCFS 顺序保留，不被踢到队尾）。
3. **Waiting queue** —— discarded 后被恢复的 + 全新的 + 之前 running 但被驱逐的 request。同样按原始到达时间 FCFS。

每次迭代两阶段决策：

**阶段 1 —— 中断 request 调度。** 对每个新被中断的 request，计算 `min(WastePreserve, WasteChunkD)`（Eq. 5）。把所有被中断 request 按该浪费 **降序** 排序，然后逐个分给 swap（一个接一个），直到这次迭代的 swap-out 预算耗尽。剩下的根据 Eq. 5 的两项哪个更小，选择 preserve 或 discard。

**阶段 2 —— 恢复 / 接纳调度。** 按 FCFS 从 waiting + swap 队列拉，接纳新 token，直到 GPU 饱和点 $S$。Swap 队列 *单独* 维护，因为 swap-in 内存是 GPU 资源的 *额外* 部分，不应该与新接纳竞争。

论文 Figure 1 的架构图把这套端到端呈现出来：

![InferCept 与替代方案：四种 timeline（Today's Discard、Discard、Swap，与 InferCept 的 MinWasteDiscard + chunked recompute + MinWasteSwap + Preserve）（论文 Fig. 1）](CN/wiki/llm-serving-for-agents/infercept-figs/fig1-architecture.png)

上面三行是 strawman 的 timeline（重算阻塞、swap 阻塞）；下面三行是 InferCept 的分块 Discard、流水线 Swap、干净 Preserve，中间的 `Executor + Offline Profiler` 块根据浪费方程做逐 request 动作决策。

### 组件 4 —— 中断时长估计（§4.4）

WastePreserve（Eq. 2）需要 $T_{INT}$，但 chatbot / image / TTS 的中断时长高方差（chatbot 方差 ≈ 均值），离线 profiling 帮不了忙。

**动态估计器**：$\hat{T}_{INT} = t_{now} - t_{call}$ —— 一个中断越久，剩余时长的估计就越大。这是简单的 **无记忆 / 指数尾先验**，对长尾中断分布有效，因为长尾分布大致重尾：等得越久，再等的预期就越长。

论文报这个估计器达到 **oracle 性能的 93%** —— 即只留 7% 性能空间给更复杂的方案，大概率不值得搞。

## 标志性证据

**端到端性能（Figure 2，混合负载）：**

![混合负载下 6B/13B/70B 模型的端到端性能：normalized latency（上）、throughput（中）、TTFT（下）。InferCept（绿）维持比 vLLM（红）/ImprovedDiscard/Preserve/Swap 更高的负载与更低延迟。（论文 Fig. 2）](CN/wiki/llm-serving-for-agents/infercept-figs/fig2-e2e-results.png)

三排图：(上) normalized latency vs 请求率，(中) 吞吐，(下) TTFT。四种配置（6B、13B、13B-TP2、70B-TP4）下，InferCept（绿）始终维持最右侧曲线（承受最高负载）和最低 TTFT。

**技术增量分解（Figure 3）：**

![技术增量分解：从 vanilla vLLM 出发依次添加 InferCept 各组件。ImprovedDiscard 单独 -24.5% latency；+MinWasteDiscard +7.8%；+MinWasteSwap +12.7%；+HeuristicPreserves +46.1%；完整 InferCept 再 +46.4%，最终 0.69% 内存浪费。（论文 Fig. 3）](CN/wiki/llm-serving-for-agents/infercept-figs/fig3-technique-breakdown.png)

Vicuna-13B 混合负载 @ 2 RPS 下每个技术的累积提升：

| 变体 | 相对 vLLM 累积 Δ | GPU 内存浪费 |
| ---- | ----------------- | ------------ |
| vLLM (Discard) | 0 % | ~25 % |
| ImprovedDiscard | −24.5 % latency | ~20 % |
| + MinWaste Discard（分块重算） | +7.8 % | ~21 % |
| + MinWaste Swap（预算化、流水线化） | +12.7 % | ~10 % |
| + Heuristic Preserves | +46.1 % | ~1.5 % |
| + Min-waste 调度（完整 InferCept） | +46.4 % | **0.69 %** |

两个最大跃升分别来自 **把 Preserve 加进选项**（系统第一次可以 pin 而不是重算/swap，+46.1%）和 **把启发式 preserve 换成基于 min-waste 的动态 preserve**（+46.4%）。重算分块和 swap 流水线各自的单独贡献较小（约 10%），但它们让 discard/swap 两个选项不再差到不行，从而让 min-waste 选择有"像样选项"可以挑。

**单 augmentation 负载：** QA-only 比 vLLM 快 2.3×（QA 工具调用多为短任务 → preserve 占主导）；Chatbot-only 快 1.9×（人类响应长时长 → swap 占主导，且选择性使用分块重算）。

**分布式设定放大收益：** 13B 跑 2 GPU（tensor parallel）→ normalized latency 改善 1.6×–10×，比单 GPU 13B 更大。原因：聚合 GPU 内存装得下更多 KV cache → 并发 request 更多 → interception 更多 → InferCept 有更多决策可优化。

**70B 跑 4 GPU：** 同 TTFT 下 2.4× 更高负载，normalized latency 降低 1.3×–12×。Llama3-70B 的 grouped-query attention（GQA）压缩 KV —— 这 *帮助* Preserve 和 Swap（要存/搬的数据更少），*削弱* 最优调度的相对收益，但绝对收益依然大，因为 baseline 的浪费也很巨大。

> [!example]- Chatbot 单 augmentation 分解（论文 §5.2）
> Chatbot 单负载下，论文报分块与流水线（针对 `Swap` 与 `Discard`）贡献 **54% 总加速** —— 远高于混合负载场景 —— 因为 Chatbot 的长中断让更多 request 可以并发，触发更多 swap 和 recompute，恰好是分块/流水线把它们做成非阻塞的那块。

## 优势与局限

**优势。**

- **第一个命名并解决增强型 LLM interception 问题的系统。** 词汇表 `Discard` / `Preserve` / `Swap` 和四条浪费方程现在是 canonical 的 —— 后续每篇 agent-serving 论文（Continuum、MORI、Autellix、Pie）要么在它基础上扩展，要么在它的分类法里对照。
- **干净的闭式决策规则。** Min-waste 选择可解释、可调试，所需的离线 profiling（一条 $T_{fwd}$ 曲线、一条 $T_{swap}$ 曲线、一个饱和点 $S$）是一次性、轻量的。
- **模块化、正交。** 论文明确指出技术设计为可整合到 DeepSpeed-Inference、Orca、TensorRT-LLM。大部分工程在 scheduler 层、不在 kernel 层（除了 swap 流水线 wrapper 外不需要 custom CUDA）。
- **重算分块是真正的新点。** Sarathi-Serve 的 chunked prefill 是最近的同辈，但作用在新 request 的 prefill 而非被中断 request 的 recompute 上。InferCept 的版本是动态的（chunk size = 饱和点 − running group size），并停留在 GPU 内存边界内。

**局限（后续工作修复了什么）。**

- **只考虑 reload cost。** 浪费模型记入 $T_{fwd} \times C_{other}$（并发迭代膨胀）但没记入 *回到队列时的 per-turn 排队代价*。[[continuum|Continuum]] 的 `OutOfOrderCost` 项就是补这个。
- **没有保留时长上界。** Preserve 下一个挂死 60 s 的 `pytest` 整段时间都在烧内存；Eq. 2 在 *下一次* 决策时建议"切到 discard"，但已经 pin 的 KV 没有任何 eviction 触发。[[continuum|Continuum]] 的 TTL 过期机制是补丁。
- **单层（GPU）。** Swap 把 *CPU 内存* 当作二元的"GPU vs 非 GPU"层级。没有 GPU 驻留但空闲 vs CPU 驻留但温热 vs 磁盘的概念。[[mori|MORI]] 的两层 idleness 指标是补丁。
- **Interception 是二元的。** 一个 request 要么完全在 `Discard`、完全 `Preserve`、完全 `Swap` —— 没有对近期 vs. 古早 KV block 做部分 pin 的支持。Prefix caching（SGLang/Preble）和 Pie 的可编程 cache 是正交扩展。
- **Workload 特定调参。** $S$（饱和点）、swap 预算、启发式 preserve 阈值都需要按（模型、硬件）元组离线 profile。混合模型机群的集群运维要付 N 次该代价。
- **没讨论多租户 SLO。** 论文优化的是平均吞吐 + 平均延迟。没有 per-tenant 公平、优先级、SLO-aware 决策。Continuum 的 program-level FCFS 是这条线里第一个 agent-aware 公平机制。

## 实现要点

- **基线：** vLLM（Kwon et al., SOSP 2023）。PagedAttention 被沿用作为 KV 内存管理器 —— InferCept 的调度层叠在 page allocation 之上。
- **代码里四个关键组件：** 调度、浪费计算、分块重算 + swap、augment-specific 旋钮（中断时长画像、按 augment 类型的分块预算）。
- **没有 custom CUDA kernel。** Swap 流水线用标准 CUDA stream + memcpy；分块重算用 vLLM 现有的 prefill kernel 加小 chunk size。
- **画像开销：** $T_{fwd}$ 和 $T_{swap}$ 曲线每个模型/硬件 profile 一次，运行时不再 profile。
- **仓库：** [WukLab/InferCept](https://github.com/WukLab/InferCept)。vLLM 基线是 2024 年初版本 —— 比 vLLM 后来的 kernel 改进（FlashAttention-3、FP8 等）要老，所以与现代 baseline 公平对比需要在更新版 vLLM 上重跑。

## 这意味着什么

**对 agent-serving 研究线的意义。** InferCept 确立了后续每篇工作都在用的 canonical 分类法。自然的后续工作各自挑一个局限：

- **[[continuum|Continuum]]（2025 年 11 月）** —— 加上 per-turn 排队代价项 + TTL 过期上界。动作不变（preserve/discard/swap-等价的 CPU offload），但浪费模型更丰富、保留有硬上界。
- **[[mori|MORI]]（2026 年 5 月）** —— 加上两层（GPU HBM + CPU DRAM）层级与连续 idleness 指标，超出 InferCept 的二元 GPU-vs-CPU 与 Continuum 的单层 TTL。
- **Autellix（2025）** —— 在 FCFS 单位层面加上 program-aware 调度（多 request 的 agent program）。
- **Pie（SOSP 2025）** —— 把 KV cache 暴露成可编程资源，让应用代码（不只是推理引擎）决定保留什么。

**对生产部署的意义。** InferCept 本身是这四个里最容易部署的 —— 纯 scheduler 层改动，没有新硬件层，没有可编程 KV。短任务（Math/QA/VE）主导的负载下，InferCept 的启发式 preserve 已经能拿到大部分收益，无需 Continuum/MORI 的运维复杂度。长任务（Chatbot/Image/TTS）主导的负载下，Continuum 的 TTL 或 MORI 的 idleness 分层会更合适。

**2027 预期。** InferCept 的词汇表（`Discard` / `Preserve` / `Swap` / `MinWaste`）会成为 agent-serving API 中的标准请求状态分类法。vLLM 的 `RequestStatus` 枚举大概率会新增 `INTERCEPTED_PRESERVE`、`INTERCEPTED_SWAPPED`、`INTERCEPTED_DISCARDED` 这些一等公民状态。

## 相关阅读

- [[continuum]] —— Continuum：基于 TTL 的 KV pinning + program 级 FCFS；直接扩展 InferCept 的浪费模型，补上 per-turn 排队代价项和保留上界。
- [[mori]] —— MORI：两层（GPU HBM + CPU DRAM）内存 offloader，配连续 idleness 指标；把 InferCept 的二元 swap 决策扩展到连续谱系。
- [[multi-turn-optimization]] —— 跨轮 KV 复用、prefix caching、session 管理的更广视野。
- [[agent-serving-challenges]] —— agent vs chatbot 负载特征；InferCept 的六类 augmentation 分类法落在这里。

## 源码

[github.com/WukLab/InferCept](https://github.com/WukLab/InferCept) —— vLLM fork，四个关键组件（`scheduler`、`waste_calc`、`chunked_swap`、`augment_profiles`）。阅读顺序：先看 `scheduler/min_waste.py` 里的 Eq. 5 决策逻辑，再看 `swap/pipeline.py` 的按层流水线 memcpy，最后看 `recompute/chunked.py` 的基于饱和点的分块。

## 论文引用

完整引用、BibTeX 与标志数字见 [[infercept-citation|sources/papers/infercept/citation.md]]。
