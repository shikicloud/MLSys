---
title: "Agentic AI 的 CPU 中心视角：Characterization + COMB + MAS"
category: llm-serving-for-agents
tags: [cpu-centric, agentic-ai, characterization, comb, mas, micro-batching, scheduling, cpu-gpu-coordination, vllm, paper-review]
created: 2026-06-02
updated: 2026-06-02
status: mature
paper: arXiv:2511.00739
---

# Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective

> [!info] 论文元数据
> - **论文**：[arXiv:2511.00739](https://arxiv.org/abs/2511.00739) —— *Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective*，2025-11-01（v3：2026-04-16）
> - **代码**：未发布
> - **作者**：Ritik Raj（Georgia Tech）、Souvik Kundu（Intel）、Ishita Vohra（Georgia Tech）、Hong Wang（Intel）、Tushar Krishna（Georgia Tech）
> - **机构**：Georgia Institute of Technology + Intel
> - **实现**：vLLM v0.14.0、PyTorch 2.8.0；闭环和开环到达系统

> [!important] 这篇论文 *不是* 什么
> 虽然标题暗示纯 characterization，但承重的贡献是**两个调度算法**（COMB + MAS）加上一个 trace 级测量 —— 关于 agentic serving 中 CPU 瓶颈实际在哪里。Characterization 数据（tool-dominated workload 上 tool 时间最多到 E2E 延迟的 88%）是 *motivation*；两个算法是交付物。两层都要读。

---

## 摘要（2 分钟读完）

**它是什么。** 一篇关于 agentic AI serving 从 CPU 视角的 characterization-and-optimization 研究。作者在两个非对称 CPU-GPU 系统（高 CPU+低 GPU vs 低 CPU+高 GPU）上 trace 五个代表性 agentic workload（Toolformer、SWE-Agent、RAG/Haystack、ChemCrow、Web-Augmented Agent / LangChain），识别 GPU 中心 serving 系统忽略的 CPU 侧瓶颈，然后提出两个调度优化 —— **COMB** 给同质 batch，**MAS** 给异质混合 workload。

**核心 idea。** **CPU 和 GPU 在 agentic pipeline 里很大程度上是 *不相交阶段*，朴素调度让闲的那个 underutilized**。三个子部件：

1. **编译期分类法 + 运行时 characterization** —— Agentic workload 沿三个正交轴变化（LLM-vs-Host orchestrator、Static-vs-Dynamic path、Single-vs-Multi-step）。论文挑了 5 个跨这些 corner 的代表性 workload 并测时间花在哪里。头号测量：**tool processing 在 tool-dominated workload 上能占 E2E 延迟的 88%**（如 Sys 2 上 RAG 的 ENNS 检索、ChemCrow Heavy 的 conformer 生成）。
2. **COMB（CPU-Aware Overlapped Micro-Batching）** —— 把大 batch 分成大小为 $B_{cap} \approx 1\text{-}2 \times \#\text{CPUs}$ 的有界 micro-batch，然后 *overlap* micro-batch $i+1$ 的 CPU 阶段跟 micro-batch $i$ 的 GPU 阶段。避免 CPU 过订阅（杀死 CPU 效率）同时维持 GPU 利用。
3. **MAS（Mixed Agentic Scheduling）** —— 给异质 workload（有些请求 CPU-heavy 有些 GPU-heavy），通过分离的队列上界（$E_{cap,CPU}$、$E_{cap,GPU}$）+ 共享弹性队列（$E_{cap,shared}$）接纳。防止主导请求类型饿死少数类型。

去掉 COMB 大 batch 上 CPU 过订阅毁掉吞吐；去掉 MAS 少数请求类型遭 head-of-line 阻塞；去掉 characterization 你的 $B_{cap}$ 或 $E_{cap}$ 大小定不对你的硬件。

**头号实验结果。**

| Setting | 优化 | 指标 | 提升 |
| ------- | --- | --- | ---- |
| 同质 standalone batch (BS=128) | COMB | P50 延迟 | **低 1.7×** |
| 同质开环，$\lambda=13$ req/s | COMB ($N_{cap}=64$) | 服务延迟 P50/P90 | **低 2.9× / 3.9×** |
| 同质开环，$\lambda=13$ req/s | COMB | 总延迟 P50/P90 | **低 1.6× / 1.8×** |
| 同质开环 | COMB ($N_{cap}=64$) | 吞吐 | **高 1.7×** |
| 异质开环，$p_{LLM}=0.25$，Sys 1 | MAS | GPU-heavy P50/P90 | **低 2.37× / 2.49×** |
| 异质开环，$p_{LLM}=0.50$，Sys 1 | MAS | 总 P50/P90 (avg) | **低 1.62× / 1.30×** |

**为什么重要。**

- **第一个端到端测量调和 "tool dominates" vs "LLM dominates" 辩论** —— 答案跟 workload 和硬件相关，不是单一数字。Tool 时间从 ~10%（Toolformer on Sys 1，GPU 瓶颈）到 88%（RAG ENNS on Sys 2，CPU 瓶颈），取决于哪边硬件更好。
- **HP CPU + LP GPU 能匹敌 HP GPU + LP CPU** 在 tool-dominated workload 上 —— 有真实成本部署影响（如果你的瓶颈是检索就别买 H200；买更多 Xeon 核）。
- **识别 CPU 并行效率根本性低于 GPU** —— 实证观察 STREAM 风格 4 进程/socket 就饱和 80%；超过核心数过订阅触发 OS 调度器争用。
- **2027 年预测。** COMB 风格 CPU-GPU overlap micro-batching 成为 vLLM/SGLang 给 agent workload 的标准。MAS 风格请求类型感知接纳成为默认调度纪律。两者会在 12 个月内集成为 `--enable-cpu-overlap` 和 `--enable-mixed-type-admission` flag。

---

# 详细内容（深入阅读从这里开始）

## 背景：没人测过的 CPU 瓶颈

生产 LLM serving 系统（vLLM、SGLang、TensorRT-LLM）疯狂为 GPU 侧优化 —— paged attention、连续批处理、PD 解耦、FP8 KV cache 等等。它们把 tool 执行和 Python 编排当 "外部"，不在 scope 内。对聊天机器人 serving 这是对的：tool 调用罕见。

**对 agent 这个假设崩**。论文枚举失败模式（Section 1）：

> "While prior approaches on AI efficiency aggressively focused on GPU kernels and KV-cache management, they become ineffective for the CPU-centric tool execution of the agentic AI workloads. A recent work shows that ENNS accounts for more than 75% of the end-to-end (E2E) latency on a 200 GB document corpus for a Retrieval Augmented Generation (RAG) workload with a Llama-3-70B."

论文的贡献是在代表性 workload 套件上让这点具体起来，并把测量转成调度算法。

### 编译期 characterization（Section 2）

作者提出 agentic 系统的**三轴分类法**：

| 轴 | 取值 | 含义 |
| --- | ---- | --- |
| **Orchestrator** | LLM-orchestrated / Host-orchestrated | 谁决定下一个 tool —— LLM（ReAct/AutoGPT 风格）或 Python 代码（LangChain/Haystack 风格）|
| **Path** | Static / Dynamic | Workflow 设计时固定（Haystack）或运行时由 LLM 选择（Reflexion、LATS）|
| **Repetitiveness** | Single-step / Multi-step | Agent 一次前向（RAG QA）或迭代 loop（WebArena、Balrog）|

这个分类法是 *先验的*（平台无关），用来挑跨 corner 的五个 workload。

### 五个代表性 workload（Table 2）

| Workload | Orchestrator | Path | Flow | Tool | 应用 |
| -------- | ------------ | ---- | ---- | --- | ---- |
| **Toolformer** | LLM | Dynamic | Single-step | Calculator API、Calendar | MLQA、数学 |
| **SWE-Agent** | LLM | Static | Multi-step | Bash File I/O、Python | SDE、数据分析 |
| **RAG (Haystack)** | Python (Host) | Static | Single-step | Web 搜索、文档检索 | RAG QA |
| **ChemCrow** | LLM | Dynamic | Multi-step | Conformer Gen、反应工具 | 化学研究 |
| **Web-Augmented Agent (LangChain)** | Python (Host) | Static | Single-step | Web 搜索、Summarizer | Web QA、DevOps |

用的模型是小 LM（≤32B），justify 为适合生产 agentic 部署（大多数 agent 步骤不需要旗舰 LLM 智能）。

### 测试硬件（Table 1）—— 不对称是重点

作者故意挑**不对称 CPU-GPU 系统**来 isolate 瓶颈实际在哪里：

| 组件 | Sys 1 (HP CPU + LP GPU) | Sys 2 (LP CPU + HP GPU) |
| --- | ----------------------- | ----------------------- |
| **CPU** | 64-core Intel Granite Rapids | 72-core Nvidia Grace |
| CPU 内存 | 512 GB DDR5 | 480 GB LPDDR5 |
| **GPU** | Nvidia RTX-Pro 6000 Blackwell | Nvidia H200 |
| GPU 内存 | 96 GB GDDR7 | 96 GB HBM3e |

跨两系统对比同 workload，看瓶颈何时 CPU↔GPU 转移。

## 三个核心组件详解

### 组件 1 —— 运行时 characterization（Section 3）

论文实证核心。Figure 2 画了所有 5 个 workload × 2 系统 × 多个 benchmark 输入的 E2E 延迟，分为 LLM 推理（GPU）和 tool 处理（CPU）。六个头号观察：

**(a) RAG / Haystack** —— ENNS 检索主导：NQ、HotpotQA、TriviaQA 上 **Sys 1 上 83%、81%、82%**；**Sys 2 上多达 89%**（LP CPU）。Sys 2 的 HP GPU 让 LLM 更快但 CPU 成约束。

**(b) Toolformer** —— Sys 1 上 LLM 主导（**~88% LLM 推理**），Sys 2 上掉到 **77%**（HP GPU）。WolframAlpha API 调用 CPU 成本可忽略。

**(c) Web-Augmented Agent (LangChain)** —— LexRank summarization 在 Sys 1 上 **55%（freshQA）和 48%（QASC）**；URL fetch 方差主导。

**(d) SWE-Agent** —— Bash + Python 执行在 Sys 2 上 **38%（APPS）、25%（BigCodeBench）**，但在某些 Sys 2 配置下多达 **65% 的 E2E 延迟**。Sys 2 上良好优化的 LLM 强制瓶颈到 CPU。

**(e) ChemCrow** —— Conformer 生成（RDKit）主导，**Heavy 85%、Sys 1 Heavy 88%**；Medium 分子 LLM 主导（Sys 1 上 58%，Sys 2 上 53%）。

> [!important] "Tool dominates" / "LLM dominates" 问题没有单一答案
> Tool 还是 LLM 主导取决于（1）哪边硬件更好、（2）特定 tool 的计算 profile、（3）输入特性（小分子 → LLM-bound，重分子 → ChemCrow 的 tool-bound）。论文贡献是在代表性 spread 上 *实证* 量化这点，而不是挑一个 workload 过度泛化。

**Key Takeaway 1（论文原话）**："Tool processing on CPUs can take significant chunk of E2E latency, motivating a CPU-centric optimization strategy."

**Key Takeaway 2**："HP CPU system can shift the bottleneck from GPU to CPU when tool execution latency is comparable to LLM inference latency, making them more CPU-bounded systems with LP GPU, motivating system-aware optimization strategies."

**Key Takeaway 3（Section 3.3）**："CPU-parallelization strategies fundamentally exhibit lower efficiency compared to GPU. In agentic AI workloads, they prematurely saturate the throughput, subsequently bottle-necking the system and degrading the utilization of costly GPU resources."

### 组件 2 —— COMB：CPU-Aware Overlapped Micro-Batching（Section 4.1）

第一个算法贡献。关键观察：**CPU 效率在 $1\text{-}2 \times \#\text{核心}$ 活跃进程处饱和**；超过这个 OS 争用主导。同时，小 batch 上 GPU 效率 *也* 受限（低利用率），但上升得陡得多。

**COMB 的两个设计动作：**

1. **Cap micro-batch 大小** 在 $B_{cap} \approx 1\text{-}2 \times \#\text{CPUs}$（从 CPU 并行效率饱和点推出）。对入站大 batch $B_{max} > B_{cap}$，切成 ceiling($B_{max} / B_{cap}$) 个 micro-batch。
2. **Overlap 相邻 micro-batch** —— 在 overlap interval $s$ 后，micro-batch $i$ 执行 GPU 阶段（LLM）时，micro-batch $i+1$ 执行 CPU 阶段（tool）。这是跨阶段 *流水线* 执行，不是阶段内。

可视化（论文 Figure 4）：

```
多进程 baseline (BS=128)：
  CPU 0-128: [████████████ tool ████████████]  ←  64 cores 过订阅 2×
  GPU:                                            [██████████ LLM ██████████]

COMB ($B_{cap}=64$, $s$ = 16s)：
  CPU 0-31:  [██████████ μ1 tool ██████████]
  CPU 31-63:                                  [██████████ μ2 tool ██████████]
  CPU 64-95:                                  [██████████ μ3 tool ██████████]
  CPU 96-128:                                                        [██ μ4 tool ██]
  GPU:                              [██ μ1 LLM ██][██ μ2 LLM ██][██ μ3 LLM ██][μ4]
                                    ↑
                                    s=16s overlap with μ1 CPU 之后开始
```

### 具体省什么

CPU 过订阅成本真实。来自 Table 3：

| Workload | Sys 1 r(64) | Sys 1 r(128) | Sys 2 r(64) | Sys 2 r(128) |
| -------- | ----------: | -----------: | ----------: | -----------: |
| Web-Augmented Agent (LangChain) | 1.94 | 1.00 | 1.76 | 1.05 |
| SWE-Agent | 1.43 | 1.18 | 1.45 | 1.15 |

其中 $r(BS) = T(BS = 2^n) / T(BS = 2^{n-1})$ 是吞吐增益比。值 $\approx 1$（如 LangChain BS=128 两系统）意味着 batch 翻倍**吞吐不改善** —— 确认 CPU 饱和。COMB 的 $B_{cap}$ 选择本质上是 "$r(BS) \approx 1$ 之前的最大 batch"。

**效果跟 $r(BS)$ 反相关**：当 $r(BS) \approx 1$（饱和），COMB 带来 ~2× P50 延迟降低；$1 < r(BS) < 1.5$（部分饱和），单 micro-batch 效果较小但 overlap 补偿。

> [!example]- COMB 开环测量（Section 5.2）
>
> Poisson 速率 $\lambda$ 的开环到达。Baseline 用 $N_{max}=256$（最大 GPU 利用）。COMB 测 $N_{cap}=64$（吞吐曲线膝点）。
>
> 在 $\lambda = 13$ req/s 时，baseline CPU 利用率 $\rho_{CPU} = 3.09$（过订阅 3×），CPU 争用主导，排队延迟飙升。COMB at $N_{cap}=64$ 把 $\rho_{CPU}$ 控制在 0.89–1.13 范围，CPU 保持接近最优负载。
>
> 结果：
> - 服务延迟 P50/P90：vs baseline **低 2.9× / 3.9×**
> - 总延迟 P50/P90：**低 1.6× / 1.8×**
> - 吞吐：**高 1.7×**
>
> 更小 $N_{cap}$（48）增加服务时间（under-utilize）；更大值（82、96）减少 $N_{cap}$ 益处。

### 组件 3 —— MAS：Mixed Agentic Scheduling（Section 4.2）

第二个算法贡献，给异质 workload（请求有不同 CPU/GPU 瓶颈）。

**MAS 针对的问题**：标准 FCFS 调度（vLLM、SGLang）一队列接纳请求。聊天服务器收到纯 LLM chat 请求（GPU-heavy）跟 tool 用 agent 请求（CPU-heavy）混合时，主导请求类型垄断接纳。少数类型在队列里 head-of-line 阻塞。

**MAS 的两个策略：**

1. **请求类型感知并发接纳**：CPU-heavy 和 GPU-heavy 请求分别有执行队列上界 $E_{cap,CPU}$ 和 $E_{cap,GPU}$。在每个上界内，请求类型不能超过其分配；一种类型 burst 不饿死另一种。
2. **共享预留队列 $E_{cap,shared}$** 给弹性吸收。超过类型上界的请求溢出到这个共享队列，提供弹性同时不损害少数类型的保护。

上界从 COMB 分析推出：$E_{cap,CPU} = N_{cap}$ 来自 COMB on CPU-heavy workload；$E_{cap,GPU}$ 是剩余并发预算分给 GPU-heavy。

**实证评估**（Sys 2，请求类型概率 $p_{LLM} \in \{0.25, 0.50, 0.75\}$）：

| $p_{LLM}$ | Workload 类型 | P50 提升 | P90 提升 |
| --------: | ------------- | -------: | -------: |
| 0.25 | GPU-heavy 少数 | **低 1.82×** | **低 1.78×** |
| 0.25 (Sys 1) | GPU-heavy 少数 | **低 2.37×** | **低 2.49×** |
| 0.50 | GPU-heavy | 低 1.39× | 低 1.18× |
| 0.50 | CPU-heavy | 低 1.1× | 低 1.1× |
| 0.50 (Sys 1, avg) | 总 | **低 1.62×** | **低 1.30×** |
| 0.75 | CPU-heavy 少数 | 低 2.09× | 低 2.15× |

少数请求类型受益最大 —— MAS 保护 head-of-line 饥饿。

> [!note]- MAS 为什么重要：chat+agent workload 混合
>
> 生产聊天服务越来越同时承载纯对话请求（无 tool）和 agentic 请求（带 tool 调用）。没 MAS，主导类型（哪个到达多哪个）主导接纳，少数类型看到 P50/P90 延迟膨胀，尽管它们用的是 *不同资源瓶颈*。
>
> MAS 的贡献是认识到 agentic 请求是跟 chat 请求 *根本性不同的调度类*。它们需要自己的接纳纪律。这泛化到 LLM serving 之外：任何请求在不同资源上竞争的异质 workload 系统都受益于同样原则。

## 头号实验证据

### COMB standalone batch（Section 5.1）

Standalone batch 处理 BS=128，对比 baseline（MP 全 128）、micro-batching（$B_{cap}=64$）、COMB（$B_{cap}=64$，varying $s$）：

| Workload | System | Baseline (s) | Micro-batching (s) | COMB best (s) |
| -------- | ------ | -----------: | -----------------: | ------------: |
| SWE-Agent | Sys 1 | 61.4 | 71.4（更糟）| **53.3** |
| SWE-Agent | Sys 2 | 38.5 | 55.9（更糟）| **45.1** |
| Web-Augmented Agent | Sys 1 | 20.3 | 21.4（相似）| **21.1** |
| Web-Augmented Agent | Sys 2 | 14.2 | 16.5（更糟）| **15.6** |

注意：单 micro-batching *可能伤害*（SWE-Agent on Sys 2：55.9s vs baseline 38.5s）当 CPU 没饱和；COMB 通过 overlap 恢复。SWE-Agent 在 $BS=128$ 时 $r(BS) \approx 1.15$ 意味着单 micro-batching 低效。

### 开环 COMB 和 MAS（Section 5.2 + 5.3）

已在头号表中总结。关键 takeaway：

- COMB 提升随 CPU 负载缩放：$\lambda \approx 11\text{-}14$ req/s 时吞吐提升 1.7×；低于此益处更少。
- MAS 保护少数请求类型但不伤主流 —— 对异质混合 Pareto-improving。

### CPU 受限平台上的 ablation（Section 5.4）

第三系统：**16-core Intel Emerald Rapids + 同 RTX-6000 Pro Blackwell GPU**。CPU 是 Sys 1（64-core）的 1/4 容量但 GPU 一样。测 COMB/MAS 在 CPU 是紧瓶颈时是否仍 work：

| 配置 | Web-Augmented Agent first-batch 时间 |
| ---- | -----------------------------------: |
| Baseline | 51.5 s |
| Micro-batching | 26.4 s（1.95× 更快）|
| COMB ($s=16$s) | **40.1 s**（比 micro-batching 更糟）|

在 16-core CPU 上，$r(BS) \approx 1$（已饱和），所以单 micro-batching 抓住收益；COMB 的 overlap *减少* 总时间但有尾延迟代价（~1.05× P90）。这是有界域 —— 论文说 COMB 有效性 "跟增益比反相关"。

> [!example]- 能耗 profile —— 次要贡献
>
> Section 5.4.2（上面没详细显示）跨 workload 测量 CPU 和 GPU 能耗。发现：agentic AI 有相当 CPU 动态能耗 overhead 超出 GPU 中心研究报告。直接量化数字：总 CPU 能量从 20%（LLM-dominated）到 70%+（tool-dominated）of E2E 系统能量。可持续 serving 部署的影响：CPU 功耗优化比当前框架认识到的更重要。

## 优势与局限

**优势。**

- **第一个端到端测量** 跨代表性 agentic workload 套件在非对称硬件上。编译期分类法是独立于特定论文的有用概念工具。
- **真正算法贡献**（COMB + MAS），不只是 characterization。两者都足够简单实现，足够有原则可辩护。
- **量化 CPU 并行效率** —— 实证观察 CPU 在 $1\text{-}2 \times \#\text{核心}$ 处饱和，可复用到 agentic serving 之外。
- **区分服务延迟跟总延迟**：总 = 服务 + 等待。许多先前论文只报服务延迟；MAS 的等待延迟降低是可分离贡献。

**局限。**

> [!warning] Tool 选择偏向"令人尴尬可并行的 CPU 工作"
> 选的五个 workload（Toolformer、SWE-Agent、RAG、ChemCrow、Web-Augmented Agent）都有定义良好的 CPU 瓶颈（检索、RDKit、summarization、Bash）。带**网络主导 tool**（WebFetch、跨 DC 向量 DB、慢第三方 API）的现代 agentic workload 有 *不同* 瓶颈结构 —— 既非 CPU 也非 GPU bound，只是 *I/O 等待*。COMB 不适用于 I/O-等待主导 tool。

- **小模型（≤32B）**。CPU/GPU 比在旗舰规模（300B+）剧烈变化。COMB 的 $B_{cap}$ 是否 scale 到大模型没测。
- **只闭环和开环**，无真实生产流量回放。合成 Poisson 到达比 CPU-Centric paper 自己 MAS 用的 on/off bursty pattern（two-state ON/OFF 模型）更简单。
- **MAS 评估用合成 $p_{LLM}$ 混合** with two-state ON/OFF 到达。真实 chat+agent workload 有相关 burst（用户 session vs agent 活动），这个没捕捉。
- **没跟 [[continuum|Continuum]] 对比**，后者从 KV cache TTL 角度攻同样问题（多轮 agent 调度）。COMB + Continuum 大概可组合但论文没 address。
- **截至 2026-06 无代码发布**。

> [!bug] CPU 效率因子估计是实证的
> $B_{cap} \approx 1\text{-}2 \times \#\text{CPUs}$ 启发式来自测过 workload 的离线 profile。新 workload 或硬件必须重新 profile。论文不提供在线估算器。

## 这意味着什么

这篇论文一次做两件有用事：**诊断** agentic serving 中 CPU 瓶颈实际在哪（量化 workload + 硬件敏感性），以及 **处方** 两个调度算法关闭 gap。诊断是更持久的贡献；一旦每个人同意失败模式，替代调度算法就会跟上。

2027 年的三个预测：

1. **COMB 风格 CPU-GPU overlap micro-batching 进入 vLLM/SGLang 作为一等调度模式**，可能作为 `--enable-cpu-tool-overlap` with 自动检测 $B_{cap}$。
2. **MAS 风格请求类型感知接纳成为默认调度器**给任何同时承载 chat 和 agent 流量的部署。来自 chat-only 部署的"单 FCFS 队列"模式在异质 workload 下不能存活。
3. **"Tool 主导 70-90%" claim 变细。** 它在特定 workload 配特定硬件上为真（RAG on Sys 2、ChemCrow Heavy）。在其它上为假（Toolformer on Sys 1 = LLM-bound）。未来 characterization paper 会采用本文的 per-workload-per-hardware 报告风格而不是单一数字总结。

这篇论文**不**解决：

- **内存带宽瓶颈** —— 焦点是 CPU 计算，不是内存带宽。不同优化故事。
- **多轮 KV cache 保留** —— 跟 [[continuum|Continuum]] 的 TTL pinning 正交且可组合。
- **分布式 serving** —— 单节点；多节点 COMB 协调未探索。
- **成本感知调度** —— 测了能耗但没作为调度输入。

## 源代码与复现

```bash
# 截至 2026-06 代码未发布。
# 实现建立在：
pip install vllm==0.14.0 torch==2.8.0
```

**复现协议**（Section 3.1）：

| 组件 | 配置 |
| ---- | --- |
| vLLM | 0.14.0 |
| PyTorch | 2.8.0 |
| Sys 1 | 64-core Intel Granite Rapids + Nvidia RTX-Pro 6000 Blackwell，512 GB DDR5 |
| Sys 2 | 72-core Nvidia Grace + Nvidia H200，480 GB LPDDR5 |
| 模型 | GPT-OSS-20B、GPT-J-6B（vLLM）、Qwen2.5-Coder-32B（vLLM）|
| Tools | ENNS 检索、WolframAlpha API、URL fetch、LexRank summarization、Bash+Python、RDKit conformer gen |
| Workload | Toolformer、SWE-Agent、RAG（Haystack）、ChemCrow、Web-Augmented Agent（LangChain）|
| 每次跑 | 5 次试验做统计方差 |

## 相关阅读

- [[continuum]] —— Continuum：互补的 KV cache TTL 系统给多轮 agent 调度。COMB 和 Continuum 正交（一个 address CPU 过订阅，另一个 address KV cache 驱逐）；组合它们能同时攻击两个瓶颈。
- [[agent-serving-challenges]] —— 更广 agent serving 挑战调研；本文的 characterization 数据是那里引用的最严谨数据点。
- [[multi-turn-optimization]] —— 多轮 KV 复用；跟这里 CPU 侧优化正交。
- [[vllm]] —— 测量用的 base 推理引擎（vLLM 0.14.0）。
- [[continuous-batching]] —— 本论文用 COMB 扩展的标准 batching 策略。
- [[paged-attention]] —— KV cache 原语；论文指出 "PagedAttention reduce memory fragmentation but they do not eliminate the underlying capacity and bandwidth limits of GPU memory."
- [[prefill-decode-disaggregation]] —— 另一个 batching/调度轴；PD-disagg 和 COMB 可组合。

## 参考文献

- Ritik Raj, Souvik Kundu, Ishita Vohra, Hong Wang, Tushar Krishna. *Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective.* arXiv:2511.00739, 2025 年 11 月（v3 2026 年 4 月）. https://arxiv.org/abs/2511.00739
- ENNS 检索瓶颈参考：Wang et al., RAG-200GB 研究（论文引用 [55]）。
- WebArena CPU 延迟分析：Yao et al.（论文引用 [73]）—— "partial tool execution can cut request latency by up to 38.8%, highlighting tool execution as a major source of E2E latency."
- LMStream micro-batching for stream processing（引用 [38]）。
- Ayo stage-local micro-batching（引用 [63]）—— cross-stage overlap 的前作。
