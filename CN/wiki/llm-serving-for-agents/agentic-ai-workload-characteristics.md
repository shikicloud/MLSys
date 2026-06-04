---
title: "Agentic AI 工作负载特性：ReAct Agent 的端到端 trace"
category: llm-serving-for-agents
tags: [agentic-ai, workload-characterization, react, opentelemetry, vllm, claude-code, qwen, gemma, swe-bench, terminal-bench, gaia, paper-review]
created: 2026-06-03
updated: 2026-06-03
status: mature
paper: arXiv:2605.26297
---

# Agentic AI Workload Characteristics

> [!info] 论文元数据
> - **论文**：[arXiv:2605.26297v1](https://arxiv.org/abs/2605.26297) —— *Agentic AI Workload Characteristics*，2026-05-25
> - **代码**：未发布
> - **作者**：Yichao Yuan（UIUC）、Ankita Nayak（Gimlet Labs）、Souvik Kundu（Intel）、Nishil Talati（UIUC）
> - **联系**：`{yichaoy2, nishil}@illinois.edu`
> - **实现**：vLLM v0.20.0、Claude Code agent 框架、Harbor 评估环境、通过 Jaeger 的 OpenTelemetry tracing

> [!important] 这篇论文是什么、不是什么
> 这是 2026 年**最重要的**端到端 agentic 工作负载 characterization 论文 —— 在真实 ReAct 风格 agent（Claude Code）跑真实 benchmark（SWE-Bench Pro、GAIA、Terminal-Bench）上做单 agent tracing，带 OpenTelemetry 级 per-turn / per-tool / per-token instrumentation。它是 [[continuum|Continuum]]、[[cpu-centric-agentic-ai|CPU-Centric Perspective]]、[[rose|ROSE]] 都建立其上或对比的**测量参考**。**不是**优化论文 —— 不提系统，只测量。贡献是数字、方法论、和大约十个综合 insight 重新校准如何思考 agent serving。

---

## 摘要（2 分钟读完）

**它是什么。** 一篇 ReAct 风格 LLM agent 跑真实 benchmark 的测量研究，从请求进入、tool 执行到下一轮 LLM 输入全程 instrumented。作者用 **Claude Code** 编排两个 dense LLM（**Qwen3.6-27B** + **Gemma4-31B**，Thinking 和 Instant 两个变体）跑 **5 个 benchmark**（ADE-Bench、DABStep、GAIA、SWE-Bench Pro、Terminal-Bench 2.0）在 2×H100 NVL + Intel Xeon 平台。每个（model, benchmark, mode）组合收 ~100 个 task trace。输出是 *生产 agentic serving 实际样子* 的结构化目录，在 turn、token、prefill/decode 阶段、cache 复用、tool 调用类型、失败模式等粒度上。

**核心 idea。** **Agentic LLM serving 不是传统长 prompt serving —— 是重复模型 re-entry over 增长的 cached context，tool 执行是小但 workload 相关的延迟尾巴。** 这个 thesis 的四个子部件：

1. **Decode 主导 91–98.6%** of LLM 时间，不是 prefill（1.4–9.0%）。之前 "长上下文 → 贵 prefill" 的假设误框 agentic workload。
2. **Context cache 复用率 84.6–99.5%** 实证 —— 每轮大部分输入 token 来自先前 cache。失去 KV state 把 decode 主导的 workload 变成贵的重新计算。
3. **Tool 执行 2–29%** of E2E 时间（LLM 71-98%），但变化大跟 workload 相关：GAIA 达 28.7%（重 WebFetch），DABStep 保持 2-4%（LLM-bound），SWE-Bench Pro 和 Terminal Bench 17.6-17.7%。
4. **推理减少病态 trajectory**，不只是 per-turn 成本 —— Gemma Thinking on ADE-Bench 平均 18 turn vs Gemma Instant 的 108.8（最差 786 turn），因为 thinking 防止 agent 陷入 edit-failure 循环。

去掉任一观察就误设计 serving 系统：优化 prefill 是误导的（1.4-9% 时间）、失去 KV state 是灾难性的、只优化 LLM 忽略 20-30% 时间、忽略 reasoning 对 workload 形状的影响低估其 serving 价值。

**头号数字。** 从图表里最被引用的：

| 指标 | 值 | 备注 |
| --- | -- | --- |
| LLM 时间 % of E2E | **71–98%** | 跨 (workload, model variant) 范围 |
| Tool 时间 % of E2E | **2–29%** | Workload 相关 |
| GAIA tool 时间 | **28.7%**（Gemma Thinking）| WebFetch / WebSearch 重 |
| SWE-Bench Pro tool 时间 | **17.7%** | Bash + Read + Edit 主导 |
| Terminal-Bench tool 时间 | **17.6%** | Bash + 脚本执行 |
| DABStep tool 时间 | **2–4%** | 几乎纯 LLM 计算 |
| Decode % of LLM 时间 | **91–98.6%** | Decode 主导 prefill |
| 实证 KV cache 命中率 | **84.6–99.5%** | 多数输入跨轮复用 |
| Append-to-output 比（中位数）| **<1.5×** | per turn 小新输入 vs 响应 |
| 平均 turn：Gemma Thinking on ADE | **18.0** | 推理紧凑轨迹 |
| 平均 turn：Gemma Instant on ADE | **108.8**（最差 786）| Edit 失败病态 |

**为什么重要。**

- **"tool=70%" 或 "tool=2%" 辩论都错。** Workload+硬件决定比率；生产编程 agent on dense model 实际范围是 **17-30%**，[[cpu-centric-agentic-ai|CPU-Centric Perspective]] 的 88% 是给故意 compute-heavy tool（200GB RAG、RDKit），不在典型 agent set 里。
- **优化 prefill 是误导的** —— 91-98.6% LLM 时间是 decode，所以 decode 上的 kernel/量化优化比 chunked-prefill 调优对 agent 多远多多。
- **推理模型有 workload 级益处** 超出 per-turn 准确性：减少 turn count、累积上下文、失败级联病态。Serving 系统设计必须 account for 这点。
- **2027 年预测。** 这篇成为 "agent workload characterization" 在任何 agentic serving 论文里的标准引用，就像 "ShareGPT trace" 在 2023-2024 是 chatbot characterization 的标准。预期新 agent serving 系统按这论文的 per-turn / per-cache / per-tool 分解报告它们的数字。

---

# 详细内容（深入阅读从这里开始）

## 背景：为什么 agentic workload 不是 chatbot workload

论文开头（Section 1）枚举把 agent serving 当 chatbot serving 误用的结构性差异：

> "Agentic AI shifts LLM serving from isolated prompt-generation requests to stateful, multi-turn executions that repeatedly invoke the model, call tools, and grow context over time."

三个具体失败模式：

1. **"长 prompt" 是错的抽象。** Chatbot serving 每请求看一个长 prompt；agentic serving 看 *增量* over 累积 cached context。优化长 prompt prefill 错过这点。
2. **Turn 数和上下文长度是不同指标。** 有些 workload（ADE Instant）有 100+ turn 但 per turn 上下文 modest；其它（SWE-Bench Pro）turn count 中等但上下文巨大。两者 stress serving stack 不同部分。
3. **Tool 执行异质长尾。** Tool 延迟从毫秒（`grep`）到分钟（`Agent` 委托、慢 API）。单一统一 "tool 延迟" 估计抓不住这点。

论文的 agent 形式化模型（Section 2.3）：

对 agent $a$ 在 step $i$，记 $H_{a,i}$ 是累积上下文（轨迹），$C_{a,i} = |H_{a,i}|$ 是上下文长度。LLM 产生三部分响应：
$$z_{a,i} = (\theta_{a,i}, m_{a,i}, u_{a,i})$$
其中 $\theta_{a,i}$ = thinking token，$m_{a,i}$ = message token，$u_{a,i}$ = tool-call token。环境然后 append tool 结果 $o_{a,i}$ 上下文演化：
$$H_{a,i+1} = H_{a,i} \| \Phi(\theta_{a,i}, m_{a,i}, u_{a,i}) \| o_{a,i}$$

其中 $\Phi(\cdot)$ 是 chat-template formatting，$\|$ 是 concatenation。**一个 agent turn 是一个 $(\text{LLM call}, \text{tool result})$ 对**。Subagent（通过 `Agent` tool 启动）被当作父级 tool 调用的一部分。

## 方法论细节

### 硬件和软件（Section 3.1）

| 组件 | 配置 |
| ---- | --- |
| GPU | 2× NVIDIA H100 NVL 由 12 NVLink 连接 |
| CPU | Intel Xeon Platinum 8592+ |
| 推理引擎 | vLLM v0.20.0，TP=2 |
| Tracing | vLLM OpenTelemetry 支持 → Jaeger |
| Agent 框架 | Claude Code |
| 评估环境 | Harbor（单 agent 执行，per task 目标 + tools）|

选 vLLM 0.20.0 + TP=2 是 deliberate：这配置代表 27-31B dense 模型的生产部署。

### 三组件 characterization 基础设施（Figure 2）

```
Harbor → Dockerized Claude Code agent
     ↓
LLM 请求
     ↓
Forwarding proxy（记录请求元数据，修改请求确保 OTEL trace）
     ↓
vLLM 服务器
     ↓
Jaeger 收集 OpenTelemetry trace（prefill/decode/queue 时间）

Per-agent wrapper 分配独特 API key → 按 agent 执行分组 trace
```

Forwarding proxy 是最重要的设计选择：坐在 agent 和 vLLM 之间，截每个 LLM 调用，确保 vLLM 发 per-request OpenTelemetry trace 带 prefill/decode 时间。这是拿到论文 per-turn / per-phase 分解的唯一办法。

### Workload 和配置（Section 3.1）

**模型**（Thinking 和 Instant 变体都测）：
- Qwen3.6-27B
- Gemma4-31B

**Benchmark**（每个采样 100 task）：
- ADE-Bench —— 数据分析
- DABStep —— 数据 agent 任务
- GAIA —— 通用助手（多步 + tool 使用）
- SWE-Bench Pro —— 软件工程
- Terminal-Bench 2.0 —— 终端软件工程 + 系统交互

这是**每 workload 8 个 cell**（Gemma-T/I × Qwen-T/I × 100 task），所以 ~4000 个 task trace 总共。

## 三个核心组件详解

### 组件 1 —— Agent 执行 characterization（Section 4）

四个子问题：多少 turn、多少上下文、输出组成是什么、失败/成功跑不同吗？

#### Turn 数（Figure 3）

| Workload | Gemma-T 平均 ± std | Gemma-I 平均 ± std | Qwen-T 平均 ± std | Qwen-I 平均 ± std | 最差 turn（任何）|
| -------- | -----------------: | -----------------: | -----------------: | -----------------: | ---------------: |
| ADE | 18 ± 15 | **109 ± 179** | 19 ± 32 | 17 ± 17 | **786**（Gemma-I）|
| DABStep | 16 ± 6 | **640** | 17 ± 8 | 16 ± 11 | 640（Gemma-I）|
| GAIA | 22 ± 33 | 617（P75 ~33）| 31 ± 26 | 17 ± 43 | 617 |
| SWE-Bench Pro | 31 ± 20 | 24 ± 23 | **41 ± 19** | 26 ± 19 | 226 |
| Terminal Bench | 74 ± 26 | 26 ± 62 | 25 ± 23 | 25 ± 43 | 197 |

两个 takeaway：
- **长尾跟模型相关，不只 workload 相关**。Gemma Instant on ADE 达 786 turn；Gemma Thinking on ADE 保持 18。原因（§4.1）：Gemma Instant on ADE 做 2,757 个失败 `Edit` 调用（95.4% 失败率），陷入 retry 循环。
- **"推理对 Gemma 益处最强"** —— Thinking 在 ADE 上减平均 turn 6×（18 vs 109）。对 Qwen，Thinking 只 modestly 更好（有时更差 —— SWE-Bench Pro 上 41 vs 26）。

#### 上下文长度（Figure 4）

| Workload | 中位上下文（K token）| 最大上下文（K token）|
| -------- | ------------------: | ------------------: |
| ADE | 37–43 | 多达 171（Gemma-I）|
| DABStep | 44–48 | 多达 126 |
| GAIA | 38–54 | 多达 174（Qwen-I）|
| **SWE-Bench Pro** | **69–80** | **146–166**（最大）|
| Terminal Bench | 45–66 | 多达 167 |

> [!important] 上下文增长 ≠ turn 数
> SWE-Bench Pro 上下文最大（平均 69-80K）但 turn 数只中等（平均 26-41）。反之 ADE 有最长 turn 数尾（786 turn）但上下文 modest（平均 37-43K）。**这是独立失败模式** 给 serving 系统设计 —— 高 turn 数 stress 调度，大上下文 stress KV cache 内存。

#### 输出 token 组成（Figure 5）

生成 token 分解（thinking / message / tool-call）：

| 模型变体 | Thinking % | Tool-call % |
| ------- | ---------: | ----------: |
| **Gemma Thinking** | 45.8–67.6% | 18–48% |
| **Gemma Instant** | ~0% | **87.8–98.2%** |
| **Qwen Thinking** | 29.0–40.7% | 48.2–63.7% |
| **Qwen Instant** | ~0% | **70.4–81.6%** |

> [!quote] 这推翻什么
> "ReAct-style agents are not simply producing long natural-language responses; much of their generated output is structured action generation that drives interaction with the environment."

Agent 解码的大部分是**结构化 tool-call token**（JSON、XML、function-call 格式），不是 prose。这对受约束生成基础设施（[[multi-turn-optimization|结构化输出]]）和要优化什么 token 有直接影响。

#### 成功 vs 失败（Figure 6）

> "**失败 agent 通常积累比成功 agent 更大的上下文**，虽然此效应强度因 workload 和 model 而异。"

GAIA 失败达 34.6K-64.7K token vs 成功 25-47K。Terminal Bench：失败 76-79K vs 成功 44-46K。SWE-Bench Pro 对成功失败都重上下文（Gemma Thinking 成功 82.2K vs 失败 72.3K —— 反例，成功跑*更长*因为合法 SWE 工作需要持续上下文）。

**失败机制**（Table 2）：失败的 Bash/Edit/Read 动作 append 错误信息和 stack trace，成为下一轮输入上下文一部分。失败跑因此在两维放大负载：更多 turn 加更大上下文。

### 组件 2 —— 运行时性能 characterization（Section 5）

重新校准 serving 系统设计的三个测量。

#### Section 5.1 —— LLM 跟 tool 调用时间（Figure 7）

这是最被引用的结果。

| Workload | LLM % | Tool % |
| -------- | ----: | -----: |
| DABStep | **96–98%** | 2–4% |
| ADE | 86–92% | 8–14% |
| SWE-Bench Pro | 82.3% | **17.7%** |
| Terminal Bench | 82.4% | **17.6%** |
| **GAIA（Gemma Thinking）** | 71.3% | **28.7%**（最高）|
| GAIA（Qwen Thinking）| 75.1% | 24.9% |

> [!quote] 头号框架
> "Across all benchmarks and models, execution is primarily LLM-dominated, with LLM inference accounting for 71–98% of total runtime. ... However, tool execution is not negligible. Tools account for 2–29% of total runtime, with the largest fractions on GAIA, where tool time reaches 28.7% for Gemma Thinking and 24.9% for Qwen Thinking. This aligns with Table 1 where GAIA includes expensive WebFetch, Agent, and TaskOutput calls."

论文在 §5.1 末段承认 **并发会移这些数字**："While the degree of concurrency (i.e., number of agents running concurrently) may change the LLM- vs. -tool time, our results are in line with other works." 这是承重 caveat —— 这些测量是给 *单 agent 执行*。多 agent 批 serving 拉长 LLM 调用排队时间改变比例。

#### Section 5.2 —— Context cache 有效性（Figure 8）

| Workload | 实证 cache 命中率 | 理论 cache 命中率 | 平均 append-to-output 比 |
| -------- | -------------: | ---------------: | ----------------------: |
| ADE | 84.6–99.5% | 87.9–99.3% | 1.5–4.5×（中位数 <1.5）|
| DABStep | 类似 | 类似 | 1.0–1.4× |
| GAIA | 类似 | 类似 | 1.0–1.3× |
| SWE-Bench Pro | 类似 | 类似 | 0.4–1.4× |
| Terminal Bench | 类似 | 类似 | 0.5–2.3× |

**两个后果**：

1. **Agent serving 不是 "重复长 prompt prefill"** —— 每轮只 append 响应长度的 ~1-2× 新输入 token。上下文其它部分从 cache 复用。
2. **失去 KV state 是灾难性的** —— 没有效 caching，turn 30 的 80K-token agent step 会重新 prefill 80K token，把 decode 主导的 workload 变成 prefill 主导。这正是为什么像 [[continuum|Continuum]]（TTL-based KV pinning）、[[multi-turn-optimization|SGLang RadixAttention]]（prefix cache）、LMCache（CPU/disk offload）这种系统是承重的。

> [!quote] 重新框架 agent serving
> "Agent execution should be modeled as a sequence of repeated model re-entry over a growing cached context, but as repeated decode over a growing cached context, rather than repeated full-context prefilling."

#### Section 5.3 —— Prefill 跟 decode 时间（Figure 9）

| Workload | Prefill % of LLM 时间 | Decode % of LLM 时间 |
| -------- | ---------------------: | -------------------: |
| 所有 workload | **1.4–9.0%** | **91.0–98.6%** |
| 最大 prefill 份额 | Gemma Thinking on SWE-Bench Pro：**9.0%** | — |
| 最小 prefill 份额 | DABStep：**~1.4%** | 98.6% |

这是 serving 优化优先级最有影响的单一数字：**decode 是 91-98.6% of LLM 时间，所以 decode 上的 kernel/量化/内存带宽优化提供远多杠杆比 prefill 优化**。PagedAttention、FlashAttention-2/3 decode kernel、KV cache 压缩 —— 这些都 target decode。Chunked prefill（Sarathi-Serve）优化 1.4-9% 切片。

### 组件 3 —— Tool 调用 characterization（Section 6）

更细粒度看哪些 tool 类型主导和它们模式如何跨 agent 生命周期 shift。

#### Tool 分解 by 类型（Figure 10）

Tool 组成跨 workload 比跨 reasoning mode 变化更多：

| Workload | Top tools |
| -------- | --------- |
| ADE | Read 29.8-41.7%、Bash 23.2-30.0%（Gemma-I：Edit 51.2%）|
| DABStep | Bash 50.5-87.8%（重）—— coding 任务依赖命令行 |
| GAIA | WebFetch 42.3%（Gemma-T）、WebSearch 29.7%（Gemma-I）—— web 工具主导 |
| SWE-Bench Pro | Edit + Bash + Read 都 substantial，with workload 特定混合 |
| Terminal Bench | Bash 重（Qwen 上 53.2-79.5%）—— 终端命令 |

#### Bash 命令分解（Figure 11）

Bash 不是单一 tool —— agent 发的命令差异大：

| Workload | Top Bash 命令 |
| -------- | ------------- |
| ADE | `dbt`（24.7-50.9%）—— 数据库转换 |
| DABStep | `python3`（Qwen 上 37.2-38.5%）或 Gemma 上 `dbt`/`python3` |
| GAIA | `curl`（Qwen 上 24.2-25.2%）—— web 面向 |
| SWE-Bench Pro / Terminal | 大 "Other" 份额（50-65%）—— 多样：`pytest`、`git`、`find`、`make`、`gcc`、`npm`、`pip` 等 |

#### Top 时间消耗 tool（Table 1）

贵 tool 跟频繁调用 tool 不同：

| Benchmark | Top 时间消耗 tool（Qwen-T）| 平均 duration |
| --------- | -------------------------: | ------------: |
| ADE | Bash（n=611）| 1.27s |
| **GAIA** | **Agent**（委托，n=11）| **916s** |
| **SWE-Bench Pro** | **Agent**（委托，n=75）| **59.15s** |

**`Agent` tool**（spawn 子 agent 做委托工作）是迄今最贵 —— GAIA 上多分钟。失败 agent 集中在产生错误的 tool。

#### Top 失败 tool（Table 2）

失败最多的 tool 是改 state 的：

| Workload | Top 失败 tool，失败率 |
| -------- | ------------------- |
| ADE | Gemma-I **Edit fail 95.4%**（n=2757）|
| ADE | Gemma-T Bash fail 28.4%（n=208）|
| GAIA | Gemma-T Read fail 48.9% |
| SWE-Bench Pro | Gemma-I Edit fail 77.8%（n=955）、Bash fail 39.8% |

**失败集中在改 state tool**（Edit、Bash），不在检索 tool（Read、WebFetch —— <10% 失败）。这是失败级联源：Edit 失败 → 错误 append 到上下文 → 下轮 LLM 试图修 → 又失败 → 上下文长 + turn 数长。

#### Tool 意图跨时间 shift（Figure 12）

一个鲁棒时间模式：agent 从探索开始，到执行结束。

```
Agent 进度 quartile 1：
  Read/Explore tool（Read、Glob、Grep、WebFetch、WebSearch）：60-80% of tool 调用
  Execute/Write tool（Bash、Edit、Write、TodoWrite）：20-40%

Agent 进度 quartile 4：
  Read/Explore tool：0-20%
  Execute/Write tool：80-100%
```

> [!quote] 时间 characterization 洞察
> "Agents typically begin in an inspection-heavy mode and then transition toward action-heavy behavior. This transition is strongest on DABStep, SWE-bench Pro, and Terminal Bench."

这对 serving 优化有直接影响：
- 早 turn 受益于快 read/list 操作（文件系统、搜索）
- 晚 turn 受益于快 write/exec 操作（进程 spawn、文件写）
- Caching 策略应 account for 时间 shift（早 turn read 可能重读；晚 turn write 是 unique 的）

## 头号实验证据（综合）

论文产生的 11 个综合 insight：

1. **LLM 时间主导** at 71-98% of E2E；tool 时间 2-29%。
2. **Decode 主导** LLM 时间 at 91-98.6%；prefill 1.4-9.0%。
3. **Context cache 命中** 实证 84.6-99.5%。
4. **Append-to-output 比** 中位 <1.5×，平均 1.5-7.3×。
5. **推理紧凑化** Gemma 在 ADE 上轨迹 6×（18 vs 109 平均 turn）。
6. **失败跑放大负载** 在 turn 数和上下文长度上。
7. **Tool 输出组成结构化 token 重** —— Instant 输出 70-98% 是 tool-call token。
8. **Tool 调用类型 workload 特定** —— DABStep dbt/python3、GAIA curl/WebFetch、SWE/Terminal Edit+Bash+Read。
9. **Tool 意图 shift** 从早 Read/Explore 到晚 Execute/Write。
10. **`Agent`（委托）是最贵 tool** —— GAIA 上多分钟。
11. **改 state tool 失败更多** —— Edit/Bash 28-95% 失败率 vs Read/WebFetch <10%。

## 优势与局限

**优势。**

- **第一个生产级端到端 agentic workload characterization** with 真 ReAct 框架（Claude Code）的 proper OpenTelemetry instrumentation，不是合成 agent。
- **多维测量** —— turn、上下文、输出组成、prefill/decode、cache 复用、tool 类型、命令类型、成功/失败、时间模式。什么都不缺。
- **推理模型测过** —— 不只 Instant。多数先前 characterization 在推理模型之前。
- **标准 benchmark**（SWE-Bench Pro、GAIA、Terminal-Bench 2.0）让结果跟其它论文直接可比。
- **方法论可复制** —— vLLM 0.20.0 + Claude Code + OpenTelemetry 是可复现 stack。

**局限。**

> [!warning] 只单 agent 执行
> 论文明确指出（§5.1 末段）："While the degree of concurrency (i.e., number of agents running concurrently) may change the LLM- vs. -tool time, our results are in line with other works." 生产多 agent serving（多个 agent 共享 GPU）有 **不同** LLM%/tool% 比，因为 LLM 调用排队。论文的数字是 **per-agent 下界** on tool% —— 并发 serving 放大它。

- **只 2 个模型家族（Qwen3.6、Gemma4）** 在一个规模范围（27-31B）。7B 或 70B+ 行为可能不同。
- **只 Claude Code agent 框架**。其它 harness（Codex、Cursor、Aider、OpenHands）有不同 turn 结构、prompt 模板、tool 集。
- **只 vLLM** —— SGLang 的 RadixAttention 可能产生不同 cache 命中率和 prefill/decode 切分。
- **不提 serving 系统优化** —— 只测量。读者必须自己把 insight 翻译成设计选择。
- **每 cell 100 task 小** —— 长尾行为（如 786-turn ADE 异常值）被采样但可能不代表完整分布。
- **TP=2 固定** —— 不测 TP=4 或 TP=8 怎么改变 decode 主导（很可能仍 decode-bound 但比例 shift）。
- **截至 2026-06 无代码发布**。

> [!bug] 28.7% GAIA 数字是给 **Gemma Thinking**，不是 Qwen
> 几篇论文（包括我先前一些 wiki entry）引用 "GAIA tool 时间 28.7%" 不注明 model。论文具体报告 **Gemma Thinking 28.7%、Qwen Thinking 24.9%**。Instant 变体数字又不同。引用时命名模型。

## 这意味着什么

这篇论文做了每个 serving-systems 研究者应该想要 workload 论文做的事：**提供数字，不是 vibes**。上面 11 个 insight 具体够在设计文档里引用且严谨够定 field 争论（特别是 "tool 主导" vs "LLM 主导" 问题 —— 见 per-workload 分解）。

Serving 系统设计三个含义：

1. **优化 decode，不是 prefill。** 91-98.6% 发现决定性。投机解码、decode 侧量化（如 [[saw-int4|SAW-INT4]]）、KV cache 压缩 —— 这些对 agent workload 比 chunked prefill 优化更多 5-10× 杠杆。
2. **让 KV cache 保留承重。** 84.6-99.5% cache 命中率意味失去 state 是灾难性的。[[continuum|Continuum]] 的 TTL-based pinning 是对的原语；[[multi-turn-optimization|RadixAttention]] in SGLang 和 prefix caching in vLLM 都有价值但 address cache 命中问题，不是 cache 保留问题。
3. **Tool 延迟优化收益有界** —— 最多在 coding workload 省 17-30% E2E 时间，web workload（GAIA）多。投机 tool 执行（[[#相关阅读|PASTE、Speculative Tool Calls、Conveyor]]）值得做但关不闭超过此 gap。

2027 年的三个预测：

1. **这论文成为标准引用** 就像 ShareGPT 对 chatbot serving 那样。任何 agentic serving 论文不对比这些数字会无人引用。
2. **多 agent 并发测量成为下一篇论文。** 单 agent caveat 是明显跟进方向。预期 6-12 个月内姊妹论文（或扩展）。
3. **Per-harness characterization 扩展。** 同 benchmark 上比较 Claude Code vs Codex vs Cursor vs Aider 会暴露这些数字多少是 harness 特定 vs LLM 特定。

这篇论文 **不**覆盖：

- **多租户带并发 serving**。只 per-agent 测量；生产多 agent 调度未 address。
- **GPU 池规模或 cost-benefit 分析**。数字 inform 设计；不挑配置。
- **长跑 agent（24+ 小时）**。每 cell 采样到 100 task；持续 agent 未测。
- **Tool 执行侧优化**。Tool 延迟测了但没优化；那是 [[cpu-centric-agentic-ai|CPU-Centric Perspective]] 的 remit。

## 源代码与复现

```bash
# 截至 2026-06 代码未发布。
# 复现 stack：
pip install vllm==0.20.0
# 加 Claude Code（专有）、Harbor 评估框架
# OpenTelemetry vLLM 支持 + Jaeger collector
```

**复现协议**（Section 3.1）：

| 组件 | 配置 |
| ---- | --- |
| 推理引擎 | vLLM v0.20.0、TP=2、OpenTelemetry 支持开启 |
| GPU | 2× NVIDIA H100 NVL with 12 NVLink |
| CPU | Intel Xeon Platinum 8592+ |
| Agent 框架 | Claude Code |
| 评估环境 | Harbor |
| Trace collector | Jaeger |
| 模型 | Qwen3.6-27B Thinking + Instant；Gemma4-31B Thinking + Instant |
| Benchmark | ADE-Bench、DABStep（100 采样）、GAIA、SWE-Bench Pro（100 采样）、Terminal-Bench 2.0 |
| Task/cell | 100（每 benchmark 采样）|
| 总 cell | 8 / workload × 5 workload = 40 |
| 总 trace | ~4000 |

**关键基础设施组件**（§3.1）：

- **Forwarding proxy** —— 截每个 LLM 请求，改 header 确保 vLLM 发 OpenTelemetry trace，附 per-agent API key 给 trace correlation
- **Per-agent wrapper** —— 在 Docker 容器里启动 Claude Code agent，分配独特 API key
- **Trace 聚合器** —— 把 Jaeger OTEL trace + Harbor 执行结果 + agent 请求 log 组合成 characterization-ready 数据

## 相关阅读

- [[continuum]] —— Continuum 直接引用这论文数字（具体 84.6-99.5% cache 命中率），Continuum 的 TTL-based KV pinning 是这论文 "失去 state 是灾难性的" insight 的系统响应。
- [[cpu-centric-agentic-ai]] —— 互补测量从 CPU 侧；这论文标准 coding workload 上 17-30% tool 数字是 CPU-Centric 上 88% 在故意 CPU-heavy tool（200GB ENNS、RDKit）的下界对应。
- [[agent-serving-challenges]] —— Agent serving 差异更高层调研；这论文提供严谨数据。
- [[multi-turn-optimization]] —— 多轮 KV 复用策略；这论文 84.6-99.5% cache 命中率是那些策略的实证 justification。
- [[prefill-decode-disaggregation]] —— PD disagg 动机重审：这论文显示 agentic workload decode 主导 91-98.6%，所以 PD-disagg 的"平衡 prefill 和 decode" framing 应用不同 —— agent 几乎全 decode。
- [[saw-int4]] —— INT4 KV 量化；特别相关给这论文显示 decode（因此 KV cache）主导 LLM 时间。
- [[paged-attention]] —— 底层 KV cache 原语；这论文验证 paged attention 跨长上下文摊销正是 agentic workload 需要的。
- [[vllm]] —— 用的推理引擎（vLLM v0.20.0）。
- [[continuous-batching]] —— 标准 agent serving 用连续批处理；这论文显示 decode 主导，正是连续批处理优化的。
- [[rose]] —— ROSE 测量 rollout > 70% training wall-clock 是这论文 per-task 测量的 agentic-RL 对应；都 motivate cache-state 保留。
- [[ai-agent-overview]] —— 更高层 ReAct 范式描述。

## 参考文献

- Yichao Yuan, Ankita Nayak, Souvik Kundu, Nishil Talati. *Agentic AI Workload Characteristics.* arXiv:2605.26297, 2026 年 5 月。 https://arxiv.org/abs/2605.26297
- ReAct：Yao et al.（论文引用 [37]）。
- Claude Code：Anthropic（[2]）。
- Codex：OpenAI（[4]）。
- Harbor 评估框架：（[10]）。
- 开放 benchmark：ADE-Bench [31]、DABStep [7]、GAIA [21]、SWE-Bench Pro [6]、Terminal-Bench [20]。
- vLLM：（[15]）。
- OpenTelemetry：（[24]）。
- Jaeger：（[11]）。
- 并发 ReAct workload 分析：KAIROS（[38]）—— 引用给多 agent 并发视角。
