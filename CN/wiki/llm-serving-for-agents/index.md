---
title: 面向 AI 智能体的 LLM 服务
---

# 面向 AI 智能体的 LLM 服务

面向 AI agent 工作负载的推理侧优化 —— 多轮对话、工具调用、结构化输出、复合系统。

## 智能体服务挑战

- [[agent-serving-challenges]] —— 智能体服务跟聊天机器人服务的差异

  - [[agent-serving-challenges#智能体 vs. 聊天机器人工作负载|智能体 vs 聊天机器人]]
  - [[agent-serving-challenges#关键挑战详解|关键挑战]]
  - [[agent-serving-challenges#专用系统|专用系统（Parrot、AsyncFlow）]]
  - [[agent-serving-challenges#基准分析|基准分析]]

## 多轮优化

- [[multi-turn-optimization]] —— KV 复用、前缀缓存、会话管理

  - [[multi-turn-optimization#跨轮 KV 缓存复用|跨轮 KV 复用]]
  - [[multi-turn-optimization#前缀缓存实现|前缀缓存实现]]
  - [[multi-turn-optimization#分离式架构下的多轮挑战|分离式架构下的多轮挑战]]
  - [[multi-turn-optimization#上下文窗口管理策略|上下文窗口管理]]
  - [[multi-turn-optimization#提示缓存 (Prompt Caching)|Prompt 缓存]]
  - [[multi-turn-optimization#会话管理与路由|会话管理与路由]]

## 论文精读

- [[continuum]] —— Continuum：基于 TTL 的 KV cache pinning + program 级 FCFS 给多轮 agent serving；第一个建模 per-turn queueing delay（不只是 reload cost）；延迟降低 1.12–3.66×，真实 SWE-agent 分布式 setting 上多达 8.18×（UC Berkeley/Stanford/Tensormesh, arXiv 2511.02230, 2025-11）
- [[cpu-centric-agentic-ai]] —— Agentic AI 的 CPU 中心视角：characterization（tool 时间在非对称 CPU-GPU 硬件上 tool 主导 workload 多达 88%）+ **COMB**（CPU 感知 overlapped micro-batching，P50 1.7× / 服务延迟 3.9×）+ **MAS**（chat+agent 混合 workload 的 mixed agentic 调度，少数请求 P50 2.37×）（Georgia Tech/Intel, arXiv 2511.00739, 2025-11）
- [[agentic-ai-workload-characteristics]] —— Agentic AI Workload Characteristics：标杆端到端 agent workload 测量论文；Claude Code + Qwen3.6-27B/Gemma4-31B on 5 个真实 benchmark with OpenTelemetry 级 tracing；**LLM=71-98% / tool=2-29% 分解的权威来源**，decode 主导 91-98.6%，cache 命中 84.6-99.5%，推理的 6× 轨迹紧凑化效应（UIUC/Intel, arXiv 2605.26297, 2026-05）
- [[speculative-actions]] —— Speculative Actions：把 speculate-verify 从 token 级提升到 **跨整个 agentic 环境的 API-call 级** 的无损框架；通过 k-way breadth speculation 把慢 Actor（GPT-5 高推理）跟快 Speculator（更小模型）配对；闭式 cost-latency 定理；4 个环境（chess 节省 19.5% 时间、e-commerce ~1/3 轮立即响应、HotpotQA 46% top-3 prediction、OS tuning P95 37.93ms vs 102.97ms）（哥伦比亚, arXiv 2510.04371, 2025-10）
- [[speceyes]] —— SpecEyes：**多模态 LLM**（DeepEyes、Thyme）的 agentic 级 speculative 加速；4 阶段 pipeline 路由 tool-free 查询到小非 agentic MLLM（Qwen3-VL-2B）；新颖 **answer separability score** $S_{sep}$ 取代 softmax 做 cognitive gating（Δ 比 softmax 大 14×）；异质并行 funnel 给吞吐；在 V*/HR-Bench/POPE 上 1.42-1.73× 平均加速 with 多达 +6.7% 准确率（厦大/罗切斯特/OSU, arXiv 2603.23483, 2026-03）
- [[speccache]] —— SpecCache + What Limits Agentic Systems Efficiency？：双贡献 ——（1）5 天、5 provider、9 模型 LLM API 延迟方差研究，发现 Reflexion-based web agent 里 **69.21× 变化** 和 **53.7% web-env 延迟份额**；（2）**SpecCache** caching 框架带 action-observation LRU + draft-LLM 模型基础 prefetching 取得 **9.4-54× cache 命中率改善**（WebWalkerQA 8.9%→83.3%，Frames 1.0%→54.0%）和 **3.2× web env 延迟降低** 不降级轨迹。Web 环境上 [[speculative-actions]] 理论框架的实证验证（UW-Madison/多伦多/NVIDIA, arXiv 2510.16276, 2025-10）
- [[mori]] —— MORI：Memory Offloader with Relative Idleness —— 同第一作者 Hanchen Li 的 **[[continuum|Continuum]] 直接后继**。Continuum 是单层（仅 GPU）带 TTL pinning，MORI 是 **两层（GPU HBM + CPU DRAM）** 带连续 **idleness metric** $\iota = T_{acting}^{(k)}/(T_{reasoning}^{(k)}+T_{acting}^{(k)})$ 沿相对谱排所有 program、三层队列（GPU/CPU/Waiting）、sticky rebalancing 避免 PCIe churn、typed 驱逐（busy/idle/inactive label 传到引擎）。在真实 Claude Code SWE-bench Pro trace 上测：**吞吐高 20-71%、TTFT 低 18-43%、TTFT 降低多达 2.8×**；80 并发 program GPU 利用 99%+ vs phase-oblivious 调度器 59-76%；B200/Llama-70B 上 +71%（承重高内存压力结果）（UC Berkeley/人大/斯坦福/Georgia Tech, arXiv 2606.00866, 2026-05）
- [[infercept]] —— InferCept：**最早的 agent-aware KV pinning 论文**，[[continuum|Continuum]] 和 [[mori|MORI]] 都把它作为 baseline 引用。第一个面向"被中断的增强型 LLM"专门设计的推理框架；确立 canonical `Discard` / `Preserve` / `Swap` 分类法和四条闭式 *浪费方程*（WasteDiscard / WastePreserve / WasteSwap / WasteChunkD）。三项工程改进（swap 流水线 + 分块 ≈ 消除 96% Swap 浪费；GPU 饱和点 S 处的重算分块 ≈ 砍半 Discard 浪费；动态 $\hat{T}_{INT} = t_{now} - t_{call}$ 估计器 ≈ 达到 oracle 的 93%）汇入 min-waste 调度器。**吞吐 1.6×–2×**、normalized latency 降低 1.9×–5.7×（6B）、1.6×–10×（13B-TP2）、**1.3×–12×（70B-TP4）**、GPU 内存浪费 ~25% → **0.69%**。局限被 Continuum（per-turn 排队代价项 + TTL 上界）和 MORI（两层 idleness vs 二元 GPU/CPU swap）补上（UCSD WukLab, ICML 2024, arXiv 2402.01869）
- [[thunderagent]] —— ThunderAgent：**ICML 2026 Spotlight（top 2.2%）** —— [[continuum|Continuum]] 的同期竞争者，也是 **[[mori|MORI]] 跑在其上的系统层**。第一个把 multi-turn workflow 当作一等公民 **agentic program** $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ 并按 program 粒度（而非 per-request）调度的系统。**三组件**：(1) 周期性 thrashing 监视器（每 $\Delta t = 5$ s），对每个 backend 评估 $C_{\text{total}} < \sum_p c_p$；Pause/Restore 原语按 shortest-context-first 优先（由 Lemma 4.1 形式化：recompute 代价 $\propto c_i^2$，所以贪心 shortest-first 驱逐最优）；时间衰减函数 $f(t) = 2^{-t}$ 作用于 Acting program（memoryless 工具时长下证最优），Continuum TTL 的反应式（不是预测式）对偶。(2) 跨 DP 节点全局 program-aware waiting queue（paused KV 节点无关，所以任何有容量的 backend 都能 restore）；跨节点内存不平衡上界 $c_{\min} \cdot \Delta t$，对比 vLLM+SGLang baseline 的 51% 峰值不平衡。(3) Hook-based 工具资源 GC + 异步环境准备（Docker 镜像 / 装依赖跟高优先级 restoring program 的 LLM 阶段并行启动，隐藏 29.9-47.2 s 设置延迟）。**头号数字**：6 个 benchmark（SWEAgent、OpenHands、ToolOrchestra、ScienceAgent × GLM-4.6 / Qwen3-235B / Qwen3-8B）上**对 vLLM 1.48–3.58×**，**6 个全胜 Continuum**，包括 ToolOrchestra-HLE（Continuum 在随机工具时长下**输给 vLLM 0.65×**）；vs vLLM+SGLang Gateway SOTA **RL rollout 1.79-3.92×**；**磁盘节省多达 4.2×**；KV 命中率 ~100% vs Continuum 在高并发下跌到 ~60%。**OpenAI 兼容 API** 只多一个 `program_id` 参数；MIT 协议；vLLM + SGLang 后端（Georgia Tech/CMU/UIUC/Together AI, arXiv 2602.13692, 2026-02）

## 复合 AI 系统

- [[compound-ai-systems]] —— 智能体服务作为复合系统

  - [[compound-ai-systems#系统组件|系统组件（LLM、RAG、tools、RM、router）]]
  - [[compound-ai-systems#架构模式|架构模式]]
  - [[compound-ai-systems#DSPy：复合系统优化框架|DSPy 框架]]
  - [[compound-ai-systems#评估挑战|评估挑战]]
  - [[compound-ai-systems#性能优化|性能优化]]
