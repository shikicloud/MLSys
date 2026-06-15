---
title: 目录索引
updated: 2026-06-15
---

# Wiki 目录索引

## LLM 推理与服务
- [[vllm]] — vLLM：基于 PagedAttention 的高吞吐量 LLM 服务
- [[sglang]] — SGLang：快速结构化生成与服务
- [[tensorrt-llm]] — TensorRT-LLM：NVIDIA 优化推理引擎
- [[paged-attention]] — PagedAttention：KV 缓存的虚拟内存管理
- [[speculative-decoding]] — 投机解码：草稿-验证加速方法
- [[continuous-batching]] — 连续批处理：动态请求调度
- [[kv-cache-optimization]] — KV 缓存优化技术
- [[quantization]] — LLM 推理量化（GPTQ、AWQ、SqueezeLLM）
- [[model-parallelism]] — 模型并行：流水线并行与上下文并行
- [[parallelism-strategies-deep-dive]] — LLM 并行策略深度解析：DP / TP / EP / EDP / ETP
- [[prefill-decode-disaggregation]] — 预填充-解码分离架构
- [[af-disaggregation]] — 注意力-FFN 分离：把算子（不是阶段）切到专配硬件（MegaScale-Infer、DP-attention + EP-MoE）
- [[saw-int4]] — SAW-INT4：基于块对角 Hadamard 旋转的系统感知 4 位 KV 缓存量化（Together AI, arXiv 2604.19157）— 论文精读
- [[rotation-based-quantization]] — 基于旋转的量化家族综览（QuIP / QuaRot / SpinQuant / BDR）
- [[das-spec-rl]] — DAS：面向 RL 训练的分布感知投机解码（suffix tree drafter + 长度感知预算；arXiv 2511.13841）— 论文精读
- [[aurora]] — Aurora：在 SGLang 实时流量上把投机解码 draft 训练做成异步 RL，Day-0 上线，Tree Attention kernel（Together AI，ICML 2026，arXiv 2602.06932）— 论文精读
- [[ring-attention]] — Ring Attention：Q 不动 + KV 旋转 + FlashAttention streaming softmax；4M token 训练上下文（Liu/Zaharia/Abbeel, ICLR 2024）— 论文精读
- [[deepspeed-ulysses]] — DeepSpeed Ulysses：基于 AllToAll 的序列并行，通信 $O(N/P)$；Ring Attention 的替代方案（Microsoft, arXiv 2309.14509）— 论文精读
- [[prfaas]] — PrfaaS：跨数据中心 PD 分离 —— hybrid attention 的 KVCache 通过普通以太网从算力密集 prefill 集群流到带宽优化 decode 集群（Moonshot/清华, arXiv 2604.15039）— 论文精读
- [[mamba]] — Mamba：基于 Selective SSM + 硬件感知并行扫描的线性时间序列建模；第一个在 3B 规模语言建模上匹敌 Transformer++ 的 SSM；2026 年所有 hybrid attention 生产 LLM 的基础架构（Gu/Dao, arXiv 2312.00752, NeurIPS 2024）— 论文精读

## 强化学习基础设施
- [[rlhf-overview]] — RLHF：基于人类反馈的强化学习
- [[ppo-for-llm]] — PPO 用于 LLM 对齐
- [[grpo]] — GRPO：组相对策略优化
- [[dpo]] — DPO：直接偏好优化
- [[reward-modeling]] — 奖励建模技术
- [[rl-training-frameworks]] — RL 训练框架（OpenRLHF、TRL、veRL）
- [[on-policy-distillation]] — On-Policy Distillation (OPD)：用稠密教师信号替代 RL —— GKD/MiniLLM 谱系、Thinking Machines Lab 重新包装、变体、生产部署、OPD-vs-RL 争论
- [[deepseek-v4-opd]] — DeepSeek-V4（2026-04）：多教师全词表 OPD 完全替代 V3.2 的 mixed-RL post-training 阶段 —— 论文分析
- [[mopd]] — MOPD（多 Domain On-Policy 蒸馏）：Nemotron-Cascade 2 在 Multi-domain RL 和 RLHF 之间的单阶段稳定器；3 个 cascade 内部 teacher 按 prompt 路由；3B 激活参数拿 IMO/IOI/ICPC 2025 金牌 —— 论文分析
- [[self-policy-distillation]] — SPD：无老师自蒸馏，用从 correctness-aligned loss 梯度提取的能力子空间对 KV 激活做投影来引导自生成（剑桥/港科/芝大，arXiv 2605.22675）—— 论文精读

## 机器学习基础设施
- [[distributed-training]] — 分布式训练：数据/模型/流水线并行
- [[gpu-cluster-management]] — GPU 集群管理与调度
- [[training-frameworks]] — 训练框架（Megatron-LM、DeepSpeed、FSDP）
- [[data-pipelines]] — LLM 训练数据流水线
- [[checkpointing]] — 检查点与容错

## 机器学习系统
- [[mlops-overview]] — MLOps：模型生命周期管理
- [[feature-stores]] — 特征存储与数据管理
- [[model-registry]] — 模型注册与版本管理
- [[ray-ecosystem]] — Ray 分布式 ML 生态系统
- [[experiment-tracking]] — 实验追踪与可复现性

## 智能体强化学习
- [[agentic-rl-overview]] — 智能体 RL：面向智能体行为的强化学习
- [[tool-use-rl]] — 工具使用与 API 调用的 RL
- [[multi-step-reasoning-rl]] — 多步推理的 RL
- [[environment-design]] — 智能体 RL 的环境设计
- [[prorl-agent]] — ProRL Agent：Rollout 即服务（NVIDIA, arXiv 2603.18815）— 论文精读 **[2026-05 被 [[polar]] 取代]**
- [[polar]] — Polar：任意 harness 上的可扩展 agentic RL —— ProRL Agent 续作；LLM-API proxy 让任何未修改 harness（Codex / Claude Code / Qwen Code / Pi）都能训练；注册为 NeMo Gym 环境（NVIDIA, arXiv 2605.24220）— 论文精读
- [[agent-lightning]] — Agent Lightning：通过 Training-Agent Disaggregation + LightningRL + OpenTelemetry 原生 trace 捕获训练任何 AI agent；最初的（2025-08）解耦范式论文，ProRL Agent 和 Polar 都是它的延伸（Microsoft, arXiv 2508.03680）— 论文精读
- [[rose]] — ROSE：通过 cooperative elasticity harvest 空闲生产 serving GPU（平均 18.9% 利用率）做 RL rollout，通过 VMM 基础 KVC 共享 + dual-SLO 接纳 + 稀疏感知（>95%）weight transfer；端到端吞吐 1.3-3.3×，分配 overhead 比 spot/serverless 好 50-80×（HKUST/阿里, arXiv 2605.06534）—— 论文精读
- [[search-r1]] — Search-R1：用 RL 训练 LLM 推理 + 调搜索引擎 —— agentic RL 的标准入门论文；R1-Zero 扩展到 tool use，带 retrieved-token loss masking（UIUC + UMass + Google，COLM 2025，arXiv 2503.09516）— 论文精读
- [[search-r1-codebase-walkthrough]] — Search-R1 逐文件代码教程，覆盖 600 行论文特有代码 + ~5000 行 veRL PPO/GRPO 机器；端到端走完最干净的 agentic-RL 参考实现
- [[agentic-rl-foundations]] — Agentic RL 入门 hub；4 阶段阅读路径、标志参考、FAQ、开放方向
- [[nemo-gym]] — NeMo Gym：NVIDIA 的 RL 环境框架 —— 84 个 benchmark、19 个 agent harness、基于 Apptainer 的隔离

## AI 智能体
- [[ai-agent-overview]] — AI 智能体架构与模式
- [[tool-use]] — 工具使用与函数调用
- [[multi-agent-systems]] — 多智能体系统
- [[agent-frameworks]] — 智能体框架（LangChain、CrewAI、AutoGen）
- [[mcp-protocol]] — 模型上下文协议（MCP）
- [[agent-memory]] — 智能体记忆与状态管理

## 面向 AI 智能体的 LLM 服务
- [[agent-serving-challenges]] — AI 智能体的独特服务挑战
- [[long-context-serving]] — 长上下文窗口服务优化
- [[structured-output-serving]] — 结构化输出与受约束解码
- [[function-calling-optimization]] — 函数调用优化
- [[compound-ai-systems]] — 复合 AI 系统架构
- [[multi-turn-optimization]] — 多轮对话服务优化
- [[continuum]] — Continuum：基于 TTL 的 KV cache pinning + program 级 FCFS 给多轮 agent serving；第一个建模 per-turn queueing delay（不只是 reload cost）；延迟降低 1.12–3.66×，真实 SWE-agent 上多达 8.18×（UC Berkeley/Stanford/Tensormesh, arXiv 2511.02230）—— 论文精读
- [[cpu-centric-agentic-ai]] — Agentic AI 的 CPU 中心视角：characterization（tool 时间在 tool 主导 workload 上多达 88%）+ COMB（CPU 感知 overlapped micro-batching，P50 1.7× / 服务延迟 3.9×）+ MAS（混合 agentic 调度，少数请求 P50 2.37×）（Georgia Tech/Intel, arXiv 2511.00739）—— 论文精读
- [[agentic-ai-workload-characteristics]] — Agentic AI Workload Characteristics：标杆端到端 agent workload 测量论文；Claude Code + Qwen3.6-27B/Gemma4-31B on SWE-Bench Pro/GAIA/Terminal-Bench；LLM=71-98%、tool=2-29%、decode=91-98.6%、cache 命中=84.6-99.5%；推理紧凑化轨迹 6×（UIUC/Intel, arXiv 2605.26297）—— 论文精读
- [[speculative-actions]] — Speculative Actions：把 speculate-verify 从 token 提升到 API-call 级别的无损框架；Actor/Speculator 分解 + k-way breadth + 安全原语；chess 上节省 19.5% 时间、HotpotQA 上 46% top-3 prediction、OS tuning 上 P95 延迟 37.93ms vs 102.97ms（哥伦比亚, arXiv 2510.04371）—— 论文精读
- [[speceyes]] — SpecEyes：通过 4 阶段 pipeline + answer-separability cognitive gating + 异质并行 funnel 给多模态 LLM 的 agentic 级 speculative 加速；在 V*/HR-Bench/POPE 上 1.42-1.73× 平均加速 with 多达 +6.7% 准确率（厦大/罗切斯特/OSU, arXiv 2603.23483）—— 论文精读
- [[speccache]] — SpecCache + What Limits Agentic Systems Efficiency？：5 天/5 provider/9 模型 LLM API 延迟研究（69.21× 方差）+ 53.7% web-env 延迟发现 + SpecCache（action-observation cache with draft-LLM 模型基础 prefetching）；cache 命中率比随机改善 58×、web env 延迟降低 3.2×（UW-Madison/多伦多/NVIDIA, arXiv 2510.16276）—— 论文精读
- [[mori]] — MORI：Memory Offloader with Relative Idleness —— Continuum 的直接后继（同第一作者 Hanchen Li）；两层（GPU + CPU）KV offloading 带连续 idleness metric + sticky rebalancing + typed 驱逐；在真实 Claude Code SWE-bench Pro trace 上吞吐高 20-71%、TTFT 低 18-43%、TTFT 降低多达 2.8×；第一个处理动态 agent workload 的 program-aware 两层调度器（UC Berkeley/人大/斯坦福/Georgia Tech, arXiv 2606.00866）—— 论文精读
- [[infercept]] — InferCept：第一个面向被中断的增强型 LLM 设计的推理框架；确立 Discard/Preserve/Swap 分类法 + 四条闭式浪费方程 + min-waste 调度；swap 流水线（消除 96% Swap 浪费）+ 重算分块（砍半 Discard 浪费）+ 动态中断时长估计器（达到 oracle 的 93%）；吞吐 1.6×–2×、Llama3-70B 上 normalized latency 降低 1.3×–12×、GPU 内存浪费从 ~25% → 0.69%；Continuum 与 MORI 都以它为 canonical baseline（UCSD WukLab, ICML 2024, arXiv 2402.01869）—— 论文精读
- [[thunderagent]] — ThunderAgent：ICML 2026 Spotlight（top 2.2%）—— 第一个把 multi-turn workflow 抽象成一等公民 **agentic program** $P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$ 并按 program 粒度调度的 agent-serving 系统；反应式 thrashing 监视器（每 $\Delta t = 5$ s）配 shortest-context-first 驱逐（由 Lemma 4.1 形式化：$\text{Cost}_{\text{recompute}} \propto c_i^2$）；全局跨节点 waiting queue；hook-based 工具 GC + 异步环境准备；**对 vLLM 1.48-3.58×、在所有 6 个 benchmark 上击败 [[continuum|Continuum]]**（Continuum 在随机 ToolOrchestra 上*输*给 vLLM）；RL rollout 加速 1.79-3.92×；**[[mori|MORI]] 建立在其之上的系统层**（MORI 的 ~3,300 行实现在 ThunderAgent 上）（Georgia Tech/CMU/UIUC/Together AI, arXiv 2602.13692）—— 论文精读
