---
title: 目录索引
updated: 2026-05-13
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
