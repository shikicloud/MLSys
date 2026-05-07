---
title: 目录索引
updated: 2026-04-14
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
- [[saw-int4]] — SAW-INT4：基于块对角 Hadamard 旋转的系统感知 4 位 KV 缓存量化（Together AI, arXiv 2604.19157）— 论文精读
- [[rotation-based-quantization]] — 基于旋转的量化家族综览（QuIP / QuaRot / SpinQuant / BDR）

## 强化学习基础设施
- [[rlhf-overview]] — RLHF：基于人类反馈的强化学习
- [[ppo-for-llm]] — PPO 用于 LLM 对齐
- [[grpo]] — GRPO：组相对策略优化
- [[dpo]] — DPO：直接偏好优化
- [[reward-modeling]] — 奖励建模技术
- [[rl-training-frameworks]] — RL 训练框架（OpenRLHF、TRL、veRL）

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
- [[prorl-agent]] — ProRL Agent：Rollout 即服务（NVIDIA, arXiv 2603.18815）— 论文精读

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
