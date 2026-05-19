---
title: Index
updated: 2026-05-13
---

# Wiki Index

## LLM Inference & Serving
- [[vllm]] — vLLM: High-throughput LLM serving with PagedAttention
- [[sglang]] — SGLang: Fast structured generation and serving
- [[tensorrt-llm]] — TensorRT-LLM: NVIDIA's optimized inference engine
- [[paged-attention]] — PagedAttention: Virtual memory for KV cache
- [[speculative-decoding]] — Speculative decoding: Draft-verify acceleration
- [[continuous-batching]] — Continuous batching: Dynamic request scheduling
- [[kv-cache-optimization]] — KV cache optimization techniques
- [[quantization]] — Quantization for LLM inference (GPTQ, AWQ, SqueezeLLM)
- [[model-parallelism]] — Model parallelism: pipeline parallelism & context parallelism
- [[parallelism-strategies-deep-dive]] — Parallelism strategies deep dive: DP / TP / EP / EDP / ETP
- [[prefill-decode-disaggregation]] — Prefill-decode disaggregation architectures
- [[af-disaggregation]] — Attention-FFN disaggregation: splitting operators (not phases) onto specialized hardware tiers (MegaScale-Infer, DP-attention + EP-MoE)
- [[saw-int4]] — SAW-INT4: System-Aware 4-bit KV-cache quantization with block-diagonal Hadamard rotation (Together AI, arXiv 2604.19157) — paper review
- [[rotation-based-quantization]] — Rotation-based quantization family overview (QuIP / QuaRot / SpinQuant / BDR)
- [[das-spec-rl]] — DAS: Distribution-Aware Speculative Decoding for RL Training (suffix-tree drafter + length-aware budget; arXiv 2511.13841) — paper review

## RL Infrastructure
- [[rlhf-overview]] — RLHF: Reinforcement Learning from Human Feedback
- [[ppo-for-llm]] — PPO for LLM alignment
- [[grpo]] — GRPO: Group Relative Policy Optimization
- [[dpo]] — DPO: Direct Preference Optimization
- [[reward-modeling]] — Reward modeling techniques
- [[rl-training-frameworks]] — RL training frameworks (OpenRLHF, TRL, veRL)
- [[on-policy-distillation]] — On-Policy Distillation (OPD): dense teacher signal as an RL replacement — GKD/MiniLLM lineage, Thinking Machines Lab reframing, variants, production deployments, the OPD-vs-RL debate
- [[deepseek-v4-opd]] — DeepSeek-V4 (Apr 2026): multi-teacher full-vocabulary OPD entirely replacing the V3.2 mixed-RL post-training stage — paper analysis

## ML Infrastructure
- [[distributed-training]] — Distributed training: data/model/pipeline parallelism
- [[gpu-cluster-management]] — GPU cluster management and scheduling
- [[training-frameworks]] — Training frameworks (Megatron-LM, DeepSpeed, FSDP)
- [[data-pipelines]] — Data pipelines for LLM training
- [[checkpointing]] — Checkpointing and fault tolerance

## ML Systems
- [[mlops-overview]] — MLOps: model lifecycle management
- [[feature-stores]] — Feature stores and data management
- [[model-registry]] — Model registries and versioning
- [[ray-ecosystem]] — Ray ecosystem for distributed ML
- [[experiment-tracking]] — Experiment tracking and reproducibility

## Agentic RL
- [[agentic-rl-overview]] — Agentic RL: RL for agent behaviors
- [[tool-use-rl]] — RL for tool use and API calling
- [[multi-step-reasoning-rl]] — RL for multi-step reasoning
- [[environment-design]] — Environment design for agentic RL
- [[prorl-agent]] — ProRL Agent: Rollout-as-a-Service (NVIDIA, arXiv 2603.18815) — paper review
- [[nemo-gym]] — NeMo Gym: NVIDIA's RL environment framework — 84 benchmarks, 19 agent harnesses, Apptainer-based isolation

## AI Agents
- [[ai-agent-overview]] — AI agent architectures and patterns
- [[tool-use]] — Tool use and function calling
- [[multi-agent-systems]] — Multi-agent systems
- [[agent-frameworks]] — Agent frameworks (LangChain, CrewAI, AutoGen)
- [[mcp-protocol]] — Model Context Protocol (MCP)
- [[agent-memory]] — Agent memory and state management

## LLM Serving for AI Agents
- [[agent-serving-challenges]] — Unique serving challenges for AI agents
- [[long-context-serving]] — Long context window serving optimization
- [[structured-output-serving]] — Structured output and constrained decoding
- [[function-calling-optimization]] — Function calling optimization
- [[compound-ai-systems]] — Compound AI systems architecture
- [[multi-turn-optimization]] — Multi-turn conversation serving optimization
