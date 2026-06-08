---
title: Index
updated: 2026-06-02
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
- [[aurora]] — Aurora: online speculative-decoding training as async RL on live SGLang traffic, day-0 deployment, Tree Attention kernel (Together AI, ICML 2026, arXiv 2602.06932) — paper review
- [[ring-attention]] — Ring Attention: persistent Q + rotating KV with FlashAttention streaming softmax; 4M-token training context (Liu/Zaharia/Abbeel, ICLR 2024) — paper review
- [[deepspeed-ulysses]] — DeepSpeed Ulysses: AllToAll-based sequence parallelism with O(N/P) communication; the Ring Attention alternative (Microsoft, arXiv 2309.14509) — paper review
- [[prfaas]] — PrfaaS: cross-datacenter PD disaggregation — hybrid-attention KVCache flows over commodity Ethernet from compute-dense prefill clusters to bandwidth-optimized decode clusters (Moonshot/Tsinghua, arXiv 2604.15039) — paper review
- [[mamba]] — Mamba: linear-time sequence modeling with Selective SSM + hardware-aware parallel scan; first SSM to match Transformer++ on language at 3B scale; the foundational architecture for every 2026 hybrid-attention production LLM (Gu/Dao, arXiv 2312.00752, NeurIPS 2024) — paper review

## RL Infrastructure
- [[rlhf-overview]] — RLHF: Reinforcement Learning from Human Feedback
- [[ppo-for-llm]] — PPO for LLM alignment
- [[grpo]] — GRPO: Group Relative Policy Optimization
- [[dpo]] — DPO: Direct Preference Optimization
- [[reward-modeling]] — Reward modeling techniques
- [[rl-training-frameworks]] — RL training frameworks (OpenRLHF, TRL, veRL)
- [[on-policy-distillation]] — On-Policy Distillation (OPD): dense teacher signal as an RL replacement — GKD/MiniLLM lineage, Thinking Machines Lab reframing, variants, production deployments, the OPD-vs-RL debate
- [[deepseek-v4-opd]] — DeepSeek-V4 (Apr 2026): multi-teacher full-vocabulary OPD entirely replacing the V3.2 mixed-RL post-training stage — paper analysis
- [[mopd]] — MOPD (Multi-Domain On-Policy Distillation): Nemotron-Cascade 2's single-stage stabilizer between Multi-domain RL and RLHF; 3 cascade-internal teachers routed per-prompt; IMO/IOI/ICPC 2025 gold at 3B active params — paper analysis
- [[self-policy-distillation]] — SPD: teacher-free self-distillation that steers self-generation via KV-activation projection onto a capability subspace extracted from gradients of correctness-aligned loss (Cambridge/HKUST/UChicago, arXiv 2605.22675) — paper review

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
- [[prorl-agent]] — ProRL Agent: Rollout-as-a-Service (NVIDIA, arXiv 2603.18815) — paper review **[superseded by [[polar]] May 2026]**
- [[polar]] — Polar: Agentic RL on Any Harness at Scale — the ProRL Agent successor; LLM-API proxy lets any unmodified harness (Codex / Claude Code / Qwen Code / Pi) be trained; registered as a NeMo Gym environment (NVIDIA, arXiv 2605.24220) — paper review
- [[agent-lightning]] — Agent Lightning: train ANY AI agent via Training-Agent Disaggregation + LightningRL + OpenTelemetry-native trace capture; the original (Aug 2025) decoupling-paradigm paper that ProRL Agent and Polar extend (Microsoft, arXiv 2508.03680) — paper review
- [[rose]] — ROSE: cooperative elasticity that harvests idle production serving GPUs (avg 18.9% util) for RL rollouts via VMM-based KVC sharing + dual-SLO admission + sparsity-aware (>95%) weight transfer; 1.3–3.3× e2e throughput, allocation overhead 50-80× better than spot/serverless (HKUST/Alibaba, arXiv 2605.06534) — paper review
- [[search-r1]] — Search-R1: Training LLMs to Reason and Leverage Search Engines with RL — the canonical entry-point paper for agentic RL; R1-Zero extended to tool use with retrieved-token loss masking (UIUC + UMass + Google, COLM 2025, arXiv 2503.09516) — paper review
- [[search-r1-codebase-walkthrough]] — Search-R1 file-by-file code tutorial covering the 600 lines of paper-specific code and the ~5000 lines of veRL PPO/GRPO machinery; the cleanest agentic-RL reference implementation walked end-to-end
- [[agentic-rl-foundations]] — Agentic RL onboarding hub; 4-phase reading path, canonical references, FAQ, open directions
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
- [[continuum]] — Continuum: TTL-based KV cache pinning + program-level FCFS for multi-turn agent serving; first to model per-turn queueing delay (not just reload cost); 1.12–3.66× delay reduction, up to 8.18× on real SWE-agent (UC Berkeley/Stanford/Tensormesh, arXiv 2511.02230) — paper review
- [[cpu-centric-agentic-ai]] — CPU-Centric Perspective on Agentic AI: characterization (tool time up to 88% on tool-dominated workloads) + COMB (CPU-aware overlapped micro-batching, 1.7× P50 / 3.9× service latency) + MAS (mixed agentic scheduling, 2.37× P50 for minority requests) (Georgia Tech/Intel, arXiv 2511.00739) — paper review
- [[agentic-ai-workload-characteristics]] — Agentic AI Workload Characteristics: the canonical end-to-end agent workload measurement paper; Claude Code + Qwen3.6-27B/Gemma4-31B on SWE-Bench Pro/GAIA/Terminal-Bench; LLM=71-98%, tool=2-29%, decode=91-98.6%, cache hit=84.6-99.5%; reasoning compactifies trajectories 6× (UIUC/Intel, arXiv 2605.26297) — paper review
- [[speculative-actions]] — Speculative Actions: lossless framework lifting speculate-verify from tokens to API-call level; Actor/Speculator decomposition + k-way breadth + safety primitives; 19.5% time saved on chess, 46% top-3 prediction on HotpotQA, P95 latency 37.93ms vs 102.97ms on OS tuning (Columbia, arXiv 2510.04371) — paper review
- [[speceyes]] — SpecEyes: agentic-level speculative acceleration for multimodal LLMs via 4-phase pipeline + answer-separability cognitive gating + heterogeneous parallel funnel; 1.42-1.73× avg speedup with up to +6.7% accuracy on V*/HR-Bench/POPE (Xiamen U/Rochester/OSU, arXiv 2603.23483) — paper review
- [[speccache]] — SpecCache + What Limits Agentic Systems Efficiency?: 5-day/5-provider/9-model LLM API latency study (69.21× variance) + 53.7% web-env latency finding + SpecCache (action-observation cache with draft-LLM model-based prefetching); 58× cache hit rate improvement vs random, 3.2× web env latency reduction (UW-Madison/Toronto/NVIDIA, arXiv 2510.16276) — paper review
- [[mori]] — MORI: Memory Offloader with Relative Idleness — direct successor to Continuum (same first author Hanchen Li); two-tier (GPU + CPU) KV offloading with continuous idleness metric + sticky rebalancing + typed eviction; 20-71% higher throughput, 18-43% lower TTFT, up to 2.8× TTFT reduction on real Claude Code SWE-bench Pro traces; first program-aware two-tier scheduler that handles dynamic agent workloads (UC Berkeley/Renmin/Stanford/Georgia Tech, arXiv 2606.00866) — paper review
