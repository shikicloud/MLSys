---
title: Change Log
updated: 2026-05-06
---

# Change Log

## 2026-05-06
- [INGEST] arXiv:2603.18815 "ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents" (NVIDIA, March 2026) — paper review at [[prorl-agent]] in `wiki/agentic-rl/`. Citation under `sources/papers/prorl-agent/`. Report-style page with concrete source code from `openhands/nvidia/registry.py` (AgentHandler ABC + JobDetails dataclass), `openhands/nvidia/async_server.py` (3-queue pipeline, min-heap LB, unified _worker), `scripts/start_server.py` (FastAPI + multiprocessing), `openhands/llm/nvidia/` (token-in/out). Cross-linked from [[agentic-rl-overview]], [[environment-design]], [[rl-training-frameworks]], [[grpo]], [[kv-cache-optimization]], [[multi-turn-optimization]].
- [INGEST] arXiv:2604.19157 "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving" (Together AI et al., May 2026) — paper review at [[saw-int4]] in `wiki/llm-inference/`. Citation under `sources/papers/saw-int4/`. Report-style page with concrete source from the SGLang fork (`memory_pool.py` env-var gate and `set_kv_buffer` BDR branch, fused Triton kernel `quantized_set_kv_int4_hadamard_fused_triton`, `tools/fit_kv_centroids.py` k-means calibration). Block-diagonal Hadamard rotation recovers Qwen3-4B-Thinking GPQA from 0% (plain INT4) → 65.82%, with throughput indistinguishable from plain INT4. Cross-linked from [[kv-cache-optimization]], [[quantization]], [[sglang]], [[long-context-serving]], [[paged-attention]], [[vllm]].
- [NEW] [[rotation-based-quantization]] — synthesis page covering the QuIP / QuIP# / QuaRot / SpinQuant / SAW-INT4-BDR family, with mathematical foundation, comparison table, where-the-rotation-absorbs analysis, practical guidance, and open questions. Cross-linked from [[saw-int4]], [[quantization]], [[kv-cache-optimization]].
- [EXPANDED] [[quantization]] — added "Rotation-based Quantization" section synthesizing QuIP/QuaRot/SpinQuant/BDR. Connected to existing QuIP# coverage. Updated KV cache table to distinguish plain INT4 vs INT4+BDR. Added QuaRot/SpinQuant/SAW-INT4 to references and related pages.
- [EXPANDED] [[kv-cache-optimization]] — added "Rotation-based KV cache quantization" subsection citing [[saw-int4]]. Updated quantization table with INT4 (plain) and INT4 + BDR rows showing the GPQA collapse and recovery. Improved KIVI description (asymmetric mixed-granularity, per-channel K + per-token V + FP16 anchors). Updated decision tree and references.

## 2026-05-07
- [Q&A] [[saw-int4]] — Shiki asked for a walkthrough of the outlier-channel paragraph. Q&A logged inline as an Obsidian `[!question]+` callout right after the GPQA-collapse table in the Background section (not as a bottom-of-page section). Compact-paragraph format (4 paragraphs) explaining outlier channel sources (RoPE / massive activations / specialized heads), the per-token scale-zero failure mode (~95% of channels collapse to 0), why surface tasks survive but multi-step reasoning compounds the error, and how BDR fixes it. Convention recorded in memory: paper Q&A goes inline at the discussed location, callout-formatted, compact paragraphs.

## 2026-04-14
- [EXPANDED] [[distributed-training]] — Comprehensive deep-dive (~300 lines EN, ~1090 lines CN), covering training memory analysis, DDP gradient sync with Ring AllReduce, ZeRO 1/2/3 memory formulas and diagrams, mixed precision (FP16/BF16/FP8), 3D/4D/5D parallelism composition, activation checkpointing, communication optimization, fault tolerance, code examples, LLaMA 3.1 / DeepSeek-V3 case studies
- [EXPANDED] [[training-frameworks]] — Comprehensive deep-dive (~280 lines EN, ~1050 lines CN), covering Megatron-LM/Core architecture and 5D parallelism, DeepSpeed ZeRO full suite and Chat/MoE, FSDP/FSDP2 and TorchTitan, detailed comparison table with benchmarks, other frameworks (Colossal-AI/Composer/Nanotron/Fairscale), selection guide, side-by-side code examples
- [EXPANDED] [[rlhf-overview]] — Comprehensive deep-dive (~230 lines EN, ~600 lines CN), covering 3-stage pipeline (SFT/RM/PPO), Bradley-Terry derivation, RM loss, RL objective with GAE, RLHF variants (Online/Offline/RLAIF/RLVR/Iterative/Best-of-N), TRL code examples, challenges
- [EXPANDED] [[ppo-for-llm]] — Comprehensive deep-dive (~230 lines EN, ~760 lines CN), covering TRPO-to-PPO evolution, clipped objective with geometric intuition, 4-model architecture, GAE derivation, token-level updates, KL integration, implementation tips and pitfalls, full PPO pseudocode, comparison table
- [EXPANDED] [[grpo]] — Comprehensive deep-dive (~260 lines EN, ~750 lines CN), covering critic elimination motivation, group-relative advantage derivation, GRPO pipeline diagram, PPO comparison, DeepSeek-R1-Zero emergent behaviors, R1 training pipeline, GRPO pseudocode and TRL usage, DAPO/Dr.GRPO/RLOO variants, benchmarks
- [EXPANDED] [[continuous-batching]] — Comprehensive deep-dive (~200 lines EN, ~1000 lines CN), covering static batching analysis, Orca, chunked prefill, scheduling strategies, vLLM/SGLang comparison, memory management, performance analysis
- [EXPANDED] [[prefill-decode-disaggregation]] — Comprehensive deep-dive (~220 lines EN, ~1100 lines CN), covering compute profile comparison, Splitwise/DistServe/Mooncake architectures, KV transfer analysis, DeepSeek-V3 deployment, cost analysis
- [EXPANDED] [[vllm]] — Comprehensive deep-dive article (~300 lines), covering V1 architecture, EngineCore, unified scheduler, prefix caching, DP Attention + EP, speculative decoding, quantization, benchmarks, deployment, engine comparison
- [NEW] [[parallelism-strategies-deep-dive]] — Deep dive into DP/TP/EP/EDP/ETP parallelism strategies with code, diagrams, DeepSeek-V3 case study
- [EXPANDED] [[model-parallelism]] — Complete rewrite as deep-dive (~800 lines CN, ~230 lines EN), focused on PP and CP: GPipe/1F1B/Interleaved/Zero Bubble/DualPipe schedules, Ring Attention/Ulysses context parallelism, inference applications, hybrid parallelism, performance analysis

## 2026-04-13
- [INIT] Wiki vault created with initial structure
- [INIT] Schema (CLAUDE.md) defined
- [INIT] Index created with planned topic coverage
- [INGEST] Initial research and population of all topic areas
