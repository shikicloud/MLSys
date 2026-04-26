---
title: Change Log
updated: 2026-04-13
---

# Change Log

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
