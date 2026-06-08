---
title: LLM Serving for AI Agents
---

# LLM Serving for AI Agents

Inference-side optimizations specific to AI agent workloads — multi-turn conversations, tool calls, structured outputs, compound systems.

## Challenges specific to agent serving

- [[agent-serving-challenges]] — what makes agent serving different from chatbot serving

  - [[agent-serving-challenges#Agent vs. Chatbot Workloads|Agent vs chatbot workloads]]
  - [[agent-serving-challenges#Key Challenges in Detail|Key challenges]]
  - [[agent-serving-challenges#Specialized Systems|Specialized systems (Parrot, AsyncFlow)]]
  - [[agent-serving-challenges#Benchmark Analysis|Benchmark analysis]]

## Multi-turn optimization

- [[multi-turn-optimization]] — KV reuse, prefix caching, session management

  - [[multi-turn-optimization#Cross-Turn KV Cache Reuse|Cross-turn KV reuse]]
  - [[multi-turn-optimization#Prefix-Cache Implementations|Prefix-cache implementations]]
  - [[multi-turn-optimization#Multi-Turn Challenges in Disaggregated Architectures|Multi-turn in disaggregated architectures]]
  - [[multi-turn-optimization#Context-Window Management Strategies|Context-window management]]
  - [[multi-turn-optimization#Prompt Caching|Prompt caching]]
  - [[multi-turn-optimization#Session Management and Routing|Session management & routing]]

## Paper reviews

- [[continuum]] — Continuum: TTL-based KV cache pinning + program-level FCFS for multi-turn agent serving; first to model per-turn queueing delay (not just reload cost); 1.12–3.66× delay reduction, up to 8.18× on real SWE-agent in distributed setting (UC Berkeley/Stanford/Tensormesh, arXiv 2511.02230, Nov 2025)
- [[cpu-centric-agentic-ai]] — CPU-Centric Perspective on Agentic AI: characterization (tool time up to 88% on tool-dominated workloads on asymmetric CPU-GPU hardware) + **COMB** (CPU-aware overlapped micro-batching, 1.7× P50 / 3.9× service latency) + **MAS** (mixed agentic scheduling for chat+agent mix workloads, 2.37× P50 for minority requests) (Georgia Tech/Intel, arXiv 2511.00739, Nov 2025)
- [[agentic-ai-workload-characteristics]] — Agentic AI Workload Characteristics: the canonical end-to-end agent workload measurement paper; Claude Code + Qwen3.6-27B/Gemma4-31B on five real benchmarks with OpenTelemetry-grade tracing; **the source-of-truth for the LLM=71-98% / tool=2-29% breakdown**, decode dominance 91-98.6%, cache hit 84.6-99.5%, and reasoning's 6× trajectory compaction effect (UIUC/Intel, arXiv 2605.26297, May 2026)
- [[speculative-actions]] — Speculative Actions: lossless framework that lifts speculate-verify from token-level to **API-call level across the entire agentic environment**; pairs slow Actor (GPT-5 high reasoning) with fast Speculator (smaller model) via k-way breadth speculation; closed-form cost-latency theorems; 4 environments (chess 19.5% time saved, e-commerce ~1/3 turns immediate, HotpotQA 46% top-3 prediction, OS tuning P95 37.93ms vs 102.97ms) (Columbia, arXiv 2510.04371, Oct 2025)
- [[speceyes]] — SpecEyes: agentic-level speculative acceleration for **multimodal LLMs** (DeepEyes, Thyme); 4-phase pipeline routes tool-free queries to small non-agentic MLLM (Qwen3-VL-2B); novel **answer separability score** $S_{sep}$ replaces softmax for cognitive gating (Δ 14× larger than softmax); heterogeneous parallel funnel for throughput; 1.42-1.73× avg speedup with up to +6.7% accuracy on V*/HR-Bench/POPE (Xiamen U/Rochester/OSU, arXiv 2603.23483, Mar 2026)
- [[speccache]] — SpecCache + What Limits Agentic Systems Efficiency?: dual contribution — (1) 5-day, 5-provider, 9-model LLM API latency variance study finding **69.21× variability** and **53.7% web-env latency share** in Reflexion-based web agents; (2) **SpecCache** caching framework with action-observation LRU + draft-LLM model-based prefetching achieves **9.4-54× cache hit rate improvement** (8.9%→83.3% WebWalkerQA, 1.0%→54.0% Frames) and **3.2× web env latency reduction** without trajectory degradation. Empirical validation of [[speculative-actions]]' theoretical framework for web environments (UW-Madison/Toronto/NVIDIA, arXiv 2510.16276, Oct 2025)
- [[mori]] — MORI: Memory Offloader with Relative Idleness — **direct successor to [[continuum|Continuum]]** by same first author Hanchen Li. Where Continuum was single-tier (GPU only) with TTL pinning, MORI is **two-tier (GPU HBM + CPU DRAM)** with a continuous **idleness metric** $\iota = T_{acting}^{(k)}/(T_{reasoning}^{(k)}+T_{acting}^{(k)})$ that ranks all programs along a relative spectrum, three-tier queue (GPU/CPU/Waiting), sticky rebalancing to avoid PCIe churn, and typed eviction (busy/idle/inactive labels propagated to engine). Tested on real Claude Code SWE-bench Pro traces: **20-71% higher throughput, 18-43% lower TTFT, up to 2.8× TTFT reduction**; 99%+ GPU utilization at 80 concurrent programs vs 59-76% for phase-oblivious schedulers; +71% on B200/Llama-70B (the load-bearing high-memory-pressure result) (UC Berkeley/Renmin/Stanford/Georgia Tech, arXiv 2606.00866, May 2026)
- [[infercept]] — InferCept: **the original agent-aware KV-pinning paper** that [[continuum|Continuum]] and [[mori|MORI]] both cite as baseline. First LLM inference framework designed for augmented LLMs with interceptions; establishes the canonical `Discard` / `Preserve` / `Swap` taxonomy and four closed-form *waste equations* (WasteDiscard / WastePreserve / WasteSwap / WasteChunkD). Three engineering improvements (swap pipelining + chunking ≈ 96% Swap-waste elimination; recomputation chunking at GPU saturation point S ≈ halves Discard waste; dynamic $\hat{T}_{INT} = t_{now} - t_{call}$ estimator ≈ 93% of oracle) feeding into a min-waste scheduler. **1.6×–2× throughput**, 1.9×–5.7× lower normalized latency (6B), 1.6×–10× (13B-TP2), **1.3×–12× (70B-TP4)**, GPU memory waste from ~25% → **0.69%**. Limitations later targeted by Continuum (per-turn queueing-delay term + TTL bound) and MORI (two-tier idleness vs binary GPU/CPU swap) (UCSD WukLab, ICML 2024, arXiv 2402.01869)

## Compound AI systems

- [[compound-ai-systems]] — agent serving as a compound system

  - [[compound-ai-systems#System Components|System components (LLM, RAG, tools, RM, router)]]
  - [[compound-ai-systems#Architecture Patterns|Architecture patterns]]
  - [[compound-ai-systems#DSPy: A Compound-System Optimization Framework|DSPy framework]]
  - [[compound-ai-systems#Evaluation Challenges|Evaluation challenges]]
  - [[compound-ai-systems#Performance Optimization|Performance optimization]]
