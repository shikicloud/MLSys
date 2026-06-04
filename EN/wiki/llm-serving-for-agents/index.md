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

## Compound AI systems

- [[compound-ai-systems]] — agent serving as a compound system

  - [[compound-ai-systems#System Components|System components (LLM, RAG, tools, RM, router)]]
  - [[compound-ai-systems#Architecture Patterns|Architecture patterns]]
  - [[compound-ai-systems#DSPy: A Compound-System Optimization Framework|DSPy framework]]
  - [[compound-ai-systems#Evaluation Challenges|Evaluation challenges]]
  - [[compound-ai-systems#Performance Optimization|Performance optimization]]
