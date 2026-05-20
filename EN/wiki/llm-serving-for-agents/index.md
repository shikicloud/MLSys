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

## Compound AI systems

- [[compound-ai-systems]] — agent serving as a compound system

  - [[compound-ai-systems#System Components|System components (LLM, RAG, tools, RM, router)]]
  - [[compound-ai-systems#Architecture Patterns|Architecture patterns]]
  - [[compound-ai-systems#DSPy: A Compound-System Optimization Framework|DSPy framework]]
  - [[compound-ai-systems#Evaluation Challenges|Evaluation challenges]]
  - [[compound-ai-systems#Performance Optimization|Performance optimization]]
