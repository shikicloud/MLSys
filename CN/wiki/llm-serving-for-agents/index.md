---
title: 面向 AI 智能体的 LLM 服务
---

# 面向 AI 智能体的 LLM 服务

面向 AI agent 工作负载的推理侧优化 —— 多轮对话、工具调用、结构化输出、复合系统。

## 智能体服务挑战

- [[agent-serving-challenges]] —— 智能体服务跟聊天机器人服务的差异

  - [[agent-serving-challenges#智能体 vs. 聊天机器人工作负载|智能体 vs 聊天机器人]]
  - [[agent-serving-challenges#关键挑战详解|关键挑战]]
  - [[agent-serving-challenges#专用系统|专用系统（Parrot、AsyncFlow）]]
  - [[agent-serving-challenges#基准分析|基准分析]]

## 多轮优化

- [[multi-turn-optimization]] —— KV 复用、前缀缓存、会话管理

  - [[multi-turn-optimization#跨轮 KV 缓存复用|跨轮 KV 复用]]
  - [[multi-turn-optimization#前缀缓存实现|前缀缓存实现]]
  - [[multi-turn-optimization#分离式架构下的多轮挑战|分离式架构下的多轮挑战]]
  - [[multi-turn-optimization#上下文窗口管理策略|上下文窗口管理]]
  - [[multi-turn-optimization#提示缓存 (Prompt Caching)|Prompt 缓存]]
  - [[multi-turn-optimization#会话管理与路由|会话管理与路由]]

## 论文精读

- [[continuum]] —— Continuum：基于 TTL 的 KV cache pinning + program 级 FCFS 给多轮 agent serving；第一个建模 per-turn queueing delay（不只是 reload cost）；延迟降低 1.12–3.66×，真实 SWE-agent 分布式 setting 上多达 8.18×（UC Berkeley/Stanford/Tensormesh, arXiv 2511.02230, 2025-11）
- [[cpu-centric-agentic-ai]] —— Agentic AI 的 CPU 中心视角：characterization（tool 时间在非对称 CPU-GPU 硬件上 tool 主导 workload 多达 88%）+ **COMB**（CPU 感知 overlapped micro-batching，P50 1.7× / 服务延迟 3.9×）+ **MAS**（chat+agent 混合 workload 的 mixed agentic 调度，少数请求 P50 2.37×）（Georgia Tech/Intel, arXiv 2511.00739, 2025-11）
- [[agentic-ai-workload-characteristics]] —— Agentic AI Workload Characteristics：标杆端到端 agent workload 测量论文；Claude Code + Qwen3.6-27B/Gemma4-31B on 5 个真实 benchmark with OpenTelemetry 级 tracing；**LLM=71-98% / tool=2-29% 分解的权威来源**，decode 主导 91-98.6%，cache 命中 84.6-99.5%，推理的 6× 轨迹紧凑化效应（UIUC/Intel, arXiv 2605.26297, 2026-05）

## 复合 AI 系统

- [[compound-ai-systems]] —— 智能体服务作为复合系统

  - [[compound-ai-systems#系统组件|系统组件（LLM、RAG、tools、RM、router）]]
  - [[compound-ai-systems#架构模式|架构模式]]
  - [[compound-ai-systems#DSPy：复合系统优化框架|DSPy 框架]]
  - [[compound-ai-systems#评估挑战|评估挑战]]
  - [[compound-ai-systems#性能优化|性能优化]]
