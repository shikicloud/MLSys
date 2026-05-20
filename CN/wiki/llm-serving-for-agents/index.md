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

## 复合 AI 系统

- [[compound-ai-systems]] —— 智能体服务作为复合系统

  - [[compound-ai-systems#系统组件|系统组件（LLM、RAG、tools、RM、router）]]
  - [[compound-ai-systems#架构模式|架构模式]]
  - [[compound-ai-systems#DSPy：复合系统优化框架|DSPy 框架]]
  - [[compound-ai-systems#评估挑战|评估挑战]]
  - [[compound-ai-systems#性能优化|性能优化]]
