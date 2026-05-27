---
title: 智能体 RL
---

# 智能体 RL

面向 LLM agent 的 RL —— 环境设计、工具使用、多步推理，及配套基础设施。

## 总览

- [[agentic-rl-overview]] —— 智能体 RL 全景

  - [[agentic-rl-overview#与传统 RLHF 的区别|与传统 RLHF 的区别]]
  - [[agentic-rl-overview#核心范式|核心范式]]
  - [[agentic-rl-overview#关键研究方向|关键研究方向]]
  - [[agentic-rl-overview#主要框架（2025-2026）|主要框架（2025–2026）]]
  - [[agentic-rl-overview#技术挑战|技术挑战]]

## 环境设计

- [[environment-design]] —— 什么样的 RL 环境对 LLM 来说算好

  - [[environment-design#环境类型|环境类型]]
  - [[environment-design#关键设计原则|设计原则]]
  - [[environment-design#代表性环境|代表性环境]]
  - [[environment-design#环境合成（2025-2026）|环境合成（2025–2026）]]
  - [[environment-design#奖励信号设计|奖励信号设计]]

## 专项 RL 训练

- [[tool-use-rl]] —— 工具 / API 使用的 RL
- [[multi-step-reasoning-rl]] —— 多步推理的 RL（R1 风格）

## 基础设施（论文精读）

- [[prorl-agent]] —— ProRL Agent：多轮 agentic RL 的 rollout 即服务（NVIDIA）**[2026-05 被 [[polar]] 取代]**
- [[polar]] —— Polar：ProRL Agent 续作；LLM-API proxy 让任何未修改 harness（Codex、Claude Code、Qwen Code、Pi）都可训练；注册为 NeMo Gym 环境（NVIDIA, 2026-05）
- [[nemo-gym]] —— NeMo Gym：NVIDIA 的 RL 环境框架（84 个 benchmark、19 个 harness）
