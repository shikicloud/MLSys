---
title: Agentic RL
---

# Agentic RL

RL for LLM agents — environment design, tool use, multi-step reasoning, and the supporting infrastructure.

## Overview

- [[agentic-rl-overview]] — agentic RL landscape

  - [[agentic-rl-overview#Differences from Traditional RLHF|Differences from traditional RLHF]]
  - [[agentic-rl-overview#Core Paradigm|Core paradigm]]
  - [[agentic-rl-overview#Key Research Directions|Key research directions]]
  - [[agentic-rl-overview#Major Frameworks (2025-2026)|Major frameworks (2025–2026)]]
  - [[agentic-rl-overview#Technical Challenges|Technical challenges]]

## Environment design

- [[environment-design]] — what makes a good RL environment for LLMs

  - [[environment-design#Environment Types|Environment types]]
  - [[environment-design#Key Design Principles|Design principles]]
  - [[environment-design#Representative Environments|Representative environments]]
  - [[environment-design#Environment Synthesis (2025-2026)|Environment synthesis (2025–2026)]]
  - [[environment-design#Reward Signal Design|Reward signal design]]

## Specialized RL training

- [[tool-use-rl]] — RL for tool / API use
- [[multi-step-reasoning-rl]] — RL for multi-step reasoning (R1-style)

## Infrastructure (paper reviews)

- [[prorl-agent]] — ProRL Agent: rollout-as-a-service for multi-turn agentic RL (NVIDIA) **[superseded by [[polar]] May 2026]**
- [[polar]] — Polar: ProRL Agent successor; LLM-API proxy lets any unmodified harness (Codex, Claude Code, Qwen Code, Pi) be trained; registered as NeMo Gym environment (NVIDIA, May 2026)
- [[nemo-gym]] — NeMo Gym: NVIDIA's RL environment framework (84 benchmarks, 19 harnesses)
