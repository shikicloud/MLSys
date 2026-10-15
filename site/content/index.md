---
title: "Shiki's Knowledge Wiki"
---

Personal knowledge base on LLM inference, RL infrastructure, agentic systems, and ML systems.

## Daily reading / 每日论文

**[今日推荐与历史归档](CN/reading/index)** · [经典阅读路线](CN/reading/roadmap) · [English](EN/reading/index) · [订阅 RSS](https://shikicloud.github.io/MLSys/reading.xml)

每天一篇主读，另附 2–4 篇选读。从经典基础逐步走向 agent serving 与 RL infrastructure；不限定论文发表时间。

## Pick a language / 选择语言

- **[English](EN/index)** — full wiki in English
- **[中文](CN/index)** — 完整 wiki，中文

---

**Topics covered** (same in both languages):

- LLM inference & serving — vLLM, SGLang, TensorRT-LLM, PagedAttention, speculative decoding, KV cache optimization, quantization, parallelism strategies, prefill-decode disaggregation, attention-FFN disaggregation
- RL infrastructure — RLHF, PPO, GRPO, DPO, on-policy distillation (OPD), DeepSeek-V4 OPD, MOPD, reward modeling, RL training frameworks
- ML infrastructure — distributed training, training frameworks (Megatron-LM, DeepSpeed, FSDP), checkpointing, GPU cluster management
- Agentic RL — ProRL Agent, NeMo Gym, environment design, tool-use RL, multi-step reasoning RL
- AI agents — agent architectures, MCP protocol, agent frameworks
- LLM serving for AI agents — long-context serving, structured-output serving, function-calling optimization

Source repo: [github.com/shikicloud/MLSys](https://github.com/shikicloud/MLSys)
