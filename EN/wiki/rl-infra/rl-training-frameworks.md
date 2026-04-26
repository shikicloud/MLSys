---
title: "RL Training Frameworks"
category: rl-infra
tags: [openrlhf, verl, trl, deepspeed-chat, nemo-aligner, training-framework, rlhf-infrastructure]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# RL Training Frameworks

## Overview

RL training for LLMs (RLHF/RLAIF/RLVR) is one of the most complex training paradigms in AI engineering. Unlike standard pretraining or fine-tuning, it requires coordinating multiple large models, alternating between generation and training, and efficiently managing GPU resources across distributed environments. Three major open-source frameworks dominate as of 2025-2026: **OpenRLHF**, **veRL**, and **TRL**.

---

## Core Challenges

### Multi-Model Coordination

PPO requires 4 models simultaneously: Actor (trainable), Critic (trainable), Reference (frozen), Reward Model (frozen). For a 70B model in bf16, that is ~560 GB just for weights — before gradients and optimizer states. This is why [[grpo|GRPO]] and REINFORCE++ (no Critic) are so valuable: ~25% less model memory.

### Generation + Training Loop

```
RL Training Loop:
  [Rollout Phase] Actor generates responses (inference-optimized: vLLM, KV cache)
       ↓
  [Evaluation Phase] RM + Reference compute rewards and KL
       ↓
  [Training Phase] PPO/GRPO update (training-optimized: ZeRO, FSDP)
       ↓ repeat
```

Inference and training have completely different GPU utilization patterns, making resource management a key challenge.

### GPU Memory Management

Peak memory during training can be 3-4x static model weights. Key optimization: reuse GPU memory across phases (KV cache and optimizer states are never needed simultaneously).

---

## OpenRLHF

The first high-performance, production-ready open-source RLHF framework. Core design: **separate generation and training** using Ray for scheduling, vLLM for inference, DeepSpeed ZeRO for training.

```
OpenRLHF Architecture:
┌───────────────────────────────────────────────┐
│                 Ray Cluster                    │
│                                                │
│  ┌──────────────────┐  ┌──────────────────┐   │
│  │ vLLM Inference    │  │ DeepSpeed Train   │   │
│  │ Actor + Ref + RM  │  │ Actor + Critic    │   │
│  │ (AutoTP, KV Cache)│  │ (ZeRO-3)         │   │
│  └──────────────────┘  └──────────────────┘   │
│                                                │
│  Ray handles: scheduling, data transfer,       │
│  weight sync (train → inference → train)       │
└───────────────────────────────────────────────┘
```

**Supported algorithms**: PPO, [[grpo|GRPO]], REINFORCE++, RLOO, DAPO, [[dpo|DPO]], SimPO, KTO, SFT, RM training

**Key features**: AutoTP (automatic tensor parallelism), RingAttention for long contexts, VLM RLHF (v0.10+), async agentic RL, 1.22-1.68x speedup vs alternatives.

```bash
# Launch OpenRLHF GRPO training
python -m openrlhf.cli.train_ppo_ray \
    --actor_num_gpus_per_node 4 \
    --ref_num_gpus_per_node 2 \
    --vllm_num_engines 2 \
    --pretrain meta-llama/Llama-3.1-8B-Instruct \
    --advantage_estimator group_norm \
    --remote_rm_url "http://localhost:5000/verify"
```

[GitHub](https://github.com/OpenRLHF/OpenRLHF) | [Paper](https://arxiv.org/abs/2405.11143) (EMNLP 2025)

---

## TRL (Transformer Reinforcement Learning)

HuggingFace's full-stack post-training library. **Lowest barrier to entry**.

**Trainers**: SFTTrainer, RewardTrainer, PPOTrainer, DPOTrainer, GRPOTrainer, KTOTrainer, ORPOTrainer, OnlineDPOTrainer

**TRL v1.0 (April 2026)**: CLI support (`trl sft`, `trl dpo`, `trl grpo`), unified post-training stack, OpenEnv integration (Meta's RL environments).

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,
    beta=0.04,
    learning_rate=1e-6,
    use_vllm=True,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[math_reward_fn],
    processing_class=tokenizer,
)
trainer.train()
```

**Trade-off**: Easiest to start with, but inference is slower than vLLM-native frameworks and multi-node scaling is weaker.

[Docs](https://huggingface.co/docs/trl/en/index) | [GitHub](https://github.com/huggingface/trl)

---

## veRL (Volcano Engine RL)

ByteDance's framework. HybridFlow paper (EuroSys 2025). Core innovation: **FSDP-based hybrid parallelism** with flexible model placement — the same GPUs are reused across generation and training phases (no separate GPU pools).

**Supported algorithms**: PPO, [[grpo|GRPO]], GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO

**Key features**: Scales to 671B models via expert parallelism, DAPO's official training infrastructure, v0.3.0 ~1.4x speedup, RL scaling preview reached o1-level math (70.0 pass@1 AIME).

**Key difference from OpenRLHF**: veRL reuses GPUs across phases (more resource-efficient) vs. OpenRLHF's Ray-based separation (better for heterogeneous clusters).

[GitHub](https://github.com/volcengine/verl) | [Paper](https://arxiv.org/abs/2409.19256) (EuroSys 2025)

---

## DeepSpeed-Chat

Microsoft's RLHF pipeline (2023). Core innovation: **Hybrid Engine** — same model switches between inference mode (TP + KV cache) and training mode (ZeRO-3) on the same GPUs.

Complete 3-stage pipeline (SFT → RM → PPO) with deep DeepSpeed ZeRO integration. However, inference speed lags vLLM, and community activity is lower than OpenRLHF/veRL. Does not support GRPO or newer algorithms.

## NeMo-Aligner

NVIDIA's alignment framework built on NeMo 2.0 with Megatron-LM backend. Supports PPO, DPO, SteerLM, SPIN. Best for teams with NVIDIA GPU clusters already in the NeMo ecosystem. Steep learning curve.

---

## Framework Comparison

| Feature | OpenRLHF | veRL | TRL | DeepSpeed-Chat | NeMo-Aligner |
|---------|:--------:|:----:|:---:|:--------------:|:------------:|
| Train backend | DeepSpeed ZeRO-3 | FSDP/Megatron | HF Trainer | DeepSpeed ZeRO | Megatron-LM |
| Rollout engine | vLLM | vLLM/SGLang/HF | HF/vLLM | Hybrid Engine | — |
| Scheduling | Ray | Custom | Single-node | Single-node | Slurm/K8s |
| PPO | Yes | Yes | Yes | Yes | Yes |
| GRPO | Yes | Yes | Yes | No | No |
| DPO | Yes | Yes | Yes | No | Yes |
| Max scale | Hundreds of GPUs | 671B models | Medium | Medium | Large |
| Ease of use | Medium | Higher | **Easiest** | Medium | Hard |
| Best for | Production RL | Large-scale research | Prototyping | DeepSpeed users | NVIDIA ecosystem |

**Performance** (14B model, 8K context, 8xH100): OpenRLHF 328.6s vs veRL 511.1s per iteration (~1.55x faster).

---

## Selection Guide

```
Quick experiment / learning / prototype → TRL
Production-scale RL training           → OpenRLHF
Ultra-large models (100B+) / MoE       → veRL
Existing DeepSpeed infra               → DeepSpeed-Chat (consider migrating)
NVIDIA NeMo ecosystem                  → NeMo-Aligner
```

**Typical researcher path**: TRL to validate ideas (1-8 GPU) → OpenRLHF/veRL to scale (32-256 GPU).

---

## Appendix: Key Algorithms

**REINFORCE++**: Enhanced REINFORCE with PPO's stabilization (clipped loss, token-level KL, global advantage normalization) but no critic. More stable than GRPO, faster than PPO. [arXiv:2501.03262](https://arxiv.org/html/2501.03262v5)

**RLOO**: Uses k-1 other samples as unbiased baseline. 50-70% less vRAM than PPO, 2-3x faster. [HuggingFace blog](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)

**DAPO** (Dynamic Advantage Policy Optimization): Dynamic sampling temperature and adaptive KL control. veRL team's contribution, with improved advantage estimation.

---

## Typical Workflow

```
Researcher path:
  1. TRL to validate ideas (1-8 GPUs)
  2. OpenRLHF/veRL to scale up (32-256 GPUs)
  3. Custom framework for extreme optimization (optional)

Engineering team path:
  1. OpenRLHF to build production pipeline
  2. Choose veRL (ultra-large models) or OpenRLHF (general) based on scale
  3. Track framework updates and new algorithm support
```

---

## References

- Hu et al. (2024) — [OpenRLHF](https://arxiv.org/abs/2405.11143) (EMNLP 2025)
- Sheng et al. (2024) — [HybridFlow (veRL)](https://arxiv.org/abs/2409.19256) (EuroSys 2025)
- von Werra et al. — [TRL](https://github.com/huggingface/trl)
- Yao et al. (2023) — [DeepSpeed-Chat](https://arxiv.org/abs/2308.01320)
- NVIDIA — [NeMo-Aligner](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-aligner/)

## Related Pages

- [[rlhf-overview]] — The pipeline these frameworks implement
- [[ppo-for-llm]] — Algorithm details
- [[grpo]] — Most popular algorithm on these frameworks
- [[dpo]] — Offline preference optimization (supported by most frameworks)
- [[reward-modeling]] — RM training as a key framework component
- [[agentic-rl-overview]] — Multi-turn RL frameworks
