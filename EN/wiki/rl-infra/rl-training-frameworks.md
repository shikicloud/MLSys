---
title: "RL Training Frameworks"
category: rl-infra
tags: [openrlhf, verl, trl, deepspeed-chat, nemo-aligner, training-framework, rlhf-infrastructure]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# RL Training Frameworks

> [!abstract]+ TL;DR
> LLM RL training (RLHF/RLAIF/RLVR) is one of the most complex training paradigms in AI engineering — coordinating multiple large models, interleaving generation and training, and managing GPUs across distributed environments. Three frameworks dominate the 2025-2026 ecosystem: **OpenRLHF** (Ray + vLLM + DeepSpeed ZeRO-3, production-grade large-scale), **veRL** (ByteDance, FSDP/Megatron + multiple rollout backends, EP scales to 671B, official DAPO trainer), **TRL** (HuggingFace, lowest barrier to entry, unified post-training stack in v1.0 April 2026). Comparison: veRL is the heaviest (32K LOC), OpenRLHF is the lightest (8.5K LOC) and 1.22-1.68x faster than alternatives.

## Overview

LLM RL training (RLHF/RLAIF/RLVR) is one of the most complex training paradigms in AI engineering today. Unlike standard pretraining or fine-tuning, RL training requires simultaneously coordinating multiple large models, alternating between generation and training, and efficiently managing GPU resources in a distributed environment.

Traditional deep learning training frameworks (DeepSpeed, Megatron-LM) focus on forward/backward pass for a single model. The unique demands of RL training have spawned dedicated RL training frameworks.

As of 2025-2026, three open-source frameworks dominate the ecosystem: **OpenRLHF**, **veRL**, and **TRL**, each with a different niche.

---

## Core Challenges

### 1. Multi-model coordination

For PPO, a complete RL training run coordinates 4 models:

```
PPO's 4 models:
┌────────────────────────────────────────────────────┐
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │  Actor    │  │  Critic  │  │ Reference│  │  Reward  │
│  │ (policy)  │  │ (value)  │  │  (ref)   │  │  (reward)│
│  │ trainable │  │ trainable│  │ frozen   │  │ frozen   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘
│       │              │             │              │
│       ▼              ▼             ▼              ▼
│  generate     estimate value   compute KL    compute reward
│                                                     │
└────────────────────────────────────────────────────┘
```

For a 70B model, the memory required for the 4 models is roughly:
- Actor (bf16): ~140 GB
- Critic (bf16): ~140 GB
- Reference (bf16): ~140 GB
- Reward Model (bf16): ~140 GB
- **Total**: ~560 GB + gradients + optimizer states

This is why [[grpo|GRPO]] and REINFORCE++ are so valuable in dropping the Critic — directly cutting 25% of model memory.

### 2. Generation + training loop

The unique aspect of RL training is alternating inference (generate responses) and training (update parameters):

```
RL training loop:
┌──────────────────────────────────────────────────┐
│  ┌─────────┐                                      │
│  │ Sample  │ ← Inference optimizations (vLLM, KV cache) │
│  │ Actor    │                                      │
│  │ generates│ ← GPU usage: compute-bound inference  │
│  └────┬────┘                                      │
│       │ rollouts                                   │
│       ▼                                            │
│  ┌─────────┐                                      │
│  │ Evaluate│ ← RM + Reference forward pass         │
│  │ rewards │                                      │
│  │ advantages│ ← GAE / group normalization         │
│  └────┬────┘                                      │
│       │ (states, actions, rewards, advantages)     │
│       ▼                                            │
│  ┌─────────┐                                      │
│  │ Train   │ ← Training optimizations (ZeRO, FSDP) │
│  │ PPO/GRPO│                                      │
│  │ update  │ ← GPU usage: gradient + optimizer     │
│  └────┬────┘                                      │
│       │                                            │
│       └──→ Back to sampling (next iteration)        │
└──────────────────────────────────────────────────┘
```

**Key challenge**: inference and training have completely different GPU usage patterns:
- **Inference**: needs KV cache, low latency, high throughput (favors tensor parallel)
- **Training**: needs gradient storage, optimizer state (favors ZeRO/FSDP/pipeline parallel)

### 3. GPU memory management

```
Memory timeline of one RL training step:

Time ──────────────────────────────────────────────→

[  Generation phase  ]
  Actor weights + KV cache       ████████████
  Other models                                  (offloadable)

[  Evaluation phase  ]
  Actor weights                  ████████
  RM weights                     ████████
  Reference weights              ████████
  Rollout data                   ████

[  Training phase  ]
  Actor weights                  ████████
  Actor gradients                ████████
  Actor optimizer state          ████████████████
  Critic w + grad + optim        ████████████████████████
  Rollout data                   ████

Peak memory can be 3-4x static model weights!
```

**Optimization strategies**:
- Reuse GPU memory between generation and training (KV cache and optimizer state aren't needed simultaneously)
- Place frozen models (Reference, RM) on separate GPUs or offload to CPU
- Use model parallelism to reduce per-GPU memory

### 4. Distributed training

Large-model RL training needs cross-node multi-GPU strategies, and different models may need different parallel strategies:

```
Example: distributed RL training of a 70B Actor

Node 0-3:  Actor (TP=8, DP=4) — 32 GPUs
Node 4-5:  Critic (TP=4, DP=4) — 16 GPUs
Node 6:    Reference (TP=8) — 8 GPUs  (inference only)
Node 7:    Reward Model (TP=8) — 8 GPUs  (inference only)

Total: 64 GPUs for one PPO training run
```

---

## OpenRLHF

### Architecture overview

OpenRLHF is the first high-performance, production-ready open-source RLHF framework. Its core design principle is **separating generation and training**, using Ray for distributed scheduling, vLLM for efficient inference, and DeepSpeed ZeRO for training.

```
OpenRLHF architecture:
┌──────────────────────────────────────────────────────────┐
│                     Ray Cluster                           │
│                                                           │
│  ┌─────────────────────┐   ┌─────────────────────────┐   │
│  │  vLLM inference      │   │  DeepSpeed training      │   │
│  │                     │   │                         │   │
│  │  ┌───────────────┐  │   │  ┌───────────────────┐  │   │
│  │  │ Actor (infer) │  │   │  │ Actor (train)     │  │   │
│  │  │ AutoTP        │  │   │  │ ZeRO-3            │  │   │
│  │  │ KV cache      │  │   │  │ grad + optimizer  │  │   │
│  │  └───────────────┘  │   │  └───────────────────┘  │   │
│  │  ┌───────────────┐  │   │  ┌───────────────────┐  │   │
│  │  │ Reference     │  │   │  │ Critic (train)    │  │   │
│  │  │ (frozen, inf) │  │   │  │ ZeRO-3            │  │   │
│  │  └───────────────┘  │   │  └───────────────────┘  │   │
│  │  ┌───────────────┐  │   │                         │   │
│  │  │ Reward Model  │  │   │                         │   │
│  │  │ (frozen, inf) │  │   │                         │   │
│  │  └───────────────┘  │   │                         │   │
│  └─────────────────────┘   └─────────────────────────┘   │
│                                                           │
│  Ray responsibilities:                                    │
│  - Cross-node scheduling                                  │
│  - Data transfer                                          │
│  - Weight sync (gen → train → gen)                        │
└──────────────────────────────────────────────────────────┘
```

### Core features

| Feature | Description |
|---------|-------------|
| **AutoTP** | Automatic tensor parallelism, no manual TP degree config |
| **RingAttention** | Long-context RL support (128K+ tokens) |
| **VLM RLHF** | v0.10+ supports RL on vision-language models |
| **Async agentic RL** | Supports multi-turn RL with environment interaction |
| **Hybrid engine** | vLLM for generation, DeepSpeed for training |
| **Weight sync** | Ray efficiently transfers updated weights to vLLM |

### Supported algorithms

- [[ppo-for-llm|PPO]] — classic on-policy policy gradient
- [[grpo|GRPO]] — critic-free group relative policy optimization
- REINFORCE++ — enhanced REINFORCE
- RLOO — leave-one-out baseline
- DAPO — dynamic-sampling policy optimization
- [[dpo|DPO]] / SimPO / KTO — offline preference optimization
- Reward model training — Bradley-Terry loss
- SFT — supervised fine-tuning

### Performance

```
14B model, 8K context, 8xH100 benchmark:

OpenRLHF:  328.6 s/iteration
veRL:      511.1 s/iteration
           ──────────────────
OpenRLHF is ~1.55x faster
```

At larger scale (hundreds of GPUs), OpenRLHF's Ray scheduling advantage becomes more pronounced.

### Code example

```bash
# Start OpenRLHF PPO training
ray start --head --num-gpus 8

python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
    --pretrain meta-llama/Llama-3.1-8B-Instruct \
    --reward_pretrain OpenRLHF/Llama-3.1-8B-RM \
    --save_path ./output \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 2048 \
    --advantage_estimator gae \
    --bf16 \
    --flash_attn
```

```bash
# Start OpenRLHF GRPO training (simpler, no Critic or RM needed)
python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --pretrain meta-llama/Llama-3.1-8B-Instruct \
    --save_path ./grpo_output \
    --advantage_estimator group_norm \
    --remote_rm_url "http://localhost:5000/verify" \
    --bf16
```

[GitHub](https://github.com/OpenRLHF/OpenRLHF) | [Paper](https://arxiv.org/abs/2405.11143) (EMNLP 2025)

---

## TRL (Transformer Reinforcement Learning)

### Positioning

TRL is HuggingFace's full-stack post-training library with the lowest barrier to entry. Suitable for rapid prototyping and medium-scale training.

### Architecture

```
TRL architecture (simplicity first):
┌──────────────────────────────────────────┐
│              TRL Library                  │
│                                           │
│  ┌──────────────────────────────────┐    │
│  │        Unified Trainer API        │    │
│  │  SFTTrainer | DPOTrainer |        │    │
│  │  GRPOTrainer | PPOTrainer |       │    │
│  │  RewardTrainer | KTOTrainer       │    │
│  └──────────────┬───────────────────┘    │
│                  │                        │
│  ┌───────────────┴──────────────────┐    │
│  │    HuggingFace Trainer Backend    │    │
│  │  (or optional vLLM for rollouts)  │    │
│  │                                   │    │
│  │  DeepSpeed / FSDP integration     │    │
│  │  (via Accelerate)                 │    │
│  └───────────────────────────────────┘    │
│                                           │
│  ┌───────────────────────────────────┐    │
│  │           TRL CLI (v1.0)          │    │
│  │  trl sft | trl dpo | trl grpo    │    │
│  │  One-line training startup        │    │
│  └───────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

### Supported methods

| Trainer | Method | Use |
|---------|--------|-----|
| `SFTTrainer` | Supervised fine-tuning | Base alignment |
| `RewardTrainer` | Reward model training | Bradley-Terry loss |
| `PPOTrainer` | PPO | On-policy gradient |
| `DPOTrainer` | DPO/IPO | Offline preference optimization |
| `KTOTrainer` | KTO | Binary feedback preference |
| `GRPOTrainer` | GRPO | Critic-free on-policy RL |
| `ORPOTrainer` | ORPO | Reference-free preference |
| `OnlineDPOTrainer` | Online DPO | Iterative DPO |

### TRL v1.0 (April 2026)

Major milestone:
- **CLI support**: `trl sft`, `trl dpo`, `trl grpo` one-liners
- **OpenEnv integration**: Meta's RL environment support (for agentic RL)
- **Unified post-training stack**: full pipeline from SFT to RL

### Code example: TRL GRPO training

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# 1. Model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 2. Dataset (only prompts needed)
dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train")

# 3. Define reward function (RLVR style)
def math_reward_fn(completions, ground_truths, **kwargs):
    """Verifiable reward: check whether math answer is correct"""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        answer = extract_answer(completion)
        rewards.append(1.0 if answer == gt else 0.0)
    return rewards

# 4. Config
training_args = GRPOConfig(
    output_dir="./grpo_math",
    per_device_train_batch_size=4,
    num_generations=8,        # 8 candidates per prompt
    max_completion_length=2048,
    max_prompt_length=512,
    learning_rate=1e-6,
    num_train_epochs=3,
    beta=0.04,                # KL penalty
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    # vLLM-accelerated generation (optional)
    use_vllm=True,
    vllm_gpu_utilization=0.7,
)

# 5. Train
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=[math_reward_fn],
)
trainer.train()
```

### TRL CLI quickstart

```bash
# SFT
trl sft \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --dataset_name tatsu-lab/alpaca \
  --output_dir ./sft_output

# DPO
trl dpo \
  --model_name_or_path ./sft_output \
  --dataset_name argilla/ultrafeedback-binarized-preferences-cleaned \
  --beta 0.1 \
  --output_dir ./dpo_output

# GRPO (needs a custom reward function script)
trl grpo \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --reward_funcs math_reward \
  --dataset_name AI-MO/NuminaMath-TIR \
  --output_dir ./grpo_output
```

### Simplicity vs performance tradeoff

| Aspect | TRL | OpenRLHF / veRL |
|--------|-----|-----------------|
| Onboarding | Extremely low | Medium |
| Code changes | Few lines | Need Ray/scheduling config |
| Single-GPU training | Good support | Not suitable |
| Multi-node large scale | Weaker | Strong |
| Inference speed | Slower (HF generate) | Fast (vLLM/SGLang) |
| Best for | Research / prototype / small-scale | Production / large-scale |

[Docs](https://huggingface.co/docs/trl/en/index) | [GitHub](https://github.com/huggingface/trl)

---

## veRL (Volcano Engine RL)

### Background

veRL is the RL training framework developed by ByteDance. Its core paper "HybridFlow" was published at EuroSys 2025.

### Core design

veRL's key innovations are **FSDP-based hybrid parallelism** and **flexible model placement strategies**:

```
veRL architecture:
┌──────────────────────────────────────────────────────────┐
│                    veRL HybridFlow                         │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │               Resource Manager                       │ │
│  │  (flexibly allocate GPUs to different models/phases) │ │
│  └────────────────────┬────────────────────────────────┘ │
│                       │                                   │
│       ┌───────────────┼───────────────┐                   │
│       │               │               │                   │
│       ▼               ▼               ▼                   │
│  ┌─────────┐   ┌─────────────┐  ┌─────────────┐         │
│  │ Generate │   │  Evaluate   │  │   Train     │         │
│  │          │   │             │  │             │         │
│  │ vLLM /   │   │ Reference   │  │ FSDP /      │         │
│  │ SGLang /  │   │ + RM        │  │ Megatron    │         │
│  │ HF       │   │ forward     │  │ full-shard   │         │
│  └─────────┘   └─────────────┘  └─────────────┘         │
│                                                           │
│  Key features:                                            │
│  - Same GPUs reused across phases                         │
│  - No Ray scheduling overhead                             │
│  - Native FSDP integration                                │
│  - Expert parallel support (MoE)                          │
└──────────────────────────────────────────────────────────┘
```

### Core features

| Feature | Description |
|---------|-------------|
| **HybridFlow** | Generate and train phases share the same GPUs |
| **FSDP + Megatron** | Two training backends supported |
| **vLLM + SGLang** | Flexible choice of inference engine |
| **671B scaling** | Expert parallelism for ultra-large MoE models |
| **Resource reuse** | No need to dedicate GPUs per model |

### Supported algorithms

PPO, [[grpo|GRPO]], GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO

veRL is the official training infrastructure for **DAPO**.

### Key achievements

- **671B model training**: scales to DeepSeek-V3 level via expert parallelism
- **Math SOTA**: RL scaling preview reached 70.0 pass@1 on AIME (o1 level)
- **v0.3.0**: ~1.4x speedup over prior versions

### Code example

```python
# veRL config example (YAML style)
"""
verl_config:
  algorithm: grpo
  model:
    actor: meta-llama/Llama-3.1-8B-Instruct
    ref: meta-llama/Llama-3.1-8B-Instruct

  training:
    backend: fsdp          # or megatron
    lr: 1e-6
    batch_size: 128
    ppo_epochs: 1

  rollout:
    engine: vllm            # or sglang, hf
    temperature: 1.0
    top_p: 0.95
    num_generations: 8
    max_new_tokens: 2048

  reward:
    type: function           # or model
    function: math_verifier

  resource:
    num_gpus: 8
    gpu_memory_utilization: 0.85
"""
```

```bash
# Launch veRL training
python -m verl.trainer.main_ppo \
    --config config/grpo_llama8b.yaml \
    --num_gpus 8
```

### Key differences vs OpenRLHF

| Dimension | OpenRLHF | veRL |
|-----------|----------|------|
| Scheduling | Ray (separate gen/train on different GPUs) | Custom (same-GPU reuse) |
| Training backend | DeepSpeed ZeRO-3 | FSDP / Megatron |
| Resource utilization | Needs more GPUs (per-role) | More efficient (reuse) |
| Ultra-large models | Hundreds of GPUs | 671B (expert parallelism) |
| Complexity | Medium | Higher |

[GitHub](https://github.com/volcengine/verl) | [HybridFlow Paper](https://arxiv.org/abs/2409.19256) (EuroSys 2025)

---

## DeepSpeed-Chat

### Background

DeepSpeed-Chat is Microsoft's RLHF training pipeline, built on the DeepSpeed ecosystem. One of the earliest open-source RLHF frameworks (released 2023).

### Architecture

```
DeepSpeed-Chat architecture:
┌───────────────────────────────────────────────────┐
│              DeepSpeed-Chat Pipeline               │
│                                                    │
│  Step 1: SFT ──→ Step 2: RM Training ──→ Step 3: RLHF
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │          DeepSpeed Hybrid Engine          │     │
│  │                                          │     │
│  │  Training mode: ZeRO Stage 3              │     │
│  │     ↕ auto switching                     │     │
│  │  Inference mode: Tensor Parallel + KV cache│     │
│  │                                          │     │
│  │  Same model on same GPUs switches between │     │
│  │  inference and training modes             │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  ZeRO optimization:                                │
│  - Stage 1: shard optimizer state                  │
│  - Stage 2: + gradient sharding                    │
│  - Stage 3: + parameter sharding                   │
│  - ZeRO-Offload: CPU/NVMe offload                  │
└───────────────────────────────────────────────────┘
```

### Core innovation: Hybrid Engine

DeepSpeed-Chat's key innovation is the Hybrid Engine — a single model can seamlessly switch between inference and training modes on the same GPUs:

- **Inference mode**: uses Tensor Parallel + KV cache for fast generation
- **Training mode**: uses ZeRO-3 for gradient and parameter updates
- **Switching cost**: requires re-sharding parameters, non-trivial overhead

### Pros and cons

**Pros**:
- Complete 3-stage RLHF pipeline (SFT → RM → PPO)
- Deep DeepSpeed ZeRO integration, memory efficient
- Single command for the full pipeline
- Good documentation

**Cons**:
- Inference is slower than vLLM (no PagedAttention, continuous batching, etc.)
- Switching overhead
- No GRPO and other new algorithms (community maintenance less active than OpenRLHF/veRL)
- Scalability lags behind Ray-based solutions

[GitHub](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) | [Blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)

---

## NeMo-Aligner

### Background

NeMo-Aligner is NVIDIA's alignment framework, built on the NeMo training platform. Targets large-scale deployments on NVIDIA GPU clusters.

### Characteristics

```
NeMo-Aligner:
┌───────────────────────────────────────────┐
│  Built on NeMo 2.0 platform                │
│                                            │
│  Training backend: Megatron-LM             │
│  - Tensor Parallel                         │
│  - Pipeline Parallel                       │
│  - Expert Parallel (MoE)                   │
│  - Context Parallel                        │
│                                            │
│  Supported methods:                         │
│  - PPO (RLHF)                              │
│  - DPO                                     │
│  - SteerLM (attribute-controlled align)    │
│  - Self-Play Fine-Tuning (SPIN)            │
│  - Reward model training                   │
│                                            │
│  Strengths:                                 │
│  - Deep NVIDIA GPU optimization             │
│  - Megatron-level scalability               │
│  - Seamless integration with NeMo ecosystem │
│                                            │
│  Weaknesses:                                │
│  - Steep learning curve                     │
│  - Tied to NVIDIA ecosystem                 │
│  - Less active community than OpenRLHF/TRL  │
└───────────────────────────────────────────┘
```

[GitHub](https://github.com/NVIDIA/NeMo-Aligner) | [Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-aligner/)

---

## Framework Comparison

### Overall comparison

| Feature | OpenRLHF | veRL | TRL | DeepSpeed-Chat | NeMo-Aligner |
|---------|:--------:|:----:|:---:|:--------------:|:------------:|
| **LOC** | ~8.5K | ~32K | ~19K | ~5K | ~15K |
| **Training backend** | DeepSpeed ZeRO-3 | FSDP/Megatron | HF Trainer | DeepSpeed ZeRO | Megatron-LM |
| **Inference engine** | vLLM | vLLM/SGLang/HF | HF/vLLM | DeepSpeed Hybrid | — |
| **Scheduling** | Ray | Custom | Mostly single-node | Mostly single-node | Slurm/K8s |
| **PPO** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **GRPO** | ✓ | ✓ | ✓ | ✗ | ✗ |
| **DPO** | ✓ | ✓ | ✓ | ✗ | ✓ |
| **REINFORCE++** | ✓ | ✓ | ✗ | ✗ | ✗ |
| **VLM RL** | ✓ (v0.10+) | ✓ | ✗ | ✗ | ✗ |
| **Max scale** | Hundreds of GPUs | 671B models | Medium | Medium | Large-scale |
| **Onboarding** | Medium | Higher | **Lowest** | Medium | High |
| **Community activity** | High | High | Highest | Med-low | Medium |
| **Best for** | Production RL | Large-scale research | Prototype | DeepSpeed users | NVIDIA ecosystem |

### Performance comparison (known benchmarks)

| Scenario | OpenRLHF | veRL | Notes |
|----------|:--------:|:----:|-------|
| 14B, 8K ctx, 8xH100 | 328.6s | 511.1s | OpenRLHF 1.55x faster |
| Large MoE | not tested | 671B | veRL supports larger |
| Multi-node scaling | 1.22-1.68x vs alt. | ~1.4x vs prev. ver. | Different baselines |

---

## Selection Guide

```
Which framework should you choose?

  ┌── What's your goal?
  │
  ├── Quick experiment / learning / prototype
  │   └── ✅ TRL
  │       - pip install trl
  │       - A few lines to start
  │       - Complete docs and tutorials
  │
  ├── Production-grade large-scale RL training
  │   └── ✅ OpenRLHF
  │       - Ray cluster scheduling
  │       - vLLM efficient inference
  │       - Battle-tested at scale
  │
  ├── Ultra-large models (100B+) / MoE research
  │   └── ✅ veRL
  │       - FSDP/Megatron hybrid parallelism
  │       - Expert parallel support
  │       - 671B-scale validated
  │
  ├── Existing DeepSpeed infrastructure
  │   └── ⚠️ DeepSpeed-Chat
  │       - Works but slow to update
  │       - Consider migrating to OpenRLHF
  │
  └── Deep NVIDIA / NeMo ecosystem integration
      └── ⚠️ NeMo-Aligner
          - Megatron-level scalability
          - Steeper learning curve
```

### Typical workflows

```
Researcher path:
  1. TRL to validate ideas (1-8 GPU)
  2. Scale up with OpenRLHF/veRL (32-256 GPU)
  3. Custom framework for extreme optimization (optional)

Engineering team path:
  1. Build production pipeline on OpenRLHF
  2. Choose veRL (ultra-large) or OpenRLHF (general) based on scale
  3. Continuously track framework updates and new algorithm support
```

---

## Appendix: Key Algorithm Quick Reference

### REINFORCE++

Enhanced REINFORCE that integrates PPO's stabilization techniques without a Critic:
- Clipped surrogate loss
- Token-level KL penalty
- Global advantage normalization

**Positioning**: more stable than [[grpo|GRPO]], faster than [[ppo-for-llm|PPO]].

[arXiv:2501.03262](https://arxiv.org/html/2501.03262v5)

### RLOO (REINFORCE Leave-One-Out)

Use the average reward of the other k-1 samples as an unbiased baseline:

```
For sample i among k samples:
baseline_i = (1/(k-1)) · Σ_{j≠i} r(x, y_j)
advantage_i = r(x, y_i) - baseline_i
```

**Advantages**: 50-70% less memory than PPO, 2-3x faster.

[HuggingFace Blog](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)

### DAPO (Dynamic Advantage Policy Optimization)

Dynamic-sampling policy optimization, from the veRL team:
- Dynamic sampling temperature
- Adaptive KL control
- Improved advantage estimation

[GitHub (veRL)](https://github.com/volcengine/verl)

---

## References

- Hu et al. (2024) — [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143) (EMNLP 2025)
- Sheng et al. (2024) — [HybridFlow: A Flexible and Efficient RLHF Framework (veRL)](https://arxiv.org/abs/2409.19256) (EuroSys 2025)
- von Werra et al. (2020-2026) — [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- Yao et al. (2023) — [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training](https://arxiv.org/abs/2308.01320)
- NVIDIA — [NeMo-Aligner Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-aligner/)
- Yu et al. (2025) — [REINFORCE++](https://arxiv.org/html/2501.03262v5)
- Ahmadian et al. (2024) — [RLOO: Back to Basics](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)

---

## Related Pages

- [[rlhf-overview]] — the RLHF pipeline these frameworks implement
- [[ppo-for-llm]] — PPO algorithm details
- [[grpo]] — the most popular critic-free RL algorithm
- [[dpo]] — offline preference optimization (supported by most frameworks)
- [[reward-modeling]] — reward model training (a key component for these frameworks)
- [[agentic-rl-overview]] — multi-turn RL / agentic RL frameworks
