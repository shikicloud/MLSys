---
title: "RL 训练框架"
category: rl-infra
tags: [openrlhf, verl, trl, deepspeed-chat, nemo-aligner, 训练框架, rlhf基础设施]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# RL 训练框架

## 概述

LLM 的 RL 训练（RLHF/RLAIF/RLVR）是当前 AI 工程中最复杂的训练范式之一。与标准的预训练或微调不同，RL 训练需要同时协调多个大模型、交替进行生成和训练、并在分布式环境中高效管理 GPU 资源。

传统的深度学习训练框架（如 DeepSpeed、Megatron-LM）专注于单一模型的前向/反向传播。而 RL 训练的独特需求催生了专门的 RL 训练框架。

截至 2025-2026 年，三大开源框架主导生态：**OpenRLHF**、**veRL** 和 **TRL**，各自定位不同。

---

## 核心挑战

### 1. 多模型协调

以 PPO 为例，一次完整的 RL 训练需要协调 4 个模型：

```
PPO 的 4 个模型：
┌────────────────────────────────────────────────────┐
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │  Actor    │  │  Critic  │  │ Reference│  │  Reward  │
│  │  (策略)   │  │  (价值)  │  │  (参考)   │  │  (奖励)  │
│  │  可训练   │  │  可训练  │  │  冻结     │  │  冻结    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘
│       │              │             │              │
│       ▼              ▼             ▼              ▼
│  生成回复      估计价值      计算 KL 惩罚    计算奖励
│                                                     │
└────────────────────────────────────────────────────┘
```

对于 70B 模型，4 个模型的显存需求约为：
- Actor (bf16): ~140 GB
- Critic (bf16): ~140 GB
- Reference (bf16): ~140 GB
- Reward Model (bf16): ~140 GB
- **总计**: ~560 GB + 梯度 + 优化器状态

这就是为什么 [[grpo|GRPO]] 和 REINFORCE++ 去掉 Critic 如此有价值——直接减少 25% 的模型显存。

### 2. 生成 + 训练循环

RL 训练的独特之处在于交替进行推理（生成回复）和训练（更新参数）：

```
RL 训练循环：
┌──────────────────────────────────────────────────┐
│  ┌─────────┐                                      │
│  │ 采样阶段 │ ← 使用推理优化（vLLM, KV Cache 等）   │
│  │ Actor    │                                      │
│  │ 生成回复 │ ← GPU 利用模式：计算密集型推理          │
│  └────┬────┘                                      │
│       │ rollouts                                   │
│       ▼                                            │
│  ┌─────────┐                                      │
│  │ 评估阶段 │ ← RM + Reference 前向传播              │
│  │ 计算奖励 │                                      │
│  │ 计算优势 │ ← GAE / 组内归一化                     │
│  └────┬────┘                                      │
│       │ (states, actions, rewards, advantages)     │
│       ▼                                            │
│  ┌─────────┐                                      │
│  │ 训练阶段 │ ← 使用训练优化（ZeRO, FSDP 等）        │
│  │ PPO/GRPO │                                      │
│  │ 更新参数 │ ← GPU 利用模式：梯度计算 + 优化器更新    │
│  └────┬────┘                                      │
│       │                                            │
│       └──→ 回到采样阶段（下一个 iteration）           │
└──────────────────────────────────────────────────┘
```

**关键挑战**：推理和训练的 GPU 利用模式完全不同：
- **推理**：需要 KV Cache、低延迟、高吞吐（适合 tensor parallel）
- **训练**：需要梯度存储、优化器状态（适合 ZeRO/FSDP/pipeline parallel）

### 3. GPU 显存管理

```
一个 RL 训练步骤的显存时间线：

时间 ──────────────────────────────────────────→

[  生成阶段  ]
  Actor 权重 + KV Cache          ████████████
  其他模型                                      (可卸载)

[  评估阶段  ]
  Actor 权重                     ████████
  RM 权重                        ████████
  Reference 权重                 ████████
  Rollout 数据                   ████

[  训练阶段  ]
  Actor 权重                     ████████
  Actor 梯度                     ████████
  Actor 优化器状态               ████████████████
  Critic 权重 + 梯度 + 优化器    ████████████████████████
  Rollout 数据                   ████

峰值显存可能是静态模型权重的 3-4 倍！
```

**优化策略**：
- 生成和训练阶段复用 GPU 显存（不同时需要 KV Cache 和优化器状态）
- 将冻结模型（Reference, RM）放在单独的 GPU 或卸载到 CPU
- 使用模型并行降低单 GPU 显存需求

### 4. 分布式训练

大模型的 RL 训练需要跨多机多卡的分布式策略，且不同模型可能需要不同的并行策略：

```
示例：70B Actor 的分布式 RL 训练

Node 0-3:  Actor (TP=8, DP=4) — 32 GPUs
Node 4-5:  Critic (TP=4, DP=4) — 16 GPUs
Node 6:    Reference (TP=8) — 8 GPUs  (只做推理)
Node 7:    Reward Model (TP=8) — 8 GPUs  (只做推理)

总计: 64 GPUs for 一次 PPO 训练
```

---

## OpenRLHF

### 架构概览

OpenRLHF 是首个高性能、生产就绪的开源 RLHF 框架。其核心设计理念是**分离生成和训练**，用 Ray 做分布式调度，vLLM 做高效推理，DeepSpeed ZeRO 做训练。

```
OpenRLHF 架构：
┌──────────────────────────────────────────────────────────┐
│                     Ray Cluster                           │
│                                                           │
│  ┌─────────────────────┐   ┌─────────────────────────┐   │
│  │  vLLM 推理引擎       │   │  DeepSpeed 训练引擎      │   │
│  │                     │   │                         │   │
│  │  ┌───────────────┐  │   │  ┌───────────────────┐  │   │
│  │  │ Actor (推理)   │  │   │  │ Actor (训练)       │  │   │
│  │  │ AutoTP        │  │   │  │ ZeRO-3            │  │   │
│  │  │ KV Cache      │  │   │  │ 梯度 + 优化器      │  │   │
│  │  └───────────────┘  │   │  └───────────────────┘  │   │
│  │  ┌───────────────┐  │   │  ┌───────────────────┐  │   │
│  │  │ Reference     │  │   │  │ Critic (训练)      │  │   │
│  │  │ (冻结, 推理)   │  │   │  │ ZeRO-3            │  │   │
│  │  └───────────────┘  │   │  └───────────────────┘  │   │
│  │  ┌───────────────┐  │   │                         │   │
│  │  │ Reward Model  │  │   │                         │   │
│  │  │ (冻结, 推理)   │  │   │                         │   │
│  │  └───────────────┘  │   │                         │   │
│  └─────────────────────┘   └─────────────────────────┘   │
│                                                           │
│  Ray 负责：                                                │
│  - 跨节点调度                                              │
│  - 数据传输                                                │
│  - 模型权重同步（生成 → 训练 → 生成）                         │
└──────────────────────────────────────────────────────────┘
```

### 核心特性

| 特性 | 描述 |
|------|------|
| **AutoTP** | 自动张量并行，无需手动配置 TP degree |
| **RingAttention** | 支持长上下文 RL（128K+ tokens） |
| **VLM RLHF** | v0.10+ 支持视觉语言模型的 RL |
| **异步智能体 RL** | 支持多轮 RL 与环境交互 |
| **混合引擎** | 生成用 vLLM，训练用 DeepSpeed |
| **权重同步** | Ray 高效传输更新后的模型权重到 vLLM |

### 支持的算法

- [[ppo-for-llm|PPO]] — 经典在线策略梯度
- [[grpo|GRPO]] — 无 Critic 的组相对策略优化
- REINFORCE++ — 增强版 REINFORCE
- RLOO — Leave-One-Out 基线
- DAPO — 动态采样的策略优化
- [[dpo|DPO]] / SimPO / KTO — 离线偏好优化
- 奖励模型训练 — Bradley-Terry 损失
- SFT — 监督微调

### 性能

```
14B 模型, 8K 上下文, 8xH100 基准测试:

OpenRLHF:  328.6 秒/iteration
veRL:      511.1 秒/iteration
           ──────────────────
OpenRLHF 快 ~1.55x
```

在更大规模（数百 GPU）上，OpenRLHF 的 Ray 调度优势更明显。

### 代码示例

```bash
# 启动 OpenRLHF PPO 训练
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
# 启动 OpenRLHF GRPO 训练（更简单，无需 Critic 和 RM）
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

[GitHub](https://github.com/OpenRLHF/OpenRLHF) | [论文](https://arxiv.org/abs/2405.11143)（EMNLP 2025）

---

## TRL (Transformer Reinforcement Learning)

### 定位

TRL 是 HuggingFace 的全栈后训练库，入门门槛最低。适合快速原型验证和中等规模训练。

### 架构

```
TRL 架构（简洁优先）：
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
│  │  (或可选 vLLM 用于 rollout)        │    │
│  │                                   │    │
│  │  DeepSpeed / FSDP 集成            │    │
│  │  (通过 Accelerate)                │    │
│  └───────────────────────────────────┘    │
│                                           │
│  ┌───────────────────────────────────┐    │
│  │           TRL CLI (v1.0)          │    │
│  │  trl sft | trl dpo | trl grpo    │    │
│  │  一行命令启动训练                   │    │
│  └───────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

### 支持的方法

| Trainer | 方法 | 用途 |
|---------|------|------|
| `SFTTrainer` | 监督微调 | 基础对齐 |
| `RewardTrainer` | 奖励模型训练 | Bradley-Terry 损失 |
| `PPOTrainer` | PPO | 在线策略梯度 |
| `DPOTrainer` | DPO/IPO | 离线偏好优化 |
| `KTOTrainer` | KTO | 二元反馈偏好 |
| `GRPOTrainer` | GRPO | 无 Critic 在线 RL |
| `ORPOTrainer` | ORPO | 无参考模型偏好 |
| `OnlineDPOTrainer` | Online DPO | 迭代 DPO |

### TRL v1.0 (2026 年 4 月)

重要里程碑：
- **CLI 支持**：`trl sft`, `trl dpo`, `trl grpo` 一行命令
- **OpenEnv 集成**：Meta 的 RL 环境支持（用于智能体 RL）
- **统一后训练栈**：从 SFT 到 RL 的完整流程

### 代码示例：TRL GRPO 训练

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# 1. 模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 2. 数据集（只需 prompt）
dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train")

# 3. 定义奖励函数（RLVR 风格）
def math_reward_fn(completions, ground_truths, **kwargs):
    """可验证奖励：检查数学答案是否正确"""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        answer = extract_answer(completion)
        rewards.append(1.0 if answer == gt else 0.0)
    return rewards

# 4. 配置
training_args = GRPOConfig(
    output_dir="./grpo_math",
    per_device_train_batch_size=4,
    num_generations=8,        # 每个 prompt 生成 8 个候选
    max_completion_length=2048,
    max_prompt_length=512,
    learning_rate=1e-6,
    num_train_epochs=3,
    beta=0.04,                # KL 惩罚
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    # vLLM 加速生成（可选）
    use_vllm=True,
    vllm_gpu_utilization=0.7,
)

# 5. 训练
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=[math_reward_fn],
)
trainer.train()
```

### TRL CLI 快速启动

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

# GRPO（需要自定义奖励函数脚本）
trl grpo \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --reward_funcs math_reward \
  --dataset_name AI-MO/NuminaMath-TIR \
  --output_dir ./grpo_output
```

### 简单 vs 性能权衡

| 方面 | TRL | OpenRLHF / veRL |
|------|-----|-----------------|
| 上手难度 | 极低 | 中等 |
| 代码改动 | 几行代码 | 需要配置 Ray/调度 |
| 单卡训练 | 良好支持 | 不适合 |
| 多节点大规模 | 较弱 | 强 |
| 推理速度 | 较慢（HF generate） | 快（vLLM/SGLang） |
| 适合场景 | 研究/原型/小规模 | 生产/大规模 |

[文档](https://huggingface.co/docs/trl/en/index) | [GitHub](https://github.com/huggingface/trl)

---

## veRL (Volcano Engine RL)

### 背景

veRL 是字节跳动（ByteDance）开发的 RL 训练框架。其核心论文 "HybridFlow" 发表在 EuroSys 2025。

### 核心设计

veRL 的核心创新是 **FSDP-based 混合并行**和**灵活的模型放置策略**：

```
veRL 架构：
┌──────────────────────────────────────────────────────────┐
│                    veRL HybridFlow                         │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │               Resource Manager                       │ │
│  │  (灵活分配 GPU 给不同模型和阶段)                       │ │
│  └────────────────────┬────────────────────────────────┘ │
│                       │                                   │
│       ┌───────────────┼───────────────┐                   │
│       │               │               │                   │
│       ▼               ▼               ▼                   │
│  ┌─────────┐   ┌─────────────┐  ┌─────────────┐         │
│  │ 生成阶段 │   │  评估阶段    │  │  训练阶段    │         │
│  │          │   │             │  │             │         │
│  │ vLLM /   │   │ Reference   │  │ FSDP /      │         │
│  │ SGLang /  │   │ + RM        │  │ Megatron    │         │
│  │ HF       │   │ 前向传播    │  │ 全分片训练   │         │
│  └─────────┘   └─────────────┘  └─────────────┘         │
│                                                           │
│  关键特性：                                                │
│  - 同一组 GPU 在不同阶段复用                                │
│  - 无需 Ray 调度开销                                       │
│  - FSDP 原生集成                                          │
│  - 支持专家并行（MoE 模型）                                │
└──────────────────────────────────────────────────────────┘
```

### 核心特性

| 特性 | 描述 |
|------|------|
| **HybridFlow** | 生成和训练阶段复用同一组 GPU |
| **FSDP + Megatron** | 支持两种训练后端 |
| **vLLM + SGLang** | 灵活选择推理引擎 |
| **671B 扩展** | 通过专家并行支持 MoE 超大模型 |
| **资源复用** | 不需要为每个模型分配固定 GPU |

### 支持的算法

PPO, [[grpo|GRPO]], GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO

veRL 是 **DAPO** 的官方训练基础设施。

### 关键成就

- **671B 模型训练**：通过专家并行扩展到 DeepSeek-V3 级别
- **数学 SOTA**：RL 扩展预览在 AIME 上达到 70.0 pass@1（o1 水平）
- **v0.3.0**: ~1.4x speedup over prior versions

### 代码示例

```python
# veRL 配置示例（YAML 风格）
"""
verl_config:
  algorithm: grpo
  model:
    actor: meta-llama/Llama-3.1-8B-Instruct
    ref: meta-llama/Llama-3.1-8B-Instruct
  
  training:
    backend: fsdp          # 或 megatron
    lr: 1e-6
    batch_size: 128
    ppo_epochs: 1
    
  rollout:
    engine: vllm            # 或 sglang, hf
    temperature: 1.0
    top_p: 0.95
    num_generations: 8
    max_new_tokens: 2048
    
  reward:
    type: function           # 或 model
    function: math_verifier
    
  resource:
    num_gpus: 8
    gpu_memory_utilization: 0.85
"""
```

```bash
# veRL 启动训练
python -m verl.trainer.main_ppo \
    --config config/grpo_llama8b.yaml \
    --num_gpus 8
```

### 与 OpenRLHF 的关键区别

| 维度 | OpenRLHF | veRL |
|------|----------|------|
| 调度 | Ray（分离生成和训练到不同 GPU） | 自定义（同 GPU 复用） |
| 训练后端 | DeepSpeed ZeRO-3 | FSDP / Megatron |
| 资源利用 | 需要更多 GPU（不同角色） | 更高效（复用） |
| 超大模型 | 数百 GPU | 671B（专家并行） |
| 复杂度 | 中等 | 较高 |

[GitHub](https://github.com/volcengine/verl) | [HybridFlow 论文](https://arxiv.org/abs/2409.19256)（EuroSys 2025）

---

## DeepSpeed-Chat

### 背景

DeepSpeed-Chat 是微软开发的 RLHF 训练管线，基于 DeepSpeed 生态。是较早的开源 RLHF 框架之一（2023 年发布）。

### 架构

```
DeepSpeed-Chat 架构：
┌───────────────────────────────────────────────────┐
│              DeepSpeed-Chat Pipeline               │
│                                                    │
│  Step 1: SFT ──→ Step 2: RM Training ──→ Step 3: RLHF
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │          DeepSpeed Hybrid Engine          │     │
│  │                                          │     │
│  │  训练模式：ZeRO Stage 3                   │     │
│  │     ↕ 自动切换                            │     │
│  │  推理模式：Tensor Parallel + KV Cache      │     │
│  │                                          │     │
│  │  同一模型在同一 GPU 上切换推理和训练模式     │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  ZeRO 优化：                                       │
│  - Stage 1: 优化器状态分片                          │
│  - Stage 2: + 梯度分片                             │
│  - Stage 3: + 参数分片                             │
│  - ZeRO-Offload: CPU/NVMe 卸载                     │
└───────────────────────────────────────────────────┘
```

### 核心创新：混合引擎（Hybrid Engine）

DeepSpeed-Chat 的核心创新是 Hybrid Engine——同一个模型可以在同一组 GPU 上无缝切换推理模式和训练模式：

- **推理模式**：使用 Tensor Parallel + KV Cache 加速生成
- **训练模式**：使用 ZeRO-3 进行梯度计算和参数更新
- **切换开销**：需要重新分片参数，有一定开销

### 优缺点

**优势**：
- 完整的 3 阶段 RLHF 流水线（SFT → RM → PPO）
- 深度集成 DeepSpeed ZeRO，显存效率高
- 单命令启动全流程
- 良好的文档

**劣势**：
- 推理速度不如 vLLM（没有 PagedAttention、连续批处理等）
- 模型切换有开销
- 不支持 GRPO 等新算法（社区维护活跃度低于 OpenRLHF/veRL）
- 扩展性不如 Ray-based 方案

[GitHub](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) | [博客](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)

---

## NeMo-Aligner

### 背景

NeMo-Aligner 是 NVIDIA 的对齐框架，基于 NeMo 训练平台构建。面向拥有 NVIDIA GPU 集群的大规模部署。

### 特点

```
NeMo-Aligner：
┌───────────────────────────────────────────┐
│  基于 NeMo 2.0 平台                        │
│                                            │
│  训练后端：Megatron-LM                      │
│  - Tensor Parallel                         │
│  - Pipeline Parallel                       │
│  - Expert Parallel (MoE)                   │
│  - Context Parallel                        │
│                                            │
│  支持方法：                                  │
│  - PPO (RLHF)                              │
│  - DPO                                     │
│  - SteerLM (属性可控对齐)                    │
│  - Self-Play Fine-Tuning (SPIN)            │
│  - 奖励模型训练                              │
│                                            │
│  优势：                                     │
│  - NVIDIA GPU 深度优化                       │
│  - Megatron 级别的扩展性                     │
│  - 与 NeMo 生态无缝集成                      │
│                                            │
│  劣势：                                     │
│  - 学习曲线陡峭                              │
│  - 绑定 NVIDIA 生态                          │
│  - 社区不如 OpenRLHF/TRL 活跃               │
└───────────────────────────────────────────┘
```

[GitHub](https://github.com/NVIDIA/NeMo-Aligner) | [文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-aligner/)

---

## 框架对比表

### 综合对比

| 特性 | OpenRLHF | veRL | TRL | DeepSpeed-Chat | NeMo-Aligner |
|------|:--------:|:----:|:---:|:--------------:|:------------:|
| **代码行数** | ~8.5K | ~32K | ~19K | ~5K | ~15K |
| **训练后端** | DeepSpeed ZeRO-3 | FSDP/Megatron | HF Trainer | DeepSpeed ZeRO | Megatron-LM |
| **推理引擎** | vLLM | vLLM/SGLang/HF | HF/vLLM | DeepSpeed Hybrid | — |
| **调度** | Ray | 自定义 | 单节点为主 | 单节点为主 | Slurm/K8s |
| **PPO** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **GRPO** | ✓ | ✓ | ✓ | ✗ | ✗ |
| **DPO** | ✓ | ✓ | ✓ | ✗ | ✓ |
| **REINFORCE++** | ✓ | ✓ | ✗ | ✗ | ✗ |
| **VLM RL** | ✓ (v0.10+) | ✓ | ✗ | ✗ | ✗ |
| **最大规模** | 数百 GPU | 671B 模型 | 中等 | 中等 | 大规模 |
| **上手难度** | 中等 | 较高 | **最低** | 中等 | 高 |
| **社区活跃度** | 高 | 高 | 最高 | 中低 | 中等 |
| **最适用于** | 生产 RL | 大规模研究 | 原型开发 | DeepSpeed 用户 | NVIDIA 生态 |

### 性能对比（已知基准）

| 场景 | OpenRLHF | veRL | 注释 |
|------|:--------:|:----:|------|
| 14B, 8K ctx, 8xH100 | 328.6s | 511.1s | OpenRLHF 快 1.55x |
| 大规模 MoE | 未测试 | 671B | veRL 支持更大模型 |
| 多节点扩展 | 1.22-1.68x vs alt. | ~1.4x vs prev. ver. | 不同基线 |

---

## 选择指南

```
你应该选择哪个框架？

  ┌── 你的目标是什么？
  │
  ├── 快速实验 / 学习 / 原型
  │   └── ✅ TRL
  │       - pip install trl
  │       - 几行代码就能开始
  │       - 完整文档和教程
  │
  ├── 生产级大规模 RL 训练
  │   └── ✅ OpenRLHF
  │       - Ray 集群调度
  │       - vLLM 高效推理
  │       - 经过验证的大规模部署
  │
  ├── 超大模型 (100B+) / MoE 研究
  │   └── ✅ veRL
  │       - FSDP/Megatron 混合并行
  │       - 专家并行支持
  │       - 671B 级别验证
  │
  ├── 已有 DeepSpeed 基础设施
  │   └── ⚠️ DeepSpeed-Chat
  │       - 可用但更新较慢
  │       - 考虑迁移到 OpenRLHF
  │
  └── NVIDIA 深度集成 / NeMo 生态
      └── ⚠️ NeMo-Aligner
          - Megatron 级扩展性
          - 学习曲线较陡
```

### 典型工作流

```
研究者的典型路径：
  1. TRL 验证想法 (1-8 GPU)
  2. OpenRLHF/veRL 扩展到大规模 (32-256 GPU)
  3. 自定义框架做极致优化 (可选)

工程团队的典型路径：
  1. OpenRLHF 搭建生产流水线
  2. 根据模型规模选择 veRL (超大模型) 或 OpenRLHF (通用)
  3. 持续跟进框架更新和新算法支持
```

---

## 附录：关键算法速览

### REINFORCE++

增强版 REINFORCE，整合了 PPO 的稳定化技术但无需 Critic：
- 裁剪损失（Clipped Surrogate Loss）
- Token 级 KL 惩罚
- 全局优势归一化

**定位**：比 [[grpo|GRPO]] 更稳定，比 [[ppo-for-llm|PPO]] 更快。

[arXiv:2501.03262](https://arxiv.org/html/2501.03262v5)

### RLOO (REINFORCE Leave-One-Out)

使用 k-1 个其他样本的平均奖励作为无偏基线：

```
对于 k 个样本中的第 i 个：
baseline_i = (1/(k-1)) · Σ_{j≠i} r(x, y_j)
advantage_i = r(x, y_i) - baseline_i
```

**优势**：比 PPO 少 50-70% 显存，快 2-3x。

[HuggingFace 博客](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)

### DAPO (Dynamic Advantage Policy Optimization)

动态采样策略优化，veRL 团队的工作：
- 动态采样温度
- 自适应 KL 控制
- 改进的优势估计

[GitHub (veRL)](https://github.com/volcengine/verl)

---

## 参考文献

- Hu et al. (2024) — [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143) (EMNLP 2025)
- Sheng et al. (2024) — [HybridFlow: A Flexible and Efficient RLHF Framework (veRL)](https://arxiv.org/abs/2409.19256) (EuroSys 2025)
- von Werra et al. (2020-2026) — [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- Yao et al. (2023) — [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training](https://arxiv.org/abs/2308.01320)
- NVIDIA — [NeMo-Aligner Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-aligner/)
- Yu et al. (2025) — [REINFORCE++](https://arxiv.org/html/2501.03262v5)
- Ahmadian et al. (2024) — [RLOO: Back to Basics](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)

---

## 相关页面

- [[rlhf-overview]] — 这些框架实现的 RLHF 流水线
- [[ppo-for-llm]] — PPO 算法细节
- [[grpo]] — 最流行的无 Critic RL 算法
- [[dpo]] — 离线偏好优化（多数框架支持）
- [[reward-modeling]] — 奖励模型训练（这些框架的关键组件）
- [[agentic-rl-overview]] — 多轮 RL / 智能体 RL 框架
