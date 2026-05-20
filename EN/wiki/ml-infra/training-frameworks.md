---
title: "Training Frameworks: Megatron-LM, DeepSpeed, FSDP"
category: ml-infra
tags: [megatron-lm, deepspeed, fsdp, torchtitan, training, frameworks, megatron-core, nemo]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Training Frameworks: Megatron-LM, DeepSpeed, FSDP

> [!abstract]+ TL;DR
> The mission of an LLM training framework is to package the parallel strategies and optimizations from [[distributed-training]] into a usable engineering system. The ecosystem has converged on three mainstream options: **Megatron-LM/Core** (NVIDIA, highest throughput, TP+PP+SP+CP+EP+DP), **DeepSpeed** (Microsoft, memory efficiency, ZeRO 1-3+++ with PP/EP/Ulysses-SP), and **FSDP2/TorchTitan** (Meta/PyTorch, PyTorch-native, FSDP+TP+PP+SP). Production-scale ultra-large training often combines Megatron's TP/PP with DeepSpeed's ZeRO via Megatron-DeepSpeed.

---

## 1. Overview

The core mission of an LLM training framework is to **package the parallel strategies and optimizations from [[distributed-training]] into a usable engineering system**. From the "every team builds its own wheel" era of GPT-3 in 2020 to 2025-2026, the ecosystem has gradually converged on a few mainstream options:

```
                LLM training framework ecosystem (2025-2026)
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Megatron-LM │  │  DeepSpeed   │  │ PyTorch FSDP2  │  │
│  │ /Megatron-  │  │  (Microsoft) │  │ + TorchTitan   │  │
│  │  Core       │  │              │  │ (Meta/PyTorch) │  │
│  │ (NVIDIA)    │  │              │  │                │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
│         │                │                   │           │
│   max throughput   memory eff/usability  PyTorch-native  │
│   TP+PP+SP+EP      ZeRO 1-3 + offload    FSDP+TP+PP     │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Higher-level wrappers / other frameworks            │ │
│  │  NeMo (NVIDIA)  |  Composer (Databricks/MosaicML)   │ │
│  │  Nanotron (HF)  |  Colossal-AI  |  Fairscale (Meta) │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Core considerations when choosing a framework**:
1. **Training throughput** — tokens processed per unit time
2. **Memory efficiency** — largest model that fits on given hardware
3. **Parallelism support** — which parallel dimensions and combinations are supported
4. **Usability** — learning curve, debugging difficulty, community support
5. **Composability** — compatibility with the rest of the PyTorch ecosystem

---

## 2. Megatron-LM / Megatron-Core

### 2.1 Architecture Overview

Megatron is NVIDIA's large-model training framework, organized in two layers:

```
┌──────────────────────────────────────────────────┐
│                   NeMo Framework                  │
│  (high-level API: config-driven, Hydra config,    │
│   training recipes)                                │
├──────────────────────────────────────────────────┤
│                  Megatron-Core                     │
│  (core library: parallel primitives, distributed   │
│   strategies, fused kernels)                       │
├──────────────────────────────────────────────────┤
│              Megatron-LM (original repo)           │
│  (training scripts, model defs, data pipeline)     │
├──────────────────────────────────────────────────┤
│  PyTorch  |  NCCL  |  CUDA  |  Transformer Engine │
└──────────────────────────────────────────────────┘
```

**Megatron-LM vs Megatron-Core**:

| Aspect | Megatron-LM | Megatron-Core |
|------|-------------|---------------|
| Positioning | Full training stack (model + data + train) | Reusable parallel primitives library |
| Repo | `NVIDIA/Megatron-LM` | `NVIDIA/Megatron-LM` (submodule) |
| Model defs | Built-in GPT/BERT/T5 | Provides composable modules |
| Use case | Direct training of NVIDIA-supported models | Integrated by other frameworks (e.g., NeMo) |
| API stability | Lower, frequently refactored | Higher, designed as a library |

### 2.2 Core Parallelism Support

Megatron-LM supports the most comprehensive combination of parallel dimensions:

```
Megatron-LM 5D parallelism:

  Dense layers: TP × CP × DP × PP
  MoE layers:   ETP × EP × EDP × PP

  ┌──────────────────────────────────────────┐
  │              Parallel dimensions          │
  │                                          │
  │  TP  — tensor parallelism (Megatron-style)│
  │        column-parallel: Y = GeLU(X · A_c) │
  │        row-parallel: Y = X · A_row        │
  │        2× AllReduce per layer             │
  │                                          │
  │  SP  — sequence parallelism (with TP)     │
  │        LayerNorm and Dropout sharded on   │
  │        sequence dim, saves activation mem │
  │                                          │
  │  PP  — pipeline parallelism               │
  │        supports: GPipe, 1F1B,             │
  │        Interleaved 1F1B (virtual stages)  │
  │                                          │
  │  CP  — context parallelism                │
  │        long sequences: Ring Attention     │
  │                                          │
  │  EP  — expert parallelism (MoE models)    │
  │                                          │
  │  DP  — data parallelism (can stack ZeRO-1)│
  └──────────────────────────────────────────┘
```

### 2.3 Key Technical Features

**1. Fused Kernels**

Megatron makes heavy use of CUDA fused kernels to cut memory traffic and kernel-launch overhead:

```
Standard PyTorch:
  x = LayerNorm(x)          # kernel 1: r/w x
  qkv = Linear(x)           # kernel 2: r/w x
  q, k, v = split(qkv)      # kernel 3: r/w qkv
  attn = softmax(q @ k^T)   # kernel 4-5
  out = attn @ v             # kernel 6

Megatron fused:
  x = FusedLayerNorm(x)                    # fused internal LN ops
  out = FlashAttention(x, fused_qkv=True)  # fused QKV + Attention
  → fewer kernels, less memory traffic, faster
```

**2. Transformer Engine + FP8**

NVIDIA Transformer Engine provides FP8-precision matmul, deeply integrated with Megatron:

```python
# Using Transformer Engine inside Megatron
import transformer_engine.pytorch as te

class TETransformerLayer(te.TransformerLayer):
    """FP8 layer using Transformer Engine"""
    def __init__(self, config):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            num_attention_heads=config.num_heads,
            fuse_qkv_params=True,           # fuse QKV
            fp8_format=te.recipe.Format.HYBRID,  # E4M3 fwd, E5M2 bwd
        )
```

**3. Selective Activation Recomputation**

```
Full recompute: nothing saved, everything recomputed on backward
  → most memory savings, but ~33% compute overhead

Selective recompute (Megatron default):
  Keep:    LayerNorm output, Linear output (cheap to compute, heavy r/w)
  Recompute: QK^T in attention, Softmax (large memory, easy to recompute)
  → ~70% memory saved, only ~5-10% compute overhead
```

**4. Distributed Optimizer**

Megatron's distributed optimizer shards optimizer states across the DP group (similar to ZeRO-1), reducing memory.

### 2.4 Configuration Example

```bash
# Megatron-LM training a 70B model (8 nodes × 8 GPU = 64 GPU)
python pretrain_gpt.py \
    # Model config
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    \
    # Parallel config
    --tensor-model-parallel-size 8 \      # TP=8 (intra-node)
    --pipeline-model-parallel-size 4 \     # PP=4
    --num-layers-per-virtual-pipeline-stage 2 \  # Interleaved PP
    # DP = 64 / (8 × 4) = 2
    \
    # Precision
    --bf16 \
    --use-flash-attn \
    \
    # Optimization
    --use-distributed-optimizer \          # ZeRO-1 style distributed optimizer
    --recompute-activations \              # activation checkpointing
    --recompute-granularity selective \    # selective recompute
    --overlap-grad-reduce \               # overlap grad comm with compute
    --overlap-param-gather \              # overlap param gather
    \
    # Training
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --train-iters 500000 \
    \
    # Checkpoints
    --save /checkpoint/70b \
    --save-interval 1000 \
    --async-save \                        # async checkpoint
    \
    # Data
    --data-path /data/tokenized_dataset \
    --tokenizer-type GPT2BPETokenizer
```

### 2.5 NeMo Framework Integration

NeMo is NVIDIA's higher-level framework on top of Megatron-Core, providing a config-driven training experience:

```python
# NeMo 2.0 training config (using Hydra)
# nemo/collections/llm/recipes/llama3_70b.py
from nemo.collections.llm import PreTrainingRecipe

recipe = PreTrainingRecipe(
    model="llama3_70b",
    trainer=dict(
        num_nodes=8,
        devices=8,
        precision="bf16-mixed",
        max_steps=500000,
    ),
    data=dict(
        paths=["/data/tokenized"],
        seq_length=8192,
        global_batch_size=1024,
        micro_batch_size=1,
    ),
    parallelism=dict(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=1,  # set for MoE
    ),
    optim=dict(
        name="distributed_fused_adam",  # Megatron distributed optimizer
        lr=1.5e-4,
        weight_decay=0.1,
    ),
)
```

### 2.6 Pros and Cons

| Aspect | Verdict |
|------|------|
| **Throughput** | Industry-best; the benchmark for other frameworks |
| **Parallelism support** | Most comprehensive: TP+PP+SP+CP+EP+DP |
| **Fused kernels** | Many hand-written CUDA kernels; deeply integrated with Transformer Engine/FP8 |
| **Learning curve** | Steep: huge codebase, sparse docs, frequent refactors |
| **Portability** | Tightly bound to NVIDIA ecosystem (NCCL, NVLink, Transformer Engine) |
| **Community** | Relatively closed, maintained mostly by NVIDIA internally |
| **Debugging** | Hard: multi-dimensional parallelism crosses, unfriendly error messages |

---

## 3. DeepSpeed

### 3.1 Architecture Overview

DeepSpeed is Microsoft's deep-learning optimization library, with a core focus on **breaking the memory wall**:

```
┌──────────────────────────────────────────────────┐
│                 DeepSpeed architecture            │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Training API                                │  │
│  │  deepspeed.initialize(model, optimizer, ...) │  │
│  └──────────────────────┬──────────────────────┘  │
│                         │                         │
│  ┌──────────┬───────────┼───────────┬──────────┐  │
│  │ ZeRO     │ Pipeline  │ MoE       │ Sequence │  │
│  │ 1/2/3    │ Parallel  │ Support   │ Parallel │  │
│  │ ++/Offload│          │ DeepSpeed │ Ulysses  │  │
│  │ Infinity │          │ -MoE      │          │  │
│  └──────────┴───────────┴───────────┴──────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Low-level optimizations                     │  │
│  │  Fused Optimizers | Activation Checkpointing │  │
│  │  Sparse Attention | Communication Optim      │  │
│  │  ZenFlow (Offload) | Quantization            │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 3.2 The ZeRO Family

ZeRO is DeepSpeed's central contribution (see [[distributed-training#3. ZeRO Optimization]] for details):

| Variant | Partitioned content | Comm volume | Notes |
|------|---------|--------|------|
| ZeRO-1 | Optimizer states | $2\Phi$ | Easiest, almost no perf loss |
| ZeRO-2 | + Gradients | $2\Phi$ | More memory savings |
| ZeRO-3 | + Parameters | $3\Phi$ | Max savings, 50% more comm |
| ZeRO-Offload | ZeRO-2 + CPU offload | Same as ZeRO-2 + PCIe | Train large models on a single GPU |
| ZeRO-Infinity | ZeRO-3 + CPU/NVMe offload | Same as ZeRO-3 + PCIe/NVMe | Extreme memory regimes |
| ZeRO++ | ZeRO-3 + quantized comm | ~$0.75\Phi$ | Cross-node comm cut by 4× |

**How ZeRO-Offload / ZeRO-Infinity works**:

```
┌──────────────────────────────────────────────┐
│            ZeRO-Infinity three-tier storage   │
│                                              │
│  GPU memory (80GB):                          │
│  ┌──────────────────────────────────────┐    │
│  │ Parameter shard for current compute (FP16)│  │
│  │ Current activations                   │    │
│  └──────────────┬───────────────────────┘    │
│                 │ PCIe 4.0: ~32 GB/s         │
│  CPU memory (1TB+):                          │
│  ┌──────────────┴───────────────────────┐    │
│  │ Optimizer states (FP32 master + m,v)  │    │
│  │ Parameter shards not currently needed │    │
│  └──────────────┬───────────────────────┘    │
│                 │ NVMe SSD: ~3-7 GB/s        │
│  NVMe SSD (multi-TB):                        │
│  ┌──────────────┴───────────────────────┐    │
│  │ Overflow of all parameter shards      │    │
│  │ Overflow of optimizer states          │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

**ZenFlow (Aug 2025)** — DeepSpeed's latest offload optimization:
- Traditional offload pauses compute during GPU↔CPU transfers
- ZenFlow implements "stall-free" offload: compute and transfer fully overlap
- Pipelined via pinned memory + async streams

### 3.3 DeepSpeed-Chat (RLHF Training)

DeepSpeed-Chat provides a complete RLHF training pipeline:

```
┌──────────────────────────────────────────────┐
│           DeepSpeed-Chat RLHF pipeline        │
│                                              │
│  Step 1: SFT (supervised fine-tuning)         │
│  ┌──────────────────────────────────────┐    │
│  │ Base Model + Instruction Data → SFT   │    │
│  │ Distributed training via ZeRO-3       │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  Step 2: Reward Model training                │
│  ┌──────────────────────────────────────┐    │
│  │ SFT Model → Reward Model              │    │
│  │ Uses human preference data (chosen/rej)│   │
│  └──────────────────────────────────────┘    │
│                                              │
│  Step 3: PPO training                         │
│  ┌──────────────────────────────────────┐    │
│  │ 4 models trained concurrently:        │    │
│  │   Actor (SFT) | Critic (RM)           │    │
│  │   Ref Model   | Reward Model          │    │
│  │ Hybrid Engine: train+inference hybrid │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

**Hybrid Engine** is DeepSpeed-Chat's core innovation:
- During PPO, Actor and Ref need to do inference (generate responses)
- Hybrid Engine dynamically switches between training (ZeRO) and inference (TP) modes
- In inference mode, kernel fusion and tensor parallelism are applied automatically
- Switching to inference mode for generation → 3-4× speedup

### 3.4 DeepSpeed-MoE

DeepSpeed-MoE supports training MoE models:

```python
# DeepSpeed MoE layer configuration
import deepspeed
from deepspeed.moe.layer import MoE

moe_layer = MoE(
    hidden_size=4096,
    expert=FeedForward(4096, 11008),  # single expert FFN
    num_experts=64,
    ep_size=8,                # expert parallel degree
    use_residual=True,        # residual MoE
    k=2,                      # Top-K routing
    capacity_factor=1.25,     # capacity factor
    eval_capacity_factor=2.0,
    min_capacity=4,
    use_rts=True,             # random token selection
    use_tutel=True,           # Tutel-accelerated AllToAll
)
```

### 3.5 Configuration Example

DeepSpeed uses a JSON config file:

```json
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1.5e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1.5e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 500000
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "bf16": {
        "enabled": true
    },

    "gradient_clipping": 1.0,

    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false
    },

    "wall_clock_breakdown": false,
    "steps_per_print": 100
}
```

```python
# DeepSpeed training script
import deepspeed

model = MyLargeModel()

# DeepSpeed init — handles distributed setup, optimizer, mixed precision
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)

for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)       # handles loss scaling, grad sync automatically
    model_engine.step()               # handles optimizer step automatically

    if step % save_interval == 0:
        model_engine.save_checkpoint("/checkpoint", tag=f"step_{step}")
```

### 3.6 DeepSpeed Inference Optimizations

DeepSpeed-Inference is not the focus of this page (see [[vllm]] and other inference frameworks), but worth mentioning:
- Automatic TP partitioning
- Kernel fusion (QKV fusion, bias add + residual + LayerNorm)
- Quantized inference (INT8, INT4)
- Deep integration with Hugging Face `transformers`

### 3.7 Pros and Cons

| Aspect | Verdict |
|------|------|
| **Memory efficiency** | Industry-best: full ZeRO family + Offload + Infinity |
| **Usability** | JSON-config-driven, integrates well with HuggingFace |
| **Parallelism support** | ZeRO + PP + EP + SP (Ulysses). TP weaker than Megatron |
| **Throughput** | Slightly below Megatron-LM (especially large-scale TP) |
| **RLHF** | DeepSpeed-Chat provides a full pipeline |
| **Community** | Active, good docs, but bug fixes sometimes slow |
| **Composability** | Deeply integrated with HuggingFace Trainer |
| **Debugging** | ZeRO-3 debugging is hard (parameters are sharded, hard to inspect) |

---

## 4. PyTorch FSDP / FSDP2

### 4.1 FSDP Basics

FSDP (Fully Sharded Data Parallel) is PyTorch's native ZeRO-3 implementation:

```
FSDP workflow (2-GPU example):

         GPU 0                    GPU 1
     ┌──────────┐            ┌──────────┐
     │ Param 1/2│            │ Param 2/2│  ← sharded parameter storage
     └────┬─────┘            └────┬─────┘
          │                       │
     AllGather (gather full param)   AllGather
          │                       │
     ┌────┴─────┐            ┌────┴─────┐
     │ Full     │            │ Full     │
     │ Param    │            │ Param    │  ← temporarily holds full params
     ├──────────┤            ├──────────┤
     │ Forward  │            │ Forward  │
     │ (data 0) │            │ (data 1) │
     ├──────────┤            ├──────────┤
     │ Discard  │            │ Discard  │  ← release after forward
     │ full param│            │ full param│
     └────┬─────┘            └────┬─────┘
          │                       │
     AllGather (gather again)     AllGather
          │                       │
     ┌────┴─────┐            ┌────┴─────┐
     │ Backward │            │ Backward │
     │ (data 0) │            │ (data 1) │
     ├──────────┤            ├──────────┤
     │ Discard  │            │ Discard  │
     │ full param│            │ full param│
     └────┬─────┘            └────┬─────┘
          │                       │
     ReduceScatter (sync sharded grads)
          │                       │
     ┌────┴─────┐            ┌────┴─────┐
     │ Grad 1/2 │            │ Grad 2/2 │  ← keep only own grad shard
     │ Update   │            │ Update   │
     │ Param 1/2│            │ Param 2/2│  ← update only own param shard
     └──────────┘            └──────────┘
```

### 4.2 FSDP vs DeepSpeed ZeRO-3

The two implement the same sharding strategy in essence (both are ZeRO-3), but with different engineering decisions:

| Aspect | PyTorch FSDP | DeepSpeed ZeRO-3 |
|------|-------------|------------------|
| **Implementation** | `FlatParameter` (FSDP1) / DTensor (FSDP2) | Custom hooks + partitioning |
| **Parameter grouping** | By FSDP wrapping unit | By parameter group |
| **Communication** | Native PyTorch c10d | DeepSpeed's own comm layer |
| **CPU Offload** | Supported (basic) | More mature (ZeRO-Offload/Infinity) |
| **PyTorch integration** | Native, torch.compile friendly | External library, partly incompatible with compile |
| **Mixed precision** | `MixedPrecision` policy | JSON config |
| **TP composition** | FSDP2 + TP composable (DeviceMesh) | Needs Megatron-DeepSpeed |
| **Debugging** | Native PyTorch tooling | DeepSpeed profiler |

### 4.3 FSDP2 Improvements

FSDP2 is a major rewrite of FSDP in PyTorch 2.x:

```
FSDP1 vs FSDP2:

FSDP1:
  - FlatParameter: all params concatenated into one big tensor
  - Issues: shape changes, can't compose with TP, special-case checkpoints
  - All params sharded as a single whole

FSDP2:
  - DTensor (Distributed Tensor): each param managed independently
  - per-parameter sharding
  - Multi-dim parallel topology defined via DeviceMesh
  - Freely composable with TP/PP/SP
  - Communication-free checkpoints
```

**Key improvements**:

| Improvement | FSDP1 | FSDP2 |
|------|-------|-------|
| Parameter representation | FlatParameter (concatenated) | DTensor (independent) |
| Checkpoints | Need resharding | Communication-free |
| TP composition | Not supported | Native composable |
| torch.compile | Partial support | Full support |
| Memory | Baseline | **7% lower** |
| Flexibility | Limited | Per-parameter policy |

### 4.4 TorchTitan

TorchTitan is the Meta/PyTorch team's "one-stop" LLM pretraining solution (ICLR 2025):

```
┌──────────────────────────────────────────────────┐
│                  TorchTitan                       │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  4D Parallelism                             │  │
│  │  FSDP2 × TP × PP × SP/CP                   │  │
│  │  Flexible configuration via DeviceMesh      │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  Key components                             │  │
│  │  - Float8 training (with Transformer Engine)│  │
│  │  - torch.compile speedup                    │  │
│  │  - Distributed checkpointing (DCP)          │  │
│  │  - Elastic training (torchrun)              │  │
│  │  - Mixed precision (BF16/FP8)               │  │
│  │  - Selective activation checkpointing       │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Performance:                                    │
│  - 128-GPU 65.08% speedup (vs naive baseline)   │
│  - Float8 + FSDP2: extra 50% throughput          │
│  - SimpleFSDP: 28.5% mem cut, 68.7% throughput   │
└──────────────────────────────────────────────────┘
```

**DeviceMesh multi-dim parallel config**:

```python
from torch.distributed.device_mesh import init_device_mesh

# 4D parallelism: (DP, PP, TP, SP)
# 64 GPU = 2 DP × 4 PP × 8 TP
mesh = init_device_mesh(
    "cuda",
    (2, 4, 8),   # (DP, PP, TP)
    mesh_dim_names=("dp", "pp", "tp"),
)

# ProcessGroup for each dim is created automatically
dp_mesh = mesh["dp"]    # DP comm group
pp_mesh = mesh["pp"]    # PP comm group
tp_mesh = mesh["tp"]    # TP comm group
```

### 4.5 SimpleFSDP (Compiler-Driven)

SimpleFSDP is the PyTorch team's next-gen exploration: implement FSDP automatically via the compiler:

```
Traditional FSDP:
  User manually specifies wrapping policy → framework inserts AllGather/ReduceScatter

SimpleFSDP:
  User just tags the model → torch.compile analyzes the graph →
  Compiler decides optimal comm schedule → AllGather/ReduceScatter inserted automatically

Advantages:
  - 28.5% memory reduction (compiler optimizes memory layout)
  - 68.7% throughput gain (better comm scheduling)
  - Simpler user API
```

### 4.6 Configuration Example

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# ============ FSDP1 config ============
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    ),
    device_id=torch.cuda.current_device(),
    use_orig_params=True,  # required when using torch.compile
)

# Optional: torch.compile
model = torch.compile(model)
```

```python
# ============ FSDP2 + TP (TorchTitan-style) ============
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# Step 1: tensor parallelism
for layer in model.layers:
    parallelize_module(
        layer.self_attn,
        tp_mesh,
        {
            "q_proj": ColwiseParallel(),
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "o_proj": RowwiseParallel(),
        },
    )
    parallelize_module(
        layer.mlp,
        tp_mesh,
        {
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
    )

# Step 2: FSDP2 sharding (on top of TP)
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)  # per-layer sharding
fully_shard(model, mesh=dp_mesh)      # top-level sharding
```

### 4.7 Pros and Cons

| Aspect | Verdict |
|------|------|
| **PyTorch-native** | Biggest advantage: no external deps, perfect integration with compile/DCP/DTensor |
| **Composability** | FSDP2 + TP + PP composed freely (DeviceMesh) |
| **torch.compile** | Full support, additional speedups possible |
| **Throughput** | Close to Megatron (TorchTitan gap is 5-10%) |
| **Memory** | ZeRO-3 level; SimpleFSDP optimizes further |
| **TP implementation** | Less mature than Megatron (no SP, CP still in development) |
| **MoE** | EP support is newer, less mature than Megatron/DeepSpeed |
| **Docs** | Official PyTorch docs + TorchTitan tutorial |
| **Future** | PyTorch's official roadmap, actively developed |

---

## 5. Framework Comparison Table

### 5.1 Detailed Feature Comparison

| Feature | Megatron-LM/Core | DeepSpeed | FSDP2/TorchTitan |
|------|------------------|-----------|------------------|
| **Maintainer** | NVIDIA | Microsoft | Meta / PyTorch |
| **First released** | 2019 | 2020 | 2022 (FSDP1), 2024 (FSDP2) |
| **Core language** | Python + CUDA | Python + C++ | Python (with C++ backend) |
| | | | |
| **--- Parallelism ---** | | | |
| Data parallelism | DP + ZeRO-1 | ZeRO 1/2/3/++/Offload/Infinity | FSDP (= ZeRO-3) |
| Tensor parallelism | Best (Megatron-style) | Basic support | DTensor-based (improving) |
| Pipeline parallelism | 1F1B, Interleaved 1F1B | PP + ZeRO | Schedule-based PP |
| Sequence/context parallelism | SP + CP | Ulysses SP | In development |
| Expert parallelism | EP + ETP + EDP | DeepSpeed-MoE | Basic support |
| | | | |
| **--- Performance ---** | | | |
| Max throughput | Highest (baseline) | ~85-95% of Megatron | ~90-95% of Megatron |
| FP8 support | Transformer Engine | Experimental | Float8 + compile |
| Fused kernels | Most (hand-written CUDA) | Many | Auto-fused via compile |
| Flash Attention | Integrated | Integrated | Integrated |
| | | | |
| **--- Engineering ---** | | | |
| Usability | Hard | Medium | Medium |
| HuggingFace integration | Requires adaptation | Trainer native support | Trainer support (improving) |
| torch.compile | Partial | Partial | Full support |
| Checkpoints | Custom format | Custom format | DCP (native) |
| Elastic training | Manual | Limited | torchrun |
| | | | |
| **--- Ecosystem ---** | | | |
| Community activity | Medium | High | High (PyTorch official) |
| Doc quality | Medium | Medium-high | Medium (improving fast) |
| Model support | GPT/LLaMA/MoE | Almost all HF models | General |
| Higher-level wrappers | NeMo | HF Trainer, DeepSpeed-Chat | TorchTitan |

### 5.2 Performance Benchmark (reference)

Training a LLaMA-like 70B model, 64× H100 80GB NVLink:

| Framework | Config | Throughput (tokens/s/GPU) | MFU | Relative |
|------|------|----------------------|-----|---------|
| Megatron-LM | TP=8, PP=4, DP=2 | ~3,800 | ~42% | 100% (baseline) |
| TorchTitan | FSDP2+TP=8, PP=4, DP=2 | ~3,500 | ~39% | ~92% |
| DeepSpeed | ZeRO-3 + PP=4 | ~3,200 | ~36% | ~84% |
| FSDP1 | FULL_SHARD | ~2,900 | ~32% | ~76% |

> **Note**: numbers above are approximate references; actual performance depends on configuration, hardware interconnect, batch size, sequence length, etc. Different benchmarks may give different conclusions.

---

## 6. Other Frameworks

### 6.1 Colossal-AI

Open-source framework developed by HPC-AI Tech:

| Aspect | Description |
|------|------|
| **Core features** | Gemini (ZeRO-like) + TP + PP + SP |
| **Pros** | Simple API, automatic parallelism (alpha), good HuggingFace integration |
| **Cons** | Smaller community, less large-scale validation |
| **Use case** | Rapid experimentation, medium-scale training |

```python
# Minimal Colossal-AI example
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin

plugin = GeminiPlugin(
    precision='bf16',
    initial_scale=2**16,
    max_norm=1.0,
)
booster = Booster(plugin=plugin)
model, optimizer, _, dataloader, _ = booster.boost(
    model, optimizer, dataloader=dataloader
)
```

### 6.2 Composer / MosaicML (Databricks)

Training framework from Databricks (formerly MosaicML):

| Aspect | Description |
|------|------|
| **Core features** | Algorithmic accelerations (MixUp, Label Smoothing, Stochastic Depth, etc.) |
| **Parallelism** | Via FSDP and DeepSpeed integration |
| **Pros** | Standardized training recipes, open-source Llama/MPT training scripts |
| **Cons** | Mainly serves the Databricks ecosystem |
| **Use case** | Rapid experimentation, medium-scale pretraining |
| **Notable models** | DBRX (132B MoE), MPT series |

### 6.3 Nanotron (HuggingFace)

HuggingFace's internal pretraining framework:

| Aspect | Description |
|------|------|
| **Core features** | TP + PP + DP, 3D parallelism |
| **Pros** | Clean code, integrated with HuggingFace ecosystem |
| **Cons** | Less complete than Megatron, small community |
| **Use case** | Pretraining within the HuggingFace ecosystem |
| **Notable models** | SmolLM series training |

### 6.4 Fairscale (Meta)

Meta's distributed-training library, the predecessor to FSDP:

| Aspect | Description |
|------|------|
| **Core features** | FSDP prototype, OSS (Optimizer State Sharding), Pipeline |
| **Status** | Absorbed into PyTorch FSDP, maintenance mode |
| **Historical role** | Pioneer of PyTorch-native ZeRO |

### 6.5 Megatron-DeepSpeed

The hybrid Megatron-LM + DeepSpeed approach (formerly widely used):

```
Megatron provides:  TP + PP + data processing
DeepSpeed provides: ZeRO optimizer + offload + mixed-precision management

Used for: BigScience BLOOM (176B), several open-source large models
Status: usage declining as Megatron-Core and FSDP2 mature
```

---

## 7. Selection Guide

### 7.1 Decision Tree

```
                      Choose a training framework
                          │
              ┌───────────┴───────────┐
              │                       │
         Model > 100B?           Model < 100B?
              │                       │
       ┌──────┴──────┐          ┌─────┴─────┐
       │             │          │           │
   Max throughput?  Quick start?  Tight mem? Quick experiment?
       │             │           │           │
       │        TorchTitan   DeepSpeed   Composer
       │        /FSDP2       ZeRO-3      or
   Megatron-LM              + Offload   FSDP + HF
   + NeMo                               Trainer
       │
       │
   Have NVIDIA expert support?
   ┌──────┴──────┐
   │             │
  Yes           No
   │             │
  NeMo +       TorchTitan
  Megatron     (easier to debug)
```

### 7.2 Recommendations by Scenario

| Scenario | Recommended | Rationale |
|------|---------|------|
| **Large-scale pretraining (>100B)** | Megatron-LM + NeMo | Highest throughput, most mature 5D parallelism |
| **Medium-scale pretraining (7-70B)** | TorchTitan / FSDP2 | PyTorch-native, debuggable, close to Megatron perf |
| **Memory-constrained (few GPUs)** | DeepSpeed ZeRO-3 + Offload | ZeRO-Infinity trains large models on few GPUs |
| **RLHF / alignment** | DeepSpeed-Chat or [[rl-training-frameworks]] | Full RLHF pipeline |
| **Fine-tuning HuggingFace models** | DeepSpeed + HF Trainer | JSON config, one-line enable |
| **MoE models** | Megatron-LM | EP + ETP + EDP most complete |
| **Academic research / prototypes** | FSDP2 or Composer | Fast onboarding, readable code |
| **Production (NVIDIA infrastructure)** | NeMo + Megatron-Core | Deep NVIDIA-hardware optimization |

### 7.3 Combined Usage

Frameworks are not mutually exclusive — common combinations:

```
1. Megatron-LM (TP+PP) + DeepSpeed (ZeRO) = Megatron-DeepSpeed
   → Classic combo, used to train BLOOM

2. FSDP2 (DP sharding) + PyTorch TP (tensor parallelism) = TorchTitan 4D
   → PyTorch-native full stack

3. DeepSpeed (training) + vLLM (inference) + DeepSpeed-Chat (RLHF)
   → End-to-end pipeline

4. NeMo (training wrapper) + Megatron-Core (lower layer) + TensorRT-LLM (inference)
   → NVIDIA full stack
```

---

## 8. Code Example: Three-Framework Comparison

### 8.1 Minimal Training Loop

**PyTorch DDP (baseline)**:
```python
model = DDP(model.cuda(), device_ids=[local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**DeepSpeed**:
```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, config="ds_config.json"
)
for batch in dataloader:
    loss = model_engine(batch.cuda())
    model_engine.backward(loss)
    model_engine.step()
```

**FSDP**:
```python
model = FSDP(model.cuda(), sharding_strategy=ShardingStrategy.FULL_SHARD)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Megatron-LM** (more complex, not a simple loop):
```python
# Megatron has its own training loop; users define:
def forward_step(data_iterator, model):
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda()
    labels = batch['labels'].cuda()
    output = model(tokens)
    loss = cross_entropy(output, labels)
    return loss, {'lm_loss': loss}

# Framework handles: parallel setup, optimizer, grad sync, PP scheduling
pretrain(
    train_valid_test_datasets_provider=get_datasets,
    model_provider=get_model,
    forward_step_func=forward_step,
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
)
```

### 8.2 Checkpoint Saving Comparison

```python
# --- DeepSpeed ---
model_engine.save_checkpoint("/ckpt", tag="step_1000")
# Handles ZeRO sharding automatically, saves full model + optimizer state

# --- FSDP + DCP ---
import torch.distributed.checkpoint as dcp
dcp.save(
    {"model": model.state_dict(), "optim": FSDP.optim_state_dict(model, optimizer)},
    storage_writer=dcp.FileSystemWriter("/ckpt/step_1000"),
)
# Parallel writes, supports resharding

# --- Megatron-LM ---
# Built into the training loop:
# --save /ckpt --save-interval 1000 --async-save
```

---

## 9. References

- Shoeybi et al., *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"* (arXiv 2019)
- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* (SC 2021)
- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (SC 2020)
- Rajbhandari et al., *"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning"* (SC 2021)
- Yao et al., *"DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales"* (2023)
- Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (VLDB 2023)
- Liang et al., *"TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training"* (ICLR 2025)
- NVIDIA, *"Megatron-Core Documentation"* (2024-2025)
- Wang et al., *"ZeRO++: Extremely Efficient Collective Communication for Giant Model Training"* (2023)
- Bian et al., *"Colossal-AI: A Unified Deep Learning System for Big Model Era"* (ICPP 2021)
- MosaicML, *"Composer: A Library for Training Neural Networks Better, Faster, and Cheaper"* (2022)

---

## 10. Related Pages

- [[distributed-training]] — parallel strategies and optimizations for distributed training
- [[parallelism-strategies-deep-dive]] — DP / TP / EP / EDP / ETP sharding details
- [[model-parallelism]] — parallel strategies in inference
- [[checkpointing]] — checkpoint save and restore
- [[rl-training-frameworks]] — RL-specific training frameworks (OpenRLHF, veRL, etc.)
- [[gpu-cluster-management]] — GPU cluster management
- [[ray-ecosystem]] — Ray distributed computing ecosystem
