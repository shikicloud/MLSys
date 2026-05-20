---
title: "Distributed Training: Data/Model/Pipeline Parallelism"
category: ml-infra
tags: [distributed-training, data-parallelism, tensor-parallelism, pipeline-parallelism, zero, fsdp, mixed-precision, gradient-checkpointing, fault-tolerance]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# Distributed Training: Data/Model/Pipeline Parallelism

> [!abstract]+ TL;DR
> Modern LLMs (Llama 3.1 405B, DeepSeek-V3) need thousands of GPUs and trillions of tokens to train. Distributed training fits them on available hardware via **data parallelism** (replicate model, shard data), **model parallelism** (tensor + pipeline + expert), and **memory optimization** (ZeRO, FSDP, mixed precision, activation checkpointing). This page focuses on training-specific problems; the sister page [[parallelism-strategies-deep-dive]] details the sharding mechanics shared with inference.

---

## 1. Overview

### 1.1 Why Distributed Training

Modern large language models long ago outgrew what a single GPU can hold:

| Model | Params | Training GPUs | Training Tokens | Training Time |
|-------|--------|---------------|-----------------|---------------|
| GPT-3 (2020) | 175B | ~1,000 V100 | 300B | ~34 days |
| LLaMA 2 70B (2023) | 70B | 2,048 A100 | 2T | ~25 days |
| LLaMA 3.1 405B (2024) | 405B | 16,384 H100 | 15.6T | ~54 days |
| DeepSeek-V3 (2024) | 671B (37B active) | 2,048 H800 | 14.8T | ~55 days |
| GPT-4 (rumored, 2023) | ~1.8T MoE | ~25,000 A100 | ~13T | ~90 days |

**Core challenges**:

1. **Memory bottleneck** — a 70B FP16 model needs ~140 GB just for weights; a single H100 only has 80 GB
2. **Compute bottleneck** — 1T tokens × 70B params ≈ 4.2×10²² FLOPs; a single H100 (990 TFLOPS BF16) would take ~490 days
3. **Communication bottleneck** — cross-GPU / cross-node data transfer becomes the new performance bottleneck

### 1.2 Training Memory Breakdown

Understanding memory consumption is the basis of distributed-training design. For a model with $\Phi$ parameters trained with AdamW + mixed precision, per-GPU memory is:

```
Training memory = model params + gradients + optimizer state + activations

Model params (mixed precision):
  - FP16/BF16 params: 2Φ bytes
  - FP32 master weights: 4Φ bytes

Gradients: 2Φ bytes (FP16/BF16)

Optimizer state (AdamW):
  - FP32 first moment (m): 4Φ bytes
  - FP32 second moment (v): 4Φ bytes

Total (excl. activations): 2Φ + 4Φ + 2Φ + 4Φ + 4Φ = 16Φ bytes
```

**Example**: 70B-param model → 16 × 70B × 1 byte = **1,120 GB** (excluding activations), far exceeding a single GPU's memory.

Activation memory depends on sequence length, batch size, and model hidden dimension:

```
Activations (per layer per sample, Transformer) ≈ s × b × h × (34 + 5·a·s/h)
  - s: sequence length
  - b: micro-batch size
  - h: hidden dim
  - a: number of attention heads
```

---

## 2. Data Parallelism (DP) Training Details

> For the basic principles and communication patterns of DP, see [[parallelism-strategies-deep-dive#2. DP — Data Parallelism]]. This section focuses on gradient synchronization in training.

### 2.1 DistributedDataParallel (DDP) Workflow

PyTorch DDP is the most basic data-parallel implementation:

```
   GPU 0              GPU 1              GPU 2              GPU 3
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│ Model    │      │ Model    │      │ Model    │      │ Model    │
│ (full replica)│ │ (full replica)│ │ (full replica)│ │ (full replica)│
├──────────┤      ├──────────┤      ├──────────┤      ├──────────┤
│ Data     │      │ Data     │      │ Data     │      │ Data     │
│ Shard 0  │      │ Shard 1  │      │ Shard 2  │      │ Shard 3  │
├──────────┤      ├──────────┤      ├──────────┤      ├──────────┤
│ Forward  │      │ Forward  │      │ Forward  │      │ Forward  │
│ Backward │      │ Backward │      │ Backward │      │ Backward │
├──────────┤      ├──────────┤      ├──────────┤      ├──────────┤
│ Grad_0   │      │ Grad_1   │      │ Grad_2   │      │ Grad_3   │
└────┬─────┘      └────┬─────┘      └────┬─────┘      └────┬─────┘
     │                 │                 │                 │
     └────────────┬────┴────────────┬────┘                 │
                  │   AllReduce     │                       │
                  │  (sum then /N)  ├───────────────────────┘
                  │                 │
     ┌────────────┴────┬────────────┴────┬─────────────────┐
     ▼                 ▼                 ▼                 ▼
   Avg Grad         Avg Grad         Avg Grad         Avg Grad
   (identical)      (identical)      (identical)      (identical)
     │                 │                 │                 │
   Update            Update           Update           Update
   Params            Params           Params           Params
```

**Key properties**:
- Each GPU holds a full model replica
- Data is sharded across workers; each worker processes a different micro-batch
- After backward, gradients are synchronized via AllReduce
- Post-update parameters are identical across all workers

### 2.2 Ring AllReduce Algorithm

Ring AllReduce is the core algorithm of DDP gradient synchronization. It arranges N GPUs into a logical ring:

```
Step 1: ReduceScatter — each GPU splits its gradient into N chunks,
        and via N-1 "pass + accumulate" rounds, each GPU ends up
        with the global sum of 1/N of the gradient

GPU 0 ──→ GPU 1 ──→ GPU 2 ──→ GPU 3
  ↑                                │
  └────────────────────────────────┘

Initial state (4 GPU, gradient split into 4 chunks):
GPU 0: [A0, B0, C0, D0]
GPU 1: [A1, B1, C1, D1]
GPU 2: [A2, B2, C2, D2]
GPU 3: [A3, B3, C3, D3]

Round 1: each GPU sends a chunk to the right, receives and accumulates
GPU 0: [A0,      B0,      C0,      D0+D3  ]
GPU 1: [A1+A0,   B1,      C1,      D1     ]
GPU 2: [A2,      B2+B1,   C2,      D2     ]
GPU 3: [A3,      B3,      C3+C2,   D3     ]

Round 2:
GPU 0: [A0,      B0,      C0+C3+C2, D0+D3  ]
GPU 1: [A1+A0,   B1,      C1,       D1+D0+D3]
GPU 2: [A2+A1+A0,B2+B1,   C2,       D2      ]
GPU 3: [A3,      B3+B2+B1,C3+C2,    D3      ]

Round 3 (ReduceScatter done):
GPU 0: [A0,       B0+B3+B2+B1, C0+C3+C2, D0+D3  ]
GPU 1: [A1+A0,    B1,          C1+C0+C3+C2, D1+D0+D3]
GPU 2: [A2+A1+A0, B2+B1,       C2,       D2+D1+D0+D3]
GPU 3: [A3+A2+A1+A0, B3+B2+B1, C3+C2,    D3      ]
         ↑ global sum  ↑ global sum  ↑ global sum  ↑ global sum
         (on GPU 3)    (on GPU 0)    (on GPU 1)    (on GPU 2)

Step 2: AllGather — another N-1 rounds broadcast each segment's
        global sum to all GPUs

Final: every GPU has the full global gradient sum
```

**Communication complexity**:

| Algorithm | Comm. per GPU | Steps | Notes |
|-----------|---------------|-------|-------|
| Naive AllReduce | $(N-1) \cdot D$ | $N-1$ | every worker sends to root |
| Ring AllReduce | $2 \cdot \frac{N-1}{N} \cdot D$ | $2(N-1)$ | almost independent of GPU count |
| Tree AllReduce | $2 \cdot D \cdot \log N$ | $2\log N$ | fewer steps but lower bandwidth utilization |

where $D$ is the gradient size and $N$ is the GPU count. Ring AllReduce's strength is that its **communication volume is essentially independent of GPU count** (with large $N$, $(N-1)/N \approx 1$), so it scales well to large clusters.

### 2.3 Gradient Accumulation

When GPU memory cannot fit the desired micro-batch size, gradient accumulation produces an equivalently large effective batch:

```python
# Gradient accumulation example
accumulation_steps = 4  # effective batch = micro_batch × accumulation_steps × world_size
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # note: scale loss
    loss.backward()  # gradients accumulated into .grad

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()   # update only when accumulation finishes
        optimizer.zero_grad()
```

**Effective batch size**:
```
effective_batch_size = micro_batch_size × accumulation_steps × world_size (DP degree)
```

**Caveats**:
- Loss must be divided by `accumulation_steps` (otherwise gradient scale is wrong)
- DDP triggers AllReduce on every `backward()` by default; during accumulation you should only sync on the last step
- In PyTorch, use the `model.no_sync()` context to skip sync on intermediate steps:

```python
for i, batch in enumerate(dataloader):
    context = model.no_sync() if (i + 1) % accum_steps != 0 else nullcontext()
    with context:
        loss = model(batch) / accum_steps
        loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2.4 Synchronous vs Asynchronous Training

| Feature | Synchronous (BSP) | Asynchronous (ASP) |
|---------|-------------------|---------------------|
| Gradient consistency | all workers see the same gradient | uses stale gradients |
| Sync barrier | yes (AllReduce) | no |
| Convergence | equivalent to single-card (ideal) | may diverge or converge slowly |
| Efficiency | bounded by slowest worker | no straggler problem |
| Use cases | **mainstream** (essentially all LLM training) | large-scale recommendation/ads |

**Status**: large-model training is almost exclusively **synchronous**. Async training suffers from stale-gradient convergence instability, and the straggler problem is mild on homogeneous GPU clusters. Async training is mainly used for embedding updates in industrial recommendation systems.

---

## 3. ZeRO Optimization

ZeRO (Zero Redundancy Optimizer) from the DeepSpeed team is built on the idea of **eliminating memory redundancy in data parallelism**. In standard DDP every GPU holds full model parameters, gradients, and optimizer state — a massive waste.

### 3.1 Three Stages

```
                  Standard DDP (N=4 GPU)
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Params  │    │Params  │    │Params  │    │Params  ││
  │ │(full)  │    │(full)  │    │(full)  │    │(full)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grads   │    │Grads   │    │Grads   │    │Grads   ││
  │ │(full)  │    │(full)  │    │(full)  │    │(full)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt State│   │Opt State│   │Opt State│   │Opt State│
  │ │(full)  │    │(full)  │    │(full)  │    │(full)  ││
  │ │ 12Φ    │    │ 12Φ    │    │ 12Φ    │    │ 12Φ    ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ per GPU: 16Φ   per GPU: 16Φ   per GPU: 16Φ  per GPU: 16Φ│
  │                  total redundancy: 4 × 16Φ = 64Φ      │
  └──────────────────────────────────────────────────────┘

  ZeRO-1: partition optimizer state only
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Params  │    │Params  │    │Params  │    │Params  ││
  │ │(full)  │    │(full)  │    │(full)  │    │(full)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grads   │    │Grads   │    │Grads   │    │Grads   ││
  │ │(full)  │    │(full)  │    │(full)  │    │(full)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 ││
  │ │ 3Φ     │    │ 3Φ     │    │ 3Φ     │    │ 3Φ     ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ per GPU: 7Φ   (vs DDP 16Φ → 56% saved)                │
  └──────────────────────────────────────────────────────┘

  ZeRO-2: partition optimizer state + gradients
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Params  │    │Params  │    │Params  │    │Params  ││
  │ │(full)  │    │(full)  │    │(full)  │    │(full)  ││
  │ │ 2Φ     │    │ 2Φ     │    │ 2Φ     │    │ 2Φ     ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grad 1/4│    │Grad 1/4│    │Grad 1/4│    │Grad 1/4││
  │ │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 ││
  │ │ 3Φ     │    │ 3Φ     │    │ 3Φ     │    │ 3Φ     ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ per GPU: 5.5Φ (vs DDP 16Φ → 66% saved)                │
  └──────────────────────────────────────────────────────┘

  ZeRO-3: partition everything (params + grads + opt state) = FSDP
  ┌──────────────────────────────────────────────────────┐
  │  GPU 0          GPU 1          GPU 2          GPU 3  │
  │ ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐│
  │ │Par 1/4 │    │Par 1/4 │    │Par 1/4 │    │Par 1/4 ││
  │ │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Grad 1/4│    │Grad 1/4│    │Grad 1/4│    │Grad 1/4││
  │ │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   │    │ 0.5Φ   ││
  │ ├────────┤    ├────────┤    ├────────┤    ├────────┤│
  │ │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 │    │Opt 1/4 ││
  │ │ 3Φ     │    │ 3Φ     │    │ 3Φ     │    │ 3Φ     ││
  │ └────────┘    └────────┘    └────────┘    └────────┘│
  │ per GPU: 4Φ   (vs DDP 16Φ → 75% saved)                │
  └──────────────────────────────────────────────────────┘
```

### 3.2 Per-stage Memory Analysis

Let model parameters be $\Phi$, DP degree be $N_d$, using AdamW + mixed precision (FP16 params + FP32 master weights/optimizer state):

| Component | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----------|-----|--------|--------|--------|
| FP16 params | $2\Phi$ | $2\Phi$ | $2\Phi$ | $2\Phi / N_d$ |
| FP16 gradients | $2\Phi$ | $2\Phi$ | $2\Phi / N_d$ | $2\Phi / N_d$ |
| FP32 master weights | $4\Phi$ | $4\Phi / N_d$ | $4\Phi / N_d$ | $4\Phi / N_d$ |
| FP32 first moment | $4\Phi$ | $4\Phi / N_d$ | $4\Phi / N_d$ | $4\Phi / N_d$ |
| FP32 second moment | $4\Phi$ | $4\Phi / N_d$ | $4\Phi / N_d$ | $4\Phi / N_d$ |
| **Total** | **$16\Phi$** | **$4\Phi + 12\Phi/N_d$** | **$2\Phi + 14\Phi/N_d$** | **$16\Phi/N_d$** |

**Numerical example** (70B params, $N_d = 64$):

| Stage | Per-GPU memory (excl. activations) | Savings |
|-------|------------------------------------|---------|
| DDP | 1,120 GB | — |
| ZeRO-1 | 280 + 13.1 = **293 GB** | 73.8% |
| ZeRO-2 | 140 + 15.3 = **155 GB** | 86.2% |
| ZeRO-3 | **17.5 GB** | 98.4% |

### 3.3 Per-stage Communication Cost

| Stage | Forward | Backward | After update | Total comm. |
|-------|---------|----------|--------------|-------------|
| DDP | none | AllReduce (gradients) | none | $2\Phi$ |
| ZeRO-1 | none | ReduceScatter (gradients) | AllGather (updated params) | $2\Phi$ |
| ZeRO-2 | none | ReduceScatter (gradients) | AllGather (updated params) | $2\Phi$ |
| ZeRO-3 | AllGather (params) | AllGather + ReduceScatter | none | $3\Phi$ |

**Key takeaways**:
- ZeRO-1/2 communicate the same volume as DDP ($2\Phi$) but dramatically reduce memory
- ZeRO-3 adds a forward AllGather, increasing comm. by 50% ($3\Phi$), but saves the most memory
- In practice ZeRO-3 / FSDP hides the extra communication via prefetch and overlap

### 3.4 ZeRO++ Enhancements

ZeRO++ (2023) further optimizes cross-node communication on top of ZeRO-3:

1. **Quantized weight communication (qwZ)** — INT8 quantization on AllGather, halving cross-node volume
2. **Hierarchical partition (hpZ)** — keep a full weight replica intra-node, shard only inter-node
3. **Quantized gradient communication (qgZ)** — INT4 quantization on ReduceScatter

Effect: cross-node communication reduced **4×**, training speedup 2.16× to 2.49×.

---

## 4. Mixed Precision Training

### 4.1 Why Lower Precision

| Dtype | Bits | Exponent | Mantissa | Dynamic range | Precision | Usage |
|-------|------|----------|----------|---------------|-----------|-------|
| FP32 | 32 | 8 | 23 | ~1e±38 | high | master weights, optimizer state |
| FP16 | 16 | 5 | 10 | ~6.5e4 | medium | forward/backward compute |
| BF16 | 16 | 8 | 7 | ~1e±38 | lower | forward/backward compute (recommended) |
| FP8 (E4M3) | 8 | 4 | 3 | ~448 | low | forward (experimental) |
| FP8 (E5M2) | 8 | 5 | 2 | ~57344 | very low | backward (experimental) |

**Benefits**:
- Memory halved (FP16/BF16 vs FP32)
- 2-3× compute speedup (Tensor Core)
- Communication bandwidth halved

### 4.2 Mixed Precision Training Loop

```
┌─────────────────────────────────────────────────────┐
│             Mixed-precision training loop            │
│                                                     │
│  ┌───────────────┐                                  │
│  │ FP32 master   │ ←───── optimizer updates in FP32 │
│  │ (Master Copy) │                                  │
│  └───────┬───────┘                                  │
│          │ Cast to BF16/FP16                         │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ BF16 work copy │ ──→ forward (BF16 Tensor Core)  │
│  └───────────────┘                                  │
│          │                                          │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ BF16 Loss     │                                  │
│  └───────┬───────┘                                  │
│          │ (Loss Scaling required for FP16)         │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ BF16 grads    │ ←── backward (BF16 Tensor Core)  │
│  └───────┬───────┘                                  │
│          │ AllReduce / ReduceScatter                 │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ FP32 grads    │ ←── Cast to FP32                 │
│  └───────┬───────┘                                  │
│          │                                          │
│          ▼                                          │
│  ┌───────────────┐                                  │
│  │ FP32 optim    │ ──→ Adam(m, v, master_weights)   │
│  │ state update  │                                  │
│  └───────────────┘                                  │
└─────────────────────────────────────────────────────┘
```

### 4.3 FP16 Loss Scaling

FP16 has a narrow dynamic range (smallest positive ~6×10⁻⁸); small gradients underflow to zero. Loss scaling preserves small-gradient precision by scaling up the loss:

```python
# Dynamic loss-scaling pseudocode
scaler = torch.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        output = model(batch)
        loss = criterion(output)

    # 1. Scale up the loss → scaled-up gradients (avoid underflow)
    scaler.scale(loss).backward()

    # 2. Unscale gradients → check for inf/nan
    scaler.unscale_(optimizer)

    # 3. Gradient clipping (after unscale)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4. Update params (skipped if grads contain inf/nan)
    scaler.step(optimizer)

    # 5. Dynamically adjust the scale factor
    scaler.update()
```

**Dynamic scaling policy**:
- Initial scale = 2¹⁶ or larger
- N consecutive steps without overflow → scale × 2 (increase)
- Overflow → scale / 2 (decrease), skip this step's update

### 4.4 BF16 Advantages

BF16 is the dominant choice for LLM training (natively supported since A100/H100):

- **Same dynamic range as FP32** (8 exponent bits) → no loss scaling needed
- Simpler training (no GradScaler)
- Slightly lower precision than FP16 (7 vs 10 mantissa bits), but the impact on LLMs is small in practice

```python
# BF16 mixed precision (simpler, no loss scaling needed)
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output)

loss.backward()
optimizer.step()  # step directly, no scaler
```

### 4.5 FP8 Training (frontier)

FP8 training uses the H100/H200 FP8 Tensor Core for 2× theoretical throughput over BF16:

- **Forward** uses E4M3 (higher precision, range ±448)
- **Backward** uses E5M2 (wider range, range ±57344)
- Requires per-tensor or per-block dynamic scaling
- TorchTitan + Float8 + FSDP2 has demonstrated **50% throughput improvement**

**Current status** (early 2026):
- Megatron-LM and TorchTitan support FP8 forward + BF16 backward
- Full FP8 training (forward + backward) is still experimental
- DeepSeek-V3 used FP8 training — the first large-scale success

---

## 5. 3D Parallelism

> For TP, PP, EP sharding principles and communication patterns, see [[parallelism-strategies-deep-dive]].
> For PP scheduling (GPipe, 1F1B, Interleaved 1F1B, Zero Bubble), see [[model-parallelism]].
> This section focuses on how to **combine** multiple parallel dimensions to train giant models.

### 5.1 Why Combine Parallelism Strategies

Single-strategy parallelism has limits:

| Strategy | Memory efficiency | Compute efficiency | Comm. needs | Scaling bottleneck |
|----------|-------------------|--------------------|-------------|---------------------|
| DP | low (full replica) | high | AllReduce (gradients) | model must fit a single GPU |
| TP | high (sharded weights) | high | AllReduce (2× per layer) | needs high-bandwidth NVLink |
| PP | high (layer-grouped) | medium (bubble) | point-to-point | bubble overhead |
| ZeRO-3/FSDP | high (full shard) | medium | AllGather + ReduceScatter | large-scale comm. volume |

**3D parallelism** = TP × PP × DP, getting the best of each:

```
                    Cluster topology mapping
  ┌─────────────────────────────────────────────────┐
  │                  All GPUs                        │
  │                                                 │
  │  DP replica 0          DP replica 1             │
  │  ┌──────────────────┐  ┌──────────────────┐     │
  │  │ PP stage 0       │  │ PP stage 0       │     │
  │  │ ┌──┬──┬──┬──┐   │  │ ┌──┬──┬──┬──┐   │     │
  │  │ │G0│G1│G2│G3│←TP│  │ │G8│G9│GA│GB│←TP│     │
  │  │ └──┴──┴──┴──┘   │  │ └──┴──┴──┴──┘   │     │
  │  │ PP stage 1       │  │ PP stage 1       │     │
  │  │ ┌──┬──┬──┬──┐   │  │ ┌──┬──┬──┬──┐   │     │
  │  │ │G4│G5│G6│G7│←TP│  │ │GC│GD│GE│GF│←TP│     │
  │  │ └──┴──┴──┴──┘   │  │ └──┴──┴──┴──┘   │     │
  │  └──────────────────┘  └──────────────────┘     │
  │                                                 │
  │  TP = 4 (intra-node NVLink)                     │
  │  PP = 2 (inter-node or intra-node)              │
  │  DP = 2 (cross-replica AllReduce)               │
  │  Total = 4 × 2 × 2 = 16 GPU                    │
  └─────────────────────────────────────────────────┘
```

### 5.2 Hardware Topology Mapping Principles

**Core principle**: comm.-heavy parallel dimensions → high-bandwidth links

| Parallel dim | Comm. pattern | Frequency | Recommended topology |
|--------------|---------------|-----------|----------------------|
| **TP** | AllReduce (2× per layer) | extreme | **intra-node NVLink** (900 GB/s) |
| **PP** | point-to-point (cross-layer activations) | medium | intra- or inter-node |
| **DP** | AllReduce (1× per step) | low | **inter-node InfiniBand** |
| **EP** | AllToAll (per MoE layer) | mid-high | choose intra/inter-node by load |
| **CP/SP** | AllToAll or Ring | medium | intra-node NVLink |

**Typical layout** (8 GPU/node):
```
TP = 8 (all 8 GPUs of one node connected via NVLink)
PP = N_nodes / DP_replicas (cross-node pipeline)
DP = Total_GPUs / (TP × PP) (remaining dimension for DP)
```

### 5.3 Large-Model Configuration Examples

**LLaMA 3.1 405B**:
```
Params: 405B (dense)
GPU: 16,384 × H100 80GB
TP = 8  (8 GPUs per node via NVLink)
PP = 16 (16 pipeline stages)
DP = 128 (128 data-parallel replicas)
Total: 8 × 16 × 128 = 16,384 GPU
Training data: 15.6T tokens
```

**DeepSeek-V3**:
```
Params: 671B (MoE, 37B active per token)
GPU: 2,048 × H800 80GB
TP = 1  (no TP! MoE models have few active params)
PP = 16
DP = 128 (with ZeRO-1)
EP = 64 (experts distributed across 64 GPUs)
Special: FP8 training + DualPipe scheduling
```

> Note: DeepSeek-V3 avoids TP because the MoE architecture activates only 37B params per token (single GPU can hold them) and avoids TP's communication overhead.

### 5.4 4D / 5D Parallelism

- **4D parallelism** = TP + PP + DP + **CP/SP** (sequence/context parallelism)
- **5D parallelism** = TP + PP + DP + CP + **EP** (MoE-specific)

In 2025 NVIDIA proposed the **MoE Parallel Folding** framework, fully decoupling parallelism in dense and MoE layers:
```
Dense layer: N_total = TP × CP × DP × PP
MoE layer:   N_total = ETP × EP × EDP × PP
```

See [[parallelism-strategies-deep-dive#MoE Parallel Folding]].

---

## 6. Gradient Checkpointing (Activation Checkpointing)

### 6.1 Basic Principle

Standard backward needs all intermediate activations, so memory grows linearly with layer count. Gradient checkpointing (a.k.a. activation recomputation) trades **compute for memory**:

```
Standard backward:
  Forward: save activations of all layers a1, a2, ..., aL
  Backward: use saved activations to compute gradients
  Memory: O(L) — all-layer activations

Gradient checkpointing:
  Forward: save only "checkpoint"-layer activations
  Backward: recompute activations from the nearest checkpoint
  Memory: O(√L) — only checkpoint-layer activations
  Cost: ~33% extra forward compute
```

```
Standard (save all activations):
Layer:  1    2    3    4    5    6    7    8
Save:  [a1] [a2] [a3] [a4] [a5] [a6] [a7] [a8]
                            ↑ large memory footprint

Gradient checkpointing (every 2 layers):
Layer:  1    2    3    4    5    6    7    8
Save:  [a1]  ×   [a3]  ×   [a5]  ×   [a7]  ×
              ↑         ↑         ↑         ↑
         recomp from  recomp from  recomp from  recomp from
            a1            a3            a5            a7
```

### 6.2 Full vs Selective Checkpointing

**Full checkpointing**:
- Each Transformer layer saves only its input; all in-layer activations are recomputed
- Maximum memory savings (activations drop from $O(L \cdot s \cdot b \cdot h)$ to $O(L \cdot s \cdot b \cdot h_{input})$)
- Compute overhead ~33%

**Selective checkpointing**:
- Only checkpoint memory-heavy ops (e.g., the QKV intermediate matrix in self-attention)
- Keep cheap-to-recompute activations (e.g., LayerNorm output)
- Megatron-LM strategy: keep non-attention activations, recompute attention only

```python
# Selective checkpointing (Megatron-LM style)
def transformer_layer(x):
    # Attention part — checkpoint (QK^T matrix is memory-heavy)
    attn_out = checkpoint(self_attention, x)

    # FFN part — no checkpoint (relatively cheap)
    ffn_out = feed_forward(attn_out + x)

    return ffn_out + attn_out + x
```

### 6.3 Memory Savings Analysis

For an L-layer Transformer with per-layer activation $A$ bytes:

| Strategy | Activation memory | Extra compute | Use case |
|----------|-------------------|---------------|----------|
| None | $L \cdot A$ | 0 | memory ample |
| Full checkpoint | $L \cdot A_{input}$ ≈ $L \cdot A / k$ | ~33% | memory tight |
| Selective | $L \cdot A_{selected}$ | ~10-15% | compromise |
| √L checkpoint | $\sqrt{L} \cdot A$ | ~33% | theoretical optimum |

**Empirical numbers** (70B model, seq=4096, micro-batch=1):
- No checkpoint: ~30 GB activations
- Full checkpoint: ~3 GB activations
- Selective: ~8 GB activations

---

## 7. Communication Optimization

### 7.1 DDP Bucketing

PyTorch DDP does not fire AllReduce per parameter; it bundles parameter gradients into **buckets**:

```
Parameter grads:  p1.grad  p2.grad  p3.grad  p4.grad  p5.grad  p6.grad
                  └───── bucket 0 ─────┘  └───── bucket 1 ─────┘
                            │                        │
                        AllReduce                AllReduce
```

**Benefits**:
- Fewer comm. calls (larger messages have higher throughput than small ones)
- Default bucket size = 25 MB
- Buckets fill from the last layer (gradients computed back-to-front), enabling comm.-compute overlap

### 7.2 Overlapping Communication with Computation

```
Timeline →

No overlap:
[=== Forward ===][=== Backward ===][== AllReduce ==][Update]
                                                     ↑ must wait until all done

With overlap (DDP bucketing):
[=== Forward ===][=== Backward ==================]
                 [bucket N AR][bucket N-1 AR]...  [Update]
                  ↑ start comm as soon as last-layer grads are ready

FSDP forward AllGather overlap:
[AG layer1][Fwd layer1][AG layer2][Fwd layer2]...
 ↑ prefetch next-layer params

FSDP backward overlap:
[AG+Bwd layer L][RS layer L][AG+Bwd layer L-1][RS layer L-1]...
 ↑ AllGather current-layer + Backward
                  ↑ ReduceScatter gradients
```

### 7.3 Gradient Compression

Techniques for cutting comm. volume:

| Technique | Compression | Accuracy impact | Use case |
|-----------|-------------|------------------|----------|
| FP16 gradient comm. | 2× | almost none | general |
| INT8 quantization (ZeRO++) | 4× | small | inter-node |
| Top-K sparsification | 10-100× | requires error feedback | research |
| Power-of-two quantization | 8× | small | research |

### 7.4 NCCL Tuning

NCCL (NVIDIA Collective Communications Library) is the standard for GPU-cluster communication:

**Key knobs**:
```bash
# Environment variable tuning
NCCL_IB_DISABLE=0                 # enable InfiniBand
NCCL_IB_GID_INDEX=3              # IB GID index
NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA level
NCCL_SOCKET_IFNAME=eth0          # network interface
NCCL_P2P_LEVEL=NVL               # P2P transport level
NCCL_ALGO=Ring                   # algorithm (Ring/Tree)
NCCL_PROTO=Simple                # protocol (Simple/LL/LL128)
NCCL_MIN_NCHANNELS=4             # minimum channel count
NCCL_CROSS_NIC=1                 # cross-NIC communication
```

**Tuning tips**:
- Small messages use Tree AllReduce (low latency); large messages use Ring AllReduce (high bandwidth)
- Enable GPU Direct RDMA (GDR) to bypass CPU staging
- Use multi-rail InfiniBand to boost cross-node bandwidth
- SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) does in-network reduction on switches

---

## 8. Fault Tolerance

### 8.1 Why Fault Tolerance Is Critical

Failure probability at scale:

```
Single-GPU annual failure rate ≈ 3-5%
16,384-GPU cluster:
  - average ~1.5-2.5 GPU failures per day
  - LLaMA 3.1 405B trained ~54 days
  - expected failures ≈ 80-135

Without fault tolerance → training cannot finish
```

### 8.2 Checkpointing Strategies

**Synchronous checkpointing** (traditional):
```
[Training step 1000] → [pause training] → [save full checkpoint] → [resume]
                                          ↑ all GPUs sync write to storage
                                          takes: tens of minutes (large models)
```

**Asynchronous checkpointing** (modern):
```
[Training step 1000] → [async background save] → [continue with step 1001]
                        ↑ does not block training
                        GPU → host pinned memory (fast) → disk/S3 (slow, background)
```

**Checkpoint frequency tradeoff**:
| Frequency | Pros | Cons |
|-----------|------|------|
| High (every 100 steps) | less lost work | high I/O overhead, high storage cost |
| Low (every 1000 steps) | low overhead | more lost progress per failure |
| Adaptive | balanced | complex to implement |

### 8.3 Distributed Checkpointing

Distributed checkpointing in Megatron-LM and PyTorch:

```python
# PyTorch Distributed Checkpoint (DCP)
import torch.distributed.checkpoint as dcp

# Save — each rank writes only its own shard
dcp.save(
    state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_writer=dcp.FileSystemWriter("/checkpoint/step_1000"),
)

# Load — supports resharding (recover with different parallelism)
dcp.load(
    state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_reader=dcp.FileSystemReader("/checkpoint/step_1000"),
)
```

**Distributed-checkpoint advantages**:
- Each rank writes its shard in parallel → big speedup
- Resharding: save with TP=8, load with TP=4
- Incremental checkpoints: save only changed parts

### 8.4 Elastic Training

Elastic training lets the worker count change during training (shrink after a failure, grow when a new node joins):

```
Normal training: 128 GPU
  ↓ GPU 17 fails
detect failure (seconds)
  ↓
remaining 127 GPUs reorganize
  ↓
recover from latest checkpoint (1 fewer DP replica)
  ↓
continue training: 127 GPU
  ↓ replacement node comes online
grow back to 128 GPU
  ↓
recover from checkpoint (1 more DP replica)
```

**PyTorch Elastic (torchrun)**:
```bash
# Elastic launch: min 120 GPU, max 128 GPU
torchrun \
    --nnodes=15:16 \        # elastic node-count range
    --nproc_per_node=8 \
    --rdzv_backend=c10d \   # rendezvous backend
    --rdzv_endpoint=master:29400 \
    --max_restarts=3 \      # max restart count
    train.py
```

### 8.5 Failure Detection and Recovery

**Common failure types**:

| Failure type | Detection | Recovery |
|--------------|-----------|----------|
| GPU hardware failure | NCCL timeout / ECC error | restart node + restore checkpoint |
| GPU OOM | CUDA OOM exception | reduce batch / find memory leak |
| Network failure | NCCL timeout | rebuild comm. group |
| Silent data corruption (SDC) | loss anomaly detection | roll back to an earlier checkpoint |
| Process crash | heartbeat timeout | torchrun auto-restart |

**Silent data corruption (SDC)** is the most dangerous failure type — the GPU produces a wrong result without raising an error. At cluster scale SDC is non-negligible:
- The LLaMA 3 team reported multiple SDC incidents during training
- Detection: periodically check for loss spikes, compare results across replicas
- Mitigation: use ECC memory, do periodic validation-loss checks

---

## 9. Code Examples

### 9.1 PyTorch DDP Basic Setup

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # model
    model = MyLargeModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # data
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # ensure different shuffle per epoch
        for batch in dataloader:
            batch = batch.to(rank)
            loss = model(batch)
            loss.backward()       # DDP triggers AllReduce automatically
            optimizer.step()
            optimizer.zero_grad()

    cleanup()

# launch
# torchrun --nproc_per_node=8 train.py
```

### 9.2 FSDP Configuration

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaDecoderLayer

# mixed-precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # compute in BF16
    reduce_dtype=torch.bfloat16,    # AllReduce in BF16
    buffer_dtype=torch.bfloat16,
)

# auto-wrap policy: each Transformer layer is one FSDP unit
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer}
)

# FSDP wrapping
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
    # ShardingStrategy.SHARD_GRAD_OP → ZeRO-2
    # ShardingStrategy.NO_SHARD → DDP
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=False),     # optional CPU offload
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,  # cap concurrent AllGathers
)
```

### 9.3 Full Mixed Precision + Gradient Accumulation Example

```python
import torch
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# config
use_bf16 = True  # H100/A100 recommended BF16
dtype = torch.bfloat16 if use_bf16 else torch.float16
accum_steps = 4
max_grad_norm = 1.0

# no GradScaler needed for BF16
scaler = None if use_bf16 else GradScaler()

optimizer.zero_grad()
for step, batch in enumerate(dataloader):
    # gradient accumulation: do not sync on intermediate steps
    is_accumulating = (step + 1) % accum_steps != 0
    sync_context = model.no_sync() if is_accumulating else nullcontext()

    with sync_context:
        with autocast(device_type='cuda', dtype=dtype):
            loss = model(batch) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if not is_accumulating:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        optimizer.zero_grad()
```

### 9.4 Activation Checkpointing Setup

```python
from torch.utils.checkpoint import checkpoint

class TransformerLayerWithCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # wrap with checkpoint: do not save intermediates at forward
        return checkpoint(self.layer, x, use_reentrant=False)

# FSDP + activation checkpointing
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)

# apply checkpointing to each Transformer layer
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
        m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
)
```

---

## 10. Large-Scale Training Case Studies

### 10.1 LLaMA 3.1 405B (Meta, 2024)

| Item | Value |
|------|-------|
| Params | 405B (dense) |
| GPU | 16,384 × H100 80GB |
| Interconnect | 400 Gbps RoCE (inter-node), NVLink (intra-node) |
| Parallelism | TP=8, PP=16, DP=128 |
| Sequence length | 8K → 128K (staged) |
| Training data | 15.6T tokens |
| Precision | BF16 mixed precision |
| Fault tolerance | avg one failure every ~12 hours, auto-detect + checkpoint restore |
| Key techniques | selective activation checkpointing, async checkpointing, NCCL tuning |
| Training efficiency | ~38-43% MFU (Model FLOPs Utilization) |

**Lessons**:
- At 16K-GPU scale, network failures outnumber GPU failures
- Silent data corruption (SDC) needs extra monitoring
- Staged training (short seq first, then long seq) improves efficiency

### 10.2 DeepSeek-V3 (DeepSeek, 2024)

| Item | Value |
|------|-------|
| Params | 671B (MoE, 256 experts, top-8, 37B active) |
| GPU | 2,048 × H800 80GB |
| Interconnect | IB (inter-node), NVLink (intra-node) |
| Parallelism | TP=1, PP=16, DP=128, EP=64 |
| Training data | 14.8T tokens |
| Precision | **FP8 mixed precision** (first large-scale use) |
| Training cost | ~$5.5M (vs LLaMA 3.1 405B ~$100M+) |
| Key techniques | DualPipe, FP8 training, Multi-Token Prediction, auxiliary-loss-free load balancing |

**DualPipe pipeline schedule**: DeepSeek-V3's new PP schedule advances forward and backward in both directions simultaneously, significantly reducing bubble time.

**Why 10-20× cheaper**:
1. MoE architecture → only 37B params compute per token
2. FP8 training → 2× compute throughput
3. No TP → saves significant intra-node communication
4. Efficient auxiliary-loss design → high expert utilization

### 10.3 Training Scale vs Efficiency Trends

```
MFU (Model FLOPs Utilization) trends:
  Single-node, small model:  50-60%
  Hundreds of GPUs:          40-50%
  Thousands of GPUs:         35-45%
  16K+ GPUs:                 30-43%

Reasons for declining GPU utilization:
  1. Higher comm. fraction
  2. PP bubbles
  3. Downtime from failure recovery
  4. Load imbalance (MoE)
  5. Checkpoint I/O
```

---

## 11. References

- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (SC'20)
- Rajbhandari et al., *"ZeRO-Infinity: Breaking the GPU Memory Wall"* (SC'21)
- Wang et al., *"ZeRO++: Extremely Efficient Collective Communication"* (2023)
- Micikevicius et al., *"Mixed Precision Training"* (ICLR 2018)
- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* (SC'21)
- Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (VLDB 2023)
- Liang et al., *"TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training"* (ICLR 2025)
- Chen et al., *"Training Deep Nets with Sublinear Memory Cost"* (arXiv 2016) — activation checkpointing
- Meta, *"Llama 3.1 Model Card and Technical Report"* (2024)
- DeepSeek, *"DeepSeek-V3 Technical Report"* (2024)

---

## 12. Related Pages

- [[parallelism-strategies-deep-dive]] — DP / TP / EP / EDP / ETP sharding principles (training + inference)
- [[model-parallelism]] — parallelism strategies and PP scheduling in inference
- [[training-frameworks]] — Megatron-LM, DeepSpeed, FSDP framework implementations
- [[gpu-cluster-management]] — GPU cluster management and hardware topology
- [[checkpointing]] — detailed discussion of checkpoint save/restore
- [[rl-training-frameworks]] — distributed strategies for reinforcement learning training
