---
title: "LLM Parallelism Strategies Complete Guide: DP / TP / PP / SP / CP / EP / EDP / ETP"
category: llm-inference
tags: [tensor-parallelism, data-parallelism, expert-parallelism, pipeline-parallelism, sequence-parallelism, context-parallelism, moe, multi-gpu, distributed-inference, distributed-training]
created: 2026-04-14
updated: 2026-05-20
status: mature
---

# LLM Parallelism Strategies Complete Guide: DP / TP / PP / SP / CP / EP / EDP / ETP

> [!abstract]+ TL;DR
> A systematic walkthrough of the eight parallelism strategies used to scale large models across GPUs: **DP** (data), **TP** (tensor / intra-layer weights), **PP** (pipeline / inter-layer), **SP** (sequence in non-TP regions), **CP** (sequence in attention), **EP** (expert / MoE), **EDP** (expert data), **ETP** (expert tensor), plus **ZeRO/FSDP** (sharded data) and **DP Attention** (inference KV partitioning). Covers what each strategy shards, the core collective primitive, and how to compose them in 3D / 4D / 5D parallel training and inference. Production-grade case study: DeepSeek-V3 deployment.

## 1. Overview

Large models long ago outgrew the memory and compute of a single GPU. Parallelism strategies decide *how* a huge model + dataset is sharded across many GPUs. The table below summarizes every strategy discussed in this page:

| Abbrev | Alias | Full Name | What It Shards | Scope | Core Collective |
|--------|-------|-----------|----------------|-------|-----------------|
| **DP** | — | Data Parallelism | Data batches | Dense layers | AllReduce |
| **ZeRO/FSDP** | — | Sharded Data Parallelism | Data + optimizer/gradient/params | Dense layers | AllGather + ReduceScatter |
| **TP** | — | Tensor Parallelism | Weight matrices within layers | Dense layers | AllReduce |
| **SP** | — | Sequence Parallelism | Sequence dim (non-TP regions) | Dense layers | AllGather + ReduceScatter |
| **PP** | — | Pipeline Parallelism | Consecutive layer groups | All layers | Point-to-point Send/Recv |
| **CP** | — | Context Parallelism | Sequence dim (attention) | Attention layers | Ring P2P / AllToAll |
| **EP** | — | Expert Parallelism | Whole MoE experts | MoE layers | AllToAll (token routing) |
| **EDP** | **DEP** | Expert Data Parallelism | MoE-layer data batches | MoE layers | AllReduce (expert gradient sync) |
| **ETP** | **TEP** | Expert Tensor Parallelism | Individual expert weight matrices | MoE layers | AllGather + ReduceScatter |
| **DP Attention** | — | Data-Parallel Attention | KV cache partitions | Inference Attention | AllGather |

**Core intuition**: DP and TP are the classic moves for dense models; EP / EDP / ETP are their MoE-architecture counterparts. PP partitions the model layer-wise; SP and CP both operate along the sequence axis but in different scopes; ZeRO/FSDP fixes DP's memory redundancy; DP Attention is an inference-only KV-cache partitioning scheme.

```
                          ┌──────────────────────────────────┐
                          │          All GPU resources        │
                          └────────────────┬─────────────────┘
                                           │
                    ┌──────────────────────┴─────────────────────────┐
                    │                                                │
             Dense layers (Attention + MLP)               MoE layers (Router + Experts)
                    │                                                │
         ┌────┬────┼────┬────┐                        ┌─────────────┼─────────────┐
         │    │    │    │    │                         │             │             │
        DP   TP   SP   PP   CP                       EP           EDP           ETP
      (replicate (shard  (seq   (layer   (seq         (shard       (replicate    (shard
       model,    weights, dim   groups,  dim in       experts      experts        weights
       split     intra-   in    across   attention)   across       to process     of a single
       batch)    layer    non-  nodes)                GPUs)        more data)     expert)
                 comm)    TP)

             ZeRO/FSDP                               DP Attention
          (shard optimizer/                       (inference only,
           gradient/params)                        KV cache partitioned)
```

> **Why split between Dense and MoE layers?** Because NVIDIA's 2025 **MoE Parallel Folding** framework fully decouples the parallel dimensions of the two:
> - Dense: `N_total = TP × SP × CP × DP × PP` (SP usually = TP)
> - MoE: `N_total = ETP × EP × EDP × PP`
>
> The only shared constraint is PP; the rest of the axes are configured independently.

---

## 2. DP — Data Parallelism

### 2.1 The Idea

Data parallelism is the simplest and most classic strategy: **every GPU holds a complete copy of the model and processes a different subset of the data**.

```
                    ┌──────────────────────┐
                    │     Global batch      │
                    │  [x₁, x₂, x₃, x₄]    │
                    └──────────┬───────────┘
                               │ split
                    ┌──────────┴───────────┐
                    │                      │
              GPU 0: [x₁, x₂]       GPU 1: [x₃, x₄]
              ┌──────────┐           ┌──────────┐
              │ Full W   │           │ Full W   │
              │ (copy 0) │           │ (copy 1) │
              └─────┬────┘           └─────┬────┘
                    │ fwd + bwd            │ fwd + bwd
                    │                      │
                    ▼                      ▼
              grad₀ = ∂L/∂W          grad₁ = ∂L/∂W
                    │                      │
                    └───────┬──────────────┘
                            │ AllReduce
                            ▼
                    avg_grad = (grad₀ + grad₁) / 2
                            │
                    ┌───────┴───────┐
                    │               │
               GPU 0: W -= lr·avg  GPU 1: W -= lr·avg
```

### 2.2 Communication Analysis

| Phase | Primitive | Per-GPU volume |
|-------|-----------|----------------|
| Gradient sync | AllReduce | `2 × P × sizeof(dtype)` (P = parameter count) |
| ZeRO-1 (optimizer-state shard) | ReduceScatter + AllGather | Same total, lower peak memory |
| ZeRO-3 / FSDP | Per-layer AllGather (fwd) + ReduceScatter (bwd) | Same total, can pipeline |

Ring AllReduce volume formula: each GPU transfers `2 × P × (N-1)/N` bytes; for large N this approaches `2P`.

### 2.3 Code: PyTorch DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Init process group
dist.init_process_group("nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# Every GPU holds the full model
model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Different GPUs get different batches (via DistributedSampler)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

for batch in loader:
    loss = model(batch)
    loss.backward()        # DDP fires AllReduce on grads during backward
    optimizer.step()
```

### 2.4 When to Use

- Model fits on one GPU → **DP first** (simplest, highest efficiency)
- Need more throughput (larger batch size)
- Gradient sync frequency is low (once per full fwd + bwd)

### 2.5 Limitations

| Limitation | Detail |
|------------|--------|
| **Memory redundancy** | Every GPU holds the full model → poor memory efficiency |
| **Comm grows with model size** | The bigger the model, the more expensive gradient sync |
| **Can't handle gigantic models** | If the model exceeds single-GPU memory → must combine with TP/PP |

> **ZeRO / FSDP** fixes the memory-redundancy problem by sharding optimizer states, gradients, and parameters within the DP group. See §3.

---

## 3. ZeRO / FSDP — Sharded Data Parallelism

### 3.1 Background: DP Memory Redundancy

In standard DP every GPU stores the full set of model parameters, gradients, and optimizer states. For Adam, training a P-parameter FP16 model costs per GPU:

```
Per-GPU memory (standard DP, FP16 mixed precision + Adam):
  Parameters (FP16):       2P bytes
  Gradients (FP16):        2P bytes
  Optimizer state (FP32):
    - Master params (FP32): 4P bytes
    - First moment m (FP32): 4P bytes
    - Second moment v (FP32): 4P bytes
  ────────────────────────────
  Total:                   16P bytes

  Example: P = 7B → 16 × 7B = 112 GB / GPU    ← exceeds an 80GB H100!
```

ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2020) core insight: **across a DP group of N GPUs there's no reason every GPU stores the full 16P; we can shard the different pieces**.

### 3.2 The Three Stages

```
Standard DP (per GPU):              ZeRO Stage 1:
┌──────────────────────┐          ┌──────────────────────┐
│  Params W   2P       │          │  Params W   2P       │
│  Grads G    2P       │          │  Grads G    2P       │
│  Opt OS    12P       │          │  Opt OS    12P/N     │ ← sharded!
│  ─────────────────── │          │  ─────────────────── │
│  Total      16P      │          │  Total   4P + 12P/N  │
└──────────────────────┘          └──────────────────────┘

ZeRO Stage 2:                     ZeRO Stage 3 (= FSDP):
┌──────────────────────┐          ┌──────────────────────┐
│  Params W   2P       │          │  Params W   2P/N     │ ← sharded!
│  Grads G    2P/N     │ ← shard! │  Grads G    2P/N     │ ← sharded!
│  Opt OS    12P/N     │ ← shard! │  Opt OS    12P/N     │ ← sharded!
│  ─────────────────── │          │  ─────────────────── │
│  Total   2P + 14P/N  │          │  Total      16P/N    │
└──────────────────────┘          └──────────────────────┘
```

### 3.3 Per-Stage Memory Formulas

| Stage | Per-GPU memory | N=8 (7B model) | What's sharded |
|-------|----------------|----------------|----------------|
| Standard DP | `16P` | 112 GB | Nothing |
| ZeRO-1 | `4P + 12P/N` | 38.5 GB | Optimizer state |
| ZeRO-2 | `2P + 14P/N` | 26.25 GB | Optimizer state + gradients |
| ZeRO-3 / FSDP | `16P/N` | 14 GB | Optimizer state + gradients + parameters |

### 3.4 Communication Patterns

```
Standard DP:
  after backward → AllReduce(grads)
  volume: 2P bytes/GPU    count: 1×/step

ZeRO-1:
  after backward → ReduceScatter(grads) → local opt update → AllGather(updated params)
  volume: 2P bytes/GPU    count: 2×/step (can pipeline)

ZeRO-2:
  during backward → ReduceScatter(grads, per layer) → local update → AllGather(params)
  volume: 2P bytes/GPU    count: same as ZeRO-1
  Advantage: gradients freed right after ReduceScatter

ZeRO-3 / FSDP:
  forward:  AllGather(params, per layer) → compute → free non-local params
  backward: AllGather(params, per layer) → grad → ReduceScatter(grads)
  volume: 3P bytes/GPU    ← 50% more than standard DP!
  count: per-layer 2 × AllGather + 1 × ReduceScatter
```

| Strategy | Comm/GPU | Count | Memory | Use case |
|----------|----------|-------|--------|----------|
| Standard DP | `2P` | 1/step | 16P | Small model, plenty of GPU memory |
| ZeRO-1 | `2P` | 2/step | 4P + 12P/N | Optimizer state is the bottleneck |
| ZeRO-2 | `2P` | 2/step | 2P + 14P/N | Gradients also large |
| ZeRO-3/FSDP | `3P` | many/layer | 16P/N | Huge model, tight memory |

### 3.5 FSDP: PyTorch's ZeRO-3

PyTorch's **Fully Sharded Data Parallelism (FSDP)** is essentially a native ZeRO-3 implementation:

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# ZeRO-3 equivalent: shard params + grads + optimizer
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # = ZeRO-3
    device_id=local_rank,
)

# ZeRO-2 equivalent: shard grads + optimizer only, keep full params
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # = ZeRO-2
    device_id=local_rank,
)

# Training loop is almost identical to DDP
for batch in loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

FSDP2 (PyTorch 2.4+) further refines the API and performance, supporting finer-grained per-parameter sharding.

### 3.6 Which Stage When

```
Decision tree for picking a ZeRO stage:

  Model + Adam fits on 1 GPU (16P < GPU_mem)?
    ├── Yes → standard DP (fastest, least comm)
    └── No  → does ZeRO-1 fit (4P + 12P/N < GPU_mem)?
              ├── Yes → ZeRO-1 (same comm volume, recommended default)
              └── No  → does ZeRO-2 fit?
                        ├── Yes → ZeRO-2 (same comm, also shards grads)
                        └── No  → ZeRO-3 / FSDP
                                  (+50% comm, but minimum memory)
                                  Still not enough → combine with TP/PP
```

### 3.7 Limitations

| Limitation | Detail |
|------------|--------|
| **ZeRO-3 adds 50% comm** | Forward also requires AllGather of params — half again more comm than standard DP |
| **Per-layer comm adds latency** | ZeRO-3 does AllGather → compute → ReduceScatter every layer |
| **Inference doesn't benefit** | ZeRO is training-only (no optimizer/grads in inference) |
| **Watch the comm groups when combined with TP/EP** | Different parallel-axis communicators must not collide |

---

## 4. TP — Tensor Parallelism

### 4.1 The Idea

Tensor parallelism (Megatron-LM, 2019) shards weight matrices **within each layer**, splitting a single matmul across multiple GPUs. Two core sharding patterns:

#### Column-parallel

Shard weight matrix `A [D, F]` by column into `A₁ [D, F/N], A₂ [D, F/N], ...`; input `X` is replicated to every GPU:

```
          Input X [B, D]  (every GPU holds the same copy)
               │
    ┌──────────┼──────────┐
    │          │          │
  GPU 0      GPU 1      GPU 2
  A₁[D,F/3]  A₂[D,F/3]  A₃[D,F/3]      ← weights sharded by column
    │          │          │
    ▼          ▼          ▼
  Y₁=X·A₁   Y₂=X·A₂   Y₃=X·A₃          ← each computes locally, no comm
  [B, F/3]   [B, F/3]   [B, F/3]

  → GeLU(Y₁)  GeLU(Y₂)  GeLU(Y₃)        ← activation applies independently!
```

**Key**: pointwise activations like GeLU apply independently after sharding because `GeLU([Y₁, Y₂]) = [GeLU(Y₁), GeLU(Y₂)]`.

#### Row-parallel

Shard weight matrix `B [F, D]` by row into `B₁ [F/N, D], B₂ [F/N, D], ...`:

```
  GeLU(Y₁)   GeLU(Y₂)   GeLU(Y₃)        ← from column-parallel output
  [B, F/3]   [B, F/3]   [B, F/3]
    │          │          │
  GPU 0      GPU 1      GPU 2
  B₁[F/3,D]  B₂[F/3,D]  B₃[F/3,D]      ← weights sharded by row
    │          │          │
    ▼          ▼          ▼
  Z₁=Y₁·B₁  Z₂=Y₂·B₂  Z₃=Y₃·B₃        ← local partial sums
  [B, D]     [B, D]     [B, D]
    │          │          │
    └──────────┼──────────┘
               │ AllReduce (sum)
               ▼
         Z = Z₁ + Z₂ + Z₃               ← final output
             [B, D]
```

#### Full MLP block

```
           ┌─── f ───┐
           │ (identity fwd, AllReduce bwd)
           │
    X ─────┤
           │    Column Parallel          Row Parallel
           │    (gate_proj + up_proj)    (down_proj)
           │         │                       │
           │      GeLU/SiLU              AllReduce ─── g ───→ output
           │         │                       │        (AllReduce fwd,
           │         └───────────────────────┘         identity bwd)
           └──────────────────────────────────────────────────────────

    ★ Per MLP block: 1 AllReduce fwd, 1 AllReduce bwd
```

#### Attention block

Q, K, V projections are sharded by **attention head** (column-parallel) — each GPU handles a head subset. The output projection is row-parallel:

```
    GPU 0: heads [0,1]     GPU 1: heads [2,3]     GPU 2: heads [4,5]
        │                      │                      │
        ▼                      ▼                      ▼
    Attention_0            Attention_1            Attention_2
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │ AllReduce (output projection)
                               ▼
                          merged output
```

**Per transformer layer total**: 4 AllReduces (MLP fwd 1 + bwd 1, Attention fwd 1 + bwd 1).

### 4.2 Communication Analysis

| Metric | Formula |
|--------|---------|
| Per AllReduce volume | `2 × B × S × D × sizeof(dtype)` |
| Per-layer fwd comm | `2 × 2BSD = 4BSD` bytes |
| Per-layer total comm | `4 × 2BSD = 8BSD` bytes |
| TP efficiency (empirical) | TP=2: 85–95%, TP=4: 70–85%, TP=8: 56–75% |

**Compute-to-comm ratio** (JAX Scaling Book):
```
T_comms = (4 × B × D) / W_ici        ← comm time
T_compute = (8 × B × D × F) / (N × C) ← compute time

When B/N > C/W_ici → compute-bound (the ideal regime)
H100 NVLink: C ≈ 990 TFLOPS, W ≈ 900 GB/s → threshold ≈ 1100 tokens/GPU
```

### 4.3 Code: Column-Parallel + Row-Parallel

```python
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """Column-parallel linear: shard weight along output dim"""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        # Each GPU holds only 1/N of the output dim
        assert out_features % self.tp_size == 0
        self.local_out = out_features // self.tp_size
        self.weight = torch.nn.Parameter(
            torch.randn(self.local_out, in_features)  # [F/N, D]
        )

    def forward(self, x):
        # x: [B, S, D] — every GPU holds the same input
        # Each computes Y_i = X @ W_i^T → [B, S, F/N]
        return torch.nn.functional.linear(x, self.weight)
        # No comm needed! GeLU can be applied directly.


class RowParallelLinear(torch.nn.Module):
    """Row-parallel linear: shard weight along input dim"""
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        assert in_features % self.tp_size == 0
        self.local_in = in_features // self.tp_size
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, self.local_in)  # [D, F/N]
        )

    def forward(self, x):
        # x: [B, S, F/N] — each GPU holds part of the input
        local_out = torch.nn.functional.linear(x, self.weight)  # [B, S, D]
        # AllReduce-sum → Z = Z₁ + Z₂ + ... + Z_N
        dist.all_reduce(local_out, group=self.tp_group)
        return local_out


class TPTransformerMLP(torch.nn.Module):
    """Tensor-parallel MLP block"""
    def __init__(self, hidden_dim, ffn_dim, tp_group):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_dim, ffn_dim, tp_group)
        self.down_proj = RowParallelLinear(ffn_dim, hidden_dim, tp_group)

    def forward(self, x):
        # Column parallel → no comm
        h = self.gate_proj(x)
        h = torch.nn.functional.silu(h)       # activation applies independently
        # Row parallel → AllReduce
        return self.down_proj(h)
```

### 4.4 When to Use

- Single-layer parameters exceed one GPU
- **Low-latency inference** (minimize single-request latency)
- Always stay **within the NVLink domain** (typically 1 node = 8 GPUs)

### 4.5 Limitations

| Limitation | Detail |
|------------|--------|
| **Comm every layer** | AllReduce is sync/blocking, twice per layer |
| **Efficiency drops with TP degree** | TP=8 efficiency may be only 56–75% |
| **NVLink required** | On PCIe at TP=4, comm can be 40–50% of inference time |
| **Limited scaling** | Typically TP ≤ 8 (one NVLink domain) |

---

## 5. SP — Sequence Parallelism

### 5.1 Motivation: TP's Activation-Memory Blind Spot

TP shards the weight matrices, so activations *inside* TP regions (Attention, MLP matmuls) shrink by `1/TP`. But Transformer layers also contain **LayerNorm** and **Dropout**, which sit *outside* TP's scope — these need full hidden-dim inputs, so every GPU holds the full activation tensor.

```
Activation memory per Transformer layer (no SP):

    LayerNorm₁   →  Attention (TP region) →  Dropout₁  →  LayerNorm₂  →  MLP (TP region) →  Dropout₂
    ┌──────┐       ┌──────────────┐         ┌──────┐     ┌──────┐       ┌──────────┐       ┌──────┐
    │ B×S×D │       │  B×S×D/TP    │         │ B×S×D │     │ B×S×D │       │ B×S×D/TP  │       │ B×S×D │
    │ (full)│       │  (sharded)   │         │ (full)│     │ (full)│       │ (sharded) │       │ (full)│
    └──────┘       └──────────────┘         └──────┘     └──────┘       └──────────┘       └──────┘
       ★ The unsharded parts account for >50% of activation memory!
```

SP (Megatron-LM, Korthikanti et al., 2022) fixes this: **shard the non-TP regions along the sequence dimension**.

### 5.2 How It Works

SP's idea is elegant: reuse TP's existing AllReduce by decomposing it into AllGather + ReduceScatter, which lets us switch seamlessly between the two region types.

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Transformer layer (TP + SP)                    │
    │                                                                 │
    │   [SP region]         [TP region]         [SP region]           │
    │   LayerNorm           Attention/MLP        Dropout              │
    │   per GPU: [B,S/TP,D] per GPU: [B,S,D/TP]  per GPU: [B,S/TP,D]  │
    │   sharded along seq   sharded along hidden  sharded along seq   │
    │        │                     │                    │             │
    │        └── AllGather(seq) ──→┘                    │             │
    │              (gather full seq)                    │             │
    │                              └── ReduceScatter ──→┘             │
    │                                   (reduce + re-shard seq)       │
    │                                                                 │
    │   ★ AllReduce = AllGather + ReduceScatter                       │
    │   ★ Total comm volume unchanged — one AllReduce split into two  │
    │   ★ But LayerNorm/Dropout activation memory drops from B×S×D    │
    │     to B×(S/TP)×D                                                │
    └─────────────────────────────────────────────────────────────────┘
```

Step-by-step flow (one Transformer layer):

```
    GPU0 holds tokens [0 : S/TP] of the sequence
    GPU1 holds tokens [S/TP : 2S/TP]
    ...

    Step 1: LayerNorm — each GPU LayerNorms its own S/TP tokens
            in: [B, S/TP, D]   out: [B, S/TP, D]

    Step 2: AllGather(seq dim) — gather the full sequence
            in: [B, S/TP, D]   out: [B, S, D]
            ★ Now each GPU has the full seq → enter TP region

    Step 3: Attention / MLP (TP region) — shard along hidden
            per GPU: [B, S, D/TP]

    Step 4: ReduceScatter(seq dim) — sum + redistribute along seq
            in: [B, S, D] (per-GPU partial results)
            out: [B, S/TP, D] (reduced result, sharded by sequence)
            ★ Back into SP region

    Step 5: Dropout — each GPU drops its own S/TP tokens
            in: [B, S/TP, D]   out: [B, S/TP, D]
```

### 5.3 Memory Savings

```
Activation memory per GPU:

              No SP (TP only)                   With SP (TP + SP)
    LayerNorm:  B × S × D                   B × S/TP × D       ← TP× smaller!
    Attention:  B × S × D/TP                B × S × D/TP       (unchanged)
    Dropout:    B × S × D                   B × S/TP × D       ← TP× smaller!
    MLP:        B × S × D/TP                B × S × D/TP       (unchanged)

    Overall: non-TP region activations shrink to 1/TP
    For TP=8, total activation memory reduced ~40–60%
```

### 5.4 Why SP is mandatory with TP+EP

When TP and EP are both on, Megatron-LM requires `--sequence-parallel`. Reasons:

1. Dense layers (Attention) use TP, MoE layers use EP.
2. TP groups and EP groups typically partition GPUs differently.
3. SP's AllGather / ReduceScatter is the natural bridge between the two layer types' data formats.
4. Without SP, TP's AllReduce can't cleanly hand off into EP's AllToAll.

### 5.5 Code

```python
import torch
import torch.distributed as dist

class SequenceParallelLayerNorm(torch.nn.Module):
    """Sequence-parallel LayerNorm: each GPU handles only S/TP tokens"""
    def __init__(self, hidden_dim, tp_group):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.tp_group = tp_group

    def forward(self, x):
        # x: [B, S/TP, D] — sequence already sharded across TP group
        return self.norm(x)  # LayerNorm operates along hidden_dim, doesn't need full seq


def allgather_seq(x, tp_group):
    """AllGather: [B, S/TP, D] → [B, S, D]"""
    tp_size = dist.get_world_size(tp_group)
    gathered = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(gathered, x, group=tp_group)
    return torch.cat(gathered, dim=1)  # concat along seq


def reducescatter_seq(x, tp_group):
    """ReduceScatter: [B, S, D] → [B, S/TP, D] (reduce then shard by seq)"""
    tp_size = dist.get_world_size(tp_group)
    chunks = list(x.chunk(tp_size, dim=1))
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, chunks, group=tp_group)
    return output


class SPTransformerBlock(torch.nn.Module):
    """Transformer block with sequence parallelism"""
    def __init__(self, hidden_dim, ffn_dim, n_heads, tp_group):
        super().__init__()
        self.ln1 = SequenceParallelLayerNorm(hidden_dim, tp_group)
        self.attn = TPAttention(hidden_dim, n_heads, tp_group)  # TP region
        self.ln2 = SequenceParallelLayerNorm(hidden_dim, tp_group)
        self.mlp = TPTransformerMLP(hidden_dim, ffn_dim, tp_group)  # TP region
        self.tp_group = tp_group

    def forward(self, x):
        # x: [B, S/TP, D] — SP region
        residual = x

        # SP → TP transition
        h = self.ln1(x)                              # [B, S/TP, D] — SP
        h = allgather_seq(h, self.tp_group)           # [B, S, D] — full seq
        h = self.attn(h)                              # [B, S, D/TP] — TP region
        h = reducescatter_seq(h, self.tp_group)       # [B, S/TP, D] — back to SP
        h = torch.nn.functional.dropout(h, p=0.1)    # SP region
        x = residual + h

        # Same pattern for MLP
        residual = x
        h = self.ln2(x)
        h = allgather_seq(h, self.tp_group)
        h = self.mlp(h)
        h = reducescatter_seq(h, self.tp_group)
        h = torch.nn.functional.dropout(h, p=0.1)
        x = residual + h

        return x  # [B, S/TP, D]
```

### 5.6 SP vs CP — the naming collision

"Sequence parallelism" is overloaded across at least three different papers, which sends people in circles. **In this wiki, SP means the Megatron-LM v2 definition** (Korthikanti et al., 2022): sharding only LayerNorm/Dropout activations *inside* a TP group. **CP means** the broader sequence-dimension sharding of the *whole* attention computation, across an independent GPU dimension.

Three things people call "sequence parallelism" in papers and codebases:

| What's actually sharded | What people call it | What this wiki calls it |
|---|---|---|
| LayerNorm / Dropout activations inside a TP group (Megatron v2) | "Sequence parallelism" | **SP** |
| Whole attention, across an independent GPU dim, via AllToAll (DeepSpeed Ulysses) | "Sequence parallelism" | **CP** (Ulysses variant) |
| Whole attention, across an independent GPU dim, via Ring P2P (Ring Attention) | "Sequence parallelism" *or* "Ring Attention" | **CP** (Ring variant) |

Two questions are enough to disambiguate any paper:

1. **What is being sharded?** Just LayerNorm/Dropout → SP. The whole attention → CP.
2. **Inside a TP group, or a separate GPU dimension?** Inside TP → SP (size always = TP). Separate dim → CP (any size, independent of TP).

Full contrast:

| Dimension | SP (Megatron v2) | CP |
|-----------|------------------|----|
| What it shards | LayerNorm + Dropout activations (non-TP regions) | The whole attention (QKV + softmax + output proj) |
| Relation to TP | Always inside a TP group; SP size = TP size | Independent GPU dim; CP size unrelated to TP |
| Communication | AllGather (entering TP region) + ReduceScatter (leaving) | Ring P2P (Ring Attention) or AllToAll (Ulysses) |
| Why it exists | Reduce activation memory inside TP regions | Support sequences too long to fit one GPU's KV cache |
| Typical size | SP = TP = 8 (whatever TP you picked) | CP = 2 / 4 / 8 / ... / 64+ |
| Mandatory? | Megatron-LM forces SP whenever TP + EP are both used | Only when sequences exceed single-GPU capacity |
| Talks to | TP-region AllReduce, decomposed into AG + RS | Other CP ranks holding adjacent sequence segments |
| Total comm volume | Same as plain TP (AllReduce decomposed, not eliminated) | Adds new comm on top of TP/DP |

The DeepSpeed Ulysses paper title — *"Sequence Parallelism for Long Sequence Training"* — is the most painful overlap, because that paper's "sequence parallelism" is exactly what every other framework now calls CP. **When reading any paper that mentions "sequence parallelism," check the dimension being sharded, not the word.** If the attention math is changed, it's CP. If only LayerNorm/Dropout activations move, it's SP.

### 5.7 Limitations

| Limitation | Detail |
|------------|--------|
| **Total comm unchanged** | Just splits AllReduce into AG + RS; same total bytes |
| **Tied to TP** | SP is an attachment to TP; no TP → no SP |
| **Implementation complexity** | Tensor shapes must be converted correctly between TP and SP regions |

---

## 6. PP — Pipeline Parallelism

### 6.1 Basic Concept

Pipeline parallelism assigns **consecutive layer groups** to different GPUs. Each GPU is responsible for forward + backward on a subset of layers, passing activations and gradients between stages via **point-to-point Send/Recv**.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   GPU 0     │    │   GPU 1     │    │   GPU 2     │    │   GPU 3     │
│  Stage 0    │───▶│  Stage 1    │───▶│  Stage 2    │───▶│  Stage 3    │
│ Layer 0-7   │    │ Layer 8-15  │    │ Layer 16-23 │    │ Layer 24-31 │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                 Send/Recv          Send/Recv          Send/Recv
            (activation only)   (activation only)   (activation only)
```

**Why PP is attractive**:
- **Low comm volume**: only one activation tensor `[B, S, H]` crosses each stage boundary — not every-layer AllReduces like TP.
- **Low bandwidth requirement**: P2P is enough; no NVLink needed; PCIe or even cross-node networks work.
- **Memory efficient**: each GPU holds only `1/PP` of the model parameters.

**Core problem**: **pipeline bubbles** — when a stage is waiting on upstream data or downstream gradients, it's idle.

### 6.2 Naive PP and the Bubble Problem

The naive form: one micro-batch flows through all stages sequentially; only one GPU works at any instant.

```
GPU 0: │██F██│                        │██B██│
GPU 1: │     │██F██│            │██B██│
GPU 2: │     │     │██F██│██B██│
GPU 3: │     │     │     │█F+B█│        blank = bubble (idle)
```

**Utilization** = `1 / (2p)`. At p=4, only 12.5% — unacceptable.

### 6.3 GPipe: Micro-batch Pipelining

**GPipe** (Huang et al., 2019) splits a mini-batch into **m micro-batches**, letting multiple micro-batches stream through the pipeline so multiple GPUs work in parallel.

**Schedule**: finish all micro-batches' forward passes (all F), then sequentially execute all backward passes (all B). A **pipeline flush** sits in the middle.

```
GPipe schedule (p=4 stages, m=4 micro-batches)

time ──────────────────────────────────────────────────────────────────────▶

GPU 0: │F₁│F₂│F₃│F₄│         │B₄│B₃│B₂│B₁│
GPU 1: │  │F₁│F₂│F₃│F₄│      │B₄│B₃│B₂│B₁│
GPU 2: │  │  │F₁│F₂│F₃│F₄│   │B₄│B₃│B₂│B₁│
GPU 3: │  │  │  │F₁│F₂│F₃│F₄│B₄│B₃│B₂│B₁│
                               ▲
                          pipeline flush
                          (sync barrier)

    █ = compute    blank = bubble
```

**Bubble-rate formula**:

```
                   p - 1
Bubble Rate = ─────────────
                m + p - 1
```

- p = number of pipeline stages
- m = number of micro-batches

| p (stages) | m (micro-batches) | Bubble rate |
|------------|-------------------|-------------|
| 4 | 4 | 42.9% |
| 4 | 8 | 27.3% |
| 4 | 16 | 15.8% |
| 4 | 32 | 8.6% |
| 8 | 32 | 17.9% |
| 16 | 64 | 19.0% |

**Rule of thumb**: `m >= 4 × p` keeps bubble rate around `< 20%`. But increasing m means smaller per-micro-batch compute (possibly hurting kernel efficiency) and more activations to store (GPipe addresses this with activation recomputation).

### 6.4 1F1B Scheduling (PipeDream)

**1F1B (One Forward One Backward)** (PipeDream, Narayanan et al., 2019): **don't wait until all forwards finish — interleave forward and backward as early as possible**.

**Three phases**:
1. **Warmup**: each stage progressively kicks off forwards, filling the pipeline.
2. **Steady state**: each GPU strictly alternates 1 F and 1 B.
3. **Cooldown**: drain remaining backward passes.

```
1F1B schedule (p=4, m=8)

time ──────────────────────────────────────────────────────────────────────▶

GPU 0: │F₁│F₂│F₃│F₄│B₁│F₅│B₂│F₆│B₃│F₇│B₄│F₈│B₅│B₆│B₇│B₈│
GPU 1: │  │F₁│F₂│F₃│B₁│F₄│B₂│F₅│B₃│F₆│B₄│F₇│B₅│F₈│B₆│B₇│B₈│
GPU 2: │  │  │F₁│F₂│B₁│F₃│B₂│F₄│B₃│F₅│B₄│F₆│B₅│F₇│B₆│F₈│B₇│B₈│
GPU 3: │  │  │  │F₁│B₁│F₂│B₂│F₃│B₃│F₄│B₄│F₅│B₅│F₆│B₆│F₇│B₇│F₈│B₈│

        ◄─warmup─▶◄────────── steady state ──────────▶◄─cooldown─▶
```

**1F1B benefits**:
- **Same bubble rate as GPipe**: `(p-1) / (m+p-1)`
- **Lower peak memory**: no need to store activations of all m micro-batches simultaneously. In steady state each GPU only keeps activations for p micro-batches (vs m in GPipe).
- **Earlier memory release**: activations free as soon as the corresponding backward runs.

**Memory comparison**:

| Schedule | Peak activation storage |
|----------|------------------------|
| GPipe | activations for m micro-batches |
| 1F1B | activations for p micro-batches |

When m >> p, 1F1B's memory advantage is significant.

### 6.5 Interleaved 1F1B (Virtual Stages)

**Interleaved 1F1B** (Narayanan et al., 2021, Megatron-LM v2) adds **virtual pipelining**: each GPU no longer owns one contiguous chunk of layers but **multiple non-contiguous chunks (model chunks)**.

E.g. 4 GPUs with virtual_pipeline_size=2:

```
Physical assignment:
  GPU 0: Layer 0-3  + Layer 16-19    (chunk 0 + chunk 4)
  GPU 1: Layer 4-7  + Layer 20-23    (chunk 1 + chunk 5)
  GPU 2: Layer 8-11 + Layer 24-27    (chunk 2 + chunk 6)
  GPU 3: Layer 12-15 + Layer 28-31   (chunk 3 + chunk 7)

Logical pipeline (8 virtual stages):
  VS0 → VS1 → VS2 → VS3 → VS4 → VS5 → VS6 → VS7
  GPU0   GPU1   GPU2   GPU3   GPU0   GPU1   GPU2   GPU3
```

**Bubble rate**:

```
                         p - 1
Bubble Rate = ──────────────────────
               m × v + p - 1

v = virtual_pipeline_model_parallel_size (chunks per GPU)
```

At v=2, the bubble shrinks to roughly half.

**Costs**:
- Comm volume grows by v× (each virtual stage adds Send/Recv)
- More implementation complexity

### 6.6 Zero Bubble PP (Qi et al., ICLR 2024)

**Core insight**: backward splits into two independent pieces:
- **B** (backward_input): compute input gradient ∂L/∂x (needs downstream gradient, time-sensitive)
- **W** (backward_weight): compute weight gradient ∂L/∂W (depends only on local activations, can be deferred)

Classic 1F1B bundles B and W together; Zero Bubble decouples them and uses W to **fill bubbles**.

#### ZB-H1: improvement on top of 1F1B

ZB-H1 largely follows 1F1B but defers W to fill the trailing bubbles. **Bubble rate is roughly 1/3 of 1F1B**.

```
ZB-H1 sketch: F...F│B│F│B│F│...│B│...│B│W│W│W│W│W│W│...
                                       ▲ W fills the original bubble region
```

#### ZB-H2: near zero bubble

Allows more memory (more in-flight micro-batches); W can fill *all* bubbles. **In theory, zero bubble**.

#### ZB-V: V-shaped virtual stages

Each GPU owns 2 chunks with V-shaped dependencies: `VS0→VS1→VS2→VS3→VS7→VS6→VS5→VS4`. Zero bubble when `T_F ≈ T_B ≈ T_W`.

**Summary of bubbles across schedules**:

| Schedule | Bubble time | Condition |
|----------|-------------|-----------|
| **GPipe** | `(p-1) × (T_F + T_B)` | — |
| **1F1B** | `(p-1) × (T_F + T_B)` | Lower peak memory |
| **Interleaved 1F1B** | `(p-1) × (T_F + T_B) / v` | v = virtual stages |
| **ZB-H1** | `(p-1) × T_W` | ≈ 1/3 of 1F1B |
| **ZB-H2** | ≈ 0 | More memory required |
| **ZB-V** | 0 (when T_F = T_B = T_W) | 2 virtual stages |

### 6.7 DualPipe (DeepSeek-V3)

**DualPipe** is the bidirectional pipeline-parallel algorithm used in DeepSeek-V3/R1 training, specifically engineered for **overlapping compute and communication**, particularly the large cross-node AllToAll volume in MoE models.

#### Core idea

1. **Decompose each chunk into 4 components**:
   - **ATTN**: attention compute (pure compute)
   - **DISPATCH**: AllToAll dispatch, routing tokens to experts (comm)
   - **MLP**: expert / FFN compute (pure compute)
   - **COMBINE**: AllToAll combine, gathering expert outputs (comm)

2. **Bidirectional schedule**: feed micro-batches from **both ends** of the pipeline simultaneously; forward and reverse streams interleave.

3. **Compute-comm overlap**: one micro-batch's comm (DISPATCH/COMBINE) overlaps another micro-batch's compute (ATTN/MLP).

```
DualPipe bidirectional schedule (p=4 stages)

                    Forward stream (micro-batch from stage 0 → stage 3)
                    ──────────────────────────────────────▶
GPU 0: │F→│F→│F→│...│B→│B→│...│                        │W│W│W│...
GPU 1: │  │F→│F→│...│   │B→│...│   │F←│...│B←│...│     │W│W│W│...
GPU 2: │  │  │F→│...│   │   │...│F←│F←│...│B←│B←│...│  │W│W│W│...
GPU 3: │                        │F←│F←│F←│...│B←│B←│B←│...│W│W│W│...
                    ◀──────────────────────────────────────
                    Reverse stream (micro-batch from stage 3 → stage 0)

    F→ = forward of forward stream    F← = forward of reverse stream
    B→ = backward of forward stream   B← = backward of reverse stream
    W  = weight gradient compute
```

#### Compute-comm overlap detail

```
A forward + backward chunk pair, 4 time steps:

  Step 1: ATTN(fwd) [compute] + COMBINE(bwd) [comm]   ← overlap
  Step 2: DISPATCH(fwd) [comm] + MLP(bwd) [compute]    ← overlap
  Step 3: MLP(fwd) [compute] + DISPATCH(bwd) [comm]    ← overlap
  Step 4: COMBINE(fwd) [comm] + ATTN(bwd) [compute]    ← overlap

  Every step has compute + comm simultaneously → full overlap
```

#### DualPipe bubble and cost

**Bubble time**:

```
DualPipe Bubble = (p/2 - 1) × T_{F&B}

where T_{F&B} = execution time of one overlapped forward+backward chunk pair
```

Compared with 1F1B's `(p-1) × (T_F + T_B)`, DualPipe cuts the bubble by roughly **50%+** at large p (and T_{F&B} < T_F + T_B because of overlap).

**Memory cost**: **2× parameter memory** (each direction stream needs its own copy); p+1 micro-batches of activations.

### 6.8 Communication Pattern Analysis

```
Topology: Stage 0 ──send──▶ Stage 1 ──send──▶ Stage 2 ──send──▶ Stage 3
                  ◀──recv──         ◀──recv──         ◀──recv──
Fwd: left → right activation flow    Bwd: right → left gradient flow
Comm volume per send = B × S × H × sizeof(dtype)   e.g. 1×4096×8192×2B = 64 MB
```

**PP vs TP comm**:

| Feature | PP | TP |
|---------|----|----|
| Primitive | Send/Recv (P2P) | AllReduce / AllGather |
| Per-layer count | 0 (only at stage boundaries) | 2 AllReduces (fwd) |
| Bandwidth need | Low (PCIe is fine) | High (NVLink) |
| Latency impact | Adds pipeline latency | Adds per-layer latency |

### 6.9 PP Design Choices

#### Number of stages

The number of stages p trades **memory savings** (`1/p`) against **bubble overhead** (`(p-1)/(m+p-1)`).

**Principles**:
- **Keep p as small as possible**: just enough to meet the memory requirement; don't over-partition.
- **m >> p**: ensure micro-batch count is much larger than stage count.
- **Typical configs**: p = 2, 4, 8. p > 16 usually unacceptable bubble (unless using Zero Bubble / DualPipe).

#### Load balancing

Different layers have different compute (e.g. MoE vs Dense, first layer with embedding, last layer with LM head). **Unbalanced stages amplify bubbles** — the slowest stage caps everything.

**Approaches**:
- **Profiling**: measure per-layer time, partition for balanced compute.
- **Megatron-LM `--pipeline-model-parallel-layout`**: supports flexible layer assignment.
- **DeepSeek-V3**: for 61 decoder layers + 1 MTP layer, uses PP=16, VPP=2, custom layout.

#### Micro-batch size

Micro-batch size b affects:
1. **Compute efficiency**: too small → low GPU utilization (kernel-launch overhead dominates).
2. **Bubble rate**: m = B/b; larger m → smaller bubble, but smaller b.
3. **Activation memory**: each in-flight micro-batch stores activations.

**Practical advice**:
- Training: pick b that maximizes per-stage GPU efficiency, then choose B so m >= 4p.
- Inference: b is usually the current concurrent-request count; PP has limited tuning room.

#### PP for training vs inference

| Dim | Training | Inference |
|-----|----------|-----------|
| **Primary goal** | Reduce bubble, maximize training throughput | Lower latency, raise serving throughput |
| **Micro-batch count** | Freely tunable (m >> p) | Bound by concurrent requests |
| **Bubble issue** | Mitigated by high m | Bad at low concurrency |
| **Backward** | Yes (B + W needed) | No (forward only) |
| **Schedule complexity** | High (1F1B, ZB, DualPipe) | Low (simple forward pipeline) |
| **Memory** | Params + optimizer + activations | Params + KV cache |

### 6.10 PP in Inference

#### When to pick PP over TP

```
Decision tree:

Model exceeds one GPU?
  ├── No  → single-GPU inference (maybe + quantization)
  └── Yes → have NVLink?
          ├── Yes → low-latency requirement?
          │       ├── Yes → TP (inside NVLink domain)
          │       └── No  → TP + PP, or pure PP (high throughput)
          └── No (PCIe only) → PP or DP
```

**PP inference is a fit for**:
1. **PCIe systems**: without NVLink, TP comm is 40–50% overhead; PP only needs P2P.
2. **Cross-node deployment**: only network between nodes; PP's point-to-point pattern fits.
3. **High-concurrency throughput**: many requests can fill the pipeline and hide bubbles.
4. **Cost optimization**: PP doesn't require expensive NVLink hardware.

#### Pipeline bubbles in inference

Inference is forward-only, no backward. But a single request still passes sequentially through every stage, creating **sequential latency**.

```
Inference PP (single request):

GPU 0: │████ F ████│                                    │
GPU 1: │            │████ F ████│                        │
GPU 2: │            │            │████ F ████│            │
GPU 3: │            │            │            │████ F ████│

Latency = p × T_stage  (serial, no parallelism)
Single-request latency worse than TP
```

**PP inference at high concurrency**:

```
Multi-request pipeline (PP=4, many requests):

GPU 0: │F_R1│F_R2│F_R3│F_R4│F_R5│F_R6│...
GPU 1: │    │F_R1│F_R2│F_R3│F_R4│F_R5│...
GPU 2: │    │    │F_R1│F_R2│F_R3│F_R4│...
GPU 3: │    │    │    │F_R1│F_R2│F_R3│...

Steady state: one request finishes every T_stage → throughput = 1/T_stage
```

### 6.11 Code Examples

#### PyTorch Pipeline Parallelism

```python
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, Schedule1F1B

# Each rank only instantiates its layers
if rank == 0:
    stage_module = nn.Sequential(model.embed, *model.layers[:8])
elif rank == 1:
    stage_module = nn.Sequential(*model.layers[8:16])
elif rank == 2:
    stage_module = nn.Sequential(*model.layers[16:24])
else:
    stage_module = nn.Sequential(*model.layers[24:], model.head)

stage = PipelineStage(stage_module, stage_index=rank,
                      num_stages=world_size, device=f"cuda:{rank}")

schedule = ScheduleGPipe(stage, n_microbatches=8)  # or Schedule1F1B
if rank == 0:
    schedule.step(input_batch)
else:
    output = schedule.step()
```

#### Megatron-LM PP configuration

```bash
# Basic PP: TP=4, PP=8
--pipeline-model-parallel-size 8 --tensor-model-parallel-size 4

# Virtual pipeline (Interleaved 1F1B): 2 model chunks per GPU
--pipeline-model-parallel-size 8 --virtual-pipeline-model-parallel-size 2

# DeepSeek-V3 custom layout (PP=16, VPP=2)
--pipeline-model-parallel-size 16 --virtual-pipeline-model-parallel-size 2 \
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

#### Recommend PP stage count

```python
def recommend_pp_stages(model_params_gb, gpu_memory_gb, kv_cache_gb, overhead=1.3):
    """Recommend PP stage count — smallest power-of-2 that satisfies memory budget"""
    available = gpu_memory_gb - kv_cache_gb
    min_gpus = math.ceil(model_params_gb * overhead / available)
    pp = 1
    while pp < min_gpus:
        pp *= 2
    return pp

# E.g. Llama-3-70B (140GB FP16) on 80GB H100, KV≈10GB → pp=4
```

### 6.12 Limitations

| Limitation | Detail |
|------------|--------|
| **Pipeline bubble** | Even the most advanced scheduling rarely eliminates bubbles entirely |
| **Load imbalance** | Per-stage compute differences amplify bubbles |
| **Single-request latency** | PP adds serial stage-to-stage latency |
| **Schedule complexity** | Advanced strategies (ZB, DualPipe) are hard to implement and debug |
| **Memory fragmentation** | Each stage holds only partial layers; GPU utilization may be uneven |

---

## 7. CP — Context Parallelism

### 7.1 Why CP

As LLM context windows expand from 4K to 128K, 1M, even 10M tokens, a single GPU's memory can no longer hold the KV cache and intermediate activations needed for attention:

```
KV cache memory (per request):
  = 2 × n_layers × n_kv_heads × d_head × seq_len × sizeof(dtype)

  E.g. Llama-3-70B, seq_len=1M, FP16
  = 2 × 80 × 8 × 128 × 1,000,000 × 2 bytes
  = 327.68 GB  ← far beyond a single GPU

Attention compute complexity: O(S²) — doubling seq quadruples attention FLOPs
```

**CP's core idea**: shard the **sequence dimension** across multiple GPUs; each GPU processes a slice of the sequence and they collaborate via communication to complete the full attention computation.

### 7.2 Ring Attention

**Ring Attention** (Liu et al., ICLR 2024) is the canonical CP implementation, inspired by the ring-communication pattern of Ring AllReduce.

#### How it works

1. Split the sequence into N equal segments across N GPUs.
2. Each GPU holds its own Q (query) segment and the current-round KV segment.
3. In a **ring topology**, KV blocks rotate around the ring; each GPU computes a partial attention per round.
4. After N rounds, every GPU has computed attention against all KV.

```
Ring Attention (4 GPUs, sequence split into 4 segments)

Initial:
  GPU 0: Q₀, KV₀    GPU 1: Q₁, KV₁    GPU 2: Q₂, KV₂    GPU 3: Q₃, KV₃

Round 1: local compute Attn(Qᵢ, KVᵢ)
  GPU 0: Attn(Q₀,KV₀)  GPU 1: Attn(Q₁,KV₁)  GPU 2: Attn(Q₂,KV₂)  GPU 3: Attn(Q₃,KV₃)
  Simultaneously: KV₀→GPU1, KV₁→GPU2, KV₂→GPU3, KV₃→GPU0  (ring rotation)

Round 2: receive neighbor's KV, compute Attn(Qᵢ, KVᵢ₋₁)
  GPU 0: Attn(Q₀,KV₃)  GPU 1: Attn(Q₁,KV₀)  GPU 2: Attn(Q₂,KV₁)  GPU 3: Attn(Q₃,KV₂)
  Simultaneously: KV continues to rotate around the ring

Round 3: ...
Round 4: final round

                 ┌─────────────────────────────────────┐
                 │           Ring Topology              │
                 │                                      │
                 │    GPU 0 ───KV──▶ GPU 1              │
                 │      ▲                │              │
                 │      │KV            KV│              │
                 │      │                ▼              │
                 │    GPU 3 ◀──KV─── GPU 2              │
                 │                                      │
                 └─────────────────────────────────────┘
```

#### Compute-comm overlap

The key Ring Attention optimization: **while round i's KV is in flight, the GPU is already computing attention with round i-1's received KV**.

```
GPU 0 timeline:

Round 1:  │ Compute: Attn(Q₀, KV₀) │ Send: KV₀ → GPU1  │
Round 2:  │ Compute: Attn(Q₀, KV₃) │ Send: KV₃ → GPU1  │  ← compute and comm overlap!
Round 3:  │ Compute: Attn(Q₀, KV₂) │ Send: KV₂ → GPU1  │
Round 4:  │ Compute: Attn(Q₀, KV₁) │                    │

If T_compute >= T_communicate, comm is fully hidden
Condition: each segment is long enough (S/N large) so compute > transfer
```

**Ring Attention properties**:
- Memory: per-GPU sequence memory ∝ S/N (linear scaling)
- Compute: total FLOPs unchanged, identical to standard attention
- Comm: KV blocks transit per round, N-1 rounds total, overlapped with compute
- Context length scales linearly with GPU count

**Real-world results**:
- Meta: 1M tokens on a single H100 host in <1 minute; 10M tokens on 32 hosts in <1 minute.
- RingX (SC'24): training Llama-3-8B at 1M sequence on 4096 Frontier GPUs at 38% MFU.

### 7.3 Ulysses (DeepSpeed)

**DeepSpeed-Ulysses** (Jacobs et al., 2023) takes a different approach: **AllToAll-based** sequence parallelism.

#### How it works

1. The sequence is sharded along the seq dim across GPUs (each holds S/N tokens).
2. Before attention compute, **AllToAll** transposes:
   - In: each GPU holds all heads, partial sequence
   - Out: each GPU holds partial heads, full sequence
3. Each GPU runs **standard attention on the full sequence for its head subset**.
4. After attention, another **AllToAll** restores the original distribution.

```
Ulysses AllToAll pattern (4 GPUs, 8 heads)

Step 1: initial distribution — sharded by sequence
  GPU 0: [seq 0:S/4, heads 0-7]     ← all heads, partial seq
  GPU 1: [seq S/4:S/2, heads 0-7]
  GPU 2: [seq S/2:3S/4, heads 0-7]
  GPU 3: [seq 3S/4:S, heads 0-7]

Step 2: AllToAll — switch to head-sharded
  GPU 0: [seq 0:S, heads 0-1]       ← partial heads, full seq
  GPU 1: [seq 0:S, heads 2-3]
  GPU 2: [seq 0:S, heads 4-5]
  GPU 3: [seq 0:S, heads 6-7]

Step 3: standard attention (each GPU runs its own heads)
  GPU 0: Attn(Q₀₋₁, K₀₋₁, V₀₋₁)   ← uses FlashAttention, full seq
  GPU 1: Attn(Q₂₋₃, K₂₋₃, V₂₋₃)
  ...

Step 4: AllToAll — restore seq sharding
  GPU 0: [seq 0:S/4, heads 0-7]
  ...
```

#### Ring Attention vs Ulysses

| Property | Ring Attention | Ulysses |
|----------|----------------|---------|
| Primitive | P2P Send/Recv (ring) | AllToAll |
| Rounds | N-1 rounds | 2 AllToAlls |
| Overlap with compute | Yes (key advantage) | No (AllToAll is blocking) |
| Intrusiveness on attention impl | High (modify attention kernel) | Low (add comm around attention) |
| Bandwidth fit | Tolerates low bandwidth | Needs high bandwidth |
| GPU-count limit | None (just P2P) | ≤ num_heads |
| FlashAttention compatibility | Needs adapting | Direct |

**Hybrid CP**: in practice, combine both — Ulysses within a node (high BW), Ring across nodes (latency hidden). PyTorch's Context Parallel implementation supports this hybrid mode. See [[#7.6 Current best practice — hybrid CP|§7.6 Hybrid CP]].

### 7.4 Megatron CP (Megatron-LM)

Megatron's CP is **Ring Attention plus three production-grade refinements**, and is the de-facto standard for long-context training inside NVIDIA / Megatron-LM-derived stacks (NeMo, NeMo-Megatron-Core, NVIDIA-style Llama training).

**Refinement 1 — Zigzag load balancing for causal masks.**

Plain Ring + causal mask gives GPU 0 almost no work and GPU $N{-}1$ all the work. Megatron splits the sequence into $2N$ blocks rather than $N$, then **pairs early + late blocks on the same GPU**:

```
Plain ring (N=4 GPU):
  GPU0: [block 0]        ← almost nothing under causal
  GPU1: [block 1]
  GPU2: [block 2]
  GPU3: [block 3]        ← almost everything under causal

Zigzag (N=4 GPU, split into 8 blocks):
  GPU0: [block 0, block 7]    ← one early + one late → balanced
  GPU1: [block 1, block 6]
  GPU2: [block 2, block 5]
  GPU3: [block 3, block 4]
```

Each GPU now holds one "early" and one "late" segment, so under causal masking every GPU does roughly the same amount of work. Effective throughput nearly doubles relative to plain Ring + causal.

**Refinement 2 — Ring P2P embedded inside the FlashAttention kernel loop.**

Rather than scheduling Ring P2P at the framework layer (with FlashAttention as a black-box kernel), Megatron-LM modifies FlashAttention's tile loop to issue `send` / `recv` directly between tile iterations. This is what enables real compute/communication overlap — the framework-layer version often stalls between iterations.

**Refinement 3 — First-class integration with TP / PP / DP / SP / EP.**

CP is exposed as another parallel dimension via `--context-parallel-size <N>`. The framework constructs the right NCCL communicator groups, manages KV cache sharding across CP, handles cross-CP boundary masking, and composes with the other 4 dimensions automatically. You don't write any of the plumbing.

This is why DeepSeek-V3, Llama-3-405B long-context, and the NVIDIA Nemotron long-context variants all use Megatron-CP rather than rolling their own.

### 7.5 CP vs TP for long sequences

| Property | CP | TP |
|----------|----|----|
| What it shards | Sequence (S/N tokens per GPU) | Weights (W/N per GPU) |
| KV cache | Sharded (S/N of KV per GPU) | Replicated (full KV per GPU) |
| When it communicates | Inside attention only | Every transformer layer |
| Scaling target | Ultra-long sequences (>128K) | Standard sequences |
| Practical N limit | Effectively unlimited | ≤ NVLink domain (≈ 8) |

**Key insight**: for long sequences, TP's KV-cache replication is a huge memory waste. With 8-way TP, every GPU stores the full 1M-token KV cache; with CP=8, every GPU stores only 125K tokens of KV.

### 7.6 Current best practice — hybrid CP

In production at 2024–2026 scale, no one uses a single CP variant in isolation. The recipe:

```
Within a node (≤ 8 GPUs):  Ulysses
   → AllToAll over NVLink is fast; head count usually sufficient.

Across nodes:              Ring / Megatron-CP
   → P2P over IB / RoCE scales; compute/comm overlap hides latency.

Composite:  hybrid Ulysses (intra-node) × Ring (inter-node)
   → Ulysses=8 × Ring=4 = 32-way CP across 4 nodes.
   → 1M-token training, 256-GPU class deployments.
```

Concrete implementations of this hybrid pattern:

- **Tencent USP** (Unified Sequence Parallel) — paper introducing the hybrid pattern explicitly.
- **Megatron-LM CP** with its `--context-parallel-size` parameter wraps both variants and auto-picks based on the comm topology.
- **PyTorch's native Context Parallel** (in newer PyTorch versions) exposes Ring + Ulysses as composable backends.
- **SGLang long-context mode** + **xDiT** (for diffusion video) both use the hybrid pattern.
- **FlashAttention-3** ships with distributed primitives matching this hybrid.

**Training vs inference.** CP is historically a *training* technique — gradient/activation memory scales with $S$ and overwhelms single GPUs first. In *inference*, the first answers are usually KV-cache compression ([[saw-int4|SAW-INT4]], MLA, KV pruning) and/or [[prefill-decode-disaggregation|PD disaggregation]] before reaching for CP. **Meta** is the one frontier lab that has publicly demonstrated CP at inference: ["Context Parallelism for Scalable Million-Token Inference" (arXiv:2411.01783)](https://arxiv.org/abs/2411.01783) shows Llama 3 405B doing 1M-token prefill in 77 s on 128 H100s via a **pass-KV / pass-Q hybrid** ring attention. Engine support today: **vLLM** ships *Decode Context Parallel (DCP)* — a narrower KV-dedup-only variant exposed via `--decode-context-parallel-size` (PR [#24864](https://github.com/vllm-project/vllm/pull/24864), Oct 2025) — but full prefill CP is still under active development. **TensorRT-LLM** exposes `context_parallel_size` as a first-class API ([docs](https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html)). **SGLang** has explicitly opted for pipeline parallelism instead of CP for million-token serving ([blog](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/), Jan 2026). **Other frontier labs (Google Gemini, Anthropic Claude, OpenAI GPT-4) have not publicly disclosed whether their long-context serving uses CP** — claims they do are unverified speculation.

### 7.7 Limitations

| Limitation | Detail |
|------------|--------|
| **Comm overhead** | If the sequence is too short, comm time can't be hidden by compute |
| **Causal triangle** | Causal attention: early-token queries don't need later KV → compute imbalance |
| **GPU-count limit (Ulysses)** | GPU count can't exceed num_heads |
| **Implementation complexity** | Modify attention impl (Ring), or inject AllToAll (Ulysses) |
| **Short sequence has no gain** | When S/N is small, comm cost > compute saving |

---

## 8. EP — Expert Parallelism

### 8.1 MoE Architecture Recap

MoE (Mixture of Experts) replaces the Transformer FFN with multiple "expert" networks + a router:

```
    input token x
         │
         ▼
    ┌─────────┐
    │  Router  │ ← gating network, selects top-K experts per token
    └────┬────┘
         │ routing decision: token → expert IDs + weights
         │
    ┌────┼────┬────┬────┬────┬────┐
    │    │    │    │    │    │    │
   E₀   E₁   E₂   E₃   E₄   E₅  E₆  E₇   ← 256 experts (e.g. DeepSeek-V3)
    │    │                   │
    ▼    ▼                   ▼
  (selected experts process the token)
    │    │                   │
    └────┴───────────────────┘
              │ weighted sum
              ▼
         output = Σ gᵢ · Expertᵢ(x)
```

**Key property**: each token activates only a few experts (e.g. DeepSeek-V3 activates 8/256), giving "huge total parameters but bounded per-token compute."

### 8.2 EP Mechanics

EP places **entire expert networks** on different GPUs. With E experts and EP_size GPUs, each GPU holds `E / EP_size` experts.

```
    Assume: 8 experts, EP_size=4

    GPU 0: [E₀, E₁]    GPU 1: [E₂, E₃]    GPU 2: [E₄, E₅]    GPU 3: [E₆, E₇]

    ┌─────────────────────────────────────────────────────────────┐
    │ Step 1: Router computes routing decisions independently on  │
    │         each GPU.                                            │
    │         token_0 → E₂, E₅   token_1 → E₀, E₇ ...             │
    │                                                             │
    │ Step 2: AllToAll DISPATCH — send tokens to GPUs holding      │
    │         the target experts.                                  │
    │                                                             │
    │   GPU 0 ──token_0──→ GPU 1 (for E₂)                          │
    │   GPU 0 ──token_0──→ GPU 2 (for E₅)                          │
    │   GPU 1 ──token_1──→ GPU 0 (for E₀)                          │
    │   GPU 1 ──token_1──→ GPU 3 (for E₇)                          │
    │                                                             │
    │ Step 3: Each GPU processes the tokens it received with local │
    │         experts.                                             │
    │                                                             │
    │ Step 4: AllToAll COMBINE — send results back to originating  │
    │         GPUs.                                                │
    │                                                             │
    │   GPU 1 ──result──→ GPU 0 (E₂'s contribution for token_0)    │
    │   GPU 2 ──result──→ GPU 0 (E₅'s contribution for token_0)    │
    │                                                             │
    │ Step 5: weighted sum                                         │
    └─────────────────────────────────────────────────────────────┘
```

### 8.3 Communication Analysis

Each MoE layer has **2 AllToAlls** (dispatch + combine):

| Metric | Formula |
|--------|---------|
| Per AllToAll volume (per GPU) | `tokens × top_k × hidden_dim × dtype_bytes × (EP-1) / EP` |
| Per MoE layer total | `2 × tokens × top_k × H × dtype × (EP-1) / EP` |

**AllToAll vs AllReduce key difference**: AllToAll's data flow is **dynamic and irregular** — exactly how much data moves between which two GPUs depends on the router's per-batch decision. "Hot" experts produce asymmetric traffic.

### 8.4 Code: EP AllToAll Routing

```python
import torch
import torch.distributed as dist

def expert_parallel_forward(
    hidden_states,    # [num_tokens, hidden_dim] — local GPU's tokens
    router_logits,    # [num_tokens, num_experts] — routing scores
    local_experts,    # nn.ModuleList — experts owned by this GPU
    ep_group,         # comm group
    num_experts,      # total experts
    top_k=2,          # experts per token
):
    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    experts_per_gpu = num_experts // ep_size

    # ---- Step 1: routing decision ----
    scores = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
    # topk_indices: [num_tokens, top_k] — selected expert IDs per token

    # ---- Step 2: group tokens by target GPU ----
    # Determine which GPU each token-expert pair goes to
    target_gpu = topk_indices // experts_per_gpu   # [num_tokens, top_k]

    # Build the AllToAll send counts
    send_counts = torch.zeros(ep_size, dtype=torch.long)
    for gpu_id in range(ep_size):
        send_counts[gpu_id] = (target_gpu == gpu_id).sum()

    # Exchange counts — each GPU learns how many tokens to receive from others
    recv_counts = torch.zeros_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)

    # ---- Step 3: AllToAll DISPATCH ----
    # Permute tokens by target GPU, then AllToAll
    # (simplified; real impl uses permutation indices)
    sorted_tokens = permute_tokens_by_target(hidden_states, topk_indices, target_gpu)
    received_tokens = all_to_all(sorted_tokens, send_counts, recv_counts, ep_group)

    # ---- Step 4: local expert compute ----
    expert_outputs = torch.zeros_like(received_tokens)
    for i, expert in enumerate(local_experts):
        local_expert_id = ep_rank * experts_per_gpu + i
        mask = (received_expert_ids == local_expert_id)
        if mask.any():
            expert_outputs[mask] = expert(received_tokens[mask])

    # ---- Step 5: AllToAll COMBINE — send results back ----
    returned_outputs = all_to_all(expert_outputs, recv_counts, send_counts, ep_group)

    # ---- Step 6: weighted sum ----
    final_output = weighted_sum(returned_outputs, topk_weights)
    return final_output
```

### 8.5 Load-Balancing Strategies

EP's biggest challenge is **load imbalance** — when some experts are picked much more often than others:

```
    Ideal (balanced):              Reality (imbalanced):
    E₀: ████  (25%)              E₀: ████████████ (60%)  ← hot expert!
    E₁: ████  (25%)              E₁: ██ (10%)
    E₂: ████  (25%)              E₂: ██ (10%)
    E₃: ████  (25%)              E₃: ████ (20%)

    → equal work per GPU            → GPU 0 becomes bottleneck, others idle
```

Mainstream fixes:

| Method | Description | Representative |
|--------|-------------|----------------|
| **Auxiliary loss** | Balance regularizer added to training loss | GShard, Switch Transformer |
| **Aux-free bias adjustment** | Dynamically adjust routing bias instead of adding loss | DeepSeek-V3 |
| **Capacity factor** | Cap max tokens per expert, drop overflow | Switch Transformer |
| **Node-limited routing** | Limit a token to at most M nodes | DeepSeek-V3 (M=4) |
| **Redundant experts** | Replicate hot experts on multiple GPUs | DeepSeek-V3 inference |

### 8.6 When to Use

- MoE models where total expert parameters exceed one GPU
- Sparsely activated models (e.g. DeepSeek-V3: 671B total, 37B active/token)
- When a single expert fits on one GPU, **EP beats ETP**

### 8.7 Limitations

| Limitation | Detail |
|------------|--------|
| **AllToAll is a sync barrier** | All GPUs must complete the token exchange before proceeding |
| **Dynamic routing → imbalance** | GPU hosting hot expert becomes the bottleneck |
| **Cross-node EP is expensive** | Needs high-bandwidth interconnect (InfiniBand) |
| **Comm grows linearly with EP degree** | Larger EP → more GPUs in the AllToAll |

---

## 9. EDP/DEP — Expert Data Parallelism

### 9.1 The idea

EDP (also written DEP) is **data parallelism applied to MoE expert layers**: multiple GPUs hold the **same** expert assignment but process **different** token batches. The key thing to understand is the **orthogonality to EP**:

- **EP dimension** (AllToAll): within an EP group, different GPUs hold different experts.
- **EDP dimension** (AllReduce): across EP groups, the entire expert assignment is **replicated** to process more data in parallel.

```
    Example: 8 experts, EP=4, EDP=2 → 4 × 2 = 8 GPUs total

    ┌──────────── EDP replica 0 ─────────────┐  ┌──────────── EDP replica 1 ─────────────┐
    │                                         │  │                                         │
    │  GPU 0    GPU 1    GPU 2    GPU 3       │  │  GPU 4    GPU 5    GPU 6    GPU 7       │
    │  [E0,E1] [E2,E3] [E4,E5] [E6,E7]       │  │  [E0,E1] [E2,E3] [E4,E5] [E6,E7]       │
    │     │        │        │        │        │  │     │        │        │        │        │
    │     └────────┴────────┴────────┘        │  │     └────────┴────────┴────────┘        │
    │           AllToAll (EP routing)         │  │           AllToAll (EP routing)         │
    │                                         │  │                                         │
    │           Processes Batch A             │  │           Processes Batch B             │
    └─────────────────────────────────────────┘  └─────────────────────────────────────────┘
            │                                            │
            └──────────── AllReduce ─────────────────────┘
              expert-gradient sync: GPU0↔GPU4, GPU1↔GPU5, GPU2↔GPU6, GPU3↔GPU7

    ★ EP intra-group: AllToAll (route tokens to the expert that owns them)
    ★ EDP inter-group: AllReduce (sync gradients of the same expert across replicas)
```

### 9.2 Communicator topology

The single most important thing about EDP is **how the communicator groups are partitioned**. For 16 GPUs with EP=4, EDP=4:

```
    GPU ID:  0   1   2   3  |  4   5   6   7  |  8   9  10  11  | 12  13  14  15
    Expert: E0  E1  E2  E3  | E0  E1  E2  E3  | E0  E1  E2  E3  | E0  E1  E2  E3
            │                │                 │                 │
            └── EP Group 0 ──┘                 └── EP Group 2 ──┘
                              └── EP Group 1 ──┘                 └── EP Group 3 ──┘

    EP groups  (AllToAll): {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}
    EDP groups (AllReduce): {0,4,8,12}, {1,5,9,13}, {2,6,10,14}, {3,7,11,15}
                              ↑ four replicas of E0       ↑ four replicas of E3
```

**Key observation**: EP groups and EDP groups are **orthogonal** — EP groups go row-wise (different experts within a row), EDP groups go column-wise (same expert replicated down a column).

### 9.3 Differences from standard DP

| Property | Standard DP | EDP |
| -------- | ----------- | --- |
| What's replicated | Whole model | MoE experts only |
| What syncs | All parameter gradients | Expert parameter gradients only |
| Scope | Dense layers (attention + MLP) | MoE layers |
| Independent of TP? | Yes | Yes (after MoE Parallel Folding) |
| When comm happens | After backward pass | After backward pass (experts only) |
| Orthogonal to | TP, PP | EP, ETP, PP |

**Historical constraint**: Before NVIDIA's **MoE Parallel Folding (2025)**, Megatron-LM required `EP ≤ DP` — EP could only be a subset of the DP group. This capped EP heavily — if DP=8, then EP ≤ 8. Parallel Folding removed this constraint, letting dense and MoE layers run completely independent parallel configurations.

### 9.4 The MoE Parallel Folding formula

```
Dense layers:  N_total = TP × SP × CP × DP × PP   (SP usually = TP)
MoE   layers:  N_total = ETP × EP × EDP × PP

Constraint: PP must be the same across layer types; everything else independent.

Worked example (128 GPUs):
  Dense:  TP=2, CP=2, PP=8  →  DP  = 128 / (2 × 2 × 8) = 4
  MoE:    ETP=1, EP=8, PP=8 →  EDP = 128 /  (1 × 8 × 8) = 2

  → Attention uses TP=2 (head sharding), CP=2 (long-context sharding), DP=4
  → MoE uses EP=8 (expert distribution) replicated across 2 EDP copies
  → Same 128 GPUs, completely different sharding on the two layer types
```

This is the unlock that made very-high-EP training (e.g. DeepSeek-V3's EP=64) feasible without forcing DP to also be 64+.

### 9.5 The "DP Attention + EP" inference pattern

EDP's most-cited *inference-side* application is **DP Attention** (a core vLLM architecture):

```
    8 GPUs, DeepSeek-R1, DP=8 + EP=8

    Attention layer (DP=8):
    ┌────────────────────────────────────────────────────────────┐
    │  GPU 0       GPU 1       GPU 2     ...     GPU 7          │
    │  Full Attn   Full Attn   Full Attn         Full Attn       │
    │  KV part 0   KV part 1   KV part 2         KV part 7       │
    │  Requests    Requests    Requests          Requests        │
    │  {0,8,16}    {1,9,17}    {2,10,18}         {7,15,23}       │
    │                                                            │
    │  ★ Each GPU independently handles different requests       │
    │  ★ KV cache partitioned by request, NOT replicated         │
    │    (vs TP mode where KV cache is fully duplicated)         │
    │  ★ Needs AllGather to assemble KV for attention compute    │
    └────────────────────────────────────────────────────────────┘

    MoE layer (EP=8):
    ┌────────────────────────────────────────────────────────────┐
    │  GPU 0       GPU 1       GPU 2     ...     GPU 7           │
    │  Experts     Experts     Experts           Experts         │
    │  {0-31}      {32-63}     {64-95}           {224-255}       │
    │                                                            │
    │  ★ AllToAll: route all GPUs' tokens to the expert owners   │
    │  ★ All GPUs' tokens mix together for routing               │
    └────────────────────────────────────────────────────────────┘

    Why this beats TP+EP at high concurrency:
    ├── KV cache not duplicated → more concurrent requests served
    ├── Attention needs no AllReduce → less communication
    └── Cost: AllGather to assemble KV — but worth it past ~256 concurrent
```

See [[#11. DP Attention — Data-Parallel Attention for MoE Inference|§11 DP Attention]] for the full inference analysis.

### 9.6 Communication volume

| Phase | Primitive | Per-GPU volume |
| ----- | --------- | -------------- |
| EP intra-group token routing (fwd) | AllToAll | `tokens × top_k × H × dtype × (EP-1)/EP` |
| EP intra-group result return (fwd) | AllToAll | Same |
| EDP inter-group gradient sync (train) | AllReduce | `2 × expert_params_per_gpu × dtype` |
| DP Attention KV assembly (inference) | AllGather | `batch × seq × kv_dim × dtype` |

**EDP's AllReduce is much smaller than full-model DP**: only expert-parameter gradients sync, not the whole model. For DeepSeek-V3, expert params are ~95 % of total (636B / 671B), but each GPU only holds `636B / EP` of them, so the per-GPU AllReduce is correspondingly smaller.

### 9.7 Communicator-group construction (code)

```python
import torch.distributed as dist

def create_edp_groups(world_size, ep_size, pp_size, etp_size=1):
    """
    Build Expert Data Parallelism communicator groups.

    world_size = ETP × EP × EDP × PP
    EDP = world_size / (ETP × EP × PP)
    """
    edp_size = world_size // (etp_size * ep_size * pp_size)
    print(f"EDP size: {edp_size} (each expert assignment replicated {edp_size}×)")

    ep_groups = []   # AllToAll communicator groups
    edp_groups = []  # AllReduce communicator groups

    for pp_rank in range(pp_size):
        base = pp_rank * (etp_size * ep_size * edp_size)

        # EP groups: within one EDP replica, GPUs holding different experts
        for edp_rank in range(edp_size):
            for etp_rank in range(etp_size):
                ranks = []
                for ep_rank in range(ep_size):
                    r = base + edp_rank * (ep_size * etp_size) + ep_rank * etp_size + etp_rank
                    ranks.append(r)
                group = dist.new_group(ranks)
                ep_groups.append((ranks, group))

        # EDP groups: GPUs holding the same expert (across replicas)
        for ep_rank in range(ep_size):
            for etp_rank in range(etp_size):
                ranks = []
                for edp_rank in range(edp_size):
                    r = base + edp_rank * (ep_size * etp_size) + ep_rank * etp_size + etp_rank
                    ranks.append(r)
                group = dist.new_group(ranks)
                edp_groups.append((ranks, group))

    return ep_groups, edp_groups


# Example: 128 GPUs, EP=8, EDP=2, PP=8, ETP=1
ep_groups, edp_groups = create_edp_groups(
    world_size=128, ep_size=8, pp_size=8, etp_size=1
)
# EP groups (AllToAll): 8 GPUs per group, 16 groups (2 EDP replicas × 8 PP stages)
# EDP groups (AllReduce): 2 GPUs per group, 64 groups (8 EP positions × 8 PP stages)
```

### 9.8 EDP vs traditional DP — visualized

```
    Pre-Parallel-Folding (Megatron-LM ≤ 2024):
    ┌──────────────────────────────────────┐
    │  DP group (say DP=8)                 │
    │  ┌─────────────────────────────┐     │
    │  │ EP sub-group  (EP ≤ DP)     │     │
    │  │ EP=4, leaving DP/EP=2       │     │
    │  │ → that's EDP=2               │     │
    │  └─────────────────────────────┘     │
    │  ★ EP forced to be a DP subset       │
    └──────────────────────────────────────┘

    MoE Parallel Folding (2025+):
    ┌─────────────────┐    ┌─────────────────┐
    │ Dense layers    │    │ MoE layers      │
    │ TP × SP × CP ×  │    │ ETP × EP × EDP × │
    │ DP × PP          │    │ PP                │
    │ (independent)    │    │ (independent)    │
    └─────────────────┘    └─────────────────┘
    ★ EP can be any size, independent of DP
    ★ E.g. Dense: TP=4, DP=4   MoE: EP=64, EDP=1
```

### 9.9 Use cases and limitations

**When EDP wins:**

| Scenario | Why |
| -------- | --- |
| EP degree < total GPUs | EDP uses the leftover GPUs to scale training throughput |
| High-concurrency inference (DP Attention) | Each GPU independently holds a KV partition; EDP scales total concurrent requests |
| Reducing EP communication pressure | Larger EDP → smaller EP groups → narrower AllToAll scope |

**Limitations:**

| Limitation | Detail |
| ---------- | ------ |
| Memory redundancy | Each EDP replica holds a full copy of the expert weights |
| Gradient sync overhead | EDP groups need AllReduce on expert gradients each step |
| Doesn't reduce per-GPU expert params | To reduce that, raise EP or ETP, not EDP |

---

## 10. ETP/TEP — Expert Tensor Parallelism

### 10.1 The idea

ETP (also written TEP) applies **tensor parallelism inside a single expert** — essentially the column-parallel + row-parallel TP recipe from §4, but applied to each expert's FFN weights individually.

**Core distinction**: TP shards attention and dense-MLP weights; **ETP shards the weights inside each MoE expert**.

```
    EP: different GPUs hold different complete experts
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │
    │ Expert 0 │  │ Expert 1 │  │ Expert 2 │  │ Expert 3 │
    │ (whole)  │  │ (whole)  │  │ (whole)  │  │ (whole)  │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘

    ETP: one expert's weights sharded across multiple GPUs
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ GPU 0    │  │ GPU 1    │  │ GPU 2    │  │ GPU 3    │
    │ Expert 0 │  │ Expert 0 │  │ Expert 1 │  │ Expert 1 │
    │ (left½)  │  │ (right½) │  │ (left½)  │  │ (right½) │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
    ←── ETP=2 ──→              ←── ETP=2 ──→
    ←──────────── EP=2 ──────────────────────→
```

### 10.2 How weights are sharded inside an expert

ETP shards each expert's FFN with the same column-parallel + row-parallel pattern as TP:

```
    Expert E0's FFN (gate_proj + up_proj + down_proj):

    Without ETP (ETP=1):
    ┌─────────────────────────────────────────────┐
    │  GPU 0 holds the whole expert E0:           │
    │                                              │
    │  gate_proj: [hidden_dim, ffn_dim]  ← whole  │
    │  up_proj:   [hidden_dim, ffn_dim]  ← whole  │
    │  down_proj: [ffn_dim, hidden_dim]  ← whole  │
    └─────────────────────────────────────────────┘

    With ETP=2:
    ┌──────────────────────┐  ┌──────────────────────┐
    │  GPU 0: E0 left half  │  │  GPU 1: E0 right half │
    │                      │  │                      │
    │  gate_proj:          │  │  gate_proj:          │
    │  [hidden, ffn/2]     │  │  [hidden, ffn/2]     │
    │  (col-parallel, L)   │  │  (col-parallel, R)   │
    │                      │  │                      │
    │  up_proj:            │  │  up_proj:            │
    │  [hidden, ffn/2]     │  │  [hidden, ffn/2]     │
    │                      │  │                      │
    │  down_proj:          │  │  down_proj:          │
    │  [ffn/2, hidden]     │  │  [ffn/2, hidden]     │
    │  (row-parallel, L)   │  │  (row-parallel, R)   │
    └──────────────────────┘  └──────────────────────┘

    ★ gate_proj, up_proj column-parallel → SiLU can be applied independently
    ★ down_proj row-parallel → needs AllReduce / ReduceScatter to sum
```

### 10.3 The full communication pipeline

ETP communication stacks on top of EP's AllToAll, forming a 6-step pipeline per MoE forward:

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │              Full forward of one MoE layer                          │
    │                                                                     │
    │  Step 1: Permutation                                                │
    │  ├── Router picks top-K experts for each token                      │
    │  └── Sort tokens by target-expert ID                                │
    │           │                                                         │
    │  Step 2: AllToAll-V (EP)                                            │
    │  ├── Send tokens to GPUs holding the target expert                  │
    │  └── Volume: tokens × top_k × H × (EP-1)/EP                         │
    │           │                                                         │
    │  Step 3: AllGather-V (ETP)                      ← ETP-only          │
    │  ├── Within ETP group, every GPU gathers full token batch           │
    │  └── Volume: received_tokens × H × (ETP-1)/ETP                      │
    │           │                                                         │
    │  Step 4: Expert compute (local)                                     │
    │  ├── Col-parallel: partial_h = SiLU(x @ gate_shard) * (x @ up_shard)│
    │  └── Row-parallel: partial_out = partial_h @ down_shard             │
    │           │                                                         │
    │  Step 5: ReduceScatter-V (ETP)                  ← ETP-only          │
    │  ├── Combine row-parallel partial results and re-shard              │
    │  └── Volume: same as Step 3                                         │
    │           │                                                         │
    │  Step 6: AllToAll-V (EP)                                            │
    │  ├── Send results back to originating GPUs                          │
    │  └── Volume: same as Step 2                                         │
    │           │                                                         │
    │  Step 7: Un-permutation + weighted sum                              │
    │                                                                     │
    │  Total comm: 2× AllToAll(EP) + AllGather(ETP) + ReduceScatter(ETP)  │
    │  Pure EP:    2× AllToAll(EP)                                        │
    │  ★ ETP adds 2 extra collective communications                       │
    └─────────────────────────────────────────────────────────────────────┘
```

### 10.4 Communication-overhead math

With $T$ tokens per micro-batch reaching the MoE layer, hidden $H$, EP degree $E$, ETP degree $P$:

| Operation | Per-GPU volume | Source |
| --------- | -------------- | ------ |
| AllToAll dispatch (EP) | $T \cdot \text{top\_k} \cdot H \cdot (E{-}1)/E$ | Token routing |
| AllToAll combine (EP) | Same | Result return |
| AllGather (ETP) | $T_{\text{recv}} \cdot H \cdot (P{-}1)/P$ | Gather full inputs |
| ReduceScatter (ETP) | $T_{\text{recv}} \cdot H \cdot (P{-}1)/P$ | Combine partial outputs |

where $T_{\text{recv}} = T \cdot \text{top\_k} / E$ is the tokens each EP position receives.

**Overhead ratio relative to pure EP**:

$$
\text{Overhead} = \frac{P-1}{P \cdot (E-1)}
$$

| $E$ | $P$ | Overhead |
| --- | --- | -------- |
| 8 | 2 | $\frac{1}{14} \approx 7$ % |
| 4 | 4 | $\frac{3}{12} = 25$ % |
| 2 | 8 | $\frac{7}{8} \approx 87$ % |

**Conclusion**: small $E$ + large $P$ is very expensive. Hence the rule of thumb: **maximize EP, minimize ETP**.

### 10.5 The local kernel (sketch)

```python
import torch
import torch.distributed as dist

def expert_tp_forward(
    dispatched_tokens,   # [T_recv, H] — tokens that landed on this ETP group
    gate_weight_shard,   # [H, ffn_dim/ETP] column-parallel slice
    up_weight_shard,     # [H, ffn_dim/ETP] column-parallel slice
    down_weight_shard,   # [ffn_dim/ETP, H] row-parallel slice
    etp_group,
):
    etp_size = dist.get_world_size(etp_group)

    # Step 1: AllGather (ETP) — every ETP rank gets the full input
    gathered = [torch.empty_like(dispatched_tokens) for _ in range(etp_size)]
    dist.all_gather(gathered, dispatched_tokens, group=etp_group)
    full_input = torch.cat(gathered, dim=0)  # [T_recv * ETP, H]

    # Step 2: column-parallel FFN compute
    gate_out = torch.nn.functional.linear(full_input, gate_weight_shard.T)
    up_out   = torch.nn.functional.linear(full_input, up_weight_shard.T)
    hidden   = torch.nn.functional.silu(gate_out) * up_out

    # Step 3: row-parallel down_proj (produces partial sums)
    partial_output = torch.nn.functional.linear(hidden, down_weight_shard.T)

    # Step 4: ReduceScatter (ETP) — combine partial sums and re-shard
    chunks = list(partial_output.chunk(etp_size, dim=0))
    output = torch.empty_like(dispatched_tokens)
    dist.reduce_scatter(output, chunks, group=etp_group)

    return output
```

### 10.6 Configuration and measured MFU

```bash
# Megatron-LM training
python pretrain_gpt.py \
    --num-experts 8 \
    --expert-model-parallel-size 4   \   # EP=4
    --expert-tensor-parallel-size 2  \   # ETP=2
    --tensor-model-parallel-size 4   \   # TP=4 (attention)
    --pipeline-model-parallel-size 2 \   # PP=2
    --sequence-parallel                  # required with TP+EP

# TensorRT-LLM inference
python convert_checkpoint.py \
    --tp_size 4  \
    --moe_tp_size 2  \   # ETP=2
    --moe_ep_size 2      # EP=2
```

**Measured MFU on Megatron-LM** (illustrative):

| Model | GPUs | Config | MFU |
| ----- | ---- | ------ | --- |
| Mixtral 8×7B | 64 | EP=8, ETP=1, TP=2, PP=4 | **49.3 %** |
| Mixtral 8×7B | 64 | EP=4, ETP=2, TP=2, PP=4 | 45.1 % |
| Qwen2-57B-A14B | 64 | EP=4, ETP=1, TP=2, PP=4 | **39.0 %** |
| Qwen2-57B-A14B | 64 | EP=2, ETP=2, TP=2, PP=4 | 35.7 % |

ETP > 1 always loses MFU; only worth it when individual experts don't fit.

### 10.7 Rule of thumb — set ETP=1 unless forced otherwise

```
    ★ Core principle: maximize EP, minimize ETP

    Fine-grained MoE (256 experts, small FFN dim, e.g. DeepSeek-V3):
    └── ETP = 1, push EP as high as possible
        └── Why: each expert is small; TP-sharding it doesn't have enough
            compute to amortize the AllGather + ReduceScatter overhead

    Coarse-grained MoE (8 experts, large FFN dim, e.g. Mixtral 8×7B):
    └── Try ETP = 1 first
        └── If OOM → ETP = 2
            └── Still OOM → ETP = 4 (last resort)

    Inference scenarios:
    └── Almost always ETP = 1
        └── No grads / optimizer state, so single GPU fits experts
            └── If still doesn't fit → quantize, don't reach for ETP

    If ETP > 1 is unavoidable:
    ├── Keep ETP group inside the NVLink domain (AG / RS need high BW)
    ├── EP × ETP ≤ 8 (one node)
    └── num_experts % (EP × ETP) == 0
```

---

## 11. DP Attention — Data-Parallel Attention (MoE Inference)

### 11.1 Background

In MoE inference, the traditional TP+EP combo has a key problem: **TP causes KV cache to be fully replicated on every GPU**.

```
    TP+EP mode (8 GPU, DeepSeek-R1):
    Attention (TP=8): every GPU stores the full KV cache → 8 copies!
    MoE (EP=8):       every GPU stores different experts → no redundancy

    Problem: KV cache memory ∝ concurrent requests × seq length
             8 copies of KV cache = 8× memory waste
             → severely caps concurrent requests
```

**DP Attention** (vLLM 2025) core idea: turn the Attention layer into data-parallel mode — **each GPU holds KV cache for only a subset of requests, not full replicas**.

### 11.2 How it works

```
DP Attention + EP architecture (8 GPUs):

┌─────────────────── Attention layer (DP=8) ──────────────────┐
│                                                              │
│  GPU 0         GPU 1         GPU 2       ...  GPU 7          │
│  ┌──────┐     ┌──────┐     ┌──────┐          ┌──────┐        │
│  │Attn  │     │Attn  │     │Attn  │          │Attn  │        │
│  │model │     │model │     │model │          │model │        │
│  │(full)│     │(full)│     │(full)│          │(full)│        │
│  ├──────┤     ├──────┤     ├──────┤          ├──────┤        │
│  │KV for│     │KV for│     │KV for│          │KV for│        │
│  │Req   │     │Req   │     │Req   │          │Req   │        │
│  │0,8,16│     │1,9,17│     │2,10  │          │7,15  │        │
│  └──────┘     └──────┘     └──────┘          └──────┘        │
│                                                              │
│  ★ Every GPU holds the full Attention parameters             │
│  ★ But only stores 1/8 of requests' KV cache → no dup!       │
│  ★ Independent compute; no comm at prefill                   │
│  ★ Decode may need AllGather across GPUs' KV (impl dependent)│
└──────────────────────────────────────────────────────────────┘
                            │
                    AllToAll (token routing)
                            │
┌─────────────────── MoE layer (EP=8) ─────────────────────────┐
│                                                              │
│  GPU 0         GPU 1         GPU 2       ...  GPU 7          │
│  Expert        Expert        Expert            Expert        │
│  {0-31}        {32-63}       {64-95}           {224-255}     │
│                                                              │
│  ★ All GPUs' tokens mix together via AllToAll routing        │
└──────────────────────────────────────────────────────────────┘
                            │
                    AllToAll (combine)
                            │
                    Attention layer (next layer)
```

### 11.3 Key differences from traditional DP

| Property | Traditional DP (training) | DP Attention (inference) |
|----------|---------------------------|--------------------------|
| Model params | Fully replicated | Fully replicated (Attention side) |
| KV cache | N/A (no KV in training) | **Partitioned**, each GPU holds 1/N of requests' KV |
| Gradient sync | AllReduce | None (no inference gradients) |
| Request assignment | Data batch split evenly | Requests round-robin across GPUs |
| Pairing with EP | Independent | Tightly coupled: DP Attention → AllToAll → EP → AllToAll → DP Attention |

### 11.4 Why this matters for MoE inference

```
Memory comparison (DeepSeek-R1, 8 GPU, 1000 concurrent requests, avg seq_len=4096):

TP=8 + EP=8:
  KV cache / GPU = 1000 × 4096 × 2 × n_layers × kv_dim × 2B
                 = 1000 full KV copies (each GPU stores all!)
  ★ Memory bottleneck: KV cache far exceeds model params

DP=8 + EP=8 (DP Attention):
  KV cache / GPU = 125 × 4096 × 2 × n_layers × kv_dim × 2B
                 = only 125 KV partitions (1000/8)
  ★ KV cache shrinks 8× → can serve 8× more concurrent requests!
```

This is why for high-concurrency MoE inference, DP Attention + EP beats TP + EP.

### 11.5 Communication pattern

```
DP Attention + EP communication flow:

  Attention layer:
    Prefill: no comm (each GPU independently serves its own requests)
    Decode:  may need AllGather KV (depends on whether requests need cross-GPU KV)
             volume: batch × seq × kv_dim × dtype

  Attention → MoE transition:
    AllToAll (dispatch): gather all GPUs' tokens and route to expert owners
    volume: total_tokens × top_k × hidden_dim × dtype × (EP-1)/EP

  MoE layer:
    Local expert compute (no comm)

  MoE → Attention transition:
    AllToAll (combine): send results back to each GPU
    volume: same as dispatch

Total: comm dominated by EP's AllToAll, not Attention
```

### 11.6 vLLM configuration and performance

```bash
# DP Attention + EP mode (vLLM)
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
    # DP=8 for attention, EP=8 for MoE

# Compare: TP + EP mode
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
    # TP=8 for attention, EP=8 for MoE
```

**Performance comparison**:

```
    Throughput (tokens/s)      DP Attention + EP vs TP + EP
    │
    │                                          DP+EP ●
    │                                       ●
    │                                    ●
    │                              ●
    │                        ●                      ← DP+EP leads
    │                  TP+EP ●─ ─ ─ ─●─ ─ ─ ─●          ~47% at high concurrency
    │              ●
    │         ●                                     ← TP+EP leads
    │    ●                                            ~52% at low concurrency
    │●
    └────────────────────────────────────────────→
    1    16   64   128  256  512  1024  2048  4096
                      concurrent requests

    Crossover: ~256-512 concurrency
```

### 11.7 Use cases and limitations

**Use cases**:
- High-concurrency MoE inference (>256 concurrent requests)
- Memory-constrained scenarios (KV cache is the bottleneck)
- Large-scale MoE models like DeepSeek-V3/R1

**Limitations**:
- Worse latency than TP+EP at low concurrency
- Attention params fully replicated (but Attention is usually much smaller than Expert params)
- Requires framework support (vLLM 2025+ natively supports it)

---

## 12. Combining Strategies

### 12.1 Total-GPU Decomposition Formulas

```
                  ┌──────────────────────────────────────────┐
                  │          N_total GPUs                     │
                  │                                          │
                  │  Dense: N = TP × SP × CP × DP × PP        │
                  │         (SP usually = TP)                 │
                  │  MoE  : N = ETP × EP × EDP × PP           │
                  │                                          │
                  │  Constraint: PP shared between both       │
                  └──────────────────────────────────────────┘
```

### 12.2 Communication-Pattern Overview

```
    ┌──────────────────────────────────────────────────────────────────┐
    │                     One Transformer block                        │
    │                                                                  │
    │  ┌─────────────────┐     ┌──────────────────────────────────┐   │
    │  │  Attention layer │     │            MoE layer              │   │
    │  │                 │     │                                    │   │
    │  │  TP: AllReduce  │     │  EP:  AllToAll (dispatch)         │   │
    │  │  SP: AG + RS    │     │       Expert compute              │   │
    │  │  CP: Ring/A2A   │     │       AllToAll (combine)          │   │
    │  │  DP: (none, inf)│     │  ETP: AllGather + ReduceScatter   │   │
    │  │      AllReduce  │     │  EDP: AllReduce (training grads)  │   │
    │  │      (train grad)│     │                                    │   │
    │  │                 │     │                                    │   │
    │  │  Comm freq: per │     │  Comm freq: per MoE layer         │   │
    │  │      layer (high)│     │  BW need: high (AllToAll volume)  │   │
    │  │  BW need: very  │     │                                    │   │
    │  │     high        │     │                                    │   │
    │  └─────────────────┘     └──────────────────────────────────┘   │
    │                                                                  │
    │  PP: only stage-boundary P2P (lowest frequency)                  │
    └──────────────────────────────────────────────────────────────────┘
```

### 12.3 Typical Combos

#### Pattern A: TP + EP (low-concurrency inference)

```
    8 GPU (1 node, NVLink)

    Attention: TP=8 (sharded across 8 GPUs)
    MoE:       EP=8 (8 expert groups, num_experts/8 experts per GPU)

    Profile: lowest single-request latency
    Comm: Attention uses AllReduce, MoE uses AllToAll
    Fit: latency-sensitive, concurrency < 256
```

```bash
# vLLM config
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

#### Pattern B: DP Attention + EP (high-concurrency inference)

```
    8 GPU (1 node, NVLink)

    Attention: DP=8 (each GPU handles different requests, KV partitioned)
    MoE:       EP=8 (experts spread across 8 GPUs)

    Profile: highest throughput, no KV duplication
    Comm: Attention uses AllGather (KV), MoE uses AllToAll
    Fit: high concurrency, > 512 concurrent requests
```

```bash
# vLLM config
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel
```

#### Pattern C: TP intra-node + PP inter-node

The most common hybrid: TP within an NVLink domain, PP across nodes.

```
Node 0 (8 GPU, NVLink)              Node 1 (8 GPU, NVLink)
┌────────────────────────────┐    ┌────────────────────────────┐
│ GPU0  GPU1  GPU2  GPU3     │    │ GPU0  GPU1  GPU2  GPU3     │
│ ◄── TP=4, Stage 0 ──▶     │    │ ◄── TP=4, Stage 2 ──▶     │
│                            │    │                            │
│ GPU4  GPU5  GPU6  GPU7     │    │ GPU4  GPU5  GPU6  GPU7     │
│ ◄── TP=4, Stage 1 ──▶     │    │ ◄── TP=4, Stage 3 ──▶     │
└────────────────────────────┘    └────────────────────────────┘
              │ PP (cross-node, InfiniBand)│
              └──────────────────────────┘

Config: TP=4, PP=4, total GPUs = 4×4 = 16
```

#### Pattern D: PP + EP (large-scale MoE training)

DeepSeek-V3 training config (2048 H800):

```
DeepSeek-V3 Training:
  PP = 16 (cross-node pipeline, DualPipe schedule)
  EP = 64 (experts spread across 64 GPUs over 8 nodes)
  DP = 2  (ZeRO-1 data parallelism)
  TP = 1  (no tensor parallelism! MLA's KV dim is small, TP overhead > benefit)

  Uses DualPipe for PP, overlapping AllToAll comm
  Node-limited routing (M=4) cuts cross-node traffic ~50%
```

#### Pattern E: 4D/5D parallelism

Full multi-dim parallelism: TP × PP × CP × DP (+EP for MoE).

```
4D parallelism example: 128 GPUs

    ┌──────────────── PP=4 (cross-node) ──────────────────┐
    │                                                      │
    │  ┌─── TP=4 (intra-node NVLink) ───┐  × CP=2  × DP=4 │
    │  │ GPU0 GPU1 GPU2 GPU3            │                  │
    │  └─────────────────────────────────┘                  │
    │                                                      │
    └──────────────────────────────────────────────────────┘

Total GPUs = TP × PP × CP × DP = 4 × 4 × 2 × 4 = 128
```

### 12.4 TP × CP composition explained

The most common confusing combo: "TP shards heads, CP shards sequence — how do these stack?" The answer is **they're orthogonal axes of a 2-D GPU grid**, and the two communication groups never overlap during a single forward pass.

**Example: 8 GPUs as TP=2 × CP=4.**

```
         CP rank 0    CP rank 1    CP rank 2    CP rank 3
        ┌──────────┬──────────┬──────────┬──────────┐
TP rank 0│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │
        ├──────────┼──────────┼──────────┼──────────┤
TP rank 1│  GPU 4   │  GPU 5   │  GPU 6   │  GPU 7   │
        └──────────┴──────────┴──────────┴──────────┘
```

At attention input the tensor is `(B, S, H, D)`. Two shardings stack:

- **TP shards $H$**: each row of the grid (same TP rank) holds $H/2$ heads.
- **CP shards $S$**: each column (same CP rank) holds $S/4$ tokens.

Each GPU owns a slab of shape `(B, S/4, H/2, D)`.

**Two independent communication groups:**

```
TP groups (vertical, 2 members each):    Used for TP AllReduce
  {GPU 0, GPU 4}, {GPU 1, GPU 5},
  {GPU 2, GPU 6}, {GPU 3, GPU 7}

CP groups (horizontal, 4 members each):  Used for Ring P2P or AllToAll
  {GPU 0, GPU 1, GPU 2, GPU 3}            (TP rank 0 row)
  {GPU 4, GPU 5, GPU 6, GPU 7}            (TP rank 1 row)
```

**One transformer layer's attention block — execution flow:**

```
1. Input LayerNorm + Dropout (SP):
     Shards along sequence dim (compatible with CP, doesn't conflict).

2. QKV projection (TP region):
     Each GPU has input (B, S/4, D) → multiply by its W_qkv slice (column-parallel)
     → (B, S/4, H/2 × D × 3). No communication.

3. Attention computation (CP region):
     Each GPU holds (B, S/4, H/2, D) of Q, K, V.
     Run Ring Attention or Ulysses *within its CP group of 4 GPUs*.
     Output: (B, S/4, H/2, D).
     → All CP communication is intra-row only. TP is not involved here;
       the H/2 head subset is each TP rank's private data.

4. Output projection (TP region):
     Row-parallel TP matmul → AllReduce across the TP group of 2 GPUs.
     → All TP communication is intra-column only. CP is not involved here;
       the sequence shard stays put.

5. Residual + LayerNorm + FFN (TP + SP) → next layer.
```

**Key observation: each GPU is in both groups, but only one group communicates at a time.** During the CP attention phase, GPU 0 only talks to {GPU 1, 2, 3} — not GPU 4. During the TP output-projection phase, GPU 0 only talks to GPU 4 — not GPU 1/2/3. The two groups operate on orthogonal axes of the same forward pass.

**Why this works mathematically.** Multi-head attention is *embarrassingly parallel across heads* — different heads $h$ never interact during attention. So TP-splitting heads needs no comm inside attention. Tokens, in contrast, *all interact* (every query reads every KV), so CP-splitting sequence needs comm inside attention. **The required-comm locations of the two axes don't overlap**, which is exactly the property that lets them compose freely.

**Generalization.** In production with 5-D parallelism (TP × PP × DP × CP × EP), each GPU sits at a coordinate in a 5-D grid; each parallel axis defines an independent NCCL communicator group, and every layer's forward chooses which axis to communicate on based on which operator is running. Different layers / phases use different groups; nothing ever needs two groups simultaneously.

### 12.5 vLLM TP+EP vs DP+EP Performance

```
    Throughput (tokens/s)
    │
    │                                          DP=8+EP ●
    │                                       ●
    │                                    ●
    │                              ●
    │                        ●
    │                  TP=8+EP ●─ ─ ─ ─●─ ─ ─ ─●─ ─ ─ ─ ─●
    │              ●
    │         ●
    │    ●
    │●
    └────────────────────────────────────────────────────→
    1    16   64   128  256  512  1024  2048  4096
                      concurrent requests

    ★ Crossover at ~256-512 concurrency
    ★ Low concurrency: TP+EP lower latency (52% throughput advantage at concurrency=64)
    ★ High concurrency: DP+EP higher throughput (47% throughput advantage at concurrency=1024)
```

### 12.6 Notes on Mixing

| Note | Why |
|------|-----|
| **SP is mandatory with TP + EP** | Megatron-LM hard requirement: `--sequence-parallel` |
| **EP × ETP should fit in NVLink domain** | ETP's AllGather/ReduceScatter need high BW |
| **PP must match across layer types** | Dense and MoE share the same PP partition |
| **num_experts % EP == 0** | Expert count must be divisible by EP degree |
| **AllToAll and AllReduce can't fuse** | Different comm patterns may contend for network bandwidth |
| **Expert activation density affects EP gain** | Density > 3% favors EP; < 1% AllToAll cost may exceed gain |

> **Expert activation density** = `experts_per_token / total_routed_experts × 100%`
> - DeepSeek-V3: 8/256 = 3.1% → good fit for EP
> - Llama-4-Maverick: 1/128 ≈ 0.8% → EP gain questionable

### 12.7 TP × CP composition (extended)

The most confusing combo: "TP splits heads, CP splits sequence — how do they stack?" Answer: **they're orthogonal axes in a GPU 2D grid**, and during a single forward pass the two communicator groups never communicate simultaneously.

(See §12.4 above for the full walk-through; this section is the conceptual extension.)

---

## 13. Selection Guide

```
                            Start
                              │
                    ┌─────────▼──────────┐
                    │ Model fits on 1 GPU?│
                    └─────────┬──────────┘
                         Yes ╱   ╲ No
                          ╱       ╲
                    ┌────▼───┐  ┌──▼─────────────────┐
                    │ Use DP │  │ NVLink available?    │
                    │simplest│  └──┬─────────────────┘
                    │(or ZeRO│  Yes╱    ╲No
                    │mem-save)│  ╱        ╲
                    └────────┘
                    ┌──────────▼───┐  ┌───▼──────────┐
                    │ TP ≤ 8       │  │ PP (cross-node)│
                    │ (NVLink dom) │  │ + DP (cross-nd)│
                    │ + SP (enable)│  └──────────────┘
                    └──────┬───────┘
                           │
                    ┌──────▼───────────────────┐
                    │ Need more GPUs?            │
                    └──────┬───────────────────┘
                      Yes ╱     ╲ No
                       ╱         ╲
              ┌───────▼──────┐  ┌─▼────────────┐
              │ Add PP cross │  │ Add DP for    │
              │ + CP long-seq│  │ throughput or │
              └───────┬──────┘  │ ZeRO mem-save │
                      │         └──────────────┘
              ┌───────▼──────────────────────┐
              │ MoE model?                    │
              └───────┬──────────────────────┘
                 Yes ╱     ╲ No
                  ╱         ╲
         ┌──────▼──────┐  ┌─▼────────────┐
         │ Add EP for  │  │ Remaining GPUs│
         │ experts     │  │ → DP for batch│
         └───────┬──────┘  └──────────────┘
                 │
         ┌───────▼──────────────────────┐
         │ Single expert fits on 1 GPU? │
         └───────┬──────────────────────┘
            Yes ╱     ╲ No
             ╱         ╲
     ┌──────▼──────┐ ┌──▼──────────┐
     │ ETP=1       │ │ ETP=2 or 4  │
     │ maximize EP │ │ (last resort)│
     └──────┬──────┘ └─────────────┘
            │
     ┌──────▼──────────────────────┐
     │ Inference or training?       │
     └──────┬──────────────────────┘
       Inf ╱     ╲ Train
          ╱         ╲
  ┌──────▼──────┐ ┌──▼──────────────┐
  │ High concur?│ │ Remaining → EDP │
  │ Y → DP Attn │ │ + ZeRO-1/FSDP  │
  │    +EP      │ │ sync expert grad│
  │ N → TP+EP   │ └─────────────────┘
  └─────────────┘
```

### 13.1 Hardware Requirements

| Strategy | Minimum interconnect | Recommended |
|----------|---------------------|-------------|
| TP / SP | NVLink (600+ GB/s) | NVLink 4.0 (900 GB/s) |
| EP (intra-node) | NVLink | NVLink |
| EP (cross-node) | InfiniBand (200+ Gbps) | IB NDR (400 Gbps) |
| PP | Any (P2P, low frequency) | 100 Gbps Ethernet is fine |
| CP (Ring) | Moderate (overlapped) | 200+ Gbps |
| CP (Ulysses) | High (blocking comm) | NVLink |
| DP / EDP | Any (low-frequency grad sync) | Depends on model size |
| ZeRO-3 / FSDP | Med-High (per-layer comm) | IB 200+ Gbps |

### 13.2 Key Rules of Thumb

1. **TP stays inside the NVLink domain** — cross-node TP is rarely viable
2. **Enable SP with TP** — saves activation memory, mandatory with TP+EP
3. **Maximize EP, minimize ETP** — EP has much better comm efficiency than ETP
4. **PP for cross-node scaling** — low-frequency P2P, lowest BW need
5. **CP for ultra-long sequences** — consider when S > 128K
6. **DP/EDP for throughput** — add DP once the model fits
7. **ZeRO-1 before ZeRO-3** — same comm, much better memory
8. **Inference: TP+EP or DP Attention+EP by concurrency** — crossover ~256-512

---

## 14. Case Study: DeepSeek-V3

### 14.1 Model Overview

- Total params: 671B, 37B activated/token
- 61 Transformer layers, 256 routed + 1 shared expert per layer
- Top-8 routing
- Uses MLA (Multi-head Latent Attention); KV compressed to 512 dims

### 14.2 Training Config (2048 H800 GPU)

```
    256 nodes × 8 GPU/node = 2048 GPU
    Intra-node: NVLink (160 GB/s)
    Inter-node: InfiniBand (50 GB/s)

    ┌─────────────────────────────────────────────────────┐
    │ PP = 16    (16 pipeline stages, DualPipe schedule)  │
    │ EP = 64    (across 8 nodes, 256/64 = 4 experts/GPU) │
    │ DP = ZeRO-1 (optimizer-state shard)                 │
    │ TP = 1     (no tensor parallelism!)                 │
    └─────────────────────────────────────────────────────┘

    Why TP=1?
    ├── MLA's KV dim is only 512; TP-sharded KV cache saves little
    ├── TP does 2 AllReduces per layer; cost across 8 nodes too high
    └── EP+PP+DP already sufficient, and DualPipe hides comm
```

### 14.3 DualPipe Schedule for PP=16

DeepSeek-V3 uses PP=16 bidirectional pipeline — the largest known DualPipe deployment:

```
    PP=16: 16 stages, ~4 layers per stage

    DualPipe bubble: (PP/2 - 1) × T_{F&B} = 7 × T_{F&B}
    Compare 1F1B:    (PP - 1) × (T_F + T_B) = 15 × (T_F + T_B)

    Because DualPipe fully overlaps compute and comm:
    T_{F&B} ≈ max(T_compute, T_comm) < T_F + T_B

    Effect:
    ├── Bubble drops from ~15 time units to ~7 (>50% reduction)
    ├── AllToAll comm fully hidden
    └── Cost: 2× parameter memory (both streams keep a copy)

    PP=16 × EP=64 interaction:
    ├── Each PP stage has ~4 layers, each with EP AllToAll
    ├── DualPipe overlaps those AllToAlls with other stages' compute
    ├── Custom layout: "Et*3|(tt|)*29,m|L"
    │   E = embedding, t = transformer layer, m = MTP, L = LM head
    └── VPP=2: each GPU owns 2 non-contiguous chunks
```

### 14.4 Node-Limited Routing

```
    EP=64 spans 8 nodes

    No restriction:
    token → may go to all 8 nodes → heavy cross-node traffic

    Node limit M=4:
    token → at most 4 nodes → cross-node traffic cut ~50%
    8 selected experts spread across ≤4 nodes → ~2 experts per node

    Comm volume upper bound = M × IB_bandwidth = 4 × 50 GB/s = 200 GB/s
```

### 14.5 Custom Communication Kernels

```
    H800 GPU: ~132 SMs
    ├── 112 SMs: for compute (Attention, MLP, ...)
    └── 20 SMs: dedicated to comm
        ├── 10 comm channels
        │   ├── IB send/recv
        │   ├── IB → NVLink forwarding
        │   └── NVLink transfer
        │
        └── enough to saturate:
            ├── IB: 50 GB/s ✓
            └── NVLink: 160 GB/s ✓
```

### 14.6 Inference Config

| Config | Prefill (32 GPU, 4 nodes) | Decode (320 GPU, 40 nodes) |
|--------|---------------------------|----------------------------|
| Attention | TP=4 + SP, DP=8 | TP=4 + SP, DP=80 |
| MoE | EP=32 | EP=320 (~1 expert/GPU) |
| Redundant experts | 32 (one hot expert copy per GPU) | — |
| Experts per GPU | 256/32 = 8 + 1 redundant | 256/320 ≈ 1 |

```
    Prefill phase (32 GPU):

    ┌─── Node 0 ──────────────────────────────────────┐
    │ GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  │
    │ TP=4 group   TP=4 group                         │
    │ ├G0─G1─G2─G3┤├G4─G5─G6─G7┤                     │
    └─────────────────────────────────────────────────┘
    ┌─── Node 1 ──┐ ┌─── Node 2 ──┐ ┌─── Node 3 ──┐
    │ same        │ │ same        │ │ same        │
    └─────────────┘ └─────────────┘ └─────────────┘

    All 32 GPUs form a single EP=32 MoE expert group
    DP=8: eight TP=4 Attention replicas
```

### 14.7 LMSYS 96 H100 Deployment

```
    12 nodes × 8 GPU = 96 H100

    Prefill: EP=32, 4 nodes
    Decode:  EP=72, 9 nodes
    DP Attention: KV cache partitioning

    Performance:
    ├── 52.3k input tokens/sec/node
    ├── 22.3k output tokens/sec/node
    └── Cost: ~$0.20/1M output tokens
```

---

## 15. Full Comparison

| Dimension | DP | ZeRO/FSDP | TP | SP | PP | CP | EP | EDP | ETP | DP Attn |
|-----------|----|-----------|----|----|----|----|----|----|----|---------|
| **What it shards** | Data batch | Data + optim/grad/params | Weight matrices | Seq (non-TP) | Layer groups | Seq (attention) | Whole experts | MoE data batches | Expert weight matrices | KV-cache partitions |
| **Scope** | Dense | Dense | Dense | Dense | All | Attention | MoE | MoE | MoE | Inference Attn |
| **Collective** | AllReduce | AG + RS | AllReduce | AG + RS | Send/Recv | Ring/A2A | AllToAll | AllReduce | AG + RS | AllGather |
| **Comm freq** | 1×/step | many/layer | 2-4×/layer | 2×/layer | stage boundary | per attention | 2×/MoE | 1×/step | 2×/MoE | per attention |
| **BW need** | Low | Med-High | Very High (NVLink) | Very High (=TP) | Low (PCIe) | Med-High | High (IB) | Low-Med | Very High (NVLink) | Med |
| **Memory** | Low (full copy) | High (16P/N) | High (1/N) | Higher (act/N) | High (1/PP) | High (KV/N) | High (1/EP) | Low (copy) | High (1/ETP) | High (KV/N) |
| **Scaling** | Excellent | Excellent | ≤8 | =TP | Good | Good | Good | Excellent | ≤8 | Good |
| **Use phase** | Train + inf | Train | Train + inf | Train + inf | Train + inf | Train + inf | Train + inf | Train + inf | Train + inf | Inference |
| **Recommendation** | ★★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★★★ | ★★ (prefer =1) | ★★★★ |

---

## 16. PP Schedule Performance Comparison

Assume p=8 stages, m=32 micro-batches, T_F = T_B/2 = T_W:

| Schedule | Bubble rate | Normalized throughput |
|----------|-------------|-----------------------|
| **GPipe** | 17.9% | 1.00× |
| **1F1B** | 17.9% | 1.00× (better memory) |
| **Interleaved 1F1B (v=2)** | 10.4% | ~1.09× |
| **ZB-H1** | ~6% | ~1.14× |
| **ZB-H2** | ~1% | ~1.20× |
| **DualPipe** | ~4% + full comm overlap | ~1.25× (MoE scenarios) |

**PP vs TP quick decision table**:

| Factor | Choose TP | Choose PP |
|--------|-----------|-----------|
| Interconnect | NVLink (900 GB/s) | PCIe (64 GB/s) or network |
| Priority metric | Low latency (TTFT, TPOT) | High throughput (tokens/s) |
| Concurrency | Low (<200 req) | High (>500 req) |
| GPU count | ≤8 (single node) | >8 (multi-node) |
| Model type | Dense | MoE (PP + EP) |

---

## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)
- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (Huang et al., 2019)](https://arxiv.org/abs/1811.06965)
- [PipeDream: Generalized Pipeline Parallelism (Narayanan et al., 2019)](https://arxiv.org/abs/1806.03377)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [Megatron-LM v2: Reducing Activation Recomputation (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198) — Sequence Parallelism
- [Efficient Large-Scale Language Model Training on GPU Clusters (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473) — Interleaved 1F1B
- [Zero Bubble Pipeline Parallelism (Qi et al., ICLR 2024)](https://arxiv.org/abs/2401.10241)
- [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437) — DualPipe
- [Ring Attention with Blockwise Transformers (Liu et al., ICLR 2024)](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses: Extreme Long Sequence Transformer (Jacobs et al., 2023)](https://arxiv.org/abs/2309.14509)
- [A Unified Sequence Parallelism Approach (2024)](https://arxiv.org/abs/2405.07719) — Hybrid Ring+Ulysses
- [MoE Parallel Folding (NVIDIA, 2025)](https://arxiv.org/abs/2504.14960)
- [DeepSpeed-TED: Efficient MoE via Tensor-Expert-Data Parallelism (2023)](https://arxiv.org/abs/2303.06318)
- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [vLLM MoE Playbook (AMD/ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [LMSYS: Large-Scale EP Deployment](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- [JAX Scaling Book - Training Chapter](https://jax-ml.github.io/scaling-book/training/)
- [NVIDIA Hybrid-EP Blog](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)
- [Meta: Scaling LLM Inference (2025)](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [TensorRT-LLM Expert Parallelism](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html)
- [PyTorch FSDP Documentation](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch Pipeline Parallelism Documentation](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)

---

## Related Pages

- [[vllm]] — vLLM inference framework, implements DP Attention+EP / TP+EP
- [[prefill-decode-disaggregation]] — Prefill-decode disaggregated deployment
- [[distributed-training]] — Training-side parallelism (related but different focus)
- [[quantization]] — Reduce parallelism needs via quantization
