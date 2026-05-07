---
title: "LLM Parallelism Strategies Complete Guide: DP / TP / PP / SP / CP / EP / EDP / ETP"
category: llm-inference
tags: [tensor-parallelism, data-parallelism, expert-parallelism, pipeline-parallelism, sequence-parallelism, context-parallelism, moe, multi-gpu, distributed-inference, distributed-training]
created: 2026-04-14
updated: 2026-05-07
status: mature
---

# LLM Parallelism Strategies Complete Guide: DP / TP / PP / SP / CP / EP / EDP / ETP

> [!abstract]+ TL;DR
> A comprehensive guide to the eight parallelism strategies used to scale LLMs across GPUs: **DP** (data), **TP** (tensor / intra-layer weights), **PP** (pipeline / inter-layer), **SP** (sequence in non-TP regions), **CP** (context / sequence in attention), **EP** (expert / MoE), **EDP** (expert data), **ETP** (expert tensor), plus **ZeRO/FSDP** (sharded data) and **DP Attention** (inference KV partitioning). Covers what each shards, the core collective primitive, and how to compose them in 3D / 4D / 5D parallel training and inference. Production-grade case study: DeepSeek-V3 deployment.

## 1. Overview

| Abbrev           | Alias   | Full Name                 | What It Shards                    | Scope               | Core Collective                       |
| ---------------- | ------- | ------------------------- | --------------------------------- | ------------------- | ------------------------------------- |
| **DP**           | —       | Data Parallelism          | Data batches                      | Dense layers        | AllReduce                             |
| **ZeRO/FSDP**    | —       | Sharded Data Parallelism  | Data + optimizer/gradient/params  | Dense layers        | AllGather + ReduceScatter             |
| **TP**           | —       | Tensor Parallelism        | Weight matrices within layers     | Dense layers        | AllReduce / AllGather + ReduceScatter |
| **SP**           | —       | Sequence Parallelism      | Sequence dim (non-TP regions)     | Dense layers        | AllGather + ReduceScatter             |
| **PP**           | —       | Pipeline Parallelism      | Consecutive layer groups          | All layers          | Point-to-point Send/Recv              |
| **CP**           | —       | Context Parallelism       | Sequence dim (attention)          | Attention layers    | Ring P2P / AllToAll                   |
| **EP**           | —       | Expert Parallelism        | Whole MoE experts                 | MoE layers          | AllToAll (token routing)              |
| **EDP**          | **DEP** | Expert Data Parallelism   | MoE data batches                  | MoE layers          | AllReduce (expert gradient sync)      |
| **ETP**          | **TEP** | Expert Tensor Parallelism | Individual expert weight matrices | MoE layers          | AllGather + ReduceScatter             |
| **DP Attention** | —       | Data-Parallel Attention   | KV Cache partitions               | Inference Attention | AllGather                             |

DP and TP are classic strategies for dense models; EP / EDP / ETP are their MoE-layer counterparts. PP partitions by layers, SP and CP both operate on the sequence dimension but in different scopes, ZeRO/FSDP optimizes DP memory efficiency, and DP Attention is an inference-specific KV cache partitioning scheme.

With **MoE Parallel Folding** (NVIDIA, 2025), dense and MoE layers are fully decoupled:
- Dense: `N_total = TP × SP × CP × DP × PP` (SP usually = TP)
- MoE: `N_total = ETP × EP × EDP × PP`
- Only PP must match between both.

---

## 2. DP — Data Parallelism

Replicate the entire model on every GPU; each processes a different micro-batch. Gradients synchronized via AllReduce after each step.

**Communication**: Ring AllReduce volume per GPU ≈ `2 × P × sizeof(dtype)` (P = parameter count). With ZeRO/FSDP, converted to per-layer AllGather + ReduceScatter for memory savings.

**Use when**: Model fits on one GPU; need to scale throughput.

**Limitations**: Memory redundancy (full model per GPU); communication ∝ model size.

---

## 3. ZeRO / FSDP — Sharded Data Parallelism

### 3.1 The Memory Problem

Standard DP with Adam on FP16 mixed precision requires `16P` bytes per GPU (2P params + 2P grads + 12P optimizer states). ZeRO shards these across the DP group.

### 3.2 Three Stages

| Stage | Per-GPU Memory | What's Sharded | Comm Volume |
|-------|---------------|----------------|-------------|
| Standard DP | `16P` | Nothing | `2P` |
| ZeRO-1 | `4P + 12P/N` | Optimizer states | `2P` (same) |
| ZeRO-2 | `2P + 14P/N` | + Gradients | `2P` (same) |
| ZeRO-3 / FSDP | `16P/N` | + Parameters | `3P` (+50%) |

### 3.3 Communication Pattern

```
Standard DP:  backward → AllReduce(grads)                    2P/GPU, 1x/step
ZeRO-1:      backward → ReduceScatter → update → AllGather  2P/GPU, 2x/step
ZeRO-3/FSDP: forward: AllGather(params/layer) → compute → free
              backward: AllGather → grad → ReduceScatter     3P/GPU, per-layer
```

### 3.4 FSDP (PyTorch ZeRO-3)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)  # ZeRO-3
# or ShardingStrategy.SHARD_GRAD_OP for ZeRO-2
```

### 3.5 When to Use Which Stage

- Model+Adam fits on 1 GPU → standard DP
- Optimizer states are bottleneck → ZeRO-1 (preferred, no extra comm)
- Still OOM → ZeRO-2
- Still OOM → ZeRO-3/FSDP (+50% comm overhead)
- Still OOM → combine with TP/PP

---

## 4. TP — Tensor Parallelism

Splits weight matrices within each layer (Megatron-LM, 2019). Column-parallel for gate/up projections, row-parallel for down projection. Attention heads distributed across GPUs.

**Communication**: 2 AllReduce per transformer layer (forward); 4 total with backward. Volume per AllReduce: `2 × B × S × D × dtype_bytes`.

**Efficiency**: TP=2: 85-95%, TP=4: 70-85%, TP=8: 56-75%. Requires NVLink. Keep TP within one NVLink domain.

---

## 5. SP — Sequence Parallelism

### 5.1 Motivation

TP shards weight matrices, reducing activation memory in TP regions (Attention, MLP matmuls). However, **LayerNorm** and **Dropout** are outside TP scope -- each GPU holds full `[B, S, D]` activations for these ops, wasting >50% of activation memory.

### 5.2 How It Works

SP (Megatron-LM v2, Korthikanti et al., 2022) distributes non-TP operations along the **sequence dimension**:

```
Transformer Layer with TP + SP:

  [SP region]           [TP region]           [SP region]
  LayerNorm             Attention/MLP         Dropout
  per GPU: [B,S/TP,D]   per GPU: [B,S,D/TP]  per GPU: [B,S/TP,D]
       |                      |                    |
       └── AllGather(seq) ───→|                    |
                              └── ReduceScatter ──→|

  AllReduce = AllGather + ReduceScatter → total comm volume unchanged
  But: LayerNorm/Dropout activation memory reduced from B×S×D to B×(S/TP)×D
```

### 5.3 Key Points

- **Memory savings**: Non-TP region activations reduced by `1/TP`
- **Total comm unchanged**: Just decomposing AllReduce into AllGather + ReduceScatter
- **Mandatory with TP+EP**: Megatron-LM requires `--sequence-parallel` when both TP and EP are used
- **SP = TP in size**: SP degree always equals TP degree

### 5.4 SP vs CP

| | SP | CP |
|--|----|----|
| Shards | LayerNorm, Dropout (non-TP regions) | Attention (QKV interaction) |
| Relation to TP | Complement, same group | Independent dimension |
| Comm | AllGather + ReduceScatter | Ring P2P or AllToAll |
| Goal | Reduce activation memory | Support ultra-long sequences |

---

## 6. PP — Pipeline Parallelism

### 6.1 Basic Concept

PP assigns consecutive layer groups to different GPUs. Communication: point-to-point Send/Recv at stage boundaries only.

```
GPU 0 (Layers 0-7) --send--> GPU 1 (Layers 8-15) --send--> GPU 2 (Layers 16-23) --send--> GPU 3 (Layers 24-31)
```

**Advantages**: Low communication (only at stage boundaries), works over PCIe/network, each GPU holds 1/PP of parameters.

**Core problem**: Pipeline bubbles -- GPUs idle while waiting.

### 6.2 Scheduling Strategies

| Schedule | Bubble Time | Notes |
|----------|-------------|-------|
| GPipe | `(p-1) × (T_F + T_B)` | All-F then all-B; m >= 4p for <20% bubble |
| 1F1B | `(p-1) × (T_F + T_B)` | Same bubble, lower memory (p vs m micro-batches) |
| Interleaved 1F1B | `(p-1) × (T_F + T_B) / v` | v = virtual stages; halves bubble at v=2 |
| ZB-H1 | `(p-1) × T_W` | ~1/3 of 1F1B; defers W to fill bubbles |
| ZB-H2 | ~0 | Needs more memory |
| ZB-V | 0 (when T_F = T_B = T_W) | V-shaped dependency, 2 virtual stages |

### 6.3 DualPipe (DeepSeek-V3)

Bidirectional pipeline for MoE models. Feeds micro-batches from **both ends** simultaneously. Decomposes each chunk into 4 components: ATTN (compute), DISPATCH (AllToAll comm), MLP (compute), COMBINE (AllToAll comm). Overlaps one micro-batch's communication with another's computation.

```
Overlap within a forward+backward pair:
  Step 1: ATTN(fwd) [compute] + COMBINE(bwd) [comm]
  Step 2: DISPATCH(fwd) [comm] + MLP(bwd) [compute]
  Step 3: MLP(fwd) [compute] + DISPATCH(bwd) [comm]
  Step 4: COMBINE(fwd) [comm] + ATTN(bwd) [compute]
```

Bubble: `(p/2 - 1) × T_{F&B}` -- roughly 50%+ reduction vs 1F1B. Costs 2x parameter memory.

### 6.4 PP for Inference

**When to use PP over TP**:
1. PCIe-only systems (TP has 40-50% comm overhead without NVLink)
2. Cross-node deployment
3. High-concurrency throughput optimization
4. Cost optimization (no NVLink needed)

Single-request latency: Always worse with PP (sequential stages). At high concurrency, pipeline stays full.

### 6.5 PP vs TP

| Factor | Choose TP | Choose PP |
|--------|-----------|-----------|
| Interconnect | NVLink (900 GB/s) | PCIe (64 GB/s) or network |
| Priority | Low latency | High throughput |
| Concurrency | Low (<200 req) | High (>500 req) |
| GPU count | <=8 (single node) | >8 (multi-node) |

---

## 7. CP — Context Parallelism

### 7.1 Why CP

At 1M+ token sequences, KV cache alone can exceed single-GPU memory. Attention computation is O(S^2). CP splits the **sequence dimension** across GPUs.

### 7.2 Ring Attention

Splits sequence into N segments. Each GPU holds its Q chunk permanently; KV blocks rotate around a ring. Communication overlaps with computation -- when T_compute >= T_communicate, communication is fully hidden.

Meta achieved 1M tokens in <1 minute on a single H100 node; 10M tokens on 32 hosts.

### 7.3 Ulysses (DeepSpeed)

Uses AllToAll to transform from (all heads, partial sequence) to (partial heads, full sequence) before attention, then reverses after.

| Feature | Ring Attention | Ulysses |
|---------|---------------|---------|
| Communication | P2P ring (N-1 rounds) | 2x AllToAll |
| Overlap with compute | Yes | No (blocking) |
| Intrusiveness | High (modify attention kernel) | Low |
| GPU count limit | None | N <= num_attention_heads |

**Hybrid**: Ulysses intra-node + Ring Attention inter-node.

### 7.4 CP vs TP for Long Sequences

CP shards the KV cache (each GPU stores S/N), while TP replicates it. For long sequences, CP is far more memory-efficient. CP also scales beyond the NVLink domain.

---

## 8. EP — Expert Parallelism

Distributes complete expert networks across GPUs. Each GPU holds `E / EP_size` experts. Three-phase forward: AllToAll dispatch → expert compute → AllToAll combine.

**Communication**: 2 AllToAll per MoE layer. Volume: `2 × tokens × top_k × H × dtype × (EP-1)/EP`. AllToAll is dynamic/irregular -- hot experts create asymmetric traffic.

**Load balancing**: auxiliary loss, bias adjustment (DeepSeek-V3), capacity factors, node-limited routing (M=4), redundant experts.

**Use when**: MoE models where total expert params exceed single GPU.

---

## 9. EDP/DEP — Expert Data Parallelism

Data parallelism for MoE expert layers. Multiple GPUs hold the **same** expert assignment but process **different** token batches.

**Key insight**: EP and EDP are orthogonal dimensions forming a 2D grid:
- EP dimension (AllToAll): within a group, different GPUs hold different experts
- EDP dimension (AllReduce): across groups, same experts replicated

From Parallel Folding: `N_total = ETP × EP × EDP × PP`. Before MoE Parallel Folding (2025), Megatron-LM required `EP ≤ DP` -- Parallel Folding removed this constraint.

**DP Attention + EP** (inference): Attention layers use data parallelism (partitioned KV cache) while MoE layers use expert parallelism. Avoids KV cache duplication.

---

## 10. ETP/TEP — Expert Tensor Parallelism

Tensor parallelism within individual experts. Communication pipeline per MoE layer:
```
Permute → AllToAll(EP) → AllGather(ETP) → Expert Compute → ReduceScatter(ETP) → AllToAll(EP) → Unpermute
```

Overhead ratio: `(ETP-1) / (ETP × (EP-1))`. EP=8, ETP=2: ~7%. EP=2, ETP=8: ~87%.

**Recommendation: set ETP=1** for fine-grained MoE. Only use ETP>1 when individual experts don't fit on one GPU.

---

## 11. DP Attention — Data-Parallel Attention for MoE Inference

### 11.1 Core Idea

In MoE inference, TP+EP causes KV cache to be **fully replicated** across all TP GPUs. DP Attention instead **partitions** the KV cache -- each GPU only stores KV for a subset of requests.

```
DP Attention + EP (8 GPUs):
  Attention (DP=8): each GPU processes different requests, KV cache partitioned
  MoE (EP=8):       experts distributed, all tokens mixed via AllToAll

  vs TP+EP:
  Attention (TP=8): all GPUs store full KV cache (8x duplication!)
  MoE (EP=8):       same
```

### 11.2 Why It Matters for MoE

With 1000 concurrent requests on 8 GPUs:
- TP=8: each GPU stores KV for all 1000 requests
- DP=8: each GPU stores KV for 125 requests (1000/8)
- **8x KV cache reduction → 8x more concurrent requests**

### 11.3 Communication

- Prefill: no communication (each GPU processes its own requests independently)
- MoE transition: AllToAll dispatch/combine (same as standard EP)
- Optional: AllGather KV during decode for cross-GPU attention

### 11.4 Configuration (vLLM)

```bash
# DP Attention + EP
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel

# vs TP + EP
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 --enable-expert-parallel
```

### 11.5 Performance

- Crossover at ~256-512 concurrency
- Low concurrency: TP+EP wins on latency (~52% throughput advantage at concurrency=64)
- High concurrency: DP+EP wins on throughput (~47% advantage at concurrency=1024)

---

## 12. Combining Strategies

### 12.1 GPU Formulas

```
Dense: N_total = TP × SP × CP × DP × PP  (SP usually = TP)
MoE:   N_total = ETP × EP × EDP × PP
```

### 12.2 Common Patterns

- **TP+EP** (low concurrency inference): TP for attention, EP for MoE. Must enable SP.
- **DP Attention+EP** (high concurrency inference): DP for attention (partitioned KV cache), EP for MoE.
- **TP intra-node + PP inter-node**: Classic hybrid for multi-node deployment.
- **PP+EP** (large MoE training): DeepSeek-V3 uses PP=16 + EP=64 with DualPipe.
- **4D/5D parallelism**: TP × PP × CP × DP (+EP for MoE).

### 12.3 Example Configurations

| Model | GPUs | TP | PP | CP | DP | EP |
|-------|------|----|----|----|----|-----|
| Llama-3-70B train | 256 | 8 | 4 | 1 | 8 | -- |
| Llama-3-405B train | 16384 | 8 | 16 | 1 | 128 | -- |
| DeepSeek-V3 train | 2048 | 1 | 16 | 1 | 2 | 64 |
| DeepSeek-V3 prefill | 32 | 4 | 1 | 1 | 8 | 32 |
| 1M-token training | 64 | 4 | 2 | 8 | 1 | -- |

### 12.4 Key Rules

| Rule | Rationale |
|------|-----------|
| TP within NVLink domain | Cross-node TP is nearly infeasible |
| Enable SP with TP | Mandatory for TP+EP; saves activation memory |
| Maximize EP, minimize ETP | EP has lower comm overhead than ETP |
| PP for cross-node scaling | Low-frequency P2P, lowest bandwidth need |
| CP for ultra-long sequences | When seq > 128K tokens |
| ZeRO-1 before ZeRO-3 | Same comm volume, much better memory |
| Concurrency decides TP+EP vs DP+EP | Crossover ~256-512 concurrent requests |

---

## 13. DeepSeek-V3 Case Study

**Architecture**: 671B total, 37B activated/token, 61 layers, 256 routed experts + 1 shared, top-8, MLA (KV compressed to 512 dims).

**Training (2048 H800)**: PP=16 (DualPipe), EP=64 (across 8 nodes), ZeRO-1 DP, **TP=1**. No tensor parallelism because MLA's small KV dims make TP overhead > benefit. DualPipe overlaps AllToAll with compute. Node-limited routing (M=4) cuts cross-node traffic ~50%.

**PP=16 + DualPipe**: Bubble reduces from `15 × (T_F + T_B)` (1F1B) to `7 × T_{F&B}` (>50% reduction). AllToAll communication fully hidden in DualPipe's compute-comm overlap. Custom layout `"Et*3|(tt|)*29,m|L"` with VPP=2. PP=16 interacts with EP=64: each PP stage has ~4 layers, each with AllToAll for EP routing, all hidden by DualPipe scheduling.

**Inference prefill (32 GPU)**: TP=4+SP, EP=32, DP=8, 32 redundant experts.
**Inference decode (320 GPU)**: TP=4+SP, EP=320, DP=80.
**LMSYS 96 H100**: EP=32 prefill, EP=72 decode, DP Attention for KV partitioning.

---

## 14. Selection Guide

1. Model fits on 1 GPU → **DP only** (or ZeRO if memory tight)
2. Doesn't fit, have NVLink → **TP ≤ 8** within NVLink domain + **SP**
3. Need more GPUs → add **PP** across nodes; **CP** for long sequences
4. MoE model → add **EP** for experts; set **ETP=1**, maximize EP
5. Remaining GPUs → **DP/EDP** for throughput; **ZeRO-1** for memory
6. Inference: low concurrency → TP+EP; high concurrency → DP Attention+EP

---

## 15. Full Comparison

| Dimension | DP | ZeRO | TP | SP | PP | CP | EP | EDP | ETP | DP Attn |
|-----------|----|----|----|----|----|----|----|----|----|----|
| **Shards** | Data batch | Data+optim/grad/params | Weight matrices | Seq (non-TP) | Layer groups | Seq (attention) | Experts | MoE data | Expert weights | KV partitions |
| **Scope** | Dense | Dense | Dense | Dense | All | Attention | MoE | MoE | MoE | Inference Attn |
| **Collective** | AllReduce | AG+RS | AllReduce | AG+RS | Send/Recv | Ring/A2A | AllToAll | AllReduce | AG+RS | AllGather |
| **BW need** | Low | Med-High | Very High | Very High | Low | Med-High | High | Low-Med | Very High | Med |
| **Memory** | Low (full copy) | High (16P/N) | High (1/N) | Higher (act/N) | High (1/PP) | High (KV/N) | High (1/EP) | Low (copy) | High (1/ETP) | High (KV/N) |
| **Scalability** | Excellent | Excellent | ≤8 | =TP | Good | Good | Good | Excellent | ≤8 | Good |

---

## 16. References

- [Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)
- [GPipe (Huang et al., 2019)](https://arxiv.org/abs/1811.06965)
- [PipeDream (Narayanan et al., 2019)](https://arxiv.org/abs/1806.03377)
- [ZeRO (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [Megatron-LM v2 — Sequence Parallelism (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198)
- [Megatron-LM v2 — Interleaved 1F1B (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473)
- [Zero Bubble PP (Qi et al., ICLR 2024)](https://arxiv.org/abs/2401.10241)
- [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437) — DualPipe
- [Ring Attention (Liu et al., ICLR 2024)](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses (Jacobs et al., 2023)](https://arxiv.org/abs/2309.14509)
- [Unified Sequence Parallelism (2024)](https://arxiv.org/abs/2405.07719) — Hybrid Ring+Ulysses
- [MoE Parallel Folding (NVIDIA, 2025)](https://arxiv.org/abs/2504.14960)
- [Megatron Core MoE Docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [JAX Scaling Book](https://jax-ml.github.io/scaling-book/training/)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## 17. Related Pages

- [[vllm]] — Implements DP+EP / TP+EP
- [[prefill-decode-disaggregation]] — Different parallelism per phase
- [[distributed-training]] — Training-side parallelism
- [[quantization]] — Reduce parallelism needs
