---
title: "LLM Parallelism Strategies Complete Guide: DP / TP / PP / SP / CP / EP / EDP / ETP"
category: llm-inference
tags: [tensor-parallelism, data-parallelism, expert-parallelism, pipeline-parallelism, sequence-parallelism, context-parallelism, moe, multi-gpu, distributed-inference, distributed-training]
created: 2026-04-14
updated: 2026-05-13
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

### 5.4 SP vs CP — the naming collision

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

| | SP (Megatron v2) | CP |
|---|---|---|
| What it shards | LayerNorm + Dropout activations (non-TP regions) | The whole attention (QKV + softmax + output proj) |
| Relation to TP | Always inside a TP group; SP size = TP size | Independent GPU dim; CP size unrelated to TP |
| Communication | AllGather (entering TP region) + ReduceScatter (leaving) | Ring P2P (Ring Attention) or AllToAll (Ulysses) |
| Why it exists | Reduce activation memory inside TP regions | Support sequences too long to fit one GPU's KV cache |
| Typical size | SP = TP = 8 (whatever TP you picked) | CP = 2 / 4 / 8 / ... / 64+ |
| Mandatory? | Megatron-LM forces SP whenever TP + EP are both used | Only when sequences exceed single-GPU capacity |
| Talks to | TP-region AllReduce, decomposed into AG + RS | Other CP ranks holding adjacent sequence segments |
| Total comm volume | Same as plain TP (AllReduce decomposed, not eliminated) | Adds new comm on top of TP/DP |

The DeepSpeed Ulysses paper title — *"Sequence Parallelism for Long Sequence Training"* — is the most painful overlap, because that paper's "sequence parallelism" is exactly what every other framework now calls CP. **When reading any paper that mentions "sequence parallelism," check the dimension being sharded, not the word.** If the attention math is changed, it's CP. If only LayerNorm/Dropout activations move, it's SP.

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

At 128 K → 1 M → 10 M token sequences, two problems compound on a single GPU:

- **KV cache exceeds memory.** Llama-3-70B at $S = 1\text{M}$, FP16: $2 \times 80 \times 8 \times 128 \times 10^6 \times 2 \approx 328\text{ GB}$ per request — far beyond one H100/H200's HBM.
- **Attention is $O(S^2)$ in FLOPs.** Doubling $S$ quadruples attention compute.

CP shards the **sequence dimension** across $N$ GPUs. Each GPU holds $S/N$ tokens of Q/K/V. The hard part is that attention requires every query to see *all* keys/values — so the question becomes: how do you make every GPU "see" all KV without materializing the whole sequence on any single GPU? Three implementations answer this question with three different communication strategies.

### 7.2 Ring Attention

**Idea** (Liu, Zaharia, Abbeel — ICLR 2024). Put the $N$ GPUs in a ring. Each GPU keeps its own Q segment in place; the **KV blocks rotate** around the ring. After $N{-}1$ rotations, every Q has been multiplied against every KV.

```
4 GPUs, sequence split into 4 segments [Q0K0V0, Q1K1V1, Q2K2V2, Q3K3V3]:

Round 0:  each GPU computes attention on its own KV
  GPU0: Q0 × (K0,V0) → O0_partial
  GPU1: Q1 × (K1,V1) → O1_partial
  GPU2: Q2 × (K2,V2) → O2_partial
  GPU3: Q3 × (K3,V3) → O3_partial

Round 1:  KV rotates one step clockwise
  GPU0: Q0 × (K3,V3) → accumulate into O0_partial   ← P2P recv (K3,V3) from GPU3
  GPU1: Q1 × (K0,V0) → accumulate into O1_partial
  GPU2: Q2 × (K1,V1) → accumulate into O2_partial
  GPU3: Q3 × (K2,V2) → accumulate into O3_partial

Round 2, 3:  continue. After N-1 = 3 rounds each Q_i has seen every K/V.
```

Three implementation details make Ring Attention work in practice:

- **Online softmax.** Partial outputs cannot be summed directly — softmax is non-linear. Use FlashAttention's streaming softmax: maintain `(running_max, running_sum, running_output)` per Q row and merge each incoming KV chunk in numerically-stable fashion. This is why Ring Attention is always implemented *with* FlashAttention.
- **Compute / communication overlap.** Round $k{+}1$'s KV transfer can run in parallel with round $k$'s attention compute. If $T_{\text{compute}} \geq T_{\text{comm}}$, communication is **fully hidden** and Ring's effective overhead vs. plain attention is near-zero. The condition holds easily for $S \geq 8192$, $N \leq 8$.
- **Causal-mask load imbalance.** Under causal masking, GPU $i$'s queries should attend only to KV at positions $\leq$ themselves. So when GPU $i$ receives a KV chunk from "later" in the sequence, the work is zero (skipped by the mask). GPU 0 ends up doing almost no work; GPU $N{-}1$ does almost everything. The **Striped Attention** follow-up (Liu et al., 2023) fixes this with a zigzag chunking that gives each GPU a mix of early and late tokens.

**Properties.**

- Memory per GPU: $O(S/N)$ — linear.
- Total compute: unchanged ($S^2/N$ per GPU, $S^2$ globally).
- Communication: $N{-}1$ rounds of P2P, each carrying $\sim$ `head_dim × num_kv_heads × S/N × dtype` bytes.
- GPU count limit: none — works at $N = 32, 256, 4096$.
- Cross-node: scales gracefully, P2P is bandwidth-friendly over IB / RoCE.

**Real-world numbers.** Meta achieved 1 M tokens in <1 minute on a single H100 node; 10 M tokens on 32 hosts. RingX (SC '24) trained Llama-3-8B at $S = 1\text{M}$ on 4096 Frontier GPUs at 38 % MFU.

### 7.3 DeepSpeed Ulysses

**Idea** (Jacobs et al., 2023). Instead of rotating KV, **AllToAll-transpose the data** so attention becomes local on each GPU.

```
N=4 GPUs, sequence length S, H total heads, head dim D:

Before attention (sharded by sequence):
  GPU_i:  (S/N, H, D)        ← my segment of the sequence, all heads

AllToAll #1: transpose sharding from sequence → heads
  GPU_i:  (S, H/N, D)        ← full sequence, my subset of heads

Attention is now LOCAL on each GPU (uses standard FlashAttention).
  GPU_i:  attention(Q[S, H/N, D], K[S, H/N, D], V[S, H/N, D])

AllToAll #2: transpose back from heads → sequence
  GPU_i:  (S/N, H, D)        ← back to sequence-sharded for output projection
```

Per attention layer: 4 AllToAll total (Q, K, V are sharded separately before fusing, then output goes back through one more).

**Properties.**

- **Attention kernel is unchanged** — directly uses stock FlashAttention on a head subset. This is the killer feature: no custom kernel work.
- **Hard limit: $N \leq \text{num\_heads}$.** Because the second sharding axis is heads, you can't have more GPUs than heads. For 32-head models, Ulysses caps at $N = 32$. GQA models are worse — capped at the much smaller $\text{num\_kv\_heads}$.
- **Communication is blocking.** AllToAll is a synchronous collective — all $N$ GPUs must complete the exchange before attention can start. No overlap with compute.
- **Per-GPU communication volume** $\sim O(S \cdot D \cdot H / N)$, roughly constant in $N$ (AllToAll's good property).
- **Best on NVLink.** AllToAll within a node screams; AllToAll across IB nodes degrades quickly.

**Causal mask: no problem.** Because each GPU processes the *full sequence* on its head subset, causal masking is internal to the local FlashAttention call — naturally balanced.

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

### 7.5 Three-way comparison

| Property | Ring Attention | DeepSpeed Ulysses | Megatron CP |
|----------|----------------|-------------------|-------------|
| What moves | KV blocks rotate around a ring | QKV reshuffled by AllToAll | KV rotates (Ring) + zigzag chunking |
| Communication primitive | $N{-}1$ rounds of P2P send/recv | 4 × AllToAll per attention layer | P2P inside FlashAttention loop |
| Overlap with compute | ✓ Fully hidden when $T_c \geq T_{\text{comm}}$ | ✗ Blocking | ✓ Strong overlap |
| GPU-count limit | None | $\leq$ num_heads (much worse for GQA) | None |
| Cross-node scaling | ✓ P2P is bandwidth-friendly | ✗ AllToAll over IB struggles | ✓ |
| Causal-mask balance | ✗ Needs Striped fix | ✓ Naturally balanced | ✓ Zigzag |
| Attention kernel changes | Yes — fused with FA streaming softmax | None | Yes — embedded in FA tile loop |
| Typical users | xFormers, research frameworks | DeepSpeed, Microsoft training stacks | Megatron-LM, NeMo, DeepSeek, NVIDIA training |

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

**Training vs inference.** CP is dominant in *training* — the gradient/activation memory scales with $S$ and overwhelms single GPUs first. In *inference*, the answer is usually compress the KV cache ([[saw-int4|SAW-INT4]], MLA, KV pruning) and/or [[prefill-decode-disaggregation|disaggregate]] before reaching for CP. But Gemini-style 1M+ inference and frontier long-context serving do use CP at inference time too.

### 7.7 CP vs TP for long sequences

| Property | CP | TP |
|----------|----|----|
| What it shards | Sequence ($S/N$ tokens per GPU) | Weights / hidden dim |
| KV cache | Sharded ($S/N$ per GPU) | **Replicated** (full KV on every GPU) |
| When it communicates | Inside attention only | Every transformer block (attention + MLP) |
| Scaling target | Ultra-long sequences ($S > 128$ K) | Standard sequences, model size |
| Practical $N$ limit | Effectively unlimited (Ring/Megatron) | $\leq$ NVLink domain ($\approx 8$) |

The KV duplication is what makes TP a bad answer for long context. TP=8 on 1 M tokens means each of 8 GPUs stores the *full* 1 M-token KV cache — 8× wasted memory. CP=8 gives each GPU just $1/8$ of the KV cache — linear scaling. For anything past 128 K, CP wins on memory regardless of the model.

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

### 12.5 Key Rules

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
