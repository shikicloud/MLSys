---
title: "DeepSpeed Ulysses: Sequence Parallelism via Head-Sharding AllToAll"
category: llm-inference
tags: [deepspeed-ulysses, context-parallelism, sequence-parallelism, long-context, attention, alltoall, microsoft, paper-review]
created: 2026-05-19
updated: 2026-05-21
status: mature
paper: arXiv:2309.14509
code: https://github.com/deepspeedai/DeepSpeed
---

# DeepSpeed Ulysses: Sequence Parallelism via Head-Sharding AllToAll

> [!info] Paper metadata
> - **Paper**: [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) — *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models* (Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, Yuxiong He; Microsoft, 2023)
> - **Code**: [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) — primary file `deepspeed/sequence/layer.py` (`DistributedAttention` class)
> - **Blog**: [DeepSpeed Ulysses README](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md) · **Tutorial**: [deepspeed.ai/tutorials/ds-sequence](https://www.deepspeed.ai/tutorials/ds-sequence/)
> - **Follow-up**: [Ulysses-Offload (FPDT)](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) — 4M tokens on 32× A100
> - **Companion page**: [[ring-attention]] — the P2P-ring-based alternative

---

## Summary (read this if you have 2 minutes)

**What it is.** DeepSpeed Ulysses (Microsoft, Sep 2023) is a sequence-parallel attention scheme that lets you train transformers on million-token sequences by *re-laying-out* activations rather than chopping the attention math. Two AllToAll transpositions per layer flip data between **sequence-sharded** `[N/P, d]` and **head-sharded** `[N, d/P]` layouts; head-sharded attention runs locally on each GPU with **standard, unmodified FlashAttention**.

**The one idea.** Make attention embarrassingly parallel by sharding the *head* axis, not the sequence axis, *during attention*. Three pieces hold it up:

1. **AllToAll transpose** — `[N/P, d]` → `[N, d/P]` before attention, reverse after. Per-link comm volume is $4Nh/P$ per layer — **constant when $N$ and $P$ scale together**.
2. **Unmodified attention kernel** — each GPU holds the *full sequence* on $1/P$ of the heads, so any stock FlashAttention 2/3 kernel runs as-is. No custom streaming softmax.
3. **Causal mask is naturally balanced** — every rank's causal triangle is identical because every rank holds the full sequence. No Striped/zigzag scheduling needed.

Remove any one: lose the $1/P$ scaling, need a custom distributed kernel, or eat causal load-imbalance.

**Headline result.** 256× A100, GPT 1.2B–30B; **sustained ~175 TFLOPs/GPU (54 % of A100 peak)** at 64K; trains **1M-token sequences**; **2.5× faster, 4× longer** than Megatron-SP baseline. Weak scaling holds at 256K × 256 GPUs (147.4 TFLOPs). The critical limit: **$P \leq \text{num\_kv\_heads}$** — for GQA models (Llama-3, Mistral) that's 8, so pure Ulysses caps at one NVLink node.

**Why it matters.**

- **Stock FlashAttention works unmodified.** No fused distributed kernel; no streaming softmax surgery. Ulysses *wraps* the local attention call.
- **Default intra-node sequence-parallel primitive in 2026.** Shipped in HuggingFace TRL/Accelerate (`sp_backend="deepspeed"`), Tencent xDiT (diffusion video), verl RL, ms-swift, Megatron-DeepSpeed.
- **The production answer is hybrid.** Inner Ulysses (NVLink node, 8 GPUs) × outer Ring (cross-node IB) — see [USP](https://arxiv.org/abs/2405.07719), [LoongTrain](https://arxiv.org/abs/2406.18485). Ulysses lost the "single-axis" race to Ring; it won the "intra-node primitive" niche.
- **12-month watch.** GQA/MQA/MLA trends keep shrinking KV head counts. Either Ulysses migrates to operating at the *full* head count (broadcasting replicated KV — adds comm), or it stays as the intra-node half of every hybrid CP scheme.

---

# Depth (drill-down starts here)

The summary above is the executive layer. Everything below is for the careful reader who wants the comm-volume derivation, the GQA head-count math, and the source code.

## Background: why sequence parallelism is its own problem

LLM training has four orthogonal parallelism axes ([[parallelism-strategies-deep-dive|TP / PP / DP / EP]]). None of them shard the **sequence dimension** of activations. Past 128 K tokens, this becomes the binding constraint:

| Axis | What it shards | Activation memory at long $N$ |
| ---- | -------------- | ----------------------------- |
| DP | Batch | Per-rank: full $N$ × full hidden |
| TP | Weights / heads | KV cache **replicated** across TP ranks |
| PP | Layers | Each pipeline stage holds full $N$ |
| EP | Experts | MoE-only; doesn't shard sequence |

What's needed is a *sequence-parallel* axis: shard the $N$ tokens across $P$ ranks. The trouble: attention requires every query to see every key, so a naïve shard breaks the math.

Two competing answers emerged in 2023:

1. **[[ring-attention|Ring Attention]]** (Liu/Zaharia/Abbeel, Oct 2023) — keep Q on each device, rotate K/V around a ring, use FlashAttention's streaming softmax to accumulate across rotations.
2. **DeepSpeed Ulysses** (Microsoft, Sep 2023) — *reshape the data* so attention can run locally on each device's slice. Use two AllToAlls to switch between sequence-sharded and head-sharded layouts.

Both produce mathematically-identical attention output. They differ in **which dimension is sharded when**, what communication primitive moves data, and what the scaling limits are.

Per the paper's Table 1:

| Method | Communication complexity | Activation memory | Param memory | Attention-mechanism agnostic |
| ------ | ------------------------ | ----------------- | ------------ | ---------------------------- |
| ColAI-SP (Ring) | $O(M)$ | ✓ | ✗ | ✗ |
| Megatron-SP | $O(M)$ | ✓ | ✗ | ✗ |
| **DS-Ulysses** | $\mathbf{O(M/P)}$ | ✓ | ✓ | ✓ |

The paper's argument verbatim: *"DeepSpeed Ulysses incurs $O(M/P)$ communication volume, allowing it to scale to longer sequences and larger parallelism degrees without communication bottleneck."*

## The architecture: AllToAll-flip between two layouts

> [!quote] The contribution in one sentence
> Before attention, AllToAll transposes the activation from **sequence-sharded** `[N/P, d]` to **head-sharded** `[N, d/P]` (so each GPU holds the full sequence on a head subset and runs standard FlashAttention locally); after attention, AllToAll transposes back.

![DeepSpeed Ulysses AllToAll architecture (paper Fig. 2). Two red AllToAll comms straddle the local attention block: pre-attention flips `[N/P, d]` → `[N, d/P]`; post-attention flips back. Inside attention, each GPU owns the full sequence for $h_c/P$ heads.](EN/wiki/llm-inference/deepspeed-ulysses-figs/alltoall-architecture.png)

### Algorithm

Symbols from the paper: $N$ = sequence length, $b$ = micro-batch, $d$ = hidden, $h_c$ = head count, $h_s = d/h_c$ = head size, $P$ = sequence parallel degree.

```
                       sequence-sharded               head-sharded
                       [N/P, d]                       [N, d/P]
                       per GPU                        per GPU

   Input X    ─────►   X_local                                       ─┐
                       │                                              │
                       ▼                                              │
                       Q_local, K_local, V_local  (each [N/P, d])     │
                       │                                              │
   AllToAll #1  ───────┘  scatter heads,  gather sequence             │
                       ▼                                              │
                       Q, K, V  (each [N, hc/P, hs])    ──► attention │
                       │                                  (local)     │
                       ▼                                              │
                       Context  [N, hc/P, hs] = [N, d/P]              │
                       │                                              │
   AllToAll #2  ───────┘  scatter sequence,  gather heads             │
                       ▼                                              │
                       Context_local [N/P, d]                         │
                       │                                              │
   Output proj W_O ───►Output  [N/P, d]                              ─┘
```

**Why this works mathematically.** After AllToAll #1, each GPU holds the full sequence's $Q, K, V$ for its assigned heads. Multi-head attention is **embarrassingly parallel across heads** — different heads never interact during attention. So a GPU with $h_c / P$ heads can compute *exactly* the attention output for those heads, using whatever standard FlashAttention kernel is available. After AllToAll #2, the output is sequence-sharded again, ready for the output projection and downstream FFN.

### Communication volume (the paper's headline)

Per Section 3.2, verbatim:

> "On modern clusters with intra-node NVSwitch interconnect and inter-node fat tree IB topology, the communication volume transmitted per link for an all-to-all for aggregate message of size $M$ over $P$ GPUs is $M/P$. For a transformer model with hidden size $h$, sequence length of $N$, and parallelism degree of $P$, DS-Sequence performs all-to-all for the QKV projections with an aggregate message size of $3Nh$ before the attention computation, and another all-to-all for output context projection with a size $Nh$ for each transformer layer. Therefore, DeepSpeed sequence parallelism incurs an aggregate communication volume per link of $\mathbf{4Nh/P}$ (or with the complexity of $O(N/P)$). Note that this communication volume is constant when both $N$ and $P$ are increased proportionally."

For comparison, Megatron-SP volume is $4Nh$ per link (no $/P$ scaling), so $P\times$ larger.

### The hard cap: $P \leq \text{num\_heads}$

After AllToAll #1, each GPU must own a *whole number of heads* — $h_c / P$ must be a positive integer. So $P \leq h_c$.

For **GQA models** (most modern open weights), the binding constraint is `num_kv_heads`, which is much smaller than `num_q_heads`:

| Model | num_q_heads | num_kv_heads | Max Ulysses $P$ |
| ----- | ----------- | ------------ | --------------- |
| Llama-3-8B | 32 | 8 | **8** |
| Llama-3-70B | 64 | 8 | **8** |
| Mistral 7B (GQA) | 32 | 8 | **8** |
| MQA models (single KV head) | $h_c$ | **1** | **1 — effectively dead** |

This is the principal reason Ulysses is used **intra-node** today (8× H100/A100 NVLink fits perfectly) and combined with Ring **inter-node** via [USP](https://arxiv.org/abs/2405.07719) or [LoongTrain](https://arxiv.org/abs/2406.18485).

### Implementation: `DistributedAttention`

From `deepspeed/sequence/layer.py`:

```python
class DistributedAttention(torch.nn.Module):
    def __init__(self, local_attention, sequence_process_group,
                 scatter_idx: int = 2, gather_idx: int = 0,
                 sp_stream=None):
        self.local_attn = local_attention      # FlashAttention / SDPA / Triton
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx          # head dim
        self.gather_idx  = gather_idx           # sequence dim

    def forward(self, query, key, value, batch_dim_idx, ...):
        # AllToAll #1: scatter heads, gather sequence  →  [b, N, hc/P, hs]
        q = _SeqAllToAll.apply(self.spg, query, self.scatter_idx,
                                self.gather_idx, ...)
        k = _SeqAllToAll.apply(self.spg, key,   self.scatter_idx,
                                self.gather_idx, ...)
        v = _SeqAllToAll.apply(self.spg, value, self.scatter_idx,
                                self.gather_idx, ...)

        context = self.local_attn(q, k, v, *args, **kwargs)
        # ↑ standard FlashAttention / SDPA call on head-sharded layout

        # AllToAll #2: scatter sequence, gather heads  →  [b, N/P, hc, hs]
        return _SeqAllToAll.apply(self.spg, context, self.gather_idx,
                                   self.scatter_idx, ...)
```

`_SeqAllToAll` is a `torch.autograd.Function`. **Backward is symmetric** — the same op with `scatter_idx` and `gather_idx` swapped. So the gradient AllToAll is free.

> [!note] Two implementation details visible only in code
> - **`sp_stream` / `sp_overlap_comm`** enables backward-overlap on a separate CUDA stream — added *post-publication*. Forward AllToAlls remain blocking (see next subsection).
> - **`scatter_idx=2, gather_idx=0`** defaults assume the standard `(batch, seq, heads, head_dim)` layout. If your model uses `(batch, heads, seq, head_dim)`, you must swap these or the AllToAll reshuffles the wrong axis silently.

### Why no forward compute-comm overlap

Both AllToAlls are collective ops that synchronize all ranks. The first AllToAll's output directly feeds the very next op (attention itself), so there is no prior independent compute on the critical path to overlap with. The paper does not claim overlap.

This is a real cost on cross-node IB (bandwidth ~25 GB/s vs intra-node NVLink ~600 GB/s) — Ulysses pays the full AllToAll latency on every layer. [[ring-attention|Ring]]'s P2P, by contrast, overlaps naturally with attention compute. This is the principal scaling weakness Ulysses-vs-Ring discussions hit.

USP / LoongTrain effectively *fix* this by splitting QKV AllToAll into per-head pipeline stages — but that's algorithmic restructuring outside the stock Ulysses recipe.

### Causal mask: naturally balanced

After AllToAll #1, each rank holds the *full sequence* for its head subset. The causal triangle on each rank is identical — no [[ring-attention|Striped/zigzag fix needed]]. This is the cleanest aspect of Ulysses vs Ring.

### Memory

Per Section 3.3: Ulysses reduces **activation memory** but not model-state memory. It's integrated with **ZeRO-3** which partitions model states (weights + grads + optimizer state) across the combined $\text{DP} \times \text{SP}$ group — each rank owns $1/(\text{DP} \cdot \text{SP})$ of model state.

KV activation during forward exists in two layouts across the layer:

- **Outside attention**: sequence-sharded `[N/P, d]` per rank — total $Nd/P$ bytes
- **Inside attention**: head-sharded `[N, d/P]` per rank — total $Nd/P$ bytes

Both are $1/P$ of full, just along different axes.

### Supporting machinery (skim or skip)

> [!note]- FlashAttention integration — open if wiring Ulysses into an existing trainer
> Per Section 3.4: *"DeepSpeed Ulysses works with efficient attention implementations such as FlashAttention v2 (Dao 2023)."* User code passes any local attention module:
>
> ```python
> from deepspeed.sequence.layer import DistributedAttention
> from flash_attn import flash_attn_func
>
> local_attn = lambda q, k, v: flash_attn_func(q, k, v, causal=True)
> dist_attn = DistributedAttention(local_attn,
>                                   sequence_process_group=spg)
> ```
>
> Then `dist_attn(q, k, v)` runs the AllToAlls + FlashAttention. FlashAttention v3 (Hopper-optimized) works the same way. SDPA, Triton sparse, and custom kernels also slot in — Ulysses is genuinely attention-kernel-agnostic.

## Headline evidence

**Setup.** Up to 256 A100 (40 GB), NVSwitch intra-node + IB fat-tree inter-node. Models: GPT 1.2B, 7B, 30B; dense and blocked-sparse attention variants.

**The main result: throughput vs Megatron-SP at scale.** On 32 A100 with a 7B dense model, Ulysses sustains ~175 TFLOPs/GPU across 16K–256K sequences; Megatron-SP OOMs beyond 32K:

![DeepSpeed-Ulysses vs Megatron-LM, 7B GPT dense, 32 A100 (paper Fig. 4). Megatron OOMs beyond 32K; Ulysses sustains 175 TFLOPs all the way to 256K.](EN/wiki/llm-inference/deepspeed-ulysses-figs/throughput-7b-vs-megatron.png)

> [!success] The headline number
> Sustained throughput of **over 175 TFLOPs/GPU (54 % of A100 hardware peak)** at 64K on the 7B model. Headline abstract claim: **"2.5× faster with 4× longer sequence length"** than Megatron-SP.

**The critical limit: $P \leq \text{num\_heads}$.** Throughput at 30B shows the same OOM story — Megatron-SP collapses past 64K, Ulysses runs to 256K — but the head-count cap means scaling to >32 GPUs requires the model's `num_q_heads` ≥ 32:

![30B GPT dense, 64 A100 (paper Fig. 5). Megatron OOMs at 128K; Ulysses runs to 256K.](EN/wiki/llm-inference/deepspeed-ulysses-figs/throughput-30b-vs-megatron.png)

**Weak scaling: the "constant comm" claim, validated.** When $N$ and $P$ scale together, throughput stays nearly flat — exactly what $4Nh/P$ being constant predicts:

| Seq × GPUs | Time/iter (s) | TFLOPs/GPU |
| ----------: | ------------: | ---------: |
| 64K × 64 | 87.6 | 161.4 |
| 128K × 128 | 175 | 157.4 |
| **256K × 256** | **376** | **147.4** |

> [!example]- All experimental results (drill-down)
> **Strong scaling (1.2B GPT, 8–64 A100, Fig 3).** Per-GPU throughput stays in the 90–110 TFLOPs band as sequence grows 8K → 1M. The 1M data point requires 64 GPUs (Ulysses degree $P = 64$ on a model with $h_c \geq 64$).
>
> **7B / 30B vs Megatron-SP head-to-head (Fig 4, 5).** Both Megatron-SP and Ulysses can do short sequences, but Megatron OOMs first:
>
> | Model | Hardware | Sequence | Ulysses TFLOPs | Megatron-SP |
> | ----- | -------- | -------- | -------------: | ----------- |
> | GPT 7B dense | 32 A100 | 8K | 159 | 106 |
> | GPT 7B dense | 32 A100 | 64K | 175 | OOM |
> | GPT 7B dense | 32 A100 | 256K | runs | OOM |
> | GPT 30B dense | 64 A100 | 8K | 165 | 45 |
> | GPT 30B dense | 64 A100 | 256K | 134 | OOM (≥128K) |
> | GPT 7B sparse | 32 A100 | 8K → 256K | 132 → 68 | — (256K OOM) |
> | GPT 30B sparse | 64 A100 | 256K | 73 | OOM (≥128K) |
>
> **Convergence is identical to Megatron-SP (Fig 8).** 1.3B GPT, 32K sequence, 8 A100, $SP=4$. Loss curves of Megatron-SP, Ulysses+ZeRO-1, ZeRO-2, ZeRO-3 overlap — Ulysses is mathematically equivalent to a non-SP run, just memory-cheaper.
>
> ![LM loss over iterations, four configs (paper Fig. 8). Curves are visually indistinguishable.](EN/wiki/llm-inference/deepspeed-ulysses-figs/loss-convergence.png)

### Production deployments (verified)

| User | What they ship | Source |
| ---- | -------------- | ------ |
| **Microsoft Megatron-DeepSpeed** | Primary integration target named in the paper | [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) — `--ds-sequence-parallel-size N` flag |
| **HuggingFace TRL / Accelerate (SFT, 2025)** | Official Ulysses SP backend; `sp_backend="deepspeed"`, `sp_size=N` | [huggingface.co/blog/ulysses-sp](https://huggingface.co/blog/ulysses-sp) |
| **Ulysses-Offload (FPDT, 2024-12)** | DeepSpeed follow-up. **4M tokens on 32× A100 with Llama-70B**, **2M tokens on 4× A100-40GB with 8B model**, "55% MFU across 2.7B to 80B" | [Ulysses-Offload blog](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) |
| **Tencent xDiT** (diffusion video) | USP (Ulysses + Ring) for HunyuanVideo, CogVideoX, Wan2.1/2.2, Mochi-1. 8 GPUs, Ulysses-2 × Ring-2 × CFG-2 → **6.12× speedup vs single-GPU** | [xdit-project/xDiT](https://github.com/xdit-project/xDiT), [arXiv:2411.01738](https://arxiv.org/abs/2411.01738) |
| **verl** RL post-training framework | Ulysses for long-context rollouts | [PyTorch forum thread](https://discuss.pytorch.org/t/support-for-ulysses-ring-distributed-attention-for-long-context-training-32k-for-32b-dense-models/223106) |
| **ms-swift (Alibaba ModelScope)** | Ulysses + Ring zigzag | community framework |

### Compared with [[ring-attention|Ring Attention]]

| Property | DeepSpeed Ulysses | [[ring-attention\|Ring Attention]] |
| -------- | ----------------- | ---------------- |
| What moves between GPUs | QKV reshuffled by AllToAll (twice per layer) | KV blocks rotate around a ring ($P{-}1$ rounds) |
| Comm volume per layer per link | $4Nh/P$ — $O(N/P)$ (sublinear in $N$ when scaling $P$) | $\sim 2c d$ per rotation × $(P{-}1)$ rotations |
| Compute / comm overlap | ✗ Blocking | ✓ Fully hidden when $c \geq F/B$ |
| Hard GPU-count limit | $P \leq \text{num\_heads}$ (much worse for GQA) | None |
| Cross-node scaling | ✗ AllToAll over IB degrades | ✓ P2P bandwidth-friendly |
| Causal-mask balance | ✓ Naturally balanced | ✗ Needs Striped / zigzag fix |
| Attention kernel changes | None — uses stock FlashAttention | Yes — fused with FA streaming softmax |
| Best fabric | NVLink / NVSwitch intra-node | NVLink or IB inter-node |

## Strengths and limitations

The two strongest points: (1) **$O(N/P)$ comm volume** is the cleanest scaling behavior in the SP zoo — constant when $N$ and $P$ scale together; (2) **the attention kernel is unmodified** — stock FlashAttention works as-is, no fused-distributed kernel needed, and causal masking is naturally balanced without scheduling tricks.

Where the work is honest about scope but the limits matter:

- **$P \leq \text{num\_heads}$ is a hard cap.** For GQA models (Llama-3, Mistral) with 8 KV heads, max Ulysses degree is 8 — fits one NVLink-connected node but cannot scale beyond. MQA models (single KV head) effectively can't use Ulysses at all.
- **No forward compute-comm overlap.** AllToAll is blocking. On cross-node IB (~25 GB/s vs NVLink ~600 GB/s), the AllToAll latency dominates each layer. This is the principal weakness vs [[ring-attention|Ring]].
- **Incompatible with Megatron TP and PP** in stock DeepSpeed (per the tutorial), locking users into ZeRO-3.
- **Backward overlap added post-publication.** The `sp_overlap_comm` / `sp_stream` path in current source overlaps backward AllToAll with next-layer compute, but this is a refinement, not in the original algorithm.
- **Communication-volume claim is per-link, not per-rank.** The headline "$4Nh/P$" is the volume each link carries, not the total comm work. Per-rank total comm is higher when accounting for participation in AllToAll.

> [!warning] When Ulysses wins, when Ring wins
> Stylized 2026 decision rule (from USP paper Tables 3–4 and Tencent xDiT benchmarks):
>
> - **Intra-node, head count ≥ GPU count, NVLink/NVSwitch**: **Ulysses wins**. AllToAll on NVLink screams; constant volume; no scheduling complexity.
> - **Cross-node IB, head count limits scaling**: **Ring wins**. P2P overlaps; no head-count cap.
> - **Both at scale (most production)**: **Hybrid — USP / LoongTrain**. Inner Ulysses degree = node size; outer Ring degree = node count. Best of both.

## What this means

Two predictions worth tracking:

1. **Ulysses survives as the intra-node primitive.** As MoE models grow and per-node GPU count stays at 8 (NVLink domain), Ulysses-degree-8 on each node will remain the default inner SP. Tencent USP, LoongTrain, and PyTorch native CP all converge on this pattern.
2. **The head-count cap will tighten further with GQA / MQA / MLA.** DeepSeek-V3's MLA collapses KV to a tiny compressed representation; Llama-4 reportedly continues the GQA trend with fewer KV heads. Pure Ulysses will look increasingly weaker against Ring as KV head counts shrink. The way out is either combining with Ring (USP) or operating at the *full* head-count level rather than KV-head-count (requires distributed broadcast of replicated KV — adds comm).

What this is *not*: a universal long-context solver (the head-count cap rules out frontier-scale pure Ulysses), nor an inference primitive (Ulysses is training-side; decode-side is its own story), nor a replacement for FlashAttention (Ulysses wraps FlashAttention, doesn't replace it).

## Source code & reproduction

### Released artifacts

| Artifact | Status |
| -------- | ------ |
| `DistributedAttention` class | ✓ Open, Apache-2.0 — `deepspeed/sequence/layer.py` |
| Tests | ✓ `tests/unit/sequence_parallelism/test_ulysses.py` |
| Megatron-DeepSpeed integration | ✓ `--ds-sequence-parallel-size N` flag |
| Ulysses-Offload (FPDT) | ✓ DeepSpeed v0.13+ |

### Minimum reproduction (raw)

```python
import torch
from deepspeed.sequence.layer import DistributedAttention
from flash_attn import flash_attn_func

# In a torch.distributed-initialized process group of size P
# where P divides num_kv_heads:
spg = ...  # sequence_process_group

local_attn = lambda q, k, v: flash_attn_func(q, k, v, causal=True)
dist_attn = DistributedAttention(local_attn, sequence_process_group=spg)

# Input is sequence-sharded: each rank has (batch, N/P, num_heads, head_dim)
output = dist_attn(q, k, v, batch_dim_idx=0)
# Output is sequence-sharded same shape
```

### Production integration recipes

**Megatron-DeepSpeed (CLI flag)**:

```bash
deepspeed --num_gpus=8 train.py \
  --ds-sequence-parallel-size 8 \
  --deepspeed_config ds_config.json \
  --use-flash-attn-triton \
  ...
```

Requires DeepSpeed v0.10.2+. Incompatible with Megatron TP and PP per the [tutorial](https://www.deepspeed.ai/tutorials/ds-sequence/).

**HuggingFace TRL / Accelerate (2025+)**:

```python
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    output_dir="./ulysses_sft",
    sp_backend="deepspeed",
    sp_size=8,
    ...
)
trainer = SFTTrainer(model=model, args=config, train_dataset=ds)
trainer.train()
```

See [huggingface.co/blog/ulysses-sp](https://huggingface.co/blog/ulysses-sp).

**Tencent USP (Ulysses + Ring hybrid)**:

```python
# Pseudocode — see github.com/feifeibear/long-context-attention
mesh = init_2d_mesh(ulysses_size=8, ring_size=4)  # 32 GPUs total
output = usp_attn(q, k, v, mesh=mesh, causal=True)
```

8-way Ulysses inside each node × 4-way Ring across nodes = 32 GPUs total SP.

### Files worth reading

| File | Role |
| ---- | ---- |
| `deepspeed/sequence/layer.py` | `DistributedAttention` class, `_SeqAllToAll` autograd Function — entire 100 LOC implementation |
| `tests/unit/sequence_parallelism/test_ulysses.py` | Round-trip test asserting AllToAll(AllToAll(x)) = x |
| `blogs/deepspeed-ulysses/README.md` | Official walkthrough |
| `blogs/ulysses-offload/README.md` | FPDT extension for 4M-token training |

## Related reading

- [[ring-attention]] — The P2P-ring-based alternative. Sister page to this one.
- [[parallelism-strategies-deep-dive#7. CP — Context Parallelism]] — Where Ulysses sits in the broader parallelism landscape; comparison table with Ring and Megatron CP.
- [[paged-attention]] — FlashAttention; Ulysses uses it unmodified as the local attention kernel after AllToAll.
- [[kv-cache-optimization]] — KV cache compression intersects long-context training and inference.
- [[parallelism-strategies-deep-dive#11. DP Attention — Data-Parallel Attention for MoE Inference]] — Different parallelism approach to attention at scale; for MoE inference vs training.
- [[long-context-serving]] — Production long-context inference; Ulysses is training-side; serving is its own story.

## References

- **DeepSpeed Ulysses paper**: Jacobs et al., Microsoft. [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) · [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2309.14509)
- **Official blog**: [DeepSpeed Ulysses README](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md)
- **Tutorial**: [deepspeed.ai/tutorials/ds-sequence](https://www.deepspeed.ai/tutorials/ds-sequence/)
- **Source**: [`deepspeed/sequence/layer.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py) · [test](https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/sequence_parallelism/test_ulysses.py)
- **Ulysses-Offload (FPDT)**: [blog](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) · [tutorial](https://www.deepspeed.ai/tutorials/ulysses-offload/)
- **USP (Ulysses × Ring hybrid)**: [arXiv:2405.07719](https://arxiv.org/abs/2405.07719) · [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention)
- **LoongTrain (2D-Attention hybrid)**: [arXiv:2406.18485](https://arxiv.org/abs/2406.18485)
- **xDiT (Ulysses + Ring for diffusion video)**: [xdit-project/xDiT](https://github.com/xdit-project/xDiT) · [arXiv:2411.01738](https://arxiv.org/abs/2411.01738)
- **HuggingFace Ulysses-SP blog**: [huggingface.co/blog/ulysses-sp](https://huggingface.co/blog/ulysses-sp)
- **PyTorch Context Parallel** (related, Ring-primary): [docs.pytorch.org/.../context_parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- **[[ring-attention|Ring Attention]] (alternative)**: [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- **Insujang's CP overview**: [insujang.github.io/.../introducing-context-parallelism](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)
- **FlashAttention** (local kernel): [Tri Dao FA blog](https://tridao.me/blog/2024/flash3/)
