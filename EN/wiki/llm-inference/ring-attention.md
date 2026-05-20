---
title: "Ring Attention: Sharding Sequence Length Across Devices with Compute-Communication Overlap"
category: llm-inference
tags: [ring-attention, context-parallelism, long-context, attention, flash-attention, blockwise-transformer, striped-attention, paper-review]
created: 2026-05-19
updated: 2026-05-19
status: mature
paper: arXiv:2310.01889
code: https://github.com/lhao499/RingAttention
---

# Ring Attention: Sharding Sequence Length Across Devices with Compute-Communication Overlap

> [!info] Paper metadata
> - **Paper**: [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) — *Ring Attention with Blockwise Transformers for Near-Infinite Context* (Hao Liu, Matei Zaharia, Pieter Abbeel; UC Berkeley)
> - **Venue**: **ICLR 2024** (poster). Earlier workshop version at NeurIPS 2023 FMDM. *Not* an ICML 2024 paper — common mis-attribution.
> - **Code (canonical JAX/TPU)**: [lhao499/RingAttention](https://github.com/lhao499/RingAttention) — Apache-2.0
> - **PyTorch ports**: [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) (most-used), [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch)
> - **Precursor**: [Blockwise Parallel Transformer (Liu & Abbeel, 2023)](https://arxiv.org/abs/2305.19370) — single-device blockwise attention+FFN; Ring Attention is its distributed sibling
> - **Companion follow-up**: [Striped Attention](https://arxiv.org/abs/2311.09431) (same group; fixes causal-mask imbalance, 1.45–1.65× speedup)
> - **Companion page**: [[deepspeed-ulysses]] — the AllToAll-based alternative

> [!abstract]+ TL;DR
> Ring Attention shards the sequence dimension of attention across $N$ devices arranged in a **ring**: each device keeps a persistent local query block, while key/value blocks **rotate** around the ring. After $N{-}1$ rotations, every Q has been multiplied against every K, V — producing **exactly the same attention output as single-device computation**, but with per-device activation memory $O(b \cdot c \cdot h)$ that's **independent of total sequence length**. The technical innovation is fusing this distributed rotation with **FlashAttention-style streaming softmax**: each Q row maintains running `(max, sum, output)` statistics across all $N$ chunks, never materializing the full attention matrix. With the right block size $c \geq F/B$ (host FLOPs / interconnect bandwidth), KV rotation **fully overlaps with attention compute** — comm is free. Liu et al. reported scaling a 7B model to **4M token context on 32× A100** and **8M tokens on TPU v4-1024**. Production today: **Megatron-Core Context Parallelism, Meta Llama 3 training (4D parallelism with CP), and Tencent USP** all are Ring Attention plus a few years of refinements (zigzag scheduling, pass-Q variant, hybrid with [[deepspeed-ulysses|Ulysses]]). The bare 2023 algorithm isn't competitive without these; the 2026 production stack is. **Compared to** [[deepspeed-ulysses|DeepSpeed Ulysses]]: Ring scales to any $N$ but needs high interconnect bandwidth; Ulysses caps at $N \leq \text{num\_heads}$ but has constant per-rank comm. Hybrid (USP) is the production answer.

---

## Background: why sharding sequence (not heads or layers) needed inventing

Three structural facts about Transformers at long context make this problem its own:

1. **KV cache exceeds single-GPU memory at the limit.** Llama-3-70B at $S = 1\text{M}$, FP16, 80 layers, 8 KV heads, $d_{\text{head}}=128$: $2 \times 80 \times 8 \times 128 \times 10^6 \times 2 \approx 328$ GB per request — far past one H100's HBM.
2. **Attention FLOPs are $O(S^2)$.** Doubling sequence quadruples attention compute. No memory trick changes that.
3. **Existing parallelism axes don't help at length.**
   - [[parallelism-strategies-deep-dive#4. TP — Tensor Parallelism|TP]] shards weights, not activations — KV cache is replicated across TP ranks.
   - [[parallelism-strategies-deep-dive#6. PP — Pipeline Parallelism|PP]] shards layers — each layer still holds full sequence.
   - [[parallelism-strategies-deep-dive#2. DP — Data Parallelism|DP]] shards batch — each rank still holds a full sequence's worth.

[[paged-attention|FlashAttention]] solved single-device memory by streaming attention in tiles, but doesn't escape $O(S/N)$ per-device memory at $N = 1$. The natural next move: **shard the sequence dimension itself across multiple devices** so each holds $S/N$ tokens.

But attention requires every query to see every key. Naïvely, each Q on device $i$ would need access to all $S$ keys distributed across all $N$ devices — a global gather, expensive and not memory-friendly. Ring Attention's contribution is *the right schedule for moving KV around without ever materializing the full attention matrix anywhere*.

The Blockwise Parallel Transformer (BPT, [arXiv:2305.19370](https://arxiv.org/abs/2305.19370), same group, 2023-05) solved the single-device memory side with blockwise streaming attention + blockwise streaming FFN. Ring Attention takes the same kernel and **shards Q across devices, rotates KV around a ring** — turning a $32\times$-longer single-device technique into a $32\times$-longer-per-device-count distributed technique.

---

## The key idea: persistent Q, rotating KV, streaming softmax

> [!quote] The contribution in one sentence
> Each of $N$ devices keeps its local Q block fixed and receives a fresh K/V block from its left neighbor every step; over $N{-}1$ steps, every Q sees every K/V, with FlashAttention-style online softmax maintaining running statistics so the per-device activation memory stays $O(b \cdot c \cdot h)$ regardless of total sequence length.

Three sub-claims hold this up:

- **Per-device memory is constant in $S$.** Per the paper's Table 1: attention activations per host are $6 \cdot b \cdot c \cdot h$ — depends on block size $c$ and hidden $h$, **not** total $S$. Sequence length scales linearly with device count $N$ for the same activation budget.
- **Communication is free under a clean condition.** Per-step compute is $4 \cdot d \cdot c^2 / F$ (block attention FLOPs / host FLOPs/s); per-step comm is $4 \cdot c \cdot d / B$ (one KV block / interconnect bytes/s). Overlap requires $c \geq F/B$. On A100 + NVLink, $F/B \approx \text{few hundred tokens}$ — easy to satisfy.
- **Mathematical equivalence.** No approximation: Ring Attention output is bit-identical to single-device attention. The trick is the online softmax, which combines partial outputs across rotation steps in numerically-stable fashion.

Remove any one: lose persistent Q and you re-materialize the full attention matrix somewhere; lose the overlap condition and comm dominates; lose the online softmax and partial outputs can't be combined correctly.

---

## How it works

### Algorithm

The paper's Algorithm 1, verbatim:

```
Required: input sequence x, number of hosts N_h
Initialize:
  Split input sequence into N_h blocks (one per host)
  Compute query, key, value for the local input block on each host

for each transformer layer do
  for count = 1 to N_h − 1 do
    for each host concurrently do:
      Compute memory-efficient attention incrementally
        using local query, key, value blocks
      Send key and value blocks to next host
      Receive key and value blocks from previous host
```

State per host: a persistent local Q block $Q_i$, plus a K/V block that on rotation step $k$ is whichever block originated $k$ positions earlier in the ring.

### The online softmax mechanics — heart of the method

Each Q row keeps three running statistics across rotation steps:

| State | Shape | Initialized to |
| ----- | ----- | -------------- |
| `numerator` | `(B, q_len, H, d_head)` | zeros |
| `denominator` | `(B, H, q_len)` | zeros |
| `prev_max_score` | `(B, H, q_len)` | `-inf` |

For each incoming KV chunk (from `ringattention/ringattention_jax.py` lines 132–143):

```python
attn_weights = einsum('bqhd,bkhd->bhqk', q_chunk, k_chunk) / scale
attn_weights += bias_chunk
max_score_chunk = maximum(prev_max_score_chunk,
                          attn_weights.max(axis=-1))
exp_weights = exp(attn_weights - max_score_chunk[..., None])
exp_values = einsum('bhqk,bkhd->bqhd', exp_weights, value_chunk)
correction = exp(prev_max_score_chunk - max_score_chunk)
numerator_chunk   = numerator_chunk   * correction + exp_values
denominator_chunk = denominator_chunk * correction + exp_weights.sum(-1)
```

Final output per Q row: `numerator / denominator`.

This is **exactly the FlashAttention / online-softmax recurrence**, but the `k_chunk, v_chunk` come from a distributed rotating buffer (`lax.ppermute` in JAX, `dist.batch_isend_irecv` in PyTorch) rather than HBM.

### The compute / communication overlap condition

The paper gives the formal condition (Section 3):

$$
\underbrace{\frac{4 d c^2}{F}}_{\text{per-step compute}} \;\geq\; \underbrace{\frac{4 c d}{B}}_{\text{per-step KV transfer}} \quad\Longrightarrow\quad c \;\geq\; \frac{F}{B}
$$

with $c$ = block size (tokens), $d$ = hidden, $F$ = host FLOPs/s, $B$ = inter-host bytes/s. If satisfied, the ring rotation completes before the local attention compute does — communication is fully hidden behind computation.

Numerical instances:

| Hardware | $F$ (FP16) | $B$ | $c^* = F/B$ |
| -------- | ---------- | --- | ----------- |
| A100 + NVLink intra-node | ~312 TFLOPs/s | ~600 GB/s | a few hundred tokens |
| TPU v4 + ICI | similar | similar | similar |
| H100 + IB inter-node | ~989 TFLOPs/s | ~25 GB/s | thousands of tokens |
| PCIe-only nodes | similar to H100 | ~25 GB/s | tens of thousands — usually infeasible |

This is the hardware story: Ring Attention is a **bandwidth-rich-fabric** technique. NVLink-class interconnect is required for practical use.

### Per-rotation transfer bytes

Per rotation, per GPU: $2 \cdot c \cdot d \cdot \text{dtype\_bytes}$ (one K block + one V block). This is **independent of total sequence length** $S$ — only depends on block size $c$.

For $c = 4096$, $d = 8192$, FP16: $2 \cdot 4096 \cdot 8192 \cdot 2 = 128$ MiB per rotation. On 600 GB/s NVLink: ~210 µs per rotation. The local attention compute on a $c \times c$ block is ~ms range — comm hidden.

### Memory analysis

Per the paper's Table 1, Ring Attention's per-host attention-activation cost is:

$$
\boxed{\,6 \cdot b \cdot c \cdot h \text{ bytes per host}\,}
$$

— independent of total sequence length $S$. Breakdown: 1 current Q block, 2 current K/V, 2 received K/V, 1 output.

Compare:

| Method | Per-host attention activation |
| ------ | ----------------------------- |
| Vanilla attention | $O(b \cdot S^2)$ |
| FlashAttention (single device) | $2 \cdot b \cdot S \cdot h$ |
| **Ring Attention** | $\mathbf{6 \cdot b \cdot c \cdot h}$ |

KV cache is sharded equally: $O(S/N)$ per host.

### Causal-mask load imbalance (the unspoken weakness)

The original paper does **not** discuss this — the algorithm is presented under generic attention. But under causal masking with contiguous token-range partitioning:

- GPU 0 (earliest tokens) has roughly half-empty attention at every step — most of its keys are in the future.
- GPU $N{-}1$ (latest tokens) is fully utilized.

Implementations skip entire ring steps where every $(q, k)$ pair is masked, which produces an **imbalanced critical path** dominated by the worst rank.

**Striped Attention** ([arXiv:2311.09431](https://arxiv.org/abs/2311.09431), same group, 2023-11) fixes this with a **striped permutation**: token $t$ → GPU $(t \bmod N)$. Each GPU then holds a mix of early and late positions, so triangular work balances. Reported gains: **1.45× throughput on A100 at 256K**, **1.65× on 16× TPU v4 at 786K**.

A separate **zigzag** scheme (in `zhuzilin/ring-flash-attention` and Megatron-Core CP) achieves similar balance with a different permutation. The 2026 production-quality Ring Attention always ships with one of these fixes.

---

## Experiments

### Scale numbers (from the paper)

| Model | Hardware | Achieved context | vs baseline |
| ----- | -------- | ---------------- | ----------- |
| 7B | 32× A100 | **4M tokens** | ~32× longer |
| 7B | TPU v4-1024 (1024 chips) | **8M tokens** | ~512× longer |
| (abstract claim) | any | up to device-count × longer | — |

These are *training* context lengths achieved without compromise on exactness — the math is identical to single-device attention.

### Downstream production demonstration: Large World Model (LWM)

[Liu, Yan, Zaharia, Abbeel, *World Model on Million-Length Video And Language with Blockwise RingAttention*](https://arxiv.org/abs/2402.08268) (Feb 2024). Trains 7B models progressively from 4K to **1M-token context** using Ring Attention. The most credible end-to-end demonstration of the technique.

### Production deployments

| Deployment | Variant | Source |
| ---------- | ------- | ------ |
| **Megatron-Core CP** (NVIDIA) | Ring + cuDNN FlashAttention + zigzag scheduling, 4D parallelism (TP × CP × PP × DP) | [docs.nvidia.com/megatron-core/.../context_parallel.html](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/context_parallel.html) |
| **Meta Llama 3 training** | "pass-KV" (classic Ring) for training + "pass-Q" variant for KV-cache-persistent decode. **1M-token prefill on Llama 3 405B in 77 s on 128× H100, 93% parallel efficiency, 63% FLOPs utilization** | [arXiv:2411.01783](https://arxiv.org/abs/2411.01783) |
| **Tencent USP** | Ring × [[deepseek-ulysses\|Ulysses]] hybrid 2D mesh; 47% MFU training Llama-3-8B at 208K on 2×8×A800 | [arXiv:2405.07719](https://arxiv.org/abs/2405.07719), [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention) |
| **PyTorch native Context Parallel** | Ring as primary, all-to-all transport as secondary (PyTorch 2.7+) | [docs.pytorch.org/.../context_parallel.html](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) |

### Compared with [[deepspeed-ulysses|DeepSpeed Ulysses]]

| Property | Ring Attention | DeepSpeed Ulysses |
| -------- | -------------- | ----------------- |
| What moves between GPUs | KV blocks rotate around a ring | QKV reshuffled by AllToAll |
| Communication primitive | $N{-}1$ rounds of P2P send/recv | 4 × AllToAll per attention layer |
| Compute / comm overlap | ✓ Fully hidden when $c \geq F/B$ | ✗ Blocking — no overlap |
| Hard GPU-count limit | None | $\leq$ num_heads (much worse for GQA) |
| Cross-node scaling | ✓ P2P is bandwidth-friendly | ✗ AllToAll over IB degrades |
| Causal-mask balance | ✗ Needs Striped / zigzag fix | ✓ Naturally balanced |
| Attention kernel changes | Yes — fused with FA streaming softmax | None — uses stock FlashAttention |

---

## Strengths and limitations

The two strongest points: (1) **exact equivalence with single-device attention** — no approximation, no quality trade-off; (2) **memory scaling is fundamentally different** — per-device activation memory is constant in $S$, so total sequence length scales linearly with device count without changing per-device pressure.

Where the work is honest about scope but the limits matter:

- **Quadratic FLOPs are unchanged.** Total compute is still $O(S^2 \cdot d)$. Ring is a memory and scheduling win, not an algorithmic-complexity one. 10M-token training is still budget-limited by compute.
- **Causal-mask imbalance** is real and not addressed in the original paper. Production usage requires Striped or zigzag scheduling.
- **Bandwidth-bound at low interconnect.** The overlap condition $c \geq F/B$ becomes painful across PCIe or Ethernet — block sizes balloon, hurting throughput. NVLink-class fabric (or TPU ICI) is effectively required.
- **Inference / decode awkwardness.** During autoregressive generation with persistent KV cache, you can't freely rotate KV. Meta's pass-Q variant exists precisely because vanilla Ring assumes KV is rotatable.
- **Many small matmuls hurt arithmetic intensity.** Compared to Ulysses (full-head attention on each GPU), Ring's per-step blocks are smaller, lower utilization.
- **Not novel as a building block.** Online softmax is FlashAttention. Blockwise streaming is BPT. The novelty is the *distributed schedule* — important, but not algorithm-novel.

> [!warning] "Near-infinite context" — marketing or substantive?
> Substantive in the **memory** sense: per-GPU activation memory is $O(S/N)$, so you can scale $S$ with $N$.
> Marketing in the **practical** sense: total cost remains $O(S^2)$ FLOPs, so 10M-token training is still budget-limited by compute, not memory. The LWM 1M-context 7B paper is the most credible demonstration. Claims that "Gemini 1.5's 10M context uses Ring Attention" are **unverified speculation** — Google hasn't disclosed.

---

## What this means

Two predictions worth tracking:

1. **Ring Attention has already won the production CP battle — under different names.** Megatron-Core CP, Llama 3 training, Tencent USP, PyTorch native CP — all are Ring Attention with productionization fixes (zigzag, hybrid, pass-Q). The bare 2023 algorithm isn't competitive; the 2026 production stack is the algorithm plus three years of refinements.
2. **The next axis is interconnect topology, not algorithm.** Ring works on NVLink; struggles on IB; fails on PCIe. As HBM bandwidth grows and inter-node fabric grows slower, the $c^* = F/B$ block-size tax goes up. Future work will be about making Ring tolerate slower fabric (pipelining, KV compression in transit) rather than reinventing the schedule.

What this is *not*: a fix for $O(S^2)$ FLOPs (no method here is), nor an inference-decode primitive in its raw form (Meta's pass-Q is the patch), nor a universal long-context solver (Ulysses + Ring + KV compression all play roles).

---

## Source code & reproduction

### Canonical implementation (JAX/TPU)

[lhao499/RingAttention](https://github.com/lhao499/RingAttention) — Apache-2.0, the reference. Key files:

| File | Role |
| ---- | ---- |
| `ringattention/ringattention_jax.py` | Core GPU/TPU forward kernel (the 20-line core quoted above) |
| `ringattention/ringattention_jax_inference.py` | Inference variant |
| `ringattention/ringattention_pallas_tpu.py` | Pallas (TPU) kernel |

The ring rotation primitive (JAX):

```python
k, v = map(lambda x: lax.ppermute(
    x, axis_name,
    perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]
), (k, v))
```

`lax.ppermute` is SPMD-aware P2P send/recv along the device mesh axis.

### PyTorch ports

| Repo | Notes |
| ---- | ----- |
| **[zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)** | Most-used PyTorch impl. Wraps Tri Dao's FlashAttention as the per-step inner kernel. Variants: `ring_flash_attn_func`, `zigzag_ring_flash_attn_func`, `stripe_flash_attn_func`, plus `_varlen` / `_qkvpacked` versions. ~90% of single-device FlashAttention throughput at fwd+bwd on H800 with zigzag. NVLink required. |
| [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch) | Phil Wang's pedagogical PyTorch port. Useful for reading. |
| [gpu-mode/ring-attention](https://github.com/gpu-mode/ring-attention) | GPU MODE community scaffolding (Lecture 13). |

### Production-grade integration: Megatron-Core CP

NVIDIA's productionized Ring Attention. Exposed via `--context-parallel-size <N>` flag. Composes as 4D parallelism: $\text{TP} \times \text{CP} \times \text{PP} \times \text{DP}$. Uses cuDNN FlashAttention as inner kernel and zigzag-style scheduling for causal balance. [Docs](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/context_parallel.html).

### Minimum reproduction (PyTorch, zhuzilin/ring-flash-attention)

```python
import torch
from ring_flash_attn import zigzag_ring_flash_attn_func

# In a torch.distributed-initialized process group of size N:
# q, k, v have shape (batch, seq_len/N, num_heads, head_dim) — sequence sharded.
out = zigzag_ring_flash_attn_func(q, k, v, causal=True)
# out has same shape as q
```

The library handles ring rotation (NCCL `batch_isend_irecv`), per-step FlashAttention call, online softmax accumulation, and zigzag chunking for causal balance. RoPE offsets need adapting for striped/zigzag layouts (per the README).

---

## Related reading

- [[deepspeed-ulysses]] — The AllToAll-based alternative. Sister page to this one.
- [[parallelism-strategies-deep-dive#7. CP — Context Parallelism]] — Where Ring Attention sits in the broader parallelism landscape; comparison table with Ulysses and Megatron CP.
- [[paged-attention]] — FlashAttention's single-device streaming kernel; Ring Attention's per-step inner loop.
- [[kv-cache-optimization]] — Where KV cache compression intersects with long context.
- [[long-context-serving]] — Production long-context inference; Ring is the training side, [[saw-int4|SAW-INT4]] / quantization is the serving-side counterpart.
- [[das-spec-rl]] — Speculative decoding for the rollout phase; orthogonal to Ring but complementary at the inference layer.

## References

- **Ring Attention paper**: Liu, Zaharia, Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context*. ICLR 2024. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) · [OpenReview](https://openreview.net/forum?id=WsRHpHH4s0) · [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/1119587863e78451f080da2a768c4935-Paper-Conference.pdf)
- **Blockwise Parallel Transformer (precursor)**: Liu, Abbeel. [arXiv:2305.19370](https://arxiv.org/abs/2305.19370)
- **Striped Attention (causal-mask fix)**: Brandon, Nrusimha, Qian, Ankner, Jin, Song, Liu, Ragan-Kelley. [arXiv:2311.09431](https://arxiv.org/abs/2311.09431)
- **Large World Model (1M-context Ring Attention application)**: Liu, Yan, Zaharia, Abbeel. [arXiv:2402.08268](https://arxiv.org/abs/2402.08268)
- **BurstAttention (double-buffered ring variant)**: [arXiv:2403.09347](https://arxiv.org/pdf/2403.09347)
- **USP (Unified Sequence Parallel — Ring × Ulysses hybrid)**: [arXiv:2405.07719](https://arxiv.org/abs/2405.07719) · [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention)
- **Meta Context Parallelism (pass-KV / pass-Q for inference)**: Yang et al. [arXiv:2411.01783](https://arxiv.org/abs/2411.01783)
- **TokenRing (bidirectional ring)**: [arXiv:2412.20501](https://arxiv.org/abs/2412.20501)
- **Megatron-Core CP docs**: [docs.nvidia.com/megatron-core/.../context_parallel.html](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/context_parallel.html)
- **Explanations**: [Coconut Mode walkthrough](https://coconut-mode.com/posts/ring-attention/) · [GPU MODE Lecture 13 notes](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/) · [Insujang CP overview](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)
- **DeepSpeed Ulysses (alternative)**: [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) — see [[deepspeed-ulysses]]
- **FlashAttention-3 (inner kernel)**: [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) · [Tri Dao blog](https://tridao.me/blog/2024/flash3/)
