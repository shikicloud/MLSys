---
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
category: llm-inference
tags: [mamba, ssm, state-space-model, selective-ssm, parallel-scan, hardware-aware, linear-attention, hybrid-attention, paper-review]
created: 2026-06-02
updated: 2026-06-02
status: mature
paper: arXiv:2312.00752
code: https://github.com/state-spaces/mamba
---

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

> [!info] Paper metadata
> - **Paper**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, December 2023, NeurIPS 2024
> - **Code**: [state-spaces/mamba](https://github.com/state-spaces/mamba) — Apache-2.0, Albert Gu's reference implementation with CUDA kernels
> - **Authors**: Albert Gu (CMU), Tri Dao (Princeton, also of FlashAttention)
> - **Lineage**: Continuation of S4 (Gu et al., ICLR 2022) → S5 (Smith et al., 2023) → **S6 / Mamba** (this paper)

> [!important] Why this page matters in 2026
> Every hybrid-attention production LLM in 2026 — [[#Successor work: the hybrid attention era|Qwen3-Next, Qwen3.5, Qwen3.6, Nemotron-H, Kimi Linear, MiMo-V2-Flash, DeepSeek-V3.2 NSA]] — has linear-attention layers descended from Mamba. Understanding the Selective SSM mechanism, the chunkwise parallel scan, and Mamba's specific weaknesses (associative recall, ICL) is prerequisite for understanding why the industry moved to *hybrid* architectures rather than going all-Mamba.

---

## Summary (read this if you have 2 minutes)

**What it is.** Mamba is a sequence model that combines the **O(N) inference complexity** of recurrent models, the **parallel training** of transformers, and the **long-range capability** of structured state-space models (SSMs). Its core technical contribution is the *Selective SSM* (also called S6) — an input-dependent state-space transition that breaks the LTI (linear time-invariant) assumption of prior SSMs, enabling content-aware filtering and selective memory.

**The one idea.** **Make SSM parameters input-dependent**, then design a hardware-aware parallel scan algorithm so the resulting non-LTI system can still train at scale. Three pieces hold this up:

1. **Selective SSM (S6)**: the matrices $\bar{A}_t, \bar{B}_t, C_t$ become functions of the current input $x_t$, computed via linear projections. This gives the SSM the ability to *decide what to remember* — exactly the missing capability that made S4 fail on text tasks like the Selective Copy task.
2. **Hardware-aware parallel scan**: because $\bar{A}_t$ varies per token, the convolution-form trick that S4 used (FFT) doesn't apply. Replace with Blelloch parallel scan + kernel fusion (discretization + scan + projection in one CUDA kernel), keeping the SSM state in SRAM throughout. This is what makes selective SSM trainable at all.
3. **Mamba block**: a single block type that replaces both attention and FFN layers — a gated MLP with SSM core instead of attention. The architecture is *homogeneous* — every block is a Mamba block, no transformer interleaving.

Remove the selectivity and you're back to S4 (can't copy / filter); remove the parallel scan and training time explodes; remove the homogeneous block and you're back to ad-hoc SSM-transformer hybrids.

**Headline result.** At 3B parameters and 300B tokens on The Pile, Mamba matches Transformer++ (modern transformer baseline with all the tricks: RoPE, GLU, RMSNorm, etc.) on language modeling perplexity. **5× faster decoding throughput** vs. Transformer++ at sequence length 2K. Scales **linearly** with sequence length where attention is quadratic. SOTA on long-context audio modeling and DNA sequence benchmarks.

**Why it matters.**

- **First credible attention replacement at scale.** Pre-Mamba SSMs (S4, S5, H3, Hyena) were research-grade and didn't match transformers on language. Mamba was the first to demonstrably match Transformer++ at 3B scale on a real language benchmark.
- **Restarted the linear-attention research program** — and revealed its specific failure modes. The associative-recall weakness Mamba exposed motivated 2 years of follow-up work (Mamba-2, Gated DeltaNet, Lightning Attention, KDA) and the hybrid-attention production architectures that dominate 2026.
- **Made hardware-aware kernel co-design canonical.** The selective scan kernel is as much a contribution as the algorithm. Subsequent linear-attention variants (Gated DeltaNet's WY recurrence, KDA's delta-rule kernel) all follow the same pattern: parallel scan + SRAM-resident state + kernel fusion.
- **Implication for inference serving.** Mamba's per-sequence state is *fixed size* (not length-dependent like KV cache). For long-context serving, this changes the memory budget calculation entirely — and forces the hybrid-attention KV manager redesign (see [[paged-attention]] hybrid extensions, e.g. vLLM PR #22688).

---

# Depth (drill-down starts here)

## Background: the SSM lineage and what was missing

State-space models in deep learning trace back through three generations.

### Classical SSMs (continuous-time)

A continuous-time linear SSM describes a system as:

$$
h'(t) = A h(t) + B x(t), \quad y(t) = C h(t)
$$

To use this in deep learning, discretize with step size $\Delta$:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t
$$

where $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \Delta B$. **Critically, $\bar{A}, \bar{B}, C, \Delta$ are all data-independent** — this is the LTI (Linear Time-Invariant) assumption.

### S4 (Gu et al., ICLR 2022)

S4 made SSMs trainable at scale by:

1. **Structured matrix initialization** (HiPPO theory): pick $A$ with a specific structured form (diagonal-plus-low-rank) that captures long-range dependencies in the SSM kernel.
2. **Convolutional view**: because LTI, the entire output can be written as a single convolution $y = \bar{K} * x$ where $\bar{K}$ is the SSM kernel.
3. **FFT for parallel training**: convolutions of length $N$ compute in $O(N \log N)$ via FFT — parallel-friendly.
4. **Recurrent inference**: at decode time, use the recurrence form $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ — O(N) instead of O(N²).

S4 set SOTA on Long Range Arena (LRA), audio, and time-series benchmarks. It did *not* match transformers on language modeling.

### S5, H3, Hyena — incremental improvements

- **S5** (Smith et al., 2023): diagonal SSMs + parallel scan for the recurrence (instead of FFT for convolution). Faster on GPU.
- **H3** (Fu et al., 2023): added "Hungry Hungry Hippos" gating that allows simple synthetic tasks like induction heads. Got closer to transformer performance but still a gap.
- **Hyena** (Poli et al., 2023): long convolutions with implicit parametrization.

All of these stayed LTI. **They all failed catastrophically on the Selective Copy and Induction Heads tasks** — tasks where the model has to decide *which* tokens to remember based on content.

> [!quote] The diagnostic Selective Copy task
> Given a long sequence with a few "key" tokens scattered through noise, output only the key tokens in order. **A LTI SSM cannot do this** because the transition matrix $\bar{A}$ has to be the same for noise tokens and key tokens. Mamba's selectivity directly targets this.

The paper diagnoses LTI as the bottleneck:

> "LTI models have a fundamental limitation in handling content-based reasoning. ... The model's dynamics are constant over time, which means the model cannot select what information to focus on or what to ignore based on the content of the sequence."

## The Selective SSM (S6) — Mamba's core contribution

The fix is conceptually simple: **make $\bar{B}, C, \Delta$ functions of the input** $x_t$.

### The selective update

For each input token $x_t \in \mathbb{R}^D$:

$$
\begin{aligned}
B_t &= s_B(x_t) \quad \in \mathbb{R}^N \\
C_t &= s_C(x_t) \quad \in \mathbb{R}^N \\
\Delta_t &= \tau_\Delta(\text{Parameter} + s_\Delta(x_t)) \quad \in \mathbb{R}_+^D
\end{aligned}
$$

where $s_B, s_C, s_\Delta$ are linear projections (with one extra broadcast for $\Delta_t$ across the $D$ channels), and $\tau_\Delta$ is softplus to ensure $\Delta_t > 0$.

$A$ remains data-independent (kept structured, typically a diagonal initialization in $A$-real-imag form). The discretization uses the input-dependent $\Delta_t$:

$$
\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = \Delta_t B_t
$$

And the recurrence is now non-LTI:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t
$$

**The interpretation**: $\Delta_t$ controls how much of the input "leaks into" state — small $\Delta_t$ means "ignore this token" (state passes through unchanged), large $\Delta_t$ means "reset / overwrite state". Combined with input-dependent $B_t, C_t$, the model can do content-aware filtering.

### What the selectivity buys

| Capability | LTI SSM (S4, S5, H3) | Selective SSM (Mamba) |
| ---------- | -------------------- | --------------------- |
| Selective Copy task | ✗ Cannot solve | ✓ Solves at 100% |
| Induction Heads | ✗ Cannot extrapolate | ✓ Extrapolates to longer sequences |
| Language modeling (Pile) | Behind Transformer++ | **Matches Transformer++** |
| Content-aware filtering | ✗ | ✓ |

These three benchmarks established Mamba as the first SSM that could compete with transformers on language tasks.

### The cost: no convolution form

LTI lets you write $y = \bar{K} * x$ and compute via FFT. Selective ($\bar{A}_t$ varies per token) **breaks this**. You're left with the sequential recurrence, which a naive implementation runs in $O(N)$ but serially — terrible for GPU.

## Hardware-aware parallel scan — what makes selective SSM trainable

The selective SSM's recurrence $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ has the form of an *associative scan*: each step combines previous state with current input via an associative operation. Associative scans parallelize via the **Blelloch scan** algorithm in $O(\log N)$ depth.

### The naive associative form

For each token, the update is:

$$
(h_t, \cdot) = (\bar{A}_t, \bar{B}_t x_t) \oplus (h_{t-1}, \cdot)
$$

where $\oplus$ is the associative operation:

$$
(A_1, b_1) \oplus (A_2, b_2) = (A_2 A_1, A_2 b_1 + b_2)
$$

A standard Blelloch scan over $N$ tokens uses $O(N)$ work and $O(\log N)$ depth — parallel across N GPUs in principle.

### Mamba's actual implementation: kernel-fusion + SRAM-resident state

The selective scan as a standalone CUDA kernel still has a fatal performance problem: **materializing intermediate states to HBM dominates runtime**. The hidden state $h_t$ has shape $(B, D, N)$ where $D \approx 2048$ and $N$ (state dim) $\approx 16$. For a sequence of length 2K, materializing all $h_t$ uses $2K \times B \times D \times N = $ hundreds of MB per layer — and HBM bandwidth becomes the bottleneck.

Mamba's selective scan kernel does three things:

1. **Fuse discretization + scan + output projection** into one CUDA kernel. Inputs come in, outputs go out, no intermediate materialization.
2. **Keep the SSM state in SRAM** throughout the scan. The state is small ($D \times N = 32K$ floats per token batch) and fits in SRAM.
3. **Recompute the scan in the backward pass** instead of storing all intermediate states. This is the same trick FlashAttention uses for attention — trade compute for memory in the backward pass.

The result, measured in the paper:

| Implementation | Throughput at seqlen 2K | Memory |
| -------------- | ----------------------: | -----: |
| Naive PyTorch (materializing state) | 1× baseline | 100% |
| Selective scan kernel | **20–40× faster** | <10% |

Without the fused kernel, selective SSM is **slower than transformers** despite the asymptotic advantage. The kernel makes the algorithm practical.

> [!note]- Chunkwise scan for very long sequences
>
> For sequences too long to scan in one kernel launch (e.g., 1M+ tokens), Mamba uses a *chunkwise* scan: process the sequence in chunks of size $C$ (e.g., 256), with each chunk computing its scan internally, and chunks chained via state passing. This is what later hybrid models (Qwen3-Next, Gated DeltaNet) inherit and optimize further.

## The Mamba block — a single homogeneous block replacing both attention and FFN

Mamba's architecture is unusually clean: **every layer is a Mamba block**, no transformer interleaving, no separate FFN layers.

A Mamba block (Fig. 3 in the paper) does the following on input $x \in \mathbb{R}^{B \times L \times D}$:

```
x_in  ──→ Linear (D → 2·E·D)  ──┬──→ Conv1d (kernel=4) ──→ SiLU ──→ SSM (Selective) ──┐
                                  │                                                       │
                                  └────────────────────────── (used as residual?) ──→ ⊗ gate
                                                                                          │
                                                                                          ▼
x_in  ──→ Linear (D → E·D) ────────────→ SiLU ──────────────────────────────────────────⊗
                                                                                          │
                                                                                          ▼
                                                                            Linear (E·D → D) ──→ x_out
```

Where:
- $E$ is an "expansion factor", typically 2.
- The SSM is the Selective SSM from §3 above.
- The Conv1d before the SSM acts as a local mixer (covers small-window patterns).
- The gating is GLU-style (SiLU(Wx) ⊗ SSM_path) — borrowed from gated MLP designs.

**No attention. No separate FFN. No layer norms inside the block** (LN is applied between blocks). The full Mamba model is just $\text{LN} \to \text{MambaBlock} \to \text{LN} \to \text{MambaBlock} \to \cdots$.

This homogeneity is intentional: it forces all sequence-modeling capability into the SSM, and lets the model scale without architectural choices about attention/FFN ratios, head counts, etc.

## Headline evidence

### Language modeling on The Pile

At 3B parameters and 300B tokens, Mamba matches Transformer++ (a strong baseline with RoPE, GLU, RMSNorm, etc.). The model-size scaling curves (paper Fig. 5) show Mamba pulling slightly *ahead* of Transformer++ at every scale 125M to 1.4B, with the gap closing at 3B (statistical noise plausible).

### Throughput at inference

| Model | Decoding throughput (tokens/sec) at batch 32, seqlen 2K |
| ----- | ------------------------------------------------------: |
| Transformer++ (3B) | 1× baseline |
| **Mamba (3B)** | **~5× faster** |

The 5× comes from two sources:
1. **No KV cache** — Mamba's state is fixed-size, doesn't grow with sequence length.
2. **O(N) per-token compute** — linear in sequence length, vs O(N) per token but with O(N) attention compute per token in transformers (so O(N²) total for generation).

### Long-range tasks

- **Audio**: SOTA on YouTubeMix, SC09. Mamba's linear-time scaling lets it train on long audio sequences (16K+ samples) that quadratic attention cannot afford.
- **DNA**: SOTA on Genomic-Benchmarks. Long DNA sequences (10K+ base pairs) are the natural domain for linear models.
- **Synthetic Selective Copy**: 100% (vs. ~0% for S4, transformers also solve via attention).

### Sequence length scaling

Mamba's compute scales **linearly** with sequence length where attention scales quadratically. Empirically, at 16K context the gap is large; at 1K context transformers still win in absolute speed due to better hardware fit.

> [!success] The genuinely surprising result
> Pre-Mamba SSMs lost language modeling to transformers by significant margins. Mamba matched a strong transformer baseline at 3B. **The selectivity was the missing piece** — not bigger SSMs, not better discretizations, not deeper stacks. The selectivity alone closed the gap.

## Strengths and limitations

**Strengths.**

- **Genuine architectural alternative to attention** — first SSM to match transformer language modeling at scale.
- **Linear-time inference** — the long-context killer feature.
- **Hardware-aware kernel design** — the selective scan kernel is a contribution in itself, not just engineering.
- **Architectural cleanliness** — homogeneous blocks, no attention/FFN ratio tuning.
- **Scales to long sequences without context-window engineering** (no RoPE extension, no positional interpolation).

**Limitations** (some not in the paper but identified by follow-up work).

> [!warning] Associative recall and in-context learning are weak
> Within 6 months of Mamba's release, the field discovered that Mamba **fails at associative recall** (Phonebook-style tasks — given a list of (key, value) pairs, retrieve a value by key). This is fundamental to the architecture: SSM state has *fixed size*, so it cannot losslessly store arbitrary key→value mappings. Attention has unbounded historical lookup; Mamba has a finite memory bottleneck.

- **In-context learning is weaker than attention.** Mamba can do *some* ICL but underperforms transformers on harder ICL benchmarks (especially when the in-context demonstrations are long).
- **No paper-level scaling beyond 3B.** The paper only validates to 3B parameters. Whether Mamba's gains hold at 70B+ is genuinely unknown from this paper alone.
- **Selective scan kernel is CUDA-specific.** Triton port came later; ROCm and TPU ports are still incomplete in 2026.
- **The Δ_t parameterization is fragile.** Initialization sensitivity is real; training instability at scale was reported by groups attempting to reproduce.
- **No discussion of inference-time *prefix sharing*** — fixed-size state means you can't share SSM state between two requests with shared prefixes the way you can share KV cache, which complicates serving optimizations like RadixAttention.

## What this means — and the hybrid attention era

Mamba was a positive result and a negative diagnosis simultaneously. It proved that selective SSMs can match transformers on standard language modeling, but it also exposed *exactly* what they can't do (associative recall, hard ICL, retrieval).

The 2024–2026 response from the field was overwhelmingly **hybrid architectures**: keep most layers as linear-attention (Mamba descendants) for compute efficiency, but insert a small number of full-attention layers (typically 1 in every 4–8) to recover retrieval / ICL capability.

### Successor work: the hybrid attention era

| Work | Year | Linear variant | Hybrid ratio | Note |
| ---- | ---- | -------------- | ------------ | ---- |
| **Mamba-2 / SSD** (Gu+Dao, May 2024, arXiv:2405.21060) | 2024 | SSD (Structured State-Space Duality) | All-Mamba | Unifies SSM and attention as structured matrices; 2–8× faster than Mamba-1 |
| **Jamba** (AI21, Mar 2024) | 2024 | Mamba | 1:7 attention:Mamba | First production hybrid; 12B active / 52B total |
| **Zamba** (Zyphra, 2024) | 2024 | Mamba | Single global attention block | 7B; SOTA at the size |
| **Samba** (Microsoft, 2024) | 2024 | Mamba + Sliding-Window Attention | Alternating | Open-source 3.8B |
| **Nemotron-H** (NVIDIA, 2024) | 2024 | Mamba-2 + attention | 3:1 ratio in hybrid layers | NVIDIA's first production hybrid |
| **Gated DeltaNet** (Yang et al., ICLR 2025, arXiv:2412.06464) | 2025 | Gated DeltaNet (delta rule) | All-Mamba in base | Powers Qwen3-Next, Qwen3.5, Qwen3.6 |
| **Lightning Attention** (Qin et al., 2024) | 2024 | Lightning-class linear | All-linear | Used by MiniMax |
| **KDA (Kimi Delta Attention)** | 2025 | Delta rule variant | Hybrid | Used by Kimi Linear, K2 family |
| **Mamba-3** (mentioned 2026) | 2026 | Next-gen Mamba | Hybrid emphasized | Active research |

### Three predictions for 2027

1. **All-Mamba production models are dead.** The associative-recall weakness is fundamental enough that nobody ships a flagship LLM as pure SSM. Hybrid 3:1 or 7:1 will remain the canonical pattern.
2. **The Mamba state becomes the next frontier for serving systems.** KV cache management is well-understood; hybrid-aware memory managers (vLLM's heheda12345 PR series, SGLang's hybrid extensions) are still maturing. Expect new papers on cross-architecture KV/state transfer protocols (PD disaggregation for hybrids, see [[prfaas]]).
3. **The selective scan kernel becomes a standard library primitive.** Like FlashAttention's `flash_attn_func`, expect `selective_scan_fn` from FLA / NVlabs to be the canonical API every framework calls into.

## Source code & reproduction

```bash
# Install (requires CUDA 11.6+ and PyTorch 1.12+)
pip install mamba-ssm
pip install causal-conv1d  # required dep
```

**Minimal Mamba block usage** (from the reference repo):

```python
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 1024, 16
x = torch.randn(batch, length, dim).cuda()
model = Mamba(
    d_model=dim,        # Model dimension
    d_state=16,         # SSM state size (N in the paper)
    d_conv=4,           # Conv1d kernel size
    expand=2,           # E expansion factor
).cuda()
y = model(x)
assert y.shape == (batch, length, dim)
```

**Pretrained checkpoints** released by the authors:

| Checkpoint | Params | Trained tokens | URL |
| ---------- | -----: | -------------: | --- |
| `mamba-130m` | 130M | 50B | https://huggingface.co/state-spaces/mamba-130m |
| `mamba-370m` | 370M | 50B | https://huggingface.co/state-spaces/mamba-370m |
| `mamba-790m` | 790M | 50B | https://huggingface.co/state-spaces/mamba-790m |
| `mamba-1.4b` | 1.4B | 300B | https://huggingface.co/state-spaces/mamba-1.4b |
| `mamba-2.8b` | 2.8B | 300B | https://huggingface.co/state-spaces/mamba-2.8b |

| File path | Role |
| --------- | ---- |
| `mamba_ssm/modules/mamba_simple.py` | The `Mamba` block — top-level user interface |
| `mamba_ssm/ops/selective_scan_interface.py` | The selective scan op wrapping the CUDA kernel |
| `csrc/selective_scan/selective_scan_fwd_kernel.cuh` | Forward CUDA kernel (the hardware-aware impl) |
| `csrc/selective_scan/selective_scan_bwd_kernel.cuh` | Backward CUDA kernel (recompute-in-backward trick) |
| `mamba_ssm/models/mixer_seq_simple.py` | Full Mamba LM stack |

**Production-quality implementations** worth knowing:

- **[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** — Triton kernels for Mamba, Mamba-2, GLA, Gated DeltaNet, and ~20 other linear-attention variants. **The de-facto kernel library** for hybrid models in vLLM and SGLang.
- **[NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)** — NVIDIA's reference implementation of GDN (the linear-attention variant in Qwen3-Next / Qwen3.5).
- **vLLM hybrid KV manager** — heheda12345's PR #22688 unifies attention KV cache and Mamba state under one allocator; required for serving Qwen3-Next, Nemotron-H.

## Related reading

- [[paged-attention]] — PagedAttention's hybrid extension is the serving-side response to Mamba: how to manage *both* attention KV blocks and Mamba state in one memory allocator.
- [[kv-cache-optimization]] — Mamba sidesteps KV cache entirely (fixed-size state). For long-context serving, this changes the memory math completely.
- [[prfaas]] — Cross-DC PD disaggregation explicitly motivates by hybrid-attention models having dramatically lower KV throughput than dense GQA (13× lower per the paper) — a direct downstream consequence of Mamba's linear architecture.
- [[ring-attention]] — Sequence-parallel attention for long contexts; the alternative path that hybrids partially obsoleted by reducing the attention-layer count.
- [[sglang]], [[vllm]] — Inference engines that ship hybrid model support; SGLang via FLA Triton, vLLM via the hybrid KV manager.
- [[speculative-decoding]] — Speculative decoding interacts non-trivially with Mamba's fixed-size state: rejection requires state checkpointing, which is more expensive than KV truncation. See *Component-Aware SSD* (arXiv:2605.01106) for the 18× α gap between parallel and sequential hybrids.
- [[quantization]] — Mamba state quantization is an underexplored area; current implementations keep state in BF16 even when KV cache is FP8 or NVFP4.
- [[continuous-batching]] — Batching of hybrid models requires Mamba-state-aware scheduling; not all schedulers handle this cleanly.

## References

- Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, December 2023. NeurIPS 2024. https://arxiv.org/abs/2312.00752
- state-spaces/mamba. https://github.com/state-spaces/mamba
- Albert Gu. *Efficiently Modeling Long Sequences with Structured State Spaces* (S4). ICLR 2022.
- Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2). arXiv:2405.21060, May 2024.
- Yang et al. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464, ICLR 2025.
- Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model.* AI21, March 2024.
- Glorioso et al. *Zamba: A Compact 7B SSM Hybrid Model.* Zyphra, 2024.
- Ren et al. *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* Microsoft, 2024.
- fla-org/flash-linear-attention. https://github.com/fla-org/flash-linear-attention
