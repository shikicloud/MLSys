---
title: "Rotation-Based Quantization (QuIP / QuaRot / SpinQuant / BDR)"
category: llm-inference
tags: [quantization, rotation, hadamard, quip, quarot, spinquant, saw-int4, bdr, family-overview]
created: 2026-05-06
updated: 2026-05-07
status: mature
---

# Rotation-Based Quantization

> [!abstract]+ Family overview
> A family of low-bit quantization techniques that share one idea: before quantizing, apply an **orthogonal transformation** that flattens outliers. The rotation preserves the $L_2$ norm but redistributes energy across all dimensions, so the post-rotation tensor is much friendlier to per-channel or per-token scale-and-zero quantization. Family members differ in *what* they rotate (weights, activations, KV cache), *how* they choose the rotation matrix (random Hadamard, learned, block-diagonal), and *where* the rotation lives at inference time (absorbed into adjacent linear layers, or fused into the quantization kernel).

> [!info] Where to go next
> - For the SAW-INT4 / BDR paper specifically, see [[saw-int4]].
> - For the broader quantization landscape, see [[quantization]].

---

## The shared insight

LLM weights and activations have **outlier channels** — a small number of dimensions with magnitudes one or two orders larger than the rest. Per-token or per-channel scale-and-zero quantization assigns most of the dynamic range to those outliers, leaving the bulk of channels under-resolved. At INT4 (16 levels) the error is severe; on multi-step reasoning it compounds across attention rounds and the model collapses.

> [!example] Concrete failure
> Qwen3-4B-Thinking-2507 drops from 66.67 % to 0 % on GPQA under naive INT4 KV. See [[saw-int4#Background: why INT4 KV breaks reasoning models]].

Multiplying a vector by an orthonormal matrix $R$ (with $R^\top R = I$) doesn't change the $L_2$ norm but **redistributes** the components across all dimensions. If outliers were concentrated in a few coordinates, the rotated vector has a more uniform distribution. Quantization error is roughly proportional to the *range* of values per quantization group, so flatter distribution = smaller error.

The Hadamard matrix is the natural choice for $R$: it's orthogonal, all entries are $\pm 1$ (so the rotation costs only additions/subtractions, no multiplies), and the matrix-vector product runs in $O(d \log d)$ via the Fast Walsh-Hadamard Transform. With the $1/\sqrt{d}$ normalization it's an isometry; without it, applying $H$ scales magnitudes by $\sqrt{d}$.

The trick to making this practical is making sure the rotation either **absorbs into existing computation** (so inference is no slower) or **fuses into a kernel** that was going to run anyway. Both moves are present in the family below.

---

## The lineage

| Year | Method | Where rotation lives | Rotation type | What absorbs the rotation | Headline result |
|------|--------|---------------------|---------------|---------------------------|-----------------|
| 2023 | **QuIP** ([arXiv:2307.13304](https://arxiv.org/abs/2307.13304)) | Weights | Random orthogonal (incoherence processing) | Modify input/output of the layer | First to formalize that rotation makes 2-bit weights tractable. |
| 2024 | **QuIP#** ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) | Weights | Random Hadamard | Same | Adds lattice-codebook vector quantization on rotated weights. SOTA at 2-bit weights. |
| 2024 | **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) | Weights, activations, **and KV cache** | Random Hadamard | **Fused into adjacent linear-layer weights** — zero inference cost | NeurIPS 2024. Full W4A4KV4 on LLaMA-2-70B with ≤0.47 WikiText-2 PPL increase, 99 % of zero-shot performance retained. |
| 2024 | **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) | Weights, activations, KV | **Learned** rotation matrices (gradient-trained) | Same as QuaRot (absorbed into linear weights) | Closes 45 % of the QuaRot gap-to-FP on hard-to-quantize LLaMA-3 8B; W4A4KV4 LLaMA-2-7B comes within 2.9 pp of full precision. |
| 2026 | **SAW-INT4 / BDR** ([arXiv:2604.19157](https://arxiv.org/abs/2604.19157)) | KV cache only | Block-diagonal Hadamard (small fixed block size) | **Fused into the INT4 KV-write Triton kernel + decode-side Q-rotation kernel** | Recovers Qwen3-4B-Thinking GPQA from 0 % (plain INT4) to 65.82 %. Throughput indistinguishable from plain INT4. See [[saw-int4]]. |

Cross-cutting observations:

- **Where the rotation absorbs is the systems contribution.** QuIP introduces the algorithm; QuaRot makes it *free at inference* by absorbing into linear-layer weights; BDR makes it free for *KV cache* by fusing into the Triton write kernel. Each generation finds a new "free" place to put the rotation.
- **Random vs. learned.** Learned rotations (SpinQuant) outperform random Hadamard on accuracy, especially on hard-to-quantize models (LLaMA-3 8B), at the cost of an offline calibration step that minimizes a quantization-aware loss. Random Hadamard is calibration-free and sufficient for many models.
- **Full vs. block-diagonal.** Full Hadamard on the head dimension is $O(d \log d)$ and the matrix is mathematically clean, but for KV cache it's incompatible with paged-by-head layouts and FA3-style fused attention. Block-diagonal Hadamard restricts mixing to fixed-size blocks within the head dimension, trading some rotation strength for kernel-friendliness.
- **Rotation is orthogonal to the quantizer.** Once the input distribution is flattened, **any** quantizer works better — GPTQ, AWQ, plain scale-and-zero, k-means. The literature mostly stacks rotation with simple per-token/per-channel scale-zero because the rotation already does the hard work.

---

## What gets rotated, in detail

**QuIP / QuIP#**: weight matrices only. Each linear layer $y = Wx$ becomes $y = (U^\top W V)(V^\top x)$ with $U$, $V$ random orthogonal; the input rotation is absorbed by modifying the previous layer, the output rotation by modifying the next. Quantization happens on the rotated $W'$. This works because LLM weights have weight-distribution outliers similar to activations.

**QuaRot**: extends to **activations and KV cache** by inserting Hadamard rotations at carefully chosen places — at the input/output of each transformer block, around RMSNorm boundaries, and inside attention. The rotations are placed so they cancel out at run time when absorbed into adjacent linear layers' weights. The result is `W4A4KV4` (4-bit weights, 4-bit activations, 4-bit KV) on LLaMA-2 with minimal quality loss.

**SpinQuant**: same insertion points as QuaRot, but the rotation matrices $R_1, R_2, \ldots$ are no longer random Hadamard — they're parameters of an **offline optimization problem** that minimizes the quantization error of the rotated network on a calibration set. Empirically the learned rotations are nearly orthogonal but have structure that random Hadamard doesn't.

**SAW-INT4 / BDR**: rotates only the **KV cache** along the head dimension, in fixed-size blocks (16 or 128). The rotation is fused into the Triton kernel that writes the paged INT4 KV cache (`quantized_set_kv_int4_hadamard_fused_triton`); the same block-diagonal Hadamard is applied to Q inside the GQA decode kernel. No effect on weights or activations, no offline calibration. See [[saw-int4#Inside the fused Triton kernel]] for the kernel walkthrough.

---

## Practical guidance

| You want to... | Use |
|---------------|-----|
| INT4 weights with no calibration data | QuIP# (or AWQ if random-Hadamard isn't worth the complexity) |
| INT4 weights + INT4 activations on LLaMA-2 family | QuaRot |
| Push to the highest accuracy at INT4 W+A on a hard-to-quantize model | SpinQuant (offline calibration cost) |
| Compress KV cache to INT4 on a serving system today | SAW-INT4 / BDR (MHA only, SGLang fork) |
| Combine: INT4 weights + INT4 activations + INT4 KV in one stack | QuaRot (all three at once) or QuaRot/SpinQuant for W+A + BDR for KV (but careful — both of these rotate KV, you need to pick one) |
| Stay portable to non-NVIDIA hardware | QuaRot/SpinQuant (the rotation is in PyTorch, not Triton); BDR currently requires Triton GQA decode |

> [!warning] Trap: don't stack BDR's KV rotation on top of QuaRot's KV rotation
> QuaRot already rotates KV (and absorbs the rotation into adjacent weights); applying BDR on top would rotate twice. If you want both weight + activation and KV quantization, either (a) use QuaRot / SpinQuant alone for all three, or (b) use QuaRot for W+A only with a non-quantized KV path, then layer BDR on the KV — which means the QuaRot W+A rotations need to *not* touch KV.

---

## Open questions

- **MLA architectures.** All of QuaRot, SpinQuant, and BDR target MHA/GQA. DeepSeek-V3-style MLA stores a low-rank projection of K, and the rotation has to interact with the up-projection — no published method handles this cleanly yet.
- **Learned block-diagonal.** SpinQuant learned the full rotation; BDR uses random block-diagonal Hadamard. A learned block-diagonal rotation (small enough to be kernel-friendly, smarter than random) is an obvious next step.
- **Rotation at non-power-of-2 head dims.** All current methods assume power-of-two block sizes. Llama-style 128-dim heads are fine; less common dims (e.g. 80 from some research models) are not.
- **Rotation + sparsity.** Rotations flatten distributions, which makes Top-K sparse attention worse (every channel matters now). The interaction is unstudied.
- **FP4 / NVFP4 + rotation.** [[quantization|NVFP4]] has its own outlier-mitigation via two-level scaling; whether rotation still helps at NVFP4 is open.

---

## Related reading

- [[saw-int4]] — Block-diagonal Hadamard rotation for KV cache (full paper review).
- [[quantization]] — Broader quantization landscape (GPTQ, AWQ, FP8, NVFP4, SmoothQuant).
- [[kv-cache-optimization]] — KV cache optimization stack; rotation-based KV is one technique among many.
- [[sglang]] — Where the SAW-INT4 fork lives.
- [[paged-attention]] — The paged KV layout that constrains BDR's block-diagonal design.

---

## References

- Chee et al., "QuIP: 2-Bit Quantization of Large Language Models with Guarantees," NeurIPS 2023. [arXiv:2307.13304](https://arxiv.org/abs/2307.13304)
- Chee et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks," ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Liu et al., "SpinQuant: LLM Quantization with Learned Rotations," 2024. [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
- Jia et al., "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving," 2026. [arXiv:2604.19157](https://arxiv.org/abs/2604.19157)
