---
title: "Quantization for LLM Inference"
category: llm-inference
tags: [quantization, gptq, awq, fp8, int4, weight-quantization, kv-cache-quantization]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# Quantization for LLM Inference

## Overview

LLM weights in FP16/BF16 consume 2 bytes per parameter. A 70B model needs ~140GB -- exceeding single-GPU capacity. **Quantization** reduces precision (FP16 -> INT8/INT4/FP8), cutting memory 2-4x and boosting throughput. The single most impactful optimization for fitting larger models on fewer GPUs.

```
Llama-3.3-70B memory footprint:
  FP16:  ████████████████  140 GB  (2x H100)
  FP8:   ████████           70 GB  (1x H100)
  INT4:  ████               35 GB  (1x A100-40G)
```

## Quantization Fundamentals

**Symmetric**: `q = round(x / scale)`, where `scale = max(|x|) / (2^(b-1) - 1)`. Zero maps to zero.

**Asymmetric**: `q = round(x / scale) + zero_point`. Better for non-zero-centered distributions (e.g., post-ReLU activations).

**Granularity** (coarse to fine):
- **Per-tensor**: One scale for entire matrix. Fastest, lowest accuracy.
- **Per-channel**: One scale per output channel. Most common.
- **Per-group** (group_size=128): One scale per 128 elements. Best accuracy, slight overhead. Default for GPTQ/AWQ.

Key insight: **Not all weights matter equally**. Weights multiplied by large activations contribute more to output error when quantized.

## Weight-Only Methods

### GPTQ (ICLR 2023)

Column-by-column quantization using Hessian-based error compensation. Quantizes one column, then adjusts remaining columns to minimize output error.

- Post-training, one-shot. ~4 GPU-hours for 175B models.
- 3-4 bit with <0.5 perplexity increase.
- [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

### AWQ (MLSys 2024 Best Paper)

Identifies ~1% salient weights via activation magnitudes and protects them with per-channel scaling before quantization.

```
Standard:  W_q = quantize(W)              -- salient weights damaged
AWQ:       W_q = quantize(W * s) / s      -- salient weights preserved
```

With Marlin kernel: 741 tok/s (10.9x over FP16 baseline). [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

### SqueezeLLM (ICML 2024)

Dense-and-sparse decomposition: outliers in full-precision sparse matrix, rest in non-uniform (K-means) quantization. Effective for sub-3-bit. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)

### QuIP / QuIP#

Random orthogonal transforms make weights "incoherent" (no outliers), enabling better low-bit quantization. QuIP# uses lattice codebooks. Best theoretical quality at 2-bit. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) This is the foundational paper for the **rotation-based quantization** family covered below.

### Comparison

| Method | Bits | Calibration | PPL Increase | Speed | Best For |
|--------|------|-------------|-------------|-------|----------|
| GPTQ | 3-4 | Required | <0.5 | Fast (Marlin) | Mature ecosystem |
| AWQ | 4 | Required | <0.3 | Fastest (Marlin) | Best overall |
| SqueezeLLM | 2-4 | Required | <1.0 (2-bit) | Medium | Extreme compression |
| QuIP# | 2-4 | Required | <0.5 (2-bit) | Slower | Theoretical best |

## Rotation-based Quantization (QuIP → QuaRot → SpinQuant → BDR)

A coherent technique family where the contribution is **not** a new quantizer — it's an *orthogonal transformation* applied **before** standard quantization that flattens outliers and makes the resulting tensor much easier to quantize. Multiplying by an orthonormal matrix preserves the L2 norm but redistributes energy across all dimensions; the post-rotation tensor has a more uniform distribution and per-token (or per-channel) scale-and-zero quantization works much better.

The lineage:

| Year | Method | Where rotation lives | Rotation type | Note |
|------|--------|---------------------|---------------|------|
| 2023 | **QuIP** ([arXiv:2307.13304](https://arxiv.org/abs/2307.13304)) | Weights | Random orthogonal | Introduced "incoherence processing" — first to formalize that random rotations make low-bit quantization tractable. |
| 2024 | **QuIP#** ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) | Weights | Random Hadamard + lattice codebook | Improved QuIP with vector quantization on the rotated weights; SOTA at 2-bit weights. |
| 2024 | **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) | Weights **and activations** | Random Hadamard fused into weight matrices | NeurIPS 2024. Showed the rotation can be *absorbed* into adjacent linear layers (so it's free at inference), enabling INT4 weight + INT4 activation Llama with little quality loss. |
| 2024 | **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) | Weights and activations | **Learned** rotation matrices | Replaced random Hadamard with rotations trained on a calibration set. Bigger accuracy gains; needs offline training. |
| 2026 | **SAW-INT4 / BDR** ([arXiv:2604.19157](https://arxiv.org/abs/2604.19157)) | **KV cache** | Block-diagonal Hadamard, fused with INT4 write | First production-friendly KV-cache version. Recovers Qwen3-4B-Thinking GPQA from 0 % (plain INT4) to 65.82 %. See [[saw-int4]]. |

A few cross-cutting observations:

- **Where the rotation absorbs matters.** QuaRot's contribution over QuIP is fusing the rotation into adjacent linear-layer weights so inference cost is unchanged. SAW-INT4 instead fuses the rotation into the KV-write Triton kernel and the decode-side Q-attention kernel. Both are forms of "system-aware" rotation.
- **Random vs. learned vs. block-diagonal.** Learned (SpinQuant) > random Hadamard > no rotation, on accuracy. Block-diagonal trades some rotation strength for kernel cache locality and paged-layout compatibility.
- **Rotation is orthogonal to the quantizer.** GPTQ, AWQ, plain scale-and-zero, k-means — any quantizer can run on top of a rotation. The literature mostly stacks rotation with simple per-channel/per-token scale-zero because the rotation already does the hard work.

See [[rotation-based-quantization]] for a deeper synthesis of this family with mathematical foundations and tradeoffs.

## FP8 Quantization

**E4M3** (1+4+3 bits): Range +/-448, higher precision. Used for inference (weights + activations).
**E5M2** (1+5+2 bits): Range +/-57344, lower precision. Used for gradients in training.

- **Hardware**: Native on Hopper+ (H100/H200/B100/B200). ~2x throughput vs FP16.
- **Dynamic scaling**: Compute scale per-tensor at runtime (~1-2% overhead, almost always worth it).
- **Near-transparent**: <0.1 perplexity increase. Default choice on Hopper+.

## KV Cache Quantization

KV cache can consume 30-50% of inference memory for long sequences and large batches. See [[kv-cache-optimization]] for the full landscape and [[saw-int4]] for the rotation-based INT4 KV story.

| Method | Memory Saving | PPL Increase | Recommendation |
|--------|--------------|-------------|----------------|
| FP8 KV | 50% | <0.1 | Default on Hopper+ |
| INT8 KV | 50% | <0.1 | Default on Ampere |
| INT4 KV (plain, g=64) | ~70% | 0.3-0.5 (collapses on reasoning models) | Avoid alone |
| **INT4 KV + BDR** | ~70% | <1 % on GPQA | New, MHA only — see [[saw-int4]] |
| KIVI (2-bit, mixed-granularity) | ~80% | ~1-2% | Custom kernel needed |

```python
# vLLM: enable FP8 KV cache
llm = LLM(model="...", kv_cache_dtype="fp8")
```

## Activation Quantization (SmoothQuant)

Activations have extreme outliers (50-100x normal values), making direct quantization destructive. **SmoothQuant** (ICML 2023) migrates quantization difficulty from activations to weights via per-channel scaling:

```
Y = X * W = (X * diag(s)^-1) * (diag(s) * W) = X_smooth * W_smooth
```

Enables **W8A8** (INT8 weights + INT8 activations) with hardware-accelerated INT8 matmul. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

**W8A8 vs W4A16**: W8A8 accelerates both compute and memory (best for large batches). W4A16 only saves memory but preserves activation precision (best for small batches, memory-bound).

## Code Examples

```python
from vllm import LLM

# AWQ model
llm = LLM(model="casperhansen/llama-3-70b-instruct-awq", quantization="awq", tensor_parallel_size=2)

# FP8 online quantization (Hopper+, no pre-quantized model needed)
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", quantization="fp8", kv_cache_dtype="fp8", tensor_parallel_size=4)

# Pre-quantized FP8
llm = LLM(model="neuralmagic/Llama-3.3-70B-Instruct-FP8", kv_cache_dtype="fp8", tensor_parallel_size=4)
```

## Selection Guide

```
GPU?
 ├─ Blackwell → NVFP4 (native 4-bit, ~4x compression)
 ├─ Hopper    → FP8 (near-lossless, 2x compression)
 └─ Ampere    → INT4 AWQ (best speed/quality) or INT8 SmoothQuant
 
CPU/Edge → GGUF Q4_K_M (llama.cpp, ~92% quality)
KV cache too large → FP8 KV cache (50% savings, negligible quality loss)
```

## Limitations

1. **Quality always degrades** -- especially below 4-bit. Evaluate on your target task, not just perplexity.
2. **Calibration data matters** -- GPTQ/AWQ quality depends on representative calibration samples.
3. **Kernel fragmentation** -- Not all formats have highly optimized kernels (2-bit is often slow).
4. **Outlier sensitivity** -- Some models (older architectures like OPT, BLOOM) are particularly hard to quantize.
5. **MoE considerations** -- Different experts may have different weight distributions; uniform quantization may not be optimal.

## References

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization," ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- Chee et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks," ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Liu et al., "SpinQuant: LLM Quantization with Learned Rotations," 2024. [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
- Jia et al., "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving," 2026. [arXiv:2604.19157](https://arxiv.org/abs/2604.19157)
- NVIDIA, "FP8 Formats for Deep Learning," 2022. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)

## Related Pages

- [[kv-cache-optimization]] -- KV cache quantization details
- [[saw-int4]] -- Block-diagonal Hadamard rotation + INT4 KV (paper review)
- [[rotation-based-quantization]] -- QuIP / QuaRot / SpinQuant / BDR family overview
- [[vllm]] -- Supports GPTQ, AWQ, FP8, and all major formats
- [[tensorrt-llm]] -- NVIDIA-native quantization (FP8, NVFP4)
- [[model-parallelism]] -- Quantization reduces parallelism needs
- [[speculative-decoding]] -- Quantized draft models reduce overhead
- [[continuous-batching]] -- Quantization improves per-GPU throughput
