---
title: "KV Cache Optimization Techniques"
category: llm-inference
tags: [kv-cache, mqa, gqa, mla, quantization, sparse-attention, memory-optimization]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# KV Cache Optimization Techniques

## Overview

The KV cache stores Key and Value vectors from computed tokens, avoiding recomputation during autoregressive decoding. It is the **primary memory bottleneck** in LLM serving — consuming up to **70% of GPU memory** — and grows linearly with both sequence length and batch size, directly limiting concurrency and context length.

This page surveys the full stack of KV cache optimizations: architecture-level, memory management, quantization, compression, caching/sharing, and distributed strategies.

---

## KV Cache Size Analysis

### Formula

```
KV_cache_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

### Example Calculations

```python
def kv_cache_gb(layers, kv_heads, head_dim, seq_len, batch=1, dtype=2.0):
    return 2 * layers * kv_heads * head_dim * seq_len * batch * dtype / 1024**3

# LLaMA-3.1-8B (GQA: 32 layers, 8 KV heads, 128 dim, FP16)
# seq=4K, bs=1:   0.50 GB
# seq=4K, bs=32:  16.00 GB
# seq=128K, bs=1: 16.00 GB

# LLaMA-3.1-70B (80 layers, 8 KV heads, 128 dim)
# seq=4K, bs=16:  20.00 GB  ← saturates KV budget on H100

# DeepSeek-V3 (MLA): ~0.5 GB for seq=4K (vs ~10 GB MHA equivalent)
```

**Key insight**: at batch=32, seq=4K, even an 8B GQA model uses 16 GB for KV alone. Optimization is not optional.

---

## Architecture-Level Optimizations

These reduce KV data at the model design level — the most fundamental optimization.

```
Comparison (8 query heads):

MHA:   Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    KV heads: K1..K8 (8)     KV size: 8x
GQA-2: Q1 Q2 Q3 Q4|Q5 Q6 Q7 Q8    KV heads: K1, K2 (2)     KV size: 2x
MQA:   Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    KV heads: K1 (1)          KV size: 1x
MLA:   Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    Latent: c_t (compressed)  KV size: ~0.25x
```

| Architecture | KV Size (vs MHA) | Quality | Representative Models |
|-------------|-------------------|---------|----------------------|
| **MHA** | 1x (baseline) | Best | GPT-3 |
| **MQA** | ~3% (1/n_heads) | Slightly lower | PaLM, Falcon |
| **GQA** | ~25% (groups/heads) | Near MHA | **LLaMA-3, Mistral, Qwen-2** |
| **MLA** | ~3-5% | Near MHA | **DeepSeek-V2/V3** |

**GQA** is the current industry standard. **MLA** achieves even higher compression by projecting KV into a low-dimensional latent space (DeepSeek-V3: 512-dim latent vs 16384-dim full KV), at the cost of requiring decompression during attention.

---

## Memory Management

### PagedAttention

[[paged-attention|PagedAttention]] is the industry standard for KV cache memory management. Reduces waste from 60-80% to <4% via block-based virtual memory. See [[paged-attention]] for deep dive.

### Management Granularity

| Granularity | Approach | Waste | Overhead |
|------------|----------|-------|----------|
| Max-length pre-alloc | Traditional | 60-80% | Lowest |
| Block-level | PagedAttention | <4% | Low |
| Token-level | TokenAttention | ~0% | Higher |
| Hardware VM | vAttention | ~0% | Minimal |

---

## KV Cache Quantization

| Format | Compression | Quality Loss | Hardware | Status |
|--------|-------------|-------------|----------|--------|
| FP16 (baseline) | 1x | None | Any | Default |
| FP8 E4M3 | 2x | Minimal (<0.5%) | Hopper+ | **Widely used** |
| INT8 | 2x | Minimal | Any | Widely used |
| NVFP4 | 4x | Small (<1%) | Blackwell | Emerging |
| KIVI (2-bit) | 8x | Small (~1-2%) | Any (custom kernel) | Research |

**FP8 KV cache** is the easiest win — 2x compression with negligible quality loss:

```python
# vLLM FP8 KV cache
llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", kv_cache_dtype="fp8")
# Saves ~10GB on seq=4K, batch=16 → ~80% more concurrent requests
```

**KIVI** (2-bit): per-channel quantization for Keys, per-token for Values, with recent tokens kept in FP16 as anchors.

**RoPE-aware quantization**: Key tensors have high variance across RoPE dimensions — keep RoPE-related dims in higher precision.

---

## KV Cache Compression & Eviction

### Token Eviction Strategies

```
H2O (Heavy-Hitter Oracle):
Keep: [Initial tokens] + [Heavy hitters] + [Recent window]
Drop: Everything else (~60-80% of tokens)

Attention pattern:
t0(HIGH) t1(low) t2(low) ... t50(HIGH) ... t95(low) t96 t97 t98 t99(recent)
 ↑ sink                        ↑ heavy hitter         ↑ sliding window
```

- **H2O**: Initial tokens + recent window + tokens with highest cumulative attention
- **Scissorhands**: Evicts based on persistent low-attention (not just current step)
- **Token Merging**: Merge similar tokens' KV vectors (less info loss than eviction)

### StreamingLLM & Sliding Window

**StreamingLLM**: Keep sink tokens (first ~4) + sliding window → fixed memory, infinite-length streaming. Exploits the **attention sink** phenomenon.

**Sliding Window Attention** (Mistral): Architectural — each token attends only to last W tokens. Fixed KV size. Multi-layer stacking extends effective range (32 layers x 4K window ~ 128K theoretical coverage).

---

## Prefix Caching & Sharing

### Automatic Prefix Caching

Reuses KV cache for shared prefixes (system prompts, few-shot, multi-turn history):

- **vLLM V1**: Hash-based, LRU eviction. Default on, <1% overhead on miss.
- **[[sglang|SGLang]] RadixAttention**: Token-level radix tree. ~29% faster in prefix-heavy workloads.

### Cross-Request Sharing (LMCache)

LMCache decouples KV storage from inference engines, supporting GPU/CPU/SSD/Redis/S3 storage hierarchy and cross-engine sharing. Up to **15x throughput** improvement in high-reuse scenarios.

---

## Distributed KV Cache

### DP Attention

In DP Attention (DeepSeek-V3), each DP rank holds KV cache for a disjoint set of requests — no cross-GPU KV synchronization needed. Scales well.

### Prefill-Decode Disaggregation

In [[prefill-decode-disaggregation|disaggregated architectures]], KV cache must be transferred from prefill to decode nodes:
- Transfer size: e.g., LLaMA-70B, 4K prompt, FP8 = ~640 MB
- Optimizations: RDMA, Mooncake Transfer Engine, pipeline transfer, KV compression (CacheGen)

### Mooncake

[[prefill-decode-disaggregation#Mooncake|Mooncake]] (FAST 2025 Best Paper): pools CPU/DRAM/SSD cluster-wide as a unified KV cache store. **59-498% capacity improvement**.

---

## Optimization Selection Guide

```
Decision tree:

Memory-bound?
├─ Use GQA model? (No → switch to LLaMA-3/Mistral)
├─ FP8 KV enabled? (No → easiest 2x win)
├─ PagedAttention? (No → use vLLM/SGLang)
├─ Long sequences (>32K)? → sliding window, H2O eviction, CPU offload
└─ Still constrained? → INT4 KV (KIVI), add GPUs

High TTFT?
├─ Repeated prefixes? → enable prefix caching
└─ Consider prefill-decode disaggregation

Low throughput?
├─ Enable continuous batching
├─ Increase batch size (→ need more KV memory → optimize above)
└─ Consider speculative decoding
```

**Typical production stack**: GQA model + PagedAttention + FP8 KV cache + prefix caching. Most optimizations are orthogonal and composable.

---

## References

1. **Kwon et al.** "PagedAttention" — SOSP 2023. [Paper](https://arxiv.org/abs/2309.06180)
2. **Shazeer** "Fast Transformer Decoding: One Write-Head is All You Need" — 2019. (MQA)
3. **Ainslie et al.** "GQA" — EMNLP 2023. [Paper](https://arxiv.org/abs/2305.13245)
4. **DeepSeek-AI** "DeepSeek-V2" — 2024. (MLA) [Paper](https://arxiv.org/abs/2405.04434)
5. **Liu et al.** "KIVI: 2bit KV Cache Quantization" — 2024. [Paper](https://arxiv.org/abs/2402.02750)
6. **Zhang et al.** "H2O: Heavy-Hitter Oracle" — NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.14048)
7. **Xiao et al.** "StreamingLLM" — ICLR 2024. [Paper](https://arxiv.org/abs/2309.17453)
8. **Zheng et al.** "SGLang" — 2024. (RadixAttention)
9. **Qin et al.** "Mooncake" — FAST 2025 Best Paper.
10. **Panwar et al.** "vAttention" — ASPLOS 2025. [Paper](https://arxiv.org/abs/2405.04437)

## Related Pages

- [[paged-attention]] — Block-based memory management deep dive
- [[vllm]] — Prefix caching & FP8 KV implementation
- [[sglang]] — RadixAttention prefix caching
- [[quantization]] — Broader quantization techniques
- [[continuous-batching]] — Scheduling & KV cache interaction
- [[prefill-decode-disaggregation]] — KV cache transfer challenges
- [[long-context-serving]] — Long context KV cache challenges
- [[multi-turn-optimization]] — Cross-turn KV reuse
