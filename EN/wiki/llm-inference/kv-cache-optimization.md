---
title: "KV Cache Optimization Techniques"
category: llm-inference
tags: [kv-cache, mqa, gqa, mla, quantization, sparse-attention, memory-optimization]
created: 2026-04-13
updated: 2026-05-20
status: mature
---

# KV Cache Optimization Techniques

> [!abstract]+ TL;DR
> KV cache is the **primary memory bottleneck** in LLM serving — up to **70 % of total GPU memory** — growing linearly with sequence length × batch size. The optimization stack spans architecture to bytes: **architecture** (MHA → GQA → MQA → MLA, ~3 % of MHA), **memory management** ([[paged-attention|PagedAttention]] cuts waste from 60–80 % to < 4 %), **quantization** (FP8 → INT4 → INT4+BDR rotation), **compression and eviction** (H2O, StreamingLLM, KVTC), **prefix caching** ([[vllm|vLLM]] hashing, [[sglang|SGLang]] RadixAttention), **distributed** (LMCache, Mooncake). Modern production stack: GQA + PagedAttention + FP8 KV + prefix caching.

## Overview

The KV cache (Key-Value Cache) is the most critical data structure in Transformer autoregressive decoding. It stores the Key and Value vectors of already-computed tokens, avoiding repeated attention computation over all previous tokens at every decode step.

KV cache is the **primary memory bottleneck** in LLM inference, consuming **up to 70 % of total GPU memory** in long-sequence, large-batch scenarios. Its size grows linearly with sequence length and linearly with batch size, directly limiting the system's maximum concurrency and the longest context it can handle.

Optimizing KV cache is the core challenge for improving LLM serving efficiency, spanning the full stack from model architecture design to system-level optimization. This page systematically surveys the current mainstream KV cache optimization techniques.

---

## KV Cache Size Analysis

### Formula

KV cache size of a single request:

```
KV_cache_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

Where:
- `2`: one for Key, one for Value
- `num_layers`: number of Transformer layers
- `num_kv_heads`: number of KV attention heads (equals `num_heads` in MHA, fewer in GQA/MQA)
- `head_dim`: dimension per attention head
- `seq_len`: current sequence length
- `dtype_bytes`: bytes per element (FP16=2, FP8=1, INT4=0.5)

### KV Cache Size of Typical Models

```python
def kv_cache_size_gb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: float = 2.0,  # FP16
) -> float:
    """Compute KV cache size (GB)"""
    size_bytes = (
        2 * num_layers * num_kv_heads * head_dim 
        * seq_len * batch_size * dtype_bytes
    )
    return size_bytes / (1024 ** 3)

# LLaMA-3.1-8B (GQA: 32 layers, 8 KV heads, 128 dim)
print(f"LLaMA-8B, seq=4K, bs=1:  {kv_cache_size_gb(32, 8, 128, 4096):.2f} GB")
print(f"LLaMA-8B, seq=4K, bs=32: {kv_cache_size_gb(32, 8, 128, 4096, 32):.2f} GB")
print(f"LLaMA-8B, seq=128K, bs=1: {kv_cache_size_gb(32, 8, 128, 131072):.2f} GB")

# LLaMA-3.1-70B (GQA: 80 layers, 8 KV heads, 128 dim)
print(f"LLaMA-70B, seq=4K, bs=1:  {kv_cache_size_gb(80, 8, 128, 4096):.2f} GB")
print(f"LLaMA-70B, seq=4K, bs=16: {kv_cache_size_gb(80, 8, 128, 4096, 16):.2f} GB")

# DeepSeek-V3 (MLA: 61 layers, very small equivalent KV heads, needs special accounting)
# MLA KV cache ≈ 512-dim compressed vector per layer (instead of conventional KV)
deepseek_v3_kv = 2 * 61 * 512 * 4096 * 2 / (1024**3)  # simplified estimate
print(f"DeepSeek-V3 (MLA), seq=4K, bs=1: {deepseek_v3_kv:.3f} GB")

# Output:
# LLaMA-8B, seq=4K, bs=1:  0.50 GB
# LLaMA-8B, seq=4K, bs=32: 16.00 GB
# LLaMA-8B, seq=128K, bs=1: 16.00 GB
# LLaMA-70B, seq=4K, bs=1:  1.25 GB
# LLaMA-70B, seq=4K, bs=16: 20.00 GB
# DeepSeek-V3 (MLA), seq=4K, bs=1: 0.476 GB (far smaller than equivalent MHA)
```

### Visualizing KV Cache Growth

```
KV cache size grows linearly with sequence length (LLaMA-8B, FP16, batch=1):

memory
(GB)
 16 ┤                                                    ●  128K
    │                                                 ╱
 12 ┤                                              ╱
    │                                           ╱
  8 ┤                                        ╱
    │                                     ╱
  4 ┤                                  ╱
    │                               ╱
  2 ┤                        ●  32K
    │                  ●  16K
  1 ┤           ●  8K
0.5 ┤    ●  4K
0.25┤ ● 2K
    └──┬────┬────┬────┬────┬────┬────┬────┬──→ seq length
       2K   8K   16K  32K  64K  96K  128K

Batch-size multiplier (seq=4K):
  bs=1:   0.5 GB
  bs=8:   4.0 GB
  bs=32:  16.0 GB    ← already eats most of an H100 80GB KV budget
  bs=128: 64.0 GB    ← exceeds a single H100!
```

### KV Cache vs Model Weights

```
LLaMA-3.1-70B (FP16) memory layout example:

Model weights:  140 GB (TP across 2 × H100)
                    ┌─────────────────────────────────┐
H100 #1 (80GB):    │ weights 70GB │ KV cache ~8GB │ other 2GB │
                    └─────────────────────────────────┘
H100 #2 (80GB):    │ weights 70GB │ KV cache ~8GB │ other 2GB │
                    └─────────────────────────────────┘

→ Only ~8GB per card available for KV cache
→ Limits maximum batch_size × seq_len
→ This is exactly why KV cache optimization matters so much!
```

---

## Architecture-Level Optimization

Architecture-level optimization reduces the KV data each attention head must store at the model-design level — the most fundamental form of optimization.

### Multi-Head Attention (MHA) — Baseline

In standard MHA, each attention head has its own Q, K, V projections:

```
MHA (standard multi-head attention):

Query heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8
               │   │   │   │   │   │   │   │
Key heads:    K1  K2  K3  K4  K5  K6  K7  K8    ← 8 independent KV heads
Value heads:  V1  V2  V3  V4  V5  V6  V7  V8

KV cache: 8 × 2 × head_dim × seq_len = 16 × head_dim × seq_len
```

### Multi-Query Attention (MQA)

MQA (Shazeer, 2019) makes all attention heads **share a single** K and V pair:

```
MQA (multi-query attention):

Query heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8
               │   │   │   │   │   │   │   │
               └───┴───┴───┼───┴───┴───┴───┘
                           │
Key head:                  K1                    ← only 1 KV head!
Value head:                V1

KV cache: 1 × 2 × head_dim × seq_len = 2 × head_dim × seq_len
KV cache reduction: 8x (vs MHA)
```

MQA properties:
- KV cache shrinks to `1/num_heads` (e.g., 32× smaller for a 32-head model)
- Decode speed improves significantly (lower memory-bandwidth demand)
- May hurt model quality (all heads share the same KV, less expressive power)
- Representative models: PaLM, StarCoder, Falcon

### Grouped-Query Attention (GQA)

GQA (Ainslie et al., 2023) is a compromise between MHA and MQA: group attention heads and let each group share one K, V pair.

```
GQA (grouped-query attention, 2 groups):

Query heads:  Q1  Q2  Q3  Q4 │ Q5  Q6  Q7  Q8
               │   │   │   │ │  │   │   │   │
               └───┴───┼───┘ │  └───┴───┼───┘
                       │     │          │
Key heads:             K1    │         K2         ← 2 KV heads
Value heads:           V1    │         V2

KV cache: 2 × 2 × head_dim × seq_len = 4 × head_dim × seq_len
KV cache reduction: 4x (vs MHA)
```

GQA properties:
- KV cache shrinks to `num_kv_groups / num_heads`
- Quality close to MHA (with a sensible group count)
- The current **mainstream** choice
- Representative models: LLaMA-2/3, Mistral, Gemma, Qwen-2

### Multi-head Latent Attention (MLA)

MLA (DeepSeek-V2/V3, 2024) takes a more aggressive compression approach: project KV into a low-dimensional **latent space** and cache only the compressed latent vector.

```
MLA (multi-head latent attention):

              ┌─── original KV (high-dim) ──┐
              │                              │
              ▼                              ▼
         ┌─────────┐                  ┌─────────┐
         │ down-proj W_d│             │         │
         │ (compress)   │             │         │
         └────┬────┘                  └─────────┘
              │
              ▼
         ┌─────────┐
         │ latent vec│  ← only this is cached! (dim: d_c << n_h × d_h)
         │ c_t      │     DeepSeek-V3: d_c = 512 vs original 16384
         └────┬────┘
              │
         ┌────┴────┐
         │ up-proj W_u│
         │ (decompress)│
         └────┬────┘
              │
              ▼
         ┌─────────┐
         │ K, V heads│  ← decompressed on-the-fly at attention time
         └─────────┘

KV cache: d_c × seq_len × dtype_bytes (far smaller than MHA/GQA)
```

MLA properties:
- Extremely high compression: KV cache is **~3 %** of MHA in DeepSeek-V3
- Up-projection at compute time (extra compute, but can be absorbed into the attention GEMM)
- RoPE positional embeddings need special handling (RoPE cannot be applied directly to the compressed latent vector)
  - DeepSeek's fix: store an uncompressed RoPE-related component separately
- Representative models: DeepSeek-V2, DeepSeek-V3

### Architecture Comparison

```
MHA vs MQA vs GQA vs MLA (8 Query heads):

         Q heads    KV heads    KV cache   Quality
MHA:    Q1..Q8      K1..K8      8x          best
         ││││││││    ││││││││
         ↓↓↓↓↓↓↓↓    ↓↓↓↓↓↓↓↓
         8-to-1      8 KV copies

GQA-2:  Q1..Q8      K1, K2      2x          near MHA
         ↓↓↓↓↓↓↓↓    ↓    ↓
         4-to-1      2 KV copies

MQA:    Q1..Q8      K1          1x          slightly lower
         ↓↓↓↓↓↓↓↓    ↓
         8-to-1      1 KV copy

MLA:    Q1..Q8      c_t (compr) ~0.25x      near MHA
         ↓↓↓↓↓↓↓↓    ↓
         decompr-map  1 compressed vec
```

| Architecture | KV cache size (rel. MHA) | Model quality | Inference speed | Representative models |
|--------------|--------------------------|---------------|-----------------|----------------------|
| MHA | 1x (baseline) | best | slowest | GPT-3 |
| MQA | 1/n_heads (~3%) | slightly lower | fastest | PaLM, Falcon |
| GQA | n_groups/n_heads (~25%) | near MHA | fast | LLaMA-3, Mistral |
| MLA | ~3-5% | near MHA | needs decompr | DeepSeek-V3 |

### Code Example: GQA Cuts KV Cache

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """GQA implementation, showing how KV cache shrinks"""
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,  # GQA: every 4 Q heads share 1 KV head
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_q_heads // num_kv_heads  # = 4
        
        # Q projection: full num_q_heads
        self.q_proj = nn.Linear(hidden_dim, num_q_heads * head_dim, bias=False)
        # K, V projection: only num_kv_heads (much fewer!)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_dim, bias=False)
    
    def forward(self, x, kv_cache=None):
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # KV cache only stores num_kv_heads copies (not num_q_heads copies)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        new_kv_cache = (k, v)
        
        # Expand KV heads to match Q heads (only at compute time, not stored)
        # [batch, seq, num_kv_heads, dim] → [batch, seq, num_q_heads, dim]
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)
        
        # standard attention computation...
        # (in practice use FlashAttention etc. optimized kernels)
        
        return output, new_kv_cache
    
    def kv_cache_size_per_token(self) -> int:
        """KV cache size per token (bytes, FP16)"""
        return 2 * self.num_kv_heads * self.head_dim * 2  # 2 for K,V; 2 for FP16
        # MHA version: 2 * self.num_q_heads * self.head_dim * 2
        # GQA savings: num_q_heads / num_kv_heads = 4x

# Memory comparison
mha_per_token = 2 * 32 * 128 * 2  # 16,384 bytes
gqa_per_token = 2 * 8 * 128 * 2   # 4,096 bytes
print(f"MHA per token: {mha_per_token:,} bytes")
print(f"GQA per token: {gqa_per_token:,} bytes")
print(f"GQA savings: {mha_per_token / gqa_per_token:.0f}x")
```

---

## Memory Management Optimization

### PagedAttention

[[paged-attention|PagedAttention]] is the single most important memory-management optimization, cutting KV cache waste from 60–80 % to < 4 %. See [[paged-attention]] for a deep dive.

Core idea: divide the KV cache into fixed-size blocks, and use a block table to present non-contiguous physical memory as a virtually contiguous range, allocated strictly on demand.

### Token-Level vs Block-Level Management

| Granularity | Representative | Memory waste | Management overhead | Suitable for |
|------------|----------------|--------------|---------------------|--------------|
| Pre-allocate max length | Legacy | 60-80% | lowest | obsolete |
| Block-level | PagedAttention | <4% | low | mainstream standard |
| Token-level | TokenAttention | ~0% | higher | long-sequence optimization |
| Hardware virtual memory | vAttention | ~0% | very low | NVIDIA GPUs |

### Dynamic Memory Allocation Strategies

Memory-management strategies used by modern inference frameworks:

```
Dynamic allocation + preemption:

1. Normal operation: allocate physical blocks on demand
   Request arrives → allocate prompt blocks → incrementally allocate generation blocks

2. Memory pressure: when free blocks run low
   ├─ Option A: pause new requests (wait for existing ones to free memory)
   ├─ Option B: preempt low-priority requests
   │         ├─ Swap: swap KV cache out to CPU memory
   │         └─ Recompute: drop KV cache, recompute when needed
   └─ Option C: compress existing KV cache (quantize/evict)

3. Reclamation: free all physical blocks immediately upon request completion
```

---

## KV Cache Quantization

### FP8 KV Cache

Quantizing the KV cache from FP16 (16 bit) to FP8 (8 bit) halves memory with negligible accuracy loss.

```
FP8 E4M3 format:
┌─────┬──────────┬─────────┐
│ sign │ exp (4) │ mantissa (3) │
└─────┴──────────┴─────────┘
  1 bit   4 bits    3 bits = 8 bits total

vs FP16:
┌─────┬──────────┬──────────────┐
│ sign │ exp (5) │ mantissa (10) │
└─────┴──────────┴──────────────┘
  1 bit   5 bits    10 bits = 16 bits total

Precision range:
  FP16: ±65504, precision ~0.001
  FP8:  ±448,   precision ~0.125
  More than enough for attention scores!
```

FP8 KV cache properties:
- **2x compression**
- Accuracy loss typically **< 0.5 %** (on perplexity and downstream tasks)
- Hopper (H100) and later support FP8 natively
- Nearly all frameworks support it out of the box

```python
# Enabling FP8 KV cache in vLLM
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    kv_cache_dtype="fp8",       # FP8 KV cache
    # kv_cache_dtype="fp8_e4m3", # or explicitly specify E4M3
    quantization="fp8",          # weights in FP8 too
    tensor_parallel_size=4,
)
```

### INT4/INT8 KV Cache

More aggressive quantization schemes:

**INT8 KV cache**:
- 2x compression (same as FP8)
- Uses per-channel or per-token scale factors
- Accuracy loss comparable to FP8

**INT4 KV cache** (e.g., KIVI):
- **4x compression**
- Requires a carefully designed quantization scheme:
  - Key cache uses per-channel quantization (channels differ greatly in value range)
  - Value cache uses per-token quantization (tokens differ greatly in value range)
  - Keep a small number of recent tokens' KV in full precision (as "anchors")

```
KIVI quantization strategy:

Key cache (per-channel quantization):
each channel has its own scale and zero-point
┌──────────────────────────────────────────┐
│ ch0: [full-precision scale] [INT4 values...] │
│ ch1: [full-precision scale] [INT4 values...] │
│ ...                                      │
│ ch_d: [full-precision scale] [INT4 values...] │
└──────────────────────────────────────────┘

Value cache (per-token quantization):
each token has its own scale and zero-point
┌──────────────────────────────────────────┐
│ tok0: [full-precision scale] [INT4 values...] │
│ tok1: [full-precision scale] [INT4 values...] │
│ ...                                      │
└──────────────────────────────────────────┘

Most recent W tokens stay in FP16 (quality anchor)
```

### NVFP4 (Blackwell)

A 4-bit floating-point format introduced with the NVIDIA Blackwell architecture:

- **4x compression** (vs FP16)
- Native hardware support, two-level scaling
- Accuracy loss **< 1 %**
- Requires Blackwell GPUs such as B200/GB200

### Quantization Accuracy vs Memory Tradeoffs

| Format | Compression | Accuracy loss | Hardware | Status |
|--------|-------------|---------------|----------|--------|
| FP16 (baseline) | 1x | none | any | default |
| FP8 E4M3 | 2x | tiny (<0.5%) | Hopper+ | widely used |
| INT8 | 2x | tiny | any | widely used |
| NVFP4 | 4x | small (<1%) | Blackwell | emerging |
| INT4 (raw scale+zero) | ~3.5x | **catastrophic collapse on reasoning models** | any | don't use alone |
| **INT4 + BDR** ([[saw-int4]]) | ~3.5x | **GPQA <1 %** | any (Triton MHA only) | new |
| KIVI (2-bit, mixed granularity) | 8x | small (~1-2%) | any (needs custom kernel) | research |

### RoPE-aware Quantization

An important implementation detail: RoPE (rotary positional embedding) concentrates energy on specific dimension pairs, making Keys carry substantially larger values on those channels across all tokens.

Mitigation:
- Split Keys into RoPE-related dimensions and non-RoPE dimensions
- RoPE dimensions are kept at higher precision or use a dedicated quantization
- Non-RoPE dimensions can be quantized more aggressively

### Rotation-Based KV Cache Quantization

A deeper fix to the K outlier problem is to apply a **rotation** to K (and optionally V) **before** quantization, smearing per-channel outliers across the head dimension and making the resulting tensor uniformly friendly to quantization. Multiplying by an orthogonal matrix preserves the L2 norm but redistributes energy; the subsequent per-token scale-and-zero quantization then has a much easier task.

This is the same idea QuIP/QuIP# and QuaRot use for **weight + activation** quantization (see [[quantization#Rotation-based quantization (QuIP → QuaRot → SpinQuant → BDR)]] and [[rotation-based-quantization]]) — the SAW-INT4 paper specializes it for **KV cache** under serving constraints:

- **Block-diagonal Hadamard rotation along the head dimension**, with a fixed block size (e.g. 16 or 128) → kernel-friendly and compatible with paged layout.
- **Fused with the INT4 write**: rotation + normalization + per-token scale/zero + INT4 packing happen inside one Triton kernel, so the rotation overhead amortizes into the memory pass INT4 needs anyway.
- **Q correction at decode time**: the same rotation is applied to Q inside the decode kernel, so the attention math stays unchanged.

Concrete effect (Qwen3-4B-Thinking-2507, GPQA): raw INT4 collapses the model to 0 %; INT4 + BDR recovers 65.82 % (vs. BF16 66.67 %). End-to-end throughput is indistinguishable from raw INT4. See [[saw-int4]] for the full paper read and kernel walkthrough.

**Caveats.** Currently MHA only (MLA architectures need a different formulation), Triton GQA decode backend, and only validated on a single-precision benchmark.

---

## KV Cache Compression

### Token Eviction Strategies

When the KV cache gets too large, you can selectively remove tokens that contribute least to attention.

#### H2O (Heavy-Hitter Oracle)

H2O observes that attention scores concentrate heavily on a few "heavy hitter" tokens:

```
H2O token retention policy:

Attention score distribution (typical pattern):
Token position:  [0] [1] [2] [3] [4] [5] [6] ... [95] [96] [97] [98] [99]
Attention:       0.3 0.1 0.01 0.02 0.01 0.15 0.01 ... 0.01 0.05 0.08 0.12 0.14

H2O keeps three groups:
┌─────────────────────────────────────────────────────────────────────┐
│ [Initial Tokens]      [Heavy Hitters]           [Recent Window]    │
│ ████                  ████  ████                 ████████████████   │
│ token 0-3            token 5, 98-99             token 85-99        │
│ (attention sink)     (highest cumulative attn)  (recent context)   │
│                                                                     │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  evicted tokens     │
└─────────────────────────────────────────────────────────────────────┘

Retention ratio: ~20-40% of tokens
Memory saved: 2.5-5x
Quality loss: 1-3% degradation possible on long-text tasks
```

#### Scissorhands

Similar to H2O, but uses a different importance metric:
- Based on the **persistence** of attention patterns (not just the current step's attention magnitude)
- A token is evicted only if it has received low attention across many consecutive steps
- Avoids evicting tokens that are "temporarily ignored but important later"

#### Token Merging

Instead of dropping tokens outright, merge the KV vectors of similar tokens:

```
Token merging:

Original KV: [t0] [t1] [t2] [t3] [t4] [t5] [t6] [t7]

After similarity detection:
Merged:      [t0] [t1+t2] [t3] [t4+t5+t6] [t7]

Pro: loses less information than dropping
Con: merge operation adds compute overhead
```

### Attention Sink

StreamingLLM observed that the **first token** in a Transformer often receives anomalously high attention scores, even when its content is irrelevant. The phenomenon is called **Attention Sink**.

```
Attention Sink phenomenon:

Attention score (typical pattern):
         ┃
    0.25 ┃ █
         ┃ █
    0.20 ┃ █
         ┃ █
    0.15 ┃ █                                           ██
         ┃ █                                          ████
    0.10 ┃ █                                        ████████
         ┃ █          █                            ████████████
    0.05 ┃ █ ░ ░ ░ █ █ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ██████████████
         ┃ █ ░ ░ ░ █ █ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ██████████████
    0.00 ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         t0 t1 t2 t3 t4 t5 ... ... ... ... ... t95 t96 t97 t98 t99
         ↑                                      └──────────────┘
      attention sink                              recent tokens

The model dumps "excess" attention onto the first token,
even when it is <bos> or meaningless padding.
```

### StreamingLLM

StreamingLLM leverages the Attention Sink phenomenon to enable indefinite-length streaming inference:

```
StreamingLLM strategy:

Keep only two parts of the KV cache:
┌──────────────┬─────────────────────────────────┐
│ Sink tokens  │      Sliding Window              │
│ (first 4)    │    (most recent N tokens)        │
│ ref=0.25     │    ref: maintain coherence       │
└──────────────┴─────────────────────────────────┘

Fixed memory budget = sink_size + window_size
e.g.: 4 + 2044 = 2048 tokens

Cache state while processing token 100,000:
[t0][t1][t2][t3] ... [t97,956][t97,957]...[t99,999]
└──sink tokens──┘     └────── window (2044) ──────┘

Pros:
- Fixed memory, can handle unbounded sequences
- Stable perplexity (does not degrade as sequence grows)
- Simple to implement

Cons:
- Loses middle-context information
- Not suitable for tasks needing full-text understanding (e.g., long-doc QA)
```

### Sliding Window Attention

Mistral uses sliding window attention directly in the model architecture:

```
Sliding window attention (window_size = 4096):

Standard (full) attention:
Each token attends to all previous tokens
Token 100: attends to tokens 0 ~ 99
Token 10000: attends to tokens 0 ~ 9999  ← huge KV cache!

Sliding window attention:
Each token attends only to the most recent W tokens
Token 100: attends to tokens 0 ~ 99 (less than W, attends to all)
Token 10000: attends to tokens 5904 ~ 9999  ← KV cache fixed at W!

KV cache size: fixed at window_size (does not grow with sequence length)

Empirical effect:
- Mistral-7B (window=4096): quality close to full attention
- Reason: stacking layers lets information propagate across layers,
  covering longer effective ranges
  - e.g., 32 layers × 4096 window ≈ theoretically covers info over 131,072 tokens
```

### Eviction Strategy Comparison

```
Different eviction/compression strategies:

Strategy          Memory savings   Quality loss   Use case
────────────────────────────────────────────────
Full KV cache     0%               none           short sequences
Sliding Window    fixed cap        moderate       streaming, model-builtin
StreamingLLM      fixed cap        moderate       unbounded streaming chat
H2O               2-5x             small          general long sequences
Scissorhands      2-5x             small          general long sequences
Token Merging     2-3x             very small     when info preservation matters
KVTC              10-20x           small          extreme compression
```

---

## Prefix Caching and Sharing

### Automatic Prefix Caching

Prefix caching shares the KV cache of identical prefixes across requests, avoiding redundant computation. See [[paged-attention#Prefix Caching]] for details.

### RadixAttention (SGLang)

[[sglang|SGLang]]'s RadixAttention uses a radix tree to perform token-level exact prefix matching:

```
RadixAttention radix tree example:

Multi-turn dialogue:
User: "What is Python?" → Assistant: "Python is..."
User: "What are its features?" → Assistant: "Key features..."

                    [root]
                      │
              [System prompt KV]
                   /        \
        [User: "What is     [User: "What are
         Python?"]           its features?"]
            │                      │
        [Asst: "Python        [Asst: "Key
         is..."]               features..."]

New request: User: "What is Java?"
  → matches [System prompt KV] (hit! skip prefill)
  → start computation from "What is Java?"

Win: the prefill of system prompt + dialogue history is skipped entirely
```

### Cross-Request KV Sharing (LMCache)

LMCache decouples KV cache management from the inference engine, supporting cross-engine and cross-node KV sharing:

```
LMCache architecture:

┌──────────┐   ┌──────────┐   ┌──────────┐
│ vLLM #1  │   │ vLLM #2  │   │ SGLang   │
└─────┬────┘   └─────┬────┘   └─────┬────┘
      │              │              │
      └──────────┬───┴──────────────┘
                 │
         ┌───────┴────────┐
         │   LMCache      │
         │  Connector     │
         └───────┬────────┘
                 │
    ┌────────┬───┴───┬──────────┐
    │GPU Cache│CPU DRAM│  SSD    │  Redis/S3
    └────────┴───────┴──────────┘

Features:
- Multi-level cache hierarchy (GPU → CPU → SSD → remote)
- KV sharing across engine instances
- Throughput up to 15x (high prefix reuse scenarios)
```

---

## Distributed KV Cache

### KV Cache in DP Attention

In DP Attention (the scheme used by DeepSeek-V3), different DP ranks hold KV cache for different requests:

```
KV cache partition in DP Attention:

4-way DP Attention:
┌────────────────────────────────────────────────────────┐
│ GPU 0: KV cache for requests {r0, r4, r8, ...}        │
│ GPU 1: KV cache for requests {r1, r5, r9, ...}        │
│ GPU 2: KV cache for requests {r2, r6, r10, ...}       │
│ GPU 3: KV cache for requests {r3, r7, r11, ...}       │
└────────────────────────────────────────────────────────┘

During attention computation:
- Each GPU computes attention only for its local requests (no communication)
- The FFN layer uses EP (Expert Parallelism), which requires All-to-All
- KV cache does not need cross-GPU synchronization (independent on each GPU)

Pro: KV cache scales very well
     Each GPU stores only 1/DP_size of the request KV
```

### KV Cache Transfer in Prefill-Decode Disaggregation

In a [[prefill-decode-disaggregation|prefill-decode disaggregated]] architecture, KV cache generated on the prefill node must be transferred to the decode node:

```
KV cache transfer flow:

Prefill Node                      Decode Node
┌─────────────┐                  ┌─────────────┐
│ 1. run Prefill │                │              │
│ 2. produce KV cache │──── xfer ──→│ 3. receive KV   │
│ 3. release local KV  │            │ 4. continue decode │
└─────────────┘                  └─────────────┘

Transfer volume = KV_cache_size(prompt_length)
e.g., LLaMA-70B, prompt=4K tokens, FP8:
  = 2 × 80 × 8 × 128 × 4096 × 1 = 640 MB

Transfer optimizations:
├─ RDMA/InfiniBand: high bandwidth, low latency
├─ Mooncake Transfer Engine: purpose-built for KV transfer
├─ Compressed KV transfer: KVTC and similar reduce volume
└─ Pipelined transfer: compute + transfer in parallel (layer-by-layer)
```

### CacheBlend and CacheGen

**CacheGen**: encodes and compresses KV cache for storage and transfer
- Uses a learned encoder to compress KV cache
- Compression ratio 3-5x
- Suited for cross-network transfer scenarios

**CacheBlend**:
- Mixes local computation with remote KV cache
- Applies local corrections to remotely fetched KV cache
- Balances transfer latency and compute overhead

### Mooncake

[[prefill-decode-disaggregation#Mooncake|Mooncake]] (FAST 2025 best paper) treats KV cache as a system primitive, pooling all CPU/DRAM/SSD resources in the cluster:

```
Mooncake KV Cache Pool:

┌─────────────────────────────────────────────────┐
│                 KV Cache Pool                    │
│                                                  │
│  Node 0          Node 1          Node 2         │
│  ┌──────┐       ┌──────┐       ┌──────┐       │
│  │GPU HBM│       │GPU HBM│       │GPU HBM│       │
│  │ (hot) │       │ (hot) │       │ (hot) │       │
│  ├──────┤       ├──────┤       ├──────┤       │
│  │CPU RAM│       │CPU RAM│       │CPU RAM│       │
│  │ (warm)│       │ (warm)│       │ (warm)│       │
│  ├──────┤       ├──────┤       ├──────┤       │
│  │ SSD   │       │ SSD   │       │ SSD   │       │
│  │ (cold)│       │ (cold)│       │ (cold)│       │
│  └──────┘       └──────┘       └──────┘       │
│                                                  │
│  Unified management, auto-tiered by hotness     │
│  Capacity up 59-498% (vs GPU HBM alone)         │
└─────────────────────────────────────────────────┘
```

---

## Code Examples

### KV Cache Size Calculator

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int      # GQA: < num_q_heads; MQA: = 1
    head_dim: int
    mla_latent_dim: Optional[int] = None  # MLA compressed dim

# Common model configurations
MODELS = {
    "llama-3.1-8b": ModelConfig("LLaMA-3.1-8B", 32, 32, 8, 128),
    "llama-3.1-70b": ModelConfig("LLaMA-3.1-70B", 80, 64, 8, 128),
    "llama-3.1-405b": ModelConfig("LLaMA-3.1-405B", 126, 128, 8, 128),
    "mistral-7b": ModelConfig("Mistral-7B", 32, 32, 8, 128),
    "deepseek-v3": ModelConfig("DeepSeek-V3", 61, 128, 128, 128, mla_latent_dim=512),
    "qwen-2.5-72b": ModelConfig("Qwen-2.5-72B", 80, 64, 8, 128),
}

def kv_cache_size(
    model: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: float = 2.0,  # FP16=2, FP8=1, INT4=0.5
) -> dict:
    """Compute KV cache size"""
    
    if model.mla_latent_dim:
        # MLA: only store the compressed latent vector
        # Plus extra storage for RoPE-related dims (typically 64-128 dims)
        rope_dim = 64  # approximation
        size_bytes = (
            model.num_layers * (model.mla_latent_dim + rope_dim) 
            * seq_len * batch_size * dtype_bytes
        )
    else:
        # MHA/GQA/MQA
        size_bytes = (
            2 * model.num_layers * model.num_kv_heads * model.head_dim
            * seq_len * batch_size * dtype_bytes
        )
    
    size_gb = size_bytes / (1024 ** 3)
    
    # MHA-equivalent size (for comparison)
    mha_equiv_bytes = (
        2 * model.num_layers * model.num_q_heads * model.head_dim
        * seq_len * batch_size * dtype_bytes
    )
    mha_equiv_gb = mha_equiv_bytes / (1024 ** 3)
    
    return {
        "model": model.name,
        "size_gb": size_gb,
        "mha_equiv_gb": mha_equiv_gb,
        "compression_vs_mha": mha_equiv_gb / size_gb if size_gb > 0 else float('inf'),
        "per_token_bytes": size_bytes / seq_len / batch_size,
    }

# Print KV cache size of each model
print(f"{'model':<20} {'seq_len':>8} {'bs':>4} {'KV cache':>10} {'MHA equiv':>10} {'ratio':>8}")
print("─" * 70)

for name, model in MODELS.items():
    for seq_len in [4096, 32768, 131072]:
        result = kv_cache_size(model, seq_len, batch_size=1)
        print(f"{result['model']:<20} {seq_len:>8} {1:>4} "
              f"{result['size_gb']:>9.2f}G {result['mha_equiv_gb']:>9.2f}G "
              f"{result['compression_vs_mha']:>7.1f}x")
    print()
```

### FP8 KV Cache Configuration

```python
# ========== vLLM FP8 KV cache configuration ==========

from vllm import LLM, SamplingParams

# Method 1: directly specify the KV cache dtype
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    kv_cache_dtype="fp8",            # KV cache in FP8
    tensor_parallel_size=4,
    gpu_memory_utilization=0.92,     # FP8 allows higher utilization
)

# Method 2: combine with weight quantization
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    quantization="fp8",              # weights in FP8
    kv_cache_dtype="fp8",            # KV cache in FP8 too
    tensor_parallel_size=2,          # FP8 saves memory, fewer cards needed
)

# Method 3: use a quantization config file
# Create quantization_config.json:
# {
#     "kv_cache": {
#         "dtype": "fp8_e4m3",
#         "static": false,           # dynamic quantization (per-token scale)
#         "scaling_factor": null      # auto-compute
#     }
# }

# ========== Memory savings comparison ==========
# LLaMA-70B, seq=4K, batch=16:
# FP16 KV: 20.0 GB
# FP8 KV:  10.0 GB  ← saves 10GB → ~80% more concurrent requests possible
```

---

## Choosing an Optimization

Choose KV cache optimization strategies based on the scenario:

```
KV cache optimization decision tree:

What is your main bottleneck?
│
├─ Out of memory (cannot fit target batch_size)
│   │
│   ├─ Are you using a GQA/MQA model?
│   │   ├─ No → switch to a GQA model (e.g., LLaMA-3)
│   │   └─ Yes → next step
│   │
│   ├─ Is FP8 KV cache enabled?
│   │   ├─ No → enable FP8 (the easiest 2x win)
│   │   └─ Yes → next step
│   │
│   ├─ Using PagedAttention?
│   │   ├─ No → use vLLM/SGLang (enabled by default)
│   │   └─ Yes → next step
│   │
│   ├─ Very long sequences (>32K)?
│   │   ├─ Yes → consider:
│   │   │   ├─ sliding-window models (Mistral)
│   │   │   ├─ KV eviction (H2O)
│   │   │   └─ KV offload to CPU (LMCache)
│   │   └─ No → consider INT4 KV quantization (rotation + INT4, see [[saw-int4]] / BDR, or KIVI)
│   │
│   └─ Tried all the above? → add GPUs or use model parallelism
│
├─ TTFT too high (first-token latency)
│   │
│   ├─ Repeated prefixes?
│   │   ├─ Yes → enable prefix caching
│   │   │   ├─ vLLM: on by default
│   │   │   └─ SGLang: RadixAttention (faster)
│   │   └─ No → consider prefill-decode disaggregation
│   │
│   └─ Use disaggregation + KV transfer optimization
│
└─ Throughput too low
    │
    ├─ Enable continuous batching ✓
    ├─ Increase batch_size (needs more KV memory → back to top)
    └─ Consider speculative decoding to speed up generation
```

### Compatibility of Optimizations

| Optimization | + PagedAttn | + FP8 KV | + Prefix Cache | + GQA |
|--------------|-------------|----------|---------------|-------|
| PagedAttention | - | compatible | complementary | compatible |
| FP8 KV cache | compatible | - | compatible | compatible |
| Prefix caching | complementary | compatible | - | compatible |
| GQA architecture | compatible | compatible | compatible | - |
| Token eviction | needs adapt | composable | conflict risk | compatible |
| Sliding window | compatible | compatible | partial | compatible |

Most optimizations are orthogonal and can be combined. A typical production setup:
- **GQA model + PagedAttention + FP8 KV + prefix caching** = best general configuration

---

## References

1. **Kwon et al.** "Efficient Memory Management for Large Language Model Serving with PagedAttention" — SOSP 2023. [paper](https://arxiv.org/abs/2309.06180)

2. **Shazeer, N.** "Fast Transformer Decoding: One Write-Head is All You Need" — 2019. [paper](https://arxiv.org/abs/1911.02150)
   - Original Multi-Query Attention (MQA) paper

3. **Ainslie et al.** "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" — EMNLP 2023. [paper](https://arxiv.org/abs/2305.13245)
   - Grouped-Query Attention (GQA)

4. **DeepSeek-AI.** "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" — 2024. [paper](https://arxiv.org/abs/2405.04434)
   - Multi-head Latent Attention (MLA)

5. **Liu et al.** "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" — 2024. [paper](https://arxiv.org/abs/2402.02750)
12. **Jia et al.** "SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving" — 2026. [paper](https://arxiv.org/abs/2604.19157) — block-diagonal Hadamard rotation makes raw INT4 KV usable on reasoning models.
13. **Ashkboos et al.** "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" — NeurIPS 2024. [paper](https://arxiv.org/abs/2404.00456) — full Hadamard rotation for weights + activations; ancestor of the BDR KV variant.

6. **Zhang et al.** "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" — NeurIPS 2023. [paper](https://arxiv.org/abs/2306.14048)

7. **Xiao et al.** "Efficient Streaming Language Models with Attention Sinks" — ICLR 2024. [paper](https://arxiv.org/abs/2309.17453)
   - StreamingLLM

8. **Zheng et al.** "SGLang: Efficient Execution of Structured Language Model Programs" — 2024.
   - RadixAttention

9. **Qin et al.** "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" — FAST 2025 Best Paper.

10. **Panwar et al.** "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention" — ASPLOS 2025. [paper](https://arxiv.org/abs/2405.04437)

---

## Related Pages

- [[paged-attention]] — block-based memory management deep dive
- [[vllm]] — prefix caching and FP8 KV implementation
- [[sglang]] — RadixAttention prefix caching
- [[quantization]] — broader quantization (incl. weight/activation rotation)
- [[saw-int4]] — block-diagonal Hadamard rotation + INT4 KV (paper read)
- [[rotation-based-quantization]] — QuIP / QuaRot / SpinQuant / BDR family
- [[continuous-batching]] — interaction of scheduling and KV cache management
- [[prefill-decode-disaggregation]] — KV cache transfer challenges
- [[long-context-serving]] — KV cache challenges in long-context scenarios
- [[multi-turn-optimization]] — cross-turn KV cache reuse
