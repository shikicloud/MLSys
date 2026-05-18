---
title: "Prefill-Decode Disaggregation"
category: llm-inference
tags: [prefill-decode, disaggregation, splitwise, distserve, mooncake, kv-transfer]
created: 2026-04-13
updated: 2026-05-13
status: mature
---

# Prefill-Decode Disaggregation

> [!abstract]+ TL;DR
> LLM inference has two phases with fundamentally different compute profiles: **prefill** (compute-bound, processes full prompt) and **decode** (memory-bandwidth-bound, generates tokens one at a time). Colocated on the same GPUs, prefill operations block concurrent decode requests, inflating TPOT by **2–30×**. PD disaggregation physically separates these phases onto different GPU pools, enabling independent optimization and scaling. By 2025–2026 it became the default production architecture — supported natively by **NVIDIA Dynamo, llm-d, [[vllm|vLLM]], [[sglang|SGLang]]**. Key systems: **DistServe** (OSDI 2024), **Splitwise** (ISCA 2024), **Mooncake** (FAST 2025).

```
Colocated:                    Disaggregated:
┌──────────────────┐          ┌──────────┐   ┌──────────┐
│  GPU Pool         │          │ Prefill  │   │ Decode   │
│  ┌────┐  ┌────┐  │          │ Pool     │   │ Pool     │
│  │P+D │  │P+D │  │    →     │ (high    │──►│ (high    │
│  └────┘  └────┘  │          │  FLOPS)  │KV │  BW)     │
│  interference!    │          └──────────┘   └──────────┘
└──────────────────┘
```

## Prefill vs Decode: Compute Profiles

| Metric | Prefill | Decode |
|--------|---------|--------|
| Input tokens | N (hundreds to tens of thousands) | 1 |
| Arithmetic intensity | High (~100+ ops/byte) | Low (~1 ops/byte) |
| GPU compute utilization | 60-80% | 1-5% |
| Memory bandwidth utilization | 20-40% | 80-95% |
| Optimal batch size | 1-4 | 64-512 |
| Latency metric | TTFT | TPOT / TBT |
| Best parallelism | TP (high degree) | DP (many replicas) |

```
Roofline Model:

Performance │
(TFLOPS)    │           / ← Compute ceiling (A100: 312 TFLOPS)
   312 ─────│─ ─ ─ ─ /─ ─ ─ ─ ─ ─ ─ ─ ─
            │      /       ★ Prefill (near compute ceiling)
            │    /
            │  /
            │/  ▲ Decode (far below ceiling, memory-bound)
            └────────────────────────────
            1    10    100    1000
              Arithmetic Intensity (FLOPS/Byte)
```

The same GPU cannot be optimally configured for both workloads simultaneously -- this is the fundamental motivation for disaggregation.

## Why Disaggregate

**1. Prefill interferes with decode latency (head-of-line blocking)**

```
Colocated execution with a new long-prompt request:

Step:     1    2    3    4    5    6    7    8    9   10
New P:   [██████ prefill (long prompt) ██████]
Decode A: ■    ·    ·    ·    ·    ·    ·    ■    ■    ■
Decode B: ■    ·    ·    ·    ·    ·    ·    ■    ■    ■

■ = normal decode    · = blocked by prefill
Normal TPOT: ~30ms → Inflated TPOT: ~210ms (7x)
```

Chunked prefill mitigates but cannot eliminate this (the chunks still consume compute time).

**2. Different optimal batch sizes**: Prefill saturates GPU compute with 1-4 requests; decode needs 64-512 to amortize weight loading.

**3. Different parallelism strategies**: Prefill benefits from high TP (large matrices); decode benefits from DP (many small independent requests).

**4. Independent scaling**: Chat workloads (short prefill, long decode) need more decode GPUs; summarization workloads (long prefill, short decode) need more prefill GPUs.

## Architecture Designs

### General Architecture

```
Client → Router/Scheduler → Prefill Pool → KV Transfer → Decode Pool → Client
                                                            ↑
                                               Continuous decode loop
```

### Splitwise (Microsoft, ISCA 2024)

First systematic PD disaggregation proposal. Uses **heterogeneous hardware** within a single machine (compute-optimized GPUs for prefill, memory-optimized for decode). KV transfer via NVLink (~900 GB/s, nearly free).

**Results**: 1.4x throughput, 20% cost reduction.

### DistServe (OSDI 2024)

Extends disaggregation to **cluster-level** with cross-machine transfer. Introduces **pull-based scheduling** where decode instances pull KV caches from prefill instances. Optimizes **goodput** (throughput under SLO constraints).

**Results**: 7.4x goodput, 12.6x under stricter SLOs vs. colocated.

### Mooncake (Moonshot AI, FAST 2025 Best Paper)

Production system serving Kimi at **100B+ tokens/day**. KVCache-centric architecture pooling CPU DRAM + SSD as distributed KV storage. RDMA-based Transfer Engine (integrated into [[vllm|vLLM]] V1).

**Results**: 59-498% capacity increase depending on workload.

## KV Cache Transfer

The critical bottleneck. After prefill, KV cache must travel from prefill GPU to decode GPU.

```
KV cache size (Llama 70B, BF16, GQA):
  Per-token: ~0.32 MB
  1K tokens:   ~320 MB  → RDMA: ~3ms
  32K tokens:  ~10 GB   → RDMA: ~80ms
  128K tokens: ~40 GB   → RDMA: ~320ms
```

| Transfer Method | Bandwidth | Latency (32K) | Use Case |
|----------------|-----------|---------------|----------|
| NVLink (intra-node) | 900 GB/s | ~6ms | Same-machine split |
| PCIe Gen5 | 64 GB/s | ~16ms | Intra/inter-node |
| RDMA (InfiniBand) | 100-400 Gbps | ~10-40ms | Cross-machine |
| TCP/IP | 10-100 Gbps | ~100-1000ms | Not recommended |

**Transfer vs recompute decision**: If transfer time > prefill time, recompute on the decode side instead. Typical crossover at ~256 tokens for Llama 70B over RDMA.

**Pipeline optimization**: Stream KV cache layer-by-layer while prefill is still running, overlapping compute and transfer.

## DeepSeek-V3 Production Deployment

DeepSeek-V3/R1 is a landmark case of PD disaggregation for MoE models:

```
Prefill Pool: 32 GPUs (4 nodes × 8 H800)
  TP=4, EP=32 → each expert on one GPU
  High compute utilization, MoE all-to-all hidden by large compute

Decode Pool: 320 GPUs (40 nodes × 8 H800)
  TP=4, EP=320 → 10x prefill pool size
  Large memory bandwidth pool, many concurrent requests

Ratio: 1:10 (prefill:decode) — typical for long-output workloads
```

Why 10x more decode GPUs: decode accounts for 70-90% of total time, each request needs KV cache memory, and memory-bandwidth scales with GPU count. TP kept at 4 for both pools to simplify KV cache transfer (same tensor shapes).

See [[parallelism-strategies-deep-dive]] for detailed parallelism configuration.

## Code Example

```python
# vLLM PD disaggregation configuration

# Prefill instance
vllm_serve_cmd_prefill = """
vllm serve deepseek-ai/DeepSeek-V3 \\
    --tensor-parallel-size 4 \\
    --kv-transfer-config '{"kv_connector":"MooncakeConnector",
                           "kv_role":"kv_producer","kv_rank":0}' \\
    --max-num-seqs 32 \\
    --max-num-batched-tokens 8192 \\
    --port 8100
"""

# Decode instance
vllm_serve_cmd_decode = """
vllm serve deepseek-ai/DeepSeek-V3 \\
    --tensor-parallel-size 4 \\
    --kv-transfer-config '{"kv_connector":"MooncakeConnector",
                           "kv_role":"kv_consumer","kv_rank":0}' \\
    --max-num-seqs 512 \\
    --max-num-batched-tokens 2048 \\
    --gpu-memory-utilization 0.92 \\
    --port 8200
"""
```

## Performance

```
Latency (Llama 70B, A100 cluster):

                Colocated       Disaggregated
TTFT P50:       150ms           120ms
TTFT P99:       800ms           400ms
TPOT P50:        35ms            28ms
TPOT P99:       250ms            45ms  ← 5.6x improvement
```

| System | Effective Throughput | vs Colocated |
|--------|---------------------|-------------|
| Colocated (vLLM) | 1.0x | baseline |
| Chunked prefill | 1.8x | 1.8x |
| DistServe | 7.4x | 7.4x |
| Mooncake | 5.0-6.0x | 5.0-6.0x |
| Splitwise | 1.4x | 1.4x |

## Composing with chunked prefill

A natural confusion: "if prefill is on its own node, why would I still need [[continuous-batching#Chunked Prefill|chunked prefill]] on top?" The answer is that PD disaggregation and chunked prefill operate at **two different granularities** and solve **two different interference problems**:

- **PD disaggregation** eliminates *prefill ↔ decode* interference at the **node** level.
- **Chunked prefill** smooths *prefill ↔ prefill* (and *prefill ↔ in-flight decode*) interference at the **iteration** level — both within the prefill pool and on the decode pool's "extension prefill" path.

Three concrete scenarios where chunked prefill is still load-bearing inside a disaggregated deployment:

**1. Prefill ↔ prefill interference on the prefill pool.** Two long requests arriving close together at the same prefill node still queue behind each other:

```
Without chunked prefill on the prefill node:
  [16K prefill of req A][16K prefill of req B][...]
  req A TTFT = 2.3 s
  req B TTFT = 2.3 s + 2.3 s = 4.6 s    ← B sits behind A

With chunked prefill on the prefill node:
  [chunk_A1 + chunk_B1][chunk_A2 + chunk_B2]...
  req A TTFT ≈ 2.5 s    ← small extra because chunk_B is co-resident
  req B TTFT ≈ 2.5 s    ← almost parallel progress, no queuing wait
```

The second request's TTFT goes from "wait 4.6 s" to "get first token at the same time as A."

**2. Extension prefill on the decode pool.** A decode node is not "decode-only" in the strict sense:

- **Multi-turn dialogue**: when a new user turn arrives, the new tokens must be prefilled into the existing KV cache before decoding resumes.
- **Tool-call returns**: the returned tool result is appended as new tokens that must be prefilled.
- **Speculative-decoding rollback**: a rejected speculation sequence requires re-prefilling a small stretch.

These "extension prefills" are typically 50–2000 tokens — short by initial-prompt standards but still long enough to block a node's in-flight decodes if not chunked. Chunked prefill on the decode pool smooths these out.

**3. Traffic shaping inside each pool.** PD disaggregation only solves *role separation*. Within each role, you still need to smooth load, control tail latency, and prevent occasional outlier requests from poisoning the batch. Chunked prefill is the load-smoothing knob for the prefill pool; small chunks on the decode pool tame the extension-prefill case above.

The mnemonic:

```
PD disaggregation  =  prefill pool ↔ decode pool      DON'T mix
Chunked prefill    =  inside prefill pool / inside    DO mix the
                      decode pool's extension prefill   chunks smartly
```

Same "avoid blocking" idea operating at different scales. They're orthogonal layers, not alternatives.

## Beyond PD: attention-FFN disaggregation

PD disaggregation splits *phases* (prefill vs decode) onto different hardware. The natural next step is to split *operators within a single forward pass* — attention onto one hardware tier, FFN onto another — because their compute / memory / bandwidth profiles also differ sharply. This is **attention-FFN (AF) disaggregation**, covered on its own page: see [[af-disaggregation]] for the full treatment (the asymmetry that motivates it, MegaScale-Infer architecture, the structural AF-shape of DP-attention + EP-MoE, activation-transfer cost analysis, and when it pays off).

## Limitations

1. **KV transfer overhead**: Dominates for very long sequences (128K tokens → 320ms over RDMA). Requires high-speed networking (RDMA/NVLink minimum).
2. **System complexity**: Distributed state management, cross-pool failure handling, two separate scaling policies, end-to-end debugging difficulty.
3. **When NOT worth it**: Small models (<7B), short sequences (<256 tokens), low-traffic scenarios, no RDMA network available, frequent short multi-turn interactions.

## References

- **Splitwise**: Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", ISCA 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
- **DistServe**: Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving", OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- **Mooncake**: Qin et al., "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving", FAST 2025 (Best Paper). [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- **TetriInfer**: Xiao et al., "TetriInfer: Distributed LLM Inference via Tetris-like Scheduling", 2024. [arXiv:2401.11181](https://arxiv.org/abs/2401.11181)
- **Sarathi-Serve**: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)

## Related Pages

- [[continuous-batching]] — Scheduling foundation for disaggregation
- [[vllm]] — Supports disaggregated serving via Mooncake Transfer Engine
- [[sglang]] — Alternative engine with first-class disaggregation API
- [[paged-attention]] — KV cache memory management
- [[kv-cache-optimization]] — KV cache compression and optimization
- [[model-parallelism]] — Parallelism strategies used with disaggregation
- [[parallelism-strategies-deep-dive]] — DeepSeek-V3 detailed parallel config
