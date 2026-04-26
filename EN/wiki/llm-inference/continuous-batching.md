---
title: "Continuous Batching: Dynamic Request Scheduling"
category: llm-inference
tags: [continuous-batching, scheduling, iteration-level, dynamic-batching, throughput]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# Continuous Batching: Dynamic Request Scheduling

## Overview

Batching is essential for GPU utilization in LLM inference. GPUs are massively parallel -- a single request uses only a fraction of available compute. Batching multiple requests together amortizes the cost of loading model weights from memory.

However, LLM outputs vary wildly in length (a few tokens to thousands). **Continuous batching** (iteration-level scheduling) solves this by dynamically adjusting batch composition at every decode step -- inserting new requests as old ones finish. It is the core scheduling mechanism in [[vllm|vLLM]], [[sglang|SGLang]], and [[tensorrt-llm|TensorRT-LLM]].

## The Problem with Static Batching

Static batching collects a fixed batch, processes all requests together, and waits for **every** request to finish before accepting the next batch.

```
Static Batching (batch of 4):

Time  ──────────────────────────────────────────►

Req A: |████████████████|                          (16 tokens)
Req B: |████████████████████████████████████████|   (40 tokens)
Req C: |████████|                                   (8 tokens)
Req D: |████████████████████████|                   (24 tokens)
        ↑                                       ↑
     Batch start                        Batch ends when B finishes
                                        (C,A,D sit idle)

Utilization: (16+40+8+24) / (4×40) = 88/160 = 55%
Wasted: 45%
```

The **convoy effect**: the entire batch's latency is dictated by the longest request. Shorter requests occupy GPU slots without doing useful work.

| Approach | Batch Formation | Execution | GPU Utilization |
|----------|----------------|-----------|-----------------|
| **Static** | Fixed size | Wait for all | Low (20-55%) |
| **Dynamic** | Time-window collection | Wait for all | Medium (40-65%) |
| **Continuous** | Per-iteration | Requests leave immediately | High (85-98%) |

## How Continuous Batching Works

Introduced by the **Orca** paper (Yu et al., OSDI 2022), which proposed **iteration-level scheduling** and **selective batching**. Orca achieved **36.9x throughput** over NVIDIA FasterTransformer on GPT-3 175B.

Core loop -- executed every decode step:

1. Run one forward pass for all active requests
2. Check which requests generated `<EOS>`
3. Remove completed requests, return results to clients
4. Pull new requests from the waiting queue into freed slots
5. Run prefill for new requests
6. Repeat

```
Continuous Batching Timeline (max concurrency = 4):

Step:    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Slot 0: [A  A  A  A  A  A  A  A][E  E  E  E  E  E  E]
Slot 1: [B  B  B  B  B  B  B  B  B  B  B  B][F  F  F...
Slot 2: [C  C  C  C][D  D  D  D  D  D  D  D  D  D][G...
Slot 3: [·  ·  ·  ·  ·  ·  ·  ·  ·  ·][·  ·  ·  ·  ·

Key events:
  Step 4:  C finishes → Slot 2 freed → D enters immediately
  Step 8:  A finishes → Slot 0 freed → E enters
  Step 12: B finishes → Slot 1 freed → F enters
```

Near-100% GPU utilization is achievable under high traffic because freed slots are immediately filled by queued requests.

## Chunked Prefill

Continuous batching solved the convoy effect for decode, but introduced a new problem: **prefill blocking**. A long prompt (e.g., 32K tokens) monopolizes GPU compute and blocks concurrent decode requests, inflating TPOT by 2-30x.

**Chunked prefill** (Sarathi-Serve, Agrawal et al. 2024) breaks long prefills into fixed-size chunks interleaved with decode steps:

```
Chunked Prefill Interleaving:

Step:     1        2        3        4        5
       ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
Pfill: │Chunk1│ │      │ │Chunk2│ │      │ │Chunk3│
       │512tok│ │      │ │512tok│ │      │ │512tok│
       └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
       ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
Decode:│batch │ │batch │ │batch │ │batch │ │batch │
       └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
```

**Token budget model**: each step has a fixed token budget (e.g., 2048). Prefill chunks and decode tokens share this budget, keeping per-step compute predictable.

**TTFT vs TBT trade-off**: Large chunks → lower TTFT (prefill finishes faster) but higher TBT (decode blocked). Small chunks → lower TBT but higher TTFT. Typical sweet spot: 512-2048 tokens.

## Scheduling Strategies

**FCFS**: Process requests in arrival order. Simple, fair, but no SLA differentiation.

**Preemption**: When GPU memory is exhausted, evict low-priority running requests via:
- **Swap**: Copy KV cache to CPU memory (good for long sequences)
- **Recompute**: Discard KV cache, re-prefill later (good for short sequences)

**vLLM V1 Unified Scheduler**: Represents scheduling decisions as `{request_id: num_tokens}`, unifying chunked prefill, prefix caching, and [[speculative-decoding|speculative decoding]] under a single abstraction.

**SGLang**: Uses RadixAttention (radix-tree prefix cache) with token-level granularity, ~29% faster than vLLM on prefix-heavy workloads (multi-turn chat).

```python
# vLLM key scheduling parameters
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    max_num_seqs=256,              # Max concurrent sequences
    max_num_batched_tokens=4096,   # Per-step token budget
    gpu_memory_utilization=0.90,
    swap_space=8,                  # CPU swap space (GB)
    preemption_mode="recompute",   # "recompute" or "swap"
    enable_prefix_caching=True,
)
```

## Memory Management Interaction

[[paged-attention|PagedAttention]] is what makes continuous batching practical:

- KV cache split into fixed-size blocks (typically 16 tokens)
- Blocks allocated on-demand from a free pool -- no pre-allocation needed
- When a request finishes, its blocks are instantly returned to the pool
- New requests can allocate blocks immediately -- no contiguous memory required

This means the scheduler can freely add/remove requests every step without memory fragmentation.

```
Memory lifecycle during continuous batching:

Step T:   Req A → blocks [0,3,7]  |  Req B → blocks [1,5]
          Req C → blocks [2,4,8,9] | Free: [6,10,11]

Step T+1: C finishes → blocks [2,4,8,9] freed
          D enters   → allocates blocks [2,4] from free pool
          A grows    → allocates block [8] for new KV
```

## Performance

Typical throughput improvements over static batching:

| Workload | Static | Continuous | Improvement |
|----------|--------|-----------|-------------|
| Chat (short output) | ~1000 tok/s | ~3000 tok/s | 3x |
| Code gen (medium) | ~800 tok/s | ~2500 tok/s | 3.1x |
| Summarization (low variance) | ~900 tok/s | ~1500 tok/s | 1.7x |
| Mixed (high variance) | ~600 tok/s | ~2800 tok/s | 4.7x |

*(Llama 2 13B, single A100 80GB -- illustrative)*

Latency comparison:

| Metric | Static | Continuous | Continuous + Chunked |
|--------|--------|-----------|---------------------|
| TTFT P50 | 200ms | 100ms | 150ms |
| TTFT P99 | 2000ms | 500ms | 600ms |
| TBT P50 | 30ms | 30ms | 25ms |
| TBT P99 | 300ms | 200ms | **50ms** |

Scheduling overhead: ~1-5% of step time (0.1-1ms), negligible for large models.

## Limitations

1. **TTFT vs TBT trade-off**: Chunked prefill improves TBT at the cost of TTFT. The ultimate solution is [[prefill-decode-disaggregation|PD disaggregation]].
2. **Scheduling complexity**: The scheduler must balance memory constraints, compute budgets, priorities, prefix cache hits, speculative decoding, and preemption -- combinatorial complexity grows with features.
3. **Low-traffic scenarios**: With few concurrent requests, continuous batching offers little advantage. Single-request optimizations like [[speculative-decoding|speculative decoding]] matter more.

## References

- **Orca**: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022. [Paper](https://www.usenix.org/conference/osdi22/presentation/yu)
- **Sarathi-Serve**: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
- **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **FastServe**: Wu et al., "Fast Distributed Inference Serving for Large Language Models", 2023. [arXiv:2305.05920](https://arxiv.org/abs/2305.05920)

## Related Pages

- [[vllm]] — Primary inference engine implementing continuous batching
- [[sglang]] — Alternative engine with RadixAttention prefix caching
- [[paged-attention]] — Memory management enabling continuous batching
- [[prefill-decode-disaggregation]] — Physical separation to eliminate prefill/decode interference
- [[kv-cache-optimization]] — KV cache optimization techniques
- [[speculative-decoding]] — Complementary single-request optimization
