---
title: "PagedAttention: Virtual Memory for KV Cache"
category: llm-inference
tags: [paged-attention, kv-cache, memory-management, vllm, virtual-memory]
created: 2026-04-13
updated: 2026-04-14
status: mature
---

# PagedAttention: Virtual Memory for KV Cache

## Overview

PagedAttention (Kwon et al., SOSP 2023) applies **OS virtual memory paging** to KV cache management. Before PagedAttention, LLM serving systems wasted **60-80%** of GPU memory due to fragmentation and pre-allocation. PagedAttention reduces this waste to **<4%**, enabling 2-4x higher throughput on the same hardware.

It is the foundational innovation behind [[vllm|vLLM]] and has been adopted by virtually every serving framework including [[sglang|SGLang]], [[tensorrt-llm|TensorRT-LLM]], and HuggingFace TGI. Combined with [[continuous-batching]], it defines the efficiency baseline for modern LLM serving.

---

## The Problem with Traditional KV Cache

Traditional systems pre-allocate contiguous memory for max sequence length per request:

```
Traditional pre-allocation (max_seq_len = 2048):

Request A (actual: 327 tokens):
[█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  84% wasted
       16% used

Request B (actual: 89 tokens):
[██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  96% wasted
    4% used
```

This causes:
- **Internal fragmentation**: allocated but unused space within each request's reservation
- **External fragmentation**: freed memory scattered in non-contiguous chunks
- **Effective batch size reduction**: can't fit more requests even when total free memory suffices, because no single contiguous block is large enough

---

## How PagedAttention Works

### OS Analogy

| OS Concept | PagedAttention |
|-----------|----------------|
| Virtual Page | Logical Block |
| Physical Frame | Physical Block |
| Page Table | Block Table |
| Process | Sequence |
| Page Size | Block Size (typically 16 tokens) |

### Block Table Mechanism

```
Sequence: "The cat sat on the mat and then ..." (block_size=4)

Logical view:
[Block 0: The cat sat on] [Block 1: the mat and then] [Block 2: the dog ...]

Block Table:
  Logical 0 → Physical 7
  Logical 1 → Physical 3
  Logical 2 → Physical 12

GPU Memory:
[0:other][1:other][2:free][3:Blk1][4:free]...[7:Blk0]...[12:Blk2]

Logical blocks are contiguous; physical blocks are scattered.
```

### On-Demand Allocation

Blocks are allocated only when needed:
1. Request arrives with prompt → allocate blocks for prompt tokens
2. Each new generated token appends to the last block
3. When last block is full → allocate one new physical block
4. On completion → return all physical blocks to free list

**Waste**: only in the last block per sequence. With block_size=16, average waste = 8 tokens per sequence.

### Custom CUDA Kernel

Since KV data is non-contiguous, PagedAttention requires a custom attention kernel that:
1. Looks up the block table to find physical block locations
2. Loads K, V vectors from scattered physical blocks
3. Uses online softmax for numerically stable cross-block attention computation
4. Achieves coalesced access within each block

Key optimizations: block-wise reduction via shared memory, online softmax (no two-pass scan), and integration with FlashAttention in later versions.

### Block Size Trade-offs

| Block Size | Pros | Cons |
|-----------|------|------|
| Small (1-4) | Minimal waste | Large block tables, kernel overhead |
| **16 (default)** | **Balanced** | **Standard choice** |
| Large (64-256) | High kernel efficiency | Last-block waste, reduced flexibility |

---

## Copy-on-Write (CoW) Memory Sharing

For parallel sampling and beam search, multiple sequences share a common prefix. PagedAttention uses reference counting + CoW:

```
Parallel sampling (n=3) sharing prompt blocks:

Seq 1: [Block A (ref=3)] [Block B (ref=3)] [private blocks...]
Seq 2: [Block A (ref=3)] [Block B (ref=3)] [private blocks...]
Seq 3: [Block A (ref=3)] [Block B (ref=3)] [private blocks...]

Physical memory: ONE copy of prefix blocks (not three).

When Seq 1 modifies shared block B:
  → ref_count > 1 → copy B to new block → update Seq 1's table
  → Seq 2, Seq 3 continue sharing original B (ref=2)
```

**Results**: Up to **55% memory reduction** for beam search, **2.2x throughput** improvement from sharing alone.

---

## Prefix Caching

Reuses KV cache across requests that share common prefixes (system prompts, few-shot examples, multi-turn history).

**Hash-based (vLLM V1)**: Hash block contents → LRU cache lookup. Default on, <1% overhead on miss.

**RadixAttention ([[sglang|SGLang]])**: Token-level radix tree. ~29% faster than hash-based in prefix-heavy workloads.

| Feature | Hash (vLLM) | Radix Tree (SGLang) |
|---------|-------------|-------------------|
| Granularity | Block-level | Token-level |
| Lookup | O(1) hash | O(n) traversal |
| Prefix-heavy perf | Good | Better (~29%) |

---

## Performance

| System | Model | Throughput vs HF Transformers |
|--------|-------|-------------------------------|
| HuggingFace TGI | OPT-13B | 3.4x |
| vLLM (PagedAttention) | OPT-13B | **14.0x** |
| vLLM (PagedAttention) | OPT-175B | **24.3x** |

Memory efficiency: >96% utilization vs ~28% for traditional approaches.

Scenario-specific gains:

| Scenario | Throughput Gain | Key Reason |
|----------|----------------|------------|
| Standard serving | 2-4x | Higher memory utilization → larger batches |
| Long sequences (>4K) | 4-8x | Longer sequences waste more in traditional approach |
| Parallel sampling (n=4) | 3-6x | CoW memory sharing |
| Beam search (beam=8) | 5-10x | CoW + extensive prefix sharing |

---

## Subsequent Developments

### vAttention (ASPLOS 2025)

Uses CUDA Virtual Memory Management APIs for contiguous virtual addresses with on-demand physical allocation:

```
PagedAttention: physical non-contiguous + virtual non-contiguous (software block table)
vAttention:     physical non-contiguous + virtual contiguous (hardware MMU)
```

- **Compatible with all existing attention kernels** (no custom paging kernel needed)
- Prefill **3.92x faster**, token generation **1.97x faster** than PagedAttention
- Limitation: NVIDIA-only (depends on CUDA VMM)

### TokenAttention

Token-level granularity (no block waste), but higher management overhead. Suitable for very long sequences.

### Dynamic Block Sizes

Adaptive block size based on workload characteristics — small blocks for short sequences, large for long.

---

## Limitations

1. **Last-block waste**: block_size=16 → avg 8 tokens wasted per sequence (minor but nonzero)
2. **Indirection overhead**: block table lookups add latency; non-contiguous access patterns reduce cache efficiency
3. **Custom kernel maintenance**: requires adapted attention kernels; must port to each new attention optimization (FlashAttention-3, etc.) and hardware platform
4. **Multi-GPU complexity**: block tables must be synchronized across TP ranks; KV transfer in disaggregated serving adds overhead

---

## References

1. **Kwon et al.** "Efficient Memory Management for Large Language Model Serving with PagedAttention" — SOSP 2023. [Paper](https://arxiv.org/abs/2309.06180)
2. **Panwar et al.** "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention" — ASPLOS 2025. [Paper](https://arxiv.org/abs/2405.04437)
3. **Yu et al.** "Orca: A Distributed Serving System for Transformer-Based Generative Models" — OSDI 2022.
4. **Dao et al.** "FlashAttention: Fast and Memory-Efficient Exact Attention" — NeurIPS 2022.
5. **Zheng et al.** "SGLang: Efficient Execution of Structured Language Model Programs" — 2024.

## Related Pages

- [[vllm]] — The serving engine built on PagedAttention
- [[kv-cache-optimization]] — Broader KV cache optimization techniques
- [[continuous-batching]] — Scheduling that PagedAttention enables
- [[prefill-decode-disaggregation]] — KV cache transfer in disaggregated serving
- [[sglang]] — RadixAttention prefix caching
- [[quantization]] — KV cache quantization combined with PagedAttention
